"""Orchestration: assemble the context, poll the registry, pool, and decode.

``build_fusion`` is deliberately thin.  It owns the sequence (prepare inputs once,
let every voter cast against the same :class:`VotingContext`, pool the ballots, apply
the hard event override, decode the fused emissions with the baseline transition
prior) and nothing else — no ballot arithmetic lives here anymore.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import GRADED_HALVES, D, X
from ..features import extract_features
from ..model import (
    DEFAULT_PARAMS,
    ModelParams,
    forward_backward,
    merge_isolated_runs,
    viterbi,
)
from .ballot import FusionResult
from .geometry import EPS, PRIOR, PRIOR_WEIGHT
from .pool import blend_ball_positions, pool_ballots
from .resample import resample_perception
from .voters import DEFAULT_WEIGHTS, VOTERS, AtdInputs, VotingContext

CONTEXT_PAD = 500  # frames decoded outside each graded half so transitions see real context


def _atd_inputs(tracking: Any) -> AtdInputs:
    players = np.asarray(tracking.players, dtype=np.float64)
    player_present = np.asarray(tracking.player_present, dtype=bool)
    if player_present.shape != players.shape[:2]:
        player_present = np.isfinite(players).all(axis=2)
    return AtdInputs(
        ball=np.asarray(tracking.ball, dtype=np.float64),
        present=np.asarray(tracking.ball_present, dtype=bool),
        players=players,
        teams=np.asarray(tracking.teams, dtype=np.int8),
        player_present=player_present,
    )


def _decode_halves(
    log_emission: np.ndarray,
    params: ModelParams,
    n_frames: int,
    halves: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi-decode each graded half with a context pad; everything else stays X."""
    codes = np.full(n_frames, X, dtype=np.int8)
    graded = np.zeros(n_frames, dtype=bool)
    for start, stop in halves:
        start, stop = int(start), int(stop)
        if not (0 <= start < stop <= n_frames):
            continue
        lo = max(0, start - CONTEXT_PAD)
        hi = min(n_frames, stop + CONTEXT_PAD)
        local = viterbi(log_emission[lo:hi], params)
        local = merge_isolated_runs(local)
        codes[start:stop] = local[start - lo : stop - lo]
        graded[start:stop] = True
    return codes, graded


def build_fusion(
    tracking: Any,
    tags: Any,
    *,
    perception: Any | None = None,
    features: Any | None = None,
    events: Any | None = None,
    track: Any | None = None,
    whistle: Any | None = None,
    params: ModelParams = DEFAULT_PARAMS,
    weights: dict[str, float] | None = None,
    use_audio: bool = True,
    include_baseline: bool = True,
    baseline_codes: np.ndarray | None = None,
    combine_mode: str = "log",
    halves: tuple[tuple[int, int], ...] | None = None,
) -> FusionResult:
    """Build the full multi-source poll and decode it.

    ``features``/``track``/``whistle``/``events`` are injectable so the CLI and tests can
    reuse the validated event products instead of re-extracting audio and the ATD ball
    tracker.  ``halves`` overrides the graded windows; the default is the module-level
    ``GRADED_HALVES``, resolved at call time so it stays patchable.
    """
    n_frames = int(tracking.n_frames)
    if features is None:
        if track is None or whistle is None or events is None:
            from ..pipeline import prepare_event_products

            track, whistle, events = prepare_event_products(
                tracking, tags, track=track, whistle=whistle, use_audio=use_audio
            )
        features = extract_features(
            tracking, tags, ball_track=track, whistle=whistle, events=events
        )
    if features.start != 0 or features.stop != n_frames:
        raise ValueError("fusion needs full-match features starting at frame 0")
    if halves is None:
        halves = GRADED_HALVES
    resolved_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        resolved_weights.update({str(k): float(v) for k, v in weights.items()})
    if baseline_codes is None:
        from ..pipeline import predict_full

        baseline_codes = predict_full(
            tracking,
            tags,
            params,
            track=track,
            whistle=whistle,
            events=events,
        )
    baseline_codes = np.asarray(baseline_codes, dtype=np.int8)

    ctx = VotingContext(
        n_frames=n_frames,
        tracking=tracking,
        features=features,
        video=resample_perception(perception, n_frames),
        events=events,
        params=params,
        weights=resolved_weights,
        atd=_atd_inputs(tracking),
        baseline_codes=baseline_codes,
    )
    sources = {voter.key: voter.cast(ctx) for voter in VOTERS}

    event_dead = np.asarray(sources["events"].active, dtype=bool)
    fused = pool_ballots(list(sources.values()), mode=combine_mode)
    fused[event_dead] = np.array([0.0, 0.0, 1.0])

    consensus, consensus_conf = blend_ball_positions(
        [
            (ctx.atd.ball, ctx.atd.present.astype(np.float64), 1.0),
            (
                np.asarray(ctx.video["ball_pitch"], dtype=np.float64),
                np.asarray(ctx.video["ball_confidence"], dtype=np.float64),
                0.8,
            ),
            (sources["yolo_ball"].ball, sources["yolo_ball"].weight, 1.0),
        ],
        n_frames,
    )

    # Decode the fused emissions with the same three-state transition prior as the baseline.
    log_emission = np.log(np.clip(fused, EPS, 1.0))
    codes, graded = _decode_halves(log_emission, params, n_frames, halves)
    codes[event_dead & graded] = D

    posterior, confidence, entropy = forward_backward(log_emission, params)
    posterior[event_dead] = np.array([0.0, 0.0, 1.0])
    confidence = np.max(posterior, axis=1)
    entropy = -np.sum(np.where(posterior > 0, posterior * np.log(np.maximum(posterior, EPS)), 0.0), axis=1)
    codes[~graded] = X

    hmm_diagnostics = sources["hmm_atd"].diagnostics
    if not include_baseline:
        baseline_codes = None
    return FusionResult(
        start=0,
        stop=n_frames,
        codes=codes,
        fused_probability=fused,
        posterior=posterior,
        confidence=confidence,
        entropy=entropy,
        consensus_ball=consensus.astype(np.float32),
        consensus_confidence=consensus_conf.astype(np.float32),
        event_dead=event_dead,
        sources=sources,
        baseline_codes=baseline_codes,
        params=params,
        meta={
            "weights": resolved_weights,
            "combine_mode": combine_mode,
            "prior": PRIOR.tolist(),
            "prior_weight": PRIOR_WEIGHT,
            "hmm_confidence_mean": hmm_diagnostics.get("confidence_mean", 0.0),
            "hmm_posterior_mean": hmm_diagnostics.get("posterior_mean", 0.0),
            "perception": bool(perception is not None),
        },
    )
