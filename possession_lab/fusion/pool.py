"""How ballots are tallied, and how ball positions are blended.

The pool filters by ``VoterRole``, never by voter key: the prior is the anchor, the
override never pools, and evidence and redundant hints are the updates.  Ballots must
arrive in registry order — float addition is order-dependent, and the validated output
hash is pinned to this exact accumulation sequence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ballot import SourceVote, VoterRole
from .geometry import EPS, PRIOR, PRIOR_WEIGHT, normalise_rows

COMBINE_MODES = ("log", "linear")


def pool_ballots(ballots: Sequence[SourceVote], *, mode: str = "log") -> np.ndarray:
    """Combine ballots into a fused (N, 3) probability, honouring each ballot's role.

    ``log``
        Log-opinion pool anchored on the validated HMM posterior: every video/tag/audio
        source is a likelihood update, not a replacement vote.  A source can move the
        result only when it is confident enough, and a near-certain HMM state cannot be
        flipped by a single noisy white blob.
    ``linear``
        Confidence-weighted linear average of the ballots, kept for ablation.
    """
    if mode not in COMBINE_MODES:
        raise ValueError(f"unknown combine_mode: {mode!r}")
    if mode == "linear":
        return _pool_linear(ballots)
    return _pool_log(ballots)


def _pool_log(ballots: Sequence[SourceVote]) -> np.ndarray:
    prior_votes = [ballot for ballot in ballots if ballot.role is VoterRole.PRIOR]
    if not prior_votes:
        raise ValueError("log pooling needs the validated HMM ballot as its prior")
    prior = normalise_rows(np.asarray(prior_votes[0].probability, dtype=np.float64))
    log_fused = np.log(np.clip(prior, EPS, 1.0))
    log_prior = np.log(np.clip(PRIOR, EPS, 1.0))[None, :]
    for ballot in ballots:
        if ballot.role in (VoterRole.PRIOR, VoterRole.OVERRIDE):
            continue
        amount = np.clip(ballot.weight, 0.0, 1.0)[:, None]
        log_fused += amount * (np.log(np.clip(ballot.probability, EPS, 1.0)) - log_prior)
    log_fused -= np.max(log_fused, axis=1, keepdims=True)
    return normalise_rows(np.exp(log_fused))


def _pool_linear(ballots: Sequence[SourceVote]) -> np.ndarray:
    n = int(next(iter(ballots)).probability.shape[0]) if ballots else 0
    total_weight = np.zeros(n, dtype=np.float64)
    weighted = np.zeros((n, 3), dtype=np.float64)
    for ballot in ballots:
        if ballot.role is VoterRole.OVERRIDE:
            continue
        weighted += ballot.weight[:, None] * ballot.probability
        total_weight += ballot.weight
    fused = (PRIOR_WEIGHT * PRIOR[None, :] + weighted) / (PRIOR_WEIGHT + total_weight[:, None])
    return normalise_rows(fused)


def blend_ball_positions(
    components: Sequence[tuple[np.ndarray, np.ndarray, float]],
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted average of the ball sources — a visual aid, not a ballot.

    Each component is ``(points, weight, factor)``; components are merged in the given
    order and the summed confidence is squashed into ``[0, 1)`` at the end.
    """
    consensus = np.full((n_frames, 2), np.nan, dtype=np.float64)
    consensus_conf = np.zeros(n_frames, dtype=np.float64)
    for points, component_weight, factor in components:
        valid = np.isfinite(points).all(axis=1) & np.isfinite(component_weight) & (component_weight > 0)
        if not valid.any():
            continue
        w = component_weight[valid] * factor
        current = consensus[valid]
        known = np.isfinite(current).all(axis=1)
        consensus[valid] = np.where(
            known[:, None],
            (current * consensus_conf[valid, None] + points[valid] * w[:, None])
            / np.maximum(1e-9, consensus_conf[valid, None] + w[:, None]),
            points[valid],
        )
        consensus_conf[valid] = consensus_conf[valid] + w
    nonzero = consensus_conf > 0
    consensus_conf[nonzero] = consensus_conf[nonzero] / (1.0 + consensus_conf[nonzero])
    return consensus, consensus_conf
