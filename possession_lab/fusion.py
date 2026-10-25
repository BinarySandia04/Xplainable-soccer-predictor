"""Transparent multi-source possession polling.

The validated ATD/HMM is no longer the only opinion.  Each independent source emits a ballot over
``H`` (Barcelona), ``A`` (Athletic), and ``D`` (dead ball), plus a confidence/weight:

``atd_ball``
    The provided ATD ball and the 46 ATD player slots.
``atd_shape``
    ATD team shape around the last known ball during a tracking gap.
``video_players``
    YOLO person detections projected to the pitch, with team labels.
``video_ball``
    The own video ball tracker.  It is a gap-filling source: while ATD is present it is anchored
    to ATD, and while ATD is missing it uses YOLO ball detections plus a white-blob fallback.
``yolo_ball``
    YOLO sports-ball detections, associated in time under a physical speed gate.  Detections
    unreachable from the previous accepted point are rejected instead of becoming a teleport.
``hmm_atd``
    The posterior of the existing validated ATD/HMM.  Keeping it as one voter means the new
    evidence has to earn a disagreement instead of replacing a strong baseline by fiat.
``tags`` and ``audio``
    Smart tags and the whistle envelope, deliberately weak votes.

The ballots are combined with a weighted linear pool.  The result is then decoded by a second,
small HMM with the same transition prior as the baseline.  Confirmed goals/stoppages are hard
``D`` overrides after the decode.  No public ground truth is read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np

from .config import D, FPS, GRADED_HALVES, H, A, PITCH, PUBLIC_START, PUBLIC_STOP, X
from .features import Features, extract_features
from .model import (
    DEFAULT_PARAMS,
    DecodeResult,
    ModelParams,
    decode,
    forward_backward,
    merge_isolated_runs,
    viterbi,
)
from .perception import select_plausible_ball_detections

SOURCE_LABELS = {
    "atd_ball": "ATD balón",
    "atd_shape": "ATD forma",
    "video_players": "Jugadores YOLO",
    "video_ball": "Balón vídeo",
    "yolo_ball": "YOLO balón",
    "hmm_atd": "HMM ATD",
    "tags": "Smart tags",
    "audio": "Silbato",
    "events": "Eventos",
}

# Evidence strengths for the log-opinion pool.  The validated HMM is the prior, not one more
# equally-weighted ballot; these weights say how much a source may move that prior.  A value of
# 0.10 is a moderate update, 0.03 is a weak hint, and the HMM prior itself is not counted here.
DEFAULT_WEIGHTS = {
    # ATD geometry, tags, and audio are already inside the HMM emission.  They stay visible as
    # ballots but have almost no marginal weight; otherwise the same evidence is counted twice.
    "atd_ball": 0.02,
    "atd_shape": 0.01,
    "video_players": 0.10,
    "video_ball": 0.14,
    "yolo_ball": 0.04,
    "hmm_atd": 1.00,
    "tags": 0.01,
    "audio": 0.02,
    "events": 1.00,
}

# Prior used only when all sources abstain, and as a small stabiliser in the pool.
PRIOR = np.array([0.48, 0.48, 0.04], dtype=np.float64)
PRIOR_WEIGHT = 0.18
EPS = 1e-9

STATUS_NAMES = {
    0: "observed",
    1: "estimated",
    2: "abstain",
    3: "event",
}


@dataclass(frozen=True)
class SourceVote:
    """A per-frame ballot from one source."""

    key: str
    label: str
    probability: np.ndarray  # (N, 3) H, A, D
    weight: np.ndarray  # (N,) actual weight in the linear pool
    active: np.ndarray  # (N,) bool
    status: np.ndarray  # (N,) 0 observed, 1 estimated, 2 abstain, 3 event
    ball: np.ndarray  # (N, 2) pitch metres, NaN when the source has no ball
    details: dict[str, np.ndarray] = field(default_factory=dict)

    def sample(self, frame: int) -> dict[str, Any]:
        frame = int(frame)
        if frame < 0 or frame >= self.probability.shape[0]:
            raise IndexError(f"frame {frame} outside source length {self.probability.shape[0]}")
        probability = self.probability[frame]
        weight = float(self.weight[frame])
        active = bool(self.active[frame])
        status = int(self.status[frame])
        ball = self.ball[frame]
        role = (
            "prior"
            if self.key == "hmm_atd"
            else "override"
            if self.key == "events"
            else "redundant"
            if self.key in {"atd_ball", "atd_shape", "tags", "audio"}
            else "evidence"
        )
        result: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "role": role,
            "active": active,
            "status": status,
            "status_name": STATUS_NAMES.get(status, "unknown"),
            "probability": _json_vector(probability),
            "weight": weight,
            "ball": {
                "x": _finite(ball[0]),
                "y": _finite(ball[1]),
            },
            "reason": _reason(self.key, frame, self.details, active),
        }
        return result


@dataclass(frozen=True)
class FusionResult:
    start: int
    stop: int
    codes: np.ndarray  # (N,)
    fused_probability: np.ndarray  # (N, 3) before the final HMM
    posterior: np.ndarray  # (N, 3) after the final HMM
    confidence: np.ndarray  # (N,)
    entropy: np.ndarray  # (N,)
    consensus_ball: np.ndarray  # (N, 2)
    consensus_confidence: np.ndarray  # (N,)
    event_dead: np.ndarray  # (N,) bool
    sources: dict[str, SourceVote]
    baseline_codes: np.ndarray | None = None
    params: ModelParams = DEFAULT_PARAMS
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.codes.size)

    def sample(self, frame: int) -> dict[str, Any]:
        frame = int(frame)
        if frame < 0 or frame >= self.n_frames:
            raise IndexError(f"frame {frame} outside fusion length {self.n_frames}")
        code = int(self.codes[frame])
        probability = self.fused_probability[frame]
        posterior = self.posterior[frame]
        consensus = self.consensus_ball[frame]
        ballots = [source.sample(frame) for source in self.sources.values()]
        return {
            "code": code,
            "label": ("H", "A", "D", "X")[code],
            "fused_probability": _json_vector(probability),
            "posterior": _json_vector(posterior),
            "confidence": float(self.confidence[frame]),
            "entropy": float(self.entropy[frame]),
            "event_dead": bool(self.event_dead[frame]),
            "baseline_code": int(self.baseline_codes[frame]) if self.baseline_codes is not None else None,
            "consensus_ball": {
                "x": _finite(consensus[0]),
                "y": _finite(consensus[1]),
                "confidence": float(self.consensus_confidence[frame]),
            },
            "sources": ballots,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "codes": self.codes.astype(np.int8),
            "fused_probability": self.fused_probability.astype(np.float32),
            "posterior": self.posterior.astype(np.float32),
            "confidence": self.confidence.astype(np.float32),
            "entropy": self.entropy.astype(np.float32),
            "consensus_ball": self.consensus_ball.astype(np.float32),
            "consensus_confidence": self.consensus_confidence.astype(np.float32),
            "event_dead": self.event_dead.astype(bool),
            "baseline_codes": (
                self.baseline_codes.astype(np.int8)
                if self.baseline_codes is not None
                else np.full(self.n_frames, X, dtype=np.int8)
            ),
        }
        for key, source in self.sources.items():
            arrays[f"src_{key}_probability"] = source.probability.astype(np.float32)
            arrays[f"src_{key}_weight"] = source.weight.astype(np.float32)
            arrays[f"src_{key}_active"] = source.active.astype(bool)
            arrays[f"src_{key}_status"] = source.status.astype(np.int8)
            arrays[f"src_{key}_ball"] = source.ball.astype(np.float32)
            for detail, values in source.details.items():
                arrays[f"src_{key}_{detail}"] = np.asarray(values)
        return arrays


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64)]


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, EPS)
    total = values.sum(axis=1, keepdims=True)
    return values / np.where(total > 0, total, 1.0)


def _reason(key: str, frame: int, details: dict[str, np.ndarray], active: bool) -> str:
    if key == "events":
        kind = details.get("kind", np.zeros(1, dtype=np.int8))[frame] if "kind" in details else 0
        return "gol / detención confirmada" if int(kind) else "sin evento"
    if not active:
        return "sin evidencia utilizable"
    if key in {"atd_ball", "video_players", "video_ball", "yolo_ball"}:
        d_home = details.get("d_home", np.full(1, np.nan))[frame]
        d_away = details.get("d_away", np.full(1, np.nan))[frame]
        count = int(details.get("count", np.zeros(1))[frame])
        pieces = []
        if key == "video_ball":
            source = int(details.get("source", np.zeros(1, dtype=np.int8))[frame])
            origin = {
                0: "detección de vídeo",
                1: "predicción en hueco",
                2: "perdido",
                3: "anclado a ATD",
            }.get(source, "vídeo")
            pieces.append(origin)
        if np.isfinite(d_home):
            pieces.append(f"Barça {float(d_home):.1f} m")
        if np.isfinite(d_away):
            pieces.append(f"Athletic {float(d_away):.1f} m")
        if count:
            pieces.append(f"{count} jugadores")
        return " · ".join(pieces) if pieces else "geometría disponible"
    if key == "atd_shape":
        gap = float(details.get("gap", np.zeros(1))[frame])
        edge = float(details.get("edge", np.full(1, np.nan))[frame])
        return f"sin balón; hueco {gap / FPS:.1f} s; borde {edge:.1f} m"
    if key == "tags":
        vote = int(details.get("vote", np.zeros(1, dtype=np.int16))[frame])
        setpiece = int(details.get("setpiece", np.zeros(1, dtype=np.int8))[frame])
        return f"voto equipo {vote:+d}" + (" · set piece" if setpiece else "")
    if key == "audio":
        env = float(details.get("env", np.zeros(1))[frame])
        return f"envolvente de silbato {env:.2f}"
    if key == "hmm_atd":
        return "posterior del HMM ATD validado"
    return ""


def _geometry_source(
    *,
    key: str,
    ball: np.ndarray,
    ball_confidence: np.ndarray,
    players: np.ndarray,
    teams: np.ndarray,
    present: np.ndarray,
    base_weight: float,
    ball_status: np.ndarray | None = None,
    sigma: float = 3.8,
    dead_r: float = 7.5,
    dead_scale: float = 1.6,
    min_players: int = 6,
) -> SourceVote:
    """Build one geometry ballot from a ball position and a set of player positions."""
    ball = np.asarray(ball, dtype=np.float64)
    n = int(ball.shape[0])
    players = np.asarray(players, dtype=np.float64)
    teams = np.asarray(teams, dtype=np.int8)
    present = np.asarray(present, dtype=bool)
    ball_confidence = np.asarray(ball_confidence, dtype=np.float64)
    probability = np.tile(PRIOR, (n, 1))
    weight = np.zeros(n, dtype=np.float64)
    active = np.zeros(n, dtype=bool)
    status = np.full(n, 2, dtype=np.int8)
    d_home = np.full(n, np.nan, dtype=np.float64)
    d_away = np.full(n, np.nan, dtype=np.float64)
    count = np.zeros(n, dtype=np.int16)
    valid_ball = np.isfinite(ball).all(axis=1) & np.isfinite(ball_confidence) & (ball_confidence > 0)
    if players.size and teams.size:
        for team, target in ((0, d_home), (1, d_away)):
            mask = present & (teams == team)
            if not mask.any():
                continue
            dx = players[:, :, 0] - ball[:, None, 0]
            dy = players[:, :, 1] - ball[:, None, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            distance = np.where(mask, distance, np.inf)
            with np.errstate(all="ignore"):
                target[:] = np.min(distance, axis=1)
            target[~np.isfinite(target)] = np.nan
        count = (present & np.isin(teams, (0, 1))).sum(axis=1).astype(np.int16)
    finite_home = np.isfinite(d_home)
    finite_away = np.isfinite(d_away)
    d_min = np.minimum(d_home, d_away)
    enough = valid_ball & (count >= min_players)
    if enough.any():
        q_home = np.where(finite_home, np.exp(-(d_home**2) / (2.0 * sigma**2)), 1e-9) + EPS
        q_away = np.where(finite_away, np.exp(-(d_away**2) / (2.0 * sigma**2)), 1e-9) + EPS
        q_dead = np.where(np.isfinite(d_min), 0.03 + _sigmoid((d_min - dead_r) / dead_scale), 0.03) + EPS
        # A ball outside the pitch is a very strong dead-ball observation.
        length, width = PITCH
        outside = (
            (ball[:, 0] < -1.5)
            | (ball[:, 0] > length + 1.5)
            | (ball[:, 1] < -1.5)
            | (ball[:, 1] > width + 1.5)
        )
        q_dead = np.where(outside, 5.0, q_dead)
        probability[enough] = _normalise_rows(np.stack([q_home, q_away, q_dead], axis=1)[enough])
        coverage = np.clip(count / 10.0, 0.0, 1.0)
        quality = np.clip(ball_confidence, 0.0, 1.0) * coverage
        weight = base_weight * quality
        active = enough & (weight > 1e-5)
    if ball_status is None:
        ball_status = np.where(ball_confidence >= 0.25, 0, 1).astype(np.int8)
    status = np.where(active, np.asarray(ball_status, dtype=np.int8), 2).astype(np.int8)
    probability[~active] = PRIOR
    weight[~active] = 0.0
    return SourceVote(
        key=key,
        label=SOURCE_LABELS[key],
        probability=probability,
        weight=weight,
        active=active,
        status=status,
        ball=ball,
        details={
            "d_home": d_home.astype(np.float32),
            "d_away": d_away.astype(np.float32),
            "d_min": d_min.astype(np.float32),
            "count": count,
        },
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _resample_perception(perception: Any | None, n_frames: int) -> dict[str, np.ndarray]:
    """Map a sampled video cache onto every video frame with an age-aware decay."""
    empty_players = {
        "player_pitch": np.full((n_frames, 40, 2), np.nan, dtype=np.float32),
        "player_team": np.full((n_frames, 40), -1, dtype=np.int8),
        "player_present": np.zeros((n_frames, 40), dtype=bool),
        "player_confidence": np.zeros((n_frames, 40), dtype=np.float32),
        "player_track": np.full((n_frames, 40), -1, dtype=np.int16),
        "ball_pitch": np.full((n_frames, 2), np.nan, dtype=np.float32),
        "ball_pixel": np.full((n_frames, 2), np.nan, dtype=np.float32),
        "ball_confidence": np.zeros(n_frames, dtype=np.float32),
        "ball_source": np.full(n_frames, 2, dtype=np.int8),
        "yolo_ball": np.full((n_frames, 6, 2), np.nan, dtype=np.float32),
        "yolo_ball_confidence": np.zeros((n_frames, 6), dtype=np.float32),
        "yolo_ball_present": np.zeros((n_frames, 6), dtype=bool),
        "age": np.zeros(n_frames, dtype=np.int32),
    }
    if perception is None or perception.samples == 0:
        return empty_players
    frames = np.arange(n_frames, dtype=np.int64)
    sample_frames = np.asarray(perception.frame_index, dtype=np.int64)
    if sample_frames.size == 1:
        index = np.zeros(n_frames, dtype=np.int64)
    else:
        pos = np.searchsorted(sample_frames, frames, side="left")
        pos = np.clip(pos, 0, sample_frames.size - 1)
        before = np.maximum(pos - 1, 0)
        after = pos
        choose_before = (frames - sample_frames[before]) <= (sample_frames[after] - frames)
        index = np.where(choose_before, before, after)
    index = np.clip(index, 0, sample_frames.size - 1)
    age = np.abs(frames - sample_frames[index]).astype(np.int32)
    # The perception cache is sampled (usually every frame on the public window, every fifth
    # frame for a full-match pass).  Do not let a stale detection live forever, and keep frames
    # outside a public-window cache at zero instead of exponentiating a six-digit age.
    max_age = max(2, int(round(0.30 * FPS)))
    fresh = age <= max_age
    decay = np.zeros(n_frames, dtype=np.float64)
    decay[fresh] = np.exp(-age[fresh].astype(np.float64) / max(2.0, 0.10 * FPS))
    player_present = np.asarray(perception.players_present[index], dtype=bool) & fresh[:, None]
    player_confidence = np.asarray(perception.players_confidence[index], dtype=np.float32)
    player_confidence = (player_confidence * decay[:, None]).astype(np.float32)
    player_present &= player_confidence > 0.02
    ball_pitch = np.asarray(perception.ball_pitch[index], dtype=np.float32).copy()
    ball_pixel = np.asarray(perception.ball_pixel[index], dtype=np.float32).copy()
    ball_confidence = np.asarray(perception.ball_confidence[index], dtype=np.float32) * decay
    ball_source = np.asarray(perception.ball_source[index], dtype=np.int8).copy()
    # A stale sample must not leave a valid-looking ball behind: the ballot gate reads the pitch
    # point and confidence, not only the source code, so both have to be cleared together.
    ball_pitch[~fresh] = np.nan
    ball_pixel[~fresh] = np.nan
    ball_confidence[~fresh] = 0.0
    ball_source[~fresh] = 2
    ball_source[ball_confidence <= 0.02] = 2
    # YOLO emits several candidates per frame and the class is noisy.  Associate them causally in
    # pitch space before they can vote: a candidate unreachable from the last accepted point at
    # football speed is rejected, and a new track needs two mutually reachable detections.
    plausible_ball, plausible_confidence, _, _ = select_plausible_ball_detections(
        perception.yolo_ball_pitch,
        perception.yolo_ball_confidence,
        perception.yolo_ball_present,
        perception.frame_index,
        fps=FPS,
    )
    candidate_count = int(np.asarray(perception.yolo_ball_pitch).shape[1])
    yolo_ball = np.full((n_frames, candidate_count, 2), np.nan, dtype=np.float32)
    yolo_confidence = np.zeros((n_frames, candidate_count), dtype=np.float32)
    yolo_present = np.zeros((n_frames, candidate_count), dtype=bool)
    mapped_ball = np.asarray(plausible_ball[index], dtype=np.float32)
    mapped_confidence = np.asarray(plausible_confidence[index], dtype=np.float32) * decay
    mapped_fresh = fresh & np.isfinite(mapped_ball).all(axis=1) & (mapped_confidence > 0.01)
    yolo_ball[mapped_fresh, 0] = mapped_ball[mapped_fresh]
    yolo_confidence[mapped_fresh, 0] = mapped_confidence[mapped_fresh]
    yolo_present[mapped_fresh, 0] = True
    out = {
        "player_pitch": np.asarray(perception.players_pitch[index], dtype=np.float32),
        "player_team": np.asarray(perception.players_team[index], dtype=np.int8),
        "player_present": player_present,
        "player_confidence": player_confidence,
        "player_track": np.asarray(perception.players_track[index], dtype=np.int16),
        "ball_pitch": ball_pitch,
        "ball_pixel": ball_pixel,
        "ball_confidence": ball_confidence.astype(np.float32),
        "ball_source": ball_source,
        "yolo_ball": yolo_ball,
        "yolo_ball_confidence": yolo_confidence,
        "yolo_ball_present": yolo_present,
        "age": age,
    }
    return out


def _source_atd_shape(features: Features, base_weight: float) -> SourceVote:
    n = features.stop - features.start
    present = np.asarray(features.present, dtype=bool)
    gap = np.asarray(features.gap, dtype=np.float64)
    d_home = np.asarray(features.d_home_last, dtype=np.float64)
    d_away = np.asarray(features.d_away_last, dtype=np.float64)
    edge = np.asarray(features.last_edge, dtype=np.float64)
    vote = np.asarray(features.tag_vote, dtype=np.float64)
    probability = np.tile(PRIOR, (n, 1))
    active = (~present) & (gap >= 5.0) & (np.asarray(features.player_count) >= 6)
    if active.any():
        q_home = np.where(np.isfinite(d_home), np.exp(-(d_home**2) / (2.0 * 6.0**2)), 1e-6) + EPS
        q_away = np.where(np.isfinite(d_away), np.exp(-(d_away**2) / (2.0 * 6.0**2)), 1e-6) + EPS
        q_dead = 0.08 + _sigmoid((5.0 - np.nan_to_num(edge, nan=20.0)) / 1.5)
        q_home *= 1.0 + 0.2 * np.maximum(vote, 0.0)
        q_away *= 1.0 + 0.2 * np.maximum(-vote, 0.0)
        probability[active] = _normalise_rows(np.stack([q_home, q_away, q_dead], axis=1)[active])
    weight = base_weight * np.clip(gap / 25.0, 0.0, 1.0) * active
    weight[~active] = 0.0
    status = np.where(active, 1, 2).astype(np.int8)
    return SourceVote(
        key="atd_shape",
        label=SOURCE_LABELS["atd_shape"],
        probability=probability,
        weight=weight,
        active=active,
        status=status,
        ball=np.full((n, 2), np.nan, dtype=np.float64),
        details={
            "gap": gap.astype(np.float32),
            "edge": edge.astype(np.float32),
            "d_home": d_home.astype(np.float32),
            "d_away": d_away.astype(np.float32),
        },
    )


def _source_tags(features: Features, base_weight: float) -> SourceVote:
    n = features.stop - features.start
    vote = np.asarray(features.tag_vote, dtype=np.float64)
    coverage = np.asarray(features.tag_coverage, dtype=bool)
    setpiece = np.asarray(features.tag_setpiece, dtype=np.float64)
    live = np.asarray(features.tag_live, dtype=bool)
    team = np.tanh(vote / 2.0)
    q_home = 0.50 + 0.32 * team
    q_away = 0.50 - 0.32 * team
    q_dead = 0.03 + 0.32 * setpiece
    q_dead = np.where(live, q_dead * 0.25, q_dead)
    probability = _normalise_rows(np.stack([q_home, q_away, q_dead], axis=1))
    active = coverage & np.isfinite(probability).all(axis=1)
    probability[~active] = PRIOR
    weight = np.where(active, base_weight * (0.35 + 0.65 * np.minimum(1.0, np.abs(team) + setpiece)), 0.0)
    status = np.where(active, 0, 2).astype(np.int8)
    return SourceVote(
        key="tags",
        label=SOURCE_LABELS["tags"],
        probability=probability,
        weight=weight,
        active=active,
        status=status,
        ball=np.full((n, 2), np.nan, dtype=np.float64),
        details={
            "vote": vote.astype(np.int16),
            "setpiece": setpiece.astype(np.int8),
            "coverage": coverage.astype(np.int8),
        },
    )


def _source_audio(features: Features, base_weight: float) -> SourceVote:
    n = features.stop - features.start
    env = np.asarray(getattr(features, "whistle_env", np.zeros(n)), dtype=np.float64)
    env = np.clip(np.nan_to_num(env, nan=0.0), 0.0, 1.0)
    active = env > 0.025
    probability = np.tile(PRIOR, (n, 1))
    probability[active] = np.array([0.08, 0.08, 0.84])
    weight = np.where(active, base_weight * np.clip(env * 2.5, 0.0, 1.0), 0.0)
    status = np.where(active, 0, 2).astype(np.int8)
    return SourceVote(
        key="audio",
        label=SOURCE_LABELS["audio"],
        probability=probability,
        weight=weight,
        active=active,
        status=status,
        ball=np.full((n, 2), np.nan, dtype=np.float64),
        details={"env": env.astype(np.float32)},
    )


def _source_events(features: Features, events: Any | None) -> SourceVote:
    n = features.stop - features.start
    goal = np.asarray(getattr(features, "goal_dead", np.zeros(n)), dtype=bool)
    stop = np.asarray(getattr(features, "stop_dead", np.zeros(n)), dtype=bool)
    if events is not None and getattr(events, "dead_mask", None) is not None:
        dead = np.asarray(events.dead_mask, dtype=bool)
        if dead.size >= n:
            dead = dead[:n]
        else:
            dead = np.pad(dead, (0, n - dead.size))
        goal = np.asarray(getattr(events, "goal_dead", goal), dtype=bool)[:n]
        stop = np.asarray(getattr(events, "stop_dead", stop), dtype=bool)[:n]
    else:
        dead = goal | stop
    kind = np.where(goal, 1, np.where(stop, 2, 0)).astype(np.int8)
    probability = np.tile(PRIOR, (n, 1))
    probability[dead] = np.array([0.0, 0.0, 1.0])
    weight = np.where(dead, 1.0, 0.0)
    status = np.where(dead, 3, 2).astype(np.int8)
    return SourceVote(
        key="events",
        label=SOURCE_LABELS["events"],
        probability=probability,
        weight=weight,
        active=dead,
        status=status,
        ball=np.full((n, 2), np.nan, dtype=np.float64),
        details={"kind": kind},
    )


def _source_hmm(
    features: Features,
    params: ModelParams,
    base_weight: float,
    baseline_codes: np.ndarray | None = None,
) -> tuple[SourceVote, np.ndarray, np.ndarray, np.ndarray]:
    """Turn the validated HMM result into a soft prior for the poll.

    The Viterbi code is the anchor.  Its probability is modulated by the forward-backward
    confidence: a confident baseline is hard to overturn, while an uncertain frame leaves room for
    the independent video ballots.  We do not use the HMM posterior argmax as the anchor because
    the Viterbi code is the validated, temporally coherent output.
    """
    decoded = decode(features, params, with_confidence=True)
    n = features.stop - features.start
    codes = np.asarray(decoded.codes if baseline_codes is None else baseline_codes, dtype=np.int8)
    confidence = np.asarray(decoded.confidence, dtype=np.float64)
    probability = np.tile(PRIOR, (n, 1))
    graded = (codes != X) & np.isin(codes, (H, A, D))
    if graded.any():
        anchor = np.clip(0.60 + 0.25 * np.nan_to_num(confidence, nan=0.0), 0.55, 0.88)
        other = (1.0 - anchor) / 2.0
        probability[graded, 0] = other[graded]
        probability[graded, 1] = other[graded]
        probability[graded, 2] = other[graded]
        rows = np.flatnonzero(graded)
        probability[rows, codes[rows]] = anchor[rows]
    probability = _normalise_rows(probability)
    active = graded
    weight = np.where(active, base_weight, 0.0)
    status = np.where(active, 0, 2).astype(np.int8)
    return (
        SourceVote(
            key="hmm_atd",
            label=SOURCE_LABELS["hmm_atd"],
            probability=probability,
            weight=weight,
            active=active,
            status=status,
            ball=np.full((n, 2), np.nan, dtype=np.float64),
            details={"code": codes, "confidence": confidence.astype(np.float32)},
        ),
        codes,
        confidence,
        np.asarray(decoded.posterior, dtype=np.float64),
    )


def _source_video_ball(
    *,
    features: Features,
    video: dict[str, np.ndarray],
    atd_players: np.ndarray,
    atd_teams: np.ndarray,
    atd_present: np.ndarray,
    base_weight: float,
) -> SourceVote:
    n = features.stop - features.start
    ball = video["ball_pitch"]
    confidence = video["ball_confidence"].astype(np.float64)
    source = video["ball_source"]
    # The source is authoritative only in ATD gaps.  While ATD is present, a video measurement is
    # a useful cross-check but not a second independent ballot.
    atd_here = np.asarray(atd_present, dtype=bool)
    weight_scale = np.where(atd_here, 0.08, 1.0)
    status = np.where(source == 2, 2, np.where(source == 1, 1, 0)).astype(np.int8)
    vote = _geometry_source(
        key="video_ball",
        ball=ball,
        ball_confidence=confidence,
        players=atd_players,
        teams=np.tile(atd_teams, (n, 1)),
        present=np.isfinite(atd_players).all(axis=2),
        base_weight=base_weight,
        ball_status=status,
        sigma=3.8,
        dead_r=7.5,
    )
    weight = vote.weight * weight_scale
    active = vote.active & (weight > 1e-5)
    probability = vote.probability.copy()
    probability[~active] = PRIOR
    return SourceVote(
        key="video_ball",
        label=SOURCE_LABELS["video_ball"],
        probability=probability,
        weight=weight,
        active=active,
        status=np.where(active, status, 2).astype(np.int8),
        ball=vote.ball,
        details={**vote.details, "source": source.astype(np.int8), "age": video["age"].astype(np.int32)},
    )


def _source_yolo_ball(
    *,
    features: Features,
    video: dict[str, np.ndarray],
    atd_players: np.ndarray,
    atd_teams: np.ndarray,
    atd_present: np.ndarray,
    base_weight: float,
) -> SourceVote:
    n = features.stop - features.start
    present = np.asarray(video["yolo_ball_present"], dtype=bool)
    confidence = np.asarray(video["yolo_ball_confidence"], dtype=np.float64)
    points = np.asarray(video["yolo_ball"], dtype=np.float64)
    # The resampler has already associated candidates under the physical speed gate, so choosing
    # the best remaining candidate cannot teleport to a far false positive.
    safe = np.where(present & np.isfinite(points).all(axis=2), confidence, -np.inf)
    best = np.argmax(safe, axis=1)
    has = np.isfinite(safe[np.arange(n), best]) & (safe[np.arange(n), best] > 0)
    ball = np.full((n, 2), np.nan, dtype=np.float64)
    ball_conf = np.zeros(n, dtype=np.float64)
    ball[has] = points[np.arange(n)[has], best[has]]
    ball_conf[has] = confidence[np.arange(n)[has], best[has]]
    status = np.where(ball_conf >= 0.25, 0, 1).astype(np.int8)
    vote = _geometry_source(
        key="yolo_ball",
        ball=ball,
        ball_confidence=ball_conf,
        players=atd_players,
        teams=np.tile(atd_teams, (n, 1)),
        present=np.isfinite(atd_players).all(axis=2),
        base_weight=base_weight,
        ball_status=status,
        sigma=3.8,
        dead_r=7.5,
    )
    # The YOLO ballot is more valuable when ATD has lost the ball, but it should not win a fight
    # with ATD while the provided ball is present.
    weight = vote.weight * np.where(np.asarray(atd_present, dtype=bool), 0.35, 1.0)
    active = vote.active & (weight > 1e-5)
    probability = vote.probability.copy()
    probability[~active] = PRIOR
    return SourceVote(
        key="yolo_ball",
        label=SOURCE_LABELS["yolo_ball"],
        probability=probability,
        weight=weight,
        active=active,
        status=np.where(active, status, 2).astype(np.int8),
        ball=ball,
        details={**vote.details, "age": video["age"].astype(np.int32)},
    )


def build_fusion(
    tracking: Any,
    tags: Any,
    *,
    perception: Any | None = None,
    features: Features | None = None,
    events: Any | None = None,
    track: Any | None = None,
    whistle: Any | None = None,
    params: ModelParams = DEFAULT_PARAMS,
    weights: dict[str, float] | None = None,
    use_audio: bool = True,
    include_baseline: bool = True,
    baseline_codes: np.ndarray | None = None,
    combine_mode: str = "log",
) -> FusionResult:
    """Build the full multi-source poll and decode it.

    ``features``/``track``/``whistle``/``events`` are injectable so the CLI and tests can reuse the
    validated event products instead of re-extracting audio and the ATD ball tracker.
    """
    n_frames = int(tracking.n_frames)
    if features is None:
        if track is None or whistle is None or events is None:
            from .pipeline import prepare_event_products

            track, whistle, events = prepare_event_products(
                tracking, tags, track=track, whistle=whistle, use_audio=use_audio
            )
        features = extract_features(
            tracking, tags, ball_track=track, whistle=whistle, events=events
        )
    if features.start != 0 or features.stop != n_frames:
        raise ValueError("fusion needs full-match features starting at frame 0")
    resolved_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        resolved_weights.update({str(k): float(v) for k, v in weights.items()})
    video = _resample_perception(perception, n_frames)
    atd_ball = np.asarray(tracking.ball, dtype=np.float64)
    atd_present = np.asarray(tracking.ball_present, dtype=bool)
    atd_players = np.asarray(tracking.players, dtype=np.float64)
    atd_teams = np.asarray(tracking.teams, dtype=np.int8)
    atd_player_present = np.asarray(tracking.player_present, dtype=bool)
    if atd_player_present.shape != atd_players.shape[:2]:
        atd_player_present = np.isfinite(atd_players).all(axis=2)

    sources: dict[str, SourceVote] = {}
    sources["atd_ball"] = _geometry_source(
        key="atd_ball",
        ball=atd_ball,
        ball_confidence=atd_present.astype(np.float64),
        players=atd_players,
        teams=np.tile(atd_teams, (n_frames, 1)),
        present=atd_player_present,
        base_weight=resolved_weights["atd_ball"],
        ball_status=np.where(atd_present, 0, 2).astype(np.int8),
        sigma=3.8,
        dead_r=7.5,
    )
    sources["atd_shape"] = _source_atd_shape(features, resolved_weights["atd_shape"])

    video_players = video["player_pitch"]
    video_teams = video["player_team"]
    video_present = video["player_present"]
    # Player evidence uses the best available ball: the own video ball when it is measured, ATD
    # when it is present, and the own prediction only as a fallback.
    player_ball = np.asarray(video["ball_pitch"], dtype=np.float64).copy()
    player_conf = np.asarray(video["ball_confidence"], dtype=np.float64).copy()
    use_video = (video["ball_source"] == 0) & (player_conf >= 0.22)
    use_atd = atd_present & ~use_video
    player_ball[use_atd] = atd_ball[use_atd]
    player_conf[use_atd] = 1.0
    valid_player_ball = np.isfinite(player_ball).all(axis=1)
    player_conf[~valid_player_ball] = 0.0
    sources["video_players"] = _geometry_source(
        key="video_players",
        ball=player_ball,
        ball_confidence=player_conf,
        players=video_players,
        teams=video_teams,
        present=video_present,
        base_weight=resolved_weights["video_players"],
        sigma=3.8,
        dead_r=7.5,
        min_players=6,
    )
    sources["video_ball"] = _source_video_ball(
        features=features,
        video=video,
        atd_players=atd_players,
        atd_teams=atd_teams,
        atd_present=atd_present,
        base_weight=resolved_weights["video_ball"],
    )
    sources["yolo_ball"] = _source_yolo_ball(
        features=features,
        video=video,
        atd_players=atd_players,
        atd_teams=atd_teams,
        atd_present=atd_present,
        base_weight=resolved_weights["yolo_ball"],
    )
    if baseline_codes is None:
        from .pipeline import predict_full

        baseline_codes = predict_full(
            tracking,
            tags,
            params,
            track=track,
            whistle=whistle,
            events=events,
        )
    baseline_codes = np.asarray(baseline_codes, dtype=np.int8)
    hmm_source, hmm_codes, hmm_confidence, hmm_posterior = _source_hmm(
        features, params, resolved_weights["hmm_atd"], baseline_codes=baseline_codes
    )
    sources["hmm_atd"] = hmm_source
    sources["tags"] = _source_tags(features, resolved_weights["tags"])
    sources["audio"] = _source_audio(features, resolved_weights["audio"])
    sources["events"] = _source_events(features, events)

    # Combine the ballots.  The default is a log-opinion pool anchored on the validated HMM
    # posterior: every video/tag/audio source is a likelihood update, not a replacement vote.  A
    # source can move the result only when it is confident enough, and a near-certain HMM state
    # cannot be flipped by a single noisy white blob.  ``linear`` remains available for ablation.
    event_dead = np.asarray(sources["events"].active, dtype=bool)
    if combine_mode == "linear":
        total_weight = np.zeros(n_frames, dtype=np.float64)
        weighted = np.zeros((n_frames, 3), dtype=np.float64)
        for source in sources.values():
            if source.key == "events":
                continue
            weighted += source.weight[:, None] * source.probability
            total_weight += source.weight
        fused = (PRIOR_WEIGHT * PRIOR[None, :] + weighted) / (
            PRIOR_WEIGHT + total_weight[:, None]
        )
    elif combine_mode == "log":
        prior = _normalise_rows(np.asarray(sources["hmm_atd"].probability, dtype=np.float64))
        log_fused = np.log(np.clip(prior, EPS, 1.0))
        log_prior = np.log(np.clip(PRIOR, EPS, 1.0))[None, :]
        for source in sources.values():
            if source.key in {"hmm_atd", "events"}:
                continue
            amount = np.clip(source.weight, 0.0, 1.0)[:, None]
            log_fused += amount * (np.log(np.clip(source.probability, EPS, 1.0)) - log_prior)
        log_fused -= np.max(log_fused, axis=1, keepdims=True)
        fused = np.exp(log_fused)
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode!r}")
    fused = _normalise_rows(fused)
    fused[event_dead] = np.array([0.0, 0.0, 1.0])

    # Consensus ball is a visual aid: a confidence-weighted average of the three ball sources,
    # not an extra ballot.
    consensus = np.full((n_frames, 2), np.nan, dtype=np.float64)
    consensus_conf = np.zeros(n_frames, dtype=np.float64)
    components = [
        (atd_ball, atd_present.astype(np.float64), 1.0),
        (np.asarray(video["ball_pitch"], dtype=np.float64), np.asarray(video["ball_confidence"], dtype=np.float64), 0.8),
        (sources["yolo_ball"].ball, sources["yolo_ball"].weight, 1.0),
    ]
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

    # Decode the fused emissions with the same three-state transition prior as the baseline.
    log_emission = np.log(np.clip(fused, EPS, 1.0))
    codes = np.full(n_frames, X, dtype=np.int8)
    graded = np.zeros(n_frames, dtype=bool)
    for start, stop in GRADED_HALVES:
        start, stop = int(start), int(stop)
        if not (0 <= start < stop <= n_frames):
            continue
        lo = max(0, start - 500)
        hi = min(n_frames, stop + 500)
        local = viterbi(log_emission[lo:hi], params)
        local = merge_isolated_runs(local)
        codes[start:stop] = local[start - lo : stop - lo]
        graded[start:stop] = True
    codes[event_dead & graded] = D

    posterior, confidence, entropy = forward_backward(log_emission, params)
    posterior[event_dead] = np.array([0.0, 0.0, 1.0])
    confidence = np.max(posterior, axis=1)
    entropy = -np.sum(np.where(posterior > 0, posterior * np.log(np.maximum(posterior, EPS)), 0.0), axis=1)
    codes[~graded] = X

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
            "hmm_confidence_mean": float(np.mean(hmm_confidence)) if hmm_confidence.size else 0.0,
            "hmm_posterior_mean": float(np.mean(hmm_posterior)) if hmm_posterior.size else 0.0,
            "perception": bool(perception is not None),
        },
    )


def save_fusion(path: str | Path, result: FusionResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": 1,
        "start": result.start,
        "stop": result.stop,
        "weights": result.meta.get("weights", {}),
        "combine_mode": result.meta.get("combine_mode", "log"),
        "prior": PRIOR.tolist(),
        "perception": bool(result.meta.get("perception", False)),
        "baseline": result.baseline_codes is not None,
        "sources": [
            {
                "key": key,
                "label": source.label,
                "details": sorted(source.details),
            }
            for key, source in result.sources.items()
        ],
    }
    np.savez_compressed(path, **result.to_arrays(), meta=np.asarray(json.dumps(meta)))
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_fusion(path: str | Path) -> FusionResult | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        arrays = {name: archive[name] for name in archive.files if name != "meta"}
    sources: dict[str, SourceVote] = {}
    for item in meta.get("sources", []):
        key = str(item["key"])
        details = {
            str(detail): arrays[f"src_{key}_{detail}"]
            for detail in item.get("details", [])
            if f"src_{key}_{detail}" in arrays
        }
        sources[key] = SourceVote(
            key=key,
            label=str(item.get("label", SOURCE_LABELS.get(key, key))),
            probability=np.asarray(arrays[f"src_{key}_probability"], dtype=np.float64),
            weight=np.asarray(arrays[f"src_{key}_weight"], dtype=np.float64),
            active=np.asarray(arrays[f"src_{key}_active"], dtype=bool),
            status=np.asarray(arrays[f"src_{key}_status"], dtype=np.int8),
            ball=np.asarray(arrays[f"src_{key}_ball"], dtype=np.float64),
            details=details,
        )
    codes = np.asarray(arrays["codes"], dtype=np.int8)
    baseline_codes = (
        np.asarray(arrays["baseline_codes"], dtype=np.int8)
        if meta.get("baseline", True)
        else None
    )
    return FusionResult(
        start=int(meta.get("start", 0)),
        stop=int(meta.get("stop", codes.size)),
        codes=codes,
        fused_probability=np.asarray(arrays["fused_probability"], dtype=np.float64),
        posterior=np.asarray(arrays["posterior"], dtype=np.float64),
        confidence=np.asarray(arrays["confidence"], dtype=np.float64),
        entropy=np.asarray(arrays["entropy"], dtype=np.float64),
        consensus_ball=np.asarray(arrays["consensus_ball"], dtype=np.float64),
        consensus_confidence=np.asarray(arrays["consensus_confidence"], dtype=np.float64),
        event_dead=np.asarray(arrays["event_dead"], dtype=bool),
        sources=sources,
        baseline_codes=baseline_codes,
        meta={
            "weights": meta.get("weights", {}),
            "combine_mode": meta.get("combine_mode", "log"),
            "perception": meta.get("perception", False),
        },
    )
