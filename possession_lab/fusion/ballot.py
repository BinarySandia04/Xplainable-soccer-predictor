"""Data contracts of the multi-source poll.

``SourceVote`` is one source's per-frame ballot and ``FusionResult`` is the tallied
outcome.  Both are plain data: no source computes anything here.  The human-facing
serialisation of a ballot (``ballot_reason``, ``STATUS_NAMES``, the JSON helpers and
``SourceVote.sample``) lives beside it because that rendering *is* part of the ballot
contract the viewer consumes, not a separate concern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..config import FPS, X
from ..model import DEFAULT_PARAMS, ModelParams


class VoterRole(StrEnum):
    """What a ballot is allowed to do inside the pool.

    ``PRIOR``
        The validated baseline the pool anchors on.  In log mode it is the starting
        point, not one more update; in linear mode it carries its full weight.
    ``OVERRIDE``
        A confirmed hard fact.  It never pools: it applies ``D`` after the decode.
    ``REDUNDANT``
        Evidence already contained in the HMM emission.  It stays on the ballot for
        transparency with a near-zero marginal weight, so the same evidence is not
        counted twice.
    ``EVIDENCE``
        An independent observation that has to earn every disagreement.
    """

    PRIOR = "prior"
    OVERRIDE = "override"
    REDUNDANT = "redundant"
    EVIDENCE = "evidence"


STATUS_NAMES = {
    0: "observed",
    1: "estimated",
    2: "abstain",
    3: "event",
}


@dataclass(frozen=True)
class SourceVote:
    """A per-frame ballot cast by one voter."""

    key: str
    label: str
    role: VoterRole = VoterRole.EVIDENCE
    probability: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) H, A, D
    weight: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (N,) actual weight in the pool
    active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # (N,) bool
    status: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))  # (N,) STATUS_NAMES keys
    ball: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (N, 2) pitch metres, NaN when absent
    details: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)  # source-level summary stats, JSON-persisted

    def sample(self, frame: int) -> dict[str, Any]:
        frame = int(frame)
        if frame < 0 or frame >= self.probability.shape[0]:
            raise IndexError(f"frame {frame} outside source length {self.probability.shape[0]}")
        probability = self.probability[frame]
        weight = float(self.weight[frame])
        active = bool(self.active[frame])
        status = int(self.status[frame])
        ball = self.ball[frame]
        result: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "role": VoterRole(self.role).value,
            "active": active,
            "status": status,
            "status_name": STATUS_NAMES.get(status, "unknown"),
            "probability": _json_vector(probability),
            "weight": weight,
            "ball": {
                "x": _finite(ball[0]),
                "y": _finite(ball[1]),
            },
            "reason": ballot_reason(self.key, frame, self.details, active),
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


def ballot_reason(key: str, frame: int, details: dict[str, np.ndarray], active: bool) -> str:
    """Human-readable justification of one ballot at one frame (viewer contract)."""
    if key == "events":
        kind = details.get("kind", np.zeros(1, dtype=np.int8))[frame] if "kind" in details else 0
        return "gol / detención confirmada" if int(kind) else "sin evento"
    if not active:
        return "sin evidencia utilizable"
    if key in {"atd_ball", "video_players", "yolo_ball"}:
        d_home = details.get("d_home", np.full(1, np.nan))[frame]
        d_away = details.get("d_away", np.full(1, np.nan))[frame]
        count = int(details.get("count", np.zeros(1))[frame])
        pieces = []
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
