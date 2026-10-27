"""Shared primitives for geometry-based ballots.

A geometry ballot answers one question for several sources: given a ball position with a
confidence and a set of player positions with team labels, how near is the closest player
of each team to the ball?  The same Gaussian-likelihood / dead-ball-sigmoid computation
backs the ATD ballot, the video-player ballot and both video-ball ballots, so it lives
here once instead of in four call sites.
"""

from __future__ import annotations

import numpy as np

from ..config import PITCH
from .ballot import SourceVote, VoterRole

# Prior used only when all sources abstain, and as a small stabiliser in the pool.
PRIOR = np.array([0.48, 0.48, 0.04], dtype=np.float64)
PRIOR_WEIGHT = 0.18
EPS = 1e-9


def normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, EPS)
    total = values.sum(axis=1, keepdims=True)
    return values / np.where(total > 0, total, 1.0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def geometry_ballot(
    *,
    key: str,
    label: str,
    role: VoterRole,
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
            # A non-finite coordinate is a missing player, not an infinitely far one: ``np.min``
            # would let that one slot poison the whole team's distance for the frame.
            distance = np.where(mask & np.isfinite(distance), distance, np.inf)
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
        q_dead = np.where(np.isfinite(d_min), 0.03 + sigmoid((d_min - dead_r) / dead_scale), 0.03) + EPS
        # A ball outside the pitch is a very strong dead-ball observation.
        length, width = PITCH
        outside = (
            (ball[:, 0] < -1.5)
            | (ball[:, 0] > length + 1.5)
            | (ball[:, 1] < -1.5)
            | (ball[:, 1] > width + 1.5)
        )
        q_dead = np.where(outside, 5.0, q_dead)
        probability[enough] = normalise_rows(np.stack([q_home, q_away, q_dead], axis=1)[enough])
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
        label=label,
        role=role,
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
