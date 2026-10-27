"""Causal ball trajectory reconstruction for tracking gaps.

The ATD drops the ball for a few frames during passes, deflections, and occlusions, and for many
seconds while the ball is retrieved or the game is stopped.  This module provides:

1. Causal forward extrapolation with exponential damping: the last observed position and a robust
   least-squares velocity are carried forward, and the estimate is abandoned once the hole exceeds
   ``max_fill_s`` instead of inventing a ball position.
2. A per-frame source code (0 observed, 1 estimated, 2 no estimate) and a confidence that decays
   with the age of the last observation.
3. A pitch-perimeter test that separates an on-pitch occlusion from a ball heading out of play.

The perimeter test is *inspired by* the boundary-grid idea in Kamble, Keskar & Bhurchandi, "A deep
learning ball tracking system in soccer videos", Opto-Electronics Review 27(1):58-69, 2019
(doi:10.1016/j.opelre.2019.02.003). It is not an implementation of it: that paper classifies
pixel-space grid cells behind a CNN ball detector and a median-filter background model, while this
is a margin test on pitch coordinates. Only the concept — a ball inside the perimeter band and
moving outward is leaving play, not occluded — carries over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    BALL_TRACK_DAMPING,
    BALL_TRACK_MAX_FILL_S,
    BALL_TRACK_SPEED_CAP,
    BALL_TRACK_TAU_S,
    BALL_TRACK_VELOCITY_S,
    FPS,
)
from .data import TrackingData


@dataclass(frozen=True)
class BallTrack:
    """Per-frame ball position, velocity, and reliability.

    ``estimate`` is observed where the ATD saw the ball and extrapolated/interpolated inside holes.
    ``confidence`` is 1 on observations and decays with the age or span of the occlusion.
    ``source`` is 0 for observed, 1 for extrapolated/interpolated, and 2 for no estimate.
    """

    estimate: np.ndarray  # (N, 2) float32, NaN where there is no estimate
    velocity: np.ndarray  # (N, 2) float32, m/s, NaN where unknown
    confidence: np.ndarray  # (N,) float32, 0..1
    observed: np.ndarray  # (N,) bool
    imputed: np.ndarray  # (N,) bool
    gap_s: np.ndarray  # (N,) float32, seconds since the last observation
    source: np.ndarray  # (N,) int8
    out_of_play: np.ndarray  # (N,) bool, estimate beyond the pitch by > 1 m

    def __len__(self) -> int:
        return int(self.estimate.shape[0])


def _is_boundary_grid(
    x: float, y: float, length: float, width: float, margin: float = 2.0
) -> bool:
    """True when a pitch coordinate sits inside the perimeter band of width ``margin``.

    This is the boundary-grid *concept* from Kamble et al. (2019) reduced to one margin test on
    pitch metres; see the module docstring for what does and does not carry over from the paper.
    """
    return x <= margin or x >= length - margin or y <= margin or y >= width - margin


def _velocity_from_window(
    history: list[tuple[int, float, float]], speed_cap: float
) -> tuple[float, float, bool]:
    """Least-squares velocity from the recent observations, with a speed sanity cap."""
    if len(history) < 2:
        return 0.0, 0.0, False
    arr = np.asarray(history, dtype=np.float64)
    t = (arr[:, 0] - arr[-1, 0]) / FPS
    x = arr[:, 1]
    y = arr[:, 2]
    denom = float(np.dot(t, t))
    if denom <= 1e-9:
        return 0.0, 0.0, False
    vx = float(np.dot(t, x - x.mean()) / denom)
    vy = float(np.dot(t, y - y.mean()) / denom)
    speed = float(np.hypot(vx, vy))
    reliable = speed <= speed_cap
    if not reliable and speed > 0:
        scale = speed_cap / speed
        vx *= scale
        vy *= scale
    return vx, vy, reliable


def track_ball(
    tracking: TrackingData,
    observed_mask: np.ndarray | None = None,
    *,
    max_fill_s: float = BALL_TRACK_MAX_FILL_S,
    tau_s: float = BALL_TRACK_TAU_S,
    velocity_window_s: float = BALL_TRACK_VELOCITY_S,
    speed_cap: float = BALL_TRACK_SPEED_CAP,
    damping: float = BALL_TRACK_DAMPING,
) -> BallTrack:
    """Reconstruct the ball trajectory across tracking dropouts.

    The tracker is strictly causal: it carries the last observed position and a robust velocity
    through a hole, damps it for rolling friction, and stops estimating once the hole exceeds
    ``max_fill_s`` rather than inventing a ball position.

    Parameters
    ----------
    tracking : TrackingData
        Match tracking coordinates and pitch dimensions.
    observed_mask : np.ndarray, optional
        Per-frame mask of valid observations.
    max_fill_s : float
        Maximum causal fill horizon in seconds.
    tau_s : float
        Confidence half-life scale in seconds.
    velocity_window_s : float
        Rolling window for least-squares velocity fitting.
    speed_cap : float
        Maximum physical soccer ball speed in m/s (default 38 m/s).
    damping : float
        Exponential velocity damping parameter for open extrapolation.
    """
    n = int(tracking.n_frames)
    ball = np.asarray(tracking.ball, dtype=np.float32)
    present = np.asarray(tracking.ball_present, dtype=bool).copy()
    if observed_mask is not None:
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != (n,):
            raise ValueError("observed_mask must have one boolean per frame")
        present &= mask
    present &= np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])

    estimate = np.full((n, 2), np.nan, dtype=np.float32)
    velocity = np.full((n, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    gap_s = np.zeros(n, dtype=np.float32)
    source = np.full(n, 2, dtype=np.int8)
    out_of_play = np.zeros(n, dtype=bool)

    length, width = tracking.pitch
    window = max(1, int(round(velocity_window_s * FPS)))

    history: list[tuple[int, float, float]] = []
    last_index = -1
    last_x = np.nan
    last_y = np.nan
    last_vx = 0.0
    last_vy = 0.0
    last_reliable = False

    # 1. Causal base pass
    for k in range(n):
        if present[k]:
            x = float(ball[k, 0])
            y = float(ball[k, 1])
            history.append((k, x, y))
            while history and k - history[0][0] > window:
                history.pop(0)
            vx, vy, reliable = _velocity_from_window(history, speed_cap)
            estimate[k] = (x, y)
            velocity[k] = (vx, vy)
            confidence[k] = 1.0
            gap_s[k] = 0.0
            source[k] = 0
            out_of_play[k] = x < -1.0 or x > length + 1.0 or y < -1.0 or y > width + 1.0
            last_index = k
            last_x, last_y = x, y
            last_vx, last_vy = vx, vy
            last_reliable = reliable
            continue

        if last_index < 0:
            # No observation yet
            gap_s[k] = (k + 1) / FPS
            continue

        dt = (k - last_index) / FPS
        gap_s[k] = dt
        if dt > max_fill_s or not np.isfinite(last_x) or not np.isfinite(last_y):
            continue

        decay = np.exp(-damping * dt)
        if damping > 1e-6:
            displacement = np.array([last_vx, last_vy]) * (1.0 - decay) / damping
        else:
            displacement = np.array([last_vx, last_vy]) * dt
        px = last_x + float(displacement[0])
        py = last_y + float(displacement[1])
        estimate[k] = (px, py)
        velocity[k] = (last_vx * decay, last_vy * decay)
        base = float(np.exp(-((dt / max(tau_s, 1e-3)) ** 2)))
        if not last_reliable:
            base *= 0.5
        confidence[k] = base
        source[k] = 1

        # Perimeter band: a ball in it and still moving outward is leaving play, not occluded.
        is_outside = px < -1.0 or px > length + 1.0 or py < -1.0 or py > width + 1.0
        in_boundary_grid = _is_boundary_grid(px, py, length, width, margin=2.5)
        moving_outward = (
            (px < 5.0 and last_vx < -0.5)
            or (px > length - 5.0 and last_vx > 0.5)
            or (py < 5.0 and last_vy < -0.5)
            or (py > width - 5.0 and last_vy > 0.5)
        )
        out_of_play[k] = is_outside or (in_boundary_grid and moving_outward and dt > 0.4)

    # 2. Bidirectional smoothing was removed here: no caller, no test, no measured benefit.

    return BallTrack(
        estimate=estimate,
        velocity=velocity,
        confidence=confidence,
        observed=present,
        imputed=source == 1,
        gap_s=gap_s,
        source=source,
        out_of_play=out_of_play,
    )
