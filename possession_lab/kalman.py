"""Causal Kalman ball layer used by the reviewer.

The production tracker in :mod:`possession_lab.balltrack` is deliberately conservative: it keeps
an observation and extrapolates a damped constant-velocity estimate for at most 1.2 s.  This module
adds a second, independent view for inspection.  It filters the observed ATD ball with a
constant-velocity Kalman model, rejects isolated jumps, and can predict through a longer hole
(the viewer labels those points as estimates).  It is not used by the model or event detector.

The filter is causal: frame ``k`` only sees observations at frames ``<= k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FPS
from .data import TrackingData


@dataclass(frozen=True)
class KalmanTrack:
    """Per-frame output of the causal constant-velocity Kalman filter.

    ``source`` is 0 after a measurement update, 1 while predicting through a hole, and 2 when
    there is no honest estimate (no observation yet or the prediction horizon was exceeded).
    """

    estimate: np.ndarray  # (N, 2) float32, NaN where unavailable
    velocity: np.ndarray  # (N, 2) float32, m/s
    confidence: np.ndarray  # (N,) float32, 0..1
    observed: np.ndarray  # (N,) bool
    source: np.ndarray  # (N,) int8
    gap_s: np.ndarray  # (N,) float32, seconds since the last accepted observation
    out_of_play: np.ndarray  # (N,) bool, estimate more than 1 m outside the pitch

    def __len__(self) -> int:
        return int(self.estimate.shape[0])


def _outside(x: float, y: float, length: float, width: float) -> bool:
    return x < -1.0 or x > length + 1.0 or y < -1.0 or y > width + 1.0


def kalman_ball_track(
    tracking: TrackingData,
    observed_mask: np.ndarray | None = None,
    *,
    max_prediction_s: float = 5.0,
    process_noise: float = 1.2,
    measurement_noise: float = 0.6,
    speed_cap: float = 38.0,
    gate: float = 25.0,
    reset_after_s: float = 1.0,
) -> KalmanTrack:
    """Filter the raw ATD ball positions with a causal constant-velocity Kalman model.

    ``process_noise`` is the acceleration standard deviation (m/s²), ``measurement_noise`` is the
    position standard deviation (m), and ``gate`` is the squared Mahalanobis threshold for a
    measurement update.  Isolated jumps are ignored; after three consecutive rejects the filter
    resets to the new observation, and it also resets after a hole longer than ``reset_after_s``.
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

    length, width = (float(v) for v in tracking.pitch)
    dt = 1.0 / float(FPS)
    max_prediction = max(0.0, float(max_prediction_s))
    measurement_var = max(1e-6, float(measurement_noise) ** 2)
    acceleration_var = max(1e-6, float(process_noise) ** 2)
    max_speed = max(0.1, float(speed_cap))

    # State is [x, y, vx, vy]; H observes x and y.
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    I = np.eye(4, dtype=np.float64)
    F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Q = acceleration_var * np.array(
        [
            [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
            [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
            [dt**3 / 2.0, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2.0, 0.0, dt**2],
        ],
        dtype=np.float64,
    )
    R = np.eye(2, dtype=np.float64) * measurement_var

    state: np.ndarray | None = None
    covariance = np.zeros((4, 4), dtype=np.float64)
    last_update_frame = -1
    rejects = 0

    def reset(x: float, y: float) -> None:
        nonlocal state, covariance, last_update_frame, rejects
        state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        covariance = np.diag([measurement_var, measurement_var, 25.0, 25.0]).astype(np.float64)
        last_update_frame = -1
        rejects = 0

    def publish(k: int, current_source: int, gap_seconds: float) -> None:
        if state is None:
            return
        x, y, vx, vy = (float(v) for v in state)
        estimate[k] = (x, y)
        velocity[k] = (vx, vy)
        confidence[k] = float(np.exp(-((max(0.0, gap_seconds) / 1.0) ** 2)))
        gap_s[k] = float(gap_seconds)
        source[k] = current_source
        out_of_play[k] = _outside(x, y, length, width)

    for k in range(n):
        if state is None:
            if not present[k]:
                gap_s[k] = (k + 1) / float(FPS)
                continue
            x, y = float(ball[k, 0]), float(ball[k, 1])
            reset(x, y)
            state[0], state[1] = x, y
            last_update_frame = k
            publish(k, 0, 0.0)
            confidence[k] = 1.0
            continue

        # Predict one frame.
        state = F @ state
        covariance = F @ covariance @ F.T + Q
        speed = float(np.hypot(state[2], state[3]))
        if speed > max_speed and speed > 0.0:
            state[2] *= max_speed / speed
            state[3] *= max_speed / speed

        gap_since_update = (k - last_update_frame) / float(FPS)
        accepted = False
        if present[k]:
            z = np.array([float(ball[k, 0]), float(ball[k, 1])], dtype=np.float64)
            innovation = z - H @ state
            innovation_cov = H @ covariance @ H.T + R
            try:
                distance = float(innovation.T @ np.linalg.solve(innovation_cov, innovation))
            except np.linalg.LinAlgError:
                distance = float("inf")

            must_reset = gap_since_update > max(0.0, float(reset_after_s)) or rejects >= 3
            if must_reset:
                reset(float(z[0]), float(z[1]))
                last_update_frame = k
                publish(k, 0, 0.0)
                confidence[k] = 1.0
                accepted = True
            elif distance <= gate:
                gain = covariance @ H.T @ np.linalg.inv(innovation_cov)
                state = state + gain @ innovation
                covariance = (I - gain @ H) @ covariance
                # Keep the covariance symmetric despite finite-precision drift.
                covariance = (covariance + covariance.T) / 2.0
                last_update_frame = k
                rejects = 0
                publish(k, 0, 0.0)
                confidence[k] = 1.0
                accepted = True
            else:
                rejects += 1

        if accepted:
            continue

        gap_since_update = (k - last_update_frame) / float(FPS)
        if gap_since_update <= max_prediction:
            publish(k, 1, gap_since_update)
        else:
            gap_s[k] = gap_since_update
            source[k] = 2

    return KalmanTrack(
        estimate=estimate,
        velocity=velocity,
        confidence=confidence,
        observed=present,
        source=source,
        gap_s=gap_s,
        out_of_play=out_of_play,
    )
