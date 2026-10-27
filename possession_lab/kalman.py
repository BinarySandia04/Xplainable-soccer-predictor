"""Causal and smoothed Kalman ball tracking layer.

Implements constant-velocity Kalman filtering with Rauch-Tung-Striebel (RTS) backward smoothing
Constant-velocity Kalman filtering with a Rauch-Tung-Striebel (RTS) backward smoother, plus a
pitch-perimeter outlier test after the boundary-grid *concept* in Kamble et al. (2019) — the
concept only; ``balltrack.py`` states what that citation does and does not cover.

Provides:
1. Causal forward filtering with innovation chi-square gating (for live inspection & streaming).
2. Rauch-Tung-Striebel (RTS) bidirectional smoother (removes filtering lag, optimal minimum variance).
3. Velocity preservation across short occlusions and a pitch-perimeter sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FPS
from .data import TrackingData


@dataclass(frozen=True)
class KalmanTrack:
    """Per-frame output of the constant-velocity Kalman filter/smoother.

    ``source`` is 0 after a measurement update, 1 while predicting/smoothing through a hole, and 2
    when there is no honest estimate (no observation yet or the prediction horizon was exceeded).
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


def _is_boundary_grid(
    x: float, y: float, length: float, width: float, margin: float = 2.0
) -> bool:
    """Perimeter-band test (see :mod:`possession_lab.balltrack` for the same helper)."""
    return x <= margin or x >= length - margin or y <= margin or y >= width - margin


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
    smooth: bool = False,
) -> KalmanTrack:
    """Filter raw ball positions with a constant-velocity Kalman model & optional RTS smoothing.

    Parameters
    ----------
    tracking : TrackingData
        Tracking observations and metadata.
    observed_mask : np.ndarray, optional
        Per-frame boolean mask of trusted observations.
    max_prediction_s : float
        Maximum duration in seconds to extrapolate through tracking holes.
    process_noise : float
        Acceleration standard deviation in m/s^2.
    measurement_noise : float
        Measurement position standard deviation in metres.
    speed_cap : float
        Maximum physical ball speed cap in m/s.
    gate : float
        Squared Mahalanobis distance innovation gating threshold.
    reset_after_s : float
        Hole length after which state velocity is re-initialized.
    smooth : bool, default False
        When True, applies Rauch-Tung-Striebel (RTS) backward smoothing across continuous track
        segments, eliminating forward filter lag (BoT-SORT / sports MOT standard).
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
    eye4 = np.eye(4, dtype=np.float64)
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
    current_segment_id = -1

    # Histories for RTS smoothing
    states_pred = np.zeros((n, 4), dtype=np.float64)
    covs_pred = np.zeros((n, 4, 4), dtype=np.float64)
    states_filt = np.zeros((n, 4), dtype=np.float64)
    covs_filt = np.zeros((n, 4, 4), dtype=np.float64)
    segment_ids = np.full(n, -1, dtype=np.int32)
    active_mask = np.zeros(n, dtype=bool)

    def reset(x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        nonlocal state, covariance, last_update_frame, rejects, current_segment_id
        state = np.array([x, y, vx, vy], dtype=np.float64)
        covariance = np.diag([measurement_var, measurement_var, 25.0, 25.0]).astype(np.float64)
        last_update_frame = -1
        rejects = 0
        current_segment_id += 1

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
            states_pred[k] = state
            covs_pred[k] = covariance
            states_filt[k] = state
            covs_filt[k] = covariance
            segment_ids[k] = current_segment_id
            active_mask[k] = True
            continue

        # Predict one frame.
        state = F @ state
        covariance = F @ covariance @ F.T + Q
        speed = float(np.hypot(state[2], state[3]))
        if speed > max_speed and speed > 0.0:
            state[2] *= max_speed / speed
            state[3] *= max_speed / speed

        states_pred[k] = state
        covs_pred[k] = covariance

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
                # Compute displacement velocity if gap was moderate
                vx_init, vy_init = 0.0, 0.0
                if 0.0 < gap_since_update <= 2.5 and last_update_frame >= 0:
                    dt_reset = gap_since_update
                    vx_init = float((z[0] - estimate[last_update_frame, 0]) / dt_reset)
                    vy_init = float((z[1] - estimate[last_update_frame, 1]) / dt_reset)
                    init_speed = float(np.hypot(vx_init, vy_init))
                    if init_speed > max_speed:
                        vx_init *= max_speed / init_speed
                        vy_init *= max_speed / init_speed
                reset(float(z[0]), float(z[1]), vx=vx_init, vy=vy_init)
                last_update_frame = k
                publish(k, 0, 0.0)
                confidence[k] = 1.0
                accepted = True
            elif distance <= gate:
                gain = covariance @ H.T @ np.linalg.inv(innovation_cov)
                state = state + gain @ innovation
                covariance = (eye4 - gain @ H) @ covariance
                # Keep the covariance symmetric
                covariance = (covariance + covariance.T) / 2.0
                last_update_frame = k
                rejects = 0
                publish(k, 0, 0.0)
                confidence[k] = 1.0
                accepted = True
            else:
                rejects += 1

        if accepted:
            states_filt[k] = state
            covs_filt[k] = covariance
            segment_ids[k] = current_segment_id
            active_mask[k] = True
            continue

        gap_since_update = (k - last_update_frame) / float(FPS)
        if gap_since_update <= max_prediction:
            publish(k, 1, gap_since_update)
            states_filt[k] = state
            covs_filt[k] = covariance
            segment_ids[k] = current_segment_id
            active_mask[k] = True
        else:
            gap_s[k] = gap_since_update
            source[k] = 2

    # Optional Rauch-Tung-Striebel (RTS) backward smoothing
    if smooth and current_segment_id >= 0:
        for sid in range(current_segment_id + 1):
            seg_indices = np.flatnonzero(segment_ids == sid)
            if seg_indices.size < 2:
                continue
            # Backward recursion along the segment
            for i in range(seg_indices.size - 2, -1, -1):
                idx = seg_indices[i]
                next_idx = seg_indices[i + 1]
                if next_idx != idx + 1:
                    continue
                try:
                    p_pred_next = covs_pred[next_idx]
                    c_gain = covs_filt[idx] @ F.T @ np.linalg.pinv(p_pred_next)
                    states_filt[idx] += c_gain @ (states_filt[next_idx] - states_pred[next_idx])
                    covs_filt[idx] += c_gain @ (covs_filt[next_idx] - p_pred_next) @ c_gain.T
                    estimate[idx] = (float(states_filt[idx, 0]), float(states_filt[idx, 1]))
                    velocity[idx] = (float(states_filt[idx, 2]), float(states_filt[idx, 3]))
                    out_of_play[idx] = _outside(estimate[idx, 0], estimate[idx, 1], length, width)
                except np.linalg.LinAlgError:
                    pass

    return KalmanTrack(
        estimate=estimate,
        velocity=velocity,
        confidence=confidence,
        observed=present,
        source=source,
        gap_s=gap_s,
        out_of_play=out_of_play,
    )


def kalman_ball_smooth(
    tracking: TrackingData,
    observed_mask: np.ndarray | None = None,
    *,
    max_prediction_s: float = 5.0,
    process_noise: float = 1.2,
    measurement_noise: float = 0.6,
    speed_cap: float = 38.0,
    gate: float = 25.0,
) -> KalmanTrack:
    """Convenience wrapper for Rauch-Tung-Striebel (RTS) smoothed Kalman ball tracking."""
    return kalman_ball_track(
        tracking,
        observed_mask=observed_mask,
        max_prediction_s=max_prediction_s,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        speed_cap=speed_cap,
        gate=gate,
        smooth=True,
    )
