"""Consensus ball from ATD and YOLO under a constant-velocity Kalman filter.

ATD and YOLO support each other as observations of one physical ball: ATD leads while it
has the ball, the best YOLO candidate takes over while ATD is missing.  A per-frame
weighted average of the two inherits both streams' pointing errors at once and reads
half-metre source hops as twelve metres per second; a constant-velocity filter instead
decides which reads belong to one trajectory — an observation accepted by the innovation
gate *is* the consensus position for its frame (a smoothed position would smear the
velocity step every kick detector hunts for), a rejected read loses to inertial
prediction, and gaps run on the prediction until a source re-locks.

The filter is deliberately small: four states (position, velocity), one linear
measurement (position), measurement variance set by the source (ATD is precise while it
tracks, YOLO noisier and scaled by its detection confidence).  Confidence in the output
belongs to observations only — a predicted frame is not a seen ball.
"""

from __future__ import annotations

import numpy as np

from ..config import FPS
from .voters import top_yolo_candidate

OBS_VARIANCE_ATD = 0.09  # m^2: the provided feed is precise while it holds the ball
OBS_VARIANCE_YOLO_MIN = 0.25  # m^2 floor for a confident YOLO detection
OBS_VARIANCE_YOLO_MAX = 25.0  # m^2 ceiling for a barely-alive candidate
ACCEL_SD = 40.0  # m/s^2 process noise: a struck ball must not be smoothed away
INNOVATION_GATE_M = 1.2  # an observation this far off the prediction is a detector hop
INIT_POS_SD = 1.0  # m
INIT_VEL_SD = 10.0  # m/s


def _kalman_cv(
    n_frames: int,
    obs_at: np.ndarray,
    obs_xy: np.ndarray,
    obs_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter (obs_at, obs_xy) frame indices onto every frame of ``n_frames``.

    Returns the consensus position per frame — an accepted observation verbatim, an
    inertial prediction elsewhere — and a mask marking the accepted-observation frames.
    """
    dt = 1.0 / FPS
    f = np.eye(4)
    f[0, 2] = dt
    f[1, 3] = dt
    q = ACCEL_SD**2 * np.array(
        [
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ]
    )
    h = np.zeros((2, 4))
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    obs_index = {int(k): i for i, k in enumerate(obs_at)}
    pos = np.full((n_frames, 2), np.nan)
    observed = np.zeros(n_frames, dtype=bool)
    x = np.zeros(4)
    p = np.diag([INIT_POS_SD**2, INIT_POS_SD**2, INIT_VEL_SD**2, INIT_VEL_SD**2])
    started = False
    for frame in range(n_frames):
        if started:
            x = f @ x
            p = f @ p @ f.T + q
        i = obs_index.get(frame)
        if i is not None:
            r = np.array([[obs_var[i], 0.0], [0.0, obs_var[i]]])
            if not started:
                x[0], x[1] = obs_xy[i]
                started = True
                observed[frame] = True
                pos[frame] = obs_xy[i]
            else:
                innovation = obs_xy[i] - h @ x
                s = h @ p @ h.T + r
                spread = float(np.sqrt(max(np.trace(s) / 2.0, 1e-9)))
                if float(np.linalg.norm(innovation)) <= max(INNOVATION_GATE_M, 5.0 * spread):
                    k = p @ h.T @ np.linalg.inv(s)
                    x = x + k @ innovation
                    kh = np.eye(4) - k @ h
                    p = kh @ p @ kh.T + k @ r @ k.T  # Joseph form: numerically stable
                    observed[frame] = True
                    pos[frame] = obs_xy[i]  # the accepted read, not the smoothed state
                # else: implausible read; keep gliding on the prediction
                else:
                    pos[frame] = x[:2]
        else:
            if started:
                pos[frame] = x[:2]
    return pos, observed


def consensus_ball(
    atd_ball: np.ndarray,
    atd_present: np.ndarray,
    yolo_ball: np.ndarray,
    yolo_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman consensus of ATD-first / YOLO-fallback ball observations.

    Returns ``(position per frame, confidence per frame)``; confidence is 1.0 on an ATD
    observation, the YOLO detection confidence on a YOLO observation, and 0.0 where the
    filter is only predicting — a predicted ball is not an observed ball.
    """
    atd = np.asarray(atd_ball, dtype=np.float64)
    n = atd.shape[0]
    usable_atd = np.asarray(atd_present, dtype=bool) & np.isfinite(atd).all(axis=1)
    yolo = np.asarray(yolo_ball, dtype=np.float64)
    conf = np.asarray(yolo_confidence, dtype=np.float64)
    usable_yolo = ~usable_atd & np.isfinite(yolo).all(axis=1) & (conf > 0)
    obs_xy = np.where(usable_atd[:, None], atd, yolo)
    obs_ok = usable_atd | usable_yolo
    variance = np.where(
        usable_atd,
        OBS_VARIANCE_ATD,
        np.clip(OBS_VARIANCE_YOLO_MIN / np.maximum(conf, 1e-3), OBS_VARIANCE_YOLO_MIN, OBS_VARIANCE_YOLO_MAX),
    )
    obs_at = np.flatnonzero(obs_ok)
    pos, observed = _kalman_cv(n, obs_at, obs_xy[obs_at], variance[obs_at])
    confidence = np.zeros(n, dtype=np.float64)
    confidence[usable_atd] = 1.0
    confidence[usable_yolo] = conf[usable_yolo]
    confidence[~observed] = 0.0
    return pos, confidence


def consensus_from_context(ctx) -> tuple[np.ndarray, np.ndarray]:
    """ATD + best-YOLO consensus for a :class:`~possession_lab.fusion.voters.VotingContext`."""
    yolo, yolo_conf = top_yolo_candidate(ctx)
    return consensus_ball(ctx.atd.ball, ctx.atd.present, yolo, yolo_conf)
