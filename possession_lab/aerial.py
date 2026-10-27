"""Causal aerial-duel attribution: who last played the ball while it was in the air.

The ATD ball is a 2-D ground projection.  While the ball is flying, the nearest tracked player
is whoever is *under* the flight, not who touched it, and a proximity emission hands the duel to
the wrong team (measured on the user-truth segment: the nearest player's team agrees with the
label only 72.6 % of the time, while the team nearest at the launch agrees 91 %).  This module
marks flight frames causally and freezes the team that was nearest at the launch — the last
toucher — for the duration of the flight.  Two signals start a flight:

* dynamics: the ball speed reaches ``launch_speed`` m/s after gaining at least ``launch_dv`` m/s
  against the previous observation (the impulse of a kick or header);
* height: the optional third ATD ball component (metres above ground) is at least ``height_air``.

The hold ends when the ball drops below ``landing_speed`` and below the height threshold, when a
new impulse (a rekick) re-anchors the attribution, or after ``max_air_s`` seconds.  Only present
ball frames are ever marked, and every decision uses the current and past frames only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FPS
from .data import TrackingData

AIR_LAUNCH_SPEED = 8.0    # m/s; a struck ball, above any run speed (players top out ~9 m/s)
AIR_LANDING_SPEED = 2.5   # m/s; below this and near the ground the ball is in play again
AIR_LAUNCH_DV = 4.0       # m/s gain in one observed step marks the impulse
AIR_HEIGHT = 1.8          # m; above head height — unambiguous flight when the provider has z
AIR_MAX_S = 3.0           # never hold one toucher across more than this much flight


@dataclass(frozen=True)
class AerialState:
    in_flight: np.ndarray  # (N,) bool, True only on observed ball frames
    team: np.ndarray       # (N,) int8 team nearest at the launch, -1 when unknown
    age: np.ndarray        # (N,) int32 frames since the launch


def _nearest_team(tracking: TrackingData, k: int) -> int:
    players = np.asarray(tracking.players[k], dtype=np.float64)
    ok = tracking.player_present[k] & np.isfinite(players).all(axis=1)
    if not ok.any():
        return -1
    ball = tracking.ball[k]
    distance = np.linalg.norm(players[ok] - ball[None, :], axis=1)
    slot = int(np.flatnonzero(ok)[int(np.argmin(distance))])
    team = int(tracking.teams[slot])
    return team if team >= 0 else -1


def launch_track(
    tracking: TrackingData,
    *,
    launch_speed: float = AIR_LAUNCH_SPEED,
    landing_speed: float = AIR_LANDING_SPEED,
    launch_dv: float = AIR_LAUNCH_DV,
    height_air: float = AIR_HEIGHT,
    max_air_s: float = AIR_MAX_S,
) -> AerialState:
    """Causally mark flight frames and freeze the launching team over each flight."""
    n = int(tracking.n_frames)
    ball = np.asarray(tracking.ball, dtype=np.float64)
    present = (
        np.asarray(tracking.ball_present, dtype=bool)
        & np.isfinite(ball[:, 0])
        & np.isfinite(ball[:, 1])
    )
    if tracking.ball_height is None:
        height = np.full(n, np.nan, dtype=np.float64)
    else:
        height = np.asarray(tracking.ball_height, dtype=np.float64)

    speed = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        step = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
        ok = present[1:] & present[:-1]
        speed[1:] = np.where(ok, step, np.nan)

    in_flight = np.zeros(n, dtype=bool)
    team = np.full(n, -1, dtype=np.int8)
    age = np.full(n, 9999, dtype=np.int32)
    max_air = int(round(max_air_s * FPS))

    air = False
    cur_team = -1
    launch = -1
    prev_speed = np.nan
    for k in range(n):
        if not present[k]:
            # The flight window stays open across a hole; unobserved frames emit no air signal.
            continue
        s = float(speed[k])
        h_air = bool(np.isfinite(height[k]) and height[k] >= height_air)
        dv = s - prev_speed if np.isfinite(s) and np.isfinite(prev_speed) else np.nan
        struck = bool(np.isfinite(s) and s >= launch_speed and np.isfinite(dv) and dv >= launch_dv)

        if not air:
            if struck or h_air:
                air = True
                launch = k
                cur_team = _nearest_team(tracking, k)
        else:
            if struck:
                launch = k
                cur_team = _nearest_team(tracking, k)
            elif (k - launch) > max_air or (np.isfinite(s) and s < landing_speed and not h_air):
                air = False
                cur_team = -1

        if air:
            in_flight[k] = True
            team[k] = cur_team
            age[k] = k - launch
        if np.isfinite(s):
            prev_speed = s

    return AerialState(in_flight=in_flight, team=team, age=age)
