"""Turn tracking and smart tags into the model's per-frame features.

The design is deliberately small.  The HMM only needs to know how close the ball is to each team,
whether it is out of play, how long it has been missing, where it was last seen, and whether a
smart tag weakly supports a team.  Player identities are not available in the ATD, so the model
never pretends to know them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .aerial import launch_track
from .balltrack import BallTrack, track_ball
from .config import FPS, PITCH, STOP_LIVE_TAG_CODES
from .data import TagTable, TrackingData

# Parked ball near a line: the shape of a throw-in, corner or goal-kick stoppage waiting for the
# restart.  The thresholds are measured on the public ground truth: the ball sits within a couple
# of metres of a boundary at under ~1.5 m/s for seconds, while a live ball near the line is on its
# way through at speed.  ``PARKED_HOLD`` is the time the condition must hold; a ball that flickers
# in and out of tracking is allowed for up to ``PARKED_FILL`` of the window.  The team-speed gate
# is essential: a player standing with the ball near the touchline is live possession, not a
# stoppage, and that false positive was the source of the 15,221 "Balón muerto" run.
PARKED_NEAR = 2.5
PARKED_SPEED = 1.0
PARKED_TEAM_SPEED = 1.35
PARKED_HOLD_S = 0.8
PARKED_FILL = 0.7


def _held(cond: np.ndarray, seconds: float, fill: float) -> np.ndarray:
    """True where ``cond`` held for at least ``fill`` of a trailing window of ``seconds``."""
    n = cond.shape[0]
    window = max(1, int(round(seconds * FPS)))
    if n == 0:
        return np.zeros(0, dtype=bool)
    cum = np.concatenate([[0.0], np.cumsum(cond.astype(np.float64))])
    out = np.zeros(n, dtype=bool)
    if n >= window:
        counts = cum[window:] - cum[:-window]
        out[window - 1:] = counts >= fill * window
    return out


@dataclass(frozen=True)
class Features:
    start: int
    stop: int
    present: np.ndarray
    d_home: np.ndarray
    d_away: np.ndarray
    near_distance: np.ndarray
    near_team: np.ndarray
    near_player: np.ndarray
    second_distance: np.ndarray
    margin: np.ndarray
    near_speed: np.ndarray
    team_speed: np.ndarray
    ball_speed: np.ndarray
    ball_acceleration: np.ndarray
    gap: np.ndarray
    edge: np.ndarray
    last_edge: np.ndarray
    out_of_play: np.ndarray
    last_touch: np.ndarray
    last_touch_age: np.ndarray
    d_home_last: np.ndarray
    d_away_last: np.ndarray
    tag_vote: np.ndarray
    tag_setpiece: np.ndarray
    tag_coverage: np.ndarray
    player_count: np.ndarray
    parked_edge: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Ball-tracker reconstruction and event support.  Defaults keep older callers and tests
    # working when the tracker/audio are not supplied.
    ball_est: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    est_confidence: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    est_source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    est_out: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    est_d_home: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    est_d_away: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    est_near_distance: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    est_near_team: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    whistle_env: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    whistle_score: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    goal_dead: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    stop_dead: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    tag_live: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Causal aerial-duel support (aerial.launch_track): flight frames and the frozen launching
    # team.  Zero-filled for older callers; only ``w_air`` reads them.
    air: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    air_team: np.ndarray = field(default_factory=lambda: np.full(0, -1, dtype=np.int8))
    air_age: np.ndarray = field(default_factory=lambda: np.full(0, 9999, dtype=np.int32))

    def __len__(self) -> int:
        return int(self.stop - self.start)


def _empty_tags(n_frames: int) -> TagTable:
    return TagTable.empty(n_frames)


def extract_features(
    tracking: TrackingData,
    tags: TagTable | None = None,
    observed_mask: np.ndarray | None = None,
    start: int = 0,
    stop: int | None = None,
    ball_track: BallTrack | None = None,
    whistle: Any | None = None,
    events: Any | None = None,
) -> Features:
    """Extract features for ``[start, stop)``.

    ``observed_mask`` is how stress tests remove the ball without editing the tracking file.  A
    masked frame is treated exactly like a frame where the ATD has no ball: the gap grows, the last
    known position stays put, and the emission becomes the missing-ball emission.  The tracker is
    run with the same mask, so it cannot "see through" a removed ball either.

    ``ball_track``, ``whistle``, and ``events`` are optional precomputed products.  Supplying them
    is purely an optimisation; omitting them gives the same features with a fresh tracker and no
    audio/event support.
    """
    n_total = tracking.n_frames
    start = int(start)
    stop = n_total if stop is None else int(stop)
    if start < 0 or stop > n_total or stop <= start:
        raise ValueError(f"invalid feature window [{start}, {stop}) for {n_total} frames")
    if tags is None:
        tags = _empty_tags(n_total)
    if tags.n_frames != n_total:
        raise ValueError(f"tag table has {tags.n_frames} frames; tracking has {n_total}")

    if observed_mask is None:
        observed = np.ones(n_total, dtype=bool)
    else:
        observed = np.asarray(observed_mask, dtype=bool)
        if observed.shape != (n_total,):
            raise ValueError("observed_mask must have one boolean per frame")
    if ball_track is None:
        ball_track = track_ball(tracking, observed_mask=observed)
    elif ball_track.estimate.shape[0] != n_total:
        raise ValueError("ball_track must cover the full tracking sequence")
    present = tracking.ball_present[start:stop] & observed[start:stop]
    n = stop - start

    ball = tracking.ball[start:stop]
    players = tracking.players[start:stop]
    speeds = tracking.speeds[start:stop]
    teams = tracking.teams
    length, width = tracking.pitch

    # (n, P) distances from the ball to each player slot.
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    distance = np.sqrt(dx * dx + dy * dy)

    safe = np.where(np.isfinite(distance), distance, np.inf)
    near_player = np.argmin(safe, axis=1).astype(np.int16)
    near_distance = safe[np.arange(n), near_player].astype(np.float32)
    no_player = ~np.isfinite(near_distance)
    near_distance[no_player] = np.nan
    near_player[no_player] = -1
    near_team = np.where(
        no_player,
        -1,
        teams[np.clip(near_player, 0, max(0, len(teams) - 1))],
    ).astype(np.int8)
    near_team[~present] = -1
    near_player[~present] = -1
    near_distance[~present] = np.nan

    near_speed = np.full(n, np.nan, dtype=np.float32)
    valid_near = present & (near_player >= 0)
    if valid_near.any():
        near_speed[valid_near] = speeds[
            np.flatnonzero(valid_near), near_player[valid_near]
        ]

    # Second-nearest player, useful for a "cluster around the ball" diagnostic.
    d2 = distance.copy()
    d2[np.arange(n), np.clip(near_player, 0, max(0, d2.shape[1] - 1))] = np.inf
    second_distance = np.nanmin(d2, axis=1)
    second_distance[~np.isfinite(second_distance)] = np.nan

    home_mask = teams == 0
    away_mask = teams == 1
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        d_home = np.nanmin(distance[:, home_mask], axis=1) if home_mask.any() else np.full(n, np.nan)
        d_away = np.nanmin(distance[:, away_mask], axis=1) if away_mask.any() else np.full(n, np.nan)
    d_home = np.asarray(d_home, dtype=np.float32)
    d_away = np.asarray(d_away, dtype=np.float32)
    d_home[~present] = np.nan
    d_away[~present] = np.nan
    margin = (second_distance - near_distance).astype(np.float32)

    # Tracker reconstruction.  Distances to the *estimated* ball are what let the HMM keep a
    # possession through a short occlusion instead of falling back to a uniform emission.
    est_ball = np.asarray(ball_track.estimate[start:stop], dtype=np.float32).copy()
    est_confidence = np.asarray(ball_track.confidence[start:stop], dtype=np.float32).copy()
    est_source = np.asarray(ball_track.source[start:stop], dtype=np.int8).copy()
    est_out = np.asarray(ball_track.out_of_play[start:stop], dtype=bool).copy()
    est_valid = est_confidence > 1e-6
    edx = players[:, :, 0] - est_ball[:, None, 0]
    edy = players[:, :, 1] - est_ball[:, None, 1]
    est_distance = np.sqrt(edx * edx + edy * edy)
    est_safe = np.where(np.isfinite(est_distance), est_distance, np.inf)
    est_near_player = np.argmin(est_safe, axis=1).astype(np.int16)
    est_near_distance = est_safe[np.arange(n), est_near_player].astype(np.float32)
    est_no_player = ~np.isfinite(est_near_distance)
    est_near_distance[est_no_player] = np.nan
    est_near_player[est_no_player] = -1
    est_near_team = np.where(
        est_no_player,
        -1,
        teams[np.clip(est_near_player, 0, max(0, len(teams) - 1))],
    ).astype(np.int8)
    est_near_team[~est_valid] = -1
    est_near_distance[~est_valid] = np.nan
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        est_d_home = (
            np.nanmin(est_distance[:, home_mask], axis=1) if home_mask.any() else np.full(n, np.nan)
        )
        est_d_away = (
            np.nanmin(est_distance[:, away_mask], axis=1) if away_mask.any() else np.full(n, np.nan)
        )
    est_d_home = np.asarray(est_d_home, dtype=np.float32)
    est_d_away = np.asarray(est_d_away, dtype=np.float32)
    est_d_home[~est_valid] = np.nan
    est_d_away[~est_valid] = np.nan

    # Audio and event support.  Both are optional and zero-filled when unavailable.
    if whistle is not None and getattr(whistle, "score", None) is not None:
        w_score = np.asarray(whistle.score, dtype=np.float32)
        w_env = np.asarray(whistle.env, dtype=np.float32)
        if w_score.size != n_total:
            w_score = np.zeros(n_total, dtype=np.float32)
            w_env = np.zeros(n_total, dtype=np.float32)
        whistle_score = w_score[start:stop]
        whistle_env = w_env[start:stop]
    else:
        whistle_score = np.zeros(n, dtype=np.float32)
        whistle_env = np.zeros(n, dtype=np.float32)

    if events is not None and getattr(events, "dead_mask", None) is not None:
        goal_mask = np.asarray(events.goal_dead, dtype=bool)
        stop_mask = np.asarray(events.stop_dead, dtype=bool)
        if goal_mask.size != n_total:
            goal_mask = np.zeros(n_total, dtype=bool)
            stop_mask = np.zeros(n_total, dtype=bool)
        goal_dead = goal_mask[start:stop]
        stop_dead = stop_mask[start:stop]
    else:
        goal_dead = np.zeros(n, dtype=bool)
        stop_dead = np.zeros(n, dtype=bool)

    tag_live = np.zeros(n, dtype=bool)
    for event in tags.events:
        if event.code not in STOP_LIVE_TAG_CODES:
            continue
        lo = max(start, int(event.start_frame))
        hi = min(stop, int(event.stop_frame))
        if hi > lo:
            tag_live[lo - start : hi - start] = True

    # Team motion energy: mean speed of the tracked players, smoothed over one second.  Players
    # walk or stand still during a stoppage, so this separates dead ball from live play without
    # looking at the ball at all.  Frames with no tracked player fall back to zero.
    ok = tracking.player_present[start:stop] & np.isfinite(speeds)
    count = ok.sum(axis=1)
    frame_speed = np.where(ok, speeds, 0.0).sum(axis=1) / np.maximum(count, 1)
    frame_speed[count == 0] = 0.0
    kernel = np.ones(FPS, dtype=np.float64) / FPS
    team_speed = np.convolve(
        np.pad(frame_speed.astype(np.float64), FPS // 2, mode="edge"), kernel, mode="valid"
    ).astype(np.float32)

    # Ball speed and acceleration, in metres/second, only across consecutive observations.
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        step = np.sqrt(np.diff(ball[:, 0]) ** 2 + np.diff(ball[:, 1]) ** 2) * FPS
        ok = present[1:] & present[:-1]
        ball_speed[1:] = np.where(ok, step, np.nan)
    ball_acceleration = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        ball_acceleration[1:] = np.diff(ball_speed)

    # Edge distance, last known edge, and gap since the last observation.  These loops are
    # intentionally simple: 199k iterations are cheap and the sequential state is the point.
    edge = np.minimum(
        np.minimum(ball[:, 0], length - ball[:, 0]),
        np.minimum(ball[:, 1], width - ball[:, 1]),
    ).astype(np.float32)
    edge[~present] = np.nan
    last_edge = np.full(n, np.nan, dtype=np.float32)
    gap = np.zeros(n, dtype=np.int32)
    last_seen = -1
    last_edge_value = np.nan
    for k in range(n):
        if present[k]:
            last_edge_value = edge[k]
            last_seen = k
            gap[k] = 0
        elif last_seen >= 0:
            gap[k] = k - last_seen
        else:
            gap[k] = k + 1
        last_edge[k] = last_edge_value

    out_of_play = present & (
        (ball[:, 0] < 0)
        | (ball[:, 0] > length)
        | (ball[:, 1] < 0)
        | (ball[:, 1] > width)
    )

    # Parked ball at a boundary: visible, within ``PARKED_NEAR`` metres of a line, moving slower
    # than ``PARKED_SPEED``, and with the *team* also close to stationary.  Frames beyond the line
    # count too: the restart spot is the same situation with better evidence.  The team-speed gate
    # is what keeps a player standing with the ball at the touchline from becoming a false D.
    parked = (
        present
        & (edge < PARKED_NEAR)
        & np.isfinite(ball_speed)
        & (ball_speed < PARKED_SPEED)
        & (team_speed < PARKED_TEAM_SPEED)
    )
    parked_edge = _held(parked, PARKED_HOLD_S, PARKED_FILL) | out_of_play

    # Last touch: nearest player within 0.8 m.  This is not an identity, just a team hint.
    last_touch = np.full(n, -1, dtype=np.int8)
    last_touch_age = np.full(n, 9999, dtype=np.int32)
    touch = -1
    touch_frame = -1
    for k in range(n):
        if present[k] and near_team[k] >= 0 and np.isfinite(near_distance[k]) and near_distance[k] <= 0.8:
            touch = int(near_team[k])
            touch_frame = k
        last_touch[k] = touch
        last_touch_age[k] = k - touch_frame if touch_frame >= 0 else 9999

    # Per-team distance to the last known ball position.  This is the long-gap evidence: after a
    # tracking hole, the players have moved, but the team shape around the old ball is still a
    # useful prior.
    last_ball_x = np.nan
    last_ball_y = np.nan
    d_home_last = np.full(n, np.nan, dtype=np.float32)
    d_away_last = np.full(n, np.nan, dtype=np.float32)
    for k in range(n):
        if present[k]:
            last_ball_x = float(ball[k, 0])
            last_ball_y = float(ball[k, 1])
        if not (np.isfinite(last_ball_x) and np.isfinite(last_ball_y)):
            continue
        ddx = players[k, :, 0] - last_ball_x
        ddy = players[k, :, 1] - last_ball_y
        dist_last = np.sqrt(ddx * ddx + ddy * ddy)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if home_mask.any():
                d_home_last[k] = np.nanmin(dist_last[home_mask])
            if away_mask.any():
                d_away_last[k] = np.nanmin(dist_last[away_mask])

    # Causal aerial attribution over the full sequence (199k cheap iterations, present-only).
    aerial = launch_track(tracking)

    return Features(
        start=start,
        stop=stop,
        present=present,
        d_home=d_home,
        d_away=d_away,
        near_distance=near_distance,
        near_team=near_team,
        near_player=near_player,
        second_distance=second_distance,
        margin=margin,
        near_speed=near_speed,
        team_speed=team_speed,
        ball_speed=ball_speed,
        ball_acceleration=ball_acceleration,
        gap=gap,
        edge=edge,
        last_edge=last_edge,
        out_of_play=out_of_play,
        last_touch=last_touch,
        last_touch_age=last_touch_age,
        d_home_last=d_home_last,
        d_away_last=d_away_last,
        tag_vote=tags.team_vote[start:stop].astype(np.int16),
        tag_setpiece=tags.setpiece[start:stop].astype(np.int8),
        tag_coverage=tags.coverage[start:stop].astype(bool),
        player_count=tracking.player_present[start:stop].sum(axis=1).astype(np.int16),
        parked_edge=parked_edge,
        ball_est=est_ball,
        est_confidence=est_confidence,
        est_source=est_source,
        est_out=est_out,
        est_d_home=est_d_home,
        est_d_away=est_d_away,
        est_near_distance=est_near_distance,
        est_near_team=est_near_team,
        whistle_env=whistle_env,
        whistle_score=whistle_score,
        goal_dead=goal_dead,
        stop_dead=stop_dead,
        tag_live=tag_live,
        air=aerial.in_flight[start:stop],
        air_team=aerial.team[start:stop],
        air_age=aerial.age[start:stop],
    )


def empty_features(n_frames: int) -> Features:
    """Small helper for smoke tests and visual placeholders."""
    tracking = TrackingData(
        ball=np.full((n_frames, 2), np.nan, np.float32),
        ball_present=np.zeros(n_frames, bool),
        players=np.full((n_frames, 1, 2), np.nan, np.float32),
        speeds=np.full((n_frames, 1), np.nan, np.float32),
        player_present=np.zeros((n_frames, 1), bool),
        teams=np.array([0], dtype=np.int8),
        pitch=PITCH,
        n_frames=n_frames,
    )
    return extract_features(tracking)
