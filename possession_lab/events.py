"""Goal and confirmed-stoppage events from tracking, smart tags, and audio.

Two things are deliberately *not* trusted here:

* the ``GOALS`` smart-tag interval on its own.  The tag is often a long window around a shot,
  celebration, or a disallowed goal; a goal is only accepted when the ball trajectory reaches the
  goal mouth, or when a tagged long disappearance is backed by a celebration and a kick-off.
* the raw whistle score.  Crowd noise and players' shouts produce loud 2.4-3.8 kHz peaks during
  live play.  A whistle is used as a *candidate* that must be followed by sustained low player
  motion before the stoppage is confirmed.

The resulting intervals are used as high-confidence overrides after the HMM decode.  Everything
else (possession, ordinary out-of-play) remains the HMM's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .balltrack import BallTrack, track_ball
from .config import (
    FPS,
    GOAL_FALLBACK_GAP_S,
    GOAL_HALF_WIDTH,
    GOAL_LOOKAHEAD_S,
    GOAL_MAX_DEFICIT,
    GOAL_MIN_SPEED,
    GOAL_Y_MARGIN,
    GRADED_HALVES,
    STOP_CONTROL_R,
    STOP_FAILED_RESTART_S,
    STOP_FAILED_RESTART_SKIP_S,
    STOP_IN_RUN_ACTIVE_FRAMES,
    STOP_IN_RUN_BALL_SPEED,
    STOP_IN_RUN_WHISTLE_S,
    STOP_LIVE_TAG_CODES,
    STOP_MIN_LOW_S,
    STOP_MIN_TRACKED_PLAYERS,
    STOP_RESTART_CONTROL_R,
    STOP_RESTART_FRAMES,
    STOP_RESTART_SPEED,
    STOP_TEAM_SPEED,
    STOP_VISIBILITY_GAP_S,
    STOP_WHISTLE_LOOKBACK_S,
)
from .data import TagTable, TrackingData


@dataclass(frozen=True)
class GoalEvent:
    frame: int
    restart: int
    last_frame: int
    last_position: tuple[float, float]
    velocity: tuple[float, float]
    goal_line: float
    crossing_y: float
    confidence: float
    kind: str
    tag_start: int
    tag_stop: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "goal",
            "frame": int(self.frame),
            "restart": int(self.restart),
            "last_frame": int(self.last_frame),
            "last_position": [float(self.last_position[0]), float(self.last_position[1])],
            "velocity": [float(self.velocity[0]), float(self.velocity[1])],
            "goal_line": float(self.goal_line),
            "crossing_y": float(self.crossing_y),
            "confidence": float(self.confidence),
            "kind": self.kind,
            "tag": [int(self.tag_start), int(self.tag_stop)],
        }


@dataclass(frozen=True)
class StopEvent:
    start: int
    end: int
    trigger: str
    confidence: float
    whistle_frame: int | None = None
    whistle_peak: float | None = None
    low_start: int | None = None
    low_stop: int | None = None
    restart: int | None = None
    restart_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "stoppage",
            "start": int(self.start),
            "end": int(self.end),
            "trigger": self.trigger,
            "confidence": float(self.confidence),
            "whistle_frame": None if self.whistle_frame is None else int(self.whistle_frame),
            "whistle_peak": None if self.whistle_peak is None else float(self.whistle_peak),
            "low_start": None if self.low_start is None else int(self.low_start),
            "low_stop": None if self.low_stop is None else int(self.low_stop),
            "restart": None if self.restart is None else int(self.restart),
            "restart_failed": bool(self.restart_failed),
        }


@dataclass(frozen=True)
class EventData:
    goals: tuple[GoalEvent, ...] = field(default_factory=tuple)
    stoppages: tuple[StopEvent, ...] = field(default_factory=tuple)
    goal_dead: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    stop_dead: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    dead_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.dead_mask.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goals": [g.to_dict() for g in self.goals],
            "stoppages": [s.to_dict() for s in self.stoppages],
            "meta": dict(self.meta),
        }


def _runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [
        (int(start), int(stop))
        for start, stop in zip(edges[::2], edges[1::2])
        if int(stop) - int(start) >= min_len
    ]


def _run_end_containing(mask: np.ndarray, start: int, lookback: int, n: int) -> int | None:
    """End (exclusive) of the True-run of ``mask`` that *contains* ``start``, else ``None``.

    The scan begins ``lookback`` frames earlier so a run that started before the trigger is still
    found.  Requiring containment matters: a run that merely begins before ``start`` is not the
    stoppage's low-motion interval, and treating it as one misplaced the dead-ball end and let a
    hard ``D`` override bleed into live frames.
    """
    offset = max(0, int(start) - int(lookback))
    for run_start, run_stop in _runs(np.asarray(mask, dtype=bool)[offset:], min_len=1):
        if offset + run_start <= start < offset + run_stop:
            return min(int(n), offset + run_stop)
    return None


def _merge_stop_events(candidates: list[StopEvent]) -> list[StopEvent]:
    """Join overlapping or adjacent stoppage intervals into one per stoppage.

    ``candidates`` must be sorted by ``(start, end)``.  Intervals that touch within a second are
    joined so a throw-in is not split into a dozen one-second pieces: the earliest start and the
    latest end win, the stronger trigger survives, and every optional field keeps the first value
    that is actually set.  ``is not None`` is used rather than truthiness because frame ``0`` and
    peak ``0.0`` are legitimate values, not missing ones.
    """
    merged: list[StopEvent] = []
    for event in candidates:
        if merged and event.start <= merged[-1].end + FPS:
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                start=min(previous.start, event.start),
                end=max(previous.end, event.end),
                trigger=previous.trigger
                if previous.confidence >= event.confidence
                else event.trigger,
                confidence=max(previous.confidence, event.confidence),
                whistle_frame=(
                    previous.whistle_frame
                    if previous.whistle_frame is not None
                    else event.whistle_frame
                ),
                whistle_peak=(
                    previous.whistle_peak
                    if previous.whistle_peak is not None
                    else event.whistle_peak
                ),
                low_start=(
                    previous.low_start if previous.low_start is not None else event.low_start
                ),
                low_stop=previous.low_stop if previous.low_stop is not None else event.low_stop,
                restart=max(
                    (frame for frame in (previous.restart, event.restart) if frame is not None),
                    default=None,
                ),
            )
        else:
            merged.append(event)
    return merged


def _missing_runs(track: BallTrack) -> list[tuple[int, int]]:
    return _runs(~np.asarray(track.observed, dtype=bool), min_len=1)


def _short_velocity(
    track: BallTrack, last: int, window_s: float = 0.60, speed_cap: float = 38.0
) -> tuple[float, float]:
    """Velocity over the last few observations, for shot extrapolation.

    The tracker's regular velocity is deliberately smoothed over 0.6 s so a gap is filled
    conservatively.  A shot needs the shortest reliable window instead: using the smoothed value
    would turn a 14 m/s shot into 3 m/s and move the goal twenty frames later.
    """
    if last < 0 or not np.isfinite(track.estimate[last, 0]):
        return 0.0, 0.0
    span = max(1, int(round(window_s * FPS)))
    candidates = [
        i
        for i in range(max(0, last - span), last + 1)
        if track.observed[i] and np.isfinite(track.estimate[i, 0])
    ]
    if len(candidates) >= 3:
        arr = np.asarray([(i, *track.estimate[i]) for i in candidates], dtype=np.float64)
        # Center the time axis.  With an uncentered axis the covariance numerator and the
        # denominator ``dot(t,t)`` are not the least-squares slope, which halved the measured shot
        # speed and put the first goal almost two seconds late.
        t = arr[:, 0] - arr[:, 0].mean()
        denom = float(np.dot(t, t))
        if denom > 1e-9:
            vx = float(np.dot(t, arr[:, 1] - arr[:, 1].mean()) / denom) * FPS
            vy = float(np.dot(t, arr[:, 2] - arr[:, 2].mean()) / denom) * FPS
            speed = float(np.hypot(vx, vy))
            if speed > speed_cap and speed > 0:
                scale = speed_cap / speed
                vx *= scale
                vy *= scale
            return vx, vy
    return float(track.velocity[last, 0]), float(track.velocity[last, 1])


def _goal_crossing(
    px: float,
    py: float,
    vx: float,
    vy: float,
    length: float,
    width: float,
) -> tuple[float, float, float] | None:
    """Return ``(goal_line, crossing_y, confidence)`` for a shot that reaches a goal mouth."""
    best: tuple[float, float, float] | None = None
    for line in (0.0, float(length)):
        if line == 0.0 and vx >= -1e-6:
            continue
        if line > 0.0 and vx <= 1e-6:
            continue
        t = (line - px) / vx
        if not (0.0 <= t <= GOAL_LOOKAHEAD_S):
            continue
        y = py + vy * t
        offset = abs(y - width / 2.0)
        if offset > GOAL_HALF_WIDTH + GOAL_Y_MARGIN:
            continue
        if abs(vx) < GOAL_MIN_SPEED:
            continue
        mouth = 1.0 - offset / (GOAL_HALF_WIDTH + GOAL_Y_MARGIN)
        speed = min(1.0, abs(vx) / 10.0)
        confidence = max(0.0, mouth) * speed * (1.0 - 0.15 * t / GOAL_LOOKAHEAD_S)
        if best is None or confidence > best[2]:
            best = (line, float(y), float(confidence))
    return best


def _behind_line_wide(
    x: float, y: float, length: float, width: float, margin: float = 1.5
) -> bool:
    """A ball behind a goal line but outside the goal mouth: a shot wide, not a goal."""
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    if x < 0.0 or x > length:
        return abs(y - width / 2.0) > GOAL_HALF_WIDTH + margin
    return False


def _kickoffs(tags: TagTable) -> list[Any]:
    return [
        event
        for event in tags.events
        if event.code == "SET PIECES" and "Kick Off (After Goal)" in event.type
    ]


def _extend_through_lost_view(
    events: list[StopEvent],
    player_count: np.ndarray,
    n: int,
) -> list[StopEvent]:
    """Keep a confirmed stoppage open while the broadcast has lost the live wide view.

    After some stoppages the feed cuts to close-ups and replays: fewer than
    ``STOP_MIN_TRACKED_PLAYERS`` players are tracked for a sustained stretch.  The game is
    still dead and no possession can be attested until the wide view returns, so a stoppage
    whose end is followed almost immediately by such a stretch extends to the recovery, the
    next graded-half bound, or the next stoppage, whichever comes first.
    """
    min_len = int(round(STOP_VISIBILITY_GAP_S * FPS))
    if min_len <= 0 or player_count.size == 0:
        return events
    lost = np.asarray(player_count, dtype=np.float64) < STOP_MIN_TRACKED_PLAYERS
    half_bounds = sorted(
        {int(bound) for half in GRADED_HALVES for bound in half}
    )
    extended: list[StopEvent] = []
    for index, event in enumerate(events):
        extended.append(event)
        for run_start, run_stop in _runs(lost, min_len=min_len):
            if not event.end - 2 * FPS <= run_start <= event.end + min_len:
                continue
            if event.restart is not None and event.restart < run_start:
                break  # play demonstrably restarted before the view was lost
            ends = [run_stop]
            ends.extend(bound for bound in half_bounds if bound > event.end)
            nxt = [other.start for other in events[index + 1 :] if other.start > event.end]
            ends.append(min(nxt) if nxt else n)
            new_end = max(event.end, min(ends))
            if new_end > event.end:
                extended[-1] = replace(event, end=int(new_end))
            break
    return extended


def _restart_after(kickoffs: list[Any], frame: int) -> int | None:
    choices = [event.start_frame for event in kickoffs if event.start_frame >= frame]
    return int(min(choices)) if choices else None


def _first_low_run(
    team_speed: np.ndarray, lo: int, hi: int, min_s: float = 1.5
) -> tuple[int, int] | None:
    lo = max(0, int(lo))
    hi = min(team_speed.size, int(hi))
    if hi <= lo:
        return None
    runs = _runs(team_speed[lo:hi] < STOP_TEAM_SPEED, min_len=max(1, int(round(min_s * FPS))))
    if not runs:
        return None
    start, stop = runs[0]
    return lo + start, lo + stop


def detect_goals(
    tracking: TrackingData,
    tags: TagTable,
    track: BallTrack | None = None,
    *,
    features: Any | None = None,
) -> tuple[GoalEvent, ...]:
    """Goals accepted from shot geometry or a tagged long disappearance plus a celebration."""
    track = track or track_ball(tracking)
    n = tracking.n_frames
    length, width = tracking.pitch
    runs = _missing_runs(track)
    kickoffs = _kickoffs(tags)
    team_speed = None
    if features is not None and getattr(features, "start", 0) == 0:
        team_speed = np.asarray(features.team_speed, dtype=np.float64)

    goals: dict[int, GoalEvent] = {}
    for tag in tags.events:
        if tag.code.upper() != "GOALS":
            continue
        restart = _restart_after(kickoffs, tag.start_frame)
        cutoff = restart if restart is not None else tag.stop_frame + 5 * FPS
        if cutoff <= tag.start_frame:
            continue
        tag_runs = [
            (start, stop)
            for start, stop in runs
            if tag.start_frame - 10 <= start < cutoff
        ]
        candidates: list[GoalEvent] = []
        for start, stop in tag_runs:
            last = start - 1
            if last < 0 or not track.observed[last]:
                continue
            px, py = (float(track.estimate[last, 0]), float(track.estimate[last, 1]))
            vx, vy = _short_velocity(track, last)
            crossing = _goal_crossing(px, py, vx, vy, length, width)
            if crossing is not None:
                line, y, confidence = crossing
                travel = abs(line - px) / max(abs(vx), 1e-6)
                need = int(round(travel * FPS))
                if stop - start >= need + 3:
                    frame = last + max(1, need - 1)
                    if restart is None or restart > frame:
                        candidates.append(
                            GoalEvent(
                                frame=int(frame),
                                restart=int(restart if restart is not None else stop + 50),
                                last_frame=int(last),
                                last_position=(px, py),
                                velocity=(vx, vy),
                                goal_line=float(line),
                                crossing_y=float(y),
                                confidence=float(confidence),
                                kind="trajectory",
                                tag_start=int(tag.start_frame),
                                tag_stop=int(tag.stop_frame),
                            )
                        )

        if not candidates:
            # Fallback: a tagged, very long ball loss whose last position is in the goal mouth,
            # with no observation of the ball behind the line and outside the posts.  The latter
            # keeps precision: it rejects the false tag at 57,283 (shot wide), and it also misses
            # the real 73,935 penalty because the ATD loses that ball for 55 s and reacquires it
            # behind the line and wide.  A real goal without a tracked trajectory is not enough
            # evidence to loosen this rule without also accepting the 57,283 false positive.
            wide = False
            for k in range(max(0, tag.start_frame), min(n, cutoff)):
                if track.observed[k] and _behind_line_wide(
                    float(track.estimate[k, 0]), float(track.estimate[k, 1]), length, width
                ):
                    wide = True
                    break
            if not wide:
                for start, stop in tag_runs:
                    last = start - 1
                    if last < 0 or not track.observed[last]:
                        continue
                    px, py = (float(track.estimate[last, 0]), float(track.estimate[last, 1]))
                    deficit = min(abs(px), abs(length - px))
                    if deficit > GOAL_MAX_DEFICIT:
                        continue
                    if abs(py - width / 2.0) > GOAL_HALF_WIDTH + GOAL_Y_MARGIN:
                        continue
                    if stop - start < GOAL_FALLBACK_GAP_S * FPS:
                        continue
                    frame = start
                    if team_speed is not None:
                        low = _first_low_run(team_speed, start, stop, min_s=1.5)
                        if low is not None:
                            frame = int(low[0])
                    if restart is None or restart > frame:
                        candidates.append(
                            GoalEvent(
                                frame=int(frame),
                                restart=int(restart if restart is not None else stop + 50),
                                last_frame=int(last),
                                last_position=(px, py),
                                velocity=(0.0, 0.0),
                                goal_line=0.0 if abs(px) <= abs(length - px) else float(length),
                                crossing_y=float(py),
                                confidence=0.55,
                                kind="celebration",
                                tag_start=int(tag.start_frame),
                                tag_stop=int(tag.stop_frame),
                            )
                        )

        if candidates:
            best = max(candidates, key=lambda g: (g.confidence, -g.frame))
            # One goal per tag; deduplicate tags that overlap the same event.
            if not any(abs(best.frame - old.frame) < 100 for old in goals.values()):
                goals[best.frame] = best

    return tuple(goals[frame] for frame in sorted(goals))


def _live_tag_end(tags: TagTable, frame: int) -> int:
    end = int(frame)
    for event in tags.events:
        if event.code in STOP_LIVE_TAG_CODES and event.start_frame <= frame < event.stop_frame:
            end = max(end, int(event.stop_frame))
    return end


def _restart_end(
    start: int,
    limit: int,
    present: np.ndarray,
    out_of_play: np.ndarray,
    ball_speed: np.ndarray,
    near_distance: np.ndarray,
    edge: np.ndarray,
    floor: int | None = None,
) -> int | None:
    """First sustained kicked restart after ``start`` (two frames of fast ball near a player)."""
    n = present.size
    lo = max(start + FPS, int(floor) if floor is not None else 0, 0)
    hi = min(n, int(limit))
    hits = 0
    for k in range(lo, hi):
        fast = (
            present[k]
            and not out_of_play[k]
            and np.isfinite(ball_speed[k])
            and ball_speed[k] > STOP_RESTART_SPEED
            and np.isfinite(near_distance[k])
            and near_distance[k] < STOP_RESTART_CONTROL_R
            and np.isfinite(edge[k])
            and edge[k] < 6.0
        )
        if fast:
            hits += 1
            if hits >= STOP_RESTART_FRAMES:
                return k
        else:
            hits = 0
    return None


def _restart_kick(
    start: int,
    end: int,
    present: np.ndarray,
    ball_speed: np.ndarray,
    near_distance: np.ndarray,
) -> int | None:
    """First slow-to-fast kick after ``start``, for the viewer's restart marker.

    Display metadata only: this never trims the stoppage.  Unlike :func:`_restart_end` it does not
    require the ball near a touchline, because a goal kick is taken well inside the box (measured
    131 427: ball at x=11.0 m, 1.1 m from the goalkeeper, 22.3 m/s) and the edge gate made the
    restart invisible exactly where the user looks for it.
    """
    n = present.size
    lo = max(start + FPS // 2, 0)
    hi = min(n, int(end))
    hits = 0
    for k in range(lo, hi):
        fast = (
            present[k]
            and np.isfinite(ball_speed[k])
            and ball_speed[k] > STOP_RESTART_SPEED
            and np.isfinite(near_distance[k])
            and near_distance[k] < STOP_RESTART_CONTROL_R
        )
        if not fast:
            hits = 0
            continue
        before = np.asarray(ball_speed[max(lo, k - 3) : k], dtype=np.float64)
        still_before = bool(np.isfinite(before).any() and np.nanmin(before) < STOP_RESTART_SPEED)
        if not still_before:
            hits = 0
            continue
        hits += 1
        if hits >= STOP_RESTART_FRAMES:
            return k - hits + 1
    return None


def _restart_is_failed(
    restart: int,
    out_of_play: np.ndarray,
    horizon: int,
    skip_max: int,
) -> bool:
    """True if the ball is out of play again within ``horizon`` frames after the restart.

    Display metadata for the viewer: a throw whose ball is back beyond the line almost immediately
    did not resume the match, so the marker is drawn as a failed throw, not a resume.  Up to
    ``skip_max`` frames of out-of-play right at the restart are the throw itself (the ball leaves
    the thrower's hand beyond the line); a new crossing inside the horizon is the failure.  No
    stoppage interval, dead mask, or possession label depends on the result.
    """
    out_of_play = np.asarray(out_of_play, dtype=bool)
    n = out_of_play.size
    r = int(restart)
    hi = min(n, r + int(horizon))
    k = r
    skipped = 0
    while k < hi and out_of_play[k] and skipped < int(skip_max):
        k += 1
        skipped += 1
    return bool(out_of_play[k:hi].any()) if k < hi else False


def _control_resume(
    start: int,
    limit: int,
    present: np.ndarray,
    near_distance: np.ndarray,
    gap: np.ndarray,
    min_frames: int = 3,
) -> int | None:
    """First sustained visible control after a missing-ball interval."""
    if present[start]:
        return None
    hits = 0
    for k in range(start + 1, min(limit, present.size)):
        good = (
            present[k]
            and gap[k] <= 3
            and np.isfinite(near_distance[k])
            and near_distance[k] <= STOP_CONTROL_R
        )
        if good:
            hits += 1
            if hits >= min_frames:
                return k + 10
        else:
            hits = 0
    return None


def detect_stoppages(
    tracking: TrackingData,
    tags: TagTable,
    track: BallTrack | None = None,
    whistle: Any | None = None,
    features: Any | None = None,
) -> tuple[StopEvent, ...]:
    """Confirmed dead-ball intervals.

    Triggers, in order of strength: a recent whistle with no controlling player, a ball beyond the
    line, or a missing ball whose last observation was already out.  Every trigger still needs a
    sustained low-motion/low-play context, either the low run itself or a long out-of-play run.
    """
    track = track or track_ball(tracking)
    if features is None:
        from .features import extract_features

        features = extract_features(tracking, tags, ball_track=track)
    if getattr(features, "start", 0) != 0 or getattr(features, "stop", tracking.n_frames) != tracking.n_frames:
        raise ValueError("detect_stoppages needs features covering the full match")

    n = tracking.n_frames
    present = np.asarray(features.present, dtype=bool)
    gap = np.asarray(features.gap, dtype=np.int64)
    near_distance = np.asarray(features.near_distance, dtype=np.float64)
    out_of_play = np.asarray(features.out_of_play, dtype=bool)
    last_edge = np.asarray(features.last_edge, dtype=np.float64)
    edge = np.asarray(features.edge, dtype=np.float64)
    ball_speed = np.asarray(features.ball_speed, dtype=np.float64)
    team_speed = np.asarray(features.team_speed, dtype=np.float64)
    low = team_speed < STOP_TEAM_SPEED

    candidates: list[StopEvent] = []

    def finish(start: int, trigger: str, confidence: float, **kwargs: Any) -> StopEvent | None:
        start = max(0, min(int(start), n - 1))
        # The low-motion run is the natural end; search a little further for a restart kick.
        search_s = float(kwargs.pop("search_s", 3.0))
        restart_floor = kwargs.pop("restart_floor", None)
        low_stop = kwargs.get("low_stop")
        if low_stop is None:
            end_of_run = _run_end_containing(low, start, FPS, n)
            low_stop = end_of_run if end_of_run is not None else min(n, start + 5 * FPS)
        end = min(n, int(low_stop) + int(round(search_s * FPS)))
        control = _control_resume(start, end + 50, present, near_distance, gap)
        if control is not None:
            end = min(end, int(control))
        restart = _restart_end(
            start,
            end + 50,
            present,
            out_of_play,
            ball_speed,
            near_distance,
            edge,
            floor=restart_floor,
        )
        if restart is not None:
            end = min(end, int(restart))
        if end <= start:
            return None
        kick = _restart_kick(start, end, present, ball_speed, near_distance)
        if kick is not None:
            # The marker must sit strictly inside the stoppage.  When the stoppage's end is pinned
            # to the restart itself (sustained kicked restart or control resume), the kick lands a
            # few frames before ``end`` by construction, so the strict boundary is the right one;
            # when the end is the plain search limit, keep a 5-frame margin so the marker does not
            # hug a stoppage that runs straight into live play.
            limit = end if (control is not None or restart is not None) else end - 5
            if not (start < kick < limit):
                kick = None
        return StopEvent(
            start=int(start),
            end=int(end),
            trigger=trigger,
            confidence=float(confidence),
            restart=kick,
            **kwargs,
        )

    # 1) Whistle candidates, confirmed by a sustained low-motion run that follows them.
    whistle_events = tuple(getattr(whistle, "events", ()) or ()) if whistle is not None else ()
    lookback = int(round(STOP_WHISTLE_LOOKBACK_S * FPS))
    for low_start, low_stop in _runs(low, min_len=int(round(STOP_MIN_LOW_S * FPS))):
        # A blast is a span, and players rest *after* it: anchor the lookback on the frame the
        # whistle stopped.  Anchoring on the peak alone drops the foul the referee blows just
        # before the field comes to a walk (measured 86 577: peak 6.9, blast 86 576-86 584, the
        # low-motion run opens at 86 645 -- six frames outside a peak-anchored lookback, so a
        # minute-long stoppage with three more whistles in it read as live play).
        near_whistles = [
            event
            for event in whistle_events
            if int(event.get("stop", event.get("frame", -10**9))) >= low_start - lookback
            and int(event.get("frame", -10**9)) <= low_start + 2
        ]
        if not near_whistles:
            continue
        candidate = max(near_whistles, key=lambda e: float(e.get("peak", 0.0)))
        wf = int(candidate.get("frame", low_start))
        controlled = (
            present[wf]
            and np.isfinite(near_distance[wf])
            and near_distance[wf] <= STOP_CONTROL_R
            and not out_of_play[wf]
            and gap[wf] <= 5
        )
        if controlled:
            continue
        start = max(wf, _live_tag_end(tags, wf))
        if start > wf and present[start]:
            # A live tag can end while the passage is still going.  Give the visible ball one
            # more second before declaring the stoppage; a missing ball has no such ambiguity.
            start = min(low_stop - 1, start + FPS)
        if start >= low_stop - 5:
            continue
        stop = finish(
            start,
            "whistle",
            0.85,
            whistle_frame=wf,
            whistle_peak=float(candidate.get("peak", 0.0)),
            low_start=int(low_start),
            low_stop=int(low_stop),
        )
        if stop is not None:
            candidates.append(stop)

    # 1b) A whistle that lands *inside* a long resting stretch.  The lookback above only sees a
    # whistle that precedes the players coming to rest, but a referee blows while the field is
    # already standing still, and the squad stops for the foul, not for the ball.  Measured
    # 86 645-88 199 (57:05-59:48): the whole pitch walks for a minute, the ball is unaccounted
    # for, and the whistle bed fires at 86 960, 87 018, 87 318, 87 432 and 87 736 -- a stopped
    # game the decoder read as live Athletic possession.
    in_run_min = int(round(STOP_IN_RUN_WHISTLE_S * FPS))
    in_run_check = max(1, int(STOP_IN_RUN_ACTIVE_FRAMES))
    # The field walking is only half of it: a team also walks with a live ball at its feet, and a
    # blast in the background of a passage of play does not stop it.  The opening second of the run
    # has to show the ball *out of play* -- untracked, or tracked but neither moving nor at anyone's
    # feet.  Measured 17 323-17 858: four whistles inside a resting minute, and the ball is in play
    # at 4-5 m and 3-5 m/s the whole opening second (the game goes on for 53 more frames, and forcing
    # dead there splits a live Barcelona spell the ground truth counts as one segment).
    ball_in_play = (
        present
        & (gap <= 5)
        & ((ball_speed >= STOP_IN_RUN_BALL_SPEED) | (near_distance <= STOP_CONTROL_R))
    )
    for low_start, low_stop in _runs(low, min_len=in_run_min):
        if ball_in_play[low_start:min(n, low_start + in_run_check)].any():
            continue
        inside = [
            event
            for event in whistle_events
            if low_start <= int(event.get("frame", -10**9)) <= low_stop
        ]
        if not inside:
            continue
        if any(c.start - 5 <= low_start < c.end for c in candidates):
            continue  # trigger 1) already owns this run
        # strongest blast first, and the first one no player is answering with the ball at their
        # feet: a referee's whistle next to a controlled visible ball is play going on
        ordered = sorted(inside, key=lambda e: -float(e.get("peak", 0.0)))
        chosen = None
        for event in ordered:
            wf = int(event.get("frame", low_start))
            controlled = (
                present[wf]
                and np.isfinite(near_distance[wf])
                and near_distance[wf] <= STOP_CONTROL_R
                and not out_of_play[wf]
                and gap[wf] <= 5
            )
            if not controlled:
                chosen = (wf, float(event.get("peak", 0.0)))
                break
        if chosen is None:
            continue
        wf, peak = chosen
        stop = finish(
            low_start,
            "whistle_in_run",
            0.8,
            whistle_frame=wf,
            whistle_peak=peak,
            low_start=int(low_start),
            low_stop=int(low_stop),
        )
        if stop is not None:
            candidates.append(stop)

    # 2) Visible ball beyond the touch/goal line: dead from the crossing, end at the restart.
    for start, stop in _runs(out_of_play, min_len=max(1, int(0.5 * FPS))):
        # Ignore a ball that is merely rolling out and is already part of a whistle interval.
        if any(c.start - 5 <= start < c.end for c in candidates):
            continue
        evidence = finish(
            start,
            "out",
            0.9,
            low_start=start,
            low_stop=stop,
            search_s=15.0,
            restart_floor=stop + FPS,
        )
        if evidence is not None:
            candidates.append(evidence)

    # 3) Missing ball whose last observation was already beyond the line.  This is the goal-kick /
    # throw-in retrieval gap; it is not used for a ball lost near the line while play continues.
    for start, stop in _runs(~present, min_len=max(1, int(1.0 * FPS))):
        edge_value = float(last_edge[start]) if np.isfinite(last_edge[start]) else 20.0
        if edge_value >= -0.2:
            continue
        if any(c.start - 5 <= start < c.end for c in candidates):
            continue
        evidence = finish(start, "out_gap", 0.8, low_start=start, low_stop=stop, search_s=6.0)
        if evidence is not None:
            candidates.append(evidence)

    # Merge overlapping/adjacent intervals.  Prefer the stronger trigger and remember the earliest
    # start; this keeps a throw-in from being split into a dozen one-second pieces.
    candidates.sort(key=lambda event: (event.start, event.end))
    merged = _merge_stop_events(candidates)

    # A kick-off tag is an explicit restart.  Without this trim, a long out-of-play/retrieval gap
    # that began before the kick-off can leak a hard D override into the first live frames (the
    # opening of the public window at 11,185 is exactly this case).
    kickoffs = [
        int(event.start_frame)
        for event in tags.events
        if event.code == "SET PIECES" and "Kick Off" in event.type
    ]
    trimmed: list[StopEvent] = []
    for event in merged:
        # The earliest kick-off inside the interval, not whichever one the tag file happens to list
        # first: trimming to a later one would keep the hard D override over live frames.
        restart = min((k for k in kickoffs if event.start <= k < event.end), default=None)
        if restart is not None and restart > event.start:
            event = replace(event, end=int(restart))
        kick = event.restart if event.restart is not None and event.restart < event.end else None
        if kick != event.restart:
            event = replace(event, restart=kick)
        if event.end > event.start:
            trimmed.append(event)
    trimmed = _extend_through_lost_view(
        trimmed, np.asarray(tracking.player_present, dtype=bool).sum(axis=1), n
    )
    # Display-only annotation for the viewer's "saque fallido" marker: a restart whose ball is
    # back out of play within a few seconds did not resume the match.  Measured 179 108: the
    # first throw-in of the 178 852-179 483 stoppage, ball back beyond the line 68 frames later;
    # the effective restart is the second throw at 179 480.  Intervals, masks, and labels stay
    # untouched.
    horizon = int(round(STOP_FAILED_RESTART_S * FPS))
    skip_max = int(round(STOP_FAILED_RESTART_SKIP_S * FPS))
    return tuple(
        replace(event, restart_failed=True)
        if event.restart is not None
        and _restart_is_failed(event.restart, out_of_play, horizon, skip_max)
        else event
        for event in trimmed
    )


def build_events(
    tracking: TrackingData,
    tags: TagTable,
    track: BallTrack | None = None,
    whistle: Any | None = None,
    features: Any | None = None,
) -> EventData:
    """Detect goals and confirmed stoppages and build the two override masks."""
    track = track or track_ball(tracking)
    n = tracking.n_frames
    if features is None:
        from .features import extract_features

        features = extract_features(tracking, tags, ball_track=track, whistle=whistle)
    goals = detect_goals(tracking, tags, track, features=features)
    stoppages = detect_stoppages(tracking, tags, track, whistle, features=features)

    goal_dead = np.zeros(n, dtype=bool)
    for goal in goals:
        goal_dead[max(0, goal.frame) : min(n, goal.restart)] = True
    stop_dead = np.zeros(n, dtype=bool)
    for stop in stoppages:
        stop_dead[max(0, stop.start) : min(n, stop.end)] = True
    dead = goal_dead | stop_dead
    return EventData(
        goals=goals,
        stoppages=stoppages,
        goal_dead=goal_dead,
        stop_dead=stop_dead,
        dead_mask=dead,
        meta={
            "n_goals": len(goals),
            "n_stoppages": len(stoppages),
            "goal_frames": [int(g.frame) for g in goals],
            "stop_intervals": [[int(s.start), int(s.end)] for s in stoppages],
            "whistle_events": len(getattr(whistle, "events", ()) or ()) if whistle is not None else 0,
        },
    )


def apply_event_overrides(codes: np.ndarray, events: EventData, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Force ``D`` over high-confidence goal/stoppage intervals in a decoded window.

    ``codes`` may be either the full match (indexed by global frame) or a local window beginning at
    ``start``.  The caller usually has the latter: the public decode is a context-padded slice.
    """
    from .config import D

    out = np.asarray(codes, dtype=np.int8).copy()
    if events.dead_mask.size == 0:
        return out
    stop = events.dead_mask.size if stop is None else min(int(stop), events.dead_mask.size)
    start = max(0, int(start))
    if stop <= start:
        return out
    local = events.dead_mask[start:stop]
    if out.size == stop - start and local.size == out.size:
        out[local] = D
    elif out.size >= stop and local.size == stop - start:
        out[start:stop][local] = D
    return out
