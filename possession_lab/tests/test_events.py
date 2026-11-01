"""Tests for goal geometry and confirmed-stoppage events."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from possession_lab.balltrack import track_ball
from possession_lab.config import D, H
from possession_lab.data import TagEvent, TagTable, TrackingData
from possession_lab.events import (
    EventData,
    StopEvent,
    _extend_through_lost_view,
    _merge_stop_events,
    _restart_kick,
    _run_end_containing,
    apply_event_overrides,
    detect_goals,
    detect_stoppages,
)
from possession_lab.features import extract_features


def make_tags(n: int, events: tuple[TagEvent, ...] = ()) -> TagTable:
    base = TagTable.empty(n)
    return TagTable(
        events=events,
        code_names=base.code_names,
        type_names=base.type_names,
        code_matrix=base.code_matrix,
        type_matrix=base.type_matrix,
        team_vote=base.team_vote,
        setpiece=base.setpiece,
        coverage=base.coverage,
        n_frames=n,
    )


def make_tracking(
    n: int,
    ball_xy: tuple[np.ndarray, np.ndarray],
    present: np.ndarray | None = None,
    speed: float = 0.2,
) -> TrackingData:
    x, y = ball_xy
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = x
    ball[:, 1] = y
    if present is None:
        present = np.ones(n, dtype=bool)
    present = np.asarray(present, dtype=bool)
    ball[~present] = np.nan
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = [10.0, 10.0]
    players[:, 1] = [90.0, 50.0]
    speeds = np.full((n, 2), speed, dtype=np.float32)
    return TrackingData(
        ball=ball,
        ball_present=present,
        players=players,
        speeds=speeds,
        player_present=np.ones((n, 2), dtype=bool),
        teams=np.array([0, 1], dtype=np.int8),
        pitch=(105.3, 68.0),
        n_frames=n,
    )


def goal_tags(n: int, start: int = 30, stop: int = 160, kickoff: int = 170) -> TagTable:
    return make_tags(
        n,
        (
            TagEvent(start, stop, "GOALS", 0, "Goal"),
            TagEvent(kickoff, kickoff + 20, "SET PIECES", 1, "Kick Off (After Goal)"),
        ),
    )


class GoalTest(unittest.TestCase):
    def test_trajectory_reaches_the_goal_mouth(self):
        n = 200
        x = np.full(n, 104.8, dtype=np.float32)
        y = np.full(n, 34.0, dtype=np.float32)
        present = np.zeros(n, dtype=bool)
        present[:50] = True
        # The ball is moving toward the line at 5 m/s; the observed positions are stationary in
        # the last frame only, so give the shot a real backward velocity history.
        x[:50] = np.linspace(95.0, 104.8, 50, dtype=np.float32)
        tracking = make_tracking(n, (x, y), present=present)
        tags = goal_tags(n)
        goals = detect_goals(tracking, tags, track_ball(tracking))
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].kind, "trajectory")
        self.assertGreaterEqual(goals[0].frame, 50)
        self.assertLess(goals[0].frame, 60)
        self.assertEqual(goals[0].restart, 170)

    def test_shot_wide_of_the_posts_is_not_a_goal(self):
        n = 200
        x = np.linspace(95.0, 104.8, 50, dtype=np.float32)
        y = np.full(50, 10.0, dtype=np.float32)
        present = np.zeros(n, dtype=bool)
        present[:50] = True
        full_x = np.full(n, 104.8, dtype=np.float32)
        full_y = np.full(n, 10.0, dtype=np.float32)
        full_x[:50] = x
        full_y[:50] = y
        tracking = make_tracking(n, (full_x, full_y), present=present)
        goals = detect_goals(tracking, goal_tags(n), track_ball(tracking))
        self.assertEqual(goals, ())

    def test_ball_observed_behind_the_line_wide_rejects_the_fallback(self):
        n = 220
        x = np.full(n, 100.0, dtype=np.float32)
        y = np.full(n, 34.0, dtype=np.float32)
        present = np.ones(n, dtype=bool)
        # A tagged long disappearance near the goal, but the ball was already seen behind the
        # goal line and outside the posts: a shot wide, not a goal.
        present[50:150] = False
        x[150] = 108.0
        y[150] = 12.0
        tracking = make_tracking(n, (x, y), present=present)
        goals = detect_goals(
            tracking, goal_tags(n, stop=180, kickoff=200), track_ball(tracking)
        )
        self.assertEqual(goals, ())


class StoppageTest(unittest.TestCase):
    def test_recent_whistle_plus_low_motion_confirms_a_stoppage(self):
        n = 400
        x = np.full(n, 50.0, dtype=np.float32)
        y = np.full(n, 34.0, dtype=np.float32)
        tracking = make_tracking(n, (x, y))
        # Live play first, then a sustained low-motion run after the whistle.
        tracking.speeds[:100] = 3.0
        features = extract_features(tracking, TagTable.empty(n))
        whistle = SimpleNamespace(events=({"frame": 90, "peak": 8.0},))
        stops = detect_stoppages(
            tracking, TagTable.empty(n), features=features, whistle=whistle
        )
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0].trigger, "whistle")
        self.assertLessEqual(stops[0].start, 100)
        self.assertGreater(stops[0].end, 200)

    def test_whistle_during_visible_control_is_not_a_stoppage(self):
        n = 300
        tracking = make_tracking(n, (np.full(n, 10.0), np.full(n, 10.0)))
        features = extract_features(tracking, TagTable.empty(n))
        whistle = SimpleNamespace(events=({"frame": 90, "peak": 8.0},))
        stops = detect_stoppages(
            tracking, TagTable.empty(n), features=features, whistle=whistle
        )
        self.assertEqual(stops, ())

    def test_visible_ball_beyond_the_line_is_dead(self):
        n = 300
        x = np.full(n, 50.0, dtype=np.float32)
        x[100:160] = 108.0
        y = np.full(n, 34.0, dtype=np.float32)
        tracking = make_tracking(n, (x, y))
        features = extract_features(tracking, TagTable.empty(n))
        stops = detect_stoppages(tracking, TagTable.empty(n), features=features)
        self.assertTrue(any(s.trigger == "out" for s in stops))
        self.assertTrue(any(s.start <= 100 < s.end for s in stops))

    def test_a_whistle_inside_a_resting_stretch_is_a_stoppage(self):
        # the referee blows while the field is already standing still: the lookback above cannot
        # see it, and a minute of walking with the ball unaccounted for read as live play
        n = 600
        x = np.full(n, 50.0, dtype=np.float32)
        y = np.full(n, 34.0, dtype=np.float32)
        tracking = make_tracking(n, (x, y), present=np.ones(n, dtype=bool))
        tracking.speeds[:150] = 3.0  # live play until 150, then everybody walks
        tracking.ball_present[200:] = False  # and the ball stops being seen
        tracking.ball[200:] = np.nan
        features = extract_features(tracking, TagTable.empty(n))
        whistle = SimpleNamespace(events=({"frame": 300, "stop": 310, "peak": 9.0},))
        stops = detect_stoppages(
            tracking, TagTable.empty(n), features=features, whistle=whistle
        )
        self.assertEqual([s.trigger for s in stops], ["whistle_in_run"])
        self.assertLessEqual(stops[0].start, 210)
        self.assertGreater(stops[0].end, 300)

    def test_a_resting_field_with_a_live_ball_is_not_a_stoppage(self):
        # a team also walks with the ball at its feet, and a blast in the background of a passage
        # of play does not stop it
        n = 600
        x = np.full(n, 50.0, dtype=np.float32)
        y = np.full(n, 34.0, dtype=np.float32)
        tracking = make_tracking(n, (x, y))
        tracking.speeds[:150] = 3.0
        features = extract_features(tracking, TagTable.empty(n))
        features.ball_speed[150:400] = 5.0  # the ball is being played through the walking stretch
        whistle = SimpleNamespace(events=({"frame": 300, "stop": 310, "peak": 9.0},))
        stops = detect_stoppages(
            tracking, TagTable.empty(n), features=features, whistle=whistle
        )
        self.assertEqual([s.trigger for s in stops], [])

    def test_a_blast_just_before_the_rest_confirms_the_stoppage(self):
        # players rest *after* the whistle: anchoring the lookback on the peak alone drops a foul
        # whose blast ends a few frames before the field comes to a walk
        n = 600
        x = np.full(n, 50.0, dtype=np.float32)
        y = np.full(n, 34.0, dtype=np.float32)
        tracking = make_tracking(n, (x, y))
        tracking.speeds[:200] = 3.0
        tracking.ball_present[210:] = False
        tracking.ball[210:] = np.nan
        features = extract_features(tracking, TagTable.empty(n))
        whistle = SimpleNamespace(events=({"frame": 194, "stop": 200, "peak": 7.0},))
        stops = detect_stoppages(
            tracking, TagTable.empty(n), features=features, whistle=whistle
        )
        self.assertIn("whistle", [s.trigger for s in stops])

    def test_event_masks_and_overrides_work_for_full_and_local_arrays(self):
        n = 80
        dead = np.zeros(n, dtype=bool)
        dead[20:30] = True
        events = EventData(
            goals=(),
            stoppages=(),
            goal_dead=np.zeros(n, dtype=bool),
            stop_dead=dead.copy(),
            dead_mask=dead,
            meta={},
        )
        full = np.full(n, H, dtype=np.int8)
        np.testing.assert_array_equal(apply_event_overrides(full, events)[20:30], D)
        local = np.full(20, H, dtype=np.int8)
        local_out = apply_event_overrides(local, events, start=10, stop=30)
        np.testing.assert_array_equal(local_out[10:], D)
        np.testing.assert_array_equal(local_out[:10], H)


class RunEndContainingTest(unittest.TestCase):
    """The low-motion run that contains the trigger, not merely one that starts earlier."""

    def test_returns_the_end_of_the_containing_run(self):
        mask = np.zeros(100, dtype=bool)
        mask[40:60] = True
        self.assertEqual(_run_end_containing(mask, 45, lookback=10, n=100), 60)
        self.assertEqual(_run_end_containing(mask, 40, lookback=10, n=100), 60)
        self.assertEqual(_run_end_containing(mask, 59, lookback=10, n=100), 60)

    def test_a_run_that_only_starts_earlier_is_not_the_stoppage(self):
        # The old fallback took the first run beginning before `start`, so this returned the end of
        # the unrelated 10..20 run and misplaced the dead-ball interval by 35 frames.
        mask = np.zeros(100, dtype=bool)
        mask[10:20] = True
        mask[40:60] = True
        self.assertEqual(_run_end_containing(mask, 45, lookback=60, n=100), 60)

    def test_no_containing_run_returns_none(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:20] = True
        self.assertIsNone(_run_end_containing(mask, 45, lookback=10, n=100))

    def test_result_is_clamped_to_n(self):
        mask = np.zeros(100, dtype=bool)
        mask[40:100] = True
        self.assertEqual(_run_end_containing(mask, 45, lookback=10, n=55), 55)


class MergeStopEventsTest(unittest.TestCase):
    """Merging must not treat frame 0 or a 0.0 peak as a missing value."""

    def test_overlapping_intervals_are_joined(self):
        first = StopEvent(start=100, end=200, trigger="out", confidence=0.6)
        second = StopEvent(start=210, end=300, trigger="low_motion", confidence=0.8)
        merged = _merge_stop_events([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (100, 300))
        self.assertEqual(merged[0].trigger, "low_motion")
        self.assertEqual(merged[0].confidence, 0.8)

    def test_distant_intervals_stay_separate(self):
        first = StopEvent(start=100, end=200, trigger="out", confidence=0.6)
        second = StopEvent(start=1000, end=1100, trigger="low_motion", confidence=0.8)
        self.assertEqual(len(_merge_stop_events([first, second])), 2)

    def test_frame_zero_is_a_real_whistle_frame(self):
        first = StopEvent(
            start=0, end=100, trigger="out", confidence=0.6, whistle_frame=0, whistle_peak=0.0
        )
        second = StopEvent(
            start=50, end=200, trigger="low_motion", confidence=0.4, whistle_frame=7, whistle_peak=1.5
        )
        merged = _merge_stop_events([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].whistle_frame, 0)
        self.assertEqual(merged[0].whistle_peak, 0.0)

    def test_unset_fields_take_the_first_value_that_is_set(self):
        first = StopEvent(start=0, end=100, trigger="out", confidence=0.6, low_start=None)
        second = StopEvent(start=50, end=200, trigger="low_motion", confidence=0.4, low_start=40)
        merged = _merge_stop_events([first, second])
        self.assertEqual(merged[0].low_start, 40)


class RestartKickTest(unittest.TestCase):
    """The viewer's restart marker needs the slow-to-fast kick, even away from a touchline."""

    def test_goal_kick_after_a_still_ball_is_found(self):
        n = 40
        present = np.ones(n, dtype=bool)
        ball_speed = np.zeros(n, dtype=np.float64)
        ball_speed[20:22] = 20.0
        near_distance = np.ones(n, dtype=np.float64)
        self.assertEqual(_restart_kick(5, 30, present, ball_speed, near_distance), 20)

    def test_a_run_that_is_already_fast_has_no_restart_marker(self):
        n = 40
        present = np.ones(n, dtype=bool)
        ball_speed = np.full(n, 20.0, dtype=np.float64)
        near_distance = np.ones(n, dtype=np.float64)
        self.assertIsNone(_restart_kick(5, 30, present, ball_speed, near_distance))

    def test_to_dict_carries_the_restart_frame(self):
        event = StopEvent(start=10, end=50, trigger="whistle", confidence=0.8, restart=32)
        self.assertEqual(event.to_dict()["restart"], 32)


class LostViewStoppageTests(unittest.TestCase):
    """A confirmed stoppage stays dead while the broadcast has lost the wide view."""

    def test_stoppage_extends_through_a_lost_wide_view(self):
        n = 1000
        counts = np.full(n, 22)
        counts[240:600] = 4  # close-ups: the feed no longer shows the live field
        events = [StopEvent(start=100, end=200, trigger="whistle_in_run", confidence=0.8)]
        out = _extend_through_lost_view(events, counts, n)
        self.assertEqual(out[0].end, 600)

    def test_no_extension_while_the_wide_view_holds(self):
        n = 1000
        counts = np.full(n, 22)
        events = [StopEvent(start=100, end=200, trigger="whistle_in_run", confidence=0.8)]
        out = _extend_through_lost_view(events, counts, n)
        self.assertEqual(out[0].end, 200)

    def test_extension_is_capped_at_the_next_stoppage(self):
        n = 1000
        counts = np.full(n, 22)
        counts[240:900] = 3
        events = [
            StopEvent(start=100, end=200, trigger="whistle_in_run", confidence=0.8),
            StopEvent(start=400, end=500, trigger="out", confidence=0.9),
        ]
        out = _extend_through_lost_view(events, counts, n)
        self.assertEqual(out[0].end, 400)

    def test_restart_before_the_view_is_lost_blocks_the_extension(self):
        n = 1000
        counts = np.full(n, 22)
        counts[240:600] = 4
        events = [StopEvent(start=100, end=200, trigger="whistle_in_run", confidence=0.8, restart=210)]
        out = _extend_through_lost_view(events, counts, n)
        self.assertEqual(out[0].end, 200)


if __name__ == "__main__":
    unittest.main()
