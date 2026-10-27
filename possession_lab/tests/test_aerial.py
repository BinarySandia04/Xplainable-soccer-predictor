"""Tests for the causal aerial-duel (launch) attribution layer.

The ATD ball is a 2-D ground projection: while the ball is in the air the nearest tracked
player is whoever is *under* the flight, not who touched it.  ``launch_track`` marks those
frames causally and freezes the team of the player nearest at the launch (the last toucher),
so proximity cannot hand an aerial duel to the defending team.
"""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.aerial import launch_track
from possession_lab.data import TrackingData


def make_flight_tracking(
    x: np.ndarray,
    players: np.ndarray,
    teams: np.ndarray,
    present: np.ndarray | None = None,
    height: np.ndarray | None = None,
    player_present: np.ndarray | None = None,
) -> TrackingData:
    n = x.shape[0]
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = x
    ball[:, 1] = 34.0
    if present is None:
        present = np.ones(n, dtype=bool)
    present = np.asarray(present, dtype=bool)
    ball[~present] = np.nan
    n_players = players.shape[1]
    if player_present is None:
        player_present = np.ones((n, n_players), dtype=bool)
    return TrackingData(
        ball=ball,
        ball_present=present,
        players=players.astype(np.float32),
        speeds=np.zeros((n, n_players), dtype=np.float32),
        player_present=np.asarray(player_present, dtype=bool),
        teams=np.asarray(teams, dtype=np.int8),
        pitch=(105.3, 68.0),
        n_frames=n,
        ball_height=None if height is None else np.asarray(height, dtype=np.float32),
    )


def fixture_pass() -> TrackingData:
    """Rolling ball at x=30; launch at frame 50 towards an away player under the flight."""
    n = 140
    x = np.full(n, 30.0, dtype=np.float32)
    x[50:100] = 30.0 + 0.5 * np.arange(1, 51)  # 12.5 m/s from frame 50
    x[100:] = x[99]  # ball stops: landing
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = [30.0, 34.0]  # home, at the launch point
    players[:, 1] = [45.0, 34.0]  # away, under the flight path
    return make_flight_tracking(x, players, teams=np.array([0, 1]))


class LaunchTrackTest(unittest.TestCase):
    def test_speed_launch_freezes_the_launch_team_through_opponent_proximity(self):
        tracking = fixture_pass()
        state = launch_track(tracking)

        self.assertFalse(state.in_flight[49])
        self.assertTrue(state.in_flight[50])
        self.assertEqual(int(state.team[50]), 0)  # home was nearest at the launch
        # frame 79: the ball is directly above the away player, the home player is 15 m back
        self.assertAlmostEqual(float(tracking.ball[79, 0]), 45.0, places=3)
        self.assertTrue(state.in_flight[79])
        self.assertEqual(int(state.team[79]), 0)
        self.assertEqual(int(state.age[79]), 29)
        # ball stopped at frame 100: flight is over, proximity rules again
        self.assertFalse(state.in_flight[100])

    def test_sustained_height_marks_flight_even_at_low_speed(self):
        n = 100
        x = 0.02 * np.arange(n, dtype=np.float32)  # ~0.5 m/s, never a speed launch
        height = np.full(n, np.nan, dtype=np.float32)
        height[30:71] = 3.0
        players = np.array([[[5.0, 34.0], [8.0, 34.0]]] * n, dtype=np.float32)
        state = launch_track(make_flight_tracking(x, players, np.array([0, 1]), height=height))

        self.assertFalse(state.in_flight[29])
        self.assertTrue(state.in_flight[30])
        self.assertTrue(state.in_flight[70])
        self.assertFalse(state.in_flight[71])  # height gone, speed below landing
        self.assertEqual(int(state.team[50]), 0)

    def test_flight_holds_are_capped_by_max_air_time(self):
        n = 130
        x = np.zeros(n, dtype=np.float32)
        height = np.full(n, np.nan, dtype=np.float32)
        height[30:] = 3.0  # the provider never sees the ball land
        players = np.array([[[0.0, 34.0]]] * n, dtype=np.float32)
        state = launch_track(make_flight_tracking(x, players, np.array([0]), height=height))

        self.assertTrue(state.in_flight[105])  # 75 frames of hold
        self.assertFalse(state.in_flight[106])

    def test_unknown_launch_team_when_no_player_is_tracked(self):
        tracking = fixture_pass()
        tracking.player_present[:] = False
        state = launch_track(tracking)

        self.assertTrue(state.in_flight[60])
        self.assertEqual(int(state.team[60]), -1)

    def test_a_rekick_reanchors_attribution_to_the_new_touch(self):
        n = 120
        x = np.full(n, 30.0, dtype=np.float32)
        x[50:80] = 30.0 + 0.4 * np.arange(1, 31)  # 10 m/s launch at frame 50
        x[80:] = x[79] + 0.64 * np.arange(1, n - 79)  # rekick at frame 80: 16 m/s
        x[110:] = x[109]  # ball stops: landing
        players = np.zeros((n, 2, 2), dtype=np.float32)
        players[:, 0] = [30.0, 34.0]
        players[:, 1] = [float(x[80]), 34.0]  # away player at the rekick point
        state = launch_track(make_flight_tracking(x, players, np.array([0, 1])))

        self.assertEqual(int(state.team[79]), 0)
        self.assertTrue(state.in_flight[81])
        self.assertEqual(int(state.team[81]), 1)
        self.assertFalse(state.in_flight[110])

    def test_detection_is_causal_to_future_positions(self):
        full = fixture_pass()
        state_full = launch_track(full)
        prefix = make_flight_tracking(
            full.ball[:90, 0].copy(),
            full.players[:90].copy(),
            np.array([0, 1]),
            present=full.ball_present[:90].copy(),
        )
        state_prefix = launch_track(prefix)

        np.testing.assert_array_equal(state_prefix.in_flight, state_full.in_flight[:90])
        np.testing.assert_array_equal(state_prefix.team, state_full.team[:90])

    def test_steady_moderate_play_never_launches(self):
        n = 60
        x = 0.16 * np.arange(n, dtype=np.float32)  # steady 4 m/s dribble
        players = np.array([[[0.0, 34.0], [10.0, 34.0]]] * n, dtype=np.float32)
        state = launch_track(make_flight_tracking(x, players, np.array([0, 1])))

        self.assertFalse(state.in_flight.any())
        self.assertTrue((state.team == -1).all())


if __name__ == "__main__":
    unittest.main()
