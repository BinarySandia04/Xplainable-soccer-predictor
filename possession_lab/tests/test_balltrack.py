"""Tests for the causal ball tracker used across short tracking holes."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.balltrack import track_ball
from possession_lab.config import BALL_TRACK_MAX_FILL_S, FPS
from possession_lab.data import TrackingData


def make_tracking(n: int, x: np.ndarray | None = None, present: np.ndarray | None = None) -> TrackingData:
    if x is None:
        x = np.arange(n, dtype=np.float32)
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = x
    ball[:, 1] = 34.0
    if present is None:
        present = np.ones(n, dtype=bool)
    present = np.asarray(present, dtype=bool)
    ball[~present] = np.nan
    players = np.zeros((n, 1, 2), dtype=np.float32)
    players[:, 0] = [10.0, 10.0]
    return TrackingData(
        ball=ball,
        ball_present=present,
        players=players,
        speeds=np.zeros((n, 1), dtype=np.float32),
        player_present=np.ones((n, 1), dtype=bool),
        teams=np.array([0], dtype=np.int8),
        pitch=(105.3, 68.0),
        n_frames=n,
    )


class BallTrackerTest(unittest.TestCase):
    def test_short_gap_is_extrapolated_from_the_last_velocity(self):
        n = 40
        present = np.ones(n, dtype=bool)
        present[10:16] = False
        track = track_ball(make_tracking(n, present=present))

        self.assertEqual(track.source[9], 0)
        self.assertEqual(track.source[10], 1)
        self.assertGreater(float(track.estimate[10, 0]), 9.0)
        self.assertLess(float(track.estimate[10, 0]), 10.1)
        self.assertFalse(track.out_of_play[10])
        self.assertGreater(float(track.confidence[10]), 0.0)
        self.assertGreater(float(track.confidence[10]), float(track.confidence[15]))

    def test_long_gap_is_left_empty_instead_of_inventing_a_position(self):
        n = 80
        present = np.ones(n, dtype=bool)
        present[10:] = False
        track = track_ball(make_tracking(n, present=present))
        last_fill = 9 + int(round(BALL_TRACK_MAX_FILL_S * FPS))

        self.assertEqual(track.source[11], 1)
        self.assertEqual(track.source[last_fill], 1)
        self.assertEqual(track.source[last_fill + 1], 2)
        self.assertTrue(np.isnan(track.estimate[last_fill + 1, 0]))
        self.assertEqual(track.source[-1], 2)
        self.assertAlmostEqual(float(track.gap_s[-1]), (n - 10) / FPS)

    def test_observed_mask_cannot_see_through_a_removed_ball(self):
        n = 20
        mask = np.ones(n, dtype=bool)
        mask[5:12] = False
        track = track_ball(make_tracking(n), observed_mask=mask)

        self.assertFalse(track.observed[5])
        self.assertEqual(track.source[5], 1)
        self.assertGreater(float(track.confidence[5]), 0.0)

    def test_ball_beyond_the_touchline_is_flagged(self):
        n = 30
        x = np.full(n, 104.0, dtype=np.float32)
        x[10:] = 140.0
        present = np.ones(n, dtype=bool)
        tracking = make_tracking(n, x=x, present=present)
        # A single observation at the new position is enough to mark it out of play; the test is
        # about geometry, not the velocity estimate.
        track = track_ball(tracking)
        self.assertFalse(bool(track.out_of_play[9]))
        self.assertTrue(bool(track.out_of_play[10]))


if __name__ == "__main__":
    unittest.main()
