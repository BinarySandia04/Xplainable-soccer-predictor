"""Tests for the causal Kalman ball layer used by the viewer."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.config import FPS
from possession_lab.data import TrackingData
from possession_lab.kalman import kalman_ball_track


def make_tracking(
    n: int,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    present: np.ndarray | None = None,
) -> TrackingData:
    if x is None:
        x = np.arange(n, dtype=np.float32)
    if y is None:
        y = np.full(n, 34.0, dtype=np.float32)
    ball = np.column_stack([x, y]).astype(np.float32)
    if present is None:
        present = np.ones(n, dtype=bool)
    present = np.asarray(present, dtype=bool)
    ball[~present] = np.nan
    players = np.zeros((n, 1, 2), dtype=np.float32)
    players[:, 0] = [10.0, 10.0]
    return TrackingData(
        n_frames=n,
        ball=ball,
        ball_present=present,
        players=players,
        speeds=np.zeros((n, 1), dtype=np.float32),
        player_present=np.ones((n, 1), dtype=bool),
        teams=np.array([0], dtype=np.int8),
        pitch=(105.3, 68.0),
    )


class KalmanBallTrackTest(unittest.TestCase):
    def test_tracks_constant_velocity_observations(self):
        n = 60
        x = 10.0 + 0.08 * np.arange(n)  # 2 m/s at 25 fps
        track = kalman_ball_track(make_tracking(n, x=x))

        self.assertEqual(track.source[0], 0)
        self.assertEqual(track.source[-1], 0)
        self.assertTrue(np.isfinite(track.estimate[-1, 0]))
        self.assertAlmostEqual(float(track.estimate[-1, 0]), float(x[-1]), delta=0.4)
        self.assertAlmostEqual(float(track.velocity[-1, 0]), 2.0, delta=0.35)
        self.assertGreater(float(track.confidence[-1]), 0.9)

    def test_predicts_short_gap_and_refuses_long_gap(self):
        n = 50
        x = 10.0 + 1.5 * np.arange(n)
        present = np.ones(n, dtype=bool)
        present[20:26] = False  # 0.24 s: inside the default prediction horizon
        track = kalman_ball_track(make_tracking(n, x=x, present=present), max_prediction_s=0.5)

        self.assertEqual(track.source[20], 1)
        self.assertTrue(np.isfinite(track.estimate[20, 0]))
        self.assertAlmostEqual(float(track.estimate[20, 0]), float(x[20]), delta=0.5)
        self.assertGreater(float(track.confidence[20]), 0.0)

        present_long = np.ones(n, dtype=bool)
        present_long[20:40] = False  # 0.80 s: beyond the configured horizon
        long_track = kalman_ball_track(
            make_tracking(n, x=x, present=present_long),
            max_prediction_s=0.5,
        )
        self.assertEqual(long_track.source[35], 2)
        self.assertFalse(np.isfinite(long_track.estimate[35, 0]))

    def test_rejects_a_single_large_outlier(self):
        n = 30
        x = np.full(n, 10.0, dtype=np.float32)
        y = np.full(n, 34.0, dtype=np.float32)
        x[15] = 80.0
        y[15] = 60.0
        track = kalman_ball_track(make_tracking(n, x=x, y=y))

        self.assertTrue(np.isfinite(track.estimate[15, 0]))
        self.assertLess(float(track.estimate[15, 0]), 30.0)
        self.assertLess(float(track.estimate[15, 1]), 42.0)

    def test_reports_missing_before_first_observation(self):
        n = 20
        present = np.zeros(n, dtype=bool)
        track = kalman_ball_track(make_tracking(n, present=present))

        self.assertTrue(np.all(track.source == 2))
        self.assertTrue(np.all(np.isnan(track.estimate)))
        self.assertEqual(int(track.observed.sum()), 0)

    def test_gap_age_is_in_seconds(self):
        n = 20
        present = np.ones(n, dtype=bool)
        present[10:14] = False
        track = kalman_ball_track(make_tracking(n, present=present), max_prediction_s=1.0)

        self.assertAlmostEqual(float(track.gap_s[10]), 1.0 / FPS, places=4)
        self.assertAlmostEqual(float(track.gap_s[13]), 4.0 / FPS, places=4)
        self.assertAlmostEqual(float(track.gap_s[14]), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
