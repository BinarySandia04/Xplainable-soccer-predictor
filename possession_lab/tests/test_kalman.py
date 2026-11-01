"""Tests for the ATD + YOLO Kalman consensus ball."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.config import FPS
from possession_lab.fusion.kalman import consensus_ball


def straight_ball(n: int, speed_ms: float = 10.0) -> np.ndarray:
    step = speed_ms / FPS
    return np.stack([np.arange(n, dtype=np.float64) * step, np.full(n, 30.0)], axis=1)


class KalmanConsensusTest(unittest.TestCase):
    def test_recovers_linear_motion_and_predicts_gaps(self):
        n = 100
        truth = straight_ball(n)
        rng = np.random.default_rng(7)
        atd = truth + rng.normal(0.0, 0.05, size=(n, 2))
        present = np.ones(n, dtype=bool)
        present[40:60] = False
        pos, conf = consensus_ball(atd, present, np.full((n, 2), np.nan), np.zeros(n))
        self.assertTrue(np.isfinite(pos).all())
        self.assertEqual(conf[40:60].tolist(), [0.0] * 20)  # predicted frames are not observed
        self.assertTrue((conf[present] > 0).all())
        after_gap = np.linalg.norm(pos[65:] - truth[65:], axis=1)
        self.assertLess(float(after_gap.max()), 0.5)

    def test_rejects_a_single_frame_hop(self):
        n = 80
        truth = np.stack([np.full(n, 50.0), np.full(n, 34.0)], axis=1)
        atd = truth + np.random.default_rng(3).normal(0.0, 0.05, size=(n, 2))
        atd[50] += [2.0, -2.0]  # a two-metre detector hop
        present = np.ones(n, dtype=bool)
        pos, _ = consensus_ball(atd, present, np.full((n, 2), np.nan), np.zeros(n))
        self.assertLess(float(np.linalg.norm(pos[50] - truth[50])), 0.6)
        self.assertLess(float(np.linalg.norm(pos[52] - truth[52])), 0.2)

    def test_yolo_takes_over_while_atd_is_missing(self):
        n = 60
        truth = straight_ball(n)
        atd = truth.copy()
        present = np.ones(n, dtype=bool)
        present[20:50] = False
        yolo = truth + np.random.default_rng(11).normal(0.0, 0.1, size=(n, 2))
        yolo_conf = np.where(present, 0.0, 0.9)
        pos, conf = consensus_ball(atd, present, yolo, yolo_conf)
        self.assertTrue((conf[20:50] > 0).all())
        self.assertLess(float(np.linalg.norm(pos[45] - truth[45])), 0.5)


if __name__ == "__main__":
    unittest.main()
