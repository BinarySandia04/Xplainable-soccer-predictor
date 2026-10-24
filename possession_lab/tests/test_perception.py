"""Tests for the independent video perception and its compact cache."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from possession_lab.perception import (
    VideoBallTracker,
    _GreedyTracker,
    _empty_perception,
    load_perception,
    pixel_to_pitch,
    pitch_to_pixel,
    save_perception,
)


class ProjectionTest(unittest.TestCase):
    def test_pitch_pixel_round_trip_with_translation(self):
        matrix = np.array(
            [[1920.0, 0.0, 100.0], [0.0, 1080.0, 20.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        pitch = (105.3, 68.0)
        points = np.array([[0.0, 0.0], [52.65, 34.0], [105.3, 68.0]])
        pixels = pitch_to_pixel(matrix, points, pitch)
        np.testing.assert_allclose(
            pixels,
            np.array([[100.0, 20.0], [1060.0, 560.0], [2020.0, 1100.0]]),
            atol=1e-9,
        )
        restored = pixel_to_pitch(matrix, pixels, pitch)
        np.testing.assert_allclose(restored, points, atol=1e-9)

    def test_invalid_projection_stays_nan(self):
        matrix = np.array(
            [[1920.0, 0.0, 0.0], [0.0, 1080.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        restored = pixel_to_pitch(
            matrix,
            np.array([[np.nan, 540.0], [960.0, 540.0]]),
            (105.3, 68.0),
        )
        self.assertTrue(np.isnan(restored[0]).all())
        np.testing.assert_allclose(restored[1], [52.65, 34.0], atol=1e-9)


class GreedyTrackerTest(unittest.TestCase):
    def test_iou_tracks_survive_small_motion_and_new_boxes_get_new_ids(self):
        tracker = _GreedyTracker(iou_threshold=0.3, max_age=2)
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        first = tracker.update(boxes, frame=0)
        second = tracker.update(boxes + 1.0, frame=1)
        np.testing.assert_array_equal(first, second)
        fresh = tracker.update(np.array([[100.0, 100.0, 110.0, 110.0]]), frame=2)
        self.assertEqual(int(fresh[0]), int(first.max()) + 1)


class VideoBallTrackerTest(unittest.TestCase):
    def setUp(self):
        self.homography = np.array(
            [[1920.0, 0.0, 0.0], [0.0, 1080.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.detection = {
            "pixel": [960.0, 540.0],
            "pitch": [52.65, 34.0],
            "confidence": 0.8,
            "origin": "yolo",
        }

    def test_seed_anchor_prediction_loss_and_reacquire(self):
        tracker = VideoBallTracker(max_gap_s=3.5)
        seeded = tracker.update(
            self.frame,
            0,
            self.homography,
            [self.detection],
            atd_anchor=None,
        )
        self.assertEqual(seeded["source"], 0)
        self.assertAlmostEqual(seeded["pitch"][0], 52.65)

        anchored = tracker.update(
            self.frame,
            1,
            self.homography,
            [],
            atd_anchor=np.array([52.65, 34.0]),
        )
        self.assertEqual(anchored["source"], 3)
        self.assertAlmostEqual(anchored["pitch"][1], 34.0)

        predicted = tracker.update(
            self.frame,
            2,
            self.homography,
            [],
            atd_anchor=None,
        )
        self.assertEqual(predicted["source"], 1)
        self.assertTrue(np.isfinite(predicted["pitch"]).all())

        lost = tracker.update(
            self.frame,
            102,
            self.homography,
            [],
            atd_anchor=None,
        )
        self.assertEqual(lost["source"], 2)
        self.assertTrue(np.isnan(lost["pitch"]).all())

        reacquired = tracker.update(
            self.frame,
            110,
            self.homography,
            [self.detection],
            atd_anchor=None,
        )
        self.assertEqual(reacquired["source"], 0)


class PerceptionCacheTest(unittest.TestCase):
    def test_save_load_round_trip_and_nearest_sample(self):
        perception = _empty_perception(
            samples=2,
            n_frames=10,
            fps=25,
            step=5,
            start=0,
            stop=10,
            model="test.pt",
            device="cpu",
            video_width=1920,
            video_height=1080,
        )
        perception.frame_index[:] = [0, 5]
        perception.players_present[0, 0] = True
        perception.players_pitch[0, 0] = [50.0, 34.0]
        perception.players_pixel[0, 0] = [910.0, 540.0]
        perception.players_team[0, 0] = 0
        perception.players_confidence[0, 0] = 0.9
        perception.ball_source[0] = 0
        perception.ball_pitch[0] = [52.65, 34.0]
        perception.ball_pixel[0] = [960.0, 540.0]
        perception.ball_confidence[0] = 0.8
        perception.yolo_ball_present[0, 0] = True
        perception.yolo_ball_pitch[0, 0] = [52.65, 34.0]
        perception.yolo_ball_pixel[0, 0] = [960.0, 540.0]
        perception.yolo_ball_confidence[0, 0] = 0.7

        with TemporaryDirectory() as temp:
            path = Path(temp) / "perception.npz"
            save_perception(path, perception)
            loaded = load_perception(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_array_equal(loaded.frame_index, [0, 5])
            self.assertEqual(loaded.samples, 2)
            self.assertEqual(loaded.model, "test.pt")
            self.assertEqual(loaded.nearest_index(4), 1)
            self.assertEqual(loaded.nearest_index(1), 0)
            sample = loaded.nearest(1)
            self.assertEqual(sample["sample_frame"], 0)
            self.assertAlmostEqual(sample["ball"]["x"], 52.65, places=4)
            self.assertEqual(len(sample["players"]), 1)
            meta = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
            self.assertEqual(meta["samples"], 2)

    def test_missing_cache_returns_none(self):
        with TemporaryDirectory() as temp:
            self.assertIsNone(load_perception(Path(temp) / "absent.npz"))


if __name__ == "__main__":
    unittest.main()
