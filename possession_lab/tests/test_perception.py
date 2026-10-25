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
    select_plausible_ball_detections,
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
        # One detection is only a candidate; a second reachable observation confirms it.
        self.assertEqual(seeded["source"], 2)
        seeded = tracker.update(
            self.frame,
            1,
            self.homography,
            [self.detection],
            atd_anchor=None,
        )
        self.assertEqual(seeded["source"], 0)
        self.assertAlmostEqual(seeded["pitch"][0], 52.65)

        anchored = tracker.update(
            self.frame,
            2,
            self.homography,
            [],
            atd_anchor=np.array([52.65, 34.0]),
        )
        self.assertEqual(anchored["source"], 3)
        self.assertAlmostEqual(anchored["pitch"][1], 34.0)

        predicted = tracker.update(
            self.frame,
            3,
            self.homography,
            [],
            atd_anchor=None,
        )
        self.assertEqual(predicted["source"], 1)
        self.assertTrue(np.isfinite(predicted["pitch"]).all())

        lost = tracker.update(
            self.frame,
            103,
            self.homography,
            [],
            atd_anchor=None,
        )
        self.assertEqual(lost["source"], 2)
        self.assertTrue(np.isnan(lost["pitch"]).all())

        reacquired = tracker.update(
            self.frame,
            111,
            self.homography,
            [self.detection],
            atd_anchor=None,
        )
        self.assertEqual(reacquired["source"], 2)
        reacquired = tracker.update(
            self.frame,
            112,
            self.homography,
            [self.detection],
            atd_anchor=None,
        )
        self.assertEqual(reacquired["source"], 0)

    def test_far_jump_is_rejected_and_does_not_move_the_ball(self):
        tracker = VideoBallTracker()
        tracker.update(self.frame, 0, self.homography, [self.detection], atd_anchor=None)
        tracker.update(self.frame, 1, self.homography, [self.detection], atd_anchor=None)
        before = tracker.last_pitch.copy()
        far = {
            "pixel": [1700.0, 900.0],
            "pitch": [90.0, 55.0],
            "confidence": 0.99,
            "origin": "yolo",
        }
        result = tracker.update(self.frame, 2, self.homography, [far], atd_anchor=None)
        self.assertEqual(result["source"], 1)
        self.assertLess(float(np.linalg.norm(np.asarray(result["pitch"]) - before)), 0.2)

    def test_high_confidence_detection_just_outside_gate_is_rejected(self):
        tracker = VideoBallTracker()
        tracker.update(self.frame, 0, self.homography, [self.detection], atd_anchor=None)
        tracker.update(self.frame, 1, self.homography, [self.detection], atd_anchor=None)
        just_outside = {
            "pixel": [1000.0, 540.0],
            "pitch": [55.65, 34.0],
            "confidence": 0.99,
            "origin": "yolo",
        }
        result = tracker.update(
            self.frame, 2, self.homography, [just_outside], atd_anchor=None
        )
        self.assertEqual(result["source"], 1)
        self.assertAlmostEqual(result["pitch"][0], 52.65, places=3)

    def test_prediction_cannot_bypass_the_last_accepted_gate(self):
        tracker = VideoBallTracker()
        start = {
            "pixel": [700.0, 540.0],
            "pitch": [50.0, 34.0],
            "confidence": 0.8,
            "origin": "yolo",
        }
        moving = {
            "pixel": [720.0, 540.0],
            "pitch": [51.0, 34.0],
            "confidence": 0.8,
            "origin": "yolo",
        }
        tracker.update(self.frame, 0, self.homography, [start], atd_anchor=None)
        tracker.update(self.frame, 1, self.homography, [moving], atd_anchor=None)
        # This point is close to the constant-velocity prediction but 3 m from the last accepted
        # point; the prediction must not widen the physical gate.
        ahead = {
            "pixel": [780.0, 540.0],
            "pitch": [54.0, 34.0],
            "confidence": 0.99,
            "origin": "yolo",
        }
        result = tracker.update(self.frame, 2, self.homography, [ahead], atd_anchor=None)
        self.assertEqual(result["source"], 1)
        self.assertLess(float(result["pitch"][0]), 52.0)

    def test_single_false_positive_cannot_seed_a_track(self):
        tracker = VideoBallTracker()
        first = tracker.update(self.frame, 0, self.homography, [self.detection], atd_anchor=None)
        self.assertEqual(first["source"], 2)
        far = {
            "pixel": [1700.0, 900.0],
            "pitch": [90.0, 55.0],
            "confidence": 0.99,
            "origin": "yolo",
        }
        second = tracker.update(self.frame, 1, self.homography, [far], atd_anchor=None)
        self.assertEqual(second["source"], 2)
        self.assertFalse(tracker.alive)

    def test_large_atd_anchor_jump_is_an_anchor_not_a_yolo_detection(self):
        tracker = VideoBallTracker()
        tracker.update(self.frame, 0, self.homography, [self.detection], atd_anchor=None)
        tracker.update(self.frame, 1, self.homography, [self.detection], atd_anchor=None)
        far = {
            "pixel": [1700.0, 900.0],
            "pitch": [90.0, 55.0],
            "confidence": 0.99,
            "origin": "yolo",
        }
        result = tracker.update(
            self.frame,
            2,
            self.homography,
            [far],
            atd_anchor=np.array([90.0, 55.0]),
        )
        self.assertEqual(result["source"], 3)
        np.testing.assert_allclose(result["pitch"], [90.0, 55.0], atol=1e-9)


class PlausibleBallSelectionTest(unittest.TestCase):
    def _inputs(self, points, frames=None):
        points = np.asarray(points, dtype=np.float64)
        count = points.shape[0]
        confidence = np.full((count, 1), 0.8, dtype=np.float64)
        present = np.ones((count, 1), dtype=bool)
        if frames is None:
            frames = np.arange(count) * 5
        return points[:, None, :], confidence, present, np.asarray(frames, dtype=np.float64)

    def test_far_detection_is_rejected_but_the_next_reachable_one_is_kept(self):
        points = np.array([[10.0, 10.0], [10.2, 10.0], [70.0, 10.0], [10.4, 10.0]])
        ball, confidence, status, selected = select_plausible_ball_detections(
            *self._inputs(points, frames=[0, 5, 10, 15])
        )
        self.assertTrue(np.isnan(ball[0]).all())
        np.testing.assert_allclose(ball[1], points[1], atol=1e-9)
        self.assertTrue(np.isnan(ball[2]).all())
        np.testing.assert_allclose(ball[3], points[3], atol=1e-9)
        self.assertEqual(int(status[1]), 0)
        self.assertEqual(int(status[2]), 2)
        self.assertEqual(int(selected[2]), -1)
        self.assertGreater(float(confidence[1]), 0.0)

    def test_two_mutually_reachable_detections_are_required_to_start(self):
        points = np.array([[70.0, 10.0], [10.0, 10.0], [70.0, 10.0]])
        ball, _, _, _ = select_plausible_ball_detections(
            *self._inputs(points, frames=[0, 5, 10])
        )
        self.assertTrue(np.isnan(ball).all())

    def test_reacquisition_after_a_gap_needs_a_reachable_confirmation(self):
        points = np.array(
            [[10.0, 10.0], [10.2, 10.0], [20.0, 10.0], [20.2, 10.0]]
        )
        ball, _, _, _ = select_plausible_ball_detections(
            *self._inputs(points, frames=[0, 5, 100, 105])
        )
        self.assertTrue(np.isfinite(ball[1]).all())
        self.assertTrue(np.isnan(ball[2]).all())
        np.testing.assert_allclose(ball[3], points[3], atol=1e-9)


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
