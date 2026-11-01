"""Tests for the transparent multi-source possession poll."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from possession_lab.config import A, D, H
from possession_lab.data import TagTable, TrackingData
from possession_lab.features import extract_features
from possession_lab.fusion import build_fusion, load_fusion, save_fusion
from possession_lab.fusion.ballot import ballot_reason
from possession_lab.fusion.resample import resample_perception
from possession_lab.fusion.voters import video_gap_dead
from possession_lab.perception import _empty_perception


def graded(n: int) -> tuple[tuple[int, int], ...]:
    """One graded window covering the whole synthetic clip, for deterministic decodes."""
    return ((0, n),)


def make_tracking(n: int = 80) -> TrackingData:
    # The ball is firmly with Barcelona for the first half and Athletic for the second. The
    # synthetic geometry is deliberately simple: fusion tests are about ballots, not football.
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = [10.0, 34.0]
    players[:, 1] = [90.0, 34.0]
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = 10.0
    ball[:, 1] = 34.0
    ball[n // 2 :] = [90.0, 34.0]
    return TrackingData(
        ball=ball,
        ball_present=np.ones(n, dtype=bool),
        players=players,
        speeds=np.full((n, 2), 2.0, dtype=np.float32),
        player_present=np.ones((n, 2), dtype=bool),
        teams=np.array([0, 1], dtype=np.int8),
        pitch=(105.3, 68.0),
        n_frames=n,
    )


class FusionBuildTest(unittest.TestCase):
    def _build(self, n: int = 80):
        tracking = make_tracking(n)
        tags = TagTable.empty(n)
        features = extract_features(tracking, tags)
        baseline = np.full(n, H, dtype=np.int8)
        baseline[n // 2 :] = A
        return tracking, tags, features, baseline

    def test_ballots_anchor_on_the_validated_hmm_and_keep_roles(self):
        tracking, tags, features, baseline = self._build()
        result = build_fusion(
            tracking,
            tags,
            features=features,
            baseline_codes=baseline,
            halves=graded(tracking.n_frames),
        )
        self.assertEqual(result.codes.shape, (tracking.n_frames,))
        self.assertEqual(result.fused_probability.shape, (tracking.n_frames, 3))
        np.testing.assert_allclose(result.fused_probability.sum(axis=1), 1.0, atol=1e-9)
        self.assertEqual(result.codes[0], H)
        self.assertEqual(result.codes[-1], A)
        self.assertEqual(
            set(result.sources),
            {
                "atd_ball",
                "atd_shape",
                "video_players",
                "video_ball",
                "yolo_ball",
                "hmm_atd",
                "tags",
                "audio",
                "events",
            },
        )
        sample = result.sample(0)
        self.assertEqual(sample["baseline_code"], H)
        roles = {source["key"]: source["role"] for source in sample["sources"]}
        self.assertEqual(roles["hmm_atd"], "prior")
        self.assertEqual(roles["events"], "override")
        self.assertEqual(roles["atd_ball"], "redundant")
        self.assertEqual(roles["video_ball"], "evidence")
        self.assertIn("HMM ATD", {source["label"] for source in sample["sources"]})

    def test_confirmed_event_is_a_hard_dead_override(self):
        tracking, tags, features, baseline = self._build()
        dead = np.zeros(tracking.n_frames, dtype=bool)
        goal = np.zeros(tracking.n_frames, dtype=bool)
        dead[40:45] = True
        goal[40:45] = True
        events = SimpleNamespace(
            dead_mask=dead,
            goal_dead=goal,
            stop_dead=np.zeros(tracking.n_frames, dtype=bool),
        )
        result = build_fusion(
            tracking,
            tags,
            features=features,
            events=events,
            baseline_codes=baseline,
            halves=graded(tracking.n_frames),
        )
        self.assertTrue(result.event_dead[40:45].all())
        np.testing.assert_array_equal(result.codes[40:45], np.full(5, D, dtype=np.int8))
        self.assertFalse(result.event_dead[39])
        self.assertFalse(result.event_dead[45])

    def test_linear_pool_is_available_and_unknown_modes_are_rejected(self):
        tracking, tags, features, baseline = self._build(40)
        linear = build_fusion(
            tracking,
            tags,
            features=features,
            baseline_codes=baseline,
            combine_mode="linear",
            halves=graded(tracking.n_frames),
        )
        self.assertEqual(linear.fused_probability.shape, (tracking.n_frames, 3))
        with self.assertRaises(ValueError):
            build_fusion(
                tracking,
                tags,
                features=features,
                baseline_codes=baseline,
                combine_mode="magic",
                halves=graded(tracking.n_frames),
            )

    def test_cache_round_trip_preserves_codes_and_combine_mode(self):
        tracking, tags, features, baseline = self._build(40)
        result = build_fusion(
            tracking,
            tags,
            features=features,
            baseline_codes=baseline,
            combine_mode="linear",
            halves=graded(tracking.n_frames),
        )
        with TemporaryDirectory() as temp:
            path = Path(temp) / "fusion.npz"
            save_fusion(path, result)
            loaded = load_fusion(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_array_equal(loaded.codes, result.codes)
            np.testing.assert_array_equal(loaded.baseline_codes, result.baseline_codes)
            self.assertEqual(loaded.meta["combine_mode"], "linear")
            self.assertEqual(set(loaded.sources), set(result.sources))
            self.assertEqual(loaded.sample(0)["sources"][0]["key"], "atd_ball")

    def test_cache_can_omit_the_baseline_comparison(self):
        tracking, tags, features, baseline = self._build(40)
        result = build_fusion(
            tracking,
            tags,
            features=features,
            baseline_codes=baseline,
            include_baseline=False,
            halves=graded(tracking.n_frames),
        )
        self.assertIsNone(result.baseline_codes)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "fusion.npz"
            save_fusion(path, result)
            loaded = load_fusion(path)
            assert loaded is not None
            self.assertIsNone(loaded.baseline_codes)

    def test_missing_fusion_cache_returns_none(self):
        with TemporaryDirectory() as temp:
            self.assertIsNone(load_fusion(Path(temp) / "absent.npz"))

    def test_halftime_between_active_halves_is_dead(self):
        tracking, tags, features, baseline = self._build()
        from possession_lab.pipeline import active_halves

        windows = active_halves(tags, ((0, 30), (35, 80)))
        result = build_fusion(
            tracking,
            tags,
            features=features,
            baseline_codes=baseline,
            halves=windows,
        )
        # the warm-up gap is neither a possession nor an abstention to forward-fill
        np.testing.assert_array_equal(result.codes[30:35], np.full(5, D, dtype=np.int8))
        self.assertFalse((result.codes[30:35] == H).any())
        self.assertFalse((result.codes[30:35] == A).any())


class VideoGapDeadTest(unittest.TestCase):
    """A blind window with the ball out on both sides is a retrieval gap, not live play."""

    @staticmethod
    def arrays(n: int = 240, gap_end: int = 160, last_out: bool = True, re_out: bool = True):
        atd_present = np.ones(n, dtype=bool)
        atd_present[55:gap_end] = False
        atd_ball = np.tile(np.array([50.0, 34.0]), (n, 1))
        atd_ball[gap_end:] = [50.0, -2.0] if re_out else [50.0, 34.0]
        video_ball = np.full((n, 2), np.nan)
        video_ball[:60] = [50.0, -1.0] if last_out else [50.0, 1.0]
        video_conf = np.zeros(n)
        video_conf[:60] = 0.5
        return atd_present, atd_ball, video_ball, video_conf

    def test_gap_between_two_outside_observations_is_dead(self):
        arrays = self.arrays()
        dead = video_gap_dead(*arrays, (105.0, 68.0))
        self.assertTrue(dead[60:160].all())
        self.assertFalse(dead[59])
        self.assertFalse(dead[160])

    def test_camera_cut_that_returns_inside_is_not_dead(self):
        arrays = self.arrays(re_out=False)
        dead = video_gap_dead(*arrays, (105.0, 68.0))
        self.assertFalse(dead.any())

    def test_gap_that_starts_inside_is_not_dead(self):
        arrays = self.arrays(last_out=False)
        dead = video_gap_dead(*arrays, (105.0, 68.0))
        self.assertFalse(dead.any())

    def test_short_gap_is_not_dead(self):
        arrays = self.arrays()
        dead = video_gap_dead(*arrays, (105.0, 68.0), min_gap_frames=120)
        self.assertFalse(dead.any())


class SourceReasonTest(unittest.TestCase):
    def test_video_ball_reason_names_the_tracker_origin(self):
        details = {
            "d_home": np.array([2.5]),
            "d_away": np.array([3.5]),
            "count": np.array([18], dtype=np.int16),
            "source": np.array([3], dtype=np.int8),
        }
        reason = ballot_reason("video_ball", 0, details, active=True)
        self.assertIn("anclado a ATD", reason)
        self.assertIn("Barça 2.5 m", reason)
        self.assertIn("18 jugadores", reason)


class PerceptionResampleTest(unittest.TestCase):
    def test_single_sample_cache_is_fresh_then_expires(self):
        perception = _empty_perception(
            samples=1,
            n_frames=30,
            fps=25,
            step=5,
            start=0,
            stop=30,
            model="test.pt",
            device="cpu",
            video_width=1920,
            video_height=1080,
        )
        perception.frame_index[0] = 0
        perception.ball_source[0] = 0
        perception.ball_pitch[0] = [52.65, 34.0]
        perception.ball_confidence[0] = 0.9
        video = resample_perception(perception, 30)
        self.assertEqual(video["ball_source"][0], 0)
        self.assertEqual(video["ball_source"][8], 0)
        self.assertEqual(video["ball_source"][9], 2)
        self.assertTrue(np.isnan(video["ball_pitch"][9]).any())

    def test_resampled_yolo_ball_uses_the_physical_gate(self):
        perception = _empty_perception(
            samples=3,
            n_frames=30,
            fps=25,
            step=5,
            start=0,
            stop=30,
            model="test.pt",
            device="cpu",
            video_width=1920,
            video_height=1080,
        )
        perception.frame_index[:] = [0, 5, 10]
        for sample, point in enumerate(([10.0, 10.0], [10.2, 10.0], [80.0, 50.0])):
            perception.yolo_ball_present[sample, 0] = True
            perception.yolo_ball_pitch[sample, 0] = point
            perception.yolo_ball_confidence[sample, 0] = 0.9
        video = resample_perception(perception, 30)
        self.assertTrue(video["yolo_ball_present"][5, 0])
        np.testing.assert_allclose(video["yolo_ball"][5, 0], [10.2, 10.0], atol=1e-6)
        self.assertFalse(video["yolo_ball_present"][10, 0])
        self.assertTrue(np.isnan(video["yolo_ball"][10, 0]).all())


if __name__ == "__main__":
    unittest.main()
