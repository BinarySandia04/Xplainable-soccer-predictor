"""Tests for the local synchronized video reviewer."""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from possession_lab.config import A, D, H, X
from possession_lab.data import TagEvent
from possession_lab.viewer import (
    FrameStore,
    build_user_truth_data,
    build_viewer_data,
    load_user_truth,
    make_server,
    move_user_truth_boundary,
    parse_byte_range,
    save_user_truth,
)


class ViewerDataTest(unittest.TestCase):
    def test_builds_compact_labels_spells_disagreements_and_tags(self):
        pred = np.array([X, H, H, A, D, D, X], dtype=np.int8)
        gt = np.array([X, H, A, A, D, D, X], dtype=np.int8)
        tag = TagEvent(
            start_frame=2,
            stop_frame=4,
            code="ATTACK",
            team=0,
            type="Phase",
            labels={"Half": "1"},
        )
        data = build_viewer_data(
            pred,
            gt,
            tags=SimpleNamespace(events=(tag,)),
            metrics=None,
            fps=25,
            halves=((1, 6),),
            public=(1, 6),
            start_frame=None,
        )
        self.assertEqual(data["labels"], "XHHADDX")
        self.assertEqual(data["gt"], "XHAADDX")
        self.assertEqual(data["pred_spells"], [[1, 3, H], [3, 4, A]])
        self.assertEqual(data["gt_spells"], [[1, 2, H], [2, 4, A]])
        self.assertEqual(data["disagreements"], [2])
        self.assertEqual(data["start_frame"], 1)
        self.assertTrue(data["gt_available"])
        self.assertEqual(len(data["tags"]), 1)
        self.assertEqual(data["tags"][0]["code"], "ATTACK")
        self.assertEqual(data["tags"][0]["team"], 0)

    def test_switch_annotations_ride_the_payload(self):
        pred = np.array([X, H, H, A, D, D, X], dtype=np.int8)
        explanations = {
            "switches": [
                {
                    "frame": 4,
                    "t_s": 0.16,
                    "from": "A",
                    "to": "D",
                    "confidence": 0.9,
                    "summary": "El balón sale del campo",
                    "reasons": [{"kind": "out_of_bounds", "text": "El balón sale del campo", "evidence": {"frame": 3}}],
                }
            ],
        }
        data = build_viewer_data(
            pred,
            None,
            tags=None,
            metrics=None,
            fps=25,
            halves=((1, 6),),
            public=(1, 6),
            start_frame=None,
            explanations=explanations,
        )
        self.assertTrue(data["explanations_available"])
        self.assertEqual(data["switches"][0]["frame"], 4)
        self.assertEqual(data["switches"][0]["reasons"][0]["kind"], "out_of_bounds")

    def test_payload_without_annotations_degrades_empty(self):
        pred = np.full(10, H, dtype=np.int8)
        data = build_viewer_data(pred, fps=25, halves=((0, 10),), public=(0, 10))
        self.assertEqual(data["switches"], [])
        self.assertFalse(data["explanations_available"])
        self.assertFalse(data["tracking_available"])
        self.assertEqual(data["events"], [])
        self.assertEqual(data["ball_gaps"], [])

    def test_event_and_gap_markers_ride_the_payload(self):
        pred = np.full(10, H, dtype=np.int8)
        goal = SimpleNamespace(to_dict=lambda: {"type": "goal", "frame": 3})
        stop = SimpleNamespace(to_dict=lambda: {"type": "stoppage", "start": 5, "end": 8})
        data = build_viewer_data(
            pred,
            tags=None,
            fps=25,
            halves=((0, 10),),
            public=(0, 10),
            events=SimpleNamespace(goals=(goal,), stoppages=(stop,)),
            ball_gaps=[[1, 2], [8, 10]],
            tracking_available=True,
        )
        self.assertTrue(data["tracking_available"])
        self.assertEqual([event["type"] for event in data["events"]], ["goal", "stoppage"])
        self.assertEqual(data["ball_gaps"], [[1, 2], [8, 10]])

    def test_prediction_spans_graded_halves_when_gt_is_public_only(self):
        pred = np.full(20, H, dtype=np.int8)
        gt = np.array([X] * 5 + [H] * 10 + [X] * 5, dtype=np.int8)
        data = build_viewer_data(
            pred,
            gt,
            tags=None,
            metrics=None,
            fps=25,
            halves=((0, 10), (12, 20)),
            public=(5, 15),
            start_frame=None,
        )
        self.assertEqual(data["pred_spells"], [[0, 10, H], [12, 20, H]])
        self.assertEqual(data["gt_spells"], [[5, 15, H]])
        self.assertEqual(data["pred_segments"], [[0, 10, H], [12, 20, H]])
        self.assertEqual(data["gt_segments"], [[5, 15, H]])

    def test_without_gt_uses_halves_and_marks_gt_unavailable(self):
        pred = np.full(10, H, dtype=np.int8)
        data = build_viewer_data(
            pred,
            None,
            tags=None,
            metrics=None,
            fps=25,
            halves=((2, 8),),
            public=(2, 8),
            start_frame=5,
        )
        self.assertFalse(data["gt_available"])
        self.assertIsNone(data["gt"])
        self.assertEqual(data["pred_spells"], [[2, 8, H]])
        self.assertEqual(data["disagreements"], [])
        self.assertEqual(data["start_frame"], 5)


class FrameStoreTopViewTest(unittest.TestCase):
    def test_payload_carries_pitch_coordinates_and_trail_for_top_view(self):
        tracking = SimpleNamespace(
            n_frames=3,
            ball=np.array([[10.0, 20.0], [np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32),
            ball_present=np.array([True, False, False]),
            players=np.array(
                [
                    [[10.0, 20.0], [30.0, 40.0]],
                    [[11.0, 21.0], [np.nan, np.nan]],
                    [[12.0, 22.0], [np.nan, np.nan]],
                ],
                dtype=np.float32,
            ),
            speeds=np.zeros((3, 2), dtype=np.float32),
            player_present=np.array([[True, True], [True, False], [True, False]]),
            teams=np.array([0, 1], dtype=np.int8),
            pitch=(105.3, 68.0),
        )
        homography = SimpleNamespace(
            matrices=np.repeat(
                np.array([[[1920.0, 0.0, 0.0], [0.0, 1080.0, 0.0], [0.0, 0.0, 1.0]]]),
                3,
                axis=0,
            )
        )
        track = SimpleNamespace(
            source=np.array([0, 1, 2], dtype=np.int8),
            estimate=np.array([[10.0, 20.0], [11.0, 21.0], [np.nan, np.nan]], dtype=np.float32),
            confidence=np.array([1.0, 0.4, 0.0], dtype=np.float32),
            gap_s=np.array([0.0, 0.04, 0.08], dtype=np.float32),
            out_of_play=np.zeros(3, dtype=bool),
        )
        store = FrameStore(
            tracking,
            homography,
            track,
            SimpleNamespace(goals=(), stoppages=()),
            SimpleNamespace(start=0),
            trail_frames=3,
        )
        payload = store.frame_payload(1)
        self.assertAlmostEqual(payload["pitch"]["length"], 105.3)
        self.assertAlmostEqual(payload["pitch"]["width"], 68.0)
        self.assertEqual(payload["players"][0]["x"], 11.0)
        self.assertEqual(payload["players"][0]["y"], 21.0)
        self.assertAlmostEqual(payload["players"][0]["px"], 11.0 / 105.3 * 1920.0, places=1)
        self.assertAlmostEqual(payload["players"][0]["py"], 21.0 / 68.0 * 1080.0, places=1)
        self.assertEqual(payload["ball"]["ex"], 11.0)
        self.assertEqual(payload["trail_pitch"], [[10.0, 20.0, 0], [11.0, 21.0, 1]])

        # The map is independent of video projection: a broken homography removes pixels
        # but keeps the pitch coordinates and pitch-space trail intact.
        homography.matrices[1] = np.nan
        payload = store.frame_payload(1)
        self.assertIsNone(payload["players"][0]["px"])
        self.assertIsNone(payload["players"][0]["py"])
        self.assertIsNone(payload["ball"]["px"])
        self.assertEqual(payload["players"][0]["x"], 11.0)
        self.assertEqual(payload["ball"]["ex"], 11.0)
        self.assertEqual(payload["trail_pitch"], [[10.0, 20.0, 0], [11.0, 21.0, 1]])


class UserTruthTest(unittest.TestCase):
    def test_scores_prediction_against_labeled_runs_only(self):
        pred = np.array([H, H, A, A, X, X], dtype=np.int8)
        user_truth = np.array([X, H, A, X, X, X], dtype=np.int8)
        data = build_user_truth_data(pred, user_truth)
        self.assertEqual(data["ut_segments"], [[1, 2, H], [2, 3, A]])
        self.assertEqual(data["ut_labeled_frames"], 2)
        self.assertEqual(data["ut_metrics"]["n_graded"], 2)
        self.assertEqual(data["ut_metrics"]["n_gt_spells"], 2)
        self.assertEqual(data["ut_metrics"]["n_pred_spells"], 2)

    def test_empty_user_truth_has_no_metrics(self):
        pred = np.full(4, H, dtype=np.int8)
        data = build_user_truth_data(pred, np.full(4, X, dtype=np.int8))
        self.assertEqual(data["ut_segments"], [])
        self.assertEqual(data["ut_labeled_frames"], 0)
        self.assertIsNone(data["ut_metrics"])

    def test_save_load_round_trip(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "user_truth.json"
            user_truth = np.array([X, H, H, X, A, A, D], dtype=np.int8)
            save_user_truth(path, user_truth)
            loaded = load_user_truth(path, user_truth.size)
            self.assertEqual(loaded.tolist(), user_truth.tolist())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["game"], "BAR-ATH")

    def test_moves_boundaries_in_both_directions(self):
        ut = np.array([H, H, X, X], dtype=np.int8)
        move_user_truth_boundary(ut, 2, 4)
        self.assertEqual(ut.tolist(), [H, H, H, H])
        with self.assertRaises(ValueError):
            move_user_truth_boundary(ut, 2, 3)

        ut = np.array([H, H, H, X], dtype=np.int8)
        move_user_truth_boundary(ut, 3, 1)
        self.assertEqual(ut.tolist(), [H, X, X, X])

    def test_moves_open_edges_and_keeps_boundary_valid(self):
        ut = np.array([X, A, A, X], dtype=np.int8)
        move_user_truth_boundary(ut, 1, 0)
        self.assertEqual(ut.tolist(), [A, A, A, X])
        move_user_truth_boundary(ut, 3, 2)
        self.assertEqual(ut.tolist(), [A, A, X, X])


class RangeTest(unittest.TestCase):
    def test_parses_normal_open_and_suffix_ranges(self):
        self.assertEqual(parse_byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(parse_byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_byte_range("bytes=-10", 100), (90, 99))

    def test_rejects_invalid_or_unsatisfiable_ranges(self):
        for header in ("not-a-range", "bytes=100-", "bytes=20-10", "bytes=0-1,3-4"):
            with self.subTest(header=header):
                with self.assertRaises(ValueError):
                    parse_byte_range(header, 100)


class ViewerServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "index.html").write_text("<html><body>viewer</body></html>", encoding="utf-8")
        (root / "viewer.js").write_text("console.log('viewer');", encoding="utf-8")
        (root / "viewer.css").write_text("body{color:white}", encoding="utf-8")
        self.video = root / "video.mp4"
        self.video.write_bytes(bytes(range(100)))
        self.payload = json.dumps({"labels": "HHAA"}).encode("utf-8")
        self.server = make_server(
            "127.0.0.1",
            0,
            payload=self.payload,
            video_path=self.video,
            assets_dir=root,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def test_serves_index_and_api_payload(self):
        with urlopen(self.base + "/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"viewer", response.read())
        with urlopen(self.base + "/api/viewer-data") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), self.payload)
            self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_range_request_returns_partial_content(self):
        request = Request(self.base + "/video", headers={"Range": "bytes=10-19"})
        with urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), bytes(range(10, 20)))
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")
            self.assertEqual(response.headers["Content-Range"], "bytes 10-19/100")

    def test_head_video_and_invalid_range(self):
        request = Request(self.base + "/video", method="HEAD")
        with urlopen(request) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")
            self.assertEqual(response.headers["Content-Length"], "100")
        request = Request(self.base + "/video", headers={"Range": "bytes=200-"})
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 416)

    def test_unknown_path_is_not_served(self):
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base + "/static/../index.html")
        self.assertEqual(raised.exception.code, 404)

    def test_frame_endpoint_is_unavailable_without_tracking(self):
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base + "/api/frame?frame=1")
        self.assertEqual(raised.exception.code, 404)

    def test_frame_endpoint_returns_compact_tracking_payload(self):
        root = Path(self.temp.name)
        store = SimpleNamespace(
            frame_payload=lambda frame: {
                "available": True,
                "frame": int(frame),
                "players": [],
                "ball": {"source": 2},
            }
        )
        server = make_server(
            "127.0.0.1",
            0,
            payload=self.payload,
            video_path=self.video,
            assets_dir=root,
            frame_store=store,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with urlopen(base + "/api/frame?frame=17") as response:
                self.assertEqual(response.status, 200)
                payload = json.loads(response.read())
            self.assertEqual(payload["frame"], 17)
            self.assertEqual(payload["ball"]["source"], 2)
            with self.assertRaises(HTTPError) as raised:
                urlopen(base + "/api/frame?frame=oops")
            self.assertEqual(raised.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_user_truth_post_is_rejected_when_not_configured(self):
        request = Request(
            self.base + "/api/user-truth",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 404)


class ViewerUserTruthServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "index.html").write_text("<html><body>viewer</body></html>", encoding="utf-8")
        (root / "viewer.js").write_text("console.log('viewer');", encoding="utf-8")
        (root / "viewer.css").write_text("body{color:white}", encoding="utf-8")
        self.video = root / "video.mp4"
        self.video.write_bytes(bytes(range(100)))
        self.truth_path = root / "user_truth.json"
        self.pred = np.array([H, H, A, A], dtype=np.int8)
        self.payload = json.dumps({"labels": "HHAA"}).encode("utf-8")
        self.server = make_server(
            "127.0.0.1",
            0,
            payload=self.payload,
            video_path=self.video,
            assets_dir=root,
            pred=self.pred,
            user_truth=np.full(self.pred.size, X, dtype=np.int8),
            user_truth_path=self.truth_path,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def post(self, body, content_type="application/json"):
        request = Request(
            self.base + "/api/user-truth",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read())

    def test_edit_updates_stats_persists_and_survives_reload(self):
        status, data = self.post({"start": 0, "stop": 2, "label": "H"})
        self.assertEqual(status, 200)
        self.assertEqual(data["ut_segments"], [[0, 2, H]])
        self.assertEqual(data["ut_labeled_frames"], 2)
        self.assertEqual(data["ut_metrics"]["n_graded"], 2)
        self.assertEqual(load_user_truth(self.truth_path, 4).tolist(), [H, H, X, X])
        with urlopen(self.base + "/api/viewer-data") as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["ut_segments"], [[0, 2, H]])
        self.assertEqual(payload["ut_labeled_frames"], 2)

    def test_painting_x_erases_and_clear_wipes_everything(self):
        self.post({"start": 0, "stop": 3, "label": "A"})
        _, data = self.post({"start": 1, "stop": 2, "label": "X"})
        self.assertEqual(data["ut_segments"], [[0, 1, A], [2, 3, A]])
        _, data = self.post({"clear": True})
        self.assertEqual(data["ut_labeled_frames"], 0)
        self.assertEqual(load_user_truth(self.truth_path, 4).tolist(), [X, X, X, X])

    def test_rejects_bad_label_and_wrong_content_type(self):
        with self.assertRaises(HTTPError) as raised:
            self.post({"start": 0, "stop": 1, "label": "Z"})
        self.assertEqual(raised.exception.code, 400)
        with self.assertRaises(HTTPError) as raised:
            self.post({"start": 0, "stop": 1, "label": "H"}, content_type="text/plain")
        self.assertEqual(raised.exception.code, 415)

    def test_moves_a_ut_boundary_and_persists(self):
        self.post({"start": 0, "stop": 2, "label": "H"})
        status, data = self.post({"boundary": 2, "to": 3})
        self.assertEqual(status, 200)
        self.assertEqual(data["ut_segments"], [[0, 3, H]])
        self.assertEqual(load_user_truth(self.truth_path, 4).tolist(), [H, H, H, X])
        with self.assertRaises(HTTPError) as raised:
            self.post({"boundary": 1, "to": 0})
        self.assertEqual(raised.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
