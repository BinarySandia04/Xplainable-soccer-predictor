"""Local, synchronized video reviewer for the possession predictions.

The reviewer is intentionally boring infrastructure: one local HTTP server, one static page, and
one MP4.  It never copies the gated video and it never writes predictions.  The browser maps the
tactical frame timeline to rows of ``predictions.csv``; the public ground truth is drawn only where
it exists.

The reviewer also lets a human write a **user truth (UT)** used only to score the fixed prediction
in the UI.  The UT is stored in ``outputs/user_truth.json`` between sessions.  Nothing in the
predict/train path imports this module, so the model can never see the UT.

Command::

    .venv/bin/python -m possession_lab viewer

The server listens on ``127.0.0.1`` and only exposes the configured MP4, the static UI, and the
viewer JSON.  Byte-range requests are implemented explicitly because the tactical MP4 is 5 GB and
its ``moov`` atom is at the end of the file.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .config import (
    CODE_TO_LABEL,
    EXPLANATIONS_PATH,
    FPS,
    GAME,
    GRADED_HALVES,
    LABEL_TO_CODE,
    N_FRAMES,
    PITCH,
    PUBLIC_GT_PATH,
    PUBLIC_START,
    PUBLIC_STOP,
    ROOT,
    SOURCE,
    USER_TRUTH_PATH,
    WHISTLE_CACHE,
    A,
    D,
    H,
    X,
    asset_paths,
)
from .data import TagTable, load_public_gt, load_smart_tags, load_submission
from .perception import _run_ball_track

# Viewer-only ball association.  The yolo layer recomputes the ball track when the viewer loads, so
# these extensions are visual evidence only: they never touch fusion.npz, predictions.csv or any
# metric, because fusion recomputes the strict track on its own.  ``low_confidence_floor`` enables a
# ByteTrack/BotSORT second stage that keeps the ball track alive across a one-frame detector flicker;
# ``occlusion_radius_m`` is the guard that drops a low-confidence candidate sitting on a
# player as a likely shirt patch.  Kept here (not in config) to make the offline/online split obvious.
YOLO_VIEW_FLOOR = 0.05
YOLO_VIEW_OCCLUSION_M = 0.5

# The official scorer is the source of truth for possession segments/spells, just as it is for the
# report.  Import it lazily-safe by putting the repository root on sys.path first.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
RANGE_ITEM_RE = re.compile(r"^(\d*)-(\d*)$")
# Largest JSON body the user-truth endpoint accepts.
MAX_POST_BYTES = 100_000
MIME_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "viewer.js": "text/javascript; charset=utf-8",
    "viewer.css": "text/css; charset=utf-8",
}


def _labels_from_codes(codes: np.ndarray) -> str:
    return "".join(CODE_TO_LABEL[int(code)] for code in codes)


def _validate_codes(codes: np.ndarray, name: str) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 1 or codes.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not set(np.unique(codes).tolist()).issubset(set(CODE_TO_LABEL)):
        raise ValueError(f"{name} contains labels outside {{H,A,D,X}}")
    return codes.astype(np.int8, copy=False)


def _metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact metric fields shared by the public and user-truth stats."""
    return {
        "spell_f1": float(metrics["spell_f1_mean"]),
        "macro_f1": float(metrics["macro_f1"]),
        "d_f1": float(metrics["per_class"]["D"]["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "n_pred_spells": int(metrics["n_pred_spells"]),
        "n_gt_spells": int(metrics["n_gt_spells"]),
        "n_graded": int(metrics.get("n_graded", 0)),
    }


def _event_payload(events: Any | None) -> list[dict[str, Any]]:
    """Compact goal/stoppage markers for the timelines."""
    if events is None:
        return []
    payload: list[dict[str, Any]] = []
    for goal in getattr(events, "goals", ()):
        payload.append(goal.to_dict())
    for stop in getattr(events, "stoppages", ()):
        payload.append(stop.to_dict())
    payload.sort(key=lambda item: int(item.get("frame", item.get("start", 0))))
    return payload


def build_viewer_data(
    pred: np.ndarray,
    gt: np.ndarray | None = None,
    tags: TagTable | Any | None = None,
    metrics: dict[str, Any] | None = None,
    *,
    fps: int = FPS,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
    start_frame: int | None = None,
    explanations: dict[str, Any] | None = None,
    events: Any | None = None,
    ball_gaps: list[list[int]] | None = None,
    tracking_available: bool = False,
    perception_available: bool = False,
    fusion_available: bool = False,
) -> dict[str, Any]:
    """Build the compact payload consumed by the browser.

    ``pred`` is a full-match array.  ``gt`` is optional; when present, its non-X runs define the
    graded sections.  All spell/segment boundaries come from the official scorer rather than a
    second run-length implementation.
    """
    pred = _validate_codes(pred, "pred")
    n_frames = int(pred.size)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if public[0] < 0 or public[1] > n_frames or public[0] >= public[1]:
        raise ValueError(f"invalid public window {public} for {n_frames} frames")
    for start, stop in halves:
        if start < 0 or stop > n_frames or start >= stop:
            raise ValueError(f"invalid graded half {(start, stop)} for {n_frames} frames")

    pred_runs = [tuple(map(int, half)) for half in halves]
    gt_runs: list[tuple[int, int]] = []
    if gt is not None:
        gt = _validate_codes(gt, "gt")
        if gt.size != n_frames:
            raise ValueError(f"gt has {gt.size} frames; pred has {n_frames}")
        gt_runs = pm.valid_runs(gt)
        if not gt_runs:
            gt_runs = pred_runs

    # Prediction is evaluated over both full graded halves.  Public GT only exists in its public
    # window, so its own runs are used exclusively for the dashed GT band.
    pred_segments = [[int(a), int(b), int(c)] for a, b, c in pm.segments(pred, pred_runs)]
    pred_spells = [[int(a), int(b), int(c)] for a, b, c in pm.possession_spells(pred, pred_runs)]

    gt_segments: list[list[int]] = []
    gt_spells: list[list[int]] = []
    disagreements: list[int] = []
    if gt is not None:
        gt_segments = [[int(a), int(b), int(c)] for a, b, c in pm.segments(gt, gt_runs)]
        gt_spells = [[int(a), int(b), int(c)] for a, b, c in pm.possession_spells(gt, gt_runs)]
        disagreements = np.flatnonzero((gt != X) & (pred != gt)).astype(int).tolist()

    changes: set[int] = set()
    for start, stop in pred_runs:
        start, stop = int(start), int(stop)
        window = pred[start:stop]
        if window.size > 1:
            changes.update((np.flatnonzero(window[1:] != window[:-1]) + 1 + start).tolist())
        if start > 0:
            changes.add(start)
        if stop < n_frames:
            changes.add(stop)

    tags_payload: list[dict[str, Any]] = []
    for event in getattr(tags, "events", ()) if tags is not None else ():
        tags_payload.append(
            {
                "start": int(event.start_frame),
                "stop": int(event.stop_frame),
                "code": str(event.code),
                "team": int(event.team),
                "type": str(event.type),
                "labels": {str(key): str(value) for key, value in event.labels.items()},
            }
        )

    if start_frame is None:
        start_frame = int(public[0])
    start_frame = max(0, min(int(start_frame), n_frames - 1))

    explanations = explanations if isinstance(explanations, dict) else {}
    switches = [dict(s) for s in explanations.get("switches", ()) if isinstance(s, dict) and "frame" in s]

    counts = {
        label: int(np.sum(pred == code))
        for label, code in (("H", H), ("A", A), ("D", D), ("X", X))
    }
    metric_payload: dict[str, Any] | None = None
    if metrics is not None:
        metric_payload = _metrics_payload(metrics)

    return {
        "version": 1,
        "game": GAME,
        "source": SOURCE,
        "fps": int(fps),
        "n_frames": n_frames,
        "duration_s": float(n_frames / fps),
        "public": [int(public[0]), int(public[1])],
        "halves": [[int(start), int(stop)] for start, stop in halves],
        "start_frame": int(start_frame),
        "labels": _labels_from_codes(pred),
        "gt": _labels_from_codes(gt) if gt is not None else None,
        "gt_available": gt is not None,
        "label_counts": counts,
        "pred_segments": pred_segments,
        "gt_segments": gt_segments,
        "pred_spells": pred_spells,
        "gt_spells": gt_spells,
        "changes": sorted(changes),
        "disagreements": disagreements,
        "tags": tags_payload,
        "metrics": metric_payload,
        "switches": switches,
        "explanations_available": bool(switches),
        "tracking_available": bool(tracking_available),
        "perception_available": bool(perception_available),
        "fusion_available": bool(fusion_available),
        "events": _event_payload(events),
        "ball_gaps": [[int(a), int(b)] for a, b in (ball_gaps or ())],
    }


def load_user_truth(path: str | Path | None, n_frames: int = N_FRAMES) -> np.ndarray:
    """Load the viewer-only user truth, or an all-``X`` array when it does not exist yet."""
    user_truth = np.full(int(n_frames), X, dtype=np.int8)
    if path is None:
        return user_truth
    path = Path(path)
    if not path.is_file():
        return user_truth
    document = json.loads(path.read_text(encoding="utf-8"))
    runs = document.get("runs") if isinstance(document, dict) else None
    if not isinstance(runs, list):
        raise ValueError(f"{path}: expected a JSON object with a 'runs' list")
    for run in runs:
        if not isinstance(run, list) or len(run) != 3:
            raise ValueError(f"{path}: every run must be [start, stop, label]")
        start, stop, label = int(run[0]), int(run[1]), str(run[2]).strip().upper()
        if label not in LABEL_TO_CODE:
            raise ValueError(f"{path}: unknown label {label!r}")
        if not (0 <= start < stop <= int(n_frames)):
            raise ValueError(f"{path}: run [{start}, {stop}) outside 0..{n_frames}")
        user_truth[start:stop] = LABEL_TO_CODE[label]
    return user_truth


def save_user_truth(path: str | Path, user_truth: np.ndarray) -> Path:
    """Persist the user truth as compact constant-label runs (atomic replace)."""
    path = Path(path)
    user_truth = _validate_codes(user_truth, "user_truth")
    runs = [
        [int(start), int(stop), CODE_TO_LABEL[int(code)]]
        for start, stop, code in pm.segments(user_truth, pm.valid_runs(user_truth))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "game": GAME, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def move_user_truth_boundary(
    user_truth: np.ndarray, from_frame: int, to_frame: int
) -> np.ndarray:
    """Move a UT run boundary in place.

    ``from_frame`` must sit between two different labels (``0`` and ``n_frames`` count as
    boundaries when the edge label differs from ``X``).  Moving right gives the left label the
    freed frames; moving left gives them to the right label.
    """
    user_truth = np.asarray(user_truth)
    n = int(user_truth.size)
    from_frame, to_frame = int(from_frame), int(to_frame)
    if not (0 <= from_frame <= n and 0 <= to_frame <= n):
        raise ValueError(f"boundary {from_frame} or target {to_frame} outside 0..{n}")
    left = user_truth[from_frame - 1] if from_frame > 0 else X
    right = user_truth[from_frame] if from_frame < n else X
    if left == right:
        raise ValueError(f"frame {from_frame} is not a user-truth boundary")
    if to_frame > from_frame:
        user_truth[from_frame:to_frame] = left
    elif to_frame < from_frame:
        user_truth[to_frame:from_frame] = right
    return user_truth


def erase_nearest_user_truth(user_truth: np.ndarray, frame: int) -> np.ndarray:
    """Erase the UT run closest to ``frame``; leave the array untouched when there are no runs."""
    user_truth = np.asarray(user_truth)
    n = int(user_truth.size)
    frame = int(frame)
    if not (0 <= frame < n):
        raise ValueError(f"frame {frame} outside 0..{n - 1}")
    runs = pm.segments(user_truth, pm.valid_runs(user_truth))
    if not runs:
        return user_truth
    start, stop, _ = min(
        runs, key=lambda run: max(run[0] - frame, frame - run[1] + 1, 0)
    )
    user_truth[start:stop] = X
    return user_truth


def build_user_truth_data(pred: np.ndarray, user_truth: np.ndarray) -> dict[str, Any]:
    """Score the fixed prediction against the user truth.

    This is the only place the UT is consumed.  ``pred`` is never modified, and the model path
    (features/model/pipeline) has no access to this function or the UT file.
    """
    pred = _validate_codes(pred, "pred")
    user_truth = _validate_codes(user_truth, "user_truth")
    if user_truth.size != pred.size:
        raise ValueError(f"user truth has {user_truth.size} frames; pred has {pred.size}")

    runs = pm.valid_runs(user_truth)
    labeled = int(np.count_nonzero(user_truth != X))
    metrics = pm.evaluate(user_truth, pred, bootstrap_n=0) if labeled else None
    return {
        "ut_segments": [[int(a), int(b), int(c)] for a, b, c in pm.segments(user_truth, runs)],
        "ut_labeled_frames": labeled,
        "ut_metrics": _metrics_payload(metrics) if metrics is not None else None,
    }


def load_viewer_data(
    predictions_path: str | Path,
    *,
    gt_path: str | Path | None = PUBLIC_GT_PATH,
    tags_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    user_truth_path: str | Path | None = USER_TRUTH_PATH,
    explanations_path: str | Path | None = EXPLANATIONS_PATH,
    start_frame: int | None = None,
    events: Any | None = None,
    ball_gaps: list[list[int]] | None = None,
    tracking_available: bool = False,
    perception_available: bool = False,
    fusion_available: bool = False,
) -> dict[str, Any]:
    """Read the generated CSV outputs and build the viewer payload."""
    pred = load_submission(predictions_path)
    if pred.size != N_FRAMES:
        raise ValueError(f"{predictions_path}: expected {N_FRAMES:,} rows, got {pred.size:,}")

    gt = None
    if gt_path is not None and Path(gt_path).is_file():
        gt = load_public_gt(gt_path)
        if gt.size != pred.size:
            raise ValueError(f"{gt_path}: expected {pred.size:,} rows, got {gt.size:,}")

    tags = None
    if tags_path is not None and Path(tags_path).is_file():
        tags = load_smart_tags(tags_path, n_frames=pred.size)

    metrics = None
    if metrics_path is not None and Path(metrics_path).is_file():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

    explanations = None
    if explanations_path is not None and Path(explanations_path).is_file():
        explanations = json.loads(Path(explanations_path).read_text(encoding="utf-8"))

    data = build_viewer_data(
        pred,
        gt,
        tags=tags,
        metrics=metrics,
        start_frame=start_frame,
        explanations=explanations,
        events=events,
        ball_gaps=ball_gaps,
        tracking_available=tracking_available,
        perception_available=perception_available,
        fusion_available=fusion_available,
    )
    data.update(build_user_truth_data(pred, load_user_truth(user_truth_path, pred.size)))
    return data


def tactical_video_path(data_dir: str | Path | None = None) -> Path:
    """Return the tactical MP4 next to the gated ATD files."""
    return asset_paths(data_dir)["atd_raw"].with_name(f"{GAME}_{SOURCE}.mp4")


# The tactical broadcast is 1920x1080.  The homography maps normalised pitch coordinates
# (x / length, y / width) to these pixels, so one projection works for every frame.
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _project(
    homography: Any,
    frame: int,
    x: float,
    y: float,
    pitch: tuple[float, float] = PITCH,
) -> list[float] | None:
    """Project a pitch position to video pixels with that frame's homography."""
    matrices = getattr(homography, "matrices", None)
    if matrices is None or frame < 0 or frame >= len(matrices):
        return None
    matrix = np.asarray(matrices[frame], dtype=np.float64)
    if matrix.shape != (3, 3):
        return None
    length, width = pitch
    xn, yn = float(x) / length, float(y) / width
    denominator = matrix[2, 0] * xn + matrix[2, 1] * yn + matrix[2, 2]
    if abs(denominator) < 1e-9:
        return None
    px = (matrix[0, 0] * xn + matrix[0, 1] * yn + matrix[0, 2]) / denominator
    py = (matrix[1, 0] * xn + matrix[1, 1] * yn + matrix[1, 2]) / denominator
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    return [round(float(px), 1), round(float(py), 1)]


def ball_gap_runs(track: Any, min_frames: int = 1) -> list[list[int]]:
    """Half-open runs where the tracker has no ball observation."""
    observed = np.asarray(getattr(track, "observed", ()), dtype=bool)
    if observed.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, ~observed, False]))
    return [
        [int(start), int(stop)]
        for start, stop in zip(edges[::2], edges[1::2])
        if int(stop) - int(start) >= int(min_frames)
    ]


class FrameStore:
    """Per-frame tracking, feature, and event lookup for the video overlay.

    The full-match arrays are already in memory for the model; keeping them here means one compact
    request per displayed frame instead of shipping 199k frames of tracking to the browser.
    """

    def __init__(
        self,
        tracking: Any,
        homography: Any,
        track: Any,
        events: Any,
        features: Any,
        *,
        trail_frames: int = 25,
        kalman: Any | None = None,
        perception: Any | None = None,
        fusion: Any | None = None,
    ) -> None:
        self.tracking = tracking
        self.homography = homography
        self.track = track
        self.kalman = kalman
        self.events = events
        self.features = features
        self.perception = perception
        self.fusion = fusion
        self.trail_frames = max(1, int(trail_frames))
        self.n_frames = int(tracking.n_frames)
        self._yolo_ball_pitch: np.ndarray | None = None
        self._yolo_ball_confidence: np.ndarray | None = None
        self._yolo_ball_selected: np.ndarray | None = None
        self._yolo_ball_status: np.ndarray | None = None
        self._yolo_ball_decisions: np.ndarray | None = None
        self._yolo_ball_track_id: np.ndarray | None = None
        self._yolo_track_count = 0
        if perception is not None and getattr(perception, "samples", 0) > 0:
            present = getattr(perception, "players_present", None)
            pitch_players = getattr(perception, "players_pitch", None)
            player_points = None
            if present is not None and pitch_players is not None:
                player_points = np.where(
                    np.asarray(present, dtype=bool)[..., None],
                    np.asarray(pitch_players, dtype=np.float64),
                    np.nan,
                )
            track = _run_ball_track(
                perception.yolo_ball_pitch,
                perception.yolo_ball_confidence,
                perception.yolo_ball_present,
                perception.frame_index,
                fps=float(getattr(perception, "fps", FPS)),
                low_confidence_floor=YOLO_VIEW_FLOOR,
                player_points=player_points,
                occlusion_radius_m=YOLO_VIEW_OCCLUSION_M,
            )
            self._yolo_ball_pitch = track.ball
            self._yolo_ball_confidence = track.confidence
            self._yolo_ball_status = track.status
            self._yolo_ball_selected = track.selected
            self._yolo_ball_decisions = track.decisions
            self._yolo_ball_track_id = track.track_id
            self._yolo_track_count = int(np.unique(track.track_id[track.track_id >= 0]).size)
        if getattr(features, "start", 0) != 0:
            raise ValueError("FrameStore needs full-match features starting at frame 0")

    def _feature(self, name: str, frame: int) -> Any:
        values = getattr(self.features, name, None)
        if values is None or len(values) != self.n_frames:
            return None
        return values[frame]

    def _features_payload(self, frame: int) -> dict[str, Any]:
        names = (
            "team_speed",
            "ball_speed",
            "edge",
            "last_edge",
            "gap",
            "near_distance",
            "near_team",
            "last_touch",
            "last_touch_age",
            "d_home",
            "d_away",
            "whistle_env",
            "whistle_score",
        )
        payload: dict[str, Any] = {}
        for name in names:
            value = self._feature(name, frame)
            if value is None:
                payload[name] = None
            elif isinstance(value, (bool, np.bool_)):
                payload[name] = bool(value)
            elif isinstance(value, (int, np.integer)):
                payload[name] = int(value)
            else:
                payload[name] = _finite(value)
        for name in ("out_of_play", "parked_edge", "stop_dead", "goal_dead", "tag_live"):
            value = self._feature(name, frame)
            payload[name] = bool(value) if value is not None else False
        return payload

    def _active_events(self, frame: int) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for goal in getattr(self.events, "goals", ()):
            if int(goal.frame) <= frame < int(goal.restart):
                active.append(
                    {
                        "type": "goal",
                        "frame": int(goal.frame),
                        "kind": str(goal.kind),
                        "confidence": float(goal.confidence),
                    }
                )
        for stop in getattr(self.events, "stoppages", ()):
            if int(stop.start) <= frame < int(stop.end):
                active.append(
                    {
                        "type": "stoppage",
                        "start": int(stop.start),
                        "end": int(stop.end),
                        "trigger": str(stop.trigger),
                        "confidence": float(stop.confidence),
                    }
                )
        return active

    def _outside_pitch(self, x: float, y: float) -> bool:
        length, width = self.tracking.pitch
        return x < -1.0 or x > length + 1.0 or y < -1.0 or y > width + 1.0

    def _perception_sample(self, frame: int, max_age: int | None = None) -> dict[str, Any] | None:
        if self.perception is None or getattr(self.perception, "samples", 0) == 0:
            return None
        sample = self.perception.nearest(int(frame))
        age = abs(int(sample.get("age_frames", 0)))
        limit = max(2, int(round(0.30 * FPS))) if max_age is None else int(max_age)
        if age > limit:
            return None
        sample["age_frames"] = age
        return sample

    def _video_layer_payload(self, frame: int) -> dict[str, Any]:
        sample = self._perception_sample(frame)
        if sample is None:
            return {
                "source": 2,
                "observed": False,
                "x": None,
                "y": None,
                "px": None,
                "py": None,
                "confidence": 0.0,
                "gap_s": None,
                "out_of_play": False,
                "age_frames": None,
            }
        ball = sample.get("ball", {})
        source = int(ball.get("source", 2))
        x = ball.get("x")
        y = ball.get("y")
        available = source != 2 and x is not None and y is not None
        gap = (abs(int(sample.get("age_frames", 0))) / FPS) if available and source != 0 else (0.0 if available else None)
        return {
            "source": source,
            "observed": bool(source == 0),
            "x": x if available else None,
            "y": y if available else None,
            "px": ball.get("px") if available else None,
            "py": ball.get("py") if available else None,
            "confidence": _finite(ball.get("confidence")) if available else 0.0,
            "gap_s": gap,
            "out_of_play": bool(available and self._outside_pitch(float(x), float(y))),
            "age_frames": int(sample.get("age_frames", 0)),
        }

    def _yolo_layer_payload(self, frame: int) -> dict[str, Any]:
        sample = self._perception_sample(frame)
        if sample is None or self._yolo_ball_pitch is None:
            return {
                "source": 2,
                "observed": False,
                "x": None,
                "y": None,
                "px": None,
                "py": None,
                "confidence": 0.0,
                "gap_s": None,
                "out_of_play": False,
                "recovered": False,
                "track": -1,
                "age_frames": None,
            }
        index = self.perception.nearest_index(int(frame))
        point = np.asarray(self._yolo_ball_pitch[index], dtype=np.float64)
        confidence = (
            float(self._yolo_ball_confidence[index])
            if self._yolo_ball_confidence is not None
            else 0.0
        )
        slot = int(self._yolo_ball_selected[index]) if self._yolo_ball_selected is not None else -1
        if (
            slot < 0
            or not np.isfinite(point).all()
            or confidence <= 0.01
        ):
            return {
                "source": 2,
                "observed": False,
                "x": None,
                "y": None,
                "px": None,
                "py": None,
                "confidence": 0.0,
                "gap_s": None,
                "out_of_play": False,
                "recovered": False,
                "track": -1,
                "age_frames": int(sample.get("age_frames", 0)),
            }
        status = int(self._yolo_ball_status[index]) if self._yolo_ball_status is not None else 0
        track = int(self._yolo_ball_track_id[index]) if self._yolo_ball_track_id is not None else -1
        pixel = np.asarray(self.perception.yolo_ball_pixel[index, slot], dtype=np.float64)
        source = 0 if confidence >= 0.25 else 1
        return {
            "source": source,
            "observed": bool(source == 0),
            "recovered": bool(status == 3),
            "track": track,
            "x": _finite(point[0]),
            "y": _finite(point[1]),
            "px": _finite(pixel[0]),
            "py": _finite(pixel[1]),
            "confidence": confidence,
            "gap_s": abs(int(sample.get("age_frames", 0))) / FPS if source != 0 else 0.0,
            "out_of_play": bool(
                np.isfinite(point).all()
                and self._outside_pitch(float(point[0]), float(point[1]))
            ),
            "age_frames": int(sample.get("age_frames", 0)),
        }

    def _yolo_candidates_payload(self, frame: int) -> list[dict[str, Any]]:
        """Every raw YOLO ball candidate for the frame, labelled with its gate decision.

        ``decision`` mirrors the ``BALL_CAND_*`` codes in :mod:`possession_lab.perception`:
        0 rejected-low, 1 accepted, 2 recovered (2nd stage), 3 rejected-unreachable,
        4 rejected-occluded-by-player, 5 unselected-but-plausible.  It is online physical/geometric
        evidence only and does not affect fusion or the submitted predictions.
        """
        sample = self._perception_sample(frame)
        if sample is None or self.perception is None or self._yolo_ball_decisions is None:
            return []
        index = self.perception.nearest_index(int(frame))
        decisions = self._yolo_ball_decisions[index]
        candidates: list[dict[str, Any]] = []
        for slot in range(int(self.perception.yolo_ball_present.shape[1])):
            if not bool(self.perception.yolo_ball_present[index, slot]):
                continue
            point = np.asarray(self.perception.yolo_ball_pitch[index, slot], dtype=np.float64)
            if not np.isfinite(point).all():
                continue
            pixel = np.asarray(self.perception.yolo_ball_pixel[index, slot], dtype=np.float64)
            candidates.append(
                {
                    "x": _finite(point[0]),
                    "y": _finite(point[1]),
                    "px": _finite(pixel[0]),
                    "py": _finite(pixel[1]),
                    "confidence": round(float(self.perception.yolo_ball_confidence[index, slot]), 3),
                    "decision": int(decisions[slot]),
                }
            )
        return candidates

    def _fusion_layer_payload(self, frame: int) -> dict[str, Any]:
        if self.fusion is None:
            return {
                "source": 2,
                "observed": False,
                "x": None,
                "y": None,
                "px": None,
                "py": None,
                "confidence": 0.0,
                "gap_s": None,
                "out_of_play": False,
            }
        point = np.asarray(self.fusion.consensus_ball[frame], dtype=np.float64)
        confidence = float(self.fusion.consensus_confidence[frame])
        available = np.isfinite(point).all() and confidence > 0
        x, y = (float(point[0]), float(point[1])) if available else (float("nan"), float("nan"))
        pixel = _project(self.homography, frame, x, y, self.tracking.pitch) if available else None
        return {
            "source": 0 if available else 2,
            "observed": bool(available),
            "x": _finite(x) if available else None,
            "y": _finite(y) if available else None,
            "px": pixel[0] if pixel is not None else None,
            "py": pixel[1] if pixel is not None else None,
            "confidence": confidence if available else 0.0,
            "gap_s": 0.0 if available else None,
            "out_of_play": bool(available and self._outside_pitch(x, y)),
        }

    def _layer_payload(self, frame: int, layer: str) -> dict[str, Any]:
        """Describe one ball layer at one frame without changing the model tracker."""
        if layer == "video":
            return self._video_layer_payload(frame)
        if layer == "yolo":
            return self._yolo_layer_payload(frame)
        if layer == "fusion":
            return self._fusion_layer_payload(frame)
        raw_present = bool(self.tracking.ball_present[frame])
        raw = self.tracking.ball[frame]
        if layer == "raw":
            available = raw_present and np.isfinite(raw[0]) and np.isfinite(raw[1])
            x = float(raw[0]) if available else np.nan
            y = float(raw[1]) if available else np.nan
            source = 0 if available else 2
            confidence = 1.0 if available else 0.0
            gap = 0.0 if available else None
            observed = available
            outside = self._outside_pitch(x, y) if available else False
        else:
            track = self.kalman if layer == "kalman" else self.track
            if track is None:
                available = False
                x = y = float("nan")
                source = 2
                confidence = 0.0
                gap = None
                observed = False
                outside = False
            else:
                source = int(track.source[frame])
                point = track.estimate[frame]
                available = source != 2 and np.isfinite(point[0]) and np.isfinite(point[1])
                x = float(point[0]) if available else np.nan
                y = float(point[1]) if available else np.nan
                confidence = _finite(track.confidence[frame])
                gap = _finite(track.gap_s[frame])
                observed = bool(getattr(track, "observed", np.zeros(self.n_frames, dtype=bool))[frame])
                outside = bool(track.out_of_play[frame]) if available else False

        pixel = _project(self.homography, frame, x, y, self.tracking.pitch) if available else None
        return {
            "source": int(source),
            "observed": bool(observed),
            "x": _finite(x) if available else None,
            "y": _finite(y) if available else None,
            "px": pixel[0] if pixel is not None else None,
            "py": pixel[1] if pixel is not None else None,
            "confidence": confidence if confidence is not None else 0.0,
            "gap_s": gap,
            "out_of_play": bool(outside),
        }

    def _trail_series(
        self, frame: int, layer: str
    ) -> list[tuple[int, float, float, int]]:
        """Trail points as ``(frame, x, y, source)``; the frame is needed for projection.

        The broadcast camera moves, so each historical pitch point must use the homography of its
        own frame on the video overlay, not the homography of the current frame.
        """
        if layer == "video":
            points: list[tuple[int, float, float, int]] = []
            seen: set[int] = set()
            start = max(0, frame - self.trail_frames + 1)
            for k in range(start, frame + 1):
                sample = self._perception_sample(k)
                if sample is None:
                    continue
                sample_frame = int(sample.get("sample_frame", k))
                if sample_frame in seen:
                    continue
                seen.add(sample_frame)
                ball = sample.get("ball", {})
                source = int(ball.get("source", 2))
                if source == 2 or ball.get("x") is None or ball.get("y") is None:
                    continue
                points.append((k, float(ball["x"]), float(ball["y"]), source))
            return points
        if layer == "yolo":
            if self._yolo_ball_pitch is None or self.perception is None:
                return []
            yolo_points: list[tuple[int, float, float, int]] = []
            yolo_seen: set[int] = set()
            start = max(0, frame - self.trail_frames + 1)
            for k in range(start, frame + 1):
                idx = self.perception.nearest_index(int(k))
                if idx in yolo_seen:
                    continue
                yolo_seen.add(idx)
                slot = int(self._yolo_ball_selected[idx]) if self._yolo_ball_selected is not None else -1
                point = np.asarray(self._yolo_ball_pitch[idx], dtype=np.float64)
                confidence = (
                    float(self._yolo_ball_confidence[idx])
                    if self._yolo_ball_confidence is not None
                    else 0.0
                )
                if slot < 0 or not np.isfinite(point).all() or confidence <= 0.01:
                    yolo_points = []  # a real gap breaks the trajectory, never bridges it
                    continue
                yolo_points.append((k, float(point[0]), float(point[1]), 0 if confidence >= 0.25 else 1))
            return yolo_points
        if layer == "fusion":
            if self.fusion is None:
                return []
            points = []
            start = max(0, frame - self.trail_frames + 1)
            for k in range(start, frame + 1):
                point = self.fusion.consensus_ball[k]
                if not (np.isfinite(point[0]) and np.isfinite(point[1])):
                    continue
                points.append((k, float(point[0]), float(point[1]), 0))
            return points

        if layer == "raw":
            points: list[tuple[int, float, float, int]] = []
            start = max(0, frame - self.trail_frames + 1)
            for k in range(start, frame + 1):
                if not bool(self.tracking.ball_present[k]):
                    points = []
                    continue
                point = self.tracking.ball[k]
                if not (np.isfinite(point[0]) and np.isfinite(point[1])):
                    points = []
                    continue
                points.append((k, float(point[0]), float(point[1]), 0))
            return points

        track = self.kalman if layer == "kalman" else self.track
        if track is None:
            return []
        points = []
        start = max(0, frame - self.trail_frames + 1)
        for k in range(start, frame + 1):
            source_k = int(track.source[k])
            if source_k == 2:
                points = []
                continue
            point = track.estimate[k]
            if not (np.isfinite(point[0]) and np.isfinite(point[1])):
                continue
            points.append((k, float(point[0]), float(point[1]), source_k))
        return points

    def _trail_payload(self, frame: int, layer: str) -> dict[str, list[list[float | int]]]:
        series = self._trail_series(frame, layer)
        pitch: list[list[float | int]] = [
            [round(x, 2), round(y, 2), source] for _, x, y, source in series
        ]
        video: list[list[float | int]] = []
        for k, x, y, source in series:
            pixel = _project(self.homography, k, x, y, self.tracking.pitch)
            if pixel is not None:
                video.append([pixel[0], pixel[1], source])
        return {"pitch": pitch, "video": video}

    def frame_payload(self, frame: int) -> dict[str, Any]:
        frame = max(0, min(int(frame), self.n_frames - 1))
        raw_present = bool(self.tracking.ball_present[frame])
        source = int(self.track.source[frame])
        estimate = self.track.estimate[frame]
        raw = self.tracking.ball[frame]
        ex, ey = float(estimate[0]), float(estimate[1])
        has_estimate = np.isfinite(ex) and np.isfinite(ey)
        ball_pixel = _project(self.homography, frame, ex, ey, self.tracking.pitch) if has_estimate else None

        trails = {
            layer: self._trail_payload(frame, layer)
            for layer in ("raw", "kalman", "combined", "video", "yolo", "fusion")
        }
        trail = trails["combined"]["video"]
        trail_pitch = trails["combined"]["pitch"]
        layers = {
            layer: self._layer_payload(frame, layer)
            for layer in ("raw", "kalman", "combined", "video", "yolo", "fusion")
        }

        players: list[dict[str, Any]] = []
        players_xy = self.tracking.players[frame]
        speeds = self.tracking.speeds[frame]
        present = self.tracking.player_present[frame]
        teams = self.tracking.teams
        for index in range(players_xy.shape[0]):
            if not bool(present[index]):
                continue
            x, y = float(players_xy[index, 0]), float(players_xy[index, 1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            pixel = _project(self.homography, frame, x, y, self.tracking.pitch)
            players.append(
                {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "px": pixel[0] if pixel is not None else None,
                    "py": pixel[1] if pixel is not None else None,
                    "team": int(teams[index]) if index < len(teams) else -1,
                    "speed": _finite(speeds[index]) if np.isfinite(speeds[index]) else None,
                }
            )

        video_players: list[dict[str, Any]] = []
        sample = self._perception_sample(frame)
        if sample is not None:
            for player in sample.get("players", []):
                if player.get("x") is None or player.get("y") is None:
                    continue
                video_players.append(
                    {
                        "x": player.get("x"),
                        "y": player.get("y"),
                        "px": player.get("px"),
                        "py": player.get("py"),
                        "team": int(player.get("team", -1)),
                        "team_confidence": _finite(player.get("team_confidence")),
                        "confidence": _finite(player.get("confidence")),
                        "track": int(player.get("track", -1)),
                        "box": player.get("box"),
                    }
                )

        fusion_payload: dict[str, Any]
        if self.fusion is None:
            fusion_payload = {"available": False, "reason": "fusion cache not loaded"}
        else:
            fusion_payload = self.fusion.sample(frame)
            fusion_payload["available"] = True
            fusion_payload["baseline_label"] = (
                CODE_TO_LABEL.get(int(fusion_payload["baseline_code"]), "X")
                if fusion_payload.get("baseline_code") is not None
                else None
            )
            fusion_payload["combine_mode"] = self.fusion.meta.get("combine_mode", "log")

        return {
            "available": True,
            "frame": frame,
            "fps": FPS,
            "time_s": round(frame / FPS, 3),
            "video_width": VIDEO_WIDTH,
            "video_height": VIDEO_HEIGHT,
            "pitch": {
                "length": float(self.tracking.pitch[0]),
                "width": float(self.tracking.pitch[1]),
            },
            "ball": {
                "present": raw_present,
                "source": source,
                "x": _finite(raw[0]) if raw_present else None,
                "y": _finite(raw[1]) if raw_present else None,
                "ex": _finite(ex) if has_estimate else None,
                "ey": _finite(ey) if has_estimate else None,
                "px": ball_pixel[0] if ball_pixel is not None else None,
                "py": ball_pixel[1] if ball_pixel is not None else None,
                "confidence": _finite(self.track.confidence[frame]),
                "gap_s": _finite(self.track.gap_s[frame]),
                "out_of_play": bool(
                    self.track.out_of_play[frame] or self._feature("out_of_play", frame)
                ),
            },
            "layers": layers,
            "trails": trails,
            "trail": trail,
            "trail_pitch": trail_pitch,
            "players": players,
            "video_players": video_players,
            "yolo_candidates": self._yolo_candidates_payload(frame),
            "yolo_track": {
                "count": int(self._yolo_track_count),
                "floor": YOLO_VIEW_FLOOR,
                "occlusion_m": YOLO_VIEW_OCCLUSION_M,
                "id": int(layers["yolo"].get("track", -1)) if layers.get("yolo") else -1,
            },
            "perception_available": bool(self.perception is not None),
            "fusion_available": bool(self.fusion is not None),
            "fusion": fusion_payload,
            "features": self._features_payload(frame),
            "active_events": self._active_events(frame),
        }


class MultiByteRange(ValueError):
    """A syntactically valid multi-part ``Range`` header, such as ``bytes=0-1,3-4``.

    Serving ``multipart/byteranges`` is optional, and RFC 9110 wants a server that will not serve it
    to ignore the header and return the whole representation with 200.  Answering 416 instead breaks
    the clients that probe with two ranges, because 416 tells them the file cannot be ranged at all.
    """


def parse_byte_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a single-range ``Range`` header.

    Returns ``None`` when there is no range, or ``(start, end)`` with inclusive bounds.  Raises
    :class:`MultiByteRange` for a well-formed multi-range header and ``ValueError`` for a malformed
    or unsatisfiable one.
    """
    if not header:
        return None
    candidate = header.strip()
    if candidate.startswith("bytes=") and "," in candidate:
        # The unit prefixes the whole list, so the individual items carry no ``bytes=`` of their own.
        items = [item.strip() for item in candidate[len("bytes=") :].split(",")]
        if items and all(RANGE_ITEM_RE.fullmatch(item) and item != "-" for item in items):
            raise MultiByteRange("multiple ranges")
        raise ValueError("unsupported range")
    match = RANGE_RE.fullmatch(candidate)
    if not match:
        raise ValueError("unsupported range")
    first, last = match.groups()
    if first == "" and last == "":
        raise ValueError("empty range")
    if size <= 0:
        raise ValueError("empty file")

    if first == "":
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("empty suffix")
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(first)
        end = size - 1 if last == "" else int(last)
        if start >= size:
            raise ValueError("range starts past end")
        end = min(end, size - 1)
        if end < start:
            raise ValueError("range end before start")
    return start, end


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Small whitelist-only HTTP handler with MP4 byte-range support."""

    protocol_version = "HTTP/1.1"
    payload: ClassVar[bytes] = b"{}"
    video_path: ClassVar[Path | None] = None
    assets: ClassVar[dict[str, tuple[bytes, str]]] = {}
    pred: ClassVar[np.ndarray | None] = None
    user_truth: ClassVar[np.ndarray | None] = None
    user_truth_path: ClassVar[Path | None] = None
    frame_store: ClassVar[FrameStore | None] = None
    ut_lock: ClassVar[threading.Lock] = threading.Lock()
    verbose: ClassVar[bool] = False

    _body_read: int = 0

    def handle_one_request(self) -> None:
        # Counted per request: ``Content-Length`` is the total the client declared, so draining has
        # to subtract whatever this handler already consumed.
        self._body_read = 0
        super().handle_one_request()

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.verbose:
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._dispatch(send_body=True)

    def do_HEAD(self) -> None:
        self._dispatch(send_body=False)

    def _read_body(self, length: int) -> bytes:
        data = self.rfile.read(length)
        self._body_read += len(data)
        return data

    def _drain_request_body(self) -> None:
        """Read and discard a declared request body that will not be used.

        A keep-alive connection parses the next request straight out of ``rfile``, so answering an
        error while body bytes are still queued makes that next request start in the middle of this
        one's body.  Two cases cannot be drained and close the socket instead: a chunked body, whose
        length is not known up front, and more bytes than the endpoint would ever accept.
        """
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            return
        try:
            declared = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self.close_connection = True
            return
        remaining = declared - self._body_read
        if remaining <= 0:
            return
        if remaining > MAX_POST_BYTES:
            self.close_connection = True
            return
        while remaining > 0:
            chunk = self.rfile.read(min(65_536, remaining))
            if not chunk:
                self.close_connection = True
                return
            remaining -= len(chunk)

    def _reject(self, status: HTTPStatus, message: str) -> None:
        """Send an error response and leave the connection safe for the next request."""
        self._drain_request_body()
        self._send_error(status, message)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/user-truth":
            self._reject(HTTPStatus.NOT_FOUND, "not found")
            return
        if self.pred is None or self.user_truth is None or self.user_truth_path is None:
            self._reject(HTTPStatus.NOT_FOUND, "user truth not configured")
            return
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            self._reject(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_POST_BYTES:
                raise ValueError
            body = json.loads(self._read_body(length))
            if not isinstance(body, dict):
                raise ValueError
            n_frames = int(self.pred.size)
            clear = bool(body.get("clear"))
            erase_nearest = boundary = target = start = stop = code = None
            if clear:
                pass
            elif "eraseNearest" in body:
                erase_nearest = int(body["eraseNearest"])
                if not (0 <= erase_nearest < n_frames):
                    raise ValueError
            elif "boundary" in body:
                boundary, target = int(body["boundary"]), int(body["to"])
                if not (0 <= boundary <= n_frames and 0 <= target <= n_frames):
                    raise ValueError
            else:
                start, stop = int(body["start"]), int(body["stop"])
                code = LABEL_TO_CODE[str(body["label"]).strip().upper()]
                if not (0 <= start < stop <= n_frames):
                    raise ValueError
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._reject(HTTPStatus.BAD_REQUEST, "invalid user-truth edit")
            return

        try:
            with self.ut_lock:
                backup = self.user_truth.copy()
                try:
                    if clear:
                        self.user_truth[:] = X
                    elif erase_nearest is not None:
                        erase_nearest_user_truth(self.user_truth, erase_nearest)
                    elif boundary is not None:
                        move_user_truth_boundary(self.user_truth, boundary, target)
                    else:
                        self.user_truth[start:stop] = code
                    save_user_truth(self.user_truth_path, self.user_truth)
                except (OSError, ValueError):
                    self.user_truth[:] = backup
                    raise
                ut_payload = build_user_truth_data(self.pred, self.user_truth)
                merged = json.loads(self.payload)
                merged.update(ut_payload)
                merged_payload = json.dumps(
                    merged, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.__class__.payload = merged_payload
        except ValueError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid user-truth edit")
            return
        except OSError:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "could not save user truth")
            return

        self._send_bytes(
            json.dumps(ut_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            send_body=True,
            cache="no-store",
        )

    def _dispatch(self, *, send_body: bool) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            entry = self.assets.get("/")
            if entry is None:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "viewer index not configured")
            else:
                self._send_bytes(entry[0], entry[1], send_body=send_body)
            return
        if path == "/api/viewer-data":
            self._send_bytes(
                self.payload,
                "application/json; charset=utf-8",
                send_body=send_body,
                cache="no-store",
            )
            return
        if path == "/api/frame":
            self._serve_frame(send_body=send_body)
            return
        if path == "/video":
            self._serve_video(send_body=send_body)
            return
        entry = self.assets.get(path)
        if entry is not None:
            self._send_bytes(entry[0], entry[1], send_body=send_body)
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found", send_body=send_body)

    def _serve_frame(self, *, send_body: bool) -> None:
        store = self.frame_store
        if store is None:
            self._send_bytes(
                json.dumps({"available": False, "reason": "tracking not loaded"}).encode("utf-8"),
                "application/json; charset=utf-8",
                send_body=send_body,
                cache="no-store",
                status=HTTPStatus.NOT_FOUND,
            )
            return
        query = parse_qs(urlsplit(self.path).query)
        try:
            frame = int(query.get("frame", ["0"])[0])
        except (TypeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid frame", send_body=send_body)
            return
        if frame < 0:
            self._send_error(HTTPStatus.BAD_REQUEST, "frame must not be negative", send_body=send_body)
            return
        try:
            payload = store.frame_payload(frame)
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            # A cache that does not cover the whole match (a public-window perception or fusion
            # .npz next to full-length labels) raises here.  Report it as a 500 the page can show:
            # letting the exception escape kills the handler thread and the browser only sees a
            # dropped connection.
            if self.verbose:
                print(f"frame {frame} failed: {type(exc).__name__}: {exc}", flush=True)
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "frame unavailable: stale or partial perception/fusion cache",
                send_body=send_body,
            )
            return
        self._send_bytes(
            body.encode("utf-8"),
            "application/json; charset=utf-8",
            send_body=send_body,
            cache="no-store",
        )

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        *,
        send_body: bool,
        cache: str = "no-cache",
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str, *, send_body: bool = True) -> None:
        body = (message + "\n").encode("utf-8")
        self._send_bytes(
            body,
            "text/plain; charset=utf-8",
            send_body=send_body,
            cache="no-store",
            status=status,
        )

    def _serve_video(self, *, send_body: bool) -> None:
        path = self.video_path
        if path is None or not Path(path).is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "video not found", send_body=send_body)
            return
        size = Path(path).stat().st_size
        range_header = self.headers.get("Range", "")
        if self.verbose:
            print(f"video request: range={range_header!r} size={size:,}", flush=True)
        try:
            parsed = parse_byte_range(range_header, size)
        except MultiByteRange:
            # We do not produce multipart/byteranges bodies.  RFC 9110 says the right answer is then
            # to ignore the header and send the whole file, not 416: 416 tells the client the file
            # cannot be ranged at all, and the two-range probe never comes back.
            parsed = None
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if size == 0:
            # Nothing to serve.  The previous code turned a no-range request into a 0..0 slice, so
            # it advertised ``Content-Length: 1`` and then wrote no body, leaving a keep-alive
            # client waiting for an octet that never arrives.  A ranged request on an empty file is
            # already answered with 416 ``bytes */0`` by the parser above.
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", "0")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return

        if parsed is None:
            start, end = 0, max(0, size - 1)
            status = HTTPStatus.OK
        else:
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT

        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        if parsed is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        if not send_body:
            return
        try:
            with Path(path).open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


class ViewerHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The browser aborts the initial open-ended video request after it reads the header/moov
        # range.  That is normal for a 5 GB MP4 and should not dump a traceback into the terminal.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def _load_assets(assets_dir: Path) -> dict[str, tuple[bytes, str]]:
    assets: dict[str, tuple[bytes, str]] = {}
    for name, mime in MIME_TYPES.items():
        path = assets_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"viewer asset not found: {path}")
        route = "/" if name == "index.html" else f"/static/{name}"
        assets[route] = (path.read_bytes(), mime)
    return assets


def make_server(
    host: str,
    port: int,
    *,
    payload: bytes,
    video_path: str | Path,
    assets_dir: str | Path,
    pred: np.ndarray | None = None,
    user_truth: np.ndarray | None = None,
    user_truth_path: str | Path | None = None,
    frame_store: FrameStore | None = None,
    verbose: bool = False,
) -> ViewerHTTPServer:
    """Create the local reviewer server.  ``port=0`` chooses an ephemeral port."""
    assets = _load_assets(Path(assets_dir))
    attributes: dict[str, Any] = {
        "payload": payload,
        "video_path": Path(video_path),
        "assets": assets,
        # Set on the generated subclass, not on ViewerRequestHandler: a second server (a test, or a
        # second run in the same process) would otherwise inherit the first one's verbosity.
        "verbose": bool(verbose),
    }
    if pred is not None and user_truth is not None and user_truth_path is not None:
        attributes.update(
            {
                "pred": np.array(pred, dtype=np.int8, copy=True),
                "user_truth": np.array(user_truth, dtype=np.int8, copy=True),
                "user_truth_path": Path(user_truth_path),
            }
        )
    if frame_store is not None:
        attributes["frame_store"] = frame_store
    handler = type("BoundViewerRequestHandler", (ViewerRequestHandler,), attributes)
    return ViewerHTTPServer((host, port), handler)


def build_frame_store(
    video_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    perception_path: str | Path | None = None,
    fusion_path: str | Path | None = None,
) -> tuple[FrameStore, Any, list[list[int]]]:
    """Load the gated tracking once and prepare the per-frame explorer products.

    Audio is used only when its cache already exists.  The viewer must stay fast to open; a first
    run without ``whistle.npz`` still gets geometry/out-of-play events, and ``predict``/``all``
    builds the cache.
    """
    from .audio import load_or_extract
    from .balltrack import track_ball
    from .events import build_events
    from .features import extract_features
    from .fusion import load_fusion
    from .kalman import kalman_ball_track
    from .perception import load_perception
    from .pipeline import load_lab

    lab = load_lab(data_dir, load_gt=False)
    track = track_ball(lab.tracking)
    kalman = kalman_ball_track(lab.tracking)
    whistle = None
    if Path(WHISTLE_CACHE).is_file():
        whistle = load_or_extract(video_path, lab.tracking.n_frames)
    features = extract_features(
        lab.tracking, lab.tags, ball_track=track, whistle=whistle
    )
    events = build_events(
        lab.tracking, lab.tags, track=track, whistle=whistle, features=features
    )
    perception = load_perception(perception_path) if perception_path else None
    fusion = load_fusion(fusion_path) if fusion_path else None
    store = FrameStore(
        lab.tracking,
        lab.homography,
        track,
        events,
        features,
        kalman=kalman,
        perception=perception,
        fusion=fusion,
    )
    return store, events, ball_gap_runs(track, min_frames=max(1, int(0.4 * FPS)))


def _open_viewer_browser(url: str) -> None:
    """Open the local viewer without leaking browser child-process stderr into the terminal.

    On Linux, ``xdg-open`` hands the URL to an existing Chromium/Brave session.  That session can
    keep the inherited stderr open and print zygote/broken-pipe noise after the server has
    started, so the launcher's streams are detached.  Fall back to :mod:`webbrowser` only when
    ``xdg-open`` is unavailable or cannot be started.
    """
    if sys.platform.startswith("linux"):
        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            return
        except OSError:
            pass
    try:
        webbrowser.open(url)
    except Exception as exc:  # pragma: no cover - platform/browser failure path
        print(
            f"viewer: no se pudo abrir el navegador; abre {url} manualmente ({exc})",
            file=sys.stderr,
            flush=True,
        )


def serve_viewer(
    predictions_path: str | Path,
    *,
    video_path: str | Path | None = None,
    gt_path: str | Path | None = PUBLIC_GT_PATH,
    tags_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    user_truth_path: str | Path | None = USER_TRUTH_PATH,
    explanations_path: str | Path | None = EXPLANATIONS_PATH,
    data_dir: str | Path | None = None,
    perception_path: str | Path | None = None,
    fusion_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    start_frame: int | None = None,
    verbose: bool = False,
) -> int:
    """Build the payload, serve the reviewer, and block until Ctrl-C."""
    video = Path(video_path) if video_path is not None else tactical_video_path(data_dir)
    if not video.is_file():
        raise FileNotFoundError(
            f"tactical video not found: {video}\n"
            "Unpack the gated bundle under data/ or pass --video."
        )
    predictions = Path(predictions_path)
    if not predictions.is_file():
        raise FileNotFoundError(
            f"predictions not found: {predictions}\n"
            "Run `python -m possession_lab all` first or pass --predictions."
        )

    frame_store = None
    events = None
    gaps: list[list[int]] = []
    try:
        frame_store, events, gaps = build_frame_store(
            video,
            data_dir=data_dir,
            perception_path=perception_path,
            fusion_path=fusion_path,
        )
        print(
            f"tracking: {len(frame_store.tracking.ball):,} frames | "
            f"{len(getattr(events, 'goals', ()))} goles | "
            f"{len(getattr(events, 'stoppages', ()))} paradas | "
            f"{len(gaps):,} huecos de balón | "
            f"percepción {'sí' if frame_store.perception is not None else 'no'} | "
            f"fusión {'sí' if frame_store.fusion is not None else 'no'}",
            flush=True,
        )
    except Exception as exc:
        print(f"tracking: no disponible ({exc})", flush=True)

    data = load_viewer_data(
        predictions,
        gt_path=gt_path,
        tags_path=tags_path,
        metrics_path=metrics_path,
        user_truth_path=user_truth_path,
        explanations_path=explanations_path,
        start_frame=start_frame,
        events=events,
        ball_gaps=gaps,
        tracking_available=frame_store is not None,
        perception_available=bool(frame_store is not None and frame_store.perception is not None),
        fusion_available=bool(frame_store is not None and frame_store.fusion is not None),
    )
    pred = load_submission(predictions)
    user_truth = load_user_truth(user_truth_path, pred.size)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assets_dir = Path(__file__).resolve().parent / "viewer_assets"
    server = make_server(
        host,
        port,
        payload=payload,
        video_path=video,
        assets_dir=assets_dir,
        pred=pred,
        user_truth=user_truth,
        user_truth_path=user_truth_path,
        frame_store=frame_store,
        verbose=verbose,
    )
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"viewer: {url}", flush=True)
    print(
        f"data: {data['n_frames']:,} frames | {data['duration_s']:.2f}s | "
        f"{len(data['pred_spells'])} predicted spells | "
        f"{len(data['switches'])} annotated switches | "
        f"GT {'yes' if data['gt_available'] else 'no'} | "
        f"UT {data['ut_labeled_frames']:,} frames",
        flush=True,
    )
    print(
        f"video: {video} ({video.stat().st_size / (1024**3):.2f} GiB, byte-range streaming)",
        flush=True,
    )
    print("Ctrl-C to stop.", flush=True)
    if open_browser:
        opener = threading.Timer(0.5, _open_viewer_browser, args=(url,))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nviewer stopped.")
    finally:
        server.server_close()
    return 0
