"""Independent video perception: YOLO players, an own ball tracker, and a compact cache.

This module is deliberately separate from the ATD model.  The ATD is still one source, but the
video pipeline does not need it to *see* players or the ball:

* a COCO YOLO model detects people and the sports ball;
* player foot points are projected to pitch metres with the per-frame homography;
* teams are inferred from the nearest ATD slot when available; jersey colour is not used, so no
  actor identity is claimed;
* the ball is tracked in pitch space from YOLO ball detections plus a small white-blob detector;
* every detection is cached in a compact ``.npz`` so the viewer and the fusion layer never have to
  decode the 5 GB MP4 again.

Heavy imports (``cv2``, ``torch``, ``ultralytics``) are lazy.  Importing this module for a unit
test or for the old HMM path does not require the perception environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np

from .config import FPS, PITCH

# COCO class ids used by the stock YOLO weights.
PERSON_CLASS = 0
BALL_CLASS = 32

# A player farther than this from the pitch is a spectator/staff detection, not a player.
PITCH_MARGIN = 1.5
# A ball farther than this from the pitch is a ball boy/advert/crowd false positive.
BALL_MARGIN = 1.2

# Cache layout.  The first dimension is a sampled frame, not necessarily every video frame.
MAX_PLAYERS = 40
MAX_BALL_DETECTIONS = 6

PLAYER_ARRAY_NAMES = (
    "players_box",
    "players_pixel",
    "players_pitch",
    "players_team",
    "players_confidence",
    "players_present",
    "players_track",
    "players_team_confidence",
)
BALL_ARRAY_NAMES = (
    "ball_pixel",
    "ball_pitch",
    "ball_confidence",
    "ball_source",
    "yolo_ball_pixel",
    "yolo_ball_pitch",
    "yolo_ball_confidence",
    "yolo_ball_present",
)


@dataclass(frozen=True)
class VideoPerception:
    """Compact, sampled video detections.

    ``frame_index`` contains the video frame of each sample.  The arrays describe that sample;
    consumers should use :func:`sample_for_frame` or :meth:`VideoPerception.nearest` to map a
    full-resolution frame to the nearest perception sample.
    """

    frame_index: np.ndarray  # (M,)
    players_box: np.ndarray  # (M, K, 4) x1,y1,x2,y2 in video pixels
    players_pixel: np.ndarray  # (M, K, 2) foot point in video pixels
    players_pitch: np.ndarray  # (M, K, 2) foot point in pitch metres
    players_team: np.ndarray  # (M, K) 0/1/-1
    players_confidence: np.ndarray  # (M, K)
    players_present: np.ndarray  # (M, K) bool
    players_track: np.ndarray  # (M, K) integer track id, -1 when absent
    players_team_confidence: np.ndarray  # (M, K)
    ball_pixel: np.ndarray  # (M, 2)
    ball_pitch: np.ndarray  # (M, 2)
    ball_confidence: np.ndarray  # (M,)
    ball_source: np.ndarray  # (M,) 0 detected, 1 predicted, 2 lost, 3 ATD anchor
    yolo_ball_pixel: np.ndarray  # (M, B, 2)
    yolo_ball_pitch: np.ndarray  # (M, B, 2)
    yolo_ball_confidence: np.ndarray  # (M, B)
    yolo_ball_present: np.ndarray  # (M, B) bool
    n_frames: int
    fps: int
    step: int
    start: int
    stop: int
    model: str
    device: str
    video_width: int
    video_height: int
    pitch: tuple[float, float] = PITCH
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.frame_index.size)

    @property
    def samples(self) -> int:
        return int(self.frame_index.size)

    def nearest_index(self, frame: int) -> int:
        """Index of the nearest sampled frame (never extrapolates beyond the cache)."""
        if self.samples == 0:
            raise ValueError("empty perception cache")
        frame = int(frame)
        position = int(np.searchsorted(self.frame_index, frame, side="left"))
        if position <= 0:
            return 0
        if position >= self.samples:
            return self.samples - 1
        before = self.frame_index[position - 1]
        after = self.frame_index[position]
        return position - 1 if frame - before <= after - frame else position

    def nearest(self, frame: int) -> dict[str, Any]:
        """Return one sample as a JSON-friendly dict for the viewer."""
        index = self.nearest_index(frame)
        sample_frame = int(self.frame_index[index])
        result: dict[str, Any] = {
            "sample_frame": sample_frame,
            "age_frames": int(frame) - sample_frame,
            "age_s": (int(frame) - sample_frame) / max(self.fps, 1),
        }
        ball = self.ball_pitch[index]
        result["ball"] = {
            "source": int(self.ball_source[index]),
            "x": _finite(ball[0]),
            "y": _finite(ball[1]),
            "px": _finite(self.ball_pixel[index, 0]),
            "py": _finite(self.ball_pixel[index, 1]),
            "confidence": _finite(self.ball_confidence[index]),
        }
        result["yolo_ball"] = [
            {
                "x": _finite(self.yolo_ball_pitch[index, slot, 0]),
                "y": _finite(self.yolo_ball_pitch[index, slot, 1]),
                "px": _finite(self.yolo_ball_pixel[index, slot, 0]),
                "py": _finite(self.yolo_ball_pixel[index, slot, 1]),
                "confidence": _finite(self.yolo_ball_confidence[index, slot]),
            }
            for slot in range(MAX_BALL_DETECTIONS)
            if bool(self.yolo_ball_present[index, slot])
        ]
        players: list[dict[str, Any]] = []
        for slot in range(MAX_PLAYERS):
            if not bool(self.players_present[index, slot]):
                continue
            point = self.players_pitch[index, slot]
            pixel = self.players_pixel[index, slot]
            box = self.players_box[index, slot]
            players.append(
                {
                    "x": _finite(point[0]),
                    "y": _finite(point[1]),
                    "px": _finite(pixel[0]),
                    "py": _finite(pixel[1]),
                    "box": [_finite(value) for value in box],
                    "team": int(self.players_team[index, slot]),
                    "team_confidence": _finite(self.players_team_confidence[index, slot]),
                    "confidence": _finite(self.players_confidence[index, slot]),
                    "track": int(self.players_track[index, slot]),
                }
            )
        result["players"] = players
        return result

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "frame_index": np.asarray(self.frame_index, dtype=np.int32),
            "players_box": np.asarray(self.players_box, dtype=np.float32),
            "players_pixel": np.asarray(self.players_pixel, dtype=np.float32),
            "players_pitch": np.asarray(self.players_pitch, dtype=np.float32),
            "players_team": np.asarray(self.players_team, dtype=np.int8),
            "players_confidence": np.asarray(self.players_confidence, dtype=np.float32),
            "players_present": np.asarray(self.players_present, dtype=bool),
            "players_track": np.asarray(self.players_track, dtype=np.int16),
            "players_team_confidence": np.asarray(self.players_team_confidence, dtype=np.float32),
            "ball_pixel": np.asarray(self.ball_pixel, dtype=np.float32),
            "ball_pitch": np.asarray(self.ball_pitch, dtype=np.float32),
            "ball_confidence": np.asarray(self.ball_confidence, dtype=np.float32),
            "ball_source": np.asarray(self.ball_source, dtype=np.int8),
            "yolo_ball_pixel": np.asarray(self.yolo_ball_pixel, dtype=np.float32),
            "yolo_ball_pitch": np.asarray(self.yolo_ball_pitch, dtype=np.float32),
            "yolo_ball_confidence": np.asarray(self.yolo_ball_confidence, dtype=np.float32),
            "yolo_ball_present": np.asarray(self.yolo_ball_present, dtype=bool),
        }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def smooth_pitch(matrix: np.ndarray, pitch: tuple[float, float] = PITCH) -> np.ndarray:
    """Invert one pitch->pixel homography for vectorised pixel->pitch projection."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {matrix.shape}")
    inverse = np.linalg.inv(matrix)
    length, width = float(pitch[0]), float(pitch[1])
    # The inverse works in normalised pitch coordinates.  Scale the first two rows so the caller
    # can multiply by ``[px, py, 1]`` and read metres directly.  The third row is unchanged.
    inverse = inverse.copy()
    inverse[0] *= length
    inverse[1] *= width
    return inverse


def pixel_to_pitch(
    matrix: np.ndarray,
    pixels: np.ndarray,
    pitch: tuple[float, float] = PITCH,
) -> np.ndarray:
    """Project pixel points (or boxes' foot points) to pitch metres.

    ``pixels`` is ``(..., 2)``; the return has the same shape.  Invalid/denominator-zero points
    become ``NaN`` rather than silently wrapping around.
    """
    array = np.asarray(pixels, dtype=np.float64)
    if array.shape[-1] != 2:
        raise ValueError("pixels must have a final dimension of 2")
    flat = array.reshape(-1, 2)
    inverse = smooth_pitch(matrix, pitch)
    homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    projected = homogeneous @ inverse.T
    denominator = projected[:, 2]
    out = np.full((flat.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.abs(denominator) > 1e-9
    out[valid] = projected[valid, :2] / denominator[valid, None]
    return out.reshape(array.shape)


def pitch_to_pixel(
    matrix: np.ndarray,
    points: np.ndarray,
    pitch: tuple[float, float] = PITCH,
) -> np.ndarray:
    """Project pitch metres to video pixels with one homography."""
    array = np.asarray(points, dtype=np.float64)
    flat = array.reshape(-1, 2)
    length, width = float(pitch[0]), float(pitch[1])
    normalized = np.concatenate(
        [flat[:, 0:1] / length, flat[:, 1:2] / width, np.ones((flat.shape[0], 1))], axis=1
    )
    projected = normalized @ np.asarray(matrix, dtype=np.float64).T
    denominator = projected[:, 2]
    out = np.full((flat.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.abs(denominator) > 1e-9
    out[valid] = projected[valid, :2] / denominator[valid, None]
    return out.reshape(array.shape)


def _inside_pitch(points: np.ndarray, pitch: tuple[float, float], margin: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    length, width = float(pitch[0]), float(pitch[1])
    return (
        np.isfinite(points[..., 0])
        & np.isfinite(points[..., 1])
        & (points[..., 0] >= -margin)
        & (points[..., 0] <= length + margin)
        & (points[..., 1] >= -margin)
        & (points[..., 1] <= width + margin)
    )


def _box_foot_points(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]), axis=1)


def _box_centres(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0), axis=1)


class _GreedyTracker:
    """Small IoU tracker for the viewer's player dots.

    It intentionally does not pretend to solve identity.  A player's team and position are the
    evidence used by fusion; track ids only keep the overlay stable across adjacent frames.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 15) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.next_id = 0
        self.tracks: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, boxes: np.ndarray, frame: int) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64)
        ids = np.full(boxes.shape[0], -1, dtype=np.int16)
        if boxes.size == 0:
            return ids
        candidates: list[tuple[float, int, int]] = []
        for row, box in enumerate(boxes):
            for track_id, track in self.tracks.items():
                if frame - int(track["frame"]) > self.max_age:
                    continue
                score = self._iou(box, track["box"])
                if score >= self.iou_threshold:
                    candidates.append((score, row, track_id))
        candidates.sort(reverse=True)
        used_rows: set[int] = set()
        used_tracks: set[int] = set()
        for _, row, track_id in candidates:
            if row in used_rows or track_id in used_tracks:
                continue
            ids[row] = track_id
            used_rows.add(row)
            used_tracks.add(track_id)
            self.tracks[track_id] = {"box": boxes[row].copy(), "frame": int(frame)}
        for row, box in enumerate(boxes):
            if row in used_rows:
                continue
            track_id = self.next_id
            self.next_id += 1
            ids[row] = track_id
            self.tracks[track_id] = {"box": box.copy(), "frame": int(frame)}
        for track_id in list(self.tracks):
            if frame - int(self.tracks[track_id]["frame"]) > self.max_age:
                del self.tracks[track_id]
        return ids


class _WhiteBlobBallDetector:
    """Find a small bright, compact blob around a predicted pixel.

    This is intentionally a classical fallback, not a second neural detector.  It is useful when
    the COCO ball class misses a distant ball but the white ball is still visible in the search
    window.  It will also find socks, line paint, and boots; the tracker's motion gate and the
    circularity/area score keep those out in most frames.
    """

    def __init__(
        self,
        *,
        search_radius: int = 68,
        min_area: int = 3,
        max_area: int = 240,
    ) -> None:
        self.search_radius = int(search_radius)
        self.min_area = int(min_area)
        self.max_area = int(max_area)

    def candidates(
        self,
        frame: np.ndarray,
        predicted_pixel: tuple[float, float],
        homography: np.ndarray,
        pitch: tuple[float, float],
        *,
        template: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        import cv2  # lazy heavy import

        height, width = frame.shape[:2]
        px, py = float(predicted_pixel[0]), float(predicted_pixel[1])
        radius = self.search_radius
        x0 = max(0, int(round(px - radius)))
        y0 = max(0, int(round(py - radius)))
        x1 = min(width, int(round(px + radius + 1)))
        y1 = min(height, int(round(py + radius + 1)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        crop = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # The ball is bright and almost neutral.  The threshold is relative to the local crop so
        # shadowed/floodlit areas do not require a global brightness constant.
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        threshold = max(145, int(np.percentile(value, 96)))
        mask = ((value >= threshold) & (saturation <= 105)).astype(np.uint8) * 255
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        results: list[dict[str, Any]] = []
        for label in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if not (self.min_area <= area <= self.max_area):
                continue
            if w < 1 or h < 1 or w > 28 or h > 28:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.32 or aspect > 3.1:
                continue
            fill = area / max(1, w * h)
            if fill < 0.22:
                continue
            cx, cy = centroids[label]
            gx, gy = x0 + float(cx), y0 + float(cy)
            point = pixel_to_pitch(homography, np.array([[gx, gy]]), pitch)[0]
            if not _inside_pitch(point, pitch, BALL_MARGIN):
                continue
            distance = math.hypot(gx - px, gy - py)
            circularity = min(1.0, 2.2 * fill * min(aspect, 1.0 / aspect))
            brightness = float(value[max(0, int(cy) - 1):int(cy) + 2, max(0, int(cx) - 1):int(cx) + 2].mean()) / 255.0
            match = 0.0
            if template is not None and template.size > 0:
                half = template.shape[0] // 2 + 2
                tx0 = max(0, int(round(cx)) - half)
                ty0 = max(0, int(round(cy)) - half)
                tx1 = min(crop.shape[1], tx0 + template.shape[1] + 2 * half)
                ty1 = min(crop.shape[0], ty0 + template.shape[1] + 2 * half)
                patch = cv2.cvtColor(crop[ty0:ty1, tx0:tx1], cv2.COLOR_BGR2GRAY)
                if patch.shape[0] >= template.shape[0] and patch.shape[1] >= template.shape[1]:
                    response = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
                    match = float(np.max(response)) if response.size else 0.0
            score = (
                1.4 * math.exp(-distance / 28.0)
                + 0.35 * circularity
                + 0.20 * brightness
                + 0.55 * max(0.0, match)
            )
            results.append(
                {
                    "pixel": [gx, gy],
                    "pitch": [float(point[0]), float(point[1])],
                    "confidence": max(0.01, min(1.0, score / 2.5)),
                    "score": score,
                    "area": area,
                    "template": match,
                    "distance_px": distance,
                    "origin": "blob",
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:6]


class VideoBallTracker:
    """Causal ball tracker in pitch space.

    The tracker is seeded by a YOLO ball detection (or by an ATD observation only when explicitly
    allowed) and then uses image evidence.  It does not consume ATD observations as measurements
    while it is alive; this is what lets it bridge an ATD gap.  If it is lost and an ATD seed is
    available, it may reacquire near that seed; the cache records the source so the fusion layer
    can discount seeded samples.
    """

    def __init__(
        self,
        *,
        max_gap_s: float = 3.5,
        gate_m: float = 9.0,
        speed_cap: float = 38.0,
        search_radius: int = 68,
    ) -> None:
        self.max_gap_s = float(max_gap_s)
        self.gate_m = float(gate_m)
        self.speed_cap = float(speed_cap)
        self.blobs = _WhiteBlobBallDetector(search_radius=search_radius)
        self.pitch = np.array(PITCH, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.alive = False
        self.last_frame = -1
        self.last_pitch = np.full(2, np.nan, dtype=np.float64)
        self.last_pixel = np.full(2, np.nan, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.confidence = 0.0
        self.template: np.ndarray | None = None
        self.missing = 0

    @staticmethod
    def _nearest_ball(
        detections: list[dict[str, Any]],
        target: np.ndarray,
        max_distance: float,
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_distance = float(max_distance)
        for detection in detections:
            point = np.asarray(detection["pitch"], dtype=np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                continue
            distance = float(np.linalg.norm(point - target))
            if distance < best_distance:
                best, best_distance = detection, distance
        return best

    def _template_from_frame(self, frame: np.ndarray, pixel: tuple[float, float]) -> np.ndarray | None:
        import cv2

        x, y = int(round(pixel[0])), int(round(pixel[1]))
        half = 7
        y0, y1 = max(0, y - half), min(frame.shape[0], y + half + 1)
        x0, x1 = max(0, x - half), min(frame.shape[1], x + half + 1)
        if y1 - y0 < 5 or x1 - x0 < 5:
            return None
        patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if patch.shape[0] > 15 or patch.shape[1] > 15:
            patch = patch[:15, :15]
        return patch.copy()

    def _off_pitch(self, point: np.ndarray, margin: float = 1.5) -> bool:
        return not bool(_inside_pitch(np.asarray(point)[None, :], self.pitch, margin)[0])

    @staticmethod
    def _as_candidate(point: np.ndarray, pixel: np.ndarray | None, confidence: float, origin: str) -> dict[str, Any]:
        point = np.asarray(point, dtype=np.float64)
        if pixel is None:
            pixel = np.full(2, np.nan, dtype=np.float64)
        return {
            "pixel": [float(pixel[0]), float(pixel[1])],
            "pitch": [float(point[0]), float(point[1])],
            "confidence": float(confidence),
            "origin": str(origin),
        }

    def _commit(
        self,
        candidate: dict[str, Any],
        frame: np.ndarray,
        frame_index: int,
        *,
        source: int,
    ) -> dict[str, Any]:
        point = np.asarray(candidate["pitch"], dtype=np.float64)
        pixel = np.asarray(candidate["pixel"], dtype=np.float64)
        if self.alive and self.last_frame >= 0 and np.isfinite(self.last_pitch).all():
            dt = max(1.0 / FPS, (frame_index - self.last_frame) / FPS)
            measured_velocity = (point - self.last_pitch) / dt
            speed = float(np.linalg.norm(measured_velocity))
            if speed > self.speed_cap:
                measured_velocity *= self.speed_cap / speed
            self.velocity = 0.55 * self.velocity + 0.45 * measured_velocity
        else:
            self.velocity = np.zeros(2, dtype=np.float64)
        self.alive = True
        self.last_frame = int(frame_index)
        self.last_pitch = point.copy()
        self.last_pixel = pixel.copy()
        self.missing = 0
        confidence = float(candidate.get("confidence", 0.5))
        self.confidence = max(0.08, min(1.0, confidence))
        template = self._template_from_frame(frame, (float(pixel[0]), float(pixel[1])))
        if template is not None and (self.template is None or self.confidence >= 0.20):
            self.template = template
        return {
            "pixel": [float(pixel[0]), float(pixel[1])],
            "pitch": [float(point[0]), float(point[1])],
            "confidence": self.confidence,
            "source": int(source),
            "origin": str(candidate.get("origin", "yolo")),
        }

    def _best_visible(
        self,
        visible: list[dict[str, Any]],
        target: np.ndarray,
        gate: float,
        *,
        min_confidence: float = 0.06,
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = float(gate)
        for detection in visible:
            point = np.asarray(detection["pitch"], dtype=np.float64)
            if not np.isfinite(point).all() or self._off_pitch(point):
                continue
            confidence = float(detection.get("confidence", 0.0))
            if confidence < min_confidence:
                continue
            distance = float(np.linalg.norm(point - target))
            # Prefer a confident detection even if it is slightly farther, but never let a weak
            # detection jump across the gate.
            score = distance - 1.5 * min(confidence, 0.6)
            if score < best_score:
                best, best_score = detection, score
        return best

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        homography: np.ndarray,
        detections: list[dict[str, Any]],
        *,
        atd_anchor: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return one tracker result with pixel/pitch/confidence/source.

        Source codes: 0 = video detection, 1 = prediction through a gap, 2 = lost, 3 = ATD anchor.
        The ATD anchor is used only while it is available.  Between ATD observations the tracker
        predicts and uses YOLO/white-blob measurements, so it can fill the 40% of frames where the
        ATD ball is absent.
        """
        visible = []
        for detection in detections:
            point = np.asarray(detection["pitch"], dtype=np.float64)
            if np.isfinite(point).all() and not self._off_pitch(point):
                visible.append(detection)
        anchor = None
        if atd_anchor is not None:
            candidate_anchor = np.asarray(atd_anchor, dtype=np.float64)
            if (
                candidate_anchor.shape == (2,)
                and np.isfinite(candidate_anchor).all()
                and not self._off_pitch(candidate_anchor)
            ):
                anchor = candidate_anchor

        if anchor is not None:
            # The ATD is the anchor while it exists.  If the image tracker has drifted, snap back;
            # this is what prevents a white touchline or a player's sock from becoming the ball.
            if (
                not self.alive
                or not np.isfinite(self.last_pitch).all()
                or self._off_pitch(self.last_pitch)
                or float(np.linalg.norm(self.last_pitch - anchor)) > 5.0
            ):
                near = self._best_visible(visible, anchor, 4.0, min_confidence=0.18)
                if near is None:
                    pixel = pitch_to_pixel(homography, anchor[None, :], self.pitch)[0]
                    candidate = self._as_candidate(anchor, pixel, 0.82, "atd")
                else:
                    candidate = near
                    candidate["origin"] = "yolo+atd"
                return self._commit(candidate, frame, frame_index, source=3 if near is None else 0)

            near = self._best_visible(visible, anchor, 3.2, min_confidence=0.12)
            if near is None:
                pixel = pitch_to_pixel(homography, anchor[None, :], self.pitch)[0]
                candidate = self._as_candidate(anchor, pixel, 0.78, "atd")
                return self._commit(candidate, frame, frame_index, source=3)
            # Use the image measurement when it is close to the anchor; otherwise trust the anchor.
            if float(np.linalg.norm(np.asarray(near["pitch"]) - anchor)) <= 4.0:
                near["origin"] = "yolo+atd"
                return self._commit(near, frame, frame_index, source=0)
            pixel = pitch_to_pixel(homography, anchor[None, :], self.pitch)[0]
            candidate = self._as_candidate(anchor, pixel, 0.78, "atd")
            return self._commit(candidate, frame, frame_index, source=3)

        # From here on the ATD ball is unavailable: this is the actual gap-filling path.
        if self.alive and self.last_frame >= 0 and np.isfinite(self.last_pitch).all():
            gap = max(0.0, (frame_index - self.last_frame) / FPS)
            if gap > self.max_gap_s or self._off_pitch(self.last_pitch):
                self.alive = False
                self.confidence = 0.0
            else:
                decay = math.exp(-0.45 * gap)
                predicted = self.last_pitch + self.velocity * (1.0 - decay) / 0.45
                if self._off_pitch(predicted):
                    self.alive = False
                    self.confidence = 0.0
                else:
                    gate = max(4.0, min(16.0, 2.5 + 8.0 * gap))
                    candidate = self._best_visible(visible, predicted, gate)
                    if candidate is None:
                        predicted_pixel = pitch_to_pixel(homography, predicted[None, :], self.pitch)[0]
                        if np.isfinite(predicted_pixel).all():
                            blobs = self.blobs.candidates(
                                frame,
                                (float(predicted_pixel[0]), float(predicted_pixel[1])),
                                homography,
                                self.pitch,
                                template=self.template,
                            )
                            best_blob = None
                            best_distance = gate
                            for blob in blobs:
                                distance = float(np.linalg.norm(np.asarray(blob["pitch"]) - predicted))
                                score = distance - 0.8 * float(blob.get("confidence", 0.0))
                                if score < best_distance:
                                    best_blob, best_distance = blob, score
                            candidate = best_blob
                    if candidate is not None:
                        return self._commit(candidate, frame, frame_index, source=0)
                    point = predicted
                    pixel = pitch_to_pixel(homography, point[None, :], self.pitch)[0]
                    self.missing += 1
                    self.confidence = max(0.03, self.confidence * math.exp(-gap / 1.2))
                    return {
                        "pixel": [float(pixel[0]), float(pixel[1])]
                        if np.isfinite(pixel).all()
                        else [float("nan")] * 2,
                        "pitch": [float(point[0]), float(point[1])],
                        "confidence": self.confidence,
                        "source": 1,
                        "origin": "prediction",
                    }

        # No anchor and no useful prediction: reacquire from a confident YOLO ball.  A weak
        # detection is not enough to start a new track, because the class has false positives in
        # crowds and advertising boards.
        visible.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        seed = next((item for item in visible if float(item.get("confidence", 0.0)) >= 0.22), None)
        if seed is not None:
            seed["origin"] = "yolo-reacquire"
            return self._commit(seed, frame, frame_index, source=0)

        self.alive = False
        self.confidence = 0.0
        return {
            "pixel": [float("nan"), float("nan")],
            "pitch": [float("nan"), float("nan")],
            "confidence": 0.0,
            "source": 2,
            "origin": "lost",
        }


def _empty_perception(
    *,
    samples: int,
    n_frames: int,
    fps: int,
    step: int,
    start: int,
    stop: int,
    model: str,
    device: str,
    video_width: int,
    video_height: int,
) -> VideoPerception:
    return VideoPerception(
        frame_index=np.zeros(samples, dtype=np.int32),
        players_box=np.full((samples, MAX_PLAYERS, 4), np.nan, dtype=np.float32),
        players_pixel=np.full((samples, MAX_PLAYERS, 2), np.nan, dtype=np.float32),
        players_pitch=np.full((samples, MAX_PLAYERS, 2), np.nan, dtype=np.float32),
        players_team=np.full((samples, MAX_PLAYERS), -1, dtype=np.int8),
        players_confidence=np.zeros((samples, MAX_PLAYERS), dtype=np.float32),
        players_present=np.zeros((samples, MAX_PLAYERS), dtype=bool),
        players_track=np.full((samples, MAX_PLAYERS), -1, dtype=np.int16),
        players_team_confidence=np.zeros((samples, MAX_PLAYERS), dtype=np.float32),
        ball_pixel=np.full((samples, 2), np.nan, dtype=np.float32),
        ball_pitch=np.full((samples, 2), np.nan, dtype=np.float32),
        ball_confidence=np.zeros(samples, dtype=np.float32),
        ball_source=np.full(samples, 2, dtype=np.int8),
        yolo_ball_pixel=np.full((samples, MAX_BALL_DETECTIONS, 2), np.nan, dtype=np.float32),
        yolo_ball_pitch=np.full((samples, MAX_BALL_DETECTIONS, 2), np.nan, dtype=np.float32),
        yolo_ball_confidence=np.zeros((samples, MAX_BALL_DETECTIONS), dtype=np.float32),
        yolo_ball_present=np.zeros((samples, MAX_BALL_DETECTIONS), dtype=bool),
        n_frames=int(n_frames),
        fps=int(fps),
        step=int(step),
        start=int(start),
        stop=int(stop),
        model=str(model),
        device=str(device),
        video_width=int(video_width),
        video_height=int(video_height),
        pitch=PITCH,
    )


def _team_match(
    detections: np.ndarray,
    atd_players: np.ndarray,
    atd_teams: np.ndarray,
    homography: np.ndarray,
    pitch: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign team labels to YOLO players by one-to-one matching to projected ATD slots."""
    if detections.size == 0 or atd_players.size == 0:
        return np.full(detections.shape[0], -1, dtype=np.int8), np.zeros(detections.shape[0], dtype=np.float32)
    atd_valid = np.isfinite(atd_players).all(axis=1)
    atd_players = atd_players[atd_valid]
    atd_teams = np.asarray(atd_teams, dtype=np.int8)[atd_valid]
    if atd_players.size == 0:
        return np.full(detections.shape[0], -1, dtype=np.int8), np.zeros(detections.shape[0], dtype=np.float32)
    projected = pitch_to_pixel(homography, atd_players, pitch)
    costs = np.linalg.norm(detections[:, None, :] - projected[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(costs)
    except Exception:  # pragma: no cover - scipy is an optional perception dependency
        rows = np.arange(min(costs.shape))
        cols = np.argmin(costs, axis=1)[rows]
    teams = np.full(detections.shape[0], -1, dtype=np.int8)
    confidence = np.zeros(detections.shape[0], dtype=np.float32)
    for row, col in zip(rows, cols):
        if row >= detections.shape[0] or col >= projected.shape[0]:
            continue
        distance = float(costs[row, col])
        if not np.isfinite(distance) or distance > 34.0:
            continue
        teams[row] = int(atd_teams[col])
        confidence[row] = max(0.0, 1.0 - distance / 34.0)
    return teams, confidence


def _parse_boxes(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int16),
        )
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidence = boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = boxes.cls.detach().cpu().numpy().astype(np.int16)
    return xyxy, confidence, classes


def perceive_video(
    video_path: str | Path,
    homography: Any,
    tracking: Any | None = None,
    *,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    model: str = "yolo11x.pt",
    device: str | int | None = None,
    imgsz: int = 1920,
    player_conf: float = 0.12,
    ball_conf: float = 0.05,
    max_players: int = MAX_PLAYERS,
    max_ball_detections: int = MAX_BALL_DETECTIONS,
    seed_from_atd: bool = True,
    progress: bool = True,
) -> VideoPerception:
    """Run the video perception on a frame range and return a compact sample array.

    ``step`` is useful for a full-match pass: 5 gives 5 fps while preserving the video's 25 fps
    timeline.  The ball tracker uses the real frame gap, so it still predicts across skipped
    frames.  For the public evaluation window, step 1 is practical.
    """
    import cv2
    from ultralytics import YOLO

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    n_frames = int(getattr(tracking, "n_frames", 0) or 0)
    if n_frames <= 0:
        n_frames = int(getattr(homography, "n_frames", 0) or 0)
    if n_frames <= 0:
        raise ValueError("need tracking or homography to determine n_frames")
    start = max(0, int(start))
    stop = n_frames if stop is None else min(n_frames, int(stop))
    step = max(1, int(step))
    max_players = max(1, min(MAX_PLAYERS, int(max_players)))
    max_ball_detections = max(1, min(MAX_BALL_DETECTIONS, int(max_ball_detections)))
    if stop <= start:
        raise ValueError(f"invalid video range [{start}, {stop})")
    if device is None:
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
    device = str(device)
    n_samples = (stop - start + step - 1) // step
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    perception = _empty_perception(
        samples=n_samples,
        n_frames=n_frames,
        fps=FPS,
        step=step,
        start=start,
        stop=stop,
        model=model,
        device=device,
        video_width=width,
        video_height=height,
    )
    yolo = YOLO(model)
    tracker = _GreedyTracker()
    ball_tracker = VideoBallTracker()
    sample = 0
    current = start
    atd_players = getattr(tracking, "players", None) if tracking is not None else None
    atd_teams = getattr(tracking, "teams", None) if tracking is not None else None
    atd_ball = getattr(tracking, "ball", None) if tracking is not None else None
    atd_present = getattr(tracking, "ball_present", None) if tracking is not None else None
    last_log = 0
    try:
        while current < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if (current - start) % step != 0:
                current += 1
                continue
            if sample >= n_samples:
                break
            result = yolo.predict(
                frame,
                classes=[PERSON_CLASS, BALL_CLASS],
                conf=min(player_conf, ball_conf),
                iou=0.5,
                imgsz=int(imgsz),
                device=device,
                verbose=False,
                max_det=120,
            )[0]
            boxes, confidence, classes = _parse_boxes(result)
            matrix = np.asarray(homography.matrices[current], dtype=np.float64)
            person_mask = (classes == PERSON_CLASS) & (confidence >= player_conf)
            person_boxes = boxes[person_mask]
            person_conf = confidence[person_mask]
            ball_mask = (classes == BALL_CLASS) & (confidence >= ball_conf)
            ball_boxes = boxes[ball_mask]
            ball_conf_values = confidence[ball_mask]

            # Players: reject crowd/staff by the foot point, then assign teams and track ids.
            player_pixel = _box_foot_points(person_boxes) if person_boxes.size else np.zeros((0, 2))
            player_pitch = (
                pixel_to_pitch(matrix, player_pixel, perception.pitch)
                if player_pixel.size
                else np.zeros((0, 2))
            )
            valid_players = _inside_pitch(player_pitch, perception.pitch, PITCH_MARGIN)
            person_boxes = person_boxes[valid_players]
            person_conf = person_conf[valid_players]
            player_pixel = player_pixel[valid_players]
            player_pitch = player_pitch[valid_players]
            if atd_players is not None and atd_teams is not None and current < len(atd_players):
                teams, team_confidence = _team_match(
                    player_pixel,
                    np.asarray(atd_players[current], dtype=np.float64),
                    np.asarray(atd_teams),
                    matrix,
                    perception.pitch,
                )
            else:
                teams = np.full(len(person_boxes), -1, dtype=np.int8)
                team_confidence = np.zeros(len(person_boxes), dtype=np.float32)
            track_ids = tracker.update(person_boxes, current)
            order = np.argsort(person_conf)[::-1][:max_players]
            for slot, source_row in enumerate(order):
                if slot >= max_players:
                    break
                box = person_boxes[source_row]
                perception.players_box[sample, slot] = box
                perception.players_pixel[sample, slot] = player_pixel[source_row]
                perception.players_pitch[sample, slot] = player_pitch[source_row]
                perception.players_team[sample, slot] = teams[source_row]
                perception.players_confidence[sample, slot] = person_conf[source_row]
                perception.players_present[sample, slot] = True
                perception.players_track[sample, slot] = track_ids[source_row]
                perception.players_team_confidence[sample, slot] = team_confidence[source_row]

            # Raw YOLO ball detections, filtered by their centre's pitch projection.
            ball_centres = _box_centres(ball_boxes) if ball_boxes.size else np.zeros((0, 2))
            ball_points = (
                pixel_to_pitch(matrix, ball_centres, perception.pitch)
                if ball_centres.size
                else np.zeros((0, 2))
            )
            valid_balls = _inside_pitch(ball_points, perception.pitch, BALL_MARGIN)
            candidates: list[dict[str, Any]] = []
            for row in np.flatnonzero(valid_balls):
                point = ball_points[row]
                candidates.append(
                    {
                        "pixel": [float(ball_centres[row, 0]), float(ball_centres[row, 1])],
                        "pitch": [float(point[0]), float(point[1])],
                        "confidence": float(ball_conf_values[row]),
                        "origin": "yolo",
                    }
                )
            candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
            for slot, candidate in enumerate(candidates[:max_ball_detections]):
                perception.yolo_ball_pixel[sample, slot] = candidate["pixel"]
                perception.yolo_ball_pitch[sample, slot] = candidate["pitch"]
                perception.yolo_ball_confidence[sample, slot] = candidate["confidence"]
                perception.yolo_ball_present[sample, slot] = True
            seed = None
            if (
                seed_from_atd
                and atd_ball is not None
                and atd_present is not None
                and current < len(atd_ball)
                and bool(atd_present[current])
                and np.isfinite(atd_ball[current]).all()
            ):
                seed = np.asarray(atd_ball[current], dtype=np.float64)
            tracked = ball_tracker.update(
                frame,
                current,
                matrix,
                candidates,
                atd_anchor=seed,
            )
            perception.ball_pixel[sample] = tracked["pixel"]
            perception.ball_pitch[sample] = tracked["pitch"]
            perception.ball_confidence[sample] = tracked["confidence"]
            perception.ball_source[sample] = tracked["source"]
            perception.frame_index[sample] = current
            sample += 1
            if progress and sample - last_log >= 200:
                last_log = sample
                print(
                    f"perception: {sample:,}/{n_samples:,} samples "
                    f"(frame {current:,}/{stop:,})",
                    flush=True,
                )
            current += 1
    finally:
        cap.release()
    if sample < n_samples:
        perception = _slice_perception(perception, sample)
    return perception


def _slice_perception(perception: VideoPerception, samples: int) -> VideoPerception:
    return VideoPerception(
        frame_index=perception.frame_index[:samples],
        players_box=perception.players_box[:samples],
        players_pixel=perception.players_pixel[:samples],
        players_pitch=perception.players_pitch[:samples],
        players_team=perception.players_team[:samples],
        players_confidence=perception.players_confidence[:samples],
        players_present=perception.players_present[:samples],
        players_track=perception.players_track[:samples],
        players_team_confidence=perception.players_team_confidence[:samples],
        ball_pixel=perception.ball_pixel[:samples],
        ball_pitch=perception.ball_pitch[:samples],
        ball_confidence=perception.ball_confidence[:samples],
        ball_source=perception.ball_source[:samples],
        yolo_ball_pixel=perception.yolo_ball_pixel[:samples],
        yolo_ball_pitch=perception.yolo_ball_pitch[:samples],
        yolo_ball_confidence=perception.yolo_ball_confidence[:samples],
        yolo_ball_present=perception.yolo_ball_present[:samples],
        n_frames=perception.n_frames,
        fps=perception.fps,
        step=perception.step,
        start=perception.start,
        stop=perception.stop,
        model=perception.model,
        device=perception.device,
        video_width=perception.video_width,
        video_height=perception.video_height,
        pitch=perception.pitch,
        meta=perception.meta,
    )


def save_perception(path: str | Path, perception: VideoPerception) -> Path:
    """Write a compressed perception cache with a small JSON metadata sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = perception.to_arrays()
    meta = {
        "version": 1,
        "n_frames": perception.n_frames,
        "fps": perception.fps,
        "step": perception.step,
        "start": perception.start,
        "stop": perception.stop,
        "model": perception.model,
        "device": perception.device,
        "video_width": perception.video_width,
        "video_height": perception.video_height,
        "pitch": list(perception.pitch),
        "samples": perception.samples,
        **perception.meta,
    }
    np.savez_compressed(path, **arrays, meta=np.asarray(json.dumps(meta, ensure_ascii=False)))
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_perception(path: str | Path) -> VideoPerception | None:
    """Read a cache written by :func:`save_perception`; return ``None`` when absent."""
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        arrays = {name: archive[name] for name in (*PLAYER_ARRAY_NAMES, *BALL_ARRAY_NAMES, "frame_index")}
    return VideoPerception(
        frame_index=arrays["frame_index"].astype(np.int32),
        players_box=arrays["players_box"].astype(np.float32),
        players_pixel=arrays["players_pixel"].astype(np.float32),
        players_pitch=arrays["players_pitch"].astype(np.float32),
        players_team=arrays["players_team"].astype(np.int8),
        players_confidence=arrays["players_confidence"].astype(np.float32),
        players_present=arrays["players_present"].astype(bool),
        players_track=arrays["players_track"].astype(np.int16),
        players_team_confidence=arrays["players_team_confidence"].astype(np.float32),
        ball_pixel=arrays["ball_pixel"].astype(np.float32),
        ball_pitch=arrays["ball_pitch"].astype(np.float32),
        ball_confidence=arrays["ball_confidence"].astype(np.float32),
        ball_source=arrays["ball_source"].astype(np.int8),
        yolo_ball_pixel=arrays["yolo_ball_pixel"].astype(np.float32),
        yolo_ball_pitch=arrays["yolo_ball_pitch"].astype(np.float32),
        yolo_ball_confidence=arrays["yolo_ball_confidence"].astype(np.float32),
        yolo_ball_present=arrays["yolo_ball_present"].astype(bool),
        n_frames=int(meta["n_frames"]),
        fps=int(meta["fps"]),
        step=int(meta["step"]),
        start=int(meta["start"]),
        stop=int(meta["stop"]),
        model=str(meta["model"]),
        device=str(meta["device"]),
        video_width=int(meta["video_width"]),
        video_height=int(meta["video_height"]),
        pitch=tuple(float(value) for value in meta.get("pitch", PITCH)),
        meta=meta,
    )


def resample_to_frames(
    perception: VideoPerception | None,
    frames: Iterable[int],
) -> dict[str, np.ndarray]:
    """Nearest-sample view for a list of full-resolution frames.

    This is a convenience for fusion tests and diagnostics.  The production fusion path uses
    :meth:`VideoPerception.nearest_index` arrays directly, avoiding a Python loop over 199k frames.
    """
    frame_array = np.asarray(list(frames), dtype=np.int64)
    if perception is None or perception.samples == 0:
        return {
            "index": np.zeros(frame_array.size, dtype=np.int64),
            "age": np.zeros(frame_array.size, dtype=np.int64),
            "present": np.zeros(frame_array.size, dtype=bool),
        }
    indices = np.array([perception.nearest_index(int(frame)) for frame in frame_array], dtype=np.int64)
    sample_frames = perception.frame_index[indices].astype(np.int64)
    return {
        "index": indices,
        "age": frame_array - sample_frames,
        "present": np.ones(frame_array.size, dtype=bool),
    }
