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

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import FPS, GRADED_HALVES, PITCH

# COCO class ids used by the stock YOLO weights.
PERSON_CLASS = 0
BALL_CLASS = 32

# A player farther than this from the pitch is a spectator/staff detection, not a player.
PITCH_MARGIN = 1.5
# A ball farther than this from the pitch is a ball boy/advert/crowd false positive.
BALL_MARGIN = 1.2
# Physical gate for temporal ball association.  Match-data ball speeds are below roughly
# 40 m/s (about 145 km/h), including shots; the extra metre covers homography/projection noise.
# This is deliberately an upper bound: a false positive that is farther than this from the last
# accepted point is not the same ball, no matter how confident the detector is.
BALL_MAX_SPEED_MPS = 40.0
BALL_GATE_TOLERANCE_M = 1.0
BALL_MIN_CONFIDENCE = 0.18
BALL_TRACK_MAX_GAP_S = 1.0

# Per-candidate decision codes written by ``_run_ball_track`` (see ``annotate_ball_candidates``).
# They make the gate transparent: every YOLO ball candidate is labelled with why it was kept or
# dropped, which is the evidence the viewer overlays.  They are an online physical/geometric prior
# only, never ground truth.
BALL_CAND_REJECTED_LOW = 0  # below min_confidence and not reachable as a low-confidence recovery
BALL_CAND_ACCEPTED = 1  # chosen for the high-confidence physical track
BALL_CAND_RECOVERED = 2  # chosen by the second (low-confidence) association pass
BALL_CAND_REJECTED_UNREACHABLE = 3  # outside the ``max_speed * dt + tolerance`` gate
BALL_CAND_REJECTED_OCCLUDED = 4  # sits on a player position: likely a shirt patch, not the ball
BALL_CAND_UNSELECTED = 5  # plausible and reachable, but another candidate won this frame

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
    players_official: np.ndarray | None = None  # (M, K) bool: referee/official, never a player
    players_class: np.ndarray | None = None  # (M, K) int8: 0 player, 1 goalkeeper, 2 referee

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
                    "official": bool(
                        self.players_official is not None and self.players_official[index, slot]
                    ),
                }
            )
        result["players"] = players
        return result

    def to_arrays(self) -> dict[str, np.ndarray]:
        out = {
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
        if self.players_official is not None:
            out["players_official"] = np.asarray(self.players_official, dtype=bool)
        if self.players_class is not None:
            out["players_class"] = np.asarray(self.players_class, dtype=np.int8)
        return out


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


def ball_gate_metres(
    dt_s: float,
    *,
    max_speed: float = BALL_MAX_SPEED_MPS,
    tolerance: float = BALL_GATE_TOLERANCE_M,
) -> float:
    """Maximum plausible ball displacement after ``dt_s`` seconds."""
    return max(0.0, float(max_speed) * max(0.0, float(dt_s)) + float(tolerance))


def _min_distance_to_players(
    point: np.ndarray, players: np.ndarray | None
) -> float:
    """Smallest pitch distance from ``point`` to any finite player position (``inf`` if none)."""
    if players is None:
        return math.inf
    players = np.asarray(players, dtype=np.float64)
    if players.size == 0:
        return math.inf
    players = players.reshape(-1, 2)
    players = players[np.isfinite(players).all(axis=1)]
    if players.size == 0:
        return math.inf
    return float(np.min(np.linalg.norm(players - np.asarray(point, dtype=np.float64), axis=1)))


@dataclass
class BallTrackDecision:
    """Rich result of the online ball-association pass.

    ``select_plausible_ball_detections`` returns the first four arrays; ``annotate_ball_candidates``
    returns ``decisions``; ``ball_track_ids`` is derived from ``status`` and ``frame_index``.
    """

    ball: np.ndarray  # (samples, 2) accepted pitch point, or NaN when nothing was usable
    confidence: np.ndarray  # (samples,)
    status: np.ndarray  # (samples,) 0 accepted, 1 low-confidence accepted, 2 none, 3 recovered
    selected: np.ndarray  # (samples,) chosen candidate row, -1 when none
    decisions: np.ndarray  # (samples, candidates) per-candidate BALL_CAND_* code
    track_id: np.ndarray  # (samples,) ball track id, -1 when no track is live


def _run_ball_track(
    points: np.ndarray,
    confidence: np.ndarray,
    present: np.ndarray,
    frame_index: np.ndarray,
    *,
    fps: float = FPS,
    max_speed: float = BALL_MAX_SPEED_MPS,
    tolerance: float = BALL_GATE_TOLERANCE_M,
    min_confidence: float = BALL_MIN_CONFIDENCE,
    max_gap_s: float = BALL_TRACK_MAX_GAP_S,
    low_confidence_floor: float | None = None,
    player_points: np.ndarray | None = None,
    occlusion_radius_m: float = 0.0,
) -> BallTrackDecision:
    """Associate YOLO ball detections into one physically reachable track per sample.

    ``points`` is ``(samples, candidates, 2)`` in pitch metres.  Stage 1 keeps the strict, high-
    confidence physical track: a measurement farther from the previous accepted ball than
    ``max_speed * dt + tolerance`` is rejected even if YOLO is very confident, and a new track needs
    two mutually reachable detections so a single false positive cannot seed or move it.

    Two optional, paper-inspired extensions are **off by default** so the strict track is unchanged:

    * ``low_confidence_floor`` adds a ByteTrack/BotSORT second stage.  Only when stage 1 found
      nothing usable *and* a track is already live, a candidate in ``[floor, min_confidence)`` that
      is still physically reachable is accepted as a recovery (``status == 3``).  It can fill a
      one-frame detector flicker but can never seed a track or outrank a confident detection.
    * ``occlusion_radius_m`` adds a generic occlusion guard: a low-confidence recovery that sits within
      this many metres of a player position (``player_points``) is treated as a shirt patch and
      dropped.  It only rejects recoveries, so it never alters the strict stage-1 track.

    This is an online physical and geometric prior only: pitch geometry and time, never ground truth.
    """
    points = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    frame_index = np.asarray(frame_index, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"points must have shape (samples, candidates, 2), got {points.shape}")
    if confidence.shape != points.shape[:2] or present.shape != points.shape[:2]:
        raise ValueError("points, confidence, and present shapes disagree")
    if frame_index.shape != (points.shape[0],):
        raise ValueError("frame_index must have one entry per sample")

    samples = points.shape[0]
    ball = np.full((samples, 2), np.nan, dtype=np.float64)
    ball_confidence = np.zeros(samples, dtype=np.float64)
    status = np.full(samples, 2, dtype=np.int8)
    selected = np.full(samples, -1, dtype=np.int16)
    decisions = np.full(points.shape[:2], BALL_CAND_UNSELECTED, dtype=np.int8)
    decisions[~present] = BALL_CAND_REJECTED_LOW
    track_id = np.full(samples, -1, dtype=np.int16)
    if samples == 0:
        return BallTrackDecision(ball, ball_confidence, status, selected, decisions, track_id)

    last: np.ndarray | None = None
    last_frame = -1
    pending: np.ndarray | None = None
    pending_frame = -1
    current_track = -1
    occlusion_on = float(occlusion_radius_m) > 0.0 and player_points is not None

    def label(sample: int, chosen: int, recovered: bool, anchor: np.ndarray | None, gate: float) -> None:
        rows = np.flatnonzero(present[sample])
        for row in rows:
            value = float(confidence[sample, row])
            if row == chosen:
                decisions[sample, row] = BALL_CAND_RECOVERED if recovered else BALL_CAND_ACCEPTED
            elif (
                occlusion_on
                and low_confidence_floor is not None
                and float(low_confidence_floor) <= value < float(min_confidence)
                and _min_distance_to_players(points[sample, row], player_points[sample])
                <= float(occlusion_radius_m)
            ):
                # A recovery candidate dropped specifically because it sits on a player patch.
                decisions[sample, row] = BALL_CAND_REJECTED_OCCLUDED
            elif value < float(min_confidence):
                decisions[sample, row] = BALL_CAND_REJECTED_LOW
            elif anchor is not None and np.linalg.norm(points[sample, row] - anchor) > gate:
                decisions[sample, row] = BALL_CAND_REJECTED_UNREACHABLE
            else:
                decisions[sample, row] = BALL_CAND_UNSELECTED

    for sample in range(samples):
        valid = present[sample] & np.isfinite(points[sample]).all(axis=1)
        rows = np.flatnonzero(valid & (confidence[sample] >= float(min_confidence)))
        if pending is not None and (frame_index[sample] - pending_frame) / fps > max_gap_s:
            pending = None
        # No confident detection and no live track: nothing can be accepted (a recovery needs a
        # track to keep alive).  With a live track we fall through so the second stage can recover.
        if rows.size == 0 and last is None:
            label(sample, -1, False, None, ball_gate_metres(1.0 / fps, max_speed=max_speed, tolerance=tolerance))
            continue

        chosen = -1
        recovered = False
        gate = ball_gate_metres(
            max(1.0 / fps, (frame_index[sample] - (pending_frame if last is None else last_frame)) / fps),
            max_speed=max_speed, tolerance=tolerance,
        )

        if last is None:
            # A single detection is not enough to start a new track.  Confirm it with a second
            # detection that is reachable at football speed.
            if pending is None:
                row = int(rows[np.argmax(confidence[sample, rows])])
                pending = points[sample, row].copy()
                pending_frame = int(round(frame_index[sample]))
                label(sample, -1, False, pending, gate)
                continue
            distances = np.linalg.norm(points[sample, rows] - pending, axis=1)
            reachable = rows[distances <= gate]
            if reachable.size == 0:
                row = int(rows[np.argmax(confidence[sample, rows])])
                pending = points[sample, row].copy()
                pending_frame = int(round(frame_index[sample]))
                label(sample, -1, False, pending, gate)
                continue
            chosen = int(reachable[np.argmin(np.linalg.norm(points[sample, reachable] - pending, axis=1))])
        else:
            dt = (frame_index[sample] - last_frame) / fps
            if dt > max_gap_s:
                last = None
                current_track = -1
                if rows.size:
                    row = int(rows[np.argmax(confidence[sample, rows])])
                    pending = points[sample, row].copy()
                    pending_frame = int(round(frame_index[sample]))
                label(sample, -1, False, pending, gate)
                continue
            dt = max(1.0 / fps, dt)
            gate = ball_gate_metres(dt, max_speed=max_speed, tolerance=tolerance)
            distances = np.linalg.norm(points[sample, rows] - last, axis=1)
            reachable = rows[distances <= gate]
            if reachable.size > 0:
                chosen = int(reachable[np.argmin(np.linalg.norm(points[sample, reachable] - last, axis=1))])
            elif low_confidence_floor is not None:
                # Second stage: a confident detection is missing this frame, so let a reachable low-
                # confidence candidate keep the live track alive (never seeds, never occluded).
                low_rows = np.flatnonzero(
                    valid
                    & (confidence[sample] >= float(low_confidence_floor))
                    & (confidence[sample] < float(min_confidence))
                )
                if occlusion_on and low_rows.size:
                    keep = [
                        row
                        for row in low_rows
                        if _min_distance_to_players(points[sample, row], player_points[sample])
                        > float(occlusion_radius_m)
                    ]
                    low_rows = np.asarray(keep, dtype=np.intp)
                if low_rows.size:
                    low_dist = np.linalg.norm(points[sample, low_rows] - last, axis=1)
                    low_reach = low_rows[low_dist <= gate]
                    if low_reach.size:
                        chosen = int(low_reach[np.argmin(np.linalg.norm(points[sample, low_reach] - last, axis=1))])
                        recovered = True
            if chosen < 0:
                # Keep the previous point for a short reacquisition window, but do not emit a
                # measurement for this frame.
                label(sample, -1, False, last, gate)
                continue

        point = points[sample, chosen]
        value = float(confidence[sample, chosen])
        ball[sample] = point
        ball_confidence[sample] = value
        status[sample] = 3 if recovered else (0 if value >= 0.25 else 1)
        selected[sample] = chosen
        if current_track < 0:
            current_track = int(np.max(track_id)) + 1 if np.any(track_id >= 0) else 0
        track_id[sample] = current_track
        last = point.copy()
        last_frame = int(round(frame_index[sample]))
        pending = None
        label(sample, chosen, recovered, point, gate)

    return BallTrackDecision(ball, ball_confidence, status, selected, decisions, track_id)


def select_plausible_ball_detections(
    points: np.ndarray,
    confidence: np.ndarray,
    present: np.ndarray,
    frame_index: np.ndarray,
    *,
    fps: float = FPS,
    max_speed: float = BALL_MAX_SPEED_MPS,
    tolerance: float = BALL_GATE_TOLERANCE_M,
    min_confidence: float = BALL_MIN_CONFIDENCE,
    max_gap_s: float = BALL_TRACK_MAX_GAP_S,
    low_confidence_floor: float | None = None,
    player_points: np.ndarray | None = None,
    occlusion_radius_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the temporal YOLO-ball track that is physically reachable.

    Returns ``(ball, confidence, status, selected)``.  ``status`` is ``0`` for an accepted
    detection, ``1`` for a low-confidence accepted detection, ``2`` for no usable measurement and
    ``3`` for a second-stage recovery.  The optional ``low_confidence_floor`` / ``occlusion_radius_m``
    extensions are off by default, which reproduces the strict physical track exactly.  See
    :func:`_run_ball_track` for the full method.
    """
    result = _run_ball_track(
        points,
        confidence,
        present,
        frame_index,
        fps=fps,
        max_speed=max_speed,
        tolerance=tolerance,
        min_confidence=min_confidence,
        max_gap_s=max_gap_s,
        low_confidence_floor=low_confidence_floor,
        player_points=player_points,
        occlusion_radius_m=occlusion_radius_m,
    )
    return result.ball, result.confidence, result.status, result.selected


def annotate_ball_candidates(*args: Any, **kwargs: Any) -> np.ndarray:
    """Return the ``(samples, candidates)`` per-candidate ``BALL_CAND_*`` decision grid.

    Takes the same arguments as :func:`select_plausible_ball_detections`.  It never changes the
    accepted track; it only reports why each candidate was kept or dropped.
    """
    return _run_ball_track(*args, **kwargs).decisions


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
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        results: list[dict[str, Any]] = []
        for label in range(1, count):
            _x, _y, w, h, area = [int(v) for v in stats[label]]
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

    The tracker is seeded by an ATD observation, or by two mutually reachable YOLO ball
    detections, and then uses image evidence.  It does not consume ATD observations as
    measurements while it is alive; this is what lets it bridge an ATD gap.  If it is lost and an
    ATD seed is available, it may reacquire near that seed; the cache records the source so the
    fusion layer can discount seeded samples.
    """

    def __init__(
        self,
        *,
        max_gap_s: float = 3.5,
        gate_m: float = 9.0,
        speed_cap: float = BALL_MAX_SPEED_MPS,
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
        # Confidence of the last genuinely observed ball; predicted frames decay from this, not from
        # their own already-decayed value.
        self.anchor_confidence = 0.0
        self.template: np.ndarray | None = None
        self.missing = 0
        self.pending: dict[str, Any] | None = None
        self.pending_frame = -1

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
        self.pending = None
        self.pending_frame = -1
        confidence = float(candidate.get("confidence", 0.5))
        self.confidence = max(0.08, min(1.0, confidence))
        self.anchor_confidence = self.confidence
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

    def _motion_gate(self, frame_index: int, reference_frame: int) -> float:
        """Max reachable distance between two samples, with a projection tolerance."""
        dt = max(1.0 / FPS, (int(frame_index) - int(reference_frame)) / FPS)
        return min(self.gate_m, ball_gate_metres(dt, max_speed=self.speed_cap))

    def _reachable(self, point: np.ndarray, frame_index: int) -> bool:
        """Whether a point can be the same ball as the last accepted observation."""
        if (
            not self.alive
            or self.last_frame < 0
            or not np.isfinite(self.last_pitch).all()
        ):
            return True
        distance = float(np.linalg.norm(np.asarray(point, dtype=np.float64) - self.last_pitch))
        return distance <= self._motion_gate(frame_index, self.last_frame)

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
            if distance > gate:
                continue
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
                near = self._best_visible(visible, anchor, 2.5, min_confidence=0.18)
                if near is None or not self._reachable(np.asarray(near["pitch"]), frame_index):
                    pixel = pitch_to_pixel(homography, anchor[None, :], self.pitch)[0]
                    candidate = self._as_candidate(anchor, pixel, 0.82, "atd")
                    return self._commit(candidate, frame, frame_index, source=3)
                candidate = near
                candidate["origin"] = "yolo+atd"
                return self._commit(candidate, frame, frame_index, source=0)

            near = self._best_visible(visible, anchor, 2.5, min_confidence=0.06)
            if near is not None and not self._reachable(np.asarray(near["pitch"]), frame_index):
                near = None
            if near is None:
                # A confident image measurement corrects a sloppy ATD anchor: the anchor itself
                # can lag the ball by metres (interpolated ATD), so let strong detections in.
                near = self._best_visible(visible, anchor, 5.0, min_confidence=0.35)
            if near is None:
                pixel = pitch_to_pixel(homography, anchor[None, :], self.pitch)[0]
                candidate = self._as_candidate(anchor, pixel, 0.78, "atd")
                return self._commit(candidate, frame, frame_index, source=3)
            # ``_best_visible`` already restricted the candidate to the 2.5 m around the anchor, so
            # this is the "image measurement agrees with the ATD" case and the anchor does not need
            # to override it again.
            near["origin"] = "yolo+atd"
            return self._commit(near, frame, frame_index, source=0)

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
                    gate = self._motion_gate(frame_index, self.last_frame)
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
                                if distance > gate:
                                    continue
                                score = distance - 0.8 * float(blob.get("confidence", 0.0))
                                if score < best_distance:
                                    best_blob, best_distance = blob, score
                            candidate = best_blob
                    if candidate is not None and not self._reachable(
                        np.asarray(candidate["pitch"]), frame_index
                    ):
                        # The prediction may be ahead of the last accepted point; the candidate
                        # still has to be physically reachable from that point, not only from the
                        # predicted position.
                        candidate = None
                    if candidate is not None:
                        return self._commit(candidate, frame, frame_index, source=0)
                    point = predicted
                    pixel = pitch_to_pixel(homography, point[None, :], self.pitch)[0]
                    self.missing += 1
                    # Decay from the confidence at the last *observed* ball, by the age of that
                    # observation.  Multiplying the previous predicted confidence instead compounds
                    # the factor every frame while ``gap`` also grows, so the exponent goes
                    # quadratic and the tracker gives up long before ``max_gap_s``.
                    self.confidence = max(0.03, self.anchor_confidence * math.exp(-gap / 1.2))
                    return {
                        "pixel": [float(pixel[0]), float(pixel[1])]
                        if np.isfinite(pixel).all()
                        else [float("nan")] * 2,
                        "pitch": [float(point[0]), float(point[1])],
                        "confidence": self.confidence,
                        "source": 1,
                        "origin": "prediction",
                    }

        # No anchor and no useful prediction: reacquire from a confident YOLO ball.  A new track
        # needs two mutually reachable detections; a single false positive in the crowd or on an
        # advertising board must not become the ball.
        visible.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        if (
            self.pending is not None
            and (frame_index - self.pending_frame) / FPS > BALL_TRACK_MAX_GAP_S
        ):
            self.pending = None
            self.pending_frame = -1
        seed = next(
            (item for item in visible if float(item.get("confidence", 0.0)) >= BALL_MIN_CONFIDENCE),
            None,
        )
        if seed is None:
            self.alive = False
            self.confidence = 0.0
            return {
                "pixel": [float("nan"), float("nan")],
                "pitch": [float("nan"), float("nan")],
                "confidence": 0.0,
                "source": 2,
                "origin": "lost",
            }
        if self.pending is None:
            self.pending = dict(seed)
            self.pending_frame = int(frame_index)
        else:
            gate = self._motion_gate(frame_index, self.pending_frame)
            pending_point = np.asarray(self.pending["pitch"], dtype=np.float64)
            confirmed = self._best_visible(
                visible,
                pending_point,
                gate,
                min_confidence=BALL_MIN_CONFIDENCE,
            )
            if confirmed is None:
                self.pending = dict(seed)
                self.pending_frame = int(frame_index)
            else:
                self.pending = None
                self.pending_frame = -1
                confirmed["origin"] = "yolo-reacquire"
                return self._commit(confirmed, frame, frame_index, source=0)

        self.alive = False
        self.confidence = 0.0
        return {
            "pixel": [float("nan"), float("nan")],
            "pitch": [float("nan"), float("nan")],
            "confidence": 0.0,
            "source": 2,
            "origin": "pending-reacquire",
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
        players_official=np.zeros((samples, MAX_PLAYERS), dtype=bool),
        players_class=np.zeros((samples, MAX_PLAYERS), dtype=np.int8),
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
    except ModuleNotFoundError:
        # The farm bundle deliberately avoids pandas/scipy.  A greedy one-to-one match is not
        # globally optimal, but it is identical in practice for these ~40x46 pixel distances.
        rows: list[int] = []
        cols: list[int] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for flat in np.argsort(costs, axis=None):
            row, col = (int(value) for value in np.unravel_index(int(flat), costs.shape))
            if row in used_rows or col in used_cols:
                continue
            rows.append(row)
            cols.append(col)
            used_rows.add(row)
            used_cols.add(col)
            if len(rows) >= min(costs.shape):
                break
        rows = np.asarray(rows, dtype=np.intp)
        cols = np.asarray(cols, dtype=np.intp)
    else:
        rows, cols = linear_sum_assignment(costs)
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
    players_model: str | None = None,
    ball_model: str | None = None,
    players_imgsz: int | None = None,
    ball_imgsz: int = 1280,
    device: str | int | None = None,
    imgsz: int = 1920,
    player_conf: float = 0.12,
    ball_conf: float = 0.05,
    max_players: int = MAX_PLAYERS,
    max_ball_detections: int = MAX_BALL_DETECTIONS,
    seed_from_atd: bool = True,
    stage: str = "both",
    progress: bool = True,
) -> VideoPerception:
    """Run the video perception on a frame range and return a compact sample array.

    ``step`` is useful for a full-match pass: 5 gives 5 fps while preserving the video's 25 fps
    timeline.  The ball tracker uses the real frame gap, so it still predicts across skipped
    frames.  For the public evaluation window, step 1 is practical.

    ``stage`` runs one detector family at a time.  ``"ball"`` and ``"players"`` leave the other
    family's arrays empty, which lets a farm run the dedicated ball model first and the player
    model second without paying for both on every frame.  ``"both"`` keeps the original behaviour.
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
    stage = str(stage).lower()
    if stage not in {"both", "ball", "players"}:
        raise ValueError(f"stage must be 'both', 'ball' or 'players', got {stage!r}")
    run_players = stage in {"both", "players"}
    run_ball = stage in {"both", "ball"}
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
    player_weights = str(players_model or model) if run_players else None
    ball_weights = str(ball_model or model) if run_ball else None
    if stage == "players":
        model_label = f"players:{player_weights}"
    elif stage == "ball":
        model_label = f"ball:{ball_weights}"
    elif player_weights == ball_weights:
        model_label = str(player_weights)
    else:
        model_label = f"players:{player_weights} + ball:{ball_weights}"
    perception = _empty_perception(
        samples=n_samples,
        n_frames=n_frames,
        fps=FPS,
        step=step,
        start=start,
        stop=stop,
        model=model_label,
        device=device,
        video_width=width,
        video_height=height,
    )
    player_yolo = YOLO(player_weights) if run_players and player_weights else None
    if run_ball and ball_weights:
        ball_yolo = (
            player_yolo
            if player_yolo is not None and ball_weights == player_weights
            else YOLO(ball_weights)
        )
    else:
        ball_yolo = None
    shared_yolo = player_yolo if run_ball and player_yolo is ball_yolo else None
    shared_names = getattr(shared_yolo, "names", {}) or {}
    if shared_yolo is not None and not {PERSON_CLASS, BALL_CLASS}.issubset(shared_names):
        raise ValueError(
            "one shared model needs COCO person and ball classes; pass separate "
            "--players-model and --ball-model weights"
        )
    dedicated_players = run_players and players_model is not None
    ball_names = getattr(ball_yolo, "names", {}) or {}
    dedicated_ball = (
        run_ball
        and ball_yolo is not None
        and len(ball_names) == 1
        and str(ball_names.get(0, "")).lower() in {"ball", "sports ball"}
    )
    perception.meta.update(
        {
            "stage": stage,
            "players_model": player_weights,
            "ball_model": ball_weights,
        }
    )
    tracker = _GreedyTracker()
    ball_tracker = VideoBallTracker()
    sample = 0
    current = start
    atd_players = getattr(tracking, "players", None) if tracking is not None else None
    atd_teams = getattr(tracking, "teams", None) if tracking is not None else None
    keepers = None
    if run_players and dedicated_players and atd_players is not None and atd_teams is not None:
        # Keeper roster slots (and the goal each defends, per half) settle team colours for
        # goalkeepers even while the ATD feed swaps or freezes those slots under pressure.
        from .keepers import derive_keeper_slots, goal_side, half_index

        keepers = derive_keeper_slots(
            np.asarray(atd_players, dtype=np.float64),
            np.asarray(atd_teams),
            PITCH,
            GRADED_HALVES,
        )
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
            boxes = np.zeros((0, 4), dtype=np.float32)
            confidence = np.zeros(0, dtype=np.float32)
            classes = np.zeros(0, dtype=np.int16)
            raw_player_classes: np.ndarray | None = None
            if shared_yolo is not None:
                result = shared_yolo.predict(
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
            else:
                if player_yolo is not None:
                    result = player_yolo.predict(
                        frame,
                        classes=None if dedicated_players else [PERSON_CLASS],
                        conf=player_conf,
                        iou=0.5,
                        imgsz=int(players_imgsz or imgsz),
                        device=device,
                        verbose=False,
                        max_det=120,
                    )[0]
                    boxes, confidence, classes = _parse_boxes(result)
                    if dedicated_players and classes.size:
                        # The dedicated player model classes (0 player, 1 goalkeeper, 2 referee)
                        # are kept aside; the tracker, fusion and viewer keep seeing COCO person.
                        raw_player_classes = classes.copy()
                        classes = np.full(classes.shape, PERSON_CLASS, dtype=np.int16)
                if ball_yolo is not None:
                    ball_result = ball_yolo.predict(
                        frame,
                        classes=None if dedicated_ball else [BALL_CLASS],
                        conf=ball_conf,
                        iou=0.5,
                        imgsz=int(ball_imgsz),
                        device=device,
                        verbose=False,
                        max_det=4 * max_ball_detections,
                    )[0]
                    ball_boxes, ball_scores, _ = _parse_boxes(ball_result)
                    if ball_boxes.size:
                        boxes = np.concatenate([boxes, ball_boxes])
                        confidence = np.concatenate([confidence, ball_scores])
                        classes = np.concatenate(
                            [classes, np.full(len(ball_boxes), BALL_CLASS, dtype=np.int16)]
                        )
            matrix = np.asarray(homography.matrices[current], dtype=np.float64)
            if run_players:
                person_mask = (classes == PERSON_CLASS) & (confidence >= player_conf)
                person_boxes = boxes[person_mask]
                person_conf = confidence[person_mask]

                # Players: reject crowd/staff by the foot point, then assign teams and track ids.
                player_pixel = (
                    _box_foot_points(person_boxes) if person_boxes.size else np.zeros((0, 2))
                )
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
                if raw_player_classes is not None:
                    # person rows come first, before the ball detections get concatenated in
                    player_class = raw_player_classes[person_mask[: len(raw_player_classes)]][
                        valid_players
                    ]
                else:
                    player_class = np.zeros(len(person_boxes), dtype=np.int16)
                is_referee = player_class == 2
                is_goalkeeper = player_class == 1
                teams = np.full(len(person_boxes), -1, dtype=np.int8)
                team_confidence = np.zeros(len(person_boxes), dtype=np.float32)
                match_rows = np.flatnonzero(~is_referee)
                if (
                    atd_players is not None
                    and atd_teams is not None
                    and current < len(atd_players)
                    and match_rows.size
                ):
                    # Officials match no roster slot: keeping them out lets every slot find its player.
                    sub_teams, sub_confidence = _team_match(
                        player_pixel[match_rows],
                        np.asarray(atd_players[current], dtype=np.float64),
                        np.asarray(atd_teams),
                        matrix,
                        perception.pitch,
                    )
                    teams[match_rows] = sub_teams
                    team_confidence[match_rows] = sub_confidence
                if keepers is not None:
                    half = half_index(current, GRADED_HALVES)
                    if half >= 0:
                        for keeper_row in np.flatnonzero(is_goalkeeper):
                            foot = player_pitch[keeper_row]
                            if np.isfinite(foot).all():
                                side = goal_side(float(foot[0]), float(perception.pitch[0]))
                                right = int(keepers.team[half, side])
                                if right >= 0:
                                    teams[keeper_row] = right
                                    team_confidence[keeper_row] = 1.0
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
                    perception.players_class[sample, slot] = player_class[source_row]
                    perception.players_official[sample, slot] = bool(is_referee[source_row])

            if run_ball:
                ball_mask = (classes == BALL_CLASS) & (confidence >= ball_conf)
                ball_boxes = boxes[ball_mask]
                ball_conf_values = confidence[ball_mask]

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
        players_official=(
            None if perception.players_official is None else perception.players_official[:samples]
        ),
        players_class=(
            None if perception.players_class is None else perception.players_class[:samples]
        ),
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


def merge_perception(
    players: VideoPerception,
    ball: VideoPerception,
) -> VideoPerception:
    """Combine a players-only cache and a ball-only cache taken over the same frame range.

    The two passes must have sampled exactly the same video frames.  Player geometry comes from
    ``players``, ball detections and the accepted ball track come from ``ball``; all unaffected
    arrays stay empty.  The resulting cache has the ordinary :class:`VideoPerception` layout, so
    the viewer and fusion need no special case.
    """
    if players.samples != ball.samples:
        raise ValueError(
            f"perception stages have different sample counts: {players.samples} != {ball.samples}"
        )
    if not np.array_equal(players.frame_index, ball.frame_index):
        raise ValueError("perception stages sampled different video frames")
    if tuple(players.pitch) != tuple(ball.pitch):
        raise ValueError(f"perception stages use different pitches: {players.pitch} != {ball.pitch}")
    meta = {**ball.meta, **players.meta}
    meta.update(
        {
            "stage": "both",
            "players_model": players.meta.get("players_model"),
            "ball_model": ball.meta.get("ball_model"),
            "stage_meta": {"players": players.meta, "ball": ball.meta},
        }
    )
    return VideoPerception(
        frame_index=players.frame_index,
        players_box=players.players_box,
        players_pixel=players.players_pixel,
        players_pitch=players.players_pitch,
        players_team=players.players_team,
        players_confidence=players.players_confidence,
        players_present=players.players_present,
        players_track=players.players_track,
        players_team_confidence=players.players_team_confidence,
        ball_pixel=ball.ball_pixel,
        ball_pitch=ball.ball_pitch,
        ball_confidence=ball.ball_confidence,
        ball_source=ball.ball_source,
        yolo_ball_pixel=ball.yolo_ball_pixel,
        yolo_ball_pitch=ball.yolo_ball_pitch,
        yolo_ball_confidence=ball.yolo_ball_confidence,
        yolo_ball_present=ball.yolo_ball_present,
        n_frames=players.n_frames,
        fps=players.fps,
        step=players.step,
        start=players.start,
        stop=players.stop,
        model=f"{players.model} + {ball.model}",
        device=players.device,
        video_width=players.video_width,
        video_height=players.video_height,
        pitch=players.pitch,
        meta=meta,
        players_official=players.players_official,
        players_class=players.players_class,
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
        official = archive["players_official"].astype(bool) if "players_official" in archive.files else None
        classes = archive["players_class"].astype(np.int8) if "players_class" in archive.files else None
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
        players_official=official,
        players_class=classes,
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


def relabel_teams_from_tracking(
    players_pitch: np.ndarray,
    present: np.ndarray,
    teams: np.ndarray,
    track_xy: np.ndarray,
    track_teams: np.ndarray,
    track_present: np.ndarray,
    pitch_size: tuple[float, float],
    max_match_m: float = 2.0,
) -> np.ndarray:
    """Adopt the tracking file's roster team for every detection it recognises.

    The jersey classifier loses the teams when the light fails (measured 155 945:
    every visible player labelled Barcelona, eight more unknown, while the camera
    plainly showed both kits).  The tactical tracking file carries each slot's
    roster team — an identity, not a guess — and its pitch positions agree with the
    calibrated detections to well under a metre.  Each present detection takes the
    team of the nearest present tracking slot within ``max_match_m`` metres; slots
    outside the pitch (ghosts, throw-in corridors) are never a match, and a
    detection nobody recognises keeps its own label.
    """
    p = np.asarray(players_pitch, dtype=np.float64)
    pres = np.asarray(present, dtype=bool)
    out_teams = np.asarray(teams).copy()
    tk = np.asarray(track_xy, dtype=np.float64)
    tk_t = np.asarray(track_teams)
    tk_p = np.asarray(track_present, dtype=bool)
    length, width = float(pitch_size[0]), float(pitch_size[1])
    n = min(p.shape[0], tk.shape[0])
    with np.errstate(invalid="ignore"):
        inside = (
            np.isfinite(tk[:n]).all(axis=2)
            & (tk[:n, :, 0] > -1.0) & (tk[:n, :, 0] < length + 1.0)
            & (tk[:n, :, 1] > -1.0) & (tk[:n, :, 1] < width + 1.0)
        )
    slot_ok = tk_p[:n] & inside & np.isin(tk_t, (0, 1))[None, :]
    for f in range(n):
        det = np.flatnonzero(pres[f] & np.isfinite(p[f]).all(axis=1))
        if det.size == 0 or not slot_ok[f].any():
            continue
        slots = np.flatnonzero(slot_ok[f])
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(p[f][det, None, :] - tk[f][slots][None], axis=2)
        nearest = d.argmin(axis=1)
        close = d[np.arange(det.size), nearest] <= max_match_m
        if close.any():
            out_teams[f, det[close]] = tk_t[slots][nearest[close]]
    return out_teams
