"""Adapt the sampled perception cache to per-frame voter inputs.

The heavy YOLO pass runs on a subset of frames; the voters want every frame.  This
adapter is the only place that knows about sampling density, staleness and the physical
speed gate applied to raw YOLO ball candidates, so the voters themselves read a plain
dict of full-length arrays and never think about cache geometry again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import FPS
from ..perception import select_plausible_ball_detections


def resample_perception(perception: Any | None, n_frames: int) -> dict[str, np.ndarray]:
    """Map a sampled video cache onto every video frame with an age-aware decay."""
    empty_players = {
        "player_pitch": np.full((n_frames, 40, 2), np.nan, dtype=np.float32),
        "player_team": np.full((n_frames, 40), -1, dtype=np.int8),
        "player_present": np.zeros((n_frames, 40), dtype=bool),
        "player_confidence": np.zeros((n_frames, 40), dtype=np.float32),
        "player_track": np.full((n_frames, 40), -1, dtype=np.int16),
        "ball_pitch": np.full((n_frames, 2), np.nan, dtype=np.float32),
        "ball_pixel": np.full((n_frames, 2), np.nan, dtype=np.float32),
        "ball_confidence": np.zeros(n_frames, dtype=np.float32),
        "ball_source": np.full(n_frames, 2, dtype=np.int8),
        "yolo_ball": np.full((n_frames, 6, 2), np.nan, dtype=np.float32),
        "yolo_ball_confidence": np.zeros((n_frames, 6), dtype=np.float32),
        "yolo_ball_present": np.zeros((n_frames, 6), dtype=bool),
        "age": np.zeros(n_frames, dtype=np.int32),
    }
    if perception is None or perception.samples == 0:
        return empty_players
    frames = np.arange(n_frames, dtype=np.int64)
    sample_frames = np.asarray(perception.frame_index, dtype=np.int64)
    if sample_frames.size == 1:
        index = np.zeros(n_frames, dtype=np.int64)
    else:
        pos = np.searchsorted(sample_frames, frames, side="left")
        pos = np.clip(pos, 0, sample_frames.size - 1)
        before = np.maximum(pos - 1, 0)
        after = pos
        choose_before = (frames - sample_frames[before]) <= (sample_frames[after] - frames)
        index = np.where(choose_before, before, after)
    index = np.clip(index, 0, sample_frames.size - 1)
    age = np.abs(frames - sample_frames[index]).astype(np.int32)
    # The perception cache is sampled (usually every frame on the public window, every fifth
    # frame for a full-match pass).  Do not let a stale detection live forever, and keep frames
    # outside a public-window cache at zero instead of exponentiating a six-digit age.
    max_age = max(2, int(round(0.30 * FPS)))
    fresh = age <= max_age
    decay = np.zeros(n_frames, dtype=np.float64)
    decay[fresh] = np.exp(-age[fresh].astype(np.float64) / max(2.0, 0.10 * FPS))
    player_present = np.asarray(perception.players_present[index], dtype=bool) & fresh[:, None]
    if perception.players_official is not None:
        # Referees are not players: they must never count toward "players near the ball".
        player_present &= ~np.asarray(perception.players_official[index], dtype=bool)
    player_confidence = np.asarray(perception.players_confidence[index], dtype=np.float32)
    player_confidence = (player_confidence * decay[:, None]).astype(np.float32)
    player_present &= player_confidence > 0.02
    ball_pitch = np.asarray(perception.ball_pitch[index], dtype=np.float32).copy()
    ball_pixel = np.asarray(perception.ball_pixel[index], dtype=np.float32).copy()
    ball_confidence = np.asarray(perception.ball_confidence[index], dtype=np.float32) * decay
    ball_source = np.asarray(perception.ball_source[index], dtype=np.int8).copy()
    # A stale sample must not leave a valid-looking ball behind: the ballot gate reads the pitch
    # point and confidence, not only the source code, so both have to be cleared together.
    ball_pitch[~fresh] = np.nan
    ball_pixel[~fresh] = np.nan
    ball_confidence[~fresh] = 0.0
    ball_source[~fresh] = 2
    ball_source[ball_confidence <= 0.02] = 2
    # YOLO emits several candidates per frame and the class is noisy.  Associate them causally in
    # pitch space before they can vote: a candidate unreachable from the last accepted point at
    # football speed is rejected, and a new track needs two mutually reachable detections.
    plausible_ball, plausible_confidence, _, _ = select_plausible_ball_detections(
        perception.yolo_ball_pitch,
        perception.yolo_ball_confidence,
        perception.yolo_ball_present,
        perception.frame_index,
        fps=FPS,
    )
    candidate_count = int(np.asarray(perception.yolo_ball_pitch).shape[1])
    yolo_ball = np.full((n_frames, candidate_count, 2), np.nan, dtype=np.float32)
    yolo_confidence = np.zeros((n_frames, candidate_count), dtype=np.float32)
    yolo_present = np.zeros((n_frames, candidate_count), dtype=bool)
    mapped_ball = np.asarray(plausible_ball[index], dtype=np.float32)
    mapped_confidence = np.asarray(plausible_confidence[index], dtype=np.float32) * decay
    mapped_fresh = fresh & np.isfinite(mapped_ball).all(axis=1) & (mapped_confidence > 0.01)
    yolo_ball[mapped_fresh, 0] = mapped_ball[mapped_fresh]
    yolo_confidence[mapped_fresh, 0] = mapped_confidence[mapped_fresh]
    yolo_present[mapped_fresh, 0] = True
    return {
        "player_pitch": np.asarray(perception.players_pitch[index], dtype=np.float32),
        "player_team": np.asarray(perception.players_team[index], dtype=np.int8),
        "player_present": player_present,
        "player_confidence": player_confidence,
        "player_track": np.asarray(perception.players_track[index], dtype=np.int16),
        "ball_pitch": ball_pitch,
        "ball_pixel": ball_pixel,
        "ball_confidence": ball_confidence.astype(np.float32),
        "ball_source": ball_source,
        "yolo_ball": yolo_ball,
        "yolo_ball_confidence": yolo_confidence,
        "yolo_ball_present": yolo_present,
        "age": age,
    }
