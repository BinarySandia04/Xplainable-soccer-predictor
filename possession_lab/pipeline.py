"""Inference and submission-writing helpers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CODE_TO_LABEL,
    CONTEXT,
    FPS,
    GRADED_HALVES,
    N_FRAMES,
    PITCH,
    PUBLIC_GT_PATH,
    PUBLIC_START,
    PUBLIC_STOP,
    VIDEO_PATH,
    D,
    X,
    asset_paths,
)
from .data import (
    Homography,
    TagTable,
    TrackingData,
    load_homography,
    load_public_gt,
    load_smart_tags,
    load_tracking,
)
from .evaluate import features_for_window
from .model import (
    DEFAULT_PARAMS,
    DecodeResult,
    ModelParams,
    decode,
    merge_isolated_runs,
)


@dataclass(frozen=True)
class LabData:
    tracking: TrackingData
    tags: TagTable
    homography: Homography
    gt: np.ndarray | None


def _require(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "the gated match bundle is not available:\n  "
            f"{joined}\n\nUnpack it under data/ or set MTAH_DATA_DIR. "
            "See DATA.md; do not commit the data."
        )


def load_lab(data_dir: str | Path | None = None, load_gt: bool = True) -> LabData:
    """Load the tactical ATD, smart tags, homography, and (optional) public GT."""
    paths = asset_paths(data_dir)
    _require(paths)
    tracking = load_tracking(paths["atd_raw"], paths["atd_meta"], pitch=PITCH, n_frames=N_FRAMES)
    tags = load_smart_tags(paths["smart_tagging"], n_frames=tracking.n_frames)
    homography = load_homography(paths["homography"])
    gt = load_public_gt(PUBLIC_GT_PATH) if load_gt and Path(PUBLIC_GT_PATH).is_file() else None
    return LabData(tracking=tracking, tags=tags, homography=homography, gt=gt)


def prepare_event_products(
    tracking: TrackingData,
    tags: TagTable,
    *,
    track: Any | None = None,
    whistle: Any | None = None,
    use_audio: bool = True,
    video_path: str | Path | None = None,
) -> tuple[Any, Any, Any]:
    """Run the ball tracker, whistle cache, and event detector once for a full match.

    The event products are deliberately full-match length.  Splitting the decode into two graded
    halves must not hide a goal or a stoppage that lies just outside a half's own window.
    """
    from .audio import load_or_extract
    from .balltrack import track_ball
    from .events import build_events
    from .features import extract_features

    if track is None:
        track = track_ball(tracking)
    video = Path(video_path) if video_path is not None else VIDEO_PATH
    if whistle is None and use_audio and tracking.n_frames == N_FRAMES and video.is_file():
        whistle = load_or_extract(video, tracking.n_frames)
    base = extract_features(tracking, tags, ball_track=track, whistle=whistle)
    events = build_events(tracking, tags, track=track, whistle=whistle, features=base)
    return track, whistle, events


def active_halves(
    tags: TagTable,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
) -> tuple[tuple[int, int], ...]:
    """Trim each recorded half to the ``Kick Off (Start)`` smart tag.

    ``GRADED_HALVES`` is the recording window, not necessarily the kickoff.  In BAR-ATH the
    second half opens at 110 808 but the organisation's XML puts ``SET PIECES · Kick Off
    (Start)`` at 113 660: the 1 900 frames in between are the warm-up, and the old window
    invented possessions there.  Pair each half with the closest start tag and use its start as
    the first live frame.  No tag (tests, another fixture) leaves the window untouched.
    """
    windows = tuple((int(a), int(b)) for a, b in halves)
    kickoffs = sorted(
        int(event.start_frame)
        for event in tags.events
        if event.code == "SET PIECES" and "Kick Off (Start)" in event.type
    )
    if not kickoffs:
        return windows
    trimmed: list[tuple[int, int]] = []
    for start, stop in windows:
        candidates = [k for k in kickoffs if start - 10 * FPS <= k < stop]
        if candidates:
            kick = min(candidates, key=lambda k: abs(k - start))
            start = max(start, kick)
        trimmed.append((start, stop))
    return tuple(trimmed)


def fill_match_breaks(
    codes: np.ndarray,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
) -> np.ndarray:
    """Write ``D`` into the gap between consecutive halves (halftime, warm-up).

    Leaving the gap as ``X`` lets spellF1 forward-fill the last first-half possession across the
    interval, and leaving it to the decoder invents warm-up possessions.  The match is not being
    played there, so the truthful and safe state is a hard ``D``.
    """
    out = np.asarray(codes).copy()
    ordered = sorted((int(a), int(b)) for a, b in halves)
    for (_, stop), (next_start, _) in itertools.pairwise(ordered):
        if stop < next_start:
            out[stop:next_start] = D
    return out


def predict_full(
    tracking: TrackingData,
    tags: TagTable,
    params: ModelParams = DEFAULT_PARAMS,
    halves: tuple[tuple[int, int], ...] | None = None,
    context: int = CONTEXT,
    *,
    use_events: bool = True,
    use_audio: bool = True,
    track: Any | None = None,
    whistle: Any | None = None,
    events: Any | None = None,
) -> np.ndarray:
    """Decode each active half with a context pad; leave the rest as X and breaks as D.

    Active halves start at the organisation's ``Kick Off (Start)`` tag, so a warm-up before the
    second-half kickoff never becomes a possession.  Confirmed goals and stoppages are applied as
    a hard ``D`` override *after* the Viterbi merge.  They are high-precision intervals: a whistle
    confirmed by sustained low player motion, a ball visibly beyond the line, or a goal whose
    trajectory reaches the goal mouth.  Ordinary possession is still decoded by the HMM.
    """
    n = tracking.n_frames
    if halves is None:
        halves = active_halves(tags)
    if use_events and events is None:
        track, whistle, events = prepare_event_products(
            tracking, tags, track=track, whistle=whistle, use_audio=use_audio
        )
    prediction = np.full(n, X, dtype=np.int8)
    for start, stop in halves:
        start, stop = int(start), int(stop)
        if not (0 <= start < stop <= n):
            raise ValueError(f"invalid half [{start}, {stop}) for {n} frames")
        lo = max(0, start - context)
        hi = min(n, stop + context)
        features = features_for_window(
            tracking,
            tags,
            public=(lo, hi),
            context=0,
            ball_track=track,
            whistle=whistle,
            events=events,
        )
        result = decode(features, params, with_confidence=False)
        codes = merge_isolated_runs(result.codes[start - lo : stop - lo])
        prediction[start:stop] = codes

    if events is not None:
        graded = np.zeros(n, dtype=bool)
        for start, stop in halves:
            graded[int(start) : int(stop)] = True
        prediction[np.asarray(events.dead_mask, dtype=bool) & graded] = D
    return fill_match_breaks(prediction, halves)


def decode_public(
    tracking: TrackingData,
    tags: TagTable,
    params: ModelParams = DEFAULT_PARAMS,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
    context: int = CONTEXT,
    *,
    use_events: bool = True,
    use_audio: bool = True,
    track: Any | None = None,
    whistle: Any | None = None,
    events: Any | None = None,
) -> tuple[Any, DecodeResult, int]:
    """Return public-window features, the decode result, and the context left offset."""
    from .events import apply_event_overrides

    lo = max(0, public[0] - context)
    hi = min(tracking.n_frames, public[1] + context)
    if use_events and events is None:
        track, whistle, events = prepare_event_products(
            tracking, tags, track=track, whistle=whistle, use_audio=use_audio
        )
    features = features_for_window(
        tracking,
        tags,
        public=public,
        context=context,
        ball_track=track,
        whistle=whistle,
        events=events,
    )
    result = decode(features, params)
    codes = merge_isolated_runs(result.codes)
    if events is not None:
        codes = apply_event_overrides(codes, events, start=lo, stop=hi)
    result = replace(result, codes=codes)
    return features, result, lo


def write_submission(pred: np.ndarray, path: str | Path) -> Path:
    """Write the exact ``frame,label`` CSV shape the challenge expects."""
    pred = np.asarray(pred, dtype=np.int8)
    labels = [CODE_TO_LABEL[int(code)] for code in pred]
    frame = np.arange(len(pred), dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": frame, "label": labels}).to_csv(path, index=False)
    return path


def validate_codes(pred: np.ndarray) -> dict[str, Any]:
    """A local preflight; the official validator remains the final word."""
    pred = np.asarray(pred)
    if pred.ndim != 1:
        raise ValueError("prediction must be one-dimensional")
    valid = {0, 1, 2, 3}
    unique = {int(value) for value in np.unique(pred)}
    unknown = sorted(unique - valid)
    counts = {
        label: int(np.sum(pred == code))
        for label, code in (("H", 0), ("A", 1), ("D", 2), ("X", 3))
    }
    runs = 0
    previous = None
    for value in pred:
        if value != previous:
            runs += 1
            previous = value
    return {
        "ok": not unknown and pred.size > 0,
        "n_rows": int(pred.size),
        "unknown": unknown,
        "counts": counts,
        "n_runs": runs,
    }


def validate_submission_file(path: str | Path) -> dict[str, Any]:
    from .data import load_submission

    codes = load_submission(path)
    report = validate_codes(codes)
    report["path"] = str(path)
    return report


def public_prediction(pred: np.ndarray, public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP)) -> np.ndarray:
    return np.asarray(pred)[public[0] : public[1]]
