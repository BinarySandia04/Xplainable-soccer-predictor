#!/usr/bin/env python3
"""Auto-label the tactical match video for soccer-ball detection.

The ATD gives a ball position in pitch metres whenever the ball is observed, and the
per-frame homography maps it to pixels.  That projection is the anchor; a compact
white-blob search snaps the label onto the visible ball and rejects frames where no
plausible blob sits at the projected point (occluded, in a player's legs, off view).
Neighbour frames are checked for the same blob so a one-frame projection glitch cannot
create a label.

Frames inside the public evaluation window (plus a guard band) are deliberately NOT
labelled: they stay an untouched hold-out for scoring the trained model.

The video is decoded in one forward pass (``grab``/``retrieve`` free when the blob probe
is cheap) and neighbour confirmation is resolved three frames later from a small cache,
so no random seeking happens.

Output layout (YOLO, one class ``ball``)::

    ball_training/match/images/<stem>.jpg
    ball_training/match/labels/<stem>.txt
    ball_training/match/stats.json

Run:  .venv/bin/python ball_training/label_match.py [--stride 25]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from possession_lab.config import PITCH, VIDEO_PATH  # noqa: E402
from possession_lab.perception import _inside_pitch, pitch_to_pixel  # noqa: E402
from possession_lab.pipeline import load_lab  # noqa: E402

PUBLIC_START, PUBLIC_STOP = 11_185, 18_685
GUARD_FRAMES = 250  # keep a margin around the evaluation window completely untouched

BLOB_MIN_AREA = 4
BLOB_MAX_AREA = 400
BLOB_MAX_SIDE = 30
BLOB_MAX_OFFSET = 10.0     # px: blob centroid must sit this close to the projection
BLOB_WINDOW = 24           # px: half-size of the crop searched around the projection
NEIGHBOR_OFFSET = 3        # frames
BOX_INFLATE = 1.3
BOX_MIN_SIDE = 6.0
BOX_MAX_SIDE = 40.0
JPEG_QUALITY = 95


def snap_blob(frame, cx: float, cy: float):
    """Locate the small bright compact blob nearest (cx, cy). None when absent."""
    import cv2

    h, w = frame.shape[:2]
    x0, y0 = max(0, int(cx) - BLOB_WINDOW), max(0, int(cy) - BLOB_WINDOW)
    x1, y1 = min(w, int(cx) + BLOB_WINDOW + 1), min(h, int(cy) + BLOB_WINDOW + 1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    thr = max(145, int(np.percentile(v, 96)))
    mask = ((v >= thr) & (s <= 110)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    best_d = BLOB_MAX_OFFSET
    for label in range(1, count):
        _x, _y, bw, bh, area = [int(t) for t in stats[label]]
        if not (BLOB_MIN_AREA <= area <= BLOB_MAX_AREA):
            continue
        if bw < 1 or bh < 1 or bw > BLOB_MAX_SIDE or bh > BLOB_MAX_SIDE:
            continue
        gx, gy = x0 + float(centroids[label][0]), y0 + float(centroids[label][1])
        d = math.hypot(gx - cx, gy - cy)
        if d <= best_d:
            best = (_x + x0, _y + y0, bw, bh, d)
            best_d = d
    return best


def label_box(bbox) -> tuple[float, float, float, float]:
    """Inflated, clamped square box in pixel units (caller normalises)."""
    x, y, w, h, _d = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(BOX_MIN_SIDE, min(BOX_MAX_SIDE, max(w, h) * BOX_INFLATE))
    return cx, cy, side, side  # caller normalises with frame size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=8, help="frames between candidates")
    parser.add_argument("--out", default=str(Path(__file__).parent / "match"))
    parser.add_argument("--max-labels", type=int, default=8000)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    import cv2

    lab = load_lab(load_gt=False)
    tracking, homography = lab.tracking, lab.homography
    n_frames = tracking.n_frames
    ball, present = tracking.ball, tracking.ball_present

    candidate_set: set[int] = set()
    lo, hi = PUBLIC_START - GUARD_FRAMES, PUBLIC_STOP + GUARD_FRAMES
    for f in range(0, n_frames, max(1, args.stride)):
        if lo <= f <= hi:
            continue
        if bool(present[f]) and np.isfinite(ball[f]).all():
            candidate_set.add(f)
    print(f"label_match: {len(candidate_set)} ATD-ball candidate frames (eval window excluded)")
    if args.max_labels and len(candidate_set) > args.max_labels:
        # Spread the budget over the whole match instead of labelling the first minutes only.
        picks = np.linspace(0, len(candidate_set) - 1, int(args.max_labels)).round().astype(int)
        candidate_set = {sorted(candidate_set)[i] for i in picks}
        print(f"label_match: evenly spread to {len(candidate_set)} candidates for the label budget")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {VIDEO_PATH}")

    stats = {"candidates": len(candidate_set), "labelled": 0, "no_blob": 0, "bad_projection": 0,
             "neighbor_fail": 0, "negatives": 0}
    blob_cache: dict[int, tuple | None] = {}   # frame index -> probe result for that frame
    pending: dict[int, np.ndarray] = {}        # candidate frame awaiting its forward neighbour
    labelled = 0
    index = 0
    matrices = homography.matrices

    def probe(frame_idx: int, frame) -> tuple | None:
        """Blob at the ATD projection of ``frame_idx``, using the already decoded frame."""
        if not bool(present[frame_idx]):
            return None
        point = ball[frame_idx]
        if not _inside_pitch(point[None, :], PITCH, 1.2)[0]:
            stats["bad_projection"] += 1
            return None
        px = pitch_to_pixel(np.asarray(matrices[frame_idx]), point[None, :])[0]
        h, w = frame.shape[:2]
        if not np.isfinite(px).all() or not (30 <= px[0] < w - 30 and 30 <= px[1] < h - 30):
            stats["bad_projection"] += 1
            return None
        return snap_blob(frame, float(px[0]), float(px[1]))

    def confirm(candidate: int, forward_blob: tuple | None, frame) -> None:
        nonlocal labelled
        own_blob = blob_cache.get(candidate)
        back_blob = blob_cache.get(candidate - NEIGHBOR_OFFSET)
        if own_blob is None or forward_blob is None or back_blob is None:
            stats["neighbor_fail"] += 1
            return
        cx, cy, side, side2 = label_box(own_blob)
        h, w = frame.shape[:2]
        stem = f"m{candidate:07d}"
        if not cv2.imwrite(str(out / "images" / f"{stem}.jpg"), frame,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise RuntimeError("image write failed")
        (out / "labels" / f"{stem}.txt").write_text(
            f"0 {cx / w:.6f} {cy / h:.6f} {side / w:.6f} {side2 / h:.6f}\n", encoding="utf-8"
        )
        labelled += 1
        stats["labelled"] = labelled
        if labelled % 500 == 0:
            print(f"label_match: {labelled} labelled (frame {candidate:,}/{n_frames:,})", flush=True)

    try:
        while index < n_frames and labelled < args.max_labels:
            ok, frame = cap.read()
            if not ok:
                break
            blob = probe(index, frame)
            blob_cache[index] = blob

            back = index - NEIGHBOR_OFFSET
            if back in pending:
                confirm(back, blob, pending.pop(back))
            if index in candidate_set and labelled < args.max_labels:
                if blob is not None:
                    pending[index] = frame.copy()
                else:
                    stats["no_blob"] += 1

            stale = [k for k in blob_cache if k <= index - (2 * NEIGHBOR_OFFSET + 2)]
            for k in stale:
                del blob_cache[k]
            index += 1
    finally:
        cap.release()
        stats["scanned"] = index
        (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"label_match: done {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
