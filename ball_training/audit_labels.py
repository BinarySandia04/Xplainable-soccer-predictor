#!/usr/bin/env python3
"""Audit YOLO ball labels numerically -- no image viewing.

A ball label is trustworthy when the pixels inside the box really are the small bright,
low-saturation object the ball is, and clearly brighter than the grass immediately around
it.  For every label this compares the box against an annulus of the same area wrapped
around it and reports how many labels pass:

* ``contrast`` = mean value (V) inside box minus mean value of the ring.
* ``white_frac`` = share of pixels inside the box that are bright and unsaturated.

A label with ``contrast`` below ``--min-contrast`` is treated as suspicious; the script
prints percentiles over the whole split so a bad source shows up immediately, plus the
label-size distribution in pixels, which is what decides the training image size.

Run:  .venv/bin/python ball_training/audit_labels.py --images DIR --labels DIR [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def box_and_ring(frame, cx: float, cy: float, w: float, h: float):
    """Pixels inside the label box, and a same-area ring just outside it."""
    H, W = frame.shape[:2]
    side = max(w, h)
    x0 = int(max(0, cx - w / 2))
    x1 = int(min(W, cx + w / 2))
    y0 = int(max(0, cy - h / 2))
    y1 = int(min(H, cy + h / 2))
    rx0 = int(max(0, cx - side))
    rx1 = int(min(W, cx + side))
    ry0 = int(max(0, cy - side))
    ry1 = int(min(H, cy + side))
    if x1 <= x0 or y1 <= y0 or rx1 <= rx0 or ry1 <= ry0:
        return None, None
    inner = frame[y0:y1, x0:x1]
    ring = frame[ry0:ry1, rx0:rx1].copy()
    ry0i, ry1i = max(0, y0 - ry0), min(ring.shape[0], y1 - ry0)
    rx0i, rx1i = max(0, x0 - rx0), min(ring.shape[1], x1 - rx0)
    ring[ry0i:ry1i, rx0i:rx1i] = 0
    if ring.size == 0 or inner.size == 0:
        return None, None
    return inner, ring


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--min-contrast", type=float, default=12.0)
    parser.add_argument("--only-class", type=int, default=None, help="keep labels of this class id only")
    args = parser.parse_args()

    import cv2

    images = sorted(Path(args.images).glob("*.jpg"))[: args.limit]
    label_dir = Path(args.labels)
    contrast, white_frac, sides, off_pitch, missing = [], [], [], 0, 0
    for image_path in images:
        label = label_dir / f"{image_path.stem}.txt"
        if not label.is_file():
            missing += 1
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            missing += 1
            continue
        H, W = frame.shape[:2]
        rows = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.only_class is not None:
            rows = [r for r in rows if int(r[0]) == args.only_class]
        if len(rows) > 1:
            off_pitch += 1
        for _cls, cxn, cyn, wn, hn in rows:
            cx, cy = float(cxn) * W, float(cyn) * H
            w, h = float(wn) * W, float(hn) * H
            inner, ring = box_and_ring(frame, cx, cy, w, h)
            if inner is None:
                continue
            iv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
            vi, si = iv[:, :, 2].astype(np.float32), iv[:, :, 1].astype(np.float32)
            vring = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
            vring = vring[vring > 0]
            if vring.size == 0:
                continue
            contrast.append(float(vi.mean() - vring.mean()))
            white_frac.append(float(((vi > 150) & (si < 90)).mean()))
            sides.append(float(max(w, h)))
    contrast = np.asarray(contrast)
    white_frac = np.asarray(white_frac)
    sides = np.asarray(sides)
    if contrast.size == 0:
        print("audit_labels: nothing to audit")
        return 1
    print(f"audit_labels: {args.labels}  labels={contrast.size} images={len(images)} missing={missing}")
    print(f"  multi-box images: {off_pitch}")
    print("  contrast(box - ring, V): "
          f"median {np.median(contrast):.1f} p10 {np.percentile(contrast, 10):.1f} "
          f"p90 {np.percentile(contrast, 90):.1f}; below {args.min_contrast:.0f}: "
          f"{int((contrast < args.min_contrast).sum())} ({100.0 * (contrast < args.min_contrast).mean():.1f}%)")
    print("  white fraction in box:   "
          f"median {np.median(white_frac):.2f} p10 {np.percentile(white_frac, 10):.2f} "
          f"p90 {np.percentile(white_frac, 90):.2f}")
    print("  box side px:             "
          f"median {np.median(sides):.1f} p5 {np.percentile(sides, 5):.1f} "
          f"p95 {np.percentile(sides, 95):.1f} max {sides.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
