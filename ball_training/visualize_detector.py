#!/usr/bin/env python3
"""Render a baseline-vs-trained ball detector comparison on the match video.

Each frame is decoded once and run through both models; the two panes are the *same
frame*, so every difference on screen is a difference in detections, not in timing.
Ball confidence is printed next to every box, and the header counts what each model
found at ``--conf``.

The default window starts on a frame that ``label_match.py`` positively labelled, so
the ball is guaranteed to be in view from frame one.

Run:
    .venv/bin/python ball_training/visualize_detector.py --start 39600 --frames 250
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COCO_BALL = 32
GREEN = (0, 220, 90)
RED = (0, 80, 255)
WHITE = (240, 240, 240)


def draw_boxes(frame: np.ndarray, boxes, confs, color) -> None:
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ball {c:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def header(frame: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        y = 34 + 26 * i
        cv2.rectangle(frame, (4, y - 22), (4 + 18 * len(line) + 10, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def predict(model, frame, imgsz: int, classes, conf: float):
    result = model.predict(frame, imgsz=imgsz, classes=classes, conf=conf, iou=0.5,
                           device="0", verbose=False, max_det=300)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    return (result.boxes.xyxy.detach().cpu().numpy(),
            result.boxes.conf.detach().cpu().numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(REPO / "data" / "tactical" / "BAR-ATH_tactical.mp4"))
    parser.add_argument("--baseline", default=str(HERE / "weights" / "yolo11x.pt"), help="COCO weights")
    parser.add_argument("--model", default=str(HERE / "weights" / "ball_yolo11s_best.pt"))
    parser.add_argument("--start", type=int, default=39600)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--baseline-imgsz", type=int, default=1920)
    parser.add_argument("--model-imgsz", type=int, default=1280)
    parser.add_argument("--pane-width", type=int, default=960)
    parser.add_argument("--out", default=None, help="output .mp4 (default: visualizations/compare_<start>.mp4)")
    parser.add_argument("--samples", type=int, default=4, help="also write N annotated PNGs (0 to skip)")
    args = parser.parse_args()

    from ultralytics import YOLO

    out = Path(args.out) if args.out else HERE / "visualizations" / f"compare_{args.start}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    baseline = YOLO(args.baseline)
    trained = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    scale = args.pane_width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (args.pane_width * 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)))
    t0, n = time.time(), 0
    sample_every = max(1, args.frames // max(1, args.samples)) if args.samples else 0
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        base_boxes, base_conf = predict(baseline, frame, args.baseline_imgsz, [COCO_BALL], args.conf)
        new_boxes, new_conf = predict(trained, frame, args.model_imgsz, None, args.conf)

        left = cv2.resize(frame.copy(), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(frame.copy(), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        draw_boxes(left, base_boxes * scale, base_conf, RED)
        draw_boxes(right, new_boxes * scale, new_conf, GREEN)
        header(left, [f"COCO yolo11x  frame {args.start + n}  balls: {len(base_boxes)}"])
        header(right, [f"ball_yolo11s  balls: {len(new_boxes)}"])
        canvas = np.hstack([left, right])
        writer.write(canvas)

        if args.samples and n % sample_every == 0:
            path = out.parent / f"sample_{args.start + n:07d}.png"
            cv2.imwrite(str(path), canvas)
        n += 1
        if n % 50 == 0:
            print(f"{n}/{args.frames} ({time.time() - t0:.0f}s)", flush=True)

    cap.release()
    writer.release()
    print(f"wrote {out} ({n} frames, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
