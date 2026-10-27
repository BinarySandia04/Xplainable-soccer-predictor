#!/usr/bin/env python3
"""Render the ball detector head-to-head onto a local video (no chat round-trip).

Draws the newly trained ball model (green) and the COCO baseline (red) on the tactical
feed and writes an MP4 plus a few highlight PNGs (frames where the new model finds the
ball and the baseline does not).  Everything stays on disk: open the MP4 with VLC or
xdg-open, so no image ever goes through the (non-vision) chat model.

Run:
    .venv/bin/python ball_training/visualize.py --start-sec 800 --seconds 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

COCO_BALL = 32
GREEN = (60, 220, 60)
RED = (0, 0, 230)
WHITE = (240, 240, 240)
BLACK = (0, 0, 0)


def detections(model, frame, classes, conf, imgsz, device):
    result = model.predict(frame, conf=conf, imgsz=imgsz, classes=classes, device=device, verbose=False, max_det=30)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    return boxes.xyxy.detach().cpu().numpy().astype(float), boxes.conf.detach().cpu().numpy().astype(float)


def draw(boxes, confs, frame, color, label):
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} {c:.2f}", (int(x1), max(12, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/tactical/BAR-ATH_tactical.mp4")
    parser.add_argument("--weights", default="ball_training/weights/ball_yolo11s_best.pt")
    parser.add_argument("--baseline", default="ball_training/weights/yolo11x.pt")
    parser.add_argument("--start-sec", type=float, default=800.0)
    parser.add_argument("--start-frame", type=int, default=None, help="exact first frame (overrides --start-sec)")
    parser.add_argument("--end-frame", type=int, default=None, help="exclusive last frame (overrides --seconds)")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz-new", type=int, default=1280)
    parser.add_argument("--imgsz-base", type=int, default=1920)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="ball_training/visualizations")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = args.start_frame if args.start_frame is not None else int(args.start_sec * fps)
    end = args.end_frame if args.end_frame is not None else start + int(args.seconds * fps)
    n_frames = end - start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    new_model = YOLO(args.weights)
    base_model = None if args.no_baseline else YOLO(args.baseline)
    out_path = out_dir / f"ball_viz_f{start}-{end}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    csv_path = out_path.with_suffix(".csv")
    csv_rows = []

    highlights = 0
    found_new = found_base = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        boxes_new, conf_new = detections(new_model, frame, None, args.conf, args.imgsz_new, args.device)
        found_new += len(boxes_new) > 0
        draw(boxes_new, conf_new, frame, GREEN, "ball_v2")
        top_new = (boxes_new[conf_new.argmax()] if len(boxes_new) else (np.nan,) * 4)
        top_new_conf = float(conf_new.max()) if len(conf_new) else 0.0
        top_base = (np.nan,) * 4
        top_base_conf = 0.0
        if base_model is not None:
            boxes_base, conf_base = detections(base_model, frame, [COCO_BALL], args.conf, args.imgsz_base, args.device)
            found_base += len(boxes_base) > 0
            draw(boxes_base, conf_base, frame, RED, "coco")
            top_base = boxes_base[conf_base.argmax()] if len(boxes_base) else top_base
            top_base_conf = float(conf_base.max()) if len(conf_base) else 0.0
            if len(boxes_new) and not len(boxes_base) and highlights < 5:
                png = out_dir / f"highlight_{start + i}.png"
                cv2.imwrite(str(png), frame)
                highlights += 1
        csv_rows.append((start + i, f"{top_new_conf:.4f}", *(f"{v:.1f}" for v in top_new),
                         f"{top_base_conf:.4f}", *(f"{v:.1f}" for v in top_base)))
        stamp = f"{(start + i) / fps:7.1f}s  frame {start + i}   green=new({len(boxes_new)})  red=coco"
        cv2.rectangle(frame, (0, 0), (w, 26), BLACK, -1)
        cv2.putText(frame, stamp, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{n_frames} frames", flush=True)
    cap.release()
    writer.release()
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("frame,new_conf,new_x1,new_y1,new_x2,new_y2,coco_conf,coco_x1,coco_y1,coco_x2,coco_y2\n")
        for row in csv_rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    print(f"wrote {out_path} ({w}x{h} @{fps:.0f}fps) and {csv_path}")
    print(f"frames with a detection -> new: {found_new}/{n_frames}, coco: {found_base}/{n_frames}")
    print(f"highlight PNGs saved: {highlights} in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
