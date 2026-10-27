#!/usr/bin/env python3
"""Score a ball detector on a YOLO split with ball-specific metrics.

Ultralytics' ``val`` reports mAP over a confidence sweep; what matters for the tactical
tracker is different: at the confidence floor the pipeline actually uses, how many balls
are found, how many false positives come with them, and how far off the centre is in
pixels (that error is what the homography turns into metres).

The model runs once at ``--sweep-conf`` and every threshold above it is scored from the
same detections, so the whole sweep costs one inference pass.

Run (COCO baseline):
    .venv/bin/python ball_training/eval_detector.py --weights yolo11x.pt --coco-ball --imgsz 1920
Run (trained ball model):
    .venv/bin/python ball_training/eval_detector.py --weights runs/ball/best.pt --imgsz 1280
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COCO_BALL = 32


def load_ground_truth(label_path: Path, w: int, h: int) -> np.ndarray:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if int(parts[0]) != 0:
            continue
        _cls, cx, cy, bw, bh = [float(v) for v in parts[:5]]
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None)
    inter_h = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def score_frame(pred: np.ndarray, pred_conf: np.ndarray, gt: np.ndarray, conf: float, iou_thr: float):
    """Greedy best-first matching at one confidence threshold. Returns (tp, fp, fn, matched_iou)."""
    keep = np.flatnonzero(pred_conf >= conf)
    preds = pred[keep]
    if preds.size == 0:
        return 0, 0, len(gt), []
    order = np.argsort(-pred_conf[keep])
    preds = preds[order]
    costs = iou_matrix(preds, gt)
    used = np.zeros(len(gt), dtype=bool)
    tp, matched = 0, []
    for row in range(len(preds)):
        if len(gt) == 0:
            break
        col = int(np.argmax(costs[row])) if costs[row].size else -1
        if col >= 0 and not used[col] and costs[row, col] >= iou_thr:
            used[col] = True
            tp += 1
            matched.append((preds[row], gt[col]))
    return tp, len(preds) - tp, int((~used).sum()), matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="ball_training/dataset")
    parser.add_argument("--subset", default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--coco-ball", action="store_true", help="weights are COCO: read class 32 as ball")
    parser.add_argument("--conf-thresholds", default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--sweep-conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--only", default=None, help="only stems with this prefix (e.g. mt__ = match frames)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    root = Path(args.split)
    images = sorted((root / "images" / args.subset).glob("*.jpg"))
    if args.only:
        images = [p for p in images if p.name.startswith(args.only)]
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images under {root / 'images' / args.subset}")

    model = YOLO(args.weights)
    classes = [COCO_BALL] if args.coco_ball else None
    thresholds = [float(t) for t in args.conf_thresholds.split(",")]
    totals = {c: {"tp": 0, "fp": 0, "fn": 0} for c in thresholds}
    errors_px: dict[float, list[float]] = {c: [] for c in thresholds}
    errors_norm: dict[float, list[float]] = {c: [] for c in thresholds}
    frames_with_gt = 0
    frames_with_zero_pred: dict[float, int] = {c: 0 for c in thresholds}

    for start in range(0, len(images), args.batch):
        chunk = images[start : start + args.batch]
        frames = []
        for path in chunk:
            frame = cv2.imread(str(path))
            if frame is None:
                raise SystemExit(f"unreadable image: {path}")
            frames.append(frame)
        results = model.predict(
            frames,
            imgsz=args.imgsz,
            conf=args.sweep_conf,
            iou=0.5,
            classes=classes,
            device=args.device,
            verbose=False,
            max_det=300,
        )
        for path, frame, result in zip(chunk, frames, results):
            h, w = frame.shape[:2]
            gt = load_ground_truth(root / "labels" / args.subset / f"{path.stem}.txt", w, h)
            if gt.size:
                frames_with_gt += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                pred = np.zeros((0, 4))
                pred_conf = np.zeros(0)
            else:
                cls = boxes.cls.detach().cpu().numpy().astype(int)
                mask = np.ones(len(cls), dtype=bool) if classes is None else (cls == classes[0])
                pred = boxes.xyxy.detach().cpu().numpy().astype(np.float64)[mask]
                pred_conf = boxes.conf.detach().cpu().numpy().astype(np.float64)[mask]
            for conf in thresholds:
                tp, fp, fn, matched = score_frame(pred, pred_conf, gt, conf, args.iou)
                totals[conf]["tp"] += tp
                totals[conf]["fp"] += fp
                totals[conf]["fn"] += fn
                if gt.size and tp == 0:
                    frames_with_zero_pred[conf] += 1
                for p_box, g_box in matched:
                    pc = (p_box[:2] + p_box[2:]) / 2.0
                    gc = (g_box[:2] + g_box[2:]) / 2.0
                    errors_px[conf].append(float(np.hypot(*(pc - gc))))
                    errors_norm[conf].append(float(np.hypot(*(pc - gc)) / np.hypot(w, h) * 100.0))

    report = {
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "images": len(images),
        "images_with_gt": frames_with_gt,
        "iou": args.iou,
        "per_threshold": {},
    }
    for conf in thresholds:
        tp, fp, fn = totals[conf]["tp"], totals[conf]["fp"], totals[conf]["fn"]
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        err = np.asarray(errors_px[conf]) if errors_px[conf] else np.zeros(0)
        report["per_threshold"][f"{conf:g}"] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(2 * recall * precision / max(1e-9, recall + precision), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "frames_with_no_detection": frames_with_zero_pred[conf],
            "centre_px_median": round(float(np.median(err)), 2) if err.size else None,
            "centre_px_p90": round(float(np.percentile(err, 90)), 2) if err.size else None,
        }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
