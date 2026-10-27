#!/usr/bin/env python3
"""Train the ball-only YOLO detector on the merged dataset.

Hyper-parameters are chosen for the actual problem: a 6-15 px ball in a 1920x1080 frame.
That rules out the library defaults, which are tuned for objects that fill a large part
of the image:

* ``scale`` is capped low: zooming out shrinks an already tiny ball below one stride of
  the P3 head, zooming in throws away the context that separates a ball from a boot.
* no rotation, shear or perspective: a broadcast camera does not roll, and resampling a
  10 px ball with a projective transform destroys it.
* ``fliplr`` only: the field is left-right symmetric, vertical flips create fake sports.
* ``mosaic`` stays on (it keeps each source tile near native scale, which is what buys
  the detector its small-object resolution) but the last epochs run without it.
* ``mixup``/``copy_paste`` off: blending a 10 px blob into the grass teaches nothing.

Run:  .venv/bin/python ball_training/train_ball.py --name ball_s1280 --epochs 60 --batch 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(HERE / "weights" / "yolo11s.pt"))
    parser.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    parser.add_argument("--name", default="ball_s1280")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        workers=int(args.workers),
        device=args.device,
        fraction=float(args.fraction),
        lr0=float(args.lr0),
        patience=int(args.patience),
        project=str(HERE / "runs"),
        name=args.name,
        seed=0,
        val=True,
        plots=True,
        save_period=10,
        # --- tiny-object augmentation ------------------------------------------------
        degrees=0.0,
        translate=0.2,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.2,
        mosaic=1.0,
        close_mosaic=15,
        multi_scale=False,
        cache=False,
        resume=args.resume,
    )
    print("train_ball: finished ->", Path(str(getattr(results, 'save_dir', HERE / 'runs' / args.name))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
