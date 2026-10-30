#!/usr/bin/env python3
"""Deep-train the best ball detector, end to end, on a GPU server.

One command takes you from an empty machine to ranked, evaluated weights:

    deps -> dataset downloads -> merged dataset -> long training -> val eval -> ranking

Everything is idempotent: each step is skipped when its output already exists, so re-running
the same command resumes an interrupted attempt.  Multi-GPU is automatic (Ultralytics DDP);
dataset prep and post-training ranking run only on the primary rank, so parallel ranks never
race on downloads.

The training recipe mirrors ``train_ball.py`` (tuned for a 6-15 px ball: no rotation/shear/
perspective, capped scale, fliplr only, mosaic with a close_mosaic tail) but runs a much deeper
schedule with cosine LR decay — the point of a server run is to reach the real plateau, not the
laptop one.  The model default is YOLO26 (Ultralytics' Jan-2026 family), whose ProgLoss + STAL
loss is specifically better at tiny objects — exactly this problem.

Run on the server from the repository root:

    python3 ball_training/train_server.py                          # yolo26x, 120 epochs
    python3 ball_training/train_server.py --models yolo26x,yolo26l --imgsz 1600
    python3 ball_training/train_server.py --dry-run                # show the plan only

Retrieval afterwards: copy ``ball_training/weights/ball_server_best.pt`` back and point the
pipeline at it (``--ball-model``); keep ``runs/<name>/results.png`` for the learning curves.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXTERNAL = HERE / "external"
WEIGHTS_DIR = HERE / "weights"
RUNS = HERE / "runs"
EVAL_DIR = HERE / "eval"

# (name, completion marker, command) - the marker path is what build_dataset.py reads.
SOURCES = (
    (
        "sn_h250 (SoccerNet v3 H250, CC-BY-4.0)",
        EXTERNAL / "sn_h250" / "YOLO",
        ["git", "clone", "https://github.com/kmouts/SoccerNet_v3_H250", str(EXTERNAL / "sn_h250")],
    ),
    (
        "infactory-ai/ball-tracking (CC-BY-NC-4.0)",
        EXTERNAL / "infactory_ball" / "data",
        ["hf-download", "infactory-ai/ball-tracking"],
    ),
    (
        "martinjolif/football-ball-detection (CC-BY-4.0)",
        EXTERNAL / "martinjolif" / "data",
        ["hf-download", "martinjolif/football-ball-detection"],
    ),
)

MIN_FREE_GB = 25


def log(msg: str) -> None:
    print(f"[train_server] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def is_ddp_child() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def is_primary_rank() -> bool:
    return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"


def sh(cmd: list[str]) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_deps(skip: bool) -> None:
    """Install what the pipeline needs (torch first so pip does not pick a CPU build)."""
    if skip:
        return
    try:
        import cv2  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        pass
    else:
        log("python deps already present")
        return
    log("installing python deps (this takes a few minutes)")
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    torch_index = os.environ.get("TORCH_PIP_INDEX", "https://download.pytorch.org/whl/cu128")
    sh([*pip, "torch==2.11.0", "torchvision==0.26.0", "--index-url", torch_index])
    sh([*pip, "ultralytics>=8.4.188", "opencv-python-headless", "scipy", "huggingface_hub"])
    sh([*pip, "-r", str(ROOT / "requirements-possession.txt")])


def check_disk() -> None:
    free_gb = shutil.disk_usage(HERE).free / 1e9
    if free_gb < MIN_FREE_GB:
        log(f"WARNING: only {free_gb:.0f} GB free on this filesystem; downloads need ~{MIN_FREE_GB} GB")
    else:
        log(f"disk ok: {free_gb:.0f} GB free")


def download_sources(dry: bool) -> None:
    for name, marker, cmd in SOURCES:
        if marker.exists():
            log(f"skip {name}: {marker.relative_to(HERE)} present")
            continue
        log(f"fetching {name}")
        if dry:
            log("  (dry-run) " + " ".join(cmd))
            continue
        try:
            if cmd[0] == "hf-download":
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id=cmd[1], repo_type="dataset", local_dir=str(marker.parent))
            else:
                sh(cmd)
        except Exception as exc:
            log(f"ERROR downloading {name}: {exc}")
            log("fix the download (HF_TOKEN for gated repos? network?) and rerun - finished sources are kept")
            raise SystemExit(2) from exc


def build_dataset(force: bool, dry: bool) -> None:
    data_yaml = HERE / "dataset" / "data.yaml"
    if data_yaml.exists() and not force:
        log("skip dataset build: dataset/data.yaml present (--force-build to redo)")
        return
    cmd = [sys.executable, str(HERE / "build_dataset.py")]
    if (HERE / "match").exists():
        cmd.append("--with-match")
    if dry:
        log("  (dry-run) " + " ".join(cmd))
        return
    sh(cmd)


def train_one(weights: str, args: argparse.Namespace) -> Path | None:
    """Run the deep recipe for one model; returns its run directory (rank 0 only result matters)."""
    import torch
    from ultralytics import YOLO

    stem = Path(weights).stem
    name = f"ball_{stem}_{args.imgsz}_srv"
    run_dir = RUNS / name
    if run_dir.exists() and any(run_dir.glob("args.yaml")) and not args.force_train:
        log(f"skip training {stem}: runs/{name} already started (--force-train to redo)")
        return run_dir

    n_gpu = torch.cuda.device_count()
    device = args.device or ("0" if n_gpu <= 1 else ",".join(str(i) for i in range(n_gpu)))
    batch = args.batch if args.batch > 0 else (-1 if n_gpu <= 1 else 8 * max(1, len(str(device).split(","))))
    log(f"training {stem}: imgsz={args.imgsz} epochs<={args.epochs} batch={batch} device={device}")

    model = YOLO(weights)
    model.train(
        data=str(HERE / "dataset" / "data.yaml"),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        workers=args.workers,
        device=device,
        fraction=args.fraction,
        patience=args.patience,
        project=str(RUNS),
        name=name,
        seed=0,
        val=True,
        plots=True,
        save_period=5,
        cos_lr=True,
        cache=args.cache,
        # --- tiny-object augmentation (mirrors train_ball.py) -----------------------------
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
        close_mosaic=args.close_mosaic,
        multi_scale=False,
    )
    return run_dir


def eval_run(run_dir: Path, imgsz: int, device: str, stem: str) -> dict | None:
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        log(f"no best.pt in {run_dir.name}")
        return None
    json_out = EVAL_DIR / f"server_{stem}.json"
    cmd = [
        sys.executable,
        str(HERE / "eval_detector.py"),
        "--weights", str(best),
        "--split", str(HERE / "dataset"),
        "--subset", "val",
        "--imgsz", str(imgsz),
        "--device", device.split(",")[0],
        "--json-out", str(json_out),
    ]
    log("$ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        log(f"eval failed for {run_dir.name}")
        return None
    report = json.loads(json_out.read_text(encoding="utf-8"))
    report["_best"] = str(best)
    return report


def resolve_models(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", item):
            raise SystemExit(f"invalid --models entry {item!r}")
        out.append(item if item.endswith(".pt") else f"{item}.pt")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="yolo26x", help="comma list: yolo26x,yolo26l,yolo11x or local .pt paths")
    parser.add_argument("--epochs", type=int, default=120, help="hard cap; patience early-stops the plateau")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=1280, help="1600 if GPUs have >= 40 GB")
    parser.add_argument("--batch", type=int, default=0, help="total batch; 0 = auto (-1 single GPU, 8/GPU multi)")
    parser.add_argument("--device", default=None, help='e.g. "0" or "0,1,2,3" (default: all visible GPUs)')
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--cache", action="store_true", help="cache images in RAM (needs ~dataset size in GB)")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = parser.parse_args()

    models = resolve_models(args.models)
    t0 = time.time()

    if args.dry_run:
        log(f"plan: deps -> download {len(SOURCES)} sources -> build dataset -> train {models} "
            f"(epochs<={args.epochs}, imgsz={args.imgsz}, batch=auto) -> eval + ranking")
        return 0

    WEIGHTS_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)
    os.chdir(WEIGHTS_DIR)  # Ultralytics drops yolo26x.pt into the cwd; keep weights tidy

    # DDP children must not redo setup (the parent already did it) - go straight to train().
    if not is_ddp_child():
        ensure_deps(args.skip_deps)
        check_disk()
        if not args.skip_download:
            download_sources(dry=False)
        build_dataset(force=args.force_build, dry=False)

    run_dirs: list[tuple[str, Path]] = []
    for weights in models:
        run_dir = train_one(weights, args)
        if not is_primary_rank():
            return 0  # non-primary DDP ranks: training done, rank 0 handles the rest
        if run_dir is not None:
            run_dirs.append((Path(weights).stem, run_dir))

    if not is_primary_rank() or not run_dirs:
        return 0

    log("ranking candidates on the val split (ball-specific, IoU 0.5, F1 @ conf 0.2)")
    ranked: list[tuple[float, str, dict]] = []
    for stem, run_dir in run_dirs:
        report = eval_run(run_dir, args.imgsz, args.device or "0", stem)
        if report is None:
            continue
        f1 = report["per_threshold"].get("0.2", {}).get("f1", 0.0)
        ranked.append((f1, stem, report))
    if not ranked:
        log("nothing to rank (no best.pt found)")
        return 1

    ranked.sort(reverse=True)
    print()
    log("==== results (val split) ====")
    for f1, stem, report in ranked:
        t = report["per_threshold"]["0.2"]
        print(f"  {stem:12s} F1@0.2={f1:.3f}  recall={t['recall']:.3f}  precision={t['precision']:.3f}"
              f"  centre_px_median={t['centre_px_median']}")
    winner_f1, winner_stem, _ = ranked[0]
    winner_best = next(run for stem, run in run_dirs if stem == winner_stem) / "weights" / "best.pt"
    target = WEIGHTS_DIR / "ball_server_best.pt"
    shutil.copyfile(winner_best, target)
    hours = (time.time() - t0) / 3600
    print()
    log(f"winner: {winner_stem} (F1@0.2={winner_f1:.3f}) -> {target}")
    log("next on the laptop: --ball-model ball_training/weights/ball_server_best.pt "
        "(plus runs/*_srv/results.png for the curves, eval/server_*.json for the numbers)")
    log(f"done in {hours:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
