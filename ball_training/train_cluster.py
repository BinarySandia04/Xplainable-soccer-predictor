#!/usr/bin/env python3
"""Train the best possible soccer-ball detector on a multi-GPU server.

ONE FILE - paste it with nano, run it. No repo, no git, no GitHub, no token.

    python3 train_cluster.py                    # full run
    python3 train_cluster.py --dry-run          # show the plan, change nothing
    python3 train_cluster.py --empty-check      # per-GPU free VRAM and exit
    python3 train_cluster.py --device 1,2,3     # pin GPUs (default: all with room)
    python3 train_cluster.py --model yolo26x    # force a model instead of auto-pick
    python3 train_cluster.py --resume           # continue an interrupted run

Pipeline, in order:
  1. checks python >= 3.10, installs every dependency (auto-creates
     ball_data/.venv when the system pip is externally managed)
  2. downloads the three best public ball datasets and merges them into one
     YOLO dataset (ball-only labels, clip / official / time-block splits, so
     neighbouring frames never straddle train and val):
       SoccerNet v3 H250            Zenodo   CC-BY-4.0       ~2.5 GB
       infactory-ai/ball-tracking   HF       CC-BY-NC-4.0    ~0.4 GB
       martinjolif/football-ball-  HF       CC-BY-4.0       ~0.2 GB
  3. measures the FREE VRAM on every GPU (a llama-server holding ~22 GB per
     card is respected and never touched), picks the largest YOLO26 model,
     resolution and batch that fits, and trains with DDP on every GPU with room
  4. scores best.pt with ball-specific metrics: recall / precision / F1 at the
     confidence floors the possession tracker uses, plus centre error in pixels
  5. runs an inference test on the real match video: coverage over the whole
     match, and by default an annotated mp4 of the same clip used on the laptop
     (frames 11197-11566 of BAR-ATH_tactical.mp4)

The match video is a gated challenge bundle, so copy it once from the laptop
(the script auto-detects any mp4 under ball_data/video/, or pass --video):

    scp data/tactical/BAR-ATH_tactical.mp4 user@server:~/ball_data/video/

Optional: copy pre-labelled home-match frames to ball_data/match/{images,labels}
(produced on the laptop by label_match.py) and they are merged in as the
deployment domain, exactly like the laptop run.

Outputs, all under ./ball_data/:
    weights/<name>.pt               stable copy of the best checkpoint
    runs/<name>/weights/best.pt     full training run (results.png, args.yaml)
    eval/<name>.json                ball-specific val metrics
    inference/                      annotated clip mp4 + coverage JSON

Run detached so an ssh drop does not kill it:

    setsid nohup python3 train_cluster.py > train.log 2>&1 &
    tail -f train.log
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------- paths
# Everything the script creates lives in one folder next to it (override with
# BALL_DATA_DIR=/scratch/me/ball_data if the home quota is small).
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BALL_DATA_DIR") or HERE / "ball_data").expanduser()
EXT = DATA / "external"
DATASET = DATA / "dataset"
RUNS = DATA / "runs"
EVAL = DATA / "eval"
INFER = DATA / "inference"
WEIGHTS_OUT = DATA / "weights"
VIDEO_DIR = DATA / "video"
MATCH = DATA / "match"           # optional: ball_data/match/{images,labels}

# ------------------------------------------------------------- dependencies
MIN_PY = (3, 10)
MIN_DISK_GB = 30
DEP_CHECK = "import torch, torchvision, ultralytics, cv2, scipy, numpy, huggingface_hub"
TORCH_INDEX = os.environ.get("TORCH_PIP_INDEX", "https://download.pytorch.org/whl/cu128")
PIP_TORCH = ("torch==2.11.0", "torchvision==0.26.0")
PIP_EXTRA = ("ultralytics>=8.4.188", "opencv-python-headless", "scipy",
             "huggingface_hub>=0.26", "nvidia-ml-py", "numpy>=1.26",
             "pyyaml", "matplotlib>=3.8")

# ---------------------------------------------------------------- datasets
SN_ZIP_URL = "https://zenodo.org/api/records/7808511/files/YOLO.zip/content"
SN_ZIP_NAME = "SoccerNet_v3_H250_YOLO.zip"
HF_SOURCES = (
    # (repo id, target folder under external/, marker proving it is there)
    ("infactory-ai/ball-tracking", "infactory_ball", "data/visible"),
    ("martinjolif/football-ball-detection", "martinjolif", "data/train/images"),
)
SN = EXT / "sn_h250" / "YOLO"
INFO = EXT / "infactory_ball" / "data"
MARTIN = EXT / "martinjolif" / "data"

# ------------------------------------------------------------------- model
MODEL_FACTOR = {"n": 0.45, "s": 0.6, "m": 1.0, "l": 1.3, "x": 1.7}
PER_SAMPLE_640 = 0.69          # GB per sample, "m" detector, AdamW state
LADDER = ["x", "l", "m", "s"]  # largest first
RESOLUTIONS = [1280, 1024, 896, 640]
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
CLIP_START, CLIP_END = 11197, 11567   # laptop reference clip, end exclusive

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def log(msg: str) -> None:
    print(f"[ball_train] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def sh(cmd: list[str]) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


# ------------------------------------------------------------------ setup
def bootstrap_python() -> None:
    if sys.version_info < MIN_PY:
        raise SystemExit(
            f"python {'.'.join(map(str, MIN_PY))}+ is required (torch/ultralytics), found "
            f"{sys.version.split()[0]}. Load a newer python (module/conda), then rerun."
        )


def deps_present(python: str) -> bool:
    return subprocess.run([python, "-c", DEP_CHECK], capture_output=True).returncode == 0


def install_into(python: str) -> None:
    pip = [python, "-m", "pip", "install", "-q"]
    sh([*pip, *PIP_TORCH, "--index-url", TORCH_INDEX])
    sh([*pip, *PIP_EXTRA])


def ensure_deps() -> None:
    if deps_present(sys.executable):
        return
    log("installing python deps (a few minutes)")
    try:
        install_into(sys.executable)
    except subprocess.CalledProcessError:
        if sys.prefix != sys.base_prefix:
            raise  # already in a venv/conda env: the failure is real, show it
        log("system python refuses pip (externally managed) - bootstrapping ball_data/.venv")
        DATA.mkdir(parents=True, exist_ok=True)
        venv = DATA / ".venv"
        sh([sys.executable, "-m", "venv", str(venv)])
        vpy = str(venv / "bin" / "python")
        sh([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"])
        install_into(vpy)
        log("re-executing this script inside the venv")
        os.execv(vpy, [vpy, *sys.argv])
    if not deps_present(sys.executable):
        raise SystemExit("deps installed but imports still fail - check the pip log above")


def gpu_free_gb() -> dict[int, float]:
    """Free VRAM per CUDA index via NVML (no CUDA context, safe next to llama-server)."""
    import torch

    n = torch.cuda.device_count()
    out: dict[int, float] = {}
    if not n:
        return out
    try:
        import pynvml

        pynvml.nvmlInit()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(_nvml_index(i))
            free, _ = pynvml.nvmlDeviceGetMemoryInfo(h)
            out[i] = free / 1024**3
        pynvml.nvmlShutdown()
        return out
    except Exception:
        for i in range(n):
            free, _ = torch.cuda.mem_get_info(i)
            out[i] = free / 1024**3
        return out


def _nvml_index(cuda_index: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        try:
            return int(visible.split(",")[cuda_index])
        except (ValueError, IndexError):
            return cuda_index
    return cuda_index


def pick_config(ids: list[int], free: dict[int, float], force: str | None) -> tuple[str, int, int]:
    """Return (model, imgsz, per-GPU batch): the biggest thing that fits."""
    budget = min(free[d] for d in ids) * 0.82
    sizes = [force] if force else [f"yolo26{s}" for s in LADDER]
    for model in sizes:
        factor = MODEL_FACTOR.get(model.removesuffix(".pt")[-1], 1.0)
        for imgsz in RESOLUTIONS:
            per_sample = PER_SAMPLE_640 * factor * (imgsz / 640) ** 2
            batch = int(budget / per_sample)
            if batch >= 2:
                return model, imgsz, max(2, min(batch, 16))
    model = sizes[-1]
    factor = MODEL_FACTOR.get(model.removesuffix(".pt")[-1], 1.0)
    return model, 640, max(1, int(budget / (PER_SAMPLE_640 * factor)))


def check_disk() -> None:
    free_gb = shutil.disk_usage(DATA if DATA.exists() else HERE).free / 1e9
    if free_gb < MIN_DISK_GB:
        log(f"WARNING: {free_gb:.0f} GB free - datasets + video + runs want ~{MIN_DISK_GB} GB")
    else:
        log(f"disk ok: {free_gb:.0f} GB free")


# --------------------------------------------------------------- downloads
def download(url: str, dest: Path, attempts: int = 4) -> None:
    """Streamed download with resume (Range) and progress lines."""
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.is_file() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "ball-trainer/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range: restart cleanly
                    have = 0
                mode = "ab" if have else "wb"
                total = int(r.headers.get("Content-Length") or 0) + have
                got, last = have, 0.0
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        now = time.time()
                        if now - last > 5 or got == total:
                            last = now
                            pct = 100 * got / total if total else 0
                            log(f"  {dest.name}: {got / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:.0f}%)")
            part.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                part.replace(dest)
                return
            if attempt == attempts:
                raise
        except Exception:
            if attempt == attempts:
                raise
            log(f"  download interrupted, retrying ({attempt}/{attempts})")
            time.sleep(5 * attempt)


def ensure_soccernet() -> None:
    if (SN / "train" / "labels").is_dir():
        log("skip SoccerNet H250: present")
        return
    zip_path = EXT / SN_ZIP_NAME
    log("fetching SoccerNet v3 H250 (Zenodo, 2.5 GB)")
    if not zip_path.is_file() or zip_path.stat().st_size < 2_000_000_000:
        download(SN_ZIP_URL, zip_path)
    log("extracting SoccerNet H250 (14k images)")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(EXT / "sn_h250")
    zip_path.unlink(missing_ok=True)
    if not (SN / "train" / "labels").is_dir():
        raise SystemExit(f"SoccerNet extraction looks wrong: {SN}")


def ensure_hf_sources() -> None:
    from huggingface_hub import snapshot_download

    for repo, subdir, marker in HF_SOURCES:
        target = EXT / subdir
        if (target / marker).exists():
            log(f"skip {repo}: present")
            continue
        log(f"fetching {repo} (Hugging Face)")
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(target),
                          max_workers=8, token=os.environ.get("HF_TOKEN") or None)


def ensure_datasets() -> None:
    ensure_soccernet()
    ensure_hf_sources()


# ----------------------------------------------------------- dataset build
def link_image(image: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(image.resolve())


def keep_class(label: Path, wanted: int = 0) -> list[str]:
    lines = []
    for line in label.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if int(parts[0]) == wanted:
            parts[0] = "0"
            lines.append(" ".join(parts))
    return lines


def sn_pairs(split: str) -> list[tuple[Path, Path]]:
    out = []
    for label in sorted((SN / split / "labels").glob("*.txt")):
        for suffix in IMAGE_SUFFIXES:
            image = SN / split / "images" / f"{label.stem}{suffix}"
            if image.is_file():
                out.append((image, label))
                break
    return out


def martin_pairs(split: str) -> list[tuple[Path, Path]]:
    out = []
    images = MARTIN / split / "images"
    if not images.is_dir():
        return out
    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = MARTIN / split / "labels" / f"{image.stem}.txt"
        if label.is_file():
            out.append((image, label))
    return out


def info_pairs() -> tuple[list[tuple[Path, list[str]]], list[tuple[Path, list[str]]]]:
    """infactory frames split by source clip (never by neighbouring frame)."""
    frames: list[tuple[str, Path, list[str]]] = []
    for kind, has_ball in (("visible", True), ("not_visible", False)):
        for image in sorted((INFO / kind).glob("*.jpg")):
            label = image.with_suffix(".txt")
            lines: list[str] = []
            if has_ball and label.is_file():
                lines = [ln for ln in keep_class(label)]
            frames.append((image.stem.split("_")[0], image, lines))
    clips = sorted({clip for clip, _i, _l in frames})
    random.Random(7).shuffle(clips)
    val_clips = set(clips[: max(1, len(clips) // 5)])
    train, val = [], []
    for clip, image, lines in frames:
        (val if clip in val_clips else train).append((image, lines))
    return train, val


def match_pairs(block_frames: int = 1500, block_mod: int = 7, block_rem: int = 3,
                guard: int = 50) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Home-match labels split by time block (never interleaved frames)."""
    images = sorted((MATCH / "images").glob("*.jpg")) if MATCH.is_dir() else []
    train, val, dropped = [], [], 0
    for image in images:
        label = MATCH / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            continue
        frame = int(image.stem.lstrip("m")) if image.stem.startswith("m") else 0
        block = frame // block_frames
        if block % block_mod == block_rem:
            val.append((image, label))
            continue
        if frame - block * block_frames < guard or (block + 1) * block_frames - frame < guard:
            dropped += 1
            continue
        train.append((image, label))
    if dropped:
        log(f"match labels: dropped {dropped} frames adjacent to a validation block")
    return train, val


def build_dataset(force: bool = False) -> None:
    data_yaml = DATASET / "data.yaml"
    if data_yaml.is_file() and not force:
        log(f"dataset already built: {DATASET}")
        return
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    counter: Counter = Counter()

    def emit(prefix: str, image: Path, lines: list[str], split: str) -> None:
        name = f"{prefix}__{image.stem}"
        link_image(image, DATASET / "images" / split / f"{name}{image.suffix.lower()}")
        (DATASET / "labels" / split / f"{name}.txt").write_text(
            "".join(f"{ln}\n" for ln in lines), encoding="utf-8")
        counter[f"{split}:{'pos' if lines else 'neg'}"] += 1

    log("merging SoccerNet H250 (ball boxes only)")
    positives, negatives = [], []
    for image, label in sn_pairs("train"):
        lines = keep_class(label)
        (positives if lines else negatives).append((image, lines))
    for image, lines in positives:
        emit("sn", image, lines, "train")
    random.Random(11).shuffle(negatives)
    for image, lines in negatives[:1500]:  # proven v2 recipe (unverified SN frames)
        emit("sn", image, lines, "train")
    sn_val = [(image, keep_class(label)) for image, label in sn_pairs("valid")]
    for image, lines in [pair for pair in sn_val if pair[1]][:800]:
        emit("sn", image, lines, "val")

    log("merging infactory ball-tracking (verified negatives included)")
    info_train, info_val = info_pairs()
    for image, lines in info_train:
        if lines:
            emit("info", image, lines, "train")
    for image, lines in [p for p in info_train if not p[1]][:2200]:
        emit("info", image, lines, "train")
    for image, lines in info_val:
        emit("info", image, lines, "val")

    log("merging martinjolif (official split)")
    for image, label in martin_pairs("train"):
        emit("mj", image, keep_class(label), "train")
    for image, label in martin_pairs("valid"):
        emit("mj", image, keep_class(label), "val")

    if MATCH.is_dir():
        log("merging home-match frames (deployment domain)")
        mt_train, mt_val = match_pairs()
        for image, label in mt_train:
            emit("mt", image, keep_class(label), "train")
        for image, label in mt_val:
            emit("mt", image, keep_class(label), "val")
    else:
        log("no ball_data/match - training on public sources only "
            "(copy pre-labelled match frames there to add the deployment domain)")

    (DATASET / "data.yaml").write_text(
        f"# Generated by train_cluster.py -- do not edit by hand.\n"
        f"path: {DATASET.resolve()}\n"
        f"train: images/train\nval: images/val\nnc: 1\nnames:\n  0: ball\n",
        encoding="utf-8")
    stats = dict(counter)
    stats["train_total"] = stats.get("train:pos", 0) + stats.get("train:neg", 0)
    stats["val_total"] = stats.get("val:pos", 0) + stats.get("val:neg", 0)
    (DATASET / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"dataset ready: {json.dumps(stats, sort_keys=True)}")


# ------------------------------------------------------------------- train
def train(model: str, imgsz: int, batch: int, device: str, epochs: int, patience: int,
          workers: int, name: str, n_gpus: int, resume: bool) -> Path:
    from ultralytics import YOLO

    run_dir = RUNS / name
    best = run_dir / "weights" / "best.pt"
    if best.is_file():
        log(f"skip training: {best} already exists")
        return run_dir
    last = run_dir / "weights" / "last.pt"
    if resume and last.is_file():
        log(f"resuming {last}")
        YOLO(str(last)).train(resume=True)
        return run_dir

    weights = model if model.endswith(".pt") else f"{model}.pt"
    workers = max(2, min(workers, (os.cpu_count() or 8) // max(1, n_gpus)))
    log(f"training {weights}: imgsz={imgsz} batch={batch}/gpu epochs<={epochs} "
        f"patience={patience} device={device} workers={workers}/gpu")
    try:
        net = YOLO(weights)
    except Exception as exc:
        raise SystemExit(
            f"could not load {weights}: {exc}\n"
            f"If this server cannot reach GitHub, upload {weights} next to this script "
            f"and rerun with --model ./{weights}"
        ) from exc
    net.train(
        data=str(DATASET / "data.yaml"),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=device,
        patience=patience,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        seed=0,
        val=True,
        plots=True,
        save_period=5,
        cos_lr=True,
        # tiny-object recipe: no rotation/perspective/shear, capped scale, fliplr only
        degrees=0.0, translate=0.2, scale=0.25, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        mixup=0.0, copy_paste=0.0, erasing=0.2,
        mosaic=1.0, close_mosaic=min(15, max(5, epochs // 8)), multi_scale=False,
    )
    return run_dir


# -------------------------------------------------------------------- eval
def _load_gt(label_path: Path, w: int, h: int):
    import numpy as np

    boxes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts or int(parts[0]) != 0:
                continue
            _c, cx, cy, bw, bh = [float(v) for v in parts[:5]]
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def _iou_matrix(a, b):
    import numpy as np

    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def _score_frame(pred, pred_conf, gt, conf: float, iou_thr: float = 0.5):
    import numpy as np

    keep = np.flatnonzero(pred_conf >= conf)
    preds = pred[keep]
    if preds.size == 0:
        return 0, 0, len(gt), []
    order = np.argsort(-pred_conf[keep])
    preds = preds[order]
    costs = _iou_matrix(preds, gt)
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


def evaluate(weights: Path, imgsz: int, device: str, name: str, batch: int = 8) -> dict | None:
    """Ball-specific val metrics (what the possession tracker actually needs)."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    images = sorted((DATASET / "images" / "val").glob("*.jpg"))
    if not images:
        log("no val images - skipping evaluation")
        return None
    model = YOLO(str(weights))
    totals = {c: {"tp": 0, "fp": 0, "fn": 0} for c in THRESHOLDS}
    errors = {c: [] for c in THRESHOLDS}
    frames_gt = 0
    frames_missed = {c: 0 for c in THRESHOLDS}

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        frames = [cv2.imread(str(p)) for p in chunk]
        results = model.predict(frames, imgsz=imgsz, conf=0.01, iou=0.5,
                                device=device, verbose=False, max_det=300)
        for path, frame, result in zip(chunk, frames, results):
            h, w = frame.shape[:2]
            gt = _load_gt(DATASET / "labels" / "val" / f"{path.stem}.txt", w, h)
            if gt.size:
                frames_gt += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                pred, confs = np.zeros((0, 4)), np.zeros(0)
            else:
                pred = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
                confs = boxes.conf.detach().cpu().numpy().astype(np.float64)
            for c in THRESHOLDS:
                tp, fp, fn, matched = _score_frame(pred, confs, gt, c)
                totals[c]["tp"] += tp
                totals[c]["fp"] += fp
                totals[c]["fn"] += fn
                if gt.size and tp == 0:
                    frames_missed[c] += 1
                for p_box, g_box in matched:
                    pc = (p_box[:2] + p_box[2:]) / 2.0
                    gc = (g_box[:2] + g_box[2:]) / 2.0
                    errors[c].append(float(np.hypot(*(pc - gc))))

    report = {"weights": str(weights), "imgsz": imgsz, "images": len(images),
              "images_with_gt": frames_gt, "iou": 0.5, "per_threshold": {}}
    for c in THRESHOLDS:
        tp, fp, fn = totals[c]["tp"], totals[c]["fp"], totals[c]["fn"]
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        err = np.asarray(errors[c]) if errors[c] else np.zeros(0)
        report["per_threshold"][f"{c:g}"] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(2 * recall * precision / max(1e-9, recall + precision), 4),
            "tp": tp, "fp": fp, "fn": fn,
            "frames_with_no_detection": frames_missed[c],
            "centre_px_median": round(float(np.median(err)), 2) if err.size else None,
            "centre_px_p90": round(float(np.percentile(err, 90)), 2) if err.size else None,
        }
    EVAL.mkdir(parents=True, exist_ok=True)
    out = EVAL / f"{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"eval -> {out}")
    return report


# --------------------------------------------------------------- inference
def find_video(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for pattern in ("*.mp4", "*.mkv", "*.avi", "*.mov"):
        hits = sorted(VIDEO_DIR.glob(pattern))
        if hits:
            return hits[0]
    return None


def _no_video_hint() -> None:
    log("no match video found under ball_data/video/ - inference test skipped")
    log("  copy it once from the laptop (it is a gated bundle, not downloadable):")
    log(f"    scp data/tactical/BAR-ATH_tactical.mp4 user@server:{VIDEO_DIR}/")
    log("  or rerun later with --video /path/to/BAR-ATH_tactical.mp4")


def run_inference(weights: Path, device: str, video: Path | None,
                  start_frame: int = 0, end_frame: int = 0, imgsz: int = 1280) -> None:
    """Ball coverage over the match (+ optional annotated clip mp4)."""
    import cv2
    from ultralytics import YOLO

    if video is None:
        video = find_video(None)
    if video is None or not video.is_file():
        _no_video_hint()
        return
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log(f"WARNING: cannot open {video} - inference skipped")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = 1 if end_frame > 0 else max(1, (total or 39000) // 600)
    clip_mode = end_frame > 0
    tag = f"clip_{start_frame}-{end_frame - 1}" if clip_mode else "full_match"
    out = INFER / f"{video.stem}_{tag}.mp4"
    writer = None
    n = hits = 0
    confs: list[float] = []
    idx = -1
    log(f"inference on {video.name} [{tag}]: every frame" if clip_mode
        else f"inference on {video.name}: ~600 frames sampled from {total or '?'}")

    while True:
        if not cap.grab():
            break
        idx += 1
        if clip_mode and idx < start_frame:
            continue
        if clip_mode and idx >= end_frame:
            break
        if idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        n += 1
        res = model.predict(frame, imgsz=imgsz, conf=0.10, device=device, verbose=False)[0]
        balls = [b for b in res.boxes if int(b.cls) == 0]
        if balls:
            hits += 1
            confs.append(float(balls[0].conf))
            for b in balls[:3]:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 14, (0, 255, 0), 2)
                cv2.putText(frame, f"ball {float(b.conf):.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if clip_mode:
            cv2.putText(frame, f"frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            if writer is None:
                INFER.mkdir(parents=True, exist_ok=True)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
        log(f"annotated clip -> {out}")

    INFER.mkdir(parents=True, exist_ok=True)
    summary = {
        "weights": str(weights), "video": str(video), "clip": tag,
        "frames_tested": n, "ball_found_frames": hits,
        "coverage": round(hits / n, 4) if n else 0.0,
        "mean_conf_of_hits": round(sum(confs) / len(confs), 4) if confs else None,
        "max_conf": round(max(confs), 4) if confs else None,
    }
    json_out = INFER / f"{video.stem}_{tag}.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"inference [{tag}]: ball found in {hits}/{n} frames "
        f"({summary['coverage']:.1%}), mean conf {summary['mean_conf_of_hits']} -> {json_out}")


# -------------------------------------------------------------------- main
def parse_device(spec: str) -> list[int]:
    return [int(x) for x in re.split(r"[,\s]+", spec.strip()) if x != ""]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None,
                   help='GPUs to use, e.g. "0,1,2,3" or "1,3" (default: all with room)')
    p.add_argument("--min-free-gb", type=float, default=8.0,
                   help="minimum free VRAM per GPU to consider it")
    p.add_argument("--model", default=None,
                   help="force a model stem (yolo26x, yolo26l, ...) or path to a .pt")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="per-GPU batch")
    p.add_argument("--epochs", type=int, default=120, help="hard cap; patience early-stops")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--workers", type=int, default=8, help="dataloader workers per GPU")
    p.add_argument("--force-build", action="store_true", help="rebuild the merged dataset")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--empty-check", action="store_true",
                   help="print per-GPU free VRAM and exit")
    p.add_argument("--no-infer", action="store_true", help="skip the inference test")
    p.add_argument("--no-infer-clip", action="store_true",
                   help="skip the annotated clip (match-wide coverage still runs)")
    p.add_argument("--video", default=None,
                   help="match mp4 (default: first mp4 under ball_data/video/)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = p.parse_args()

    t0 = time.time()
    bootstrap_python()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ensure_deps()

    free = gpu_free_gb()
    if not free:
        raise SystemExit("no CUDA GPUs visible - check the NVIDIA driver / nvidia-smi")
    for i in sorted(free):
        log(f"GPU{i}: {free[i]:.1f} GB free")
    if args.empty_check:
        return 0

    ids = parse_device(args.device) if args.device else \
        [i for i in sorted(free) if free[i] >= args.min_free_gb]
    if not ids:
        raise SystemExit(f"no GPU has >= {args.min_free_gb} GB free right now. "
                         f"Free one up (a llama-server may hold ~22 GB per card), "
                         f"or lower --min-free-gb.")
    model, imgsz, batch = pick_config(ids, free, args.model)
    imgsz = args.imgsz or imgsz
    batch = args.batch or batch
    device = ",".join(str(i) for i in ids)
    name = f"ball_{re.sub(r'[^a-z0-9]', '', model.lower().replace('.pt', ''))}_{imgsz}_clu"
    log(f"config: {model} @ imgsz {imgsz}, batch {batch}/gpu on GPU[{device}] "
        f"(tightest card: {min(free[i] for i in ids):.1f} GB free)")

    if args.dry_run:
        log(f"dry-run: would download 3 datasets from Zenodo/Hugging Face into {EXT}")
        log(f"dry-run: would build {DATASET} then train {model} with {len(ids)} GPU(s)")
        log("dry-run: nothing executed")
        return 0

    check_disk()
    ensure_datasets()
    build_dataset(args.force_build)

    run_dir = train(model, imgsz, batch, device, args.epochs, args.patience,
                    args.workers, name, len(ids), args.resume)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training finished but {best} is missing - check the log above")

    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    stable = WEIGHTS_OUT / f"{name}.pt"
    shutil.copy2(best, stable)
    shutil.copy2(best, WEIGHTS_OUT / "ball_best.pt")

    log("evaluating best.pt on the val split (ball-specific, IoU 0.5)")
    report = evaluate(best, imgsz, device.split(",")[0], name)
    if report:
        t = report["per_threshold"]["0.2"]
        log(f"val F1@0.2={t['f1']} recall={t['recall']} precision={t['precision']} "
            f"centre_px_median={t['centre_px_median']}")

    if not args.no_infer:
        video = find_video(args.video)
        try:
            run_inference(best, device.split(",")[0], video, 0, 0, imgsz)
            if video is not None and not args.no_infer_clip:
                run_inference(best, device.split(",")[0], video, CLIP_START, CLIP_END, imgsz)
        except Exception as exc:
            log(f"WARNING: inference test failed ({exc}) - training/eval unaffected")

    log("================= done =================")
    log(f"weights:   {stable}")
    log(f"run:       {run_dir}")
    log(f"metrics:   {EVAL / (name + '.json')}")
    log(f"copy back: scp <server>:{stable} .")
    log(f"time:      {(time.time() - t0) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
