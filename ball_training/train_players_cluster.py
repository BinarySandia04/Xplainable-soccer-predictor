#!/usr/bin/env python3
"""Train the best possible soccer-player detector on a multi-GPU server.

ONE FILE - paste it with nano, run it. No repo, no git, no token, no code
hosting: datasets and base weights come only from Hugging Face.

    python3 train_players_cluster.py                    # full run
    python3 train_players_cluster.py --dry-run          # show the plan, change nothing
    python3 train_players_cluster.py --empty-check      # per-GPU free VRAM and exit
    python3 train_players_cluster.py --device 1,2,3     # pin GPUs (default: all with room)
    python3 train_players_cluster.py --model yolo26x    # force a model instead of auto-pick
    python3 train_players_cluster.py --resume           # continue an interrupted run

Pipeline, in order:
  1. checks python >= 3.10 and every dependency. If the current interpreter is
     not usable it re-executes inside an existing ball_data/.venv (the one the
     ball run already built), otherwise it creates player_data/.venv itself.
  2. downloads the best public player datasets and merges them into one YOLO
     dataset with three classes (player / goalkeeper / referee; ball boxes are
     ignored because the separate ball detector owns that class):
       SoccerNet-Tracking 2023       HF   research terms   57+49 broadcast games
       martinjolif/football-player-  HF   CC-BY-4.0        Roboflow, 4 classes
       keremberke/football-object-   HF   CC-BY-4.0        Roboflow COCO, 1.2k
       manot/football-players        HF   MIT              Roboflow COCO, 206
     SoccerNet's MOT gt.txt has track ids and gameinfo.ini maps each id to
     player / goalkeeper / referee, so no manual labelling is needed. Train uses
     the official 57 train games, val uses the official 49 test games: whole
     games are held out, never neighbouring frames.
  3. measures the FREE VRAM on every GPU (a llama-server holding ~22 GB per
     card is respected and never touched), picks the largest YOLO26 model,
     resolution and batch that fits, and trains with DDP on every GPU with room.
  4. scores best.pt with player-specific metrics: per-class recall / precision /
     F1 at the confidence floors a tracker uses, plus player-count error.
  5. runs an inference test on the real match video: player-count coverage over
     the whole match, and by default an annotated mp4 of the same clip used on
     the laptop (frames 11197-11566 of BAR-ATH_tactical.mp4), with a box around
     every player, goalkeeper and referee.

The match video is a gated challenge bundle, so copy it once from the laptop.
The script auto-detects any mp4 under player_data/video/, and also reuses the
copy the ball run already has under ball_data/video/ - no second upload needed.

Outputs, all under ./player_data/ (override with PLAYER_DATA_DIR):
    weights/<name>.pt               stable copy of the best checkpoint
    weights/players_best.pt         same checkpoint with a stable name
    runs/<name>/weights/best.pt     full training run (results.png, args.yaml)
    eval/<name>.json                per-class val metrics + per-source breakdown
    inference/                      annotated clip mp4 + coverage JSON

Run detached so an ssh drop does not kill it (the script lives in ~/ball_data).
Run it only after the ball training has finished: both jobs pick GPUs by free
VRAM, and starting this one early would put two runs on the same cards.

    cd ~/ball_data
    setsid nohup .venv/bin/python train_players_cluster.py > train_players.log 2>&1 &
    tail -f train_players.log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------- paths
# Everything this script creates lives under player_data/ next to the script.
# The ball script keeps its own ball_data/ root, so the two runs cannot clash.
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PLAYER_DATA_DIR") or HERE / "player_data").expanduser()
EXT = DATA / "external"
DATASET = DATA / "dataset"
RUNS = DATA / "runs"
EVAL = DATA / "eval"
INFER = DATA / "inference"
WEIGHTS_OUT = DATA / "weights"
VIDEO_DIR = DATA / "video"

SN = EXT / "soccernet_tracking"      # train/ and test/ with SNMOT-* sequences
MJP = EXT / "martinjolif"            # data/{train,valid,test}/{images,labels}
KFB = EXT / "keremberke"             # train/ valid/ test/ with COCO json
MNP = EXT / "manot"                  # train/ valid/ with COCO json

# -------------------------------------------------------------------- video
CLIP_START, CLIP_END = 11197, 11567  # laptop reference clip, end exclusive
VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov")

# ------------------------------------------------------------------- classes
CLASSES = ("player", "goalkeeper", "referee")
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}
SHORT = {0: "P", 1: "GK", 2: "R"}
COLORS = {0: (0, 255, 0), 1: (0, 215, 255), 2: (255, 0, 255)}
BALL_ID = -1                          # internal marker: box is ignored

# ------------------------------------------------------------- dependencies
MIN_PY = (3, 10)
MIN_DISK_GB = 50                     # 18 GB of zips + 18 GB extracted + run
DEP_CHECK = ("import torch, torchvision, ultralytics, cv2, scipy, numpy, "
             "huggingface_hub, yaml, matplotlib")
TORCH_INDEX = os.environ.get("TORCH_PIP_INDEX", "https://download.pytorch.org/whl/cu128")
PIP_TORCH = ("torch==2.11.0", "torchvision==0.26.0")
PIP_EXTRA = ("ultralytics>=8.4.188", "opencv-python-headless", "scipy",
             "huggingface_hub>=0.26", "nvidia-ml-py", "numpy>=1.26",
             "pyyaml", "matplotlib>=3.8")

# ----------------------------------------------------------------- datasets
HF = "https://huggingface.co/datasets"
SN_URL = HF + "/SoccerNet/SN-Tracking-2023/resolve/main/{split}.zip"
SN_MIN = {"train": 50, "test": 40}   # archives hold 57 / 49 sequences
MJP_REPO = "martinjolif/football-player-detection"
KFB_REPO = "keremberke/football-object-detection"
MNP_REPO = "manot/football-players"

# ------------------------------------------------------------------- model
MODEL_FACTOR = {"n": 0.45, "s": 0.6, "m": 1.0, "l": 1.3, "x": 1.7}
PER_SAMPLE_640 = 0.69                # GB per sample, "m" detector, AdamW state
LADDER = ["x", "l", "m", "s"]        # largest first
RESOLUTIONS = [1280, 1024, 896, 640]
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
SOURCE_CONF = 0.3                    # confidence used for the per-source table

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# base checkpoints + the AMP-check model come from these official HF mirrors
HF_WEIGHT_REPOS = {"yolo26": "Ultralytics/YOLO26",
                   "yolo11": "Ultralytics/YOLO11",
                   "yolov8": "Ultralytics/YOLOv8",
                   "yolo8": "Ultralytics/YOLOv8"}


def log(msg: str) -> None:
    print(f"[player_train] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def sh(cmd: list[str]) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


# -------------------------------------------------------------------- setup
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
    # ultralytics pulls the GUI OpenCV build; force the headless one last so a
    # server without libGL still imports cv2.
    sh([*pip, "--force-reinstall", "--no-deps", "opencv-python-headless"])


def _working_venv() -> str | None:
    """Find an interpreter that already imports every dependency.

    On the cluster the ball run leaves ball_data/.venv behind; reusing it avoids
    a second 3 GB torch download. PLAYER_VENV=/path/to/venv overrides it.
    """
    candidates: list[Path] = []
    override = os.environ.get("PLAYER_VENV")
    if override:
        candidates.append(Path(override).expanduser())
    candidates += [HERE / "ball_data" / ".venv",
                   HERE.parent / "ball_data" / ".venv",
                   Path.home() / "ball_data" / ".venv"]
    seen: set[Path] = set()
    for venv in candidates:
        py = venv / "bin" / "python"
        if py in seen:
            continue
        seen.add(py)
        if py.is_file() and deps_present(str(py)):
            return str(py)
    return None


def ensure_deps() -> None:
    if deps_present(sys.executable):
        return
    alt = _working_venv()
    if alt:
        log(f"re-executing inside existing venv: {alt}")
        os.execv(alt, [alt, *sys.argv])
    log("installing python deps (a few minutes)")
    try:
        install_into(sys.executable)
    except subprocess.CalledProcessError:
        if sys.prefix != sys.base_prefix:
            raise  # already in a venv/conda env: the failure is real, show it
        log("system python refuses pip (externally managed) - bootstrapping player_data/.venv")
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


# --------------------------------------------------------------------- GPU
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
    probe = DATA
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < 35:
        raise SystemExit(
            f"only {free_gb:.0f} GB free under {probe}; SoccerNet needs ~18 GB "
            f"of zips plus ~18 GB extracted. Free space or set PLAYER_DATA_DIR."
        )
    if free_gb < MIN_DISK_GB:
        log(f"WARNING: {free_gb:.0f} GB free - peak usage is ~{MIN_DISK_GB} GB")
    else:
        log(f"disk ok: {free_gb:.0f} GB free")


# ---------------------------------------------------------------- downloads
def download(url: str, dest: Path, attempts: int = 6) -> None:
    """Streamed download with resume (Range), progress lines and a size check."""
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.is_file() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "player-trainer/1.0"})
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
                        if now - last > 5 or (total and got >= total):
                            last = now
                            pct = 100 * got / total if total else 0
                            log(f"  {dest.name}: {got / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:.0f}%)")
            if total and got < total:
                if attempt == attempts:
                    raise RuntimeError(f"{dest.name}: got {got}/{total} bytes")
                log(f"  download ended early ({got}/{total}), retrying ({attempt}/{attempts})")
                time.sleep(5 * attempt)
                continue
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


def extract_zip(zip_path: Path, dest: Path, ready, label: str) -> None:
    """Extract once, verify a marker, then drop the archive to save disk."""
    if ready():
        log(f"skip {label}: present")
        zip_path.unlink(missing_ok=True)
        return
    log(f"extracting {label} ({zip_path.stat().st_size / 1e9:.1f} GB)")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        zip_path.unlink(missing_ok=True)
        raise SystemExit(f"{zip_path.name} is corrupt or truncated - rerun to download it again") from exc
    if not ready():
        raise SystemExit(f"{label}: extraction finished but the expected files are missing in {dest}")
    zip_path.unlink(missing_ok=True)
    log(f"{label} ready")


def sn_ready(split: str, minimum: int) -> bool:
    return len(list((SN / split).glob("SNMOT-*/gt/gt.txt"))) >= minimum


def coco_ready(dest: Path) -> bool:
    return (dest / "_annotations.coco.json").is_file() and any(dest.glob("*.jpg"))


def ensure_soccernet_tracking() -> None:
    for split, minimum in SN_MIN.items():
        if sn_ready(split, minimum):
            log(f"skip SoccerNet-Tracking {split}: present")
            continue
        zip_path = EXT / f"soccernet_tracking_{split}.zip"
        if zip_path.is_file() and zip_path.stat().st_size < 1_000_000_000:
            zip_path.unlink()
        if not zip_path.is_file():
            log(f"fetching SoccerNet-Tracking {split} (Hugging Face, ~9 GB)")
            download(SN_URL.format(split=split), zip_path)
        extract_zip(zip_path, SN, lambda s=split, m=minimum: sn_ready(s, m),
                    f"SoccerNet-Tracking {split}")


def ensure_martinjolif() -> None:
    from huggingface_hub import snapshot_download

    marker = MJP / "data" / "data.yaml"
    images = MJP / "data" / "train" / "images"
    if marker.is_file() and images.is_dir():
        log("skip martinjolif: present")
        return
    log("fetching martinjolif/football-player-detection (Hugging Face)")
    snapshot_download(repo_id=MJP_REPO, repo_type="dataset", local_dir=str(MJP),
                      allow_patterns=["data/**"], max_workers=8,
                      token=os.environ.get("HF_TOKEN") or None)
    if not marker.is_file():
        raise RuntimeError("martinjolif snapshot is missing data/data.yaml")


def ensure_coco_source(repo: str, root: Path, splits: tuple[str, ...], label: str) -> None:
    for split in splits:
        dest = root / split
        if coco_ready(dest):
            log(f"skip {label} {split}: present")
            continue
        zip_path = EXT / "zips" / f"{label}_{split}.zip"
        if zip_path.is_file() and zip_path.stat().st_size < 1_000_000:
            zip_path.unlink()
        if not zip_path.is_file():
            log(f"fetching {label} {split} (Hugging Face)")
            download(f"{HF}/{repo}/resolve/main/data/{split}.zip", zip_path)
        extract_zip(zip_path, dest, lambda d=dest: coco_ready(d), f"{label} {split}")


def ensure_datasets() -> None:
    ensure_soccernet_tracking()
    for fn, label in ((ensure_martinjolif, "martinjolif"),
                      (lambda: ensure_coco_source(KFB_REPO, KFB, ("train", "valid", "test"),
                                                  "keremberke"), "keremberke"),
                      (lambda: ensure_coco_source(MNP_REPO, MNP, ("train", "valid"),
                                                  "manot"), "manot")):
        try:
            fn()
        except Exception as exc:
            log(f"WARNING: optional source {label} failed ({exc}) - continuing without it")


# ----------------------------------------------------------- dataset build
def class_id_from_name(name: str) -> int | None:
    """Map a dataset category to 0/1/2, or None when it is not a person class."""
    low = name.strip().lower().replace("_", " ").replace("-", " ")
    if "goalkeeper" in low or "goalie" in low or "keeper" in low:
        return 1
    if "referee" in low or low in {"ref", "refs"}:
        return 2
    if "player" in low or "footballer" in low:
        return 0
    return None


def tracklet_classes(gameinfo: Path) -> dict[int, int]:
    """SoccerNet gameinfo.ini: trackletID_18= ball;1 -> -1 (ignore)."""
    out: dict[int, int] = {}
    if not gameinfo.is_file():
        return out
    for line in gameinfo.read_text(encoding="utf-8").splitlines():
        m = re.match(r"trackletID_(\d+)\s*=\s*(.+)", line.strip())
        if not m:
            continue
        low = m.group(2).strip().lower()
        if "goalkeeper" in low or "goalie" in low or "keeper" in low:
            out[int(m.group(1))] = 1
        elif "referee" in low:
            out[int(m.group(1))] = 2
        elif "player" in low:
            out[int(m.group(1))] = 0
        elif any(k in low for k in ("ball", "other", "staff", "coach", "crowd", "spectator")):
            out[int(m.group(1))] = BALL_ID
    return out


def read_ini(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def mot_labels(seq_dir: Path, width: int, height: int, mapping: dict[int, int],
               wanted: set[int]) -> tuple[dict[int, list[str]], int, int]:
    """Read one MOT gt.txt and return YOLO lines grouped by frame."""
    labels: dict[int, list[str]] = defaultdict(list)
    gt = seq_dir / "gt" / "gt.txt"
    if not gt.is_file():
        return labels, 0, 0
    unknown = total = 0
    with gt.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            try:
                frame = int(float(row[0]))
                tid = int(float(row[1]))
                x, y, w, h = (float(v) for v in row[2:6])
            except ValueError:
                continue
            # MOT convention: confidence 0 marks a region to ignore
            if len(row) >= 7:
                try:
                    if float(row[6]) <= 0:
                        continue
                except ValueError:
                    pass
            if frame not in wanted:
                continue
            total += 1
            cid = mapping.get(tid)
            if cid is None:
                unknown += 1
                continue
            if cid == BALL_ID:
                continue
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            labels[frame].append(
                f"{cid} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
    return labels, unknown, total


def link_image(image: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(image.resolve())


def _reset_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def emit(prefix: str, image: Path, lines: list[str], split: str,
         counter: Counter, seen: set[str]) -> None:
    name = f"{prefix}__{image.stem}"
    if name in seen:
        i = 2
        while f"{name}_{i}" in seen:
            i += 1
        name = f"{name}_{i}"
    seen.add(name)
    dest_img = DATASET / "images" / split / f"{name}{image.suffix.lower()}"
    link_image(image, dest_img)
    (DATASET / "labels" / split / f"{name}.txt").write_text(
        "".join(f"{ln}\n" for ln in lines), encoding="utf-8")
    counter[f"{split}:frames"] += 1
    for line in lines:
        counter[f"class:{CLASSES[int(line.split()[0])]}"] += 1


def sn_split(split: str, target: str, max_frames: int, prefix: str,
             counter: Counter, seen: set[str]) -> int:
    seq_dirs = sorted(p for p in (SN / split).glob("SNMOT-*") if (p / "gt" / "gt.txt").is_file())
    if not seq_dirs:
        log(f"  SoccerNet {split}: no sequences found under {SN / split}")
        return 0
    lengths: dict[Path, int] = {}
    for seq in seq_dirs:
        cfg = read_ini(seq / "seqinfo.ini")
        lengths[seq] = int(cfg.get("seqLength") or 750)
    total_frames = sum(lengths.values())
    stride = max(1, math.ceil(total_frames / max(1, max_frames)))
    log(f"  SoccerNet {split}: {len(seq_dirs)} games, {total_frames} frames, stride {stride}")
    emitted = unknown = rows = 0
    for seq in seq_dirs:
        cfg = read_ini(seq / "seqinfo.ini")
        width = int(cfg.get("imWidth") or 1920)
        height = int(cfg.get("imHeight") or 1080)
        length = lengths[seq]
        mapping = tracklet_classes(seq / "gameinfo.ini")
        if not mapping:
            log(f"  WARNING: {seq.name}: empty gameinfo.ini, skipping")
            continue
        wanted = {f for f in range(1, length + 1) if (f - 1) % stride == 0}
        labels, miss, total = mot_labels(seq, width, height, mapping, wanted)
        unknown += miss
        rows += total
        for frame in sorted(labels):
            image = seq / "img1" / f"{frame:06d}.jpg"
            if not image.is_file():
                continue
            emit(f"{prefix}_{seq.name}", image, labels[frame], target, counter, seen)
            emitted += 1
    if rows and unknown / rows > 0.02:
        log(f"  WARNING: {unknown}/{rows} SoccerNet {split} boxes had an unmapped track id")
    log(f"  SoccerNet {split}: {emitted} frames emitted")
    return emitted


def yolo_pairs(root: Path, split: str) -> list[tuple[Path, list[str]]]:
    """Read a Roboflow YOLO split and remap its names to player/GK/referee."""
    import yaml

    data_yaml = root / "data" / "data.yaml"
    images_dir = root / "data" / split / "images"
    labels_dir = root / "data" / split / "labels"
    if not data_yaml.is_file() or not images_dir.is_dir():
        return []
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw = cfg.get("names")
    if isinstance(raw, list):
        names = {i: str(v) for i, v in enumerate(raw)}
    elif isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items()}
    else:
        names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    pairs: list[tuple[Path, list[str]]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            continue
        lines: list[str] = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                src = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            cid = class_id_from_name(names.get(src, ""))
            if cid is None or not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                continue
            x1, x2 = max(0.0, cx - bw / 2), min(1.0, cx + bw / 2)
            y1, y2 = max(0.0, cy - bh / 2), min(1.0, cy + bh / 2)
            if x2 - x1 < 0.002 or y2 - y1 < 0.002:
                continue
            lines.append(f"{cid} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                         f"{x2 - x1:.6f} {y2 - y1:.6f}")
        if lines:
            pairs.append((image, lines))
    return pairs


def coco_pairs(root: Path, split: str) -> list[tuple[Path, list[str]]]:
    """Read a Roboflow COCO split; category names decide player/GK/referee."""
    ann_path = root / split / "_annotations.coco.json"
    if not ann_path.is_file():
        return []
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"  WARNING: cannot read {ann_path}: {exc}")
        return []
    cats = {c.get("id"): c.get("name", "") for c in data.get("categories", [])}
    images = {img.get("id"): img for img in data.get("images", [])}
    anns: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        cid = class_id_from_name(str(cats.get(ann.get("category_id"), "")))
        bbox = ann.get("bbox")
        if cid is None or not bbox or len(bbox) < 4:
            continue
        anns[ann.get("image_id")].append((cid, bbox[:4]))
    pairs: list[tuple[Path, list[str]]] = []
    for image_id, items in anns.items():
        info = images.get(image_id)
        if not info:
            continue
        image = root / split / str(info.get("file_name", ""))
        if not image.is_file():
            continue
        width = float(info.get("width") or 0)
        height = float(info.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        lines: list[str] = []
        for cid, bbox in items:
            try:
                x, y, bw, bh = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x, y, bw, bh)):
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(width, x + bw), min(height, y + bh)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            lines.append(f"{cid} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        if lines:
            pairs.append((image, lines))
    return pairs


def build_dataset(force: bool = False, max_train: int = 24000, max_val: int = 3000) -> None:
    data_yaml = DATASET / "data.yaml"
    if data_yaml.is_file() and not force:
        log(f"dataset already built: {DATASET}")
        return
    # drop the completion marker first: a rebuild that dies halfway must never
    # leave an old data.yaml pointing at wiped image folders
    data_yaml.unlink(missing_ok=True)
    (DATASET / "manifest.json").unlink(missing_ok=True)
    if not sn_ready("train", SN_MIN["train"]) or not sn_ready("test", SN_MIN["test"]):
        raise SystemExit(f"SoccerNet-Tracking is not ready under {SN} - rerun the download step")
    for path in (DATASET / "images" / "train", DATASET / "images" / "val",
                 DATASET / "labels" / "train", DATASET / "labels" / "val"):
        _reset_dir(path)
    for cache in (DATASET / "labels" / "train.cache", DATASET / "labels" / "val.cache"):
        cache.unlink(missing_ok=True)

    counter: Counter = Counter()
    seen = {"train": set(), "val": set()}

    def emit_source(label: str, pairs: list[tuple[Path, list[str]]], target: str, tag: str) -> None:
        before = counter[f"{target}:frames"]
        for image, lines in pairs:
            emit(tag, image, lines, target, counter, seen[target])
        log(f"  {label}: +{counter[f'{target}:frames'] - before} {target} frames")

    log("merging SoccerNet-Tracking 2023 (broadcast, 3 person classes)")
    sn_split("train", "train", max_train, "snt_tr", counter, seen["train"])
    sn_split("test", "val", max_val, "snt_va", counter, seen["val"])

    log("merging martinjolif/football-player-detection (YOLO, official split)")
    emit_source("martinjolif", yolo_pairs(MJP, "train"), "train", "mjp_tr")
    emit_source("martinjolif", yolo_pairs(MJP, "valid") + yolo_pairs(MJP, "test"),
                "val", "mjp_va")

    log("merging keremberke/football-object-detection (COCO, official split)")
    emit_source("keremberke", coco_pairs(KFB, "train"), "train", "kfb_tr")
    emit_source("keremberke", coco_pairs(KFB, "valid") + coco_pairs(KFB, "test"),
                "val", "kfb_va")

    log("merging manot/football-players (COCO, official split)")
    emit_source("manot", coco_pairs(MNP, "train"), "train", "mnp_tr")
    emit_source("manot", coco_pairs(MNP, "valid"), "val", "mnp_va")

    stats = dict(counter)
    stats.update({
        "classes": list(CLASSES),
        "train_frames": counter["train:frames"],
        "val_frames": counter["val:frames"],
        "boxes": {name: counter[f"class:{name}"] for name in CLASSES},
        "sources": {
            "snt": "SoccerNet-Tracking 2023 (HF SoccerNet/SN-Tracking-2023)",
            "mjp": "martinjolif/football-player-detection (HF, CC-BY-4.0)",
            "kfb": "keremberke/football-object-detection (HF, CC-BY-4.0)",
            "mnp": "manot/football-players (HF, MIT)",
        },
    })
    if not stats["train_frames"] or not stats["val_frames"]:
        raise SystemExit(
            f"dataset build produced {stats['train_frames']} train / "
            f"{stats['val_frames']} val frames - inspect the extraction and label "
            f"parsing warnings above")
    # manifest first, data.yaml last: data.yaml is the completion marker, so an
    # interrupted build must never look finished
    (DATASET / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (DATASET / "data.yaml").write_text(
        "# Generated by train_players_cluster.py -- do not edit by hand.\n"
        f"path: {DATASET.resolve()}\n"
        "train: images/train\nval: images/val\nnc: 3\nnames:\n"
        "  0: player\n  1: goalkeeper\n  2: referee\n",
        encoding="utf-8")
    log(f"dataset ready: train={stats['train_frames']} val={stats['val_frames']} "
        f"boxes={stats['boxes']}")


# ------------------------------------------------------------------- train
def ensure_base_weights(model: str) -> str:
    """Resolve base weights locally, downloading from Hugging Face only.

    Ultralytics' own loader fetches yolo*.pt from its release-asset page, and
    its AMP check does the same for yolo26n.pt. That host may be unreachable on
    this server, so every checkpoint is pulled from the official HF mirrors.
    """
    import ultralytics.utils as uutils
    from huggingface_hub import hf_hub_download

    weights_dir = DATA / "base_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    # DDP workers re-import ultralytics in fresh processes, so the weights directory
    # has to be recorded in the settings file, not only in this process' module attr.
    if Path(uutils.SETTINGS["weights_dir"]).resolve() != weights_dir.resolve():
        uutils.SETTINGS["weights_dir"] = str(weights_dir)
    uutils.WEIGHTS_DIR = weights_dir

    stem = model if model.endswith(".pt") else f"{model}.pt"
    if Path(model).is_file():  # user-supplied local checkpoint
        return model
    if not (weights_dir / stem).is_file():
        repo = next((r for k, r in HF_WEIGHT_REPOS.items() if stem.startswith(k)), None)
        if repo is None:
            raise SystemExit(f"unknown base model {stem!r} - pass a local .pt with --model")
        log(f"downloading base weights {stem} from Hugging Face ({repo})")
        hf_hub_download(repo_id=repo, filename=stem, local_dir=str(weights_dir))
    # the AMP check hardcodes yolo26n.pt; pre-place it so it never goes online
    amp = weights_dir / "yolo26n.pt"
    if not amp.is_file():
        log("downloading AMP-check weights yolo26n.pt from Hugging Face")
        hf_hub_download(repo_id="Ultralytics/YOLO26", filename="yolo26n.pt",
                        local_dir=str(weights_dir))
    # plots look for Arial.ttf online; a local font skips that probe entirely
    font = uutils.USER_CONFIG_DIR / "Arial.ttf"
    if not font.is_file():
        try:
            from matplotlib import font_manager

            uutils.USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(font_manager.findfont("DejaVu Sans"), font)
        except Exception:
            pass
    return str(weights_dir / stem)


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

    weights = ensure_base_weights(model)
    workers = max(2, min(workers, (os.cpu_count() or 8) // max(1, n_gpus)))
    # ultralytics reads `batch` as the GLOBAL batch and gives each rank batch/world_size,
    # so a per-GPU figure below the GPU count would divide down to 0.
    global_batch = max(batch, 1) * max(n_gpus, 1)
    log(f"training {Path(weights).name}: imgsz={imgsz} batch={batch}/gpu ({global_batch} total) "
        f"epochs<={epochs} patience={patience} device={device} workers={workers}/gpu")
    try:
        net = YOLO(weights)
    except Exception as exc:
        raise SystemExit(
            f"could not load {weights}: {exc}\n"
            f"Download that checkpoint manually and rerun with --model /path/to/{Path(weights).name}"
        ) from exc
    net.train(
        data=str(DATASET / "data.yaml"),
        imgsz=imgsz,
        epochs=epochs,
        batch=global_batch,
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
        # player recipe: people are big and upright, so no rotation/shear/perspective;
        # moderate scale + translation handle wide shots vs close-ups, fliplr only.
        degrees=0.0, translate=0.15, scale=0.35, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        mixup=0.0, copy_paste=0.0, erasing=0.1,
        mosaic=1.0, close_mosaic=min(15, max(5, epochs // 8)), multi_scale=False,
    )
    return run_dir


# -------------------------------------------------------------------- eval
def _load_gt(label_path: Path, w: int, h: int):
    import numpy as np

    boxes, classes = [], []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                _c, cx, cy, bw, bh = [float(v) for v in parts[:5]]
            except ValueError:
                continue
            if cid not in (0, 1, 2):
                continue
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            classes.append(cid)
    return (np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            np.asarray(classes, dtype=np.int64))


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


def _score_frame(pred, pred_cls, pred_conf, gt, gt_cls, conf: float, iou_thr: float = 0.5):
    """Greedy per-class matching; returns {class: (tp, fp, fn, n_pred, n_gt)}."""
    import numpy as np

    out = {}
    for cid in range(len(CLASSES)):
        keep = (pred_cls == cid) & (pred_conf >= conf)
        p = pred[keep]
        pc = pred_conf[keep]
        if p.size:
            p = p[np.argsort(-pc)]
        g = gt[gt_cls == cid]
        if p.size == 0:
            out[cid] = (0, 0, len(g), 0, len(g))
            continue
        if g.size == 0:
            out[cid] = (0, len(p), 0, len(p), 0)
            continue
        costs = _iou_matrix(p, g)
        used = np.zeros(len(g), dtype=bool)
        tp = 0
        for row in range(len(p)):
            # never let a prediction claim a ground-truth box that is already matched
            scores = costs[row].copy()
            scores[used] = -1.0
            col = int(np.argmax(scores))
            if scores[col] >= iou_thr:
                used[col] = True
                tp += 1
        out[cid] = (tp, len(p) - tp, int((~used).sum()), len(p), len(g))
    return out


def _prf(tp: int, fp: int, fn: int) -> dict:
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * recall * precision / max(1e-9, recall + precision)
    return {"recall": round(recall, 4), "precision": round(precision, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def _source_of(path: Path) -> str:
    stem = path.stem.split("__")[0]
    for tag in ("snt", "mjp", "kfb", "mnp"):
        if stem.startswith(tag):
            return tag
    return "other"


def evaluate(weights: Path, imgsz: int, device: str, name: str, batch: int = 8) -> dict | None:
    """Player-specific val metrics (what a possession tracker actually needs)."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    images = sorted(p for p in (DATASET / "images" / "val").iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        log("no val images - skipping evaluation")
        return None
    model = YOLO(str(weights))
    totals = {c: {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in range(len(CLASSES))}
              for c in THRESHOLDS}
    count_abs = {c: [] for c in THRESHOLDS}
    source_totals: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in range(len(CLASSES))})
    source_frames: Counter = Counter()
    frames_gt = 0
    seen = 0

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        frames, paths = [], []
        for path in chunk:
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)
                paths.append(path)
        if not frames:
            continue
        results = model.predict(frames, imgsz=imgsz, conf=0.01, iou=0.6,
                                device=device, verbose=False, max_det=300,
                                classes=[0, 1, 2])
        for path, frame, result in zip(paths, frames, results):
            h, w = frame.shape[:2]
            gt, gt_cls = _load_gt(DATASET / "labels" / "val" / f"{path.stem}.txt", w, h)
            if gt.size:
                frames_gt += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                pred, pred_cls, confs = np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0)
            else:
                pred = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
                pred_cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
                confs = boxes.conf.detach().cpu().numpy().astype(np.float64)
            seen += 1
            src = _source_of(path)
            source_frames[src] += 1
            for c in THRESHOLDS:
                scores = _score_frame(pred, pred_cls, confs, gt, gt_cls, c)
                for cid, (tp, fp, fn, n_pred, n_gt) in scores.items():
                    totals[c][cid]["tp"] += tp
                    totals[c][cid]["fp"] += fp
                    totals[c][cid]["fn"] += fn
                    if cid == 0:
                        count_abs[c].append(abs(n_pred - n_gt))
                    if c == SOURCE_CONF:
                        source_totals[src][cid]["tp"] += tp
                        source_totals[src][cid]["fp"] += fp
                        source_totals[src][cid]["fn"] += fn

    report: dict = {"weights": str(weights), "imgsz": imgsz, "images": seen,
                    "images_with_gt": frames_gt, "iou": 0.5,
                    "classes": list(CLASSES), "per_threshold": {}, "per_source": {}}
    for c in THRESHOLDS:
        per_class = {}
        for cid, cname in enumerate(CLASSES):
            t = totals[c][cid]
            per_class[cname] = _prf(t["tp"], t["fp"], t["fn"])
        macros = {k: round(sum(per_class[n][k] for n in CLASSES) / len(CLASSES), 4)
                  for k in ("recall", "precision", "f1")}
        report["per_threshold"][f"{c:g}"] = {
            "per_class": per_class, "macro": macros,
            "player_count_mae": round(float(np.mean(count_abs[c])), 2) if count_abs[c] else None,
        }
    for src, vals in source_totals.items():
        report["per_source"][src] = {
            "frames": source_frames[src], "conf": SOURCE_CONF,
            "per_class": {cname: _prf(vals[cid]["tp"], vals[cid]["fp"], vals[cid]["fn"])
                          for cid, cname in enumerate(CLASSES)},
        }
    EVAL.mkdir(parents=True, exist_ok=True)
    out = EVAL / f"{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = report["per_threshold"][f"{SOURCE_CONF:g}"]
    log(f"val player R={t['per_class']['player']['recall']} "
        f"P={t['per_class']['player']['precision']} F1={t['per_class']['player']['f1']} | "
        f"GK R={t['per_class']['goalkeeper']['recall']} | "
        f"ref R={t['per_class']['referee']['recall']} | "
        f"player count MAE={t['player_count_mae']}")
    log(f"eval -> {out}")
    return report


# --------------------------------------------------------------- inference
def _video_dirs() -> list[Path]:
    raw = [VIDEO_DIR, HERE / "ball_data" / "video",
           HERE.parent / "ball_data" / "video", Path.home() / "ball_data" / "video"]
    out: list[Path] = []
    seen: set[Path] = set()
    for d in raw:
        d = d.expanduser()
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def find_video(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        log(f"WARNING: --video {p} not found; searching the default folders")
    for d in _video_dirs():
        p = d / "BAR-ATH_tactical.mp4"
        if p.is_file():
            return p
    for d in _video_dirs():
        for pattern in VIDEO_SUFFIXES:
            hits = sorted(d.glob(f"*{pattern}"))
            if hits:
                return hits[0]
    return None


def _no_video_hint() -> None:
    log("no match video found - inference test skipped")
    log("  reuse the ball-run copy (no second upload): ~/ball_data/video/BAR-ATH_tactical.mp4")
    log("  or copy it once from the laptop (it is a gated bundle, not downloadable):")
    log(f"    scp data/tactical/BAR-ATH_tactical.mp4 user@server:{VIDEO_DIR}/")
    log("  or rerun later with --video /path/to/BAR-ATH_tactical.mp4")


def _draw_players(frame, result) -> list[int]:
    """Draw one box per person; returns [players, goalkeepers, referees]."""
    import cv2

    counts = [0, 0, 0]
    _h, w = frame.shape[:2]
    thick = max(2, round(w / 960))
    font = max(0.5, w / 1920 * 0.6)
    for box in result.boxes:
        cid = int(box.cls[0])
        if cid not in COLORS:
            continue
        counts[cid] += 1
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        color = COLORS[cid]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        text = f"{SHORT[cid]} {float(box.conf[0]):.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, max(1, thick - 1))
        top = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)
    return counts


def _count_players(result) -> list[int]:
    counts = [0, 0, 0]
    for box in result.boxes:
        cid = int(box.cls[0])
        if cid in COLORS:
            counts[cid] += 1
    return counts


def run_inference(weights: Path, device: str, video: Path | None,
                  start_frame: int = 0, end_frame: int = 0, imgsz: int = 1024,
                  conf: float = 0.25, annotate: bool = False, tag: str = "full_match") -> None:
    """Player coverage over the match (+ optional annotated clip mp4)."""
    import cv2
    import numpy as np
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
    clip_mode = end_frame > start_frame
    step = 1 if clip_mode else max(1, (total or 39000) // 600)
    out = INFER / f"{video.stem}_{tag}_players.mp4"
    writer = None
    write_video = annotate
    counts: list[list[int]] = []
    frame_counts: list[list[int]] = []
    n = 0
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
        res = model.predict(frame, imgsz=imgsz, conf=conf, iou=0.6, device=device,
                            verbose=False, max_det=300, classes=[0, 1, 2])[0]
        one = _draw_players(frame, res) if annotate else _count_players(res)
        counts.append(one)
        frame_counts.append([idx, one[0], one[1], one[2]])
        if clip_mode and write_video:
            if writer is None:
                INFER.mkdir(parents=True, exist_ok=True)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    log("WARNING: cannot open the mp4 writer - continuing without video")
                    writer = None
                    write_video = False
            if writer is not None:
                cv2.putText(frame, f"frame {idx}  P:{one[0]} GK:{one[1]} R:{one[2]}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
        log(f"annotated clip -> {out}")

    arr = np.asarray(counts, dtype=np.float64) if counts else np.zeros((0, 3))
    players = arr[:, 0] if len(arr) else np.zeros(0)
    summary = {
        "weights": str(weights), "video": str(video), "clip": tag,
        "conf": conf, "imgsz": imgsz, "frames_tested": n,
        "frames_with_5plus_players": int((players >= 5).sum()) if n else 0,
        "coverage_5plus": round(float((players >= 5).mean()), 4) if n else 0.0,
        "mean_players": round(float(players.mean()), 2) if n else None,
        "median_players": round(float(np.median(players)), 2) if n else None,
        "p10_players": round(float(np.percentile(players, 10)), 2) if n else None,
        "p90_players": round(float(np.percentile(players, 90)), 2) if n else None,
        "max_players": int(players.max()) if n else 0,
        "mean_goalkeepers": round(float(arr[:, 1].mean()), 2) if n else None,
        "mean_referees": round(float(arr[:, 2].mean()), 2) if n else None,
        "annotated_clip": str(out) if writer is not None else None,
        "frame_counts": frame_counts,  # [frame_index, players, goalkeepers, referees]
    }
    INFER.mkdir(parents=True, exist_ok=True)
    json_out = INFER / f"{video.stem}_{tag}_players.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if n:
        log(f"inference [{tag}]: {summary['coverage_5plus']:.1%} of frames have >=5 players, "
            f"mean {summary['mean_players']} players/frame, max {summary['max_players']} "
            f"-> {json_out}")
    else:
        log(f"inference [{tag}]: no frames processed -> {json_out}")


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
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--workers", type=int, default=8, help="dataloader workers per GPU")
    p.add_argument("--force-build", action="store_true", help="rebuild the merged dataset")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--empty-check", action="store_true",
                   help="print per-GPU free VRAM and exit")
    p.add_argument("--no-infer", action="store_true", help="skip the inference test")
    p.add_argument("--no-infer-clip", action="store_true",
                   help="skip the annotated clip (match-wide coverage still runs)")
    p.add_argument("--video", default=None,
                   help="match mp4 (default: first mp4 under player_data/video/, then ball_data/video/)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    p.add_argument("--max-train-frames", type=int, default=24000,
                   help="cap on SoccerNet train frames (stride is chosen automatically)")
    p.add_argument("--max-val-frames", type=int, default=3000,
                   help="cap on SoccerNet val frames (stride is chosen automatically)")
    p.add_argument("--clip-start", type=int, default=CLIP_START)
    p.add_argument("--clip-end", type=int, default=CLIP_END, help="exclusive")
    p.add_argument("--infer-conf", type=float, default=0.25,
                   help="confidence floor for the annotated clip")
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
    missing = [i for i in ids if i not in free]
    if missing:
        raise SystemExit(f"GPU(s) {missing} not visible; nvidia-smi lists {sorted(free)}")
    if not args.device and len(ids) > 1:
        best_free = max(free[i] for i in ids)
        keep = [i for i in ids if free[i] >= 0.6 * best_free]
        if len(keep) < len(ids):
            dropped = [i for i in ids if i not in keep]
            log("auto-pick leaves GPU(s) " +
                ", ".join(f"{i} ({free[i]:.1f} GB free)" for i in dropped) +
                f" out: the roomiest card has {best_free:.1f} GB, and DDP runs every rank "
                f"at the same batch, so one small card would slow all of them")
            ids = keep
    model, imgsz, batch = pick_config(ids, free, args.model)
    imgsz = args.imgsz or imgsz
    batch = args.batch or batch
    device = ",".join(str(i) for i in ids)
    stem = Path(model).stem if model.endswith(".pt") else model
    name = f"players_{re.sub(r'[^a-z0-9]', '', stem.lower())}_{imgsz}_clu"
    log(f"config: {model} @ imgsz {imgsz}, batch {batch}/gpu on GPU[{device}] "
        f"(tightest card: {min(free[i] for i in ids):.1f} GB free)")

    if args.dry_run:
        log(f"dry-run: dataset root {DATA}")
        log("dry-run: would download SoccerNet-Tracking 2023 (~9 GB train + ~9 GB test) "
            "and three Roboflow player sets from Hugging Face")
        log(f"dry-run: would build {DATASET} (train<={args.max_train_frames}, "
            f"val<={args.max_val_frames} SoccerNet frames)")
        log(f"dry-run: would train {model} with {len(ids)} GPU(s), then evaluate and infer")
        log("dry-run: nothing executed")
        return 0

    check_disk()
    ensure_datasets()
    build_dataset(args.force_build, args.max_train_frames, args.max_val_frames)

    run_dir = train(model, imgsz, batch, device, args.epochs, args.patience,
                    args.workers, name, len(ids), args.resume)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training finished but {best} is missing - check the log above")

    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    stable = WEIGHTS_OUT / f"{name}.pt"
    shutil.copy2(best, stable)
    shutil.copy2(best, WEIGHTS_OUT / "players_best.pt")

    log("evaluating best.pt on the val games (per-class, IoU 0.5)")
    evaluate(best, imgsz, device.split(",")[0], name)

    if not args.no_infer:
        video = find_video(args.video)
        try:
            run_inference(best, device.split(",")[0], video, 0, 0, imgsz,
                          args.infer_conf, annotate=False, tag="full_match")
            if video is not None and not args.no_infer_clip:
                run_inference(best, device.split(",")[0], video,
                              args.clip_start, args.clip_end, imgsz, args.infer_conf,
                              annotate=True,
                              tag=f"clip_{args.clip_start}-{args.clip_end - 1}")
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
