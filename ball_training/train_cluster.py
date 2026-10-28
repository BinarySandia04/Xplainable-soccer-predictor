#!/usr/bin/env python3
"""Download the best ball datasets and train the best ball detector on a GPU cluster.

Self-bootstrapping single file - upload it anywhere and run it. It installs every dependency
(torch cu128, ultralytics, opencv, scipy, huggingface_hub, numpy stack; auto-venvs if system
pip is externally managed), fetches the repo helpers (build_dataset.py / eval_detector.py)
from GitHub, downloads the three best public ball sources, merges them into one YOLO set,
then trains and evaluates on the val split.

Hardware-aware: it measures the FREE memory on every GPU (other processes - e.g. a running
llama-server - are respected), then picks the largest YOLO26 model, resolution and batch that
fits, and uses every GPU that has room (DDP across them).

Run on the cluster (merit-gpu1), inside tmux or nohup so it survives detach:

    python3 train_cluster.py                        # auto-pick GPUs, model, batch
    python3 train_cluster.py --dry-run              # show exactly what would run
    python3 train_cluster.py --device 1,2,3         # pin GPUs explicitly
    python3 train_cluster.py --model yolo26x        # override auto model pick
    python3 train_cluster.py --empty-check          # report per-GPU free memory and exit

Tip: the llama-server on these nodes holds ~22 GB per card. Freeing one card roughly doubles
throughput - but only if it is not the server answering your chats. The script never touches
it.

Outputs:  <data>/runs/<name>/weights/best.pt + results.png, and <data>/eval/server_<model>.json.
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

REPO_URL = "https://github.com/BinarySandia04/Xplainable-soccer-predictor.git"
REPO_BRANCH = "feature/ball-detector-retraining"
MIN_PY = (3, 10)
DEP_CHECK = ("import torch, torchvision, ultralytics, cv2, scipy, numpy, huggingface_hub")

HERE = Path(__file__).resolve().parent
DATA = HERE / "ball_data"           # everything this script creates lives here
REPO = DATA / "repo"
BALLKIT = REPO / "ball_training"

# Model capacity factors relative to a "medium" detector (activation-dominated VRAM).
MODEL_FACTOR = {"n": 0.45, "s": 0.6, "m": 1.0, "l": 1.3, "x": 1.7}
# Reference: an "m" detector at imgsz 640 costs ~0.69 GB per sample (train, with AdamW state).
PER_SAMPLE_640 = 0.69
LADDER = ["x", "l", "m", "s"]        # tried largest-first
RESOLUTIONS = [1280, 1024, 896, 640]  # tried largest-first


def log(msg: str) -> None:
    print(f"[cluster_train] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(cwd) if cwd else None)


def is_ddp_child() -> bool:
    # Ultralytics/ torchrun relaunch this file once per GPU; children must not redo downloads.
    return os.environ.get("LOCAL_RANK") is not None


# ------------------------------------------------------------------ setup steps

def clone_repo() -> None:
    if (REPO / ".git").exists():
        return
    log(f"cloning helper repo ({REPO_BRANCH}) -> {REPO}")
    DATA.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, REPO])


def bootstrap_python() -> None:
    """Fail fast on the two things pip cannot fix: an old python and a missing git."""
    if sys.version_info < MIN_PY:
        raise SystemExit(
            f"python {'.'.join(map(str, MIN_PY))}+ is required (ultralytics/torch), found "
            f"{sys.version.split()[0]}. Load a newer python (module/conda), then rerun."
        )
    if not shutil.which("git"):
        raise SystemExit("git is required (helper clone + SoccerNet source). Load/install git, then rerun.")


def deps_present(python: str) -> bool:
    probe = subprocess.run([python, "-c", DEP_CHECK], capture_output=True)
    return probe.returncode == 0


def install_into(python: str) -> None:
    pip = [python, "-m", "pip", "install", "-q"]
    index = os.environ.get("TORCH_PIP_INDEX", "https://download.pytorch.org/whl/cu128")
    sh([*pip, "torch==2.11.0", "torchvision==0.26.0", "--index-url", index])
    sh([*pip, "ultralytics>=8.4.188", "opencv-python-headless", "scipy",
        "huggingface_hub", "nvidia-ml-py", "-r", str(REPO / "requirements-possession.txt")])


def ensure_deps() -> None:
    if deps_present(sys.executable):
        return
    log("installing python deps (few minutes)")
    try:
        install_into(sys.executable)
    except subprocess.CalledProcessError:
        if sys.prefix != sys.base_prefix:
            raise  # already inside a venv/conda env: the failure is real, show it
        log("system python refuses pip (externally managed) - bootstrapping ball_data/.venv")
        DATA.mkdir(parents=True, exist_ok=True)
        venv = DATA / ".venv"
        sh([sys.executable, "-m", "venv", str(venv)])
        vpy = str(venv / "bin" / "python")
        sh([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"])
        install_into(vpy)
        log("re-executing this script inside the venv")
        os.execv(vpy, [vpy, *sys.argv])
    if not deps_present(sys.executable):  # execv did not happen and imports still fail
        raise SystemExit("deps installed but imports still fail - check the pip log above")


def gpu_free_gb() -> dict[int, float]:
    """Free VRAM per CUDA index, via NVML (no CUDA context, safe next to llama-server)."""
    import torch

    n = torch.cuda.device_count()
    out: dict[int, float] = {}
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
    # With CUDA_VISIBLE_DEVICES set, torch index maps through the visible list.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        try:
            return int(visible.split(",")[cuda_index])
        except (ValueError, IndexError):
            return cuda_index
    return cuda_index


def pick_config(device_ids: list[int], per_gpu_free: dict[int, float], force_model: str | None) -> tuple[str, int, int]:
    """Return (model_name, imgsz, per-gpu batch) - the biggest thing that fits."""
    budget = min(per_gpu_free[d] for d in device_ids) * 0.82
    sizes = [force_model] if force_model else [f"yolo26{s}" for s in LADDER]
    for model in sizes:
        factor = MODEL_FACTOR.get(model[-1], 1.0)
        for imgsz in RESOLUTIONS:
            per_sample = PER_SAMPLE_640 * factor * (imgsz / 640) ** 2
            batch = int(budget / per_sample)
            if batch >= 2:
                return model, imgsz, max(2, min(batch, 16))
    # nothing fits at batch >= 2: take the smallest with batch 1
    model = sizes[-1]
    factor = MODEL_FACTOR.get(model[-1], 1.0)
    return model, 640, max(1, int(budget / (PER_SAMPLE_640 * factor)))


def check_disk() -> None:
    free_gb = shutil.disk_usage(DATA if DATA.exists() else HERE).free / 1e9
    if free_gb < 25:
        log(f"WARNING: {free_gb:.0f} GB free - datasets + runs want ~25 GB")
    else:
        log(f"disk ok: {free_gb:.0f} GB free")


def download_sources() -> None:
    ext = BALLKIT / "external"
    ext.mkdir(parents=True, exist_ok=True)
    sources = (
        ("SoccerNet v3 H250 (CC-BY-4.0)", ext / "sn_h250" / "YOLO",
         ["git", "clone", "https://github.com/kmouts/SoccerNet_v3_H250", str(ext / "sn_h250")]),
        ("infactory-ai/ball-tracking (CC-BY-NC-4.0)", ext / "infactory_ball" / "data",
         ("hf", "infactory-ai/ball-tracking")),
        ("martinjolif/football-ball-detection (CC-BY-4.0)", ext / "martinjolif" / "data",
         ("hf", "martinjolif/football-ball-detection")),
    )
    from huggingface_hub import snapshot_download

    for name, marker, cmd in sources:
        if marker.exists():
            log(f"skip {name}: present")
            continue
        log(f"fetching {name} ...")
        try:
            if isinstance(cmd, tuple) and cmd[0] == "hf":
                snapshot_download(repo_id=cmd[1], repo_type="dataset", local_dir=str(marker.parent))
            else:
                sh(cmd)
        except Exception as exc:  # keep completed sources, fail loudly with guidance
            raise SystemExit(2) from exc


def build_dataset(force: bool) -> None:
    data_yaml = BALLKIT / "dataset" / "data.yaml"
    if data_yaml.exists() and not force:
        log("dataset already built (delete ball_data/repo/ball_training/dataset to rebuild)")
        return
    sh([sys.executable, str(BALLKIT / "build_dataset.py")])


# ------------------------------------------------------------------ train + eval

def train(model: str, imgsz: int, batch: int, device: str, epochs: int, patience: int,
          workers: int, name: str) -> Path | None:
    from ultralytics import YOLO

    run_dir = BALLKIT / "runs" / name
    if run_dir.exists() and (run_dir / "args.yaml").exists():
        log(f"skip training {name}: already started")
        return run_dir
    weights = model if model.endswith(".pt") else f"{model}.pt"
    log(f"training {weights}: imgsz={imgsz} batch={batch}/gpu epochs<={epochs} patience={patience} device={device}")
    results = YOLO(weights).train(
        data=str(BALLKIT / "dataset" / "data.yaml"),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=device,
        patience=patience,
        project=str(BALLKIT / "runs"),
        name=name,
        seed=0,
        val=True,
        plots=True,
        save_period=5,
        cos_lr=True,
        # tiny-object recipe (no rotation/perspective/shear, capped scale, fliplr only)
        degrees=0.0, translate=0.2, scale=0.25, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        mixup=0.0, copy_paste=0.0, erasing=0.2,
        mosaic=1.0, close_mosaic=min(15, max(5, epochs // 8)), multi_scale=False,
    )
    return Path(str(results.save_dir))


def evaluate(run_dir: Path, imgsz: int, device: str) -> dict | None:
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        return None
    json_out = BALLKIT / "eval" / f"server_{run_dir.name}.json"
    json_out.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, str(BALLKIT / "eval_detector.py"), "--weights", str(best),
           "--split", str(BALLKIT / "dataset"), "--subset", "val", "--imgsz", str(imgsz),
           "--device", str(device).split(",")[0], "--json-out", str(json_out)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        log(f"eval failed for {run_dir.name}")
        return None
    return json.loads(json_out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ main

def parse_device(spec: str) -> list[int]:
    return [int(x) for x in re.split(r"[,\s]+", spec.strip()) if x != ""]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help='GPUs to use, e.g. "0,1,2,3" or "1,3" (default: all with room)')
    p.add_argument("--min-free-gb", type=float, default=8.0, help="minimum free VRAM per GPU to consider it")
    p.add_argument("--model", default=None, help="force a model stem (yolo26x, yolo26l, ...) instead of auto-pick")
    p.add_argument("--imgsz", type=int, default=None, help="force resolution instead of auto-pick")
    p.add_argument("--batch", type=int, default=None, help="force per-GPU batch instead of auto-pick")
    p.add_argument("--epochs", type=int, default=120, help="hard cap; patience early-stops")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force-build", action="store_true", help="rebuild merged dataset even if present")
    p.add_argument("--empty-check", action="store_true", help="print per-GPU free memory and exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = p.parse_args()

    t0 = time.time()
    bootstrap_python()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    clone_repo()
    ensure_deps()
    free = gpu_free_gb()
    for i in sorted(free):
        log(f"GPU{i}: {free[i]:.1f} GB free")
    if args.empty_check:
        return 0

    if args.device:
        ids = parse_device(args.device)
    else:
        ids = [i for i in sorted(free) if free[i] >= args.min_free_gb]
    if not ids:
        raise SystemExit(f"no GPU has >= {args.min_free_gb} GB free right now. Free one up (note the "
                         f"llama-server holding ~22 GB per card), or lower --min-free-gb.")

    model, imgsz, batch = pick_config(ids, free, args.model)
    if args.imgsz:
        imgsz = args.imgsz
    if args.batch:
        batch = args.batch
    device = ",".join(str(i) for i in ids)
    name = f"ball_{''.join(ch for ch in model if ch.isalnum())}_{imgsz}_clu"
    log(f"config: {model} @ imgsz {imgsz}, batch {batch}/gpu on GPU[{device}] "
        f"(tightest card: {min(free[i] for i in ids):.1f} GB free)")

    if args.dry_run:
        log("dry-run: nothing executed")
        return 0

    if not is_ddp_child():
        check_disk()
        download_sources()
        build_dataset(args.force_build)

    run_dir = train(model, imgsz, batch, device, args.epochs, args.patience, args.workers, name)
    if not is_ddp_child():
        if run_dir is None:
            run_dir = BALLKIT / "runs" / name
        log("evaluating best.pt on the val split (ball-specific, IoU 0.5)")
        report = evaluate(run_dir, imgsz, device.split(",")[0])
        if report:
            t = report["per_threshold"].get("0.2", {})
            log(f"val F1@0.2={t.get('f1')} recall={t.get('recall')} precision={t.get('precision')} "
                f"centre_px_median={t.get('centre_px_median')}")
        log(f"weights:  {run_dir / 'weights' / 'best.pt'}")
        log(f"curves:   {run_dir / 'results.png'}")
        log(f"numbers:  {BALLKIT / 'eval' / f'server_{name}.json'}")
        log(f"done in {(time.time() - t0) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
