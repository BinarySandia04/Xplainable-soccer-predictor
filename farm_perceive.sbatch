#!/bin/bash
#SBATCH --job-name=mtah-perceive
#SBATCH --output=farm_perceive-%j.log
#SBATCH --error=farm_perceive-%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Change the partition below to one that has GPUs on your cluster (`sinfo -o "%P %G"`).
#SBATCH --partition=gpu

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$HERE/farm_perceive.py" ]]; then
    cd "$HERE"
else
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
fi

PYTHON="${PYTHON:-$HOME/ball_data/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

DATASET_DIR="${DATASET_DIR:-/home/alumnes/joan.ignasi.cid/dataset/tactical}"
BALL_WEIGHTS="${BALL_WEIGHTS:-$HOME/ball_data/weights/ball_yolo26x_1024_clu.pt}"
PLAYERS_WEIGHTS="${PLAYERS_WEIGHTS:-$HOME/ball_data/player_data/weights/players_yolo26x_1024_clu.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/perception_runs}"
MATCH="${MATCH:-BAR-ATH}"
STEP="${STEP:-1}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "python=$PYTHON"
"$PYTHON" - <<'PY'
import cv2, numpy, torch, ultralytics
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible; check --gres=gpu:1 and CUDA_VISIBLE_DEVICES")
print("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0))
print("numpy", numpy.__version__, "ultralytics", ultralytics.__version__)
PY

"$PYTHON" farm_perceive.py \
    --dataset-dir "$DATASET_DIR" \
    --ball-weights "$BALL_WEIGHTS" \
    --players-weights "$PLAYERS_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --match "$MATCH" \
    --step "$STEP" \
    "$@"
