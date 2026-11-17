# Reproducing the submission

This document describes how to regenerate the submitted possession CSV from the organisers'
data, bit for bit, and lists the exact artefacts the pipeline consumes and produces.

The submission CSV is `frame,label` over all 199,319 frames of `BAR-ATH_tactical-001.mp4`,
labels `H` / `A` / `D` / `X`. Upload it named `<team_name>.csv`.

## Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # CPU inference for fusion/transitions
.venv/bin/python -m pytest possession_lab -q     # 375 passed (+4 subtests)
```

The match data lives in `data/` (tactical video, tracking, homography, ATD, smart tags).
It is **not** in this repository — it is distributed separately by the organisers.

## Model weights

All learned components were trained on **external data only**; no frame of the provided match
and no ground-truth possession label was used in any training.

| weight | role | SHA-256 |
|---|---|---|
| `ball_yolo26x_1024_clu.pt` | ball detector (YOLO26x @ 1024) | `f69ff921f8ae32eeba75aa6dc8730eefe3b4389232e0f8b0948482ea7f412053` |
| `players_yolo26x_1024_clu.pt` | player + keeper detector (YOLO26x @ 1024) | `03031012298e134eb13f4dd03c25651cc3ffc231c27f93f16fb38c4b2de04865` |

Both weights ship as assets of the
[`champion-run` release](https://github.com/BinarySandia04/Xplainable-soccer-predictor/releases/tag/champion-run).
The release intentionally contains **only the two weights**: no match data, no perception cache
and no frame-level tracking export, in line with the data rules:

```bash
gh release download champion-run --pattern "*.pt" --dir .
```

**Training provenance** — both detectors are YOLO26x @ 1024, DDP on a multi-GPU server, trained
on **external public data only** (no frame of the provided match and no possession label):

- `ball_yolo26x_1024_clu.pt` — training script `ball_training/train_cluster.py` (committed):
  - SoccerNet v3 H250 — [Zenodo 10.5281/zenodo.7808511](https://zenodo.org/records/7808511) — CC-BY-4.0
  - infactory-ai/ball-tracking — [Hugging Face](https://huggingface.co/datasets/infactory-ai/ball-tracking) — CC-BY-NC-4.0
  - martinjolif/football-ball-detection — [Hugging Face](https://huggingface.co/datasets/martinjolif/football-ball-detection) — CC-BY-4.0
- `players_yolo26x_1024_clu.pt` — training script `ball_training/train_players_cluster.py` (committed; classes: player / goalkeeper / referee):
  - SoccerNet/SN-Tracking-2023 — [Hugging Face](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023) — research terms
  - martinjolif/football-player-detection — [Hugging Face](https://huggingface.co/datasets/martinjolif/football-player-detection) — CC-BY-4.0
  - keremberke/football-object-detection — [Hugging Face](https://huggingface.co/datasets/keremberke/football-object-detection) — CC-BY-4.0
  - manot/football-players — [Hugging Face](https://huggingface.co/datasets/manot/football-players) — MIT

The stock COCO weights (`yolo11x.pt` etc.) are only used by the excluded side experiment in
`ball_training/`, which is **not** part of this chain — see that README's status banner.

## The chain

Only step 1 needs a GPU; everything after it is deterministic CPU code (no RNG in the pipeline).
The perception output is a local frame-level detection cache; it is **not** distributed because it
would reconstruct part of the match tracking.

```bash
P=possession_lab/outputs

# 0. Weights (external-data training only; no match data in the release).
gh release download champion-run --pattern "*.pt" --dir .

# 1. Perception (GPU): ball + players over the tactical video, keeper rescue included.
#    Run farm_perceive.py (SLURM wrapper: farm_perceive.sbatch) against the two weights above.
#    The homography/ATD sidecars must share the video's stem (see FARM_README.txt).
python farm_perceive.py \
    --video data/tactical/BAR-ATH_tactical-001.mp4 \
    --ball-weights  ball_yolo26x_1024_clu.pt \
    --players-weights players_yolo26x_1024_clu.pt \
    --output-dir possession_lab/outputs

# 2. Fusion: multi-source evidence pool (tracking, tags, events, perception).
.venv/bin/python -m possession_lab fusion --perception $P/perception_new_fixed.npz

# 3. Transition layer: possession spells, dead-ball locks, tag/possession revival.
.venv/bin/python -m possession_lab transitions \
    --perception $P/perception_new_fixed.npz --fusion $P/fusion.npz

# 4. Explanations (viewer/report artefacts; not needed for the CSV).
.venv/bin/python -m possession_lab explain \
    --perception $P/perception_new_fixed.npz \
    --predictions $P/transition_predictions.csv

# 5. Validate the exact file you will upload.
cp $P/transition_predictions.csv <team_name>.csv
python evaluation/validate_submission.py <team_name>.csv
```

## Expected checksums

`sha256sum` of the intermediate and final artefacts of the champion run:

| artefact | SHA-256 |
|---|---|
| `possession_lab/outputs/transition_predictions.csv` | `9d79ad4bd83327f03299e02919623ef89215145eecabf1aa0a1a9bb4a5ce66fb` |
| `possession_lab/outputs/explanations.json` | `2e563e7c044e4c198757a9620f5f77515f5de68964e756ed3557ca45330a6a70` |
| `possession_lab/outputs/perception_new_fixed.npz` (local only) | `9dddaa128dc9eb39eae8419958e4252bb5ed65a423418f33a08ac835ff855f1b` |
| `possession_lab/outputs/fusion.npz` (local only) | `e7ea998b74b963af25413d1938bf5659e56efec1f155e0a668c4dbf4dcae0369` |

The last two are reference hashes for a local run; they are not published. Perception is
deterministic given the same weights, `imgsz` (1024), device and library versions (see
`requirements-perception.txt`), but small GPU-library differences can change those intermediate
files. If your perception hash matches, steps 2–4 are pure CPU and reproduce the CSV exactly;
otherwise check the final CSV hash, which is the submission.

## Compliance notes

- **No hand labels.** Every `frame,label` row is produced by the pipeline above.
- **Training on external data only**, per the organisers' clarification — see *Model weights*.
- **No private ground truth** was used anywhere; the public 5-minute window is used only for
  correctness checks of the pipeline, never as a tuning target.
- The provided data (video, tracking, tags) never enters this repository: `data/` is
  `.gitignore`d and no derived data of the match is committed except predictions.
- No frame-level tracking export, perception cache or fusion pool is published. The release
  contains only the two weights trained on external data; the match itself is not redistributed.
