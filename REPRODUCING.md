# Reproducing the submission

This document describes how to regenerate the submitted possession CSV from the organisers'
data, bit for bit, and lists the exact artefacts the pipeline consumes and produces.

The submission CSV is `frame,label` over all 199,319 frames of `BAR-ATH_tactical-001.mp4`,
labels `H` / `A` / `D` / `X`. Upload it named `<team_name>.csv`.

## Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # CPU inference for fusion/transitions
.venv/bin/python -m pytest possession_lab -q     # 315 passed (+4 subtests)
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

<!-- TODO(joan): upload both files as assets of a GitHub release (each 113 MB — too large for
     a plain git commit) and put the release URL here. Training provenance: external dataset(s)
     used, link + license, dates, training config. Fill in before the deadline:
     - dataset(s): <name, source, license>
     - training config / log: <path or release asset> -->

The stock COCO weights (`yolo11x.pt` etc.) are only used by the excluded side experiment in
`ball_training/`, which is **not** part of this chain — see that README's status banner.

## The chain

Only step 1 needs a GPU; everything after it is deterministic CPU code (no RNG in the pipeline).

```bash
P=possession_lab/outputs

# 1. Perception (GPU): ball + players over the tactical video, keeper rescue included.
#    Run farm_perceive.py (SLURM wrapper: farm_perceive.sbatch) against the two weights above.
#    The homography/ATD sidecars must share the video's stem (see FARM_README.txt).
python farm_perceive.py \
    --video data/tactical/BAR-ATH_tactical-001.mp4 \
    --ball-weights  <release>/ball_yolo26x_1024_clu.pt \
    --players-weights <release>/players_yolo26x_1024_clu.pt \
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
| `possession_lab/outputs/perception_new_fixed.npz` | `9dddaa128dc9eb39eae8419958e4252bb5ed65a423418f33a08ac835ff855f1b` |
| `possession_lab/outputs/fusion.npz` | `e7ea998b74b963af25413d1938bf5659e56efec1f155e0a668c4dbf4dcae0369` |
| `possession_lab/outputs/transition_predictions.csv` | `3c0694272da36a648061d72e9931422e66bed80ba96b7e465b19c261b366e59e` |
| `possession_lab/outputs/explanations.json` | `25023e5b1702613030f20ae6af61af97c51907e8bcc9c294f5e0e09130072b64` |

If your perception cache matches the checksum above, steps 2–4 are pure CPU and reproduce the
CSV exactly. Perception itself is deterministic given the same weights, `imgsz` (1024), device
and library versions (see `requirements-perception.txt`); small GPU-library differences are the
only known source of drift — check the perception checksum first if the CSV differs.

## Compliance notes

- **No hand labels.** Every `frame,label` row is produced by the pipeline above.
- **Training on external data only**, per the organisers' clarification — see *Model weights*.
- **No private ground truth** was used anywhere; the public 5-minute window is used only for
  correctness checks of the pipeline, never as a tuning target.
- The provided data (video, tracking, tags) never enters this repository: `data/` is
  `.gitignore`d and no derived data of the match is committed except predictions.
