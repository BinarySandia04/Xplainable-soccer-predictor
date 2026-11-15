# Reproducing the submission

This document describes how to regenerate the submitted possession CSV from the organisers'
data, bit for bit, and lists the exact artefacts the pipeline consumes and produces.

The submission CSV is `frame,label` over all 199,319 frames of `BAR-ATH_tactical-001.mp4`,
labels `H` / `A` / `D` / `X`. Upload it named `<team_name>.csv`.

## Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # CPU inference for fusion/transitions
.venv/bin/python -m pytest possession_lab -q     # 329 passed (+4 subtests)
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
[`champion-run` release](https://github.com/BinarySandia04/Xplainable-soccer-predictor/releases/tag/champion-run),
along with the full-match perception cache and the fusion pool, so the chain can be viewed or
reproduced without retraining anything:

```bash
gh release download champion-run --dir .     # or pick single assets with --pattern
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

## Viewing the real detections on any PC (no GPU, no model needed)

The heavy step is perception (YOLO26x over all 199,319 frames). Its finished output is shipped
in the release, so a personal PC can play the actual ball and player detections without running
any model — everything after perception is pure CPU:

```bash
# requires the gated bundle unpacked under data/ (tactical video, tracking, homography, tags)
gh release download champion-run --pattern "perception_new_fixed.npz" --dir possession_lab/outputs
gh release download champion-run --pattern "fusion.npz"               --dir possession_lab/outputs
gh release download champion-run --pattern "outputs_view_bundle.tar.gz" --dir /tmp
tar xzf /tmp/outputs_view_bundle.tar.gz -C possession_lab/outputs/

python -m possession_lab viewer \
    --perception  possession_lab/outputs/perception_new_fixed.npz \
    --fusion      possession_lab/outputs/fusion.npz \
    --predictions possession_lab/outputs/transition_predictions.csv
```

The viewer plays the tactical video with the detected ball, every player/goalkeeper/referee box,
the possession spell per frame and the explanation layer.

## The chain

Only step 1 needs a GPU; everything after it is deterministic CPU code (no RNG in the pipeline).

```bash
P=possession_lab/outputs

# 1. Perception (GPU): ball + players over the tactical video.
#    Run farm_perceive.py (SLURM wrapper: farm_perceive.sbatch) against the two weights above.
#    The homography/ATD sidecars must share the video's stem (see FARM_README.txt).
python farm_perceive.py \
    --video data/tactical/BAR-ATH_tactical-001.mp4 \
    --ball-weights  ball_yolo26x_1024_clu.pt \
    --players-weights players_yolo26x_1024_clu.pt \
    --output-dir possession_lab/outputs

# 2. Cache rescue (CPU): keeper teams by the goal each keeper defends, duplicated
#    keeper boxes merged, officials flagged.  Writes <cache>_fixed.npz.
#    (Rename the merged cache to perception_new.npz first, as used below.)
.venv/bin/python -m possession_lab perception-fix \
    --perception $P/perception_new.npz --out $P/perception_new_fixed.npz

# 3. Fusion: multi-source evidence pool (tracking, tags, events, perception).
.venv/bin/python -m possession_lab fusion --perception $P/perception_new_fixed.npz

# 4. Transition layer: possession spells, dead-ball locks, tag/possession revival,
#    no-possession-change-without-a-touch.
.venv/bin/python -m possession_lab transitions \
    --perception $P/perception_new_fixed.npz --fusion $P/fusion.npz

# 5. Explanations (viewer/report artefacts; not needed for the CSV).
.venv/bin/python -m possession_lab explain \
    --perception $P/perception_new_fixed.npz \
    --predictions $P/transition_predictions.csv

# 6. Validate the exact file you will upload.
cp $P/transition_predictions.csv <team_name>.csv
python evaluation/validate_submission.py <team_name>.csv
```

## Expected checksums

`sha256sum` of the intermediate and final artefacts of the champion run:

| artefact | SHA-256 |
|---|---|
| `possession_lab/outputs/perception_new_fixed.npz` | `9d63f358c035fba6e80ca959f913ce90d3cae34f5e8115e221158dd2d7465bf1` |
| `possession_lab/outputs/fusion.npz` | `fb11ccb7f8e91d5dc79c2bf23f4f7445af4bf846b61c787e682576d2c9354117` |
| `possession_lab/outputs/transition_predictions.csv` | `0bb5500b69850e6d36f2e477078772a9010d6ce1449d2d6a6db81ebbb5b86ba6` |
| `possession_lab/outputs/explanations.json` | `f0a455a80a369eed7453c58c736d7684b45554ddd5c0417c10d3bbcc0fed59b1` |

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
