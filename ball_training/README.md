# Ball detector retraining (YOLO11s)

Problem: the pipeline's stock COCO `yolo11x.pt` (`sports ball`, class 32) misses the 6–15 px
ball in the tactical wide view — the user sees the ball, the detector does not.

## Result (val split, ball-specific metrics from `eval_detector.py`, IoU 0.5)

Home-match frames (`mt__`, 300 images), at the thresholds the pipeline cares about:

| conf | baseline R / P | new R / P | baseline frames w/o detection | new frames w/o detection |
|------|----------------|-----------|-------------------------------|--------------------------|
| 0.05 | 0.62 / 0.21    | 0.72 / 0.59 | 115 | 85 |
| 0.10 | 0.60 / 0.29    | 0.70 / 0.67 | 121 | 89 |
| 0.20 | 0.55 / 0.36    | 0.70 / 0.72 | 136 | 90 |
| 0.30 | 0.48 / 0.42    | 0.70 / 0.77 | 157 | 91 |

Centre error of matched balls: median **0.55 px** (sub-pixel at 1080p).
Ultralytics val on the full split: mAP50 **0.699**, mAP50-95 **0.352**.

Raw JSON: `ball_training/eval/*.json`. Weights: `ball_training/weights/ball_yolo11s_best.pt`
(also `runs/ball_s1280_v2/weights/best.pt`).

## Dataset (ball only, class `ball`)

`build_dataset.py` merges four sources into `dataset/` with a **time-block split with guard
band** (min train/val gap on match frames 2.1 s — no neighbouring-frame leakage):

- **SoccerNet v3 H250** (`external/sn_h250`) — broadcast long shots in YOLO format, ball = class 0.
- **infactory-ai/ball-tracking** (`external/infactory_ball`) — human-verified visible frames plus
  true `not_visible` hard negatives.
- **martinjolif/football-ball-detection** (`external/martinjolif`) — very tiny balls, official split.
- Home match (`match/` via `label_match.py`) — the deployment domain itself: ball projected from
  the ATD ball track through the per-frame homography; single sequential decode pass
  (~5 min for the whole video vs ~14 h with random seeks).

Manifest: 23,097 train (19,406 pos / 3,691 neg), 2,902 val (2,248 pos / 654 neg).
`audit_labels.py` verifies label geometry numerically (no image viewing).

### Getting the data (none of it is committed — raw data, dataset and runs are git-ignored)

```bash
cd ball_training/external

# 1. SoccerNet v3 H250 (CC-BY-4.0): Zenodo 10.5281/zenodo.7808511 /
#    https://github.com/kmouts/SoccerNet_v3_H250
#    expected layout after unpack: sn_h250/YOLO/{train,valid,test}/{images,labels}
git clone https://github.com/kmouts/SoccerNet_v3_H250 sn_h250

# 2. infactory ball-tracking (CC-BY-NC-4.0): https://huggingface.co/datasets/infactory-ai/ball-tracking
hf download infactory-ai/ball-tracking --repo-type dataset --local-dir infactory_ball

# 3. martinjolif football-ball-detection (CC-BY-4.0):
#    https://huggingface.co/datasets/martinjolif/football-ball-detection
hf download martinjolif/football-ball-detection --repo-type dataset --local-dir martinjolif
```

The home-match source is the gated challenge bundle (`data/tactical/BAR-ATH_tactical.mp4` +
ATD + homography, see `DATA.md`); label it with:

```bash
.venv/bin/python ball_training/label_match.py        # ~5 min, writes ball_training/match/
```

## Training

`train_ball.py` — YOLO11s @ 1280, augmentation tuned for a tiny ball (no rotation/shear/
perspective, low `scale`, fliplr only, mosaic with close_mosaic, no mixup/copy_paste).

Final run `runs/ball_s1280_v2`: 8 epochs total, batch 6, imgsz 1280, RTX 5070 Ti.
(NOTE: the original 30-epoch plan was cut to 8 after loss curves showed convergence;
`epochs` had to be patched inside `last.pt` because `ultralytics` resume ignores the
`--epochs` override.) Epoch-wise mAP50: 0.54 → 0.55 → 0.54 → 0.60 → 0.65 → 0.68 → 0.70 → 0.70.

## Reproduce / continue

```bash
# 0. environment
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-perception.txt   # torch + ultralytics + cv2
# base weights: ultralytics downloads yolo11s.pt / yolo11x.pt on first use

# 1. external datasets + match labels (see "Getting the data" above)

# 2. build the merged dataset
.venv/bin/python ball_training/build_dataset.py --with-match

# 3. train (or --resume from runs/.../last.pt)
.venv/bin/python ball_training/train_ball.py --name ball_s1280_v3 --epochs 8 --batch 6

# 4. score any checkpoint, with or without --coco-ball for stock COCO weights
.venv/bin/python ball_training/eval_detector.py --weights ball_training/weights/ball_yolo11s_best.pt \
    --imgsz 1280 --json-out ball_training/eval/eval_new.json

# 5. render a head-to-head video (green = this model, red = COCO baseline)
.venv/bin/python ball_training/visualize.py --start-sec 1638 --seconds 30

## Wiring into the pipeline

The pipeline now runs two YOLO instances when asked:

```bash
.venv/bin/python -m possession_lab perceive --ball-model ball_training/weights/ball_yolo11s_best.pt
.venv/bin/python -m possession_lab fusion
.venv/bin/python -m possession_lab viewer
```

`--ball-model` (see `possession_lab/PERCEPTION.md`) keeps `yolo11x.pt` for people (COCO class 0)
and takes the ball stream from the ball-only weights (re-stamped to COCO id 32, so the tracker,
fusion and viewer are unchanged).  `VideoBallTracker` was tuned for it: a confident detection
(≥ 0.35) within 5 m overrides a sloppy ATD anchor, and the anchor-agreement confidence floor is
0.06.  In the public window this lifted ball-visible frames from 72.9% to 79.6% with fusion
metrics unchanged (spellF1 0.6519, macroF1 0.9469 vs baseline 0.9466).
