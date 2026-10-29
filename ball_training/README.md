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

---

# Player detector retraining (YOLO26, multi-GPU)

`train_players_cluster.py` is a **single self-contained file**: no repo, no git, no token, no
code hosting. Datasets and base weights come only from Hugging Face.
It trains three person classes — `player`, `goalkeeper`, `referee` — and ignores the ball,
which the detector above owns.

## Data (all public, nothing committed)

| source | licence | what it adds |
|--------|---------|--------------|
| `SoccerNet/SN-Tracking-2023` (HF) | SoccerNet research terms | 57 broadcast train games / 49 test games, MOT boxes + per-track id classes |
| `martinjolif/football-player-detection` (HF) | CC-BY-4.0 | ~600 Roboflow images, 4 classes |
| `keremberke/football-object-detection` (HF) | CC-BY-4.0 | 1,232 Roboflow images |
| `manot/football-players` (HF) | MIT | 206 Roboflow images |

SoccerNet `gt.txt` only stores track ids; `gameinfo.ini` maps each id to player / goalkeeper /
referee / ball. The script reads that mapping, drops every ball box, and links images into
`player_data/dataset/` (symlinks — no second copy). Train uses the official 57 train games,
val the official 49 test games: whole games are held out, never neighbouring frames.
SoccerNet is capped at ~24k train / ~3k val frames with a deterministic stride so the run
stays comparable to the ball run; the Roboflow sets are small and used in full.

## Training on the cluster

The script reuses `~/ball_data/.venv` (override with `PLAYER_VENV`), measures free VRAM per
GPU and trains the largest YOLO26 / resolution / batch that fits (on three idle 32 GB cards
this is typically `yolo26x` at 1024–1280), DDP across every GPU with at least 8 GB free.
Cards with far less headroom than the roomiest one are left out automatically, so the card
shared with `llama-server` cannot drag every rank down to its batch size. `llama-server` is
never touched: it is only measured. Peak disk is ~36 GB (zips are deleted after each verified
extraction; the script refuses below 35 GB free).

```bash
# local: copy the single file to the server (or paste it with nano)
scp ball_training/train_players_cluster.py user@server:~/ball_data/

# server: sanity check - plan + per-GPU VRAM, changes nothing
cd ~/ball_data
.venv/bin/python train_players_cluster.py --dry-run

# server: full run, detached
setsid nohup .venv/bin/python train_players_cluster.py > train_players.log 2>&1 &
tail -f train_players.log
```

Run it only after the ball training has finished: both scripts pick GPUs by free VRAM, and
starting this one early would put two jobs on the same cards.

The match video is auto-detected: any mp4 in `player_data/video/`, then the ball run's copy at
`~/ball_data/video/BAR-ATH_tactical.mp4` — no second 4.9 GB upload.

## Outputs (`player_data/`)

| path | contents |
|------|----------|
| `weights/players_best.pt` | stable copy of the best checkpoint |
| `runs/players_*/` | full run (`results.png`, `args.yaml`, `weights/`) |
| `eval/players_*.json` | per-class recall / precision / F1 at IoU 0.5, conf 0.05–0.5, per-source breakdown, player-count MAE |
| `inference/*_full_match_players.json` | player-count coverage over the whole match |
| `inference/*_clip_11197-11566_players.mp4` | annotated BAR-ATH clip, green player / yellow GK / magenta referee |

Re-run pieces directly:

```bash
cd ~/ball_data

# rebuild the merged dataset after changing a source
.venv/bin/python train_players_cluster.py --force-build

# resume an interrupted run
.venv/bin/python train_players_cluster.py --resume

# pin GPUs / force a model / change clip bounds
.venv/bin/python train_players_cluster.py --device 1,2,3 --model yolo26x \
    --clip-start 11197 --clip-end 11567
```
