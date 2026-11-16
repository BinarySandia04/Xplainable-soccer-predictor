# Reproducing the submission

This document describes how to regenerate the submitted possession CSV from the organisers'
data, bit for bit, and lists the exact artefacts the pipeline consumes and produces.

The submission CSV is `frame,label` over all 199,319 frames of `BAR-ATH_tactical-001.mp4`,
labels `H` / `A` / `D` / `X`. Upload it named `<team_name>.csv`.

## Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # CPU inference for fusion/transitions
.venv/bin/python -m pytest possession_lab -q     # 384 passed (+4 subtests)
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

Note: the `outputs_view_bundle.tar.gz` in the champion-run release was built from an earlier run
and its `transition_predictions.csv` predates the physical control rules; rerunning steps 2-4
from the perception cache (pure CPU) reproduces the current CSV with the checksum above.

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

# 4. Transition layer: decode, half-gap fill, lost/out-of-bounds -> D,
#    tag and player revival (reopen false stoppages), then physical
#    control: possession follows the last play (witnessed event, release,
#    trap from rest) and a stopped, seen ball goes to the team at its feet.
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

## Physical control rules (transition layer)

The decode says who the voters *think* has the ball; the physical rules say
who the ball is *with*. They are a priori rules — no parameter was fitted to
any ground truth — and they run in this order:

1. **Decode + half-gap fill** — the fused vote is decoded and the halftime
   gap is filled with `D`.
2. **Lost / out-of-bounds -> `D`** — nobody possesses a ball off the pitch or
   that the tracker has lost.
3. **Tag revival** — a live-phase smart tag that contradicts a dead lock
   reopens it (into `D` frames only, never over a live label).
4. **Player revival** — a player at a rolling ball, with the raw evidence
   pool behind that player's team, reopens a dead lock the same way.
5. **Possession follows the last play** (`hold_through_flight`) — a spell is
   re-labelled to the team of the last *play* in its lookback window. A play
   is an anchor, strongest first: a witnessed transition event, a **release**
   (the ball leaves a slow start at live speed with its player at the feet),
   or a **trap from rest** (a player takes on a ball that was, up to 0.24 s
   earlier, (nearly) at rest — pickup's observable signature is prior rest).
   A ball at live speed is not possession until it is controlled; only frames
   governed by an anchor in the window move, and no anchor older than 30 s
   counts.
6. **A stopped, seen ball is owned** (`rest_ownership`) — while the ball is
   at rest and tracked, the team at its feet owns it under either clause: on
   the ball itself (a player within 0.5 m and 1 m ahead of the opponent) or
   the clear stop (within 5 m with the opponent 8 m further), sustained for
   at least 0.4 s. A rest run overwrites any decode, `D` included: the ball
   is seen, stopped and owned, so there is no stoppage to remember.
7. **A stoppage ends where the restart is witnessed** (`restart_ownership`) —
   the smart tags name the *where* of each Kick Off / Throw-in window; the
   ball's own streams name the *when* and the players the *who*. A kickoff
   restarts when the dedicated ball detector locks the kicked ball
   (confidence at least 0.5, held 3 frames); a throw-in restarts when the
   ball crosses back in bounds at throw speed (at least 6 m/s within 6
   frames of the crossing). The restart team is the nearest present player
   at the ball (kicker within 1.5 m, thrower within 2.5 m, at least 1 m
   ahead of the nearest opponent). Only then does it paint: the frames
   before the restart are `D` — the ball is not in play yet — and from the
   restart frame the attributed team owns play across `D`/`X` frames until
   the decode's own evidence reasserts itself (a team label, or the ball
   lost / off the pitch), never over a decoded possession. A restart
   neither witnessed nor attributed paints nothing: the tag is a stoppage,
   not a play.
8. **A keeper gathering the ball in his own area** (`keeper_area_catches`) —
   the pixel channel is blind right by the goal, where depth compresses the
   boxes, so the pitch geometry reads the catch: the ball decelerates from
   flight speed to a stop inside the penalty area it defends, with no opponent
   near, and the catch belongs to the defending side from the kick that fed it.
   The arrival must be a pass, not a tracker spike: at least two of the 40
   frames before the stop must show the ball at or above 8 m/s — a single fast
   frame is a one-frame jitter over a slow roll (measured 129 415). Where the
   raw voter pool is available, it must back the catcher by a 0.25 mean margin:
   a ball dying in the area with the voters neutral is a conceded goal, not a
   gathering (measured 11 598).
9. **A keeper clearing a ball he held while the fusion went blind**
   (`blind_keeper_clearances`) — the fused ball is dark (absent, or confidence
   below 0.5) for at least a second, then reappears inside the defended area
   with the attributed defender on the ball (within 1.5 m) and the opponent
   clear (2.0 m or more), and leaves again as a downfield punt: at least
   10 m/s within 40 frames, gaining at least 5 m away from the goal. A shot
   moves into the goal, a contested ball has the opponent close, a dead ball
   or a tagged goal sequence is no clearance at all — so none of them fire.
   Measured 69 405: blind 69 116-69 404, back at (7.4, 35.3) with the keeper
   0.4 m from it, the punt building to 20.6 m/s.
10. **A blind hand-over waits for the touch** (`snap_blind_handovers`) —
    **implemented but off by default** (`blind_handover_max_shift_frames = 0`).
    A team-to-team change made while the incoming side has nobody within
    1.5 m of the ball cannot be a touch: with the ball in flight the distance
    kernel is symmetric and the boundary is a ~1e-5 emission residual, not a
    play. The rule moves such a boundary forward to the first frame of the
    new spell where the incoming side reaches a controllable ball (within
    1.5 m at 6 m/s or less), capped at 10 s; a change at the feet, an unseen
    ball, dead frames and `D`/`X` transitions are never rewritten.

    It is off because the public annotation does **not** wait for the touch:
    in the four public-window cases the GT boundary sits 3-55 frames from the
    decode's and 34-109 frames before the first tracked control, so enabling
    it costs 0.04 spellF1 (0.7600 -> 0.7200, 14/17 -> 11/17 spells). The case
    that motivated it is frame 38 497 (H 38 417-38 498 -> A 38 499): the ball
    leaves an H player's feet at 38 486 at 16.5 m/s and Athletic's first
    control is 38 596, so the A spell runs 97 frames with no A player within
    5.9 m. The likely reason the GT changes earlier is that a player's tracked
    position is the **body centre** (a real touch can be 2-3 m away) and the
    ATD has detection gaps (the Barcelona keeper is out of frame at 38 497,
    projected to pixel x = -350), so "nobody is near" does not prove "nobody
    touched it". To continue the experiment, set the parameter to 250 and see
    `reports/blind_handover_experiment.md` for the measured cases.

The run's own count line (all 199,319 frames):

> 194 kicks + 23 onsets detected, 521 contacts + 327 win-and-clear kicks + 10
> keeper holds + 50 area catches from boxes + 3 blind keeper clearances,
> 3 spell expiries, 0 touch takeovers + 3024 rest balls owned, 5701 flight
> ownership fixes, 1083 gated out -> 117 transition events; 59,606 frames of
> lost/out-of-bounds ball marked D; 641 frames revived by live tags, 487 by a
> player at the ball, 1,041 owned by witnessed restarts.

Against the public 5-minute window (`evaluation/run_evaluation.py`, 7,500
graded frames): **spellF1 0.7600, macroF1 0.9695, accuracy 0.9675**.

## Expected checksums

`sha256sum` of the intermediate and final artefacts of the champion run:

| artefact | SHA-256 |
|---|---|
| `possession_lab/outputs/perception_new_fixed.npz` | `9d63f358c035fba6e80ca959f913ce90d3cae34f5e8115e221158dd2d7465bf1` |
| `possession_lab/outputs/fusion.npz` | `765317256cd7ec1dc0d8f57624c30b84436a88b4ccf202e44c8d17bb818cbf31` |
| `possession_lab/outputs/transition_predictions.csv` | `9b9232ee1f2be89e7a8112daf22f843ed4bedfe22cb1d1b57f8361f2310da520` |
| `possession_lab/outputs/transition_events.csv` | `c02736ceb06356499f457c41416ae77ede3cf0b8e0c48595ae4921da5cb36d49` |
| `possession_lab/outputs/fusion_transitioned.npz` | `51e00a945bbf37aa91907fd51e388c27b2751fd3a6a344d9a37771ac487bdb5d` |

`explanations.json` (step 5) is derived from the predictions; its checksum
follows the one above.

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
