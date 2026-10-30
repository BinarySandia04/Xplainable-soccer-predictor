# Optional video perception and multi-source fusion

The validated ATD/HMM remains the default prediction path.  This document describes the optional,
independent video evidence and the transparent poll that can be layered on top of it.  The public
ground truth is used only to **evaluate** the result; it is never an online input.

## Why a second pipeline

The ATD has a ball on 56.94% of frames and no ball on 40.83%.  The HMM bridges short holes and
falls back to player shape, but there are passages where an independent view of the video could
carry information the ATD lacks.  The video pipeline therefore produces its own detections instead
of only re-drawing the ATD:

```text
tactical MP4
  -> YOLO11x: people + sports ball (full frame, 1920)
  -> per-frame homography: foot points and ball centre to pitch metres
  -> player team labels: nearest ATD slot (jersey colour is deliberately not claimed)
  -> physical temporal gate: reject ball detections unreachable at football speed
  -> own ball tracker: YOLO ball + white-blob fallback + motion prediction
  -> compact perception.npz cache
                              \
ATD ball + players -----------> multi-source ballots -> temporal HMM -> fused file
smart tags --------------------/                          |
whistle envelope -------------/                           +-> event D overrides
goals / stoppages -----------/
```

The own tracker is **anchored to ATD** while ATD is present, so a false YOLO ball cannot pull it
away from the validated observation.  In an ATD gap it predicts through the gap and accepts YOLO or
white-blob measurements only inside a time-aware motion gate.  If it has neither an anchor nor a
usable measurement it reports `lost`; it never invents a position silently.

The gate is physical, not a ground-truth lookup.  It uses a match-data upper bound of `40 m/s`
(about `145 km/h`, a generous bound for shots); the gate allows `40 m/s * dt + 1 m` for
homography/projection error.  A detection outside that radius is rejected even when YOLO is very
confident, and a velocity prediction cannot widen the gate: the candidate must be reachable from
the last accepted point, not only from the predicted position.  Starting or reacquiring a YOLO
track needs two mutually reachable detections, so a single false positive on a crowd or advertising
board cannot seed it.  The same association is applied to the independent `yolo_ball` ballot and
to the viewer's YOLO layer; public ground truth is still used only for evaluation, never as an
online gate.  A tracker update pinned to ATD is labelled source `3`; only YOLO/white-blob
measurements are source `0`, so an anchor snap cannot be mistaken for a YOLO jump in diagnostics.

## Ball-layer source codes

| code | meaning |
|---|---|
| `0` | video/YOLO detection |
| `1` | prediction through a gap |
| `2` | lost / no estimate |
| `3` | ATD anchor |

The viewer draws the own tracker in green, the temporally gated YOLO ball as an orange square,
and the fused consensus ball as a magenta diamond.  The raw ATD, diagnostic Kalman, and production
combined layers remain available unchanged.  The YOLO layer shows only detections that passed the
physical association; the cache still retains all raw candidates for diagnostics.

## The poll

Each source emits a probability over `H` (Barcelona), `A` (Athletic), and `D` (dead ball).  The
default is a **log-opinion pool** anchored on the validated HMM Viterbi code, not on the HMM
posterior argmax.  The HMM vote has confidence `0.60 + 0.25 * HMM confidence`; the independent
sources are likelihood updates with effective weights:

| source | role | default base weight |
|---|---|---:|
| `atd_ball` | diagnostic (already in the HMM emission) | 0.02 |
| `atd_shape` | diagnostic | 0.01 |
| `video_players` | independent evidence | 0.10 |
| `video_ball` | independent evidence, strongest in ATD gaps | 0.14 |
| `yolo_ball` | independent detector evidence, physically associated in time | 0.04 |
| `hmm_atd` | prior | 1.00 |
| `tags` | diagnostic | 0.01 |
| `audio` | diagnostic | 0.02 |
| `events` | hard `D` override | 1.00 |

The effective weight is further reduced by source quality: ball confidence, player coverage,
whether ATD already supplies the ball, and sample age.  ATD, tags, and audio are deliberately weak
because they are already inside the HMM emission; counting them strongly again would double-count
evidence.  The weights are a design prior, not fitted labels.  The `linear` pool is available for
ablation, but `log` is the default.

After pooling, a small HMM with the same transition prior decodes the sequence.  Confirmed goals
and stoppages are applied as hard `D` overrides after that decode.  Frames outside the graded halves
stay `X`, exactly like the baseline.

## Commands

```bash
# One-time optional dependencies (YOLO, OpenCV, SciPy).
.venv/bin/python -m pip install -r requirements-perception.txt

# Public evaluation window at full frame rate.  About 13 minutes on an RTX 5070 Ti.
# With the dedicated ball detector (`--ball-model`), the COCO weights below are used for
# people only and the ball stream comes from the ball-only model; about 16 minutes.
.venv/bin/python -m possession_lab perceive \
  --start 11185 --stop 18685 --step 1 \
  --ball-model ball_training/weights/ball_yolo11s_best.pt \
  --perception possession_lab/outputs/perception_public.npz

# Dedicated player + ball checkpoints.  The player model's player/goalkeeper/referee classes
# are all re-stamped as COCO person so tracking, team matching and the viewer are unchanged.
# `--stage players` (or `--stage ball`) runs only one family; the farm script below uses that to
# run the two models sequentially and merge the two caches.
.venv/bin/python -m possession_lab perceive \
  --start 11185 --stop 18685 --step 1 \
  --players-model weights/players_weights/players_yolo26x_1024_clu.pt \
  --ball-model weights/moncuniserver/ball_yolo26x_1024_clu.pt \
  --imgsz 1024 --perception possession_lab/outputs/perception_models.npz

# SLURM farm: ball pass first, player pass second, then merge.  The bundle contains only the
# two package files the farm needs (no pandas/scipy).  See FARM_README.txt for the full recipe.
#   python farm_perceive.py --list
#   sbatch farm_perceive.sbatch

# Fusion with the public cache.  Writes fusion.npz, fusion_predictions.csv,
# fusion_metrics_public.json, and fusion_validation.json.
.venv/bin/python -m possession_lab fusion \
  --perception possession_lab/outputs/perception_public.npz

# Viewer: auto-loads perception.npz (or perception_public.npz) and fusion.npz.
.venv/bin/python -m possession_lab viewer
```

For a full-match pass, `--step 5` is the practical first option (about 1.2 hours); the cache is
sampled at 5 fps and the age gate prevents stale detections from becoming evidence.  A full
`--step 1` pass is possible but takes roughly six hours on the same GPU.

## What the viewer shows

The **Poll multi-fuente** card shows, for the current frame:

* the final fused label and its posterior HMM confidence; the summary bars show the per-frame
  fused probability before the temporal decode;
* the baseline HMM label for comparison;
* one row per source, with its role, status, probability bars for H/A/D, effective weight, and a
  short reason (video-tracker origin, nearest-player distances, gap length, tag vote, whistle
  envelope, or event);
* the agreement/disagreement between the fused label and the validated baseline.

The new map and video layers show the detector and fusion layers, not just the final label.  The
YOLO ball layer is off by default and now applies the same physical association as the ballot, so
it cannot jump to an unreachable false positive.  The cache still stores every raw candidate for
offline inspection, but raw candidates are never drawn as an accepted ball.  The YOLO player rings
can be toggled independently.  The browser never receives the 199k-frame tracking table; it fetches
small per-frame payloads.

## Physical-gate check

On the 7,500-frame public cache, the raw best YOLO candidate had 566 consecutive frame-to-frame
displacements above `2.6 m` (the 25 fps gate), with a maximum of `84.9 m` in one frame.  After the
physical association, 4,497 frames have an accepted YOLO observation and no consecutive accepted
pair exceeds the gate; the maximum observed displacement is `1.86 m` per frame.  The own tracker's
source-`0` measurements are also below the gate (4,700 observations, 4,340 consecutive pairs,
maximum `2.55 m`, zero violations).  These numbers are diagnostics from the public window, not an
online input.  The fused label is unchanged, which is the intended safety result: the gate fixes
continuity without promoting a noisy detector over the validated HMM.

## Current public-window check (development result)

The public 7,500-frame window is an evaluation set only.  With the current conservative weights,
the fused and baseline files are deliberately kept separate:

| run | accuracy | macro F1 | spell F1 | D-F1 | segments |
|---|---:|---:|---:|---:|---:|
| validated baseline | 94.17% | 94.66% | 65.19% | 97.27% | 15 |
| fusion, no video cache | 94.20% | 94.69% | 65.19% | 97.33% | 14 |
| fusion with public video cache | 94.20% | 94.69% | 65.19% | 97.33% | 14 |

The video ballots are currently **diagnostic evidence**, not a proven improvement.  Their argmax
is much less accurate than the validated baseline (about 60% versus 94% on the public window), and
the HMM prior correctly refuses to hand them the decision.  The fused file differs from the
baseline on only a few frames.  This is the intended safety property: a noisy detector cannot
silently replace a validated model.  Do not promote `fusion_predictions.csv` to the default unless
a later evaluation shows a repeatable gain, especially in spell F1 rather than frame accuracy.

## Known limitations

* YOLO's COCO sports-ball class produces false positives in crowds and near advertising boards;
  the pitch gate, the `40 m/s + 1 m` physical gate, the two-detection reacquisition rule, and the
  ATD anchor suppress most of them.  The gate is deliberately conservative: a real ball faster
  than the bound would be dropped rather than allowed to teleport.
* Player team labels come from the nearest ATD slot.  Jersey-colour classification is a possible
  future source but is not enabled, so no actor identity is claimed.
* The public cache covers only frames `[11,185, 18,685)`.  Outside that window the viewer and
  fusion treat the video source as stale and abstain.
* The own tracker is intentionally causal.  It does not use future frames, public GT, or offline
  smoothing.
