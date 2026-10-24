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
white-blob measurements only inside a motion gate.  If it has neither an anchor nor a usable
measurement it reports `lost`; it never invents a position silently.

## Ball-layer source codes

| code | meaning |
|---|---|
| `0` | video/YOLO detection |
| `1` | prediction through a gap |
| `2` | lost / no estimate |
| `3` | ATD anchor |

The viewer draws the own tracker in green, the raw YOLO ball as an orange square, and the fused
consensus ball as a magenta diamond.  The raw ATD, diagnostic Kalman, and production combined
layers remain available unchanged.

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
| `yolo_ball` | independent raw detector evidence | 0.04 |
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
.venv/bin/python -m possession_lab perceive \
  --start 11185 --stop 18685 --step 1 \
  --perception possession_lab/outputs/perception_public.npz

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

The new map and video layers show all detector outputs, not just the final label.  The raw YOLO
ball layer is intentionally off by default; enable it to inspect false positives.  The YOLO player
rings can be toggled independently.  The browser never receives the 199k-frame tracking table; it
fetches small per-frame payloads.

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
  the tracker's pitch gate, motion gate, and ATD anchor suppress most, not all, of them.
* Player team labels come from the nearest ATD slot.  Jersey-colour classification is a possible
  future source but is not enabled, so no actor identity is claimed.
* The public cache covers only frames `[11,185, 18,685)`.  Outside that window the viewer and
  fusion treat the video source as stale and abstain.
* The own tracker is intentionally causal.  It does not use future frames, public GT, or offline
  smoothing.
