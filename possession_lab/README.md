# Possession Lab — BAR-ATH

A data-only, gap-aware hidden Markov model for the **More than a Hack 2027 — Phase 1: Ball
Possession** task. It is a separate implementation from the earlier video-tracking experiments:
no MOSSE, no GrayDiff, no NanoTrack, no broadcast-video indexing, and no private full-match truth.

The model reads the tactical ATD, smart tags, and homography diagnostics, decodes the two graded
halves, and writes the required `frame,label` submission. The public 5-minute ground truth is used
only as a correctness check.

## Quick start

Use Python 3.12.3, the version used for the validated run. The pinned environment includes the
runtime and plotting dependencies; no GPU, notebook, or frontend toolchain is required.

```bash
# From the repository root
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-possession.txt

# Unpack the gated bundle under data/ (see ../data/README.md), then regenerate everything
.venv/bin/python -m possession_lab all --bootstrap-n 2000

# Verify the implementation and the submission format
.venv/bin/python -m unittest discover -s possession_lab/tests -v
.venv/bin/python evaluation/validate_submission.py possession_lab/outputs/predictions.csv
```

Inference requires these four files under `data/tactical/`:

```text
BAR-ATH_tactical_atd.txt
BAR-ATH_tactical_atd.xml
BAR-ATH_tactical_smart_tagging.xml
BAR-ATH_tactical_homography.csv
```

They are gated challenge assets and are not stored in Git. Obtain them through the enrollment
process described in [DATA.md](../DATA.md); keep the supplied names and paths. The tactical MP4
is only required for the optional local reviewer, not to reproduce predictions or metrics.
Alternatively, set `MTAH_DATA_DIR` to the folder containing `tactical/`.

Outputs are written to `possession_lab/outputs/` (git-ignored):

```
predictions.csv          199,319 rows: frame,label
explanations.json        per-switch "why" annotations (clickable in the viewer)
metrics_public.json      official public metrics + bootstrap
stress.csv               clean / random / long stress rows
full_stats.json          full-match spell structure
timings.json             measured CPU timings and GPU note
report.html              self-contained report with eight figures
01_data_overview.png ... 08_tag_context.png
user_truth.json          optional viewer-only UT (written by the local reviewer)
```

## Local synchronized reviewer

The viewer synchronizes the tactical MP4 with the generated `frame,label` rows, the public ground
truth, and a **user truth (UT)** that you can write by hand. It never rewrites
`predictions.csv` or the public GT. The UT exists only to score the fixed prediction inside the
viewer; the model, features, and training code never read it.

```bash
# From the repository root, with the gated tactical video under data/
.venv/bin/python -m possession_lab viewer

# No browser launch, different port, or start at another frame
.venv/bin/python -m possession_lab viewer --no-open --port 8766 --start-frame 11185
```

The default URL is `http://127.0.0.1:8765/`; the page opens at the first public GT frame
(**11,185**). Useful options:

| Option | Meaning |
|---|---|
| `--predictions PATH` | generated CSV (default `possession_lab/outputs/predictions.csv`) |
| `--video PATH` | tactical MP4 (default `data/tactical/BAR-ATH_tactical.mp4`) |
| `--gt PATH` | public GT CSV (default `evaluation/gt/BAR-ATH_possession_gt_public.csv`) |
| `--tags PATH` | smart-tag XML (default: gated tactical tags) |
| `--metrics PATH` | public metrics JSON (default `possession_lab/outputs/metrics_public.json`) |
| `--user-truth PATH` | UT JSON (default `possession_lab/outputs/user_truth.json`) |
| `--start-frame N` | initial frame; URL `?frame=N` overrides it |
| `--no-open` | do not launch a browser |
| `--port N` | local port (default `8765`) |

What is shown:

* Video, current frame, timecode, predicted label, public GT (only inside frames
  `[11,185, 18,685)`; outside it the GT is shown as `X` / “sin evaluar”), and the UT.
* Current H/A possession spell, smart-tag context, public metrics, and **UT metrics**: spell F1,
  macro F1, D-F1, accuracy, and spell counts scored only over the frames you tagged.
* Full-match overview timeline with PRED / GT / UT channels, a ±20 s detail strip, and navigation
  by label change, possession, or GT disagreement. Click or drag either timeline to seek; the
  browser's native video controls still work.
* **Decision annotations.** Every predicted state switch is a clickable tick on both timelines and
  a timestamped row in the "Decisiones · por qué" card: the `from → to` transition, the emission
  confidence of the entered state, and the ordered evidence reasons from `explain.py` (ball out of
  bounds, parked at the line, restart ramp, group walking, ball lost, `SET PIECES` tag, nearest
  player, or the transition prior) — each reason's evidence frame is itself a seek button. Filters
  all / → D / D → / H ↔ A; `j`/`k` jump between annotated switches; the selection auto-follows
  playback within ±1.5 s. Annotations come from `outputs/explanations.json`, built automatically by
  `predict` / `all` and rebuildable with `.venv/bin/python -m possession_lab explain`.
* UT editing: press **Ancla aquí** at the start of a passage, seek to its end, then press `H`, `A`,
  or `D` to write that span; without an anchor a label applies to the current frame. `X` erases
  the span and **Borrar UT** clears everything. The same `H/A/D/X` keys work anywhere on the page
  (except while typing in a text field) and every label press leaves an anchor at the end of the
  written span. Anchors are drawn on both timelines; drag one to move the label boundary. The UT is
  written to `outputs/user_truth.json` after every edit and reloaded on the next session.
* Keyboard: `Espacio` play/pause, `←/→` one frame, `Shift` + arrows one second, `[`/`]` previous/next
  label change, `,`/`.` previous/next possession, `n`/`N` next/previous GT disagreement, `j`/`k`
  previous/next annotated switch, `H/A/D/X` user-truth labels, `Home`/`End`.

The tactical timeline is exact: `frame i` is row `i`, `currentTime = (i + 0.5) / 25`. The viewer
uses the synchronized tactical video only; the unsynchronized broadcast is excluded. The 5 GB MP4
is streamed with HTTP byte ranges, so only the requested tail/header and playback ranges are read.
No new Python dependencies are needed: the server is Python's standard library HTTP server and the
frontend is plain HTML/CSS/JS.

## What the model does

Three states: **H** (Barcelona), **A** (Atletic Club), **D** (dead ball / out of play).

* **Visible ball.** A Gaussian in the distance from the ball to each team's nearest tracked player.
  The dead-ball emission is a sigmoid on the nearest-player distance plus an out-of-play bonus.
* **Dead-ball evidence.** Independent of the team Gaussians, `D` also gets support from team motion
  energy (mean tracked-player speed over one second), a ball close to a touchline, and a small
  base-rate offset. Players walk or stand still during a stoppage, so this catches dead balls where
  the ball is visible and a player is standing next to it. This is the single biggest change:
  D-F1 rises from 0.62 to 0.78 and team accuracy (both alive) from 0.948 to 0.968.
* **Speed gate.** The emission is blended with a uniform distribution using
  `1 - sigmoid((ball_speed - gate) / 2)`. The validated model uses `gate = 0`, so a stationary or
  unknown-speed ball gets a 50/50 blend; this is intentional and is required to reproduce the
  reference scores.
* **Short gaps (≤120 frames, 4.8 s).** Uniform emissions. The transition prior carries the state.
* **Long gaps (>120 frames).** The team shape around the last known ball position, the last known
  edge distance, and the smart-tag vote provide a weak geometric prior.
* **Sparse tracking (<6 visible players).** Treated as missing evidence rather than reading team
  shape from one or two lonely slots. This guard only touches full-match blackout frames; the public
  window has at least 19 visible players, so public and stress results are unchanged.
* **Loose ball (>7 m from every player).** A ball that far from everyone is loose or in flight, not
  a stoppage, so the emission is uniform and the prior carries the possession. If it is also slower
  than `2 m/s`, the team that touched it last gets 70% of the emission mass when that touch is under
  5 s old. This removed the false `D` stretches on rolling balls and recoveries.
* **Ball parked at a boundary.** A ball seen within `2.5 m` of a line at walking pace (`≤ 1.5 m/s`)
  for at least `0.8 s` — tolerating tracking flicker inside the window — gets a strong dead-ball
  bonus (`w_parked_edge = 1.5`): the throw-in / corner / goal-kick signature. It is the strongest
  single dead-ball signal measured on the public window: spell F1 **0.6074 → 0.6857** (D-F1 −0.014).
* **Transitions.** `p_stay=0.995`, `p_ha=0.0005`, `p_hd=0.001`, `p_dh=0.01`.
* **Decode.** Viterbi over each graded half with a leading 500-frame context pad. Frames outside the graded
  halves are `X`. Forward–backward posteriors are computed only for diagnostics; they never change
  the submission labels. A run shorter than 1 s that returns to the same label is absorbed afterwards:
  it is a flicker, not a possession change.

The exact parameter dictionary is `possession_lab/config.py:MODEL_PARAMS`. The implementation is in
`model.py` (emissions, dead-ball evidence, Viterbi, forward–backward), `features.py` (geometry,
motion energy, gaps, tags), `explain.py` (per-switch reason rules and the explanation payload),
`data.py` (ATD/tag/homography loaders), and `pipeline.py` (inference and
CSV writing).

The current default was retuned on the public window against the possession folds and the
missing-ball stress suite: the team Gaussian widened from `σ=3.0` to `3.5`, leaving dead ball
became ten times more likely (`p_dh` 0.001 → 0.01), the dead-ball evidence described above was
added (`w_motion=1.0`, `w_edge=0.5`, `w_dead=0.02`), the loose-ball rule was added
(`loose_R=7 m`, `loose_speed=2 m/s`, `touch_p=0.7`, `touch_age=125`), and the parked-at-boundary
signal was added (`w_parked_edge=1.5`).

Two alternatives were measured and rejected. A fully split decode (an independent two-state
alive/dead HMM combined with a two-state H/A HMM) reached only 0.547 spell F1 and 0.686 macro F1 on
the public window. Decaying the last-ball memory with the gap length lowered macro F1 to 0.846. A
Kalman/RTS ball smoother with short-gap bridging was not pursued: the public window has only 6
ball-gap runs, and 0.1% of full-match missing frames are holes shorter than 12 frames, so there is
nothing to bridge. A walk/run fraction, a global last-touch term, a smart-tag team vote, and
ball-speed gating of the dead-ball evidence were also measured and each lowered public spell F1 by
0.03–0.07.

## Results

### Public 5-minute ground truth

Public window: frames **11,185–18,684** (7,500 frames, 17 possessions).

| Metric | Value |
|---|---:|
| Spell F1 (headline) | **0.6857** |
| Macro F1 (H/A/D) | **0.8843** |
| D-F1 | **0.8441** |
| Accuracy | 0.8943 |
| Cohen's kappa | 0.8342 |
| Alive/dead F1 | 0.9610 |
| Team accuracy (both alive) | 0.9436 |
| Predicted spells | 11 (GT 17) |
| Spell F1 95% bootstrap | [0.457, 0.834] |

These are the values of the current run. The files under `reports/data/` record an earlier
parameter set and are not updated by the retune. The bootstrap interval
is wide because there are only 17 GT possessions; do not over-read small differences. The spell
count is still below 17: a few sub-second GT possessions are single deflections that the model
does not split out, and the 1 s flicker cleanup removes any that it produces.

### Missing-ball stress test

`stress.csv` removes about 40.7% of the public window. The clean row is the untouched public
window. The baseline is a centred H/A vote with no gap model.

| Mask | Method | Spell F1 | Macro F1 | D-F1 |
|---|---|---:|---:|---:|
| clean | HMM gap-aware | **0.6857** | **0.8843** | **0.8441** |
| random | HMM gap-aware | **0.4615** | **0.7976** | **0.7385** |
| long | HMM gap-aware | **0.4000** | **0.6121** | **0.5687** |

The gap-aware model beats the HMM-persistence and smoothing baselines on every mask. The long-gap
spell F1 is still slightly below the pre-retune parameter set (0.4000 vs 0.4160), and the random
mask is lower than the previous run (0.4615 vs 0.5280): the loose-ball rule changes the decode on
the unmasked frames, and short holes interact with it. The clean and long masks both improve. The
random mask is not automatically harder than the clean window; the score depends on which GT
boundaries survive, so the number is reported, not editorialised.

### Full submission

* **199,319 rows**, frames 0–199,318, labels in `{H,A,D,X}` (H 66,930 · A 50,637 · D 41,290 · X 40,462).
* 188 H/A spells (96 first half, 92 second) over 158,857 graded frames; mean 25.0 s; median 18.6 s.
* Shares: H 42.1%, A 31.9%, D 26.0%.
* Official validator: `OK predictions.csv` (format only).
* Public metrics with the pinned environment and `--bootstrap-n 2000`: spell F1 **0.6857**,
  macro F1 **0.8843**, and D-F1 **0.8441**.

Predictions, metric values, and seeded bootstrap intervals are reproducible from the same data
and pinned dependencies. Wall-clock timings in `timings.json` and the HTML report naturally vary
by machine.

A previous reference run reported 256 spells and H 41.4% / A 32.5% / D 26.1%; the pre-retune run
reported 255 spells and H 40.5% / A 32.5% / D 26.9%; the previous parameter set reported 239
spells and H 44.1% / A 34.6% / D 21.3%. The current run reports 188 spells and
H 42.1% / A 31.9% / D 26.0%: the loose-ball rule hands a slow ball that is far from
every player back to the team that touched it last, instead of calling a stoppage, so the D share
falls and more possessions are found. Separately, the earlier difference was a 747-frame tracking
blackout at the end of the first half (frames 89,905–90,651): no home players and only 1–3 away
players were tracked. The old run's NaN emissions effectively reset to H; this implementation
treats that sparse stretch as missing evidence and predicts D. Public and stress scores are
unaffected by the guard. See the reconciliation note in `outputs/report.html`.

## Data and privacy

* Uses the gated tactical **ATD**, **smart-tag XML**, and **homography CSV**.
* Broadcast video is **not** used: it is unsynchronised and adds no label evidence.
* The public GT is read from `evaluation/gt/BAR-ATH_possession_gt_public.csv` only for evaluation.
* The viewer-only UT is stored in `possession_lab/outputs/user_truth.json`. It scores the fixed
  prediction for the user and is never read by the model, features, or training code.
* No private full-match ground truth is read, written, or inferred.
* No match video frames are copied into the report.
* `data/` and `possession_lab/outputs/` are git-ignored. Do not commit the gated bundle or the
  generated full-match labels.

## CPU vs GPU

An NVIDIA RTX 5070 Ti Laptop GPU (12 GB) is available, but the model is a 3-state dynamic program over 199,319
frames plus a handful of 46-player distance arrays. Measured on CPU:

* load: ~2.8–2.9 s
* full Viterbi decode: ~1.8 s
* 12-row stress suite: ~0.7 s

A GPU would spend more time transferring arrays than computing. No GPU path is included because it
would be slower and harder to reproduce. Revisit GPU only for a pixel-space tracker or a learned
sequence model.

## Layout

```
possession_lab/
  config.py       match constants, labels, paths, exact parameters
  data.py         ATD / smart-tag / homography / GT loaders
  features.py     distances, gap, edge, last touch, sparse-player count, tags
  model.py        emissions, Viterbi, forward-backward
  explain.py      per-switch reason rules and the viewer explanation payload
  evaluate.py     official metrics, stress masks, full-match stats
  pipeline.py     full-match inference, CSV writing, preflight
  visualize.py    eight figures + HTML report
  viewer.py       local reviewer: UT editor, stats, byte-range streaming
  viewer_assets/  index.html, viewer.css, viewer.js (no build step)
  cli.py          all / predict / explain / evaluate / stress / visualize / viewer
  tests/          stdlib unittest suite
  outputs/        generated, git-ignored
```

## Skills used

* `writing-plans` — implementation plan and acceptance criteria
* `test-driven-development` — tests first for loaders, features, HMM, metrics, and pipeline
* `verification-before-completion` — reference equality, official validator, and test evidence

## Limitations

* The public GT has 17 possessions; the confidence interval is wide.
* The ATD has 46 anonymous player slots. The model never infers player identity.
* Smart tags cover 70.1% of the full recording (75.5% of the public window), and their team vote is
  weak; every throw-in is `N/A`.
* The model is tuned on the public window. The full match is a generalization check, not a
  validation set.
* Homography is used for diagnostics and pitch checks, not as a label source. The tactical
  homography is valid on 100% of frames with median reprojection error 0.24 px.
