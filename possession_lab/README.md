# Possession Lab — BAR-ATH

A data-only, gap-aware hidden Markov model for the **More than a Hack 2027 — Phase 1: Ball
Possession** task. It is a separate implementation from the earlier video-tracking experiments:
no MOSSE, no GrayDiff, no NanoTrack, no broadcast-video indexing, and no private full-match truth.

The pipeline combines four signals:

* a **causal ball tracker** that bridges only short tracking holes (≤ 1.2 s) and admits longer
  gaps instead of inventing a position;
* a **three-state HMM** (`H` Barcelona, `A` Atletic Club, `D` dead ball) over tracking geometry,
  team motion, boundary distance, last touch, and smart tags;
* a **whistle detector** on the tactical MP4's audio track, used to confirm referee stoppages;
* an **event layer** for goals, visible out-of-play balls, and confirmed stoppages, applied as
  high-precision `D` overrides after the Viterbi decode.

The model reads the tactical ATD, smart tags, homography diagnostics, and the tactical audio. The
public 5-minute ground truth is used only as a correctness check.

## Quick start

Use Python 3.12.3, the version used for the validated run. The pinned environment includes the
runtime, plotting dependencies, and `imageio-ffmpeg` for audio decoding. No GPU, notebook, or
frontend toolchain is required.

```bash
# From the repository root
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-possession.txt

# Unpack the gated bundle under data/ (see ../data/README.md), then regenerate everything.
# The first run builds outputs/whistle.npz from the MP4 audio; later runs reuse the cache.
.venv/bin/python -m possession_lab all --bootstrap-n 2000

# Verify the implementation and the submission format
.venv/bin/python -m pytest -q possession_lab/tests
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
process described in [DATA.md](../DATA.md); keep the supplied names and paths. Alternatively, set
`MTAH_DATA_DIR` to the folder containing `tactical/`.

The tactical MP4 is optional for predictions but required for whistle confirmation. If the video
or ffmpeg is unavailable, whistle support is zero-filled and the event layer still uses ball
geometry and smart tags. `outputs/whistle.npz` is a generated cache and is git-ignored.

Outputs are written to `possession_lab/outputs/` (git-ignored):

```text
predictions.csv          199,319 rows: frame,label
explanations.json        per-switch "why" annotations (clickable in the viewer)
metrics_public.json      official public metrics + bootstrap
stress.csv               clean / random / long missing-ball stress rows
full_stats.json          full-match spell structure
timings.json             measured CPU timings and GPU note
whistle.npz              cached per-frame whistle score/envelope/events
report.html              self-contained report with eight figures
01_data_overview.png ... 08_tag_context.png
user_truth.json          optional viewer-only UT (written by the local reviewer)
```

## Local synchronized reviewer

The viewer synchronizes the tactical MP4 with the generated `frame,label` rows, the public ground
truth, a **user truth (UT)** that you can write by hand, and the tracking/event evidence behind
the prediction. It never rewrites `predictions.csv` or the public GT. The UT exists only to score
the fixed prediction inside the viewer; the model, features, and training code never read it.

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
| `--explanations PATH` | switch annotations JSON (default `outputs/explanations.json`) |
| `--no-explanations` | start without the annotation layer |
| `--start-frame N` | initial frame; URL `?frame=N` overrides it |
| `--no-open` | do not launch a browser |
| `--port N` | local port (default `8765`) |

What is shown:

* Video, current frame, timecode, predicted label, public GT (only inside frames
  `[11,185, 18,685)`; outside it the GT is shown as `X` / “sin evaluar”), and the UT.
* **Tracking overlay.** Player slots and the ball are projected onto the video with the frame's
  homography (1920×1080). The three ball layers can be toggled independently under **Capas de
  balón**: **ATD bruto** draws only ATD observations (white solid ring); **Kalman** is a causal
  constant-velocity filter that rejects isolated jumps and predicts through holes up to 5 s (cyan
  ring, dashed while predicting); **Combinado** is the production tracker, ATD observations plus
  damped extrapolation for holes of at most 1.2 s (white ring when observed, dashed amber when
  estimating). A red ring means the point is beyond the pitch. The last ~1 s of each enabled
  layer is drawn as a fading trail. The `Seguimiento` checkbox toggles the whole overlay; the
  layer checkboxes select which estimates are drawn.
* **Mapa cenital.** A separate pitch canvas under the video draws the same frame directly from
  ATD metres, with **no homography and no video dependency**: pitch lines, blue/red player slots,
  the enabled ball layers, and their ~1 s trails. It is the honest view of what the model sees
  when the broadcast projection does not match. The 46 slots are anonymous; colour is the team in
  the XML, not a player identity. The layer checkboxes affect both canvases; the map status line
  reports each enabled layer and its gap. A ball outside the touchline is drawn in the map
  padding in red instead of disappearing at the clip boundary.
* **Dataset explorer.** A frame card shows the ball source/coordinates, ball and team speed,
  nearest player and distance, distance to the line, ball gap, last touch and age, and the active
  goal/stoppage event. The header also reports the state of each enabled ball layer. This is
  computed from the same full-match feature arrays the model used, sliced per frame by the local
  server; the browser never receives the 199k-frame tracking table.
  The Kalman layer is **diagnostic only**: it does not feed `features.py`, the HMM, the event
  detector, or the submission. A video-domain MOSSE/GrayDiff/NanoTrack fusion was evaluated on
  paper but not implemented: this workspace has no tracker code or weights and no OpenCV/SciPy/
  filterpy runtime, and the tactical video still needs frame-accurate alignment to make a pixel
  tracker comparable with the ATD. The dataset-native Kalman layer is the implemented alternative.
* **Event markers.** The full-match and detail timelines mark goals (green), confirmed stoppages
  (amber dashes), and ball-tracking gaps (grey). Hovering a timeline shows the frame, time, pred,
  GT, and UT.
* Current H/A possession spell, smart-tag context, public metrics, and **UT metrics**: spell F1,
  macro F1, D-F1, accuracy, and spell counts scored only over the frames you tagged.
* Full-match overview with PRED / GT / UT channels, a ±20 s detail strip, and navigation by label
  change, possession, or GT disagreement. Click or drag either timeline to seek; the browser's
  native video controls still work.
* **Decision annotations.** Every predicted state switch is a clickable tick on both timelines and
  a timestamped row in the "Decisiones · por qué" card: the `from → to` transition, the emission
  confidence of the entered state, and the ordered evidence reasons from `explain.py` (ball out of
  bounds, parked at the line, restart ramp, group walking, ball lost, `SET PIECES` tag, nearest
  player, or the transition prior) — each reason's evidence frame is itself a seek button. Filters
  all / → D / D → / H ↔ A; `j`/`k` jump between annotated switches; the selection auto-follows
  playback within ±1.5 s.
* UT editing: press **Ancla aquí** at the start of a passage, seek to its end, then press `H`, `A`,
  or `D` to write that span; without an anchor a label applies to the current frame. `X` erases
  the span and **Borrar UT** erases the UT run nearest to the current frame. The same `H/A/D/X`
  keys work anywhere on the page
  (except while typing in a text field) and every label press leaves an anchor at the end of the
  written span. Anchors are drawn on both timelines; drag one to move the label boundary. The UT is
  written to `outputs/user_truth.json` after every edit and reloaded on the next session.
* Keyboard: `Espacio` play/pause, `←/→` one frame, `Shift` + arrows one second, `[`/`]` previous/next
  label change, `,`/`.` previous/next possession, `n`/`N` next/previous GT disagreement, `j`/`k`
  previous/next annotated switch, `H/A/D/X` user-truth labels, `Home`/`End`.

The tactical timeline is exact: `frame i` is row `i`, `currentTime = (i + 0.5) / 25`. The viewer
uses the synchronized tactical video only; the unsynchronized broadcast is excluded. The 5 GB MP4
is streamed with HTTP byte ranges, so only the requested header/tail and playback ranges are read.
The server (`/api/frame?frame=N`) returns one compact JSON frame with projected player/ball
positions (`px`/`py` are nullable), pitch size, the projected trail, the pitch-space trail
(`trail_pitch`), the feature slice, and active events. The map uses `pitch` and `trail_pitch`
only, so it remains usable if a homography is missing. The frontend is plain HTML/CSS/JS; the
server is Python's standard library HTTP server. `imageio-ffmpeg` is only needed by the offline
audio extractor, not by the viewer page.

## Tracking representation

The viewer and the model consume Metrica's ATD export, not an in-browser detector. `data.py`
reads **46 fixed player slots** per frame, scales the ATD's normalised coordinates to a
**105.3 × 68 m** pitch, and maps `frame_id - 1` to the video frame. Team labels come from the XML
metadata; the slots are anonymous and no jersey/identity is inferred. The file has 24 Barcelona
slots and 22 Athletic slots, and the active count varies (median 21); some frames contain more
than 11 tracks per team, so the viewer shows every visible slot rather than pretending to know
the active XI. The model only uses aggregate nearest-player/team geometry from those slots.

The ball is observed directly from the ATD; `balltrack.py` adds a causal tracker. Source `0` is
an observation, source `1` a short-gap extrapolation capped at 1.2 s, and source `2` a long gap
with no estimate. Over the full file: **113,488 observed (56.94%)**, **4,441 estimated (2.23%)**,
and **81,390 without an estimate (40.83%)**. That missing/estimated fraction is why the ball can
look “wrong” against the video even when the code is working: an amber ring is a guess and no ring
is an explicit refusal to guess.

## What the model does

Three states: **H** (Barcelona), **A** (Atletic Club), **D** (dead ball / out of play).

* **Visible ball.** A Gaussian in the distance from the ball to each team's nearest tracked player
  (`σ=4.0 m`). The dead-ball emission is a sigmoid on the nearest-player distance
  (`d_dead=8.0 m`) plus independent dead-ball evidence.
* **Dead-ball evidence.** Team motion energy (mean tracked-player speed smoothed over one second),
  ball close to a touchline, a visible out-of-play ball, and a parked-ball-at-the-line signal.
  Players walk or stand still during a stoppage, so this separates a stoppage where the ball is
  visible and a player is standing next to it from live play.
* **Last touch.** A visible ball within `lt_R=0.8 m` of the player who last touched it gets a small
  bonus for that team (`w_lt=0.1`). The weight is deliberately low: a stronger term improved
  public spell F1 slightly but invented a long H spell after a single defensive touch at frame
  15,656.
* **Speed gate.** The emission is blended with a uniform distribution using
  `1 - sigmoid((ball_speed - gate) / 2)`. The validated model uses `gate=0`, so a stationary or
  unknown-speed ball gets a 50/50 blend; this is intentional and required to reproduce the
  reference scores.
* **Loose ball (>7 m from every player).** A ball that far from everyone is loose or in flight,
  not a stoppage, so the emission is uniform and the prior carries the possession. If it is also
  slower than `2 m/s`, the team that touched it last gets 70% of the emission mass when that touch
  is under 5 s old. This removed false `D` stretches on rolling balls and recoveries.
* **Ball parked at a boundary.** A ball seen within `2.5 m` of a line at under `1.0 m/s`, while
  the team is also barely moving (`<1.35 m/s`), for at least `0.8 s` — tolerating tracking
  flicker inside the window — gets a strong dead-ball bonus (`w_parked_edge=1.5`): the throw-in /
  corner / goal-kick signature.
* **Short gaps.** A causal tracker keeps the last observed ball position and a robust velocity,
  extrapolates through holes up to **1.2 s** with a damped constant-velocity model, and exposes a
  confidence/gap. Long dead-ball gaps are left empty: inventing a ball position for a ten-second
  stoppage is worse than admitting the gap. In the validated parameter set the HMM's short-gap
  emission stays uniform (`est_blend=0`); the tracker estimate feeds goal geometry and the viewer
  rather than the emission.
* **Missing-ball emissions.** Uniform for gaps ≤120 frames (4.8 s). For longer holes, the team
  shape around the last known ball, the last known edge distance, and the smart-tag vote provide a
  weak geometric prior. Sparse tracking (<6 visible players) is treated as missing evidence rather
  than reading team shape from one or two lonely slots.
* **Whistle evidence.** The tactical MP4 audio is decoded through ffmpeg and scored on relative
  spectral shape in the 2.4–3.8 kHz whistle band (the feed is quiet, so level is not used). A
  whistle is confirmed only when followed by sustained low team speed (`<1.4 m/s` for ≥2 s) and
  the ball is not visibly controlled. Visible out-of-play and missing-after-out intervals are
  direct dead-ball evidence rather than whistle candidates. A live smart tag (`PROGRESSION`,
  `ATTACKING TRANSITION`) can delay the cut past short stoppage candidates so live play is not
  truncated.
* **Goals and confirmed stoppages.** Goal detection combines a ball trajectory that reaches the
  goal mouth with tagged long disappearances backed by a celebration and `Kick Off (After Goal)`;
  balls observed behind the line but outside the posts are rejected. Confirmed goals and stoppages
  become a hard `D` mask after Viterbi and run merging. A `Kick Off` tag trims an overlapping
  stoppage so the public window does not inherit a false `D`.
* **Transitions.** `p_stay=0.995`, `p_ha=0.0005`, `p_hd=0.001`, `p_dh=0.007`.
* **Decode.** Viterbi over each graded half with a leading 500-frame context pad. Frames outside
  the graded halves are `X`. Forward–backward posteriors are computed only for diagnostics; they
  never change the submission labels. A run shorter than 1 s that returns to the same label is
  absorbed afterwards: it is a flicker, not a possession change.

The exact parameter dictionary is `possession_lab/config.py:MODEL_PARAMS`. The implementation is in
`model.py` (emissions, dead-ball evidence, Viterbi, forward–backward), `features.py` (geometry,
motion energy, gaps, tags, tracker/whistle/event features), `balltrack.py` (causal tracker),
`audio.py` (ffmpeg decode, whistle score/cache), `events.py` (goals, stoppages, overrides),
`explain.py` (per-switch reason rules and the explanation payload), `data.py` (ATD/tag/homography
loaders), and `pipeline.py` (inference and CSV writing).

The current defaults were retuned on the public window against the possession folds and the
missing-ball stress suite. `σ` widened from 3.0 to **4.0**, leaving dead ball from `D` became about
seven times more likely (`p_dh` 0.001 → **0.007**), the team Gaussian dead zone widened to
`d_dead=8.0 m`, and the last-touch term was set to **0.1**. A stronger last-touch weight (0.2)
raised spell F1 by ~0.005 but lowered macro F1 by ~0.019 and created a false H spell after the
15,656 defensive touch, so 0.1 was kept.

Measured alternatives that were rejected: a fully split two-state alive/dead + H/A decode reached
only 0.547 spell F1 and 0.686 macro F1; decaying the last-ball memory with gap length lowered macro
F1 to 0.846; a walk/run fraction, a smart-tag team vote, and ball-speed gating of the dead-ball
evidence each lowered public spell F1 by 0.03–0.07. An offline Kalman/RTS smoother is unnecessary
for the model: the causal tracker already fills the 0.4–1.2 s holes for goal geometry and the
production overlay, while the HMM keeps short gaps uniform and long holes are not bridged. The
causal Kalman filter in `kalman.py` is a **viewer diagnostic** only; it is not blended into the
HMM or the event detector.

## Results

### Public 5-minute ground truth

Public window: frames **11,185–18,684** (7,500 frames, 17 possessions).

| Metric | Value |
|---|---:|
| Spell F1 (headline) | **0.6519** |
| Spell F1 at IoU 0.5 | 0.6667 |
| Macro F1 (H/A/D) | **0.9466** |
| D-F1 | **0.9727** |
| Accuracy | 0.9417 |
| Cohen's kappa | 0.9083 |
| Alive/dead F1 | 0.9933 |
| Team accuracy (both alive) | 0.9404 |
| Predicted spells | 10 (GT 17) |
| Spell F1 95% bootstrap | [0.430, 0.817] |

The bootstrap interval is wide because there are only 17 GT possessions; do not over-read small
differences. The spell count is still below 17: a few sub-second GT possessions are single
deflections that the model does not split out, and the 1 s flicker cleanup removes any that it
produces. The event layer is deliberately high precision rather than high recall: it fixes the
referee stop, goal, and out-of-play cases without turning every rolling ball into `D`.

### Missing-ball stress test

`stress.csv` removes about 40.7% of the public window. The clean row is the untouched public window.
The baseline is a centred H/A vote with no gap model. The stress suite isolates the **HMM core**:
it deliberately does **not** apply the goal/stoppage hard overrides, so its clean row is lower than
the final public score above. This keeps the test focused on gap handling rather than on event
detection, which has its own tests.

| Mask | Method | Spell F1 | Macro F1 | D-F1 |
|---|---|---:|---:|---:|
| clean | baseline w50 | 0.3256 | 0.6207 | 0.3225 |
| clean | baseline w100 | 0.4485 | 0.5976 | 0.1992 |
| clean | HMM persistence | 0.4308 | 0.6045 | 0.2341 |
| clean | HMM gap-aware | **0.6143** | **0.7299** | **0.5144** |
| random | baseline w50 | 0.2414 | 0.5730 | 0.2974 |
| random | baseline w100 | 0.2900 | 0.5542 | 0.1945 |
| random | HMM persistence | 0.2957 | 0.7045 | 0.5494 |
| random | HMM gap-aware | **0.3500** | **0.7049** | **0.5209** |
| long | baseline w50 | 0.1789 | 0.4790 | 0.3022 |
| long | baseline w100 | 0.2452 | 0.4897 | 0.2630 |
| long | HMM persistence | 0.1818 | 0.6171 | 0.2341 |
| long | HMM gap-aware | **0.3840** | **0.6387** | **0.5602** |

The HMM core beats both baselines on every mask. The long-gap row is the hardest because two long
holes remove most of the public window's possession evidence.

### Full submission

* **199,319 rows**, frames 0–199,318, labels in `{H,A,D,X}` (H 65,338 · A 42,387 · D 51,132 · X 40,462).
* 198 H/A spells (100 first half, 98 second) over 158,857 graded frames; mean 21.8 s; median 15.6 s.
* Shares: H 41.1%, A 26.7%, D 32.2%.
* Official validator: `OK predictions.csv` (format only).
* Public metrics with the pinned environment and `--bootstrap-n 2000`: spell F1 **0.6519**,
  macro F1 **0.9466**, and D-F1 **0.9727**.

Predictions, metric values, and seeded bootstrap intervals are reproducible from the same data
and pinned dependencies. Wall-clock timings in `timings.json` and the HTML report naturally vary
by machine. The detected goals are at frames **46,691** (`trajectory`) and **157,825**
(`celebration`). The tagged candidate at **57,283** is a false positive: the ball is observed
behind the line but wide before its gap. The **73,935** penalty (tag opens at 73,710) is a real
goal that the conservative detector misses: the ATD loses the ball for **55 s** across the kick,
so there is no tracked trajectory; its first re-observation at 73,951 is already behind the line
and wide, and the last pre-gap point is 13.5 m from the line, outside the 12 m fallback. The
fallback therefore rejects it. Loosening that rule would also accept the 57,283 false tag, so the
miss is documented rather than hidden. This is a known recall limitation of the
high-precision event layer.

## Data and privacy

* Uses the gated tactical **ATD**, **smart-tag XML**, and **homography CSV**.
* Uses the tactical **MP4 audio** only for whistle detection. Match video frames are not copied
  into any output.
* Broadcast video is **not** used: it is unsynchronised and adds no label evidence.
* The public GT is read from `evaluation/gt/BAR-ATH_possession_gt_public.csv` only for evaluation.
* The viewer-only UT is stored in `possession_lab/outputs/user_truth.json`. It scores the fixed
  prediction for the user and is never read by the model, features, or training code.
* No private full-match ground truth is read, written, or inferred.
* `data/` and `possession_lab/outputs/` are git-ignored. Do not commit the gated bundle, the
  whistle cache, or the generated full-match labels.

## CPU vs GPU

An NVIDIA RTX 5070 Ti Laptop GPU (12 GB) is available, but the model is a 3-state dynamic program
over 199,319 frames plus a handful of 46-player distance arrays. Measured on CPU:

* load: ~2.8 s
* full decode including tracker/event preparation: ~4.6 s
* 12-row stress suite: ~3.2 s

A GPU would spend more time transferring arrays than computing. No GPU path is included because it
would be slower and harder to reproduce. Revisit GPU only for a pixel-space tracker or a learned
sequence model.

## Layout

```
possession_lab/
  config.py       match constants, labels, paths, exact parameters
  data.py         ATD / smart-tag / homography / GT loaders
  balltrack.py    causal short-gap ball tracker
  kalman.py       causal constant-velocity Kalman viewer layer (diagnostic)
  audio.py        ffmpeg audio decode, whistle score, cache
  events.py       goal/stoppage detection and hard D overrides
  features.py     distances, gap, edge, last touch, tags, tracker/whistle/event features
  model.py        emissions, Viterbi, forward-backward
  explain.py      per-switch reason rules and the viewer explanation payload
  evaluate.py     official metrics, stress masks, full-match stats
  pipeline.py     full-match inference, CSV writing, preflight
  visualize.py    eight figures + HTML report
  viewer.py       local reviewer, frame explorer, byte-range streaming
  viewer_assets/  index.html, viewer.css, viewer.js (no build step)
  cli.py          all / predict / explain / evaluate / stress / visualize / viewer
  tests/          stdlib unittest suite
  outputs/        generated, git-ignored
```

## Skills used

* `writing-plans` — implementation plan and acceptance criteria
* `test-driven-development` — tests for loaders, features, tracker, audio, events, HMM, metrics
* `verification-before-completion` — official validator, browser smoke test, and test evidence
* `systematic-debugging` — isolated the overlay stretch to `canvas { height: 100% }` filling the
  whole video card (including the caption), instead of guessing at the tracker
* `frontend-ui-engineering` — the native top-view canvas card, responsive layout, legend, and
  keyboard/touch-safe viewer wiring
* `browser-testing-with-devtools` — measured video/overlay rectangles and checked the map at
  frames 12,000 and 12,851 with Chrome DevTools

## Limitations

* The public GT has 17 possessions; the confidence interval is wide.
* The ATD has 46 anonymous player slots (24 Barcelona / 22 Athletic). It is not a guaranteed
  11v11 active lineup: some frames contain more than 11 tracks per team (extra/substitute/staff
  tracks or provider duplicates; up to 35 visible slots). The viewer shows all visible slots, and
  the model's nearest-team distances use all present slots. The model never infers player identity.
* The ball is not observed for 43.06% of the full file (2.23% estimated, 40.83% no estimate), so
  an overlay comparison on those frames is testing the tracker's gap policy, not its projection.
* Smart tags cover 70.1% of the full recording (75.5% of the public window), and their team vote is
  weak; every throw-in is `N/A`.
* The model is tuned on the public window. The full match is a generalization check, not a
  validation set.
* Whistle detection depends on the tactical audio; without the MP4/cache, the event layer falls
  back to geometry and tags but cannot confirm a referee-only stoppage.
* Homography is used for diagnostics, pitch checks, and the viewer overlay, not as a label source.
  The tactical homography is valid on 100% of frames with median reprojection error 0.24 px.
