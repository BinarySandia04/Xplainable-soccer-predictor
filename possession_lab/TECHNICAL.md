# Possession Lab — technical note

## Objective

Assign one label per tactical video frame: `H` (Barcelona possession), `A` (Atletic Club
possession), `D` (dead ball / out of play), or `X` (ungraded). The submission is 199,319 rows.

The model is a **data-only, gap-aware 3-state HMM**. It deliberately excludes the earlier
MOSSE + GrayDiff + NanoTrack video ensemble: the tactical tracking already provides a cleaner,
synchronised observations stream, and the public window is only a correctness check. Around the
HMM, a causal ball tracker supplies short-hole geometry, a whistle detector reads the tactical
audio, and a high-precision event layer pins confirmed goals and referee stoppages to `D`.

## Inputs

| Input | Use |
|---|---|
| `BAR-ATH_tactical_atd.txt/.xml` | Ball and 46 anonymous player slots, metres, 25 fps |
| `BAR-ATH_tactical_smart_tagging.xml` | 638 intervals, 14 codes, 6 types, team/type labels |
| `BAR-ATH_tactical_homography.csv` | Per-frame camera geometry; diagnostics, pitch validity, viewer projection |
| `BAR-ATH_tactical.mp4` audio | Whistle candidates; 7,972.69 s at 16 kHz mono |
| `evaluation/gt/...public.csv` | 7,500-frame public GT; evaluation only |

`video_frame = frame_id - 1`. Smart-tag intervals use
`[round(start*25), round(end*25)+1)`. The tactical homography is valid on 100% of frames (median
reprojection error 0.24 px, p99 4.81 px).

The validated configuration uses the tactical audio through the event layer, not through a
whistle emission weight. If the MP4 or ffmpeg is unavailable, whistle support is empty and the
event layer still uses visible out-of-play balls, missing-ball last edges, and tags.

## Pipeline

```text
ATD + tags + homography
        │
        ├── causal ball tracker ── short-hole estimate, source, confidence, out-of-play
        │
        ├── whistle cache ──────── 2.4–3.8 kHz score, envelope, 274 candidate blasts
        │
        └── features ───────────── distances, motion, edge, gap, last touch, tags
                                      │
                              goal/stoppage detector
                                      │
                              HMM Viterbi decode
                                      │
                         merge 1 s flickers + hard D overrides
                                      │
                              predictions.csv
```

The full-match diagnostics are built once, then the two graded halves are decoded independently
with a leading 500-frame context pad. Stoppages and goals are detected on the full match so an
event just outside a half is not hidden by the split.

## Causal ball tracker (`balltrack.py`)

The ATD drops the ball for a few frames during passes, deflections, and occlusions, and for many
seconds during stoppages. The tracker is causal: frame `k` uses observations at frames `<= k`
only.

* Keeps the last observed position and a least-squares velocity over the last 0.6 s of
  observations.
* Extrapolates a short hole with a damped constant-velocity model
  (`v(t) = v0 * exp(-0.35 t)`), capped at 38 m/s for outliers.
* Never fills more than **1.2 s**. Longer holes have `source=2` (no estimate), because inventing a
  position across a dead-ball gap is worse than admitting the gap.
* Confidence is 1 on observations and decays as `exp(-(gap/0.9)²)` inside a filled hole; an
  unreliable velocity estimate halves it.
* Flags an estimate beyond the pitch by more than 1 m as `out_of_play`.

The tracker is used by goal geometry, the viewer overlay, and the gap diagnostics. In the final
parameter set the HMM's short-gap emission is **uniform** (`est_blend=0.0`); the tracker estimate
is not blended into the HMM emission. This is deliberate: blending a short extrapolation into the
emission improved some public frames but created false possession changes in longer gaps, so the
short-gap prior remains the validated behaviour.

## Features (`features.py`)

For each frame:

* `d_H`, `d_A`: distance from the ball to the nearest home/away tracked player.
* `near_team`, `near_distance`, `second_distance`, `margin`.
* `near_speed`; `ball_speed`, `ball_acceleration` (only across consecutive observations).
* `team_speed`: mean speed of the tracked players, smoothed over one second.
* `gap`: frames since the last observed ball; `edge`, `last_edge`: distance to the nearest pitch
  boundary.
* `out_of_play`: observed ball outside `[0,L] × [0,W]`.
* `last_touch`, `last_touch_age`: nearest player within 0.8 m; a team hint, not an identity.
* `d_H_last`, `d_A_last`: current team shape around the last observed ball.
* `tag_vote`, `tag_setpiece`, `tag_coverage`, `tag_live`.
* `player_count`: visible slots for the sparse-tracking guard.
* `parked_edge`: visible ball within `2.5 m` of a line at `<= 1.0 m/s`, with the team also below
  `1.35 m/s`, meeting the condition in at least 70% of a trailing `0.8 s` window (30% tracking
  flicker allowed). Observed-out frames count.
* Tracker diagnostics: `ball_est`, `est_confidence`, `est_source`, `est_out`, `est_d_home`,
  `est_d_away`.
* Event/audio support: `whistle_score`, `whistle_env`, `goal_dead`, `stop_dead`.

## Emissions

At visible frames, with `σ = 4.0 m`, `d_dead = 8.0 m`:

```text
q_H = exp(-d_H² / (2σ²)) + 1e-12
q_A = exp(-d_A² / (2σ²)) + 1e-12
q_D = sigmoid((min(d_H,d_A) - d_dead) / 1.5) + 1e-12
q_D += w_dead + w_motion * sigmoid((motion0 - team_speed) / motion_scale)
     + w_edge * sigmoid((edge0 - edge) / edge_scale)
q_D += w_out * out_of_play
q_D += w_parked_edge * parked_edge
q   = [q_H, q_A, q_D] / sum(q)
logB = log(q)
```

The validated values are `w_out=1`, `w_dead=0.02`, `w_motion=1`, `motion0=1.6 m/s`,
`motion_scale=0.35`, `w_edge=0.5`, `edge0=5.0 m`, `edge_scale=1.5`, and
`w_parked_edge=1.5`.

Team motion is only evidence when no player is controlling the ball: if the ball is visible,
within `motion_control_R=4.0 m` of a player, more than `2.5 m` from a line, and not inside a
confirmed stop, the motion term is zero. This is what stops a player standing with the ball at
the touchline from being labelled “Balón muerto”.

The speed gate is applied even when `gate=0`:

```text
α = 1 - sigmoid((ball_speed - gate) / 2)
logB = α * logB + (1 - α) * log(1/3)
```

A stationary or unknown-speed ball is therefore blended 50/50 with uniform. This is a deliberate
regulariser and is required to reproduce the validated scores.

### Loose balls and last touch

A ball farther than `loose_R=7 m` from every player is loose or in flight, not a dead ball. Its
emission is uniform, so distance alone cannot vote `D`. If it is also slower than `loose_speed=2
m/s`, the team that touched it last receives `touch_p=0.7` of the mass when the touch is younger
than `touch_age=125` frames (5 s); the remainder is split over the other two states. This removed
false `D` stretches where a rolling or in-flight ball sits 7–10 m from both teams.

The validated `loose_stop_gate` is off. The optional gate would exempt a parked ball from the
loose-ball flattening, which is useful for some ablations but was not selected.

A small last-touch bonus (`w_lt=0.1`) applies when a visible ball is within `lt_R=0.8 m` of the
player who last touched it. It is intentionally weak: a stronger term improved public spell F1 by
about 0.005 but invented a long H spell after a single defensive touch at frame 15,656 and lowered
macro F1 by about 0.019.

The flight rule (`flight_R`, `flight_p`) exists in `model.py` but is disabled in the validated
configuration (`flight_R=0`). The loose-ball + last-touch path was more stable.

### Missing-ball emissions

* `persist`: uniform for every missing frame (ablation baseline).
* `hybrid` (validated): uniform for `gap <= 120` frames (4.8 s). For longer holes:

```text
m_H = exp(-d_H_last² / (2σ_miss²)) + 1e-6
m_A = exp(-d_A_last² / (2σ_miss²)) + 1e-6
m_D = 0.15 + sigmoid((5 - last_edge) / 1.5)
m_H *= 1 + max(tag_vote, 0)
m_A *= 1 + max(-tag_vote, 0)
m_D *= 1 + 0.2 * tag_setpiece
```

with `σ_miss=6 m`. If fewer than `min_players=6` players are visible, the frame is treated as
missing evidence and the emission is uniform; this prevents a near-empty tracking feed from
inventing a team. A missing frame covered by a live smart tag (`PROGRESSION`, `ATTACKING
TRANSITION`) is also uniform, so a live-play gap does not flip possession from the last known
team shape. Confirmed stoppages are hard overrides after decoding, so this does not hide a real
dead ball.

## Whistle detection (`audio.py`)

The tactical feed has a quiet, tonal referee whistle; absolute level is not useful. The detector
uses relative spectral shape:

* Decode the MP4 audio to mono float32 at 16 kHz with ffmpeg (`imageio-ffmpeg`'s static binary or
  a system ffmpeg).
* Compute an STFT with a 2,048-sample Hann window and 512-sample hop.
* For each STFT frame, score the 2.4–3.8 kHz band as
  `peakiness * band_energy / low_energy`, where low is 0.4–1.8 kHz.
* Collapse STFT frames onto video frames by taking the maximum; frame 0 is the first video frame.
* Threshold at `max(4.0, 2.0 * local_median)`, where the local median spans ±120 frames.
* Join dropouts of at most 1 s inside one blast, reject runs shorter than 0.12 s, and record the
  peak frame, half-open start/stop, peak score, and width.

The result is cached in `outputs/whistle.npz` (score, envelope, events, metadata). On this match
it contains **274 candidate blasts**. A candidate is not a stoppage by itself: crowd noise and
players' shouts can produce similar peaks. It becomes evidence only when followed by sustained low
player motion and the ball is not visibly controlled.

## Goals and confirmed stoppages (`events.py`)

### Goals

A `GOALS` smart tag is not trusted on its own. For every tag interval, the detector looks at the
ball-loss run before the restart:

* **Trajectory path.** Extrapolate the last observed ball position with a short-window velocity
  and require it to reach a goal line within 3 s, inside the posts (`GOAL_HALF_WIDTH=3.66 m` plus
  a 1.6 m tolerance), at no less than 3 m/s along the line. Confidence combines mouth offset,
  speed, and travel time.
* **Tagged-disappearance path.** If no trajectory crosses, accept a long tagged disappearance
  only when the last position is within 12 m of the goal line and inside the mouth, the ball is
  missing for at least 8 s, and a celebration/kick-off tag backs it. A ball observed behind the
  line but outside the posts (`_behind_line_wide`) rejects the goal.

Accepted goals: frame **46,691** (`trajectory`) and frame **157,825** (`celebration`). The tagged
candidates at 57,283 and 73,710 are rejected because the ball was observed behind the line but
wide. Goal intervals extend to the restart: 46,691–49,631 and 157,825–163,000.

### Stoppages

Detected intervals, in order of evidence strength:

1. **Whistle.** A low-motion run (`team_speed < 1.4 m/s` for at least 2 s) with a whistle
   candidate no more than 2.5 s before the low run starts. A visible ball within 3 m of a player
   and not out of play suppresses the whistle. If a live smart tag covers the whistle, the start
   is delayed to the end of the tag (plus one second when the ball is visible) so live play is not
   cut short.
2. **Visible out.** A ball observed beyond the touch/goal line for at least 0.5 s; end at the
   first sustained kicked restart.
3. **Missing after out.** A missing ball whose last observed edge was beyond the line, for at
   least 1 s; this covers goalkeeper/throw-in retrieval gaps, not a ball lost near the line during
   live play.

Overlapping or adjacent candidates merge, preferring the stronger trigger. A `Kick Off` tag trims
an overlapping interval at the kick-off, so a retrieval gap that began earlier cannot leak a hard
`D` into the first live frames. The public window contains these confirmed stoppages after the
trim:

```text
11,035–11,097   (trimmed at the kick-off; originally longer)
12,852–12,991   whistle (injury/referee stoppage)
15,888–16,405   whistle
17,377–17,828   whistle
18,025–18,373   out (visible ball beyond the line)
```

Full match: 68 stoppages and 2 goals. The goal/stoppage masks are unioned into `dead_mask`.

### Overrides

Confirmed events are applied **after** Viterbi and the 1 s flicker merge. `apply_event_overrides`
sets every graded frame inside the mask to `D`; outside the graded halves the label remains `X`.
This is the final authority for referee stops and goals. The HMM is still responsible for ordinary
possession and ordinary out-of-play, which avoids turning every rolling ball into `D`.

## Transition, decode, and smoothing

```text
T = [[.995,   .0005, .001],
     [.0005,  .995,  .001],
     [.007,   .007,  .986]]
```

`p_dh=0.007` is the retuned exit rate from `D`; it is deliberately less sticky than the earlier
`0.01` because the hard event mask now guarantees confirmed stoppages, while the HMM should enter
`D` promptly around them. `p_stay=0.995`, `p_ha=0.0005`, `p_hd=0.001`.

Viterbi runs over each graded half with a leading 500-frame context pad. Frames outside
`[11,185, 90,652)` and `[110,808, 190,198)` are `X`. Forward–backward posteriors are computed only
for diagnostics; they never change the submission. After decoding, a run shorter than one second
that is flanked by the same label is absorbed by it: a few frames of `D` (or of the other team)
between two equal labels are a tracking or ball-in-flight flicker, not a possession change. `X`
is never changed.

The final validated parameters are in `config.py:MODEL_PARAMS`:

```text
sigma 4.0      d_dead 8.0      dead_scale 1.5
edge0 5.0      edge_scale 1.5  w_edge 0.5
w_out 1.0      w_dead 0.02     w_motion 1.0
motion0 1.6    motion_scale 0.35  motion_control_R 4.0
w_lt 0.1       lt_R 0.8        p_stay 0.995
p_ha 0.0005    p_hd 0.001      p_dh 0.007
gate 0.0       loose_R 7.0     loose_speed 2.0
touch_age 125  touch_p 0.7     miss_mode hybrid
sigma_miss 6.0 gap_thresh 120  min_players 6
w_parked_edge 1.5
disabled in this run: w_gap, w_tag, w_sp, w_whistle, est_blend, w_est_out, flight_R
```

## Explanation layer (`explain.py`)

The Viterbi answer to “why did the model switch here?” is “the balance of evidence moved”.
`explain.py` rebuilds that balance for every predicted switch from the same feature windows the
model saw and the smart tags, and renders ordered, checkable reasons:

| rule | evidence shown |
|---|---|
| `out_of_bounds` | ball coordinates crossed the line; depth in metres, frame |
| `parked` | longest parked-at-line run before the switch; seconds, edge distance, speed |
| `restart` | sustained ball ramp < 1.5 m/s → ≥ 5 m/s within 2 s after the switch (measured ramps take 0.3–1.5 s, so this is a ramp rule, not a one-frame jump rule) |
| `teams_slow` | window median team speed below 0.62 × the window median of the half |
| `ball_missing` | tracker lost the ball ≥ 1 s around the switch |
| `tag` | covering or nearest `SET PIECES` smart tag with type and team (other tag codes run for minutes and explain nothing) |
| `geometry` | nearest player to the ball just after the switch, team and distance |
| `prior` | fallback when no local signature fired: the transition prior carried the state |

Switches are deduplicated below 0.5 s (flickers collapse into the first), `X` transitions are
excluded, and each half is explained with the same 500-frame context-padded window the decode
used, so annotations describe evidence the model actually saw. `confidence` is the emission
probability of the entered state at the switch frame (max over ±0.5 s) **before** temporal
smoothing: how strongly that single frame favoured it — values near 1/3 honestly say “the frame
itself carried no evidence; the prior decided”.

`explain_prediction` walks the whole match; `python -m possession_lab explain` regenerates
`outputs/explanations.json` from any existing `predictions.csv`. `predict` and `all` build it
automatically (`--no-explain` skips; failures never break a run). The model never reads anything
`explain.py` produces — explanations are built downstream of the prediction.

## Local synchronized reviewer

`python -m possession_lab viewer` builds a compact JSON payload from `outputs/predictions.csv`, the
public GT, the smart-tag table, `metrics_public.json`, and the tracker/event products, then serves
it from a local stdlib HTTP server. The page is plain HTML/CSS/JS; no frontend build step is
required. `imageio-ffmpeg` is only used by the offline audio extractor, not by the viewer page.

* **Synchronization.** `frame i` is exactly row `i` of the submission. Browser seeking uses
  `currentTime = (i + 0.5) / 25`; the displayed frame is
  `clamp(floor(currentTime * 25 + 1e-6), 0, 199318)`. The video duration is checked against
  `199,319 / 25 = 7972.76 s` when metadata loads.
* **Payload.** Labels are compact `H/A/D/X` strings; prediction segments/spells cover the full
  graded halves, while GT segments/spells cover only the public window because the public CSV has
  `X` outside it. Changes, disagreement runs, 638 tag intervals, public metrics, goals,
  stoppages, and ball-gap runs are included. When `outputs/explanations.json` exists it is folded
  in as `switches` and `explanations_available`; the UI degrades quietly without it.
* **Per-frame explorer.** `GET /api/frame?frame=N` returns one compact payload built by
  `FrameStore`: projected 1920×1080 player/ball pixels, the ball's raw and tracked state, a
  ~1 s trail, the feature slice (team/ball speed, edge, gap, near player/team, last touch,
  whistle score/envelope, event masks), and active events. The browser never receives the
  199k-frame tracking arrays. If tracking is unavailable the endpoint returns a JSON 404 and the
  viewer still shows the labels and timelines.
* **Overlay.** Players are projected with the frame's homography; the ball is a white ring when
  observed, a dashed amber ring when the tracker is filling a short hole, and red when beyond the
  pitch. The `Seguimiento` checkbox toggles the canvas.
* **Decision annotations.** Every predicted switch is a thin tick on the PRED band of both
  timelines; ticks are hit-tested within 6 px, seeking and opening the decision card on click.
  The card lists every switch with a clickable timestamp, the `from → to` transition, the
  emission confidence bar, and the ordered reason chips from `explain.py` — each reason's
  evidence frame is itself a seek button. Filters: all / → D / D → / H ↔ A; `j`/`k` jump to the
  previous/next annotated switch; selection auto-follows playback within ±1.5 s.
* **User truth (UT).** The browser can paint `H/A/D/X` on the current frame or an anchored range,
  from the buttons or globally from the keyboard. Each label press leaves an anchor at the end of
  the written span; run boundaries are drawn on both timelines and can be dragged to move a cut.
  `POST /api/user-truth` updates the UT array (paint, boundary move, or clear), writes
  `outputs/user_truth.json` atomically, and returns recomputed UT segments and official-scorer
  metrics over exactly the tagged frames. The UT is viewer-only: `features.py`, `model.py`, and
  `pipeline.py` never import or read it, and the fixed `predictions.csv` is never modified.
* **Streaming.** `/video` implements a single-range `Range` parser and returns `206`, `200`, or
  `416` as appropriate. The 5.2 GB tactical MP4 has its `moov` atom at the end; the browser reads
  the tail range for metadata and then fetches playback ranges. Routes are whitelist-only:
  `/`, `/static/viewer.css`, `/static/viewer.js`, `/api/viewer-data`, `/api/frame`,
  `/api/user-truth`, `/video`, `/favicon.ico`.
* **Scope.** The UI edits only the UT, has no correction export, and has no broadcast-video path.
  GT is shown only on non-`X` public frames. The page works entirely from `127.0.0.1`; the POST
  endpoint requires `application/json`.

The reviewer is covered by `tests/test_viewer.py`: payload semantics, public-only GT behavior,
UT scoring/persistence/editing, range parsing (`normal`, open, suffix, invalid), HTTP
`206`/`416`/`HEAD`, API payload, per-frame payload/404, and unknown-route rejection.

## Validation

* Unit tests: **74 stdlib `unittest` tests (+4 subtests)**, all passing — including switch/reason
  rules (`test_explain.py`), stop-signal parameter neutrality and gating (`test_model.py`),
  tracker/audio/event units (`test_balltrack.py`, `test_audio.py`, `test_events.py`), and
  annotation/frame-explorer plumbing (`test_viewer.py`).
* Public metrics for the final parameters: spell F1 **0.6519**, macro F1 **0.9466**, D-F1
  **0.9727**, accuracy 0.9417, kappa 0.9083, alive/dead F1 0.9933, team accuracy (both alive)
  0.9404, 10 predicted spells (17 in GT). The 95% spell-F1 bootstrap is [0.430, 0.817].
* Missing-ball stress suite (HMM core only; **event overrides disabled**, so the clean row is not
  the final public score): clean 0.6143 / 0.7299 macro / 0.5144 D-F1; random
  0.3500 / 0.7049 / 0.5209; long 0.3840 / 0.6387 / 0.5602. The full 12-row table is
  `outputs/stress.csv`; every mask beats both centred baselines on macro F1.
* Browser smoke test (Chrome headless + CDP, tactical video streamed from the local server):
  metadata `readyState=4`, duration 7972.76 s; initial frame 12,851 at `currentTime=514.06` with
  `video sincronizado`; seeking to frame 46,691 reached `currentTime=1867.66` and displayed
  `GOL · trajectory`; play advanced the frame counter and `currentTime`; frame 157,825 showed
  `GOL · celebration`, frame 15,888 showed `PARADA · whistle`; the overlay canvas and frame
  explorer rendered. A 5 s virtual-time screenshot stalls media metadata in headless Chrome; the
  CDP run is the authoritative browser evidence.
* Submission: 199,319 rows, frames 0–199,318, official validator `OK`, `n_runs=305`.

## Full-match reconciliation

The final run reports **198 spells** (100 first half, 98 second) over 158,857 graded frames,
mean 21.8 s and median 15.6 s; shares H 41.1% / A 26.7% / D 32.2%. Frame counts are H 65,338 ·
A 42,387 · D 51,132 · X 40,462.

One structural difference from the earlier reference run is frames 89,905–90,651: the ATD has no
home players and only 1–3 away players, and the ball is missing. The sparse-tracking guard treats
the stretch as missing evidence and predicts `D`, which the public and stress metrics are
insensitive to. The rest is the retune: a wider team Gaussian (`σ=4.0`), a wider dead zone
(`d_dead=8.0`), easier exits from `D` (`p_dh=0.007`), the dead-ball evidence model, the loose-ball
rule that returns a slow ball far from every player to its last toucher, and the goal/stoppage
event layer.

## Reproducibility and privacy

* `python -m possession_lab all --bootstrap-n 2000` regenerates every output, including the
  whistle cache on first run.
* `python -m pytest -q possession_lab/tests` runs the suite.
* `python evaluation/validate_submission.py possession_lab/outputs/predictions.csv` runs the
  official format gate.
* `python -m possession_lab explain` rebuilds only `outputs/explanations.json` from an existing
  `predictions.csv`.
* The gated `data/` bundle, `outputs/`, and the whistle cache are not committed.
* Uses the tactical MP4 audio only for whistle detection. Match video frames are not copied into
  any output. Broadcast video is not used.
* No private full-match GT is read, written, or inferred.
* CPU is intentional: the 3-state Viterbi pass plus tracker/event preparation took 4.56 s
  (load 2.77 s; stress suite 3.24 s) on the validated machine. A GPU transfer would dominate.
