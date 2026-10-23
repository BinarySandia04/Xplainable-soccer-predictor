# Possession Lab — technical note

## Objective

Assign one label per tactical video frame: `H` (Barcelona possession), `A` (Atletic Club
possession), `D` (dead ball / out of play), or `X` (ungraded). The submission is 199,319 rows.

The model is a **data-only, gap-aware 3-state HMM**. It deliberately excludes the earlier
MOSSE + GrayDiff + NanoTrack video ensemble: the tactical tracking already provides a cleaner,
synchronised observations stream, and the public window is only a correctness check.

## Inputs

| Input | Use |
|---|---|
| `BAR-ATH_tactical_atd.txt/.xml` | Ball and 46 anonymous player slots, metres, 25 fps |
| `BAR-ATH_tactical_smart_tagging.xml` | 638 intervals, 14 codes, 6 types, team/type labels |
| `BAR-ATH_tactical_homography.csv` | Per-frame camera geometry; diagnostics and pitch-space validity |
| `evaluation/gt/...public.csv` | 7,500-frame public GT; evaluation only |

`video_frame = frame_id - 1`. Smart-tag intervals use
`[round(start*25), round(end*25)+1)`. The tactical homography is valid on 100% of frames (median
reprojection error 0.24 px, p99 4.81 px).

## Features

For each frame:

* `d_H`, `d_A`: distance from the ball to the nearest home/away tracked player.
* `near_team`, `near_distance`, `second_distance`.
* `ball_speed`, `ball_acceleration` (raw, only across consecutive observations).
* `team_speed`: mean speed of the tracked players, smoothed over one second.
* `gap`: frames since the last observed ball.
* `edge`, `last_edge`: distance of the ball (or last known ball) from the nearest pitch boundary.
* `out_of_play`: observed ball outside `[0,L] × [0,W]`.
* `last_touch`, `last_touch_age`: nearest player within 0.8 m.
* `d_H_last`, `d_A_last`: current team shape around the last observed ball.
* `tag_vote`, `tag_setpiece`, `tag_coverage`.
* `player_count`: visible player slots, used for the sparse-tracking guard.
* `parked_edge`: the ball visible within `PARKED_NEAR = 2.5 m` of a boundary at
  `PARKED_SPEED = 1.5 m/s` or less, held for `PARKED_HOLD_S = 0.8 s` (up to 70% tracking
  flicker inside the window is tolerated). This is the throw-in / corner / goal-kick signature;
  observed-out frames (`out_of_play`) are included.

## Emissions

At visible frames, with `σ = 3.5 m`, `d_dead = 7 m`:

```
q_H = exp(-d_H² / (2σ²)) + 1e-12
q_A = exp(-d_A² / (2σ²)) + 1e-12
q_D = sigmoid((min(d_H, d_A) - d_dead) / 1.5) + 1e-12
q_D += w_dead + w_motion * sigmoid((motion0 - team_speed) / motion_scale)
     + w_edge * sigmoid((edge0 - edge) / edge_scale)
q_D += w_out * out_of_play
q_D += w_parked_edge * parked_edge
q   = [q_H, q_A, q_D] / sum(q)
logB = log(q)
```

The validated model sets `w_out = 1`, `w_dead = 0.02`, `w_motion = 1`, `w_edge = 0.5`,
`w_parked_edge = 1.5`, `motion0 = 1.6 m/s`, `motion_scale = 0.35 m/s`. The motion and edge terms
are the dead-ball evidence model: they do not look at the ball at all, which is what lets `D`
separate a stoppage where the ball is visible and a player is standing next to it. The
`parked_edge` term is the strongest single dead-ball signature measured on the public window
(spell F1 0.6074 → 0.6857): a ball stopped within 2.5 m of a line for at least 0.8 s is a set
piece being prepared. `w_dead` is the dead-ball base rate; it keeps the `D` prior independent of
the scale of the team Gaussians. The speed gate is applied even when `gate = 0`:

```
α = 1 - sigmoid((ball_speed - gate) / 2)
logB = α * logB + (1 - α) * log(1/3)
```

so a stationary or unknown-speed ball is blended 50/50 with uniform. This is a deliberate
regulariser and is required to reproduce the reference metrics.

Loose-ball rule. A ball far from every player is not evidence of a stoppage: it is loose or in
flight. When `d_min > loose_R = 7 m` and the ball is not out of play, the emission is uniform and
the transition prior carries the possession. If the ball is also slower than
`loose_speed = 2 m/s`, the only possession evidence left is the team that touched it last, so that
team gets `touch_p = 0.7` of the emission mass, provided the touch is younger than
`touch_age = 125` frames (5 s). The remaining mass is split evenly over the other two states. This
removed the false `D` stretches where a rolling or in-flight ball sits 7–10 m from both teams.
An optional `loose_stop_gate` (off in the validated model, kept for ablations) exempts loose
frames that carry stoppage support — `parked_edge` — from this flattening: a ball parked near a
line is a restart waiting to be taken, not a live loose ball.

Missing-ball emissions:

* `persist`: uniform for every missing frame.
* `hybrid` (recommended): uniform for `gap ≤ 120` frames (4.8 s); for longer holes use

```
m_H = exp(-d_H_last² / (2σ_miss²)) + 1e-6
m_A = exp(-d_A_last² / (2σ_miss²)) + 1e-6
m_D = 0.15 + sigmoid((5 - last_edge) / 1.5)
m_H *= 1 + max(tag_vote, 0)
m_A *= 1 + max(-tag_vote, 0)
m_D *= 1 + 0.2 * tag_setpiece
```

with `σ_miss = 6 m`. If fewer than 6 players are visible, the frame is treated as missing
evidence and the emission is uniform; this avoids reading team shape from a near-empty tracking
feed. The public window has at least 19 visible players, so this guard does not affect public or
stress scores.

## Transition and decode

```
T = [[.995,   .0005, .001],
     [.0005,  .995,  .001],
     [.01,    .01,   .98]]
```

Viterbi runs over each graded half with a leading 500-frame context pad. Frames outside
`(11,185, 90,652)` and `(110,808, 190,198)` are `X`. Forward–backward posterior confidence is
computed for diagnostics only. After decoding, a run shorter than one second that is flanked by
the same label is absorbed by it: a few frames of `D` (or of the other team) between two equal
labels are a tracking or ball-in-flight flicker, not a possession change. `X` is never changed.

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
public GT, the smart-tag table, and `metrics_public.json`, then serves it from a local stdlib HTTP
server. The page is plain HTML/CSS/JS; no frontend build step or extra Python dependency is added.

* **Synchronization.** `frame i` is exactly row `i` of the submission. Browser seeking uses
  `currentTime = (i + 0.5) / 25`; the displayed frame is
  `clamp(floor(currentTime * 25 + 1e-6), 0, 199318)`. The video duration is checked against
  `199,319 / 25 = 7972.76 s` when metadata loads.
* **Payload.** Labels are compact `H/A/D/X` strings; prediction segments/spells cover both the full
  graded halves, while GT segments/spells cover only the public window because the public CSV has
  `X` outside it. Changes, disagreement runs (grouped in the browser), 638 tag intervals, and
  public metrics are included. When `outputs/explanations.json` exists it is folded in as
  `switches` and `explanations_available`; the UI degrades quietly without it.
* **Decision annotations.** Every predicted switch is a thin tick on the PRED band of both
  timelines; ticks are hit-tested within 6 px, seeking and opening the decision card on click.
  The card lists every switch with a clickable timestamp, the
  `from → to` transition, the emission confidence bar, and the ordered reason chips from
  `explain.py` — each reason's evidence frame is itself a seek button. Filters: all / → D /
  D → / H ↔ A; `j`/`k` jump to the previous/next annotated switch; selection auto-follows
  playback within ±1.5 s.
* **User truth (UT).** The browser can paint `H/A/D/X` on the current frame or an anchored range,
  from the buttons or globally from the keyboard. Each label press leaves an anchor at the end of
  the written span; run boundaries are drawn on both timelines and can be dragged to move a cut.
  `POST /api/user-truth` updates the UT array (paint, boundary move, erase nearest run, or
  clear), writes
  `outputs/user_truth.json` atomically, and returns recomputed UT segments and official-scorer
  metrics over exactly the tagged frames. The UT is viewer-only: `features.py`, `model.py`, and
  `pipeline.py` never import or read it, and the fixed `predictions.csv` is never modified.
* **Streaming.** `/video` implements a single-range `Range` parser and returns `206`, `200`, or
  `416` as appropriate. The 5.2 GB tactical MP4 has its `moov` atom at the end; the browser reads
  the tail range for metadata and then fetches playback ranges. Routes are whitelist-only:
  `/`, `/static/viewer.css`, `/static/viewer.js`, `/api/viewer-data`, `/api/user-truth`, `/video`,
  `/favicon.ico`.
* **Scope.** The UI edits only the UT, has no correction export, and has no broadcast-video path.
  GT is shown only on non-`X` public frames. The page works entirely from `127.0.0.1`; the POST
  endpoint requires `application/json`.

The reviewer is covered by `tests/test_viewer.py`: payload semantics, public-only GT behavior,
UT scoring/persistence/editing, range parsing (`normal`, open, suffix, invalid), HTTP
`206`/`416`/`HEAD`, API payload, and unknown-route rejection.

## Validation

* Unit tests: 53 stdlib `unittest` tests (+4 subtests), all passing — including switch/reason
  rules (`test_explain.py`), stop-signal parameter neutrality and gating (`test_model.py`), and
  annotation payload plumbing (`test_viewer.py`).
* Public metrics for the current parameters: spell F1 0.6857, macro F1 0.8843,
  D-F1 0.8441, 11 predicted spells (17 in GT).
* Stress (HMM gap-aware, current parameters): clean 0.6857 / random 0.4615 / long 0.4000 spell F1.
* Weight-zero neutrality: with `w_parked_edge = 0` the new code
  reproduces the previous `predictions.csv` bit-for-bit (verified before adopting the tuned row).
* The files in `reports/data/` were produced by the pre-retune parameter set (`σ=3.0`,
  `p_dh=0.001`, no dead-ball evidence, no loose-ball rule) and are kept as history, not as an
  equality target.
* Submission: 199,319 rows, frames 0–199,318, official validator `OK`.

## Full-match reconciliation

A previous reference run reported 256 spells and H 41.4% / A 32.5% / D 26.1%; the pre-retune run
reported 255 spells and H 40.5% / A 32.5% / D 26.9%; the previous parameter set reported 239
spells and H 44.1% / A 34.6% / D 21.3%. The current run reports 186 spells and
H 42.3% / A 31.9% / D 25.8%. One part of the earlier difference is frames 89,905–90,651: the ATD
has no home players and only 1–3 away players, and the ball is missing. The old run's NaN
emissions effectively reset to H. The sparse-tracking guard treats the stretch as missing evidence
and predicts D, which the public and stress metrics are insensitive to. The rest is the retune:
wider team Gaussian, easier exits from D, the dead-ball evidence model, and the loose-ball rule
that returns a slow ball far from every player to the team that touched it last. This is a
full-match structural diagnostic, not a score.

## Reproducibility and privacy

* `python -m possession_lab all --bootstrap-n 2000` regenerates every output.
* `python -m unittest discover -s possession_lab/tests -v` runs the tests.
* `python evaluation/validate_submission.py possession_lab/outputs/predictions.csv` runs the
  official format gate.
* `python -m possession_lab explain` rebuilds only `outputs/explanations.json` from an existing
  `predictions.csv`.
* The gated `data/` bundle and generated `outputs/` are not committed.
* No private full-match GT and no match video frames are used.
* CPU is intentional: the 3-state Viterbi pass takes ~1.8 s; a GPU transfer would dominate.
