# Verification evidence

Environment: repository virtualenv (`.venv`), Python 3.12.3, Linux CPU. No GPU path is used.
Unless noted, commands run from the repository root.

## Unit and syntax checks

```text
$ .venv/bin/python -m pytest -q possession_lab/tests
75 passed, 4 subtests passed in 4.79s

$ .venv/bin/python -m compileall -q possession_lab
$ node --check possession_lab/viewer_assets/viewer.js
compileall + node check OK
```

The suite covers ATD/tag/homography parsing, feature geometry and gap state, ball-tracker holes,
whistle scoring/cache fallbacks, goal/stoppage detection and overrides, HMM emissions and
Viterbi/posterior consistency, sparse tracking, official metric wiring, stress masks, full-match
structure, CSV writing, preflight validation, the explanation layer, and the local viewer
(payload, public-only GT band, per-frame `/api/frame`, top-view pitch/trail payload, UT
scoring/editing/persistence and boundary moves, byte ranges, HTTP 206/416/HEAD, API, and route
whitelist).

## Regenerated outputs

```text
$ .venv/bin/python -m possession_lab all
loaded tactical data in 2.77s: 199,319 frames, 46 player slots, 638 tag intervals
decoded full match in 4.56s
explanations: 298 switches anotados -> possession_lab/outputs/explanations.json
public: spellF1 0.6519 | macroF1 0.9466 | D-F1 0.9727 | pred spells 10
submission: {'ok': True, 'n_rows': 199319, 'unknown': [],
             'counts': {'H': 65338, 'A': 42387, 'D': 51132, 'X': 40462}, 'n_runs': 305}
```

| Artifact | MD5 |
|---|---|
| `predictions.csv` | `75b1968adad6795169aca4813d4ef38e` |
| `metrics_public.json` | `29b983a9a3a037363cbe71f632b87881` |
| `stress.csv` | `88279db642566e5540a53f7139d3115b` |
| `explanations.json` | `619e4bbcd958c9781d3f680d1138252f` |

The run used the tactical MP4's cached audio score (`outputs/whistle.npz`: 199,319 frames, 274
candidate blasts, 7,972.69 s decoded) and the final parameters in `config.py:MODEL_PARAMS`.

## Public metrics

Public window: frames 11,185–18,684 (7,500 frames, 17 GT possessions).

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
| Spell-F1 95% bootstrap | [0.4296, 0.8175] (n=2000) |

The wide interval is expected with 17 GT possessions. The D-F1 is high because the event layer
pins confirmed goals/stoppages; ordinary possession and ordinary out-of-play remain HMM decisions.

## Missing-ball stress

The stress suite isolates the HMM core and deliberately **does not apply the goal/stoppage hard
overrides**, so its clean row is lower than the final public score above.

```text
mask    method              spellF1   macroF1   D-F1    n_pred
clean   baseline_w50        0.3256    0.6207    0.3225    26
clean   baseline_w100       0.4485    0.5976    0.1992    16
clean   HMM persistence    0.4308    0.6045    0.2341     9
clean   HMM gap-aware      0.6143    0.7299    0.5144    11
random  baseline_w50        0.2414    0.5730    0.2974    41
random  baseline_w100       0.2900    0.5542    0.1945    23
random  HMM persistence    0.2957    0.7045    0.5494     6
random  HMM gap-aware      0.3500    0.7049    0.5209     7
long    baseline_w50        0.1789    0.4790    0.3022    21
long    baseline_w100       0.2452    0.4897    0.2630    14
long    HMM persistence    0.1818    0.6171    0.2341     5
long    HMM gap-aware      0.3840    0.6387    0.5602     8
```

Random and long each remove about 40.7% of the public window. The gap-aware HMM beats both
baselines on every mask; the long holes are the hardest case.

## Events

Goal evidence:

```text
46,691  trajectory   46,691–49,631
157,825 celebration 157,825–163,000
```

The tagged candidates at 57,283 and 73,710 are rejected: the ball was observed behind the goal
line but wide of the posts.

Confirmed public stoppages after the kick-off trim (hard `D` overrides):

```text
11,035–11,097   out_gap, trimmed at the kick-off
12,852–12,991   whistle, peak 4.62, low motion starts 12,886
15,888–16,405   whistle, peak 5.98
17,377–17,828   whistle
18,025–18,373   out (visible ball beyond the line)
```

Full match: 68 stoppages and 2 goals. The 12,851 injury/referee stop is handled as
`D` from 12,852; the whistle candidate is at 12,841 and the ball is missing from 12,831 to 12,979.
The 15,221 false “Balón muerto” and the 17,230 gap are not hard stops.

## Full submission and validator

```text
rows: 199,319; frames 0..199,318
counts: H 65,338 | A 42,387 | D 51,132 | X 40,462
spells: 198 (100 first half, 98 second) over 158,857 graded frames
duration: mean 21.8 s, median 15.6 s (min 0.84 s, max 88.8 s)
shares: H 41.1% | A 26.7% | D 32.2%
runs: 305 (includes X transitions)
```

```text
$ .venv/bin/python evaluation/validate_submission.py possession_lab/outputs/predictions.csv
OK predictions.csv
```

The validator is a format gate, not a score. `reports/data/` still contains historical pre-retune
reference outputs; it is not an equality target for the final run.

## Viewer integration

The real 5.2 GB tactical MP4 was served with HTTP byte ranges and loaded in headless Chrome
(Chromium 1243, CDP-driven) against the live server on `127.0.0.1:8765`.

```text
GET /api/viewer-data       200  application/json  656,805 bytes
pred segments/spells       302 / 198 (full graded halves)
GT segments/spells          21 / 17  (public window only)
changes / disagreements    304 changes
tags                       638 intervals
events / ball gaps          70 / 149
switches / explanations    298 / true
GET /api/frame?frame=12851 200  3,230 bytes
  keys                      active_events, available, ball, features, fps, frame,
                            pitch, players, trail, trail_pitch, time_s,
                            video_height, video_width
  pitch / players           105.3 × 68.0 m / 21 slots
  trails                    25 projected / 25 pitch-space points
GET /video Range: bytes=0-1023
                           206  1,024 bytes
Content-Range              bytes 0-1023/5222307334
GET /video Range: bytes=5222305286-5222307333
                           206  2,048 bytes
Content-Range              bytes 5222305286-5222307333/5222307334
```

Browser checks:

* Metadata loaded: `readyState=4`, duration `7972.76 s`; initial frame 12,851 at
  `currentTime=514.06` with status `video sincronizado`.
* Overlay: video and overlay rectangles both measured `1139.4 × 640.9` CSS px after moving the
  canvas into the `.video-frame` wrapper. Before the fix the overlay was `1139.4 × 678.5` because
  the global `canvas { height: 100% }` filled the whole video card, including the caption.
* Frame 12,000: 22 projected players, observed ball; a screenshot showed the projected dots on
  the players and the white ball ring on the ball.
* Frame 12,851: 21 projected players and the estimated ball at `30.98, 55.79 m`; the ball gap was
  `0.84 s (21 f)`, ball state `estimado`.
* Top-view map: canvas `658 × 424` CSS px, non-blank; frame 12,000 status `22 slots visibles ·
  balón observado`, frame 12,851 status `21 slots visibles · balón estimado · hueco 0.84 s`.
  The map uses `pitch`, `players[].x/y`, `ball.ex/ey`, and `trail_pitch`; it does not read video
  pixels or call the homography.
* Explorer at 12,851: `Jugador cerca —`, `Dist. a línea —`, `Vel. equipos 3.79 m/s`, last touch
  `Barcelona · 0.84 s`, event `juego abierto` (the stop begins on 12,852).
* Seeking through the UI's own Go button to frame 46,691 moved video `currentTime` to `1867.66`
  and displayed `GOL · trajectory`.
* Frame 157,825 displayed `GOL · celebration`; frame 15,888 displayed `PARADA · whistle`.
* Play advanced `currentTime` from `1867.66` to `1869.58` and the frame counter from 46,691 to
  46,738, confirming video playback and frame sync.
* The overlay toggle cleared/restored the tracking layer; the decision annotations rendered on
  both timelines.
* A headless screenshot with only a short virtual-time budget stalls video metadata (media time
  does not advance under virtual time); the CDP run is the authoritative browser evidence.

## Tracking data audit

The audit is deliberately about the supplied ATD, not about the viewer's rendering:

```text
rows                        199,319; every row has exactly 46 player tokens
team slots                  24 Barcelona, 22 Athletic Club
visible slots (median)      21; min 1; max 35
home > 11 in                22,859 frames (max 24)
away > 11 in                 8,905 frames (max 15)
ball observed               113,488 (56.94%)
ball estimated               4,441 (2.23%)
ball without estimate       81,390 (40.83%)
```

The viewer therefore shows anonymous tracking slots, not a guaranteed active XI. The extra slots
are not filtered; the model's nearest-player/team features use every present slot. No ReID,
jersey OCR, or identity assignment is performed. This is the main reason a dot can occasionally
look like the “wrong” player, and why the GUI must not be read as a named-player tracker.

## Data / privacy check

* No private full-match ground truth is read, written, or inferred.
* The public GT (`evaluation/gt/BAR-ATH_possession_gt_public.csv`) is used only for evaluation.
* The viewer-only UT (`outputs/user_truth.json`) is never read by the model, features, or
  prediction path; it scores the fixed prediction inside the viewer.
* The tactical MP4 audio is used only for whistle detection. Match video frames are not copied
  into outputs.
* Broadcast video is not used.
* `data/` is gated and git-ignored; `possession_lab/outputs/` is git-ignored, including
  `predictions.csv`, `whistle.npz`, and `explanations.json`.
