# Verification evidence

> Historical log: the metrics, label counts, and reference equality below were recorded before the
> `σ=3.5`, `p_dh=0.01` retune. They still describe the pre-retune run; current values are in
> `README.md`.

Environment: repository virtualenv (`.venv`), Python 3.12, Linux CPU. No GPU path is used.

## Unit tests

```text
$ python -m unittest discover -s possession_lab/tests -v
Ran 26 tests
OK
```

The suite covers ATD/tag/homography parsing, feature geometry and gap state, HMM emissions and
Viterbi/posterior consistency, sparse-tracking handling, official metric wiring, stress masks,
full-match structure, inference, CSV writing, preflight validation, and the local viewer (payload,
public-only GT band, UT scoring/editing/persistence and boundary moves, byte ranges, HTTP
206/416/HEAD, API, and route whitelist).

## Public prediction equality

```text
rows: 199,319
frames 0..199,318: True
public window (11,185:18,685) equal to reports/data/final_window_codes.npy: True
difference: 0 frames
label counts: H 64,393 | A 51,695 | D 42,769 | X 40,462
```

## Public metrics equality

`possession_lab/outputs/metrics_public.json` matches `reports/data/final_metrics.json` for every
field, including the 4x4 confusion matrix (the X row is zero after scoring only the public window):

| Field | Lab | Reference |
|---|---:|---:|
| `spell_f1_mean` | 0.49655172413793114 | 0.49655172413793114 |
| `macro_f1` | 0.770925425712762 | 0.770925425712762 |
| `D` F1 | 0.5916230366492146 | 0.5916230366492146 |
| predicted spells | 12 | 12 |
| confusion matrix | equal | equal |

## Stress-table equality

Every row and every numeric column matches `reports/data/final_stress.csv` exactly (12/12 rows,
max absolute difference 0.0 for spellF1, macroF1, and D-F1; `n_pred` also equal).

## Official format validator

```text
$ python evaluation/validate_submission.py possession_lab/outputs/predictions.csv
OK    predictions.csv
      199319 rows, frame 0..199318
      H 64393 32.3% | A 51695 25.9% | D 42769 21.5% | X 40462 20.3%
```

The validator reports 255 possessions using its X-as-gap convention. This is a format/structural
check, not a score.

## Full-match reconciliation

The lab reports 255 H/A spells, mean 18.21 s, H 40.5% / A 32.5% / D 26.9%. The earlier reference
run reported 256 spells, mean 18.36 s, H 41.4% / A 32.5% / D 26.1%. The difference is the first-half
tracking blackout at frames 89,905–90,651 (no home players, 1–3 away players, ball missing). The
older run's NaN emissions effectively reset to H; the lab's sparse-tracking guard treats this as
missing evidence and predicts D. Public and stress metrics are unaffected.

## Viewer integration

The real 5.2 GB tactical MP4 was served with byte ranges and loaded in Brave 155 (headless,
CDP-driven) against the live server.

```text
GET /api/viewer-data       200  application/json  515,072 bytes
pred segments/spells       393 / 255 (full graded halves)
GT segments/spells          21 / 17  (public window only)
changes / disagreements    395 / 1,367 frames
tags                       638 intervals
GET /video Range: bytes=5216206848-
                           206  6,100,486 bytes
Content-Range              bytes 5216206848-5222307333/5222307334
GET /video Range: bytes=0-1023
                           206  1,024 bytes
```

Browser checks:

* Initial frame **11,185** displayed `0:07:27.400`; video `currentTime` settled at `447.42 s`
  after metadata, with status `video sincronizado`.
* `next change` moved to frame **11,333** (`453.34 s`) and showed prediction `A` = GT `A`.
* `next disagreement` moved to frame **11,351** (`GT H`, pred `A`); `previous disagreement` returned
  to **11,185**.
* GT boundary: frame **18,684** showed `H`/`H`; frame **18,685** showed GT `X` /
  “GT pública no disponible”.
* Overview click at 50% moved to frame **99,659** (`3986.38 s`); detail click at 75% moved to
  **99,909** (`3996.38 s`).
* Play advanced from `3996.38 s` to `3998.18 s` (45 frames at 25 fps), confirming video playback is
  synchronized with the UI.
* `HEAD /video` returned `Content-Length: 5222307334`; an unsatisfiable range returned `416`.

## Data / privacy check

* No private full-match ground truth is read by the package.
* The viewer-only UT (`outputs/user_truth.json`) is never read by the prediction or training path.
* No match video frames are copied into outputs.
* `data/` is gated and git-ignored; `possession_lab/outputs/` is git-ignored.
* The only GT used is the public CSV under `evaluation/gt/` (7,500 graded frames; X outside).

---

# Retune evidence — parked ball at a boundary (2026-10-22)

All values below are the current validated state; they supersede the pre-retune numbers in the
first part of this file.

## Parked-edge weight sweep (public window)

`w_parked_edge` added `D` evidence on frames where the ball is seen within 2.5 m of a boundary at
≤ 1.5 m/s for at least 0.8 s (up to 70% tracking flicker tolerated). Below 1.5 the term never
outweighs the team Gaussians, so the decode is unchanged:

| `w_parked_edge` | spell F1 | macro F1 | D-F1 | predicted spells |
|---:|---:|---:|---:|---:|
| 0.00 | 0.6074 | 0.8925 | 0.8584 | 10 |
| 0.10 | 0.6074 | 0.8925 | 0.8584 | 10 |
| 0.30 | 0.6074 | 0.8925 | 0.8584 | 10 |
| 0.50 | 0.6074 | 0.8925 | 0.8584 | 10 |
| 1.00 | 0.6074 | 0.8925 | 0.8584 | 10 |
| **1.50** | **0.6857** | 0.8843 | 0.8441 | 11 |

Adopted 1.50: spell F1 +0.078 at a D-F1 cost of 0.014. The optional `loose_stop_gate` was measured
in the same grid and left `False` (it changed nothing on this window at the adopted weight).

## Equality and regression

```text
$ python -m possession_lab all --bootstrap-n 2000
predictions.csv: 199,319 rows | H 66,930 | A 50,637 | D 41,290 | X 40,462 | md5 29ec9f9ebe1e026db89972bc3d127b4b
$ python evaluation/validate_submission.py possession_lab/outputs/predictions.csv
Valid. (format only)
```

* With `w_parked_edge = 0` the code reproduces the previous `predictions.csv` bit-for-bit, so the
  parked signal is provably isolated (verified before adopting the tuned row).
* After the current refactor (explanation layer, viewer annotations, parameter cleanup) the
  regenerated `predictions.csv` is byte-identical to the tuned run above (same md5).
* Public window (7,500 frames): spell F1 **0.6857**, macro F1 **0.8843**, D-F1 **0.8441**,
  accuracy 0.8943, kappa 0.8342, alive/dead F1 0.9610, team accuracy (both alive) 0.9436,
  11 predicted spells (GT 17), spell-F1 95% bootstrap **[0.457, 0.834]** (n=2000).
* Stress (HMM gap-aware): clean 0.6857 / 0.8843 / 0.8441 · random 0.4615 / 0.7976 / 0.7385 ·
  long 0.4000 / 0.6121 / 0.5687.
* Full submission: 188 H/A spells (96 + 92) over 158,857 graded frames; mean 25.0 s, median 18.6 s;
  shares H 42.1% / A 31.9% / D 26.0%.

## Unit tests

```text
$ python -m unittest discover -s possession_lab/tests
Ran 53 tests in 4.059s
OK
```

(`pytest possession_lab/tests -q` reports the same suite: 53 passed + 4 subtests.)

## Explanation layer output

```text
$ python -m possession_lab explain
explanations: 295 switches anotados -> possession_lab/outputs/explanations.json
```

295 annotated switches across both graded halves (613 reason instances, a switch can carry several).
Reason instances per rule: `ball_missing` 143 · `geometry` 132 · `tag` 123 · `out_of_bounds` 80 ·
`parked` 57 · `teams_slow` 50 · `restart` 17 · `prior` 11 (switches where no local signature fired
and the transition prior carried the state). The payload is built downstream of
`predictions.csv` only.

## Viewer smoke

```text
$ python -m possession_lab viewer --no-open --port 8179
data: 199,319 frames | 7972.76s | 188 predicted spells | 295 annotated switches | GT yes | UT 0 frames
GET /api/viewer-data      200  application/json  650,954 bytes
switches                  295 | explanations_available: true
GET /                     200
```

The switch ticks, decision card (confidence bar, ordered reason chips with per-frame seek buttons,
all / → D / D → / H ↔ A filters) and `j`/`k` navigation respond on both canvases; ticks are
hit-tested within 6 px.
