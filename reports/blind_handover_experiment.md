# Experiment: blind hand-over snap (rule 10, off by default)

Status: **implemented, tested, not shipped**. `main` runs with the rule disabled so the
submission CSV stays bit-identical to the validated `v9` build (spellF1 0.7600).

## The case that motivated it

Frame 38 497, outside the public ground-truth window (11 185-18 684):

- Barcelona (H) has the ball at trk1's feet until 38 486; the ball leaves at 16.5 m/s.
- The fused decode changes H -> A at **38 499**.
- No Athletic player is within **5.9 m** of the ball until 38 594; the first real control
  (trk40, within 1.5 m, ball below 6 m/s) is **38 596**.
- The boundary is decided by the speed gate: at 16.5 m/s `alpha = 1 - sigmoid(v/2) = 2.6e-4`,
  so the emission is uniform. The residual `log p_H - log p_A` flips sign at 38 499
  (+2.5e-5 -> -1.3e-5) because `dA` (6.26 m) becomes smaller than `dH` (6.40 m). Viterbi puts
  the single switch at that sign flip: a 1e-5-nat difference, not a touch.
- `hold_through_flight` cannot help: there is no witnessed release. The ball speed ramps
  smoothly (max 3-frame gain 1.53 m/s against the 6.0 m/s release threshold), so no kick
  anchor fires.

The missing player at 38 497 is trk23, Barcelona's goalkeeper, at (4.78, 36.25). He is **out
of the camera frame**: the homography covers pitch x 17.5-87.1 m and the keeper projects to
pixel x = -350. He is detected in 28.6% of the first-half frames where the ATD places him.

## The rule

`possession_lab/fusion/transitions.py::snap_blind_handovers`, parameters
`blind_handover_*` in `TransitionParams`.

A team-to-team change whose incoming side has nobody within `blind_handover_dist_m` (1.5 m)
of the ball cannot be a touch. Move the boundary forward to the first frame of the new spell
where the incoming side reaches a controllable ball (`blind_handover_control_m` 1.5 m,
`blind_handover_control_speed_m` 6 m/s), never more than
`blind_handover_max_shift_frames` (250 = 10 s). A change at the feet, an unseen ball, dead
frames and changes into/out of `D`/`X` are left alone, so no possession is invented.

## Measured cost

`evaluation/results/v10_blind_handover.csv`, official `run_evaluation.py`:

| build | spellF1 | macroF1 | accuracy | spells matched |
|---|---|---|---|---|
| main (`v9_clearance`) | **0.7600** | 0.9695 | 0.9675 | 14/17 |
| rule on (`v10_blind_handover`) | 0.7200 | 0.9525 | 0.9471 | 11/17 |

The rule moves four boundaries inside the public window and lands late every time:

| decode boundary | GT boundary | rule boundary |
|---|---|---|
| 11 342 | 11 351 | 11 460 (+109) |
| 12 647 | 12 665 | 12 706 (+41) |
| 13 730 | 13 785 | 13 839 (+54) |
| 14 843 | 14 846 | 14 880 (+34) |

## Why it fails (and what to try next)

1. **Body-centre geometry.** Tracked positions are body centres; a real control can be 2-3 m
   from the ball, so a 1.5 m "touch" radius is too tight. A larger radius would preserve
   cases 12 647 (1.69 m) and 14 843 (2.26 m) without touching the encode.
2. **ATD gaps.** "Nobody within X" only proves nobody is *tracked*. In case 13 730 the GT
   gives H the ball with the nearest tracked H player 9.7 m away. The same gap hides the
   Barcelona keeper at 38 497.
3. **Annotation convention.** The public GT changes possession *during the flight*: in three
   of four cases the GT boundary is within 18 frames of the decode's. Waiting for the
   tracked control is not what the labels do.

Next steps, in order of promise: (a) raise `blind_handover_dist_m`/`control_m` to a
geometrically justified ~2.5 m (body centre + reach + tracking noise) and re-measure —
**do not tune the threshold on the four GT cases**, that is fitting on possession GT;
(b) only snap when the ball is fast (>8 m/s) at the boundary, so slow-ball hand-overs are
untouched; (c) combine with a detection-coverage guard: skip the rule when any player is
missing from the tracking in the window (impossible to prove a non-touch).

## How to run the experiment

```bash
# enable (single parameter, default is 0 = off)
#   possession_lab/fusion/transitions.py: blind_handover_max_shift_frames = 250

P=possession_lab/outputs
.venv/bin/python -m possession_lab transitions \
    --perception $P/perception_new_fixed.npz --fusion $P/fusion.npz
.venv/bin/python -m possession_lab explain \
    --perception $P/perception_new_fixed.npz --predictions $P/transition_predictions.csv
cp $P/transition_predictions.csv evaluation/results/v10_blind_handover.csv
.venv/bin/python evaluation/run_evaluation.py
```

Expected: the CLI prints `blind handovers 2845`; the viewer shows H through 38 595 and A
from 38 596; the scoreboard puts `v10_blind_handover` at spellF1 0.7200. To go back, restore
the parameter to 0 and re-run the two commands (the fusion cache does not change).
