# Xplainable Soccer Predictor — More than a Hack 2027

> **Our submission:** deep video perception with two dedicated YOLO26x detectors (ball, players),
> fused with the provided tracking, smart tagging and whistle audio into a per-frame evidence pool
> (`fusion`), then a transition layer that turns the pool into possession spells — dead-ball locks,
> tag-based revival, possession-based revival — and an interactive explainability viewer.
>
> * **Initialize it:** quick start with the two execution paths (no-GPU re-run from the release,
>   full GPU regeneration) is in **[Initialize the system](#initialize-the-system)** below; the
>   exact commands, checksums and the full chain are in **[REPRODUCING.md](REPRODUCING.md)**.
> * **How the models were trained:** external public datasets only — no frame of the provided
>   match and no possession label — in **[How the models were trained](#how-the-models-were-trained)** below.
> * **Performance:** **0.7600 spellF1** and **0.9691 macroF1** on the public 5-minute window
>   (best published baseline: 0.4242 spellF1 / 0.6207 macroF1).
> * **Verification:** 315 passing tests (`pytest`), clean linter (`ruff check .`), official format
>   validation, and every artefact of the champion run SHA-256-pinned.
> * **Docs:** **[possession_lab/README.md](possession_lab/README.md)** (implementation + CLI),
>   **[PROPOSAL.md](PROPOSAL.md)** (original design), **[TODO.md](TODO.md)** (open items).

---

# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play                                                                                           |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and its format, and the frame alignment: **[DATA.md](DATA.md)**.

---

## Initialize the system

Python 3.12 (the validated run uses 3.12.3), one venv, nothing else. The match assets are **not**
in this repository — see *Getting the data* and unpack the bundle under `data/`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-possession.txt   # CPU chain (fusion, transitions, viewer)
.venv/bin/python -m pip install -r requirements-perception.txt   # + YOLO (only for the GPU perception path)
.venv/bin/python -m pytest -q possession_lab/tests               # 315 passed (+4 subtests)
```

### Path A — no GPU: regenerate the submission CSV from the released artefacts

The only GPU step is perception (YOLO over all 199,319 frames). Its finished output — and the
fusion pool — ship in the [`champion-run` release](https://github.com/BinarySandia04/Xplainable-soccer-predictor/releases/tag/champion-run),
so the exact submission CSV can be regenerated on a laptop:

```bash
gh release download champion-run --pattern "perception_new_fixed.npz" --dir possession_lab/outputs
gh release download champion-run --pattern "fusion.npz"               --dir possession_lab/outputs

.venv/bin/python -m possession_lab transitions \
    --perception possession_lab/outputs/perception_new_fixed.npz \
    --fusion possession_lab/outputs/fusion.npz
.venv/bin/python -m possession_lab explain \
    --perception possession_lab/outputs/perception_new_fixed.npz \
    --predictions possession_lab/outputs/transition_predictions.csv
.venv/bin/python evaluation/validate_submission.py possession_lab/outputs/transition_predictions.csv
```

`sha256sum` of the produced files must match the table in [REPRODUCING.md](REPRODUCING.md) —
that is the definition of "reproduced" for this repository.

To **watch the real detections** on the tactical video without running any model, add the viewer
bundle from the same release and:

```bash
gh release download champion-run --pattern "outputs_view_bundle.tar.gz" --dir /tmp
tar xzf /tmp/outputs_view_bundle.tar.gz -C possession_lab/outputs/
.venv/bin/python -m possession_lab viewer \
    --perception  possession_lab/outputs/perception_new_fixed.npz \
    --fusion      possession_lab/outputs/fusion.npz \
    --predictions possession_lab/outputs/transition_predictions.csv
```

then open `http://localhost:8765` (or `--port N` to change it).

### Path B — full regeneration from raw data (GPU)

```bash
# 1. Trained weights (external data only; SHA-256 in REPRODUCING.md)
gh release download champion-run --pattern "*.pt" --dir .

# 2. Perception: ball + players over the tactical video, keeper rescue included.
#    A SLURM wrapper for GPU clusters is provided: farm_perceive.sbatch (see FARM_README.txt).
python farm_perceive.py \
    --video data/tactical/BAR-ATH_tactical-001.mp4 \
    --ball-weights ball_yolo26x_1024_clu.pt \
    --players-weights players_yolo26x_1024_clu.pt \
    --output-dir possession_lab/outputs

# 3. Evidence pool, transition layer, explanations, validation.
.venv/bin/python -m possession_lab fusion \
    --perception possession_lab/outputs/perception_new_fixed.npz
# then the same two commands as Path A: transitions, explain, validate.
```

### Path C — data-only HMM (no video, no YOLO)

The original gap-aware HMM over tracking + tags + whistle audio, before video perception was
added:

```bash
.venv/bin/python -m possession_lab all --bootstrap-n 2000
.venv/bin/python evaluation/validate_submission.py possession_lab/outputs/predictions.csv
```

---

## How the models were trained

The only learned components in the submission are two YOLO26x @ 1024 px detectors. Both were
trained with DDP on a multi-GPU server, on **external public data only**: no frame of the provided
match and no possession ground-truth label entered any training or tuning step (per the challenge
rules and the organisers' clarification of November 2026).

| weight | external training data | training script |
|---|---|---|
| `ball_yolo26x_1024_clu.pt` | SoccerNet v3 H250 ([Zenodo](https://zenodo.org/records/7808511), CC-BY-4.0) · infactory-ai/ball-tracking ([HF](https://huggingface.co/datasets/infactory-ai/ball-tracking), CC-BY-NC-4.0) · martinjolif/football-ball-detection ([HF](https://huggingface.co/datasets/martinjolif/football-ball-detection), CC-BY-4.0) | `ball_training/train_cluster.py` |
| `players_yolo26x_1024_clu.pt` | SoccerNet/SN-Tracking-2023 ([HF](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023), research terms) · martinjolif/football-player-detection ([HF](https://huggingface.co/datasets/martinjolif/football-player-detection), CC-BY-4.0) · keremberke/football-object-detection ([HF](https://huggingface.co/datasets/keremberke/football-object-detection), CC-BY-4.0) · manot/football-players ([HF](https://huggingface.co/datasets/manot/football-players), MIT) | `ball_training/train_players_cluster.py` |

The player model covers classes *player / goalkeeper / referee*. All training code is committed;
the trained weights are in the `champion-run` release with SHA-256 pinned in
[REPRODUCING.md](REPRODUCING.md). The YOLO11s experiment in `ball_training/` mixed frames of the
provided match into its training set; it is **not** part of this chain and is documented as
excluded (see its README's status banner).

---

## Reproduce the possession lab

Everything the champion run consumes and produces, the exact commands, the model weights with
SHA-256, and the expected checksums of every intermediate artefact are documented in
[**REPRODUCING.md**](REPRODUCING.md). Do not commit the gated data or generated full-match
predictions; the repository must regenerate them, not ship them.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[REPRODUCING.md](REPRODUCING.md)** | the champion chain: commands, weights, checksums |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
farm_perceive.py            GPU perception driver (two-stage inference + keeper rescue)
farm_perceive.sbatch        SLURM wrapper for the perception step
FARM_README.txt             how the farm bundle is built and run
REPRODUCING.md              how to regenerate the exact submission CSV — weights, commands, checksums
TODO.md                     open items before the deadline
ball_training/              detector training scripts + the excluded YOLO11s experiment
weights/                    local copies of the trained detectors (git-ignored; published in the champion-run release)
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
