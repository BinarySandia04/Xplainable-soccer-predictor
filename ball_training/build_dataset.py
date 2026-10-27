#!/usr/bin/env python3
"""Merge every ball source into one YOLO dataset for training.

Sources (all already on disk under ``ball_training/external``, fetched as described in
``ball_training/README.md``):

* SoccerNet v3 H250 (Zenodo 7808511, CC-BY-4.0): broadcast long shots, class ``0`` is
  the ball and class ``1`` is a person.  Only the ball boxes are kept.  Frames without a
  ball annotation are *not* verified negatives (SoccerNet annotates the ball when it
  considers it visible), so they enter training as negatives up to ``--sn-negative-cap``.
* infactory-ai/ball-tracking (CC-BY-NC-4.0): human verified visible/not-visible frames.
  The ``not_visible`` frames are true hard negatives.  Train/val is split per source clip
  so neighbouring frames of one clip never straddle the split.
* martinjolif/football-ball-detection (Roboflow, CC-BY-4.0): very tiny balls, official
  train/valid split is respected.
* ``ball_training/match`` from ``label_match.py``: the deployment domain itself (tactical
  wide view, ~10 px ball).  Every ``--match-val-every``-th label is held out for
  validation, spread over the whole match.

Images are symlinked, labels are rewritten (filtered / normalised), so the merged dataset
costs only the label text.

Run:  .venv/bin/python ball_training/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE / "external"
SN = EXTERNAL / "sn_h250" / "YOLO"
INFO = EXTERNAL / "infactory_ball" / "data"
MARTIN = EXTERNAL / "martinjolif" / "data"
MATCH = HERE / "match"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def stem_of(path: Path) -> str:
    return path.stem


def link_image(image: Path, dest: Path) -> None:
    """Symlink an image into the merged dataset (re-creating a stale or copied link)."""
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(image.resolve())


def write_label(lines: list[str], dest: Path) -> None:
    dest.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def sn_pairs(split: str) -> list[tuple[Path, Path]]:
    out = []
    for label in sorted((SN / split / "labels").glob("*.txt")):
        for suffix in IMAGE_SUFFIXES:
            image = SN / split / "images" / f"{label.stem}{suffix}"
            if image.is_file():
                out.append((image, label))
                break
    return out


def martin_pairs(split: str) -> list[tuple[Path, Path]]:
    directory = MARTIN / split
    out = []
    images = directory / "images"
    labels = directory / "labels"
    if not images.is_dir():
        return out
    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels / f"{image.stem}.txt"
        if label.is_file():
            out.append((image, label))
    return out


def info_pairs() -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """infactory frames as (clip, image, label-lines); returns (train, val) split by clip."""
    frames: list[tuple[str, Path, list[str]]] = []
    for kind, lines_for in (("visible", True), ("not_visible", False)):
        for image in sorted((INFO / kind).glob("*.jpg")):
            label = image.with_suffix(".txt")
            lines: list[str] = []
            if lines_for and label.is_file():
                lines = [ln.strip() for ln in label.read_text().splitlines() if ln.strip()]
                lines = [ln for ln in lines if int(ln.split()[0]) == 0]
            frames.append((stem_of(image).split("_")[0], image, lines))
    clips = sorted({clip for clip, _img, _lines in frames})
    random.Random(7).shuffle(clips)
    val_clips = set(clips[: max(1, len(clips) // 5)])
    train, val = [], []
    for clip, image, lines in frames:
        (val if clip in val_clips else train).append((image, lines))
    return train, val


def match_pairs(block_frames: int, block_mod: int, block_rem: int, guard: int = 50):
    """Match labels split by *time block*, never by interleaved index.

    Labelled frames sit seconds apart, so an every-nth split would put two frames of the
    same pass of the ball on both sides of the split and validate on near-duplicates.
    Whole ``block_frames``-long blocks of the match go to validation instead, and training
    drops anything within ``guard`` frames of a validation block edge, because two blocks
    are otherwise neighbours on the timeline.
    """
    images = sorted((MATCH / "images").glob("*.jpg")) if MATCH.is_dir() else []
    train, val, dropped = [], [], 0
    for image in images:
        label = MATCH / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            continue
        frame = int(image.stem.lstrip("m")) if image.stem.startswith("m") else 0
        block = frame // block_frames
        if block % block_mod == block_rem:
            val.append((image, label))
            continue
        edge = (block + 1) * block_frames
        if frame - block * block_frames < guard or edge - frame < guard:
            dropped += 1  # too close to a validation block to score against
            continue
        train.append((image, label))
    if dropped:
        print(f"build_dataset: dropped {dropped} match frames adjacent to a validation block")
    return train, val


def keep_class(label: Path, wanted: int) -> list[str]:
    lines = []
    for line in label.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if int(parts[0]) == wanted:
            parts[0] = "0"
            lines.append(" ".join(parts))
    return lines


def emit(prefix: str, image: Path, lines: list[str], split: str, counter: Counter) -> None:
    images_dir = OUT / "images" / split
    labels_dir = OUT / "labels" / split
    name = f"{prefix}__{stem_of(image)}"
    link_image(image, images_dir / f"{name}{image.suffix.lower()}")
    write_label(lines, labels_dir / f"{name}.txt")
    counter[f"{split}:{'pos' if lines else 'neg'}"] += 1


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "dataset"))
    parser.add_argument("--sn-negative-cap", type=int, default=3000)
    parser.add_argument("--sn-val-images", type=int, default=800)
    parser.add_argument("--info-negative-cap", type=int, default=2200)
    parser.add_argument("--match-block-frames", type=int, default=1500, help="60 s block at 25 fps")
    parser.add_argument("--match-block-mod", type=int, default=7)
    parser.add_argument("--match-block-rem", type=int, default=3)
    parser.add_argument("--with-match", action="store_true")
    args = parser.parse_args()

    OUT = Path(args.out)
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    counter: Counter = Counter()
    rng = random.Random(11)

    # --- SoccerNet v3 H250: ball boxes only -------------------------------------------
    positives, negatives = [], []
    for image, label in sn_pairs("train"):
        lines = keep_class(label, 0)
        (positives if lines else negatives).append((image, lines))
    for image, lines in positives:
        emit("sn", image, lines, "train", counter)
    rng.shuffle(negatives)
    for image, lines in negatives[: max(0, args.sn_negative_cap)]:
        emit("sn", image, lines, "train", counter)

    sn_val = [(image, keep_class(label, 0)) for image, label in sn_pairs("valid")]
    sn_val_pos = [pair for pair in sn_val if pair[1]]
    for image, lines in sn_val_pos[: max(0, args.sn_val_images)]:
        emit("sn", image, lines, "val", counter)

    # --- infactory: human verified visible + verified not-visible negatives -----------
    info_train, info_val = info_pairs()
    info_pos = [pair for pair in info_train if pair[1]]
    info_neg = [pair for pair in info_train if not pair[1]]
    for image, lines in info_pos:
        emit("info", image, lines, "train", counter)
    for image, lines in info_neg[: max(0, args.info_negative_cap)]:
        emit("info", image, lines, "train", counter)
    for image, lines in info_val:
        emit("info", image, lines, "val", counter)

    # --- martinjolif: official split, tiny balls --------------------------------------
    for image, label in martin_pairs("train"):
        emit("mj", image, keep_class(label, 0), "train", counter)
    for image, label in martin_pairs("valid"):
        emit("mj", image, keep_class(label, 0), "val", counter)

    # --- the tactical match itself -----------------------------------------------------
    if args.with_match:
        match_train, match_val = match_pairs(
            args.match_block_frames, args.match_block_mod, args.match_block_rem
        )
        for image, label in match_train:
            emit("mt", image, keep_class(label, 0), "train", counter)
        for image, label in match_val:
            emit("mt", image, keep_class(label, 0), "val", counter)

    yaml = (
        f"# Generated by ball_training/build_dataset.py -- do not edit by hand.\n"
        f"path: {OUT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: ball\n"
    )
    (OUT / "data.yaml").write_text(yaml, encoding="utf-8")

    stats = dict(counter)
    stats["train_total"] = stats.get("train:pos", 0) + stats.get("train:neg", 0)
    stats["val_total"] = stats.get("val:pos", 0) + stats.get("val:neg", 0)
    (OUT / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("build_dataset:", json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
