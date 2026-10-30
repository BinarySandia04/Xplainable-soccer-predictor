#!/usr/bin/env python3
"""Two-stage tactical-video perception for a SLURM GPU farm.

This is the farm-side companion to ``possession_lab``.  It deliberately imports only
``numpy``, ``opencv`` and ``ultralytics`` (plus the two local package files it needs), so it
does not require pandas, scipy or the rest of the laptop environment.

What it does, in order:

1. **ball pass** -- runs the dedicated ball checkpoint on every sampled frame, projects every
   candidate through the per-frame homography, and writes a ball-only cache;
2. **players pass** -- runs the dedicated player checkpoint (player / goalkeeper / referee are all
   treated as people) on the same frames and writes a players-only cache;
3. **merge** -- combines the two caches into the normal ``perception.npz`` layout used by the
   dashboard, writes a ``.json`` sidecar, and prints the file to download.

The two stages are separate on purpose: the ball model and the player model are never loaded at
the same time, which keeps VRAM low and lets a preempted job resume after either stage.

Typical use on ``slurm-cm``::

    tar xzf mtah_farm_perceive.tar.gz -C ~/mtah_farm
    cd ~/mtah_farm
    sbatch farm_perceive.sbatch

For a quick smoke test over 500 frames::

    python farm_perceive.py --start 11185 --stop 11685 --step 5

Then download the result on the laptop::

    scp joan.ignasi.cid@slurm-cm:~/perception_runs/BAR-ATH_tactical_perception.npz \
        possession_lab/outputs/perception_new.npz
    scp joan.ignasi.cid@slurm-cm:~/perception_runs/BAR-ATH_tactical_perception.npz.json \
        possession_lab/outputs/perception_new.npz.json
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import sys
import time
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "possession_lab"
DEFAULT_DATASET = Path("/home/alumnes/joan.ignasi.cid/dataset/tactical")
DEFAULT_BALL_WEIGHTS = Path.home() / "ball_data/weights/ball_yolo26x_1024_clu.pt"
DEFAULT_PLAYERS_WEIGHTS = (
    Path.home() / "ball_data/player_data/weights/players_yolo26x_1024_clu.pt"
)
DEFAULT_OUTPUT = Path.home() / "perception_runs"


def log(message: str) -> None:
    print(f"[farm_perceive] {time.strftime('%H:%M:%S')} {message}", flush=True)


# ---------------------------------------------------------------------------
# Load possession_lab.perception without executing possession_lab/__init__.py
# ---------------------------------------------------------------------------
# The package __init__ imports pandas/matplotlib/evaluation.  The farm venv only needs the
# perception module and its constants, so load those two files as a tiny synthetic package.
def load_perception_module() -> Any:
    name = "_farm_lab.perception"
    if name in sys.modules:
        return sys.modules[name]
    if not (PACKAGE_DIR / "perception.py").is_file():
        raise SystemExit(f"missing {PACKAGE_DIR / 'perception.py'}; upload the farm bundle")
    package = types.ModuleType("_farm_lab")
    package.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
    sys.modules["_farm_lab"] = package
    for module_name in ("config", "perception"):
        path = PACKAGE_DIR / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(f"_farm_lab.{module_name}", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_farm_lab.{module_name}"] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


perception = load_perception_module()
VideoPerception = perception.VideoPerception
load_perception = perception.load_perception
merge_perception = perception.merge_perception
perceive_video = perception.perceive_video
save_perception = perception.save_perception
PITCH = perception.PITCH


# ---------------------------------------------------------------------------
# Lightweight asset readers (no pandas / scipy)
# ---------------------------------------------------------------------------
@dataclass
class TrackingLite:
    """The subset of ``possession_lab.data.TrackingData`` perception uses."""

    n_frames: int
    players: np.ndarray
    teams: np.ndarray
    ball: np.ndarray
    ball_present: np.ndarray


@dataclass
class HomographyLite:
    matrices: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.matrices.shape[0])


def load_homography_lite(path: str | Path) -> HomographyLite:
    """Read the challenge homography CSV without pandas.

    Columns: ``frame,h00..h22,reprojection_error,valid``.  Only the nine matrix values are kept.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"homography not found: {path}")
    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 10:
        raise ValueError(f"{path}: expected at least 10 columns, got {table.shape}")
    if not np.array_equal(table[:, 0].astype(np.int64), np.arange(len(table))):
        raise ValueError(f"{path}: frame column is not contiguous from zero")
    matrices = table[:, 1:10].reshape(-1, 3, 3)
    valid = table[:, 11].astype(bool) if table.shape[1] > 11 else np.ones(len(table), bool)
    matrices = np.where(valid[:, None, None], matrices, np.nan)
    return HomographyLite(matrices=matrices)


def _team_id_map(root: ET.Element) -> tuple[dict[str, int], list[str]]:
    ids = [team.get("id", "") for team in root.iter("Team")]
    mapping = {team_id: index for index, team_id in enumerate(ids)}
    return mapping, ids


def load_tracking_lite(
    raw_path: str | Path,
    meta_path: str | Path,
    *,
    n_frames: int,
    pitch: tuple[float, float] = PITCH,
) -> TrackingLite:
    """Parse the ATD raw text + XML metadata, matching ``possession_lab.data.load_tracking``."""
    raw_path = Path(raw_path)
    meta_path = Path(meta_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"ATD raw file not found: {raw_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"ATD metadata not found: {meta_path}")

    root = ET.parse(meta_path).getroot()
    team_map, _ = _team_id_map(root)
    xml_players = list(root.iter("Player"))
    teams = np.array(
        [team_map.get(player.get("teamId", ""), -1) for player in xml_players],
        dtype=np.int8,
    )

    with raw_path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    n_frames = int(n_frames)
    if row_count > n_frames:
        raise ValueError(f"{raw_path}: {row_count} rows exceeds n_frames={n_frames}")

    first_tokens: list[str] = []
    if not xml_players:
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    first_tokens = line.rstrip("\n").split(":")[1].split(";")
                    break
    n_players = len(xml_players) or len(first_tokens)
    if n_players <= 0:
        raise ValueError(f"{raw_path}: cannot determine player-slot count")
    if len(teams) != n_players:
        teams = np.full(n_players, -1, dtype=np.int8)

    length, width = float(pitch[0]), float(pitch[1])
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_present = np.zeros(n_frames, dtype=bool)
    players = np.full((n_frames, n_players, 2), np.nan, dtype=np.float32)

    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"{raw_path}: malformed ATD row (expected 3 colon fields): {line[:80]!r}"
                )
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                raise ValueError(f"{raw_path}: frame id {frame + 1} outside 0..{n_frames - 1}")

            ball_token = parts[2].strip()
            if ball_token and not ball_token.upper().startswith("N"):
                values = ball_token.split(",")
                if len(values) >= 2:
                    ball[frame, 0] = float(values[0]) * length
                    ball[frame, 1] = float(values[1]) * width
                    ball_present[frame] = True

            player_tokens = parts[1].split(";") if parts[1] else []
            if len(player_tokens) > n_players:
                raise ValueError(
                    f"{raw_path}: row {frame + 1} has {len(player_tokens)} player slots; "
                    f"expected <= {n_players}"
                )
            for slot, token in enumerate(player_tokens):
                token = token.strip()
                if not token or token.upper().startswith("N"):
                    continue
                values = token.split(",")
                if len(values) >= 2:
                    players[frame, slot, 0] = float(values[0]) * length
                    players[frame, slot, 1] = float(values[1]) * width

    return TrackingLite(
        n_frames=n_frames,
        players=players,
        teams=teams,
        ball=ball,
        ball_present=ball_present,
    )


# ---------------------------------------------------------------------------
# Match discovery
# ---------------------------------------------------------------------------
@dataclass
class Match:
    stem: str
    video: Path
    homography: Path
    atd_txt: Path | None
    atd_xml: Path | None

    def describe(self) -> str:
        extras = []
        extras.append("ATD" if self.atd_txt and self.atd_xml else "no ATD")
        return f"{self.stem} ({extras[0]})"


def _pick_videos(dataset_dir: Path) -> list[Path]:
    for pattern in ("*_tactical.mp4", "*.mp4"):
        videos = sorted(dataset_dir.rglob(pattern))
        if videos:
            return videos
    return []


def discover_matches(dataset_dir: Path, match_filter: str) -> list[Match]:
    videos = _pick_videos(dataset_dir)
    if not videos:
        raise SystemExit(f"no .mp4 files under {dataset_dir}")
    matches: list[Match] = []
    for video in videos:
        if match_filter.lower() != "all" and not any(
            fnmatch.fnmatch(video.name.lower(), f"*{part.strip().lower()}*")
            for part in match_filter.split(",")
        ):
            continue
        stem = video.stem
        homography = video.parent / f"{stem}_homography.csv"
        atd_txt = video.parent / f"{stem}_atd.txt"
        atd_xml = video.parent / f"{stem}_atd.xml"
        matches.append(
            Match(
                stem=stem,
                video=video,
                homography=homography,
                atd_txt=atd_txt if atd_txt.is_file() else None,
                atd_xml=atd_xml if atd_xml.is_file() else None,
            )
        )
    if not matches:
        names = ", ".join(video.name for video in videos)
        raise SystemExit(f"no video matched {match_filter!r}; available: {names}")
    return matches


# ---------------------------------------------------------------------------
# One match
# ---------------------------------------------------------------------------
def _stage_paths(output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"{stem}_ball.npz",
        output_dir / f"{stem}_players.npz",
        output_dir / f"{stem}_perception.npz",
    )


def _load_cache(path: Path) -> VideoPerception | None:
    try:
        return load_perception(path)
    except Exception as exc:
        log(f"cannot read {path}: {type(exc).__name__}: {exc}; rebuilding")
        return None


def _cache_covers(
    cached: VideoPerception,
    args: argparse.Namespace,
    homography: HomographyLite,
    stage: str | None = None,
) -> bool:
    """Whether an existing cache was built for this exact frame range, step and stage."""
    expected_stop = args.stop if args.stop is not None else homography.n_frames
    cached_stage = cached.meta.get("stage")
    return (
        int(cached.start) == int(args.start)
        and int(cached.stop) == int(expected_stop)
        and int(cached.step) == int(args.step)
        and (stage is None or cached_stage is None or str(cached_stage) == stage)
    )


def _run_stage(
    *,
    stage: str,
    match: Match,
    output_dir: Path,
    tracking: TrackingLite | None,
    homography: HomographyLite,
    args: argparse.Namespace,
) -> VideoPerception:
    ball_path, players_path, _ = _stage_paths(output_dir, match.stem)
    stage_path = ball_path if stage == "ball" else players_path
    if args.resume and stage_path.is_file():
        cached = _load_cache(stage_path)
        if cached is not None and _cache_covers(cached, args, homography, stage=stage):
            log(f"{match.stem}: {stage} cache exists, reusing {stage_path.name}")
            return cached
        if cached is not None:
            log(
                f"{match.stem}: ignoring {stage_path.name} (it covers "
                f"{cached.start}..{cached.stop} step {cached.step})"
            )
    log(f"{match.stem}: {stage} pass starts ({match.video.name})")
    started = time.time()
    result = perceive_video(
        match.video,
        homography,
        tracking,
        start=args.start,
        stop=args.stop,
        step=args.step,
        model=args.model,
        players_model=str(args.players_weights),
        ball_model=str(args.ball_weights),
        players_imgsz=args.imgsz,
        ball_imgsz=args.imgsz,
        device=args.device,
        imgsz=args.imgsz,
        player_conf=args.player_conf,
        ball_conf=args.ball_conf,
        stage=stage,
        progress=True,
    )
    result.meta.update(
        {
            "video": str(match.video),
            "homography": str(match.homography),
            "atd": str(match.atd_txt) if match.atd_txt else None,
        }
    )
    saved = save_perception(stage_path, result)
    elapsed = time.time() - started
    log(
        f"{match.stem}: {stage} pass wrote {saved.name} "
        f"({result.samples:,} samples in {elapsed / 60:.1f} min)"
    )
    return result


def run_match(
    match: Match,
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    if not match.video.is_file():
        raise FileNotFoundError(f"video not found: {match.video}")
    if not match.homography.is_file():
        raise FileNotFoundError(
            f"homography not found: {match.homography}\n"
            "Copy the matching *_tactical_homography.csv next to the video, or pass --dataset-dir."
        )

    log(f"{match.stem}: loading homography {match.homography.name}")
    homography = load_homography_lite(match.homography)
    tracking = None
    if match.atd_txt is not None and match.atd_xml is not None:
        log(f"{match.stem}: loading ATD {match.atd_txt.name}")
        tracking = load_tracking_lite(
            match.atd_txt,
            match.atd_xml,
            n_frames=homography.n_frames,
            pitch=PITCH,
        )
    else:
        log(f"{match.stem}: no ATD pair found; teams and the ATD ball anchor will be unavailable")

    _ball_path, _, final_path = _stage_paths(output_dir, match.stem)
    if args.resume and final_path.is_file():
        cached = _load_cache(final_path)
        if cached is not None and _cache_covers(cached, args, homography, stage=args.stage):
            log(f"{match.stem}: final cache exists, skipping (delete it to re-run): {final_path}")
            return final_path
        if cached is not None:
            log(
                f"{match.stem}: final cache covers {cached.start}..{cached.stop} step {cached.step}; "
                "rebuilding"
            )

    ball: VideoPerception | None = None
    players: VideoPerception | None = None
    if args.stage in {"both", "ball"}:
        ball = _run_stage(
            stage="ball",
            match=match,
            output_dir=output_dir,
            tracking=tracking,
            homography=homography,
            args=args,
        )
    if args.stage in {"both", "players"}:
        players = _run_stage(
            stage="players",
            match=match,
            output_dir=output_dir,
            tracking=tracking,
            homography=homography,
            args=args,
        )

    if args.stage == "both":
        assert ball is not None and players is not None
        log(f"{match.stem}: merging ball + players caches")
        merged = merge_perception(players, ball)
        merged.meta.update(
            {
                "video": str(match.video),
                "homography": str(match.homography),
                "atd": str(match.atd_txt) if match.atd_txt else None,
                "farm": {
                    "start": args.start,
                    "stop": args.stop,
                    "step": args.step,
                    "imgsz": args.imgsz,
                },
            }
        )
        saved = save_perception(final_path, merged)
    else:
        stage_result = ball if args.stage == "ball" else players
        assert stage_result is not None
        saved = save_perception(final_path, stage_result)

    summary_path = output_dir / f"{match.stem}_run.json"
    summary_path.write_text(
        json.dumps(
            {
                "match": match.stem,
                "video": str(match.video),
                "homography": str(match.homography),
                "atd": str(match.atd_txt) if match.atd_txt else None,
                "perception": str(saved),
                "perception_json": str(saved) + ".json",
                "stage": args.stage,
                "start": args.start,
                "stop": args.stop,
                "step": args.step,
                "imgsz": args.imgsz,
                "ball_weights": str(args.ball_weights),
                "players_weights": str(args.players_weights),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"{match.stem}: final cache {saved}")
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the dedicated ball and player models sequentially and build a "
        "dashboard-compatible perception cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--match",
        default="BAR-ATH",
        help="substring(s) of the video name to process, comma separated, or 'all'",
    )
    parser.add_argument("--video", type=Path, default=None, help="explicit video; overrides --match")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ball-weights", type=Path, default=DEFAULT_BALL_WEIGHTS)
    parser.add_argument("--players-weights", type=Path, default=DEFAULT_PLAYERS_WEIGHTS)
    parser.add_argument(
        "--stage",
        choices=["both", "ball", "players"],
        default="both",
        help="run both passes or only one",
    )
    parser.add_argument("--start", type=int, default=0, help="first video frame")
    parser.add_argument("--stop", type=int, default=None, help="exclusive last video frame")
    parser.add_argument("--step", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--imgsz", type=int, default=1024, help="inference size for both models")
    parser.add_argument("--model", default="yolo11x.pt", help="fallback COCO model when a weight is omitted")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--player-conf", type=float, default=0.12)
    parser.add_argument("--ball-conf", type=float, default=0.05)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse existing stage/final caches",
    )
    parser.add_argument("--list", action="store_true", help="list discovered matches and exit")
    return parser


def _check_weights(path: Path, label: str, stage: str) -> None:
    if stage in {"both", "ball"} and label == "ball" and not path.is_file():
        raise SystemExit(f"ball weights not found: {path}")
    if stage in {"both", "players"} and label == "players" and not path.is_file():
        raise SystemExit(f"player weights not found: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dataset_dir = args.dataset_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.ball_weights = args.ball_weights.expanduser()
    args.players_weights = args.players_weights.expanduser()
    if args.step < 1:
        raise SystemExit("--step must be >= 1")
    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    if args.imgsz < 320:
        raise SystemExit("--imgsz must be >= 320")

    if args.video is not None:
        video = args.video.expanduser()
        if not video.is_file():
            raise SystemExit(f"video not found: {video}")
        stem = video.stem
        parent = video.parent
        matches = [
            Match(
                stem=stem,
                video=video,
                homography=parent / f"{stem}_homography.csv",
                atd_txt=parent / f"{stem}_atd.txt" if (parent / f"{stem}_atd.txt").is_file() else None,
                atd_xml=parent / f"{stem}_atd.xml" if (parent / f"{stem}_atd.xml").is_file() else None,
            )
        ]
    else:
        matches = discover_matches(args.dataset_dir, args.match)

    if args.list:
        for match in matches:
            print(
                f"{match.stem}\n"
                f"  video:      {match.video}\n"
                f"  homography: {match.homography}"
                + ("" if match.homography.is_file() else "  [MISSING]")
                + f"\n  atd:        {match.atd_txt or 'none'}"
                + ("" if match.atd_txt else "  [MISSING]")
            )
        return 0

    _check_weights(args.ball_weights, "ball", args.stage)
    _check_weights(args.players_weights, "players", args.stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"dataset: {args.dataset_dir}")
    log(f"matches: {', '.join(match.stem for match in matches)}")
    log(f"ball weights: {args.ball_weights}")
    log(f"player weights: {args.players_weights}")
    log("stage order: ball -> players -> merge" if args.stage == "both" else f"stage: {args.stage}")
    log(f"frames: {args.start}..{args.stop if args.stop is not None else 'end'} step {args.step}")
    log(f"imgsz: {args.imgsz}, device: {args.device}")
    log(f"output: {args.output_dir}")

    outputs: list[Path] = []
    failures: list[tuple[str, str]] = []
    for match in matches:
        try:
            outputs.append(
                run_match(
                    match,
                    output_dir=args.output_dir,
                    args=args,
                )
            )
        except Exception as exc:  # keep a multi-match run alive
            failures.append((match.stem, f"{type(exc).__name__}: {exc}"))
            log(f"{match.stem}: FAILED: {type(exc).__name__}: {exc}")

    if failures:
        log("failures:")
        for stem, message in failures:
            log(f"  {stem}: {message}")
    if not outputs:
        return 1
    print("\nDone. Download these files:")
    for output in outputs:
        print(f"  {output}")
        print(f"  {output}.json")
    print("\nOn the laptop:")
    print(
        "  scp joan.ignasi.cid@slurm-cm:"
        f"{outputs[-1]} possession_lab/outputs/perception_new.npz"
    )
    print(
        "  scp joan.ignasi.cid@slurm-cm:"
        f"{outputs[-1]}.json possession_lab/outputs/perception_new.npz.json"
    )
    return 1 if failures and len(failures) == len(matches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
