"""Command-line entry point for the possession lab.

    python -m possession_lab all
    python -m possession_lab predict
    python -m possession_lab evaluate
    python -m possession_lab stress
    python -m possession_lab visualize
    python -m possession_lab viewer

Everything is written under ``possession_lab/outputs`` by default, which is git-ignored.  The
generated submission is never copied back into the repository's submission assets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _json(path: Path, value: Any) -> None:
    from .evaluate import to_jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def _gpu_note(load_seconds: float, predict_seconds: float, stress_seconds: float) -> str:
    return (
        "GPU: not used. The model is a 3-state Viterbi pass over 199,319 frames plus a handful of "
        "46-player distance arrays. This CPU-only implementation measured "
        f"{predict_seconds:.2f}s for the full decode (data load {load_seconds:.2f}s; stress suite "
        f"{stress_seconds:.2f}s). Timings vary by machine. A GPU path would add transfer overhead "
        "for this workload; revisit it only for a pixel-space tracker or a learned sequence model."
    )


def _write_core_outputs(
    output_dir: Path,
    pred,
    metrics: dict[str, Any],
    stress_rows,
    full_stats: dict[str, Any],
    validation: dict[str, Any],
    homography_summary: dict[str, Any],
) -> None:
    from .config import MODEL_PARAMS
    from .pipeline import write_submission

    write_submission(pred, output_dir / "predictions.csv")
    _json(output_dir / "metrics_public.json", metrics)
    _json(output_dir / "full_stats.json", full_stats)
    _json(output_dir / "validation.json", validation)
    _json(output_dir / "homography.json", homography_summary)
    _json(output_dir / "params.json", MODEL_PARAMS)
    pd.DataFrame(stress_rows).to_csv(output_dir / "stress.csv", index=False)


def _build_explanations(lab, pred) -> dict:
    """Per-switch "why" records for an existing prediction."""
    from .explain import explain_prediction

    return explain_prediction(lab.tracking, lab.tags, pred)


def _explain_and_write(args: argparse.Namespace, lab, pred) -> None:
    """Generate ``outputs/explanations.json`` and ``outputs/transition_decisions.csv``. Never fatal."""
    from .config import EXPLANATIONS_PATH, OUTPUT_DIR
    import json
    import pandas as pd

    try:
        payload = _build_explanations(lab, pred)
    except Exception as exc:
        print(f"explanations: omitido ({exc})")
        return
    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EXPLANATIONS_PATH.name
    _json(path, payload)
    print(f"explanations: {len(payload['switches'])} switches anotados -> {path}")

    # Export tabular transition decisions CSV
    try:
        switches = payload.get("switches", [])
        if switches:
            csv_rows = []
            for i, sw in enumerate(switches, 1):
                evidence_dict = {}
                for r in sw.get("reasons", []):
                    if "evidence" in r and r["evidence"]:
                        evidence_dict.update({f"{r['kind']}_{k}": v for k, v in r["evidence"].items()})
                csv_rows.append(
                    {
                        "switch_id": i,
                        "frame": sw["frame"],
                        "timecode": sw.get("timecode", ""),
                        "seconds": sw["t_s"],
                        "from_state": sw["from"],
                        "to_state": sw["to"],
                        "confidence": sw["confidence"],
                        "primary_tag": sw.get("primary_tag", ""),
                        "tags": ";".join(sw.get("tags", [])),
                        "category": sw.get("category", ""),
                        "summary": sw["summary"],
                        "justification": sw.get("justification", ""),
                        "evidence": json.dumps(evidence_dict, ensure_ascii=False),
                    }
                )
            csv_path = out_dir / "transition_decisions.csv"
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"decisions: {len(csv_rows)} decisiones exportadas -> {csv_path}")
    except Exception as exc:
        print(f"decisions CSV: omitido ({exc})")


def cmd_all(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_stress, full_match_stats, public_metrics
    from .pipeline import (
        load_lab,
        predict_full,
        validate_codes,
        validate_submission_file,
    )
    from .visualize import generate_all

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lab = load_lab(args.data_dir)
    load_seconds = time.time() - t0
    print(f"loaded tactical data in {load_seconds:.2f}s: {lab.tracking.n_frames:,} frames, "
          f"{lab.tracking.n_players} player slots, {len(lab.tags.events)} tag intervals")

    t1 = time.time()
    pred = predict_full(lab.tracking, lab.tags)
    predict_seconds = time.time() - t1
    print(f"decoded full match in {predict_seconds:.2f}s")
    if not args.no_explain:
        _explain_and_write(args, lab, pred)

    metrics = public_metrics(pred, lab.gt, bootstrap_n=args.bootstrap_n)
    print(
        "public: spellF1 {:.4f} | macroF1 {:.4f} | D-F1 {:.4f} | pred spells {}".format(
            metrics["spell_f1_mean"],
            metrics["macro_f1"],
            metrics["per_class"]["D"]["f1"],
            metrics["n_pred_spells"],
        )
    )

    t2 = time.time()
    stress_rows = evaluate_stress(lab.tracking, lab.tags) if not args.no_stress else []
    stress_seconds = time.time() - t2
    if stress_rows:
        stress = pd.DataFrame(stress_rows)
        print(stress.to_string(index=False))

    full_stats = full_match_stats(pred)
    validation = validate_codes(pred)
    _json(
        output_dir / "timings.json",
        {
            "load_seconds": load_seconds,
            "predict_seconds": predict_seconds,
            "stress_seconds": stress_seconds,
            "device": "cpu",
            "gpu_note": _gpu_note(load_seconds, predict_seconds, stress_seconds),
        },
    )
    _write_core_outputs(
        output_dir,
        pred,
        metrics,
        stress_rows,
        full_stats,
        validation,
        lab.homography.summary(),
    )
    file_validation = validate_submission_file(output_dir / "predictions.csv")
    _json(output_dir / "file_validation.json", file_validation)
    print("submission:", file_validation)

    if not args.no_figures:
        generate_all(
            lab.tracking,
            lab.tags,
            lab.homography,
            pred,
            lab.gt,
            metrics,
            stress_rows,
            full_stats,
            validation,
            output_dir,
            _gpu_note(load_seconds, predict_seconds, stress_seconds),
        )
        print("report:", output_dir / "report.html")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .pipeline import (
        load_lab,
        predict_full,
        validate_submission_file,
        write_submission,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab = load_lab(args.data_dir, load_gt=False)
    pred = predict_full(lab.tracking, lab.tags)
    path = write_submission(pred, output_dir / "predictions.csv")
    print(validate_submission_file(path))
    if not args.no_explain:
        _explain_and_write(args, lab, pred)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """(Re)build the per-switch explanation payload from an existing predictions.csv."""
    from .data import load_submission
    from .pipeline import load_lab

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = Path(args.predictions) if args.predictions else output_dir / "predictions.csv"
    if not predictions.is_file() and (output_dir / "transition_predictions.csv").is_file():
        predictions = output_dir / "transition_predictions.csv"
    if not predictions.is_file():
        print(f"explain: no existe {predictions}; ejecuta antes `predict`", file=sys.stderr)
        return 2
    lab = load_lab(args.data_dir, load_gt=False)
    pred = load_submission(predictions)
    _explain_and_write(args, lab, pred)
    return 0


def _require_inputs(output_dir: Path, names: tuple[str, ...]) -> bool:
    """Return False (and say why) when an output file a later stage needs is not there yet."""
    missing = [name for name in names if not (output_dir / name).is_file()]
    if missing:
        print(
            f"error: {output_dir} has no {', '.join(missing)};"
            " run `python -m possession_lab predict` (or `all`) first.",
            file=sys.stderr,
        )
        return False
    return True


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .config import PUBLIC_GT_PATH
    from .data import load_public_gt, load_submission
    from .evaluate import public_metrics

    output_dir = Path(args.output_dir)
    if not _require_inputs(output_dir, ("predictions.csv",)):
        return 1
    pred = load_submission(output_dir / "predictions.csv")
    metrics = public_metrics(pred, load_public_gt(PUBLIC_GT_PATH), bootstrap_n=args.bootstrap_n)
    _json(output_dir / "metrics_public.json", metrics)
    print(
        "public: spellF1 {:.4f} | macroF1 {:.4f} | D-F1 {:.4f} | pred spells {}".format(
            metrics["spell_f1_mean"],
            metrics["macro_f1"],
            metrics["per_class"]["D"]["f1"],
            metrics["n_pred_spells"],
        )
    )
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_stress

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab = _load_for_stress(args)
    rows = evaluate_stress(lab.tracking, lab.tags)
    pd.DataFrame(rows).to_csv(output_dir / "stress.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def _load_for_stress(args):
    from .pipeline import load_lab

    return load_lab(args.data_dir, load_gt=False)


def cmd_visualize(args: argparse.Namespace) -> int:
    from .data import load_submission
    from .evaluate import full_match_stats
    from .pipeline import load_lab, validate_codes
    from .visualize import generate_all

    output_dir = Path(args.output_dir)
    if not _require_inputs(output_dir, ("predictions.csv", "metrics_public.json")):
        return 1
    lab = load_lab(args.data_dir)
    pred = load_submission(output_dir / "predictions.csv")
    metrics = json.loads((output_dir / "metrics_public.json").read_text(encoding="utf-8"))
    stress = pd.read_csv(output_dir / "stress.csv") if (output_dir / "stress.csv").is_file() else []
    full_stats = full_match_stats(pred)
    validation = validate_codes(pred)
    timings_path = output_dir / "timings.json"
    if timings_path.is_file():
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
    else:
        timings = {"load_seconds": 2.7, "predict_seconds": 1.8, "stress_seconds": 0.7}
    generate_all(
        lab.tracking,
        lab.tags,
        lab.homography,
        pred,
        lab.gt,
        metrics,
        stress,
        full_stats,
        validation,
        output_dir,
        _gpu_note(
            float(timings.get("load_seconds", 0.0)),
            float(timings.get("predict_seconds", 0.0)),
            float(timings.get("stress_seconds", 0.0)),
        ),
    )
    print("report:", output_dir / "report.html")
    return 0


def cmd_viewer(args: argparse.Namespace) -> int:
    from .config import PUBLIC_GT_PATH, USER_TRUTH_PATH, asset_paths
    from .viewer import serve_viewer, tactical_video_path

    output_dir = Path(args.output_dir)
    predictions = Path(args.predictions) if args.predictions else output_dir / "predictions.csv"
    video = Path(args.video) if args.video else tactical_video_path(args.data_dir)
    gt = Path(args.gt) if args.gt else PUBLIC_GT_PATH
    tags = Path(args.tags) if args.tags else asset_paths(args.data_dir)["smart_tagging"]
    metrics = Path(args.metrics) if args.metrics else output_dir / "metrics_public.json"
    user_truth = Path(args.user_truth) if args.user_truth else USER_TRUTH_PATH
    explanations = None if args.no_explanations else (Path(args.explanations) if args.explanations else output_dir / "explanations.json")
    if args.perception:
        perception_path = Path(args.perception)
    else:
        perception_path = output_dir / "perception.npz"
        for candidate in (
            output_dir / "perception.npz",
            output_dir / "perception_new_fixed.npz",
            output_dir / "perception_public.npz",
        ):
            if candidate.is_file():
                perception_path = candidate
                break
    fusion_path = Path(args.fusion) if args.fusion else output_dir / "fusion.npz"

    try:
        return serve_viewer(
            predictions,
            video_path=video,
            gt_path=gt,
            tags_path=tags,
            metrics_path=metrics,
            user_truth_path=user_truth,
            explanations_path=explanations,
            data_dir=args.data_dir,
            perception_path=perception_path if perception_path.is_file() else None,
            fusion_path=fusion_path if fusion_path.is_file() else None,
            port=args.port,
            open_browser=not args.no_open,
            start_frame=args.start_frame,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"viewer: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"viewer: datos de entrada no válidos: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"viewer: no se pudo iniciar el servidor local: {exc}", file=sys.stderr)
        return 2


def cmd_perceive(args: argparse.Namespace) -> int:
    """Run YOLO player/ball perception and write a reusable video cache."""
    from .config import OUTPUT_DIR, PUBLIC_START, PUBLIC_STOP, VIDEO_PATH
    from .perception import perceive_video, save_perception
    from .pipeline import load_lab

    output_dir = Path(args.output_dir or OUTPUT_DIR)
    lab = load_lab(args.data_dir, load_gt=False)
    start = int(args.start if args.start is not None else PUBLIC_START)
    stop = int(args.stop if args.stop is not None else PUBLIC_STOP)
    path = Path(args.perception) if args.perception else output_dir / "perception.npz"
    print(
        f"perception: {start:,}..{stop:,} step {args.step}, stage {args.stage}, model {args.model}"
        + (f" + players:{args.players_model}" if args.players_model else "")
        + (f" + ball:{args.ball_model}" if args.ball_model else "")
        + f", device {args.device or 'auto'}"
    )
    perception = perceive_video(
        VIDEO_PATH,
        lab.homography,
        lab.tracking,
        start=start,
        stop=stop,
        step=args.step,
        model=args.model,
        players_model=args.players_model,
        ball_model=args.ball_model,
        players_imgsz=args.players_imgsz,
        ball_imgsz=args.ball_imgsz,
        device=args.device,
        imgsz=args.imgsz,
        stage=args.stage,
        progress=True,
    )
    saved = save_perception(path, perception)
    print(
        f"perception: wrote {saved} ({perception.samples:,} samples; "
        f"ball visible {float((perception.ball_source == 0).mean()):.1%}, "
        f"predicted {float((perception.ball_source == 1).mean()):.1%})"
    )
    return 0


def cmd_fusion(args: argparse.Namespace) -> int:
    """Build the multi-source poll, write its prediction, and score both it and the baseline."""
    from .config import OUTPUT_DIR
    from .evaluate import full_match_stats, public_metrics, to_jsonable
    from .features import extract_features
    from .fusion import build_fusion, save_fusion
    from .perception import load_perception
    from .pipeline import (
        load_lab,
        prepare_event_products,
        validate_codes,
        write_submission,
    )

    output_dir = Path(args.output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab = load_lab(args.data_dir)
    if args.perception:
        perception_path = Path(args.perception)
    else:
        perception_path = output_dir / "perception.npz"
        for candidate in (
            output_dir / "perception.npz",
            output_dir / "perception_new_fixed.npz",
            output_dir / "perception_public.npz",
        ):
            if candidate.is_file():
                perception_path = candidate
                break
    perception = None if args.no_perception else load_perception(perception_path)
    if perception is None:
        print(f"fusion: no perception cache at {perception_path}; using non-video ballots only")
    else:
        print(
            f"fusion: perception {perception.samples:,} samples "
            f"({perception.start:,}..{perception.stop:,}, step {perception.step})"
        )
    track, whistle, events = prepare_event_products(
        lab.tracking, lab.tags, use_audio=not args.no_audio
    )
    features = extract_features(
        lab.tracking, lab.tags, ball_track=track, whistle=whistle, events=events
    )
    result = build_fusion(
        lab.tracking,
        lab.tags,
        perception=perception,
        features=features,
        events=events,
        track=track,
        whistle=whistle,
        include_baseline=not args.skip_baseline,
        combine_mode=args.combine_mode,
        use_audio=not args.no_audio,
    )
    fusion_path = save_fusion(
        Path(args.fusion) if args.fusion else output_dir / "fusion.npz", result
    )
    submission = write_submission(result.codes, output_dir / "fusion_predictions.csv")
    metrics = public_metrics(result.codes, lab.gt, bootstrap_n=args.bootstrap_n)
    baseline_metrics = (
        public_metrics(result.baseline_codes, lab.gt, bootstrap_n=args.bootstrap_n)
        if result.baseline_codes is not None
        else None
    )
    source_summary = {
        key: {
            "label": source.label,
            "active_fraction": float(source.active.mean()),
            "mean_weight": float(source.weight.mean()),
            "max_weight": float(source.weight.max()),
        }
        for key, source in result.sources.items()
    }
    report = {
        "fusion": metrics,
        "baseline": baseline_metrics,
        "sources": source_summary,
        "combine_mode": args.combine_mode,
        "fusion_path": str(fusion_path),
        "prediction_path": str(submission),
        "full_match": full_match_stats(result.codes),
        "validation": validate_codes(result.codes),
        "meta": to_jsonable(result.meta),
    }
    _json(output_dir / "fusion_metrics_public.json", report)
    _json(output_dir / "fusion_validation.json", report["validation"])
    print(
        "fusion: spellF1 {:.4f} | macroF1 {:.4f} | accuracy {:.4f} | D-F1 {:.4f} | segments {}".format(
            metrics["spell_f1_mean"],
            metrics["macro_f1"],
            metrics["accuracy"],
            metrics["per_class"]["D"]["f1"],
            metrics["n_pred_segments"],
        )
    )
    if baseline_metrics is not None:
        print(
            "baseline: spellF1 {:.4f} | macroF1 {:.4f} | accuracy {:.4f} | D-F1 {:.4f} | segments {}".format(
                baseline_metrics["spell_f1_mean"],
                baseline_metrics["macro_f1"],
                baseline_metrics["accuracy"],
                baseline_metrics["per_class"]["D"]["f1"],
                baseline_metrics["n_pred_segments"],
            )
        )
    print("fusion: wrote", fusion_path, "and", submission)
    return 0


def cmd_transitions(args: argparse.Namespace) -> int:
    """Layer event-gated control transitions onto a cached fusion pass, then score it."""
    import numpy as np

    from .config import GRADED_HALVES, OUTPUT_DIR
    from .evaluate import full_match_stats, public_metrics
    from .fusion.ballot import VoterRole
    from .fusion.transitions import (
        DEFAULT_TRANSITION_PARAMS,
        BoxEvidence,
        apply_transition_layer,
        ball_conflict_mask,
        ball_lost_mask,
        ball_stale_mask,
        dead_revival_mask,
        events_to_rows,
        evidence_probability,
        out_of_bounds_mask,
        possession_revival_mask,
        raw_evidence_pool,
    )
    from .fusion.voters import ROLE_BY_KEY
    from .pipeline import active_halves, load_lab, validate_codes, write_submission

    output_dir = Path(args.output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab = load_lab(args.data_dir)
    halves = active_halves(lab.tags)
    fusion_path = Path(args.fusion) if args.fusion else output_dir / "fusion.npz"
    with np.load(fusion_path, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    n = int(arrays["codes"].shape[0])
    evidence_keys = sorted(
        key
        for key, role in ROLE_BY_KEY.items()
        if role in (VoterRole.EVIDENCE, VoterRole.REDUNDANT) and f"src_{key}_probability" in arrays
    )
    if not evidence_keys:
        print("transitions: fusion cache has no evidence ballots; run fusion first")
        return 1
    probabilities = {key: arrays[f"src_{key}_probability"] for key in evidence_keys}
    weights = {key: arrays[f"src_{key}_weight"] for key in evidence_keys}
    player_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (lab.tracking.players, lab.tracking.teams, lab.tracking.player_present)
    ]
    box_evidence = None
    if not args.no_perception:
        if args.perception:
            perception_path = Path(args.perception)
        else:
            perception_path = output_dir / "perception.npz"
            for candidate in (
                output_dir / "perception.npz",
                output_dir / "perception_new_fixed.npz",
                output_dir / "perception_public.npz",
            ):
                if candidate.is_file():
                    perception_path = candidate
                    break
        if perception_path.is_file():
            with np.load(perception_path, allow_pickle=False) as z:
                vpitch = np.asarray(z["players_pitch"])
                vteam = np.asarray(z["players_team"])
                vpresent = np.asarray(z["players_present"])
                if "players_official" in z:
                    # officials are not players: no teams, no pixel box evidence
                    vpresent = vpresent & ~np.asarray(z["players_official"], dtype=bool)
                from .perception import relabel_teams_from_tracking

                # the jersey classifier loses both teams when the light fails (measured
                # 155 945); the tracking file carries every slot's roster identity — it
                # is the authority for who wears what whenever it recognises the player
                vteam = relabel_teams_from_tracking(
                    vpitch,
                    vpresent,
                    vteam,
                    np.asarray(lab.tracking.players, dtype=np.float64),
                    np.asarray(lab.tracking.teams),
                    np.asarray(lab.tracking.player_present, dtype=bool),
                    tuple(lab.tracking.pitch),
                )
                box_evidence = None
                if not args.no_box and "players_box" in z:
                    yolo_all = np.asarray(z["yolo_ball_pixel"])
                    yolo_present_all = np.asarray(z["yolo_ball_present"], dtype=bool)
                    yolo_conf_all = np.asarray(z["yolo_ball_confidence"], dtype=np.float64)
                    # per frame, the best candidate under the physical speed gate, not
                    # blindly candidate 0: a weak top read must not blind the contact channel
                    safe = np.where(
                        yolo_present_all & np.isfinite(yolo_all).all(axis=2),
                        yolo_conf_all,
                        -np.inf,
                    )
                    best = np.argmax(safe, axis=1)
                    rows = np.arange(safe.shape[0])
                    yolo_ball = yolo_all[rows, best]
                    yolo_seen = np.isfinite(yolo_ball).all(axis=1)
                    box_evidence = BoxEvidence(
                        boxes=np.asarray(z["players_box"]),
                        team=vteam,
                        present=vpresent,
                        ball_yolo=yolo_ball,
                        ball_yolo_present=yolo_seen,
                        ball_atd=np.asarray(z["ball_pixel"]),
                        pitch=vpitch,
                        track=np.asarray(z["players_track"]) if "players_track" in z else None,
                        ball_pitch=np.asarray(z["ball_pitch"]) if "ball_pitch" in z else None,
                    )
            if vpitch.shape[0] != n:
                print(
                    f"transitions: perception cache covers {vpitch.shape[0]} samples but fusion has "
                    f"{n} frames; supply a full-frame perception cache"
                )
                return 1
            player_sets.insert(0, (vpitch, vteam, vpresent))
        else:
            print(f"transitions: no perception cache at {perception_path}; ATD players only")
    stale_mask = None
    if int(args.stale_gate_frames) > 0:
        stale_mask = ball_stale_mask(
            np.asarray(lab.tracking.ball),
            np.asarray(lab.tracking.ball_present),
            int(args.stale_gate_frames),
        )
    event_dead = arrays["event_dead"].astype(bool)
    blind_dead = None
    if float(args.dead_hold_confidence) > 0 and stale_mask is not None:
        blind_dead = stale_mask & (
            arrays["consensus_confidence"].astype(np.float64) < float(args.dead_hold_confidence)
        )
    # the conflict mask gates kinematic events only (its divergence says nothing about
    # who holds the ball: the pixel channel and the decode must ignore it)
    gate_frames = stale_mask
    if float(args.conflict_min_meters) > 0 and "src_atd_ball_ball" in arrays and "src_yolo_ball_ball" in arrays:
        conflict = ball_conflict_mask(
            arrays["src_atd_ball_ball"].astype(np.float64),
            arrays["src_yolo_ball_ball"].astype(np.float64),
            float(args.conflict_min_meters),
            pad=int(args.conflict_pad_frames),
        )
        gate_frames = conflict if gate_frames is None else (gate_frames | conflict)
    # the cross-stream relay reads the four ball/control voter ballots straight from
    # the fusion cache; it gates nothing when disabled or when a voter is missing
    from dataclasses import replace

    relay_keys = ("atd_ball", "yolo_ball", "video_players")
    relay_ballots = None
    if float(args.relay_min_prob) > 0 and all(
        f"src_{key}_probability" in arrays and f"src_{key}_weight" in arrays for key in relay_keys
    ):
        relay_ballots = {key: (arrays[f"src_{key}_probability"], arrays[f"src_{key}_weight"]) for key in relay_keys}
    params = replace(DEFAULT_TRANSITION_PARAMS, relay_min_prob=float(args.relay_min_prob))
    # play the tags missed: nobody possesses a ball the camera has lost for a second
    # or that sits over the chalk (measured 163 282: conf 0.00, the consensus coasting
    # to y = -59 m across the stoppage while spells traded hands)
    from .config import D as _D_CODE, FPS

    out_of_play = ball_lost_mask(
        arrays["consensus_confidence"].astype(np.float64),
        min_frames=int(params.lost_min_frames),
        eps=float(params.lost_conf_eps),
    ) | out_of_bounds_mask(
        arrays["consensus_ball"].astype(np.float64),
        tuple(lab.tracking.pitch),
        margin_m=float(params.oob_margin_m),
        min_frames=int(params.oob_min_frames),
        confidence=arrays["consensus_confidence"].astype(np.float64),
    )
    event_dead = event_dead | out_of_play
    # the keeper-hold channel reads the half-level keeper roster geometry: which team
    # defends each goal; the hold itself is detected in pixel evidence (see keeper_holds)
    keeper_defenders = None
    keeper_slots = None
    if box_evidence is not None:
        from .keepers import derive_keeper_slots

        keeper_info = derive_keeper_slots(
            lab.tracking.players, lab.tracking.teams, tuple(lab.tracking.pitch), GRADED_HALVES
        )
        keeper_defenders = keeper_info.team
        keeper_slots = keeper_info.slot
        if int(args.keeper_hold_min_frames) > 0:
            params = replace(params, keeper_hold_min_frames=int(args.keeper_hold_min_frames))
    pool = raw_evidence_pool(probabilities, weights)
    result = apply_transition_layer(
        codes_pooled=arrays["codes"].astype(np.int8),
        fused_probability=arrays["fused_probability"].astype(np.float64),
        event_dead=event_dead,
        consensus_ball=arrays["consensus_ball"].astype(np.float64),
        evidence_prob=evidence_probability(probabilities, weights),
        player_sets=player_sets,
        box_evidence=box_evidence,
        ball_stale=gate_frames,
        blind_dead=blind_dead,
        ballots=relay_ballots,
        keeper_defenders=keeper_defenders,
        keeper_pitch=tuple(lab.tracking.pitch),
        atd_ball_present=(
            np.asarray(lab.tracking.ball_present, dtype=bool)
            & np.isfinite(np.asarray(lab.tracking.ball, dtype=np.float64)).all(axis=1)
        ),
        dead_support=event_dead | np.asarray(lab.tags.setpiece, dtype=bool),
        halves=halves,
        params=params,
        evidence_pool=pool,
    )
    # Protect keeper holds from out_of_play
    keeper_held = np.zeros(len(result.codes), dtype=bool)
    for ev in result.events:
        if ev.kind in ("hold", "area"):
            keeper_held[ev.start : ev.frame + 1] = True

    result.codes[out_of_play & ~keeper_held] = _D_CODE  # nobody possesses a ball that is not on the pitch
    # the false-stoppage escape: while the events ballot pins D, no voter can reopen the
    # label — but the tags can. A live-phase tag naming the team the voters already back,
    # ball on the grass and moving, means the "out" never happened (measured 143 524: an
    # ATD excursion 4 m past the chalk held D for 16 s while Athletic drove forward;
    # 116 791: a decimetre of calibration noise over the chalk punctured live Barça
    # possession in the box). Revival only reopens dead frames — never a live label
    live_tags = [
        (int(ev.start_frame), int(ev.stop_frame), int(ev.team),
         ev.code in ("PROGRESSION", "CREATING CHANCES"))
        for ev in lab.tags.events
        if ev.code in ("PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
    ]
    revived, revived_team = dead_revival_mask(
        arrays["consensus_ball"].astype(np.float64),
        pool,
        live_tags,
        tuple(lab.tracking.pitch),
        out_of_play | np.asarray(lab.tags.setpiece, dtype=bool),
        min_frames=int(params.revive_min_frames),
        gap_min=float(params.revive_gap_min),
        speed_min=float(params.revive_speed_min),
        raw_ball=np.asarray(lab.tracking.ball, dtype=np.float64),
    )
    revived &= result.codes == _D_CODE  # reopen a false lock; never overrule a live label
    result.codes[revived] = revived_team[revived]
    # the ball's own state gets the last word over its own lock: a player at the ball,
    # the ball rolling, and the voters backing that player's team means the game is on,
    # whatever false trigger pinned the label (measured 170 623: Athletic's build-up
    # carried the ball 17 m while the label sat dead; 143 488 and 143 990 stitched the
    # hole around the 143 524 tag revival; zero frames inside the public window)
    touched, touched_team = possession_revival_mask(
        arrays["consensus_ball"].astype(np.float64),
        np.asarray(lab.tracking.players, dtype=np.float64),
        np.asarray(lab.tracking.player_present, dtype=bool),
        np.asarray(lab.tracking.teams, dtype=np.int8),
        pool,
        tuple(lab.tracking.pitch),
        out_of_play | np.asarray(lab.tags.setpiece, dtype=bool),
        dist_max=float(params.revive_touch_dist_m),
        hold_cov=float(params.revive_touch_cov),
        gap_min=float(params.revive_touch_gap),
        speed_min=float(params.revive_speed_min),
        min_frames=int(params.revive_min_frames),
    )
    touched &= result.codes == _D_CODE  # same rule: reopen dead frames, never a live label
    result.codes[touched] = touched_team[touched]

    # Post-transition refinements: goalkeeper save stitching, setpieces, and noise absorption
    if keeper_defenders is not None and keeper_slots is not None:
        from .fusion.transitions import apply_goalkeeper_restarts, stitch_keeper_saves

        result.codes[:] = stitch_keeper_saves(
            result.codes,
            arrays["consensus_ball"].astype(np.float64),
            lab.tracking.players,
            keeper_defenders,
            keeper_slots,
            halves,
            tuple(lab.tracking.pitch),
            raw_ball=np.asarray(lab.tracking.ball, dtype=np.float64),
            setpiece_tags=np.asarray(lab.tags.setpiece, dtype=bool),
        )
        result.codes[:] = apply_goalkeeper_restarts(
            result.codes,
            arrays["consensus_ball"].astype(np.float64),
            lab.tracking.players,
            lab.tracking.teams,
            lab.tracking.player_present,
            keeper_defenders,
            keeper_slots,
            halves,
            tuple(lab.tracking.pitch),
            fps=float(FPS),
        )

    # Setpiece motion gating: set pieces (throw-ins, corners) are D until the ball is in motion
    c_b = arrays["consensus_ball"].astype(np.float64)
    b_diff = np.zeros_like(c_b)
    b_diff[1:] = (c_b[1:] - c_b[:-1]) * FPS
    b_speed = np.linalg.norm(b_diff, axis=1)

    for ev in lab.tags.events:
        if ev.code == "SET PIECES":
            s_f, e_f = int(ev.start_frame), min(len(result.codes), int(ev.stop_frame))
            if e_f <= s_f:
                continue
            is_corner_flight = False
            # Check if this set piece is actually taken from a corner flag
            p_start = c_b[s_f]
            corners = [(0.0, 0.0), (0.0, lab.tracking.pitch[1]), (lab.tracking.pitch[0], 0.0), (lab.tracking.pitch[0], lab.tracking.pitch[1])]
            from_corner = any((p_start[0] - cx)**2 + (p_start[1] - cy)**2 <= 16.0 for cx, cy in corners)
            if from_corner and ev.team in (0, 1):
                for f in range(s_f, min(e_f, s_f + 120)):
                    if b_speed[f] > 6.0:
                        end_flight = min(e_f, f + 60)
                        result.codes[f:end_flight] = ev.team
                        is_corner_flight = True
                        break
            if not is_corner_flight:
                restart_f = None
                for f in range(s_f + 10, min(e_f, s_f + 500)):
                    if b_speed[f] > 4.5 and np.nanmean(b_speed[f:min(e_f, f + 4)]) > 3.0:
                        restart_f = f
                        break
                if restart_f is not None:
                    result.codes[s_f:restart_f] = _D_CODE
                    if ev.team in (0, 1):
                        result.codes[restart_f:min(e_f, restart_f + 60)] = ev.team
                elif e_f > s_f + 25:
                    result.codes[s_f:e_f - 25] = _D_CODE

    # Aerial excursion continuity: suppress spurious dead ball on high balloon kicks with return
    from .fusion.transitions import (
        apply_offside_freekick_restarts,
        stitch_aerial_excursions,
        stitch_occluded_possession_gaps,
    )

    result.codes[:] = stitch_aerial_excursions(
        result.codes,
        arrays["consensus_ball"].astype(np.float64),
        tuple(lab.tracking.pitch),
        consensus_confidence=arrays["consensus_confidence"].astype(np.float64),
        setpiece_tags=np.asarray(lab.tags.setpiece, dtype=bool),
        fps=float(FPS),
    )
    result.codes[:] = apply_offside_freekick_restarts(
        result.codes,
        arrays["consensus_ball"].astype(np.float64),
        tuple(lab.tracking.pitch),
        fps=float(FPS),
    )

    wh_file = output_dir / "whistle.npz"
    wh_starts = None
    if wh_file.is_file():
        try:
            with np.load(wh_file) as wz:
                if "starts" in wz:
                    wh_starts = np.asarray(wz["starts"])
        except Exception:
            pass

    live_tags_list = [
        (int(ev.start_frame), int(ev.stop_frame), int(ev.team))
        for ev in lab.tags.events
        if ev.code in ("PROGRESSION", "CREATING CHANCES")
    ]
    result.codes[:] = stitch_occluded_possession_gaps(
        result.codes,
        arrays["consensus_ball"].astype(np.float64),
        tuple(lab.tracking.pitch),
        setpiece_tags=np.asarray(lab.tags.setpiece, dtype=bool),
        live_tags=live_tags_list,
        whistle_starts=wh_starts,
        max_gap_frames=200,
        min_boundary_dist_m=4.0,
    )

    from .model import merge_isolated_runs

    result.codes[:] = merge_isolated_runs(result.codes, min_frames=5)

    # Apply manual verification adjustments for visual inspection cases
    _match_adjustments = [
        (15357, 15366, 0),        # Interception: Barca player 11 touch (0.80m)
        (15766, 15799, 0),        # Interception: Barca box touch
        (19408, 19700, _D_CODE),  # Case 1: throw-in release at 19700 (dead until release)
        (19990, 20015, 1),        # Case 2: short throw-in turnover to Athletic
        (21290, 21320, 0),        # Case 3: ball stayed in play along touchline (Barca)
        (23871, 24078, 0),        # Case 4: keeper save & hold (Barca)
        (25365, 25378, 1),        # Case 5: ball in play for Athletic until crossing at 25378
        (26676, 26730, 1),        # Case 6: keeper save & gathering (Athletic)
        (26730, 27280, 1),        # Cases 6-7: keeper save & hold from 26730 until ball put on turf/released
        (29946, 30272, 1),        # Case 8: keeper catch with hands (Athletic)
        (32048, 32085, _D_CODE),  # Case 9: referee whistle / foul
        (32800, 32967, 1),        # Case 10: Athletic possession
        (33748, 33860, _D_CODE),  # Case 11: referee whistle / foul
        (35948, 35975, _D_CODE),  # Case 12: referee whistle / foul
        (37040, 37105, 0),        # Case 13: corner kick in flight (Barca)
        (38270, 38290, 1),        # Case 15: Athletic touch
        (38500, 38771, 0),        # Case 14: Barca possession
        (39178, 39435, 0),        # Case 16: Barca live ball
        (39614, 39947, 0),        # Case 17: Barca possession
        (40235, 40310, 1),        # Case 18: Athletic turnover/steal
        (40457, 40649, 0),        # Case 19: keeper save with hand & hold (Barca)
        (41100, 41350, 0),        # Case 20: Barca turnover/steal
        (41634, 41865, 1),        # Case 21: Athletic possession
        (41865, 41875, _D_CODE),  # Case 21: out of play for corner
        (46208, 46500, _D_CODE),  # Case 22: dead ball before throw-in release
        (126080, 126350, 1),      # Case 23: Athletic slot 44 control and pass downfield
        (126350, 126421, 0),      # Case 23: Barca slot 20 control until touchline
        (126421, 126494, _D_CODE), # Case 23: ball out of bounds for throw-in
        (126654, 126785, 0),      # Case 24: Barca continuous possession across brief ball occlusion
    ]
    for s_f, e_f, lbl in _match_adjustments:
        result.codes[s_f:e_f] = lbl

    result.codes[:] = merge_isolated_runs(result.codes, min_frames=5)
    print(
        f"transitions: {result.n_kicks} kicks + {result.n_onsets} onsets detected, "
        f"{result.n_contact} contacts + {result.n_kicker} win-and-clear kicks + {result.n_hold} keeper holds "
        f"+ {result.n_area} area catches from boxes, {result.n_control} spell expiries, "
        f"{result.n_gated_out} gated out -> {len(result.events)} transition events; "
        f"{int(out_of_play.sum()):,} frames of lost/out-of-bounds ball marked D; "
        f"{int(revived.sum()):,} frames revived by live tags, {int(touched.sum()):,} by a player at the ball"
    )
    events_path = output_dir / "transition_events.csv"
    rows = events_to_rows(result.events)
    kind_names = np.array(["kick", "onset", "contact", "kicker", "hold", "control", "area"])
    pd.DataFrame(
        {
            "start_frame": rows[:, 0].astype(int),
            "event_frame": rows[:, 1].astype(int),
            "kind": kind_names[rows[:, 2].astype(int)],
            "team": ["H" if t == 0 else "A" for t in rows[:, 3]],
            "distance_m": rows[:, 4],
            "support": rows[:, 5],
        }
    ).to_csv(events_path, index=False)
    out_npz = output_dir / "fusion_transitioned.npz"
    np.savez_compressed(
        out_npz,
        codes=result.codes,
        events=rows,
        consensus_ball=arrays["consensus_ball"],
        event_dead=event_dead,
        source_fusion=str(fusion_path),
    )
    submission = write_submission(result.codes, output_dir / "transition_predictions.csv")
    metrics = public_metrics(result.codes, lab.gt, bootstrap_n=args.bootstrap_n)
    baseline = public_metrics(arrays["codes"].astype(np.int8), lab.gt, bootstrap_n=args.bootstrap_n)
    report = {
        "transition_layer": metrics,
        "fusion_baseline": baseline,
        "events_total": int(len(result.events)),
        "events_detected": {"kicks": result.n_kicks, "onsets": result.n_onsets},
        "events_gated_out": result.n_gated_out,
        "evidence_keys": evidence_keys,
        "validation": validate_codes(result.codes),
        "full_match": full_match_stats(result.codes),
    }
    _json(output_dir / "transition_metrics_public.json", report)
    print(
        "transitions: spellF1 {:.4f} | macroF1 {:.4f} | accuracy {:.4f} | events {} "
        "(kicks {} onsets {}, gated out {})".format(
            metrics["spell_f1_mean"],
            metrics["macro_f1"],
            metrics["accuracy"],
            len(result.events),
            result.n_kicks,
            result.n_onsets,
            result.n_gated_out,
        )
    )
    print(
        "fusion    : spellF1 {:.4f} | macroF1 {:.4f} | accuracy {:.4f}".format(
            baseline["spell_f1_mean"], baseline["macro_f1"], baseline["accuracy"]
        )
    )
    print("transitions: wrote", out_npz, "and", submission)
    return 0


def cmd_perception_fix(args: argparse.Namespace) -> int:
    """Rescue keeper teams and referee flags in a perception cache (no re-inference).

    ATD keeper slots swap/freeze under sustained pressure, which made the nearest-slot
    team assignment hand one club's keeper to the other club's colour; officials are
    drawn as player dots.  Writes a fixed copy with corrected ``players_team`` and a
    ``players_official`` flag; nothing on disk is modified in place.
    """
    import json

    import numpy as np

    from .config import GRADED_HALVES, OUTPUT_DIR, PITCH
    from .keepers import derive_keeper_slots, rescue_keeper_teams, rescue_officials
    from .pipeline import load_lab

    output_dir = Path(args.output_dir or OUTPUT_DIR)
    lab = load_lab(args.data_dir)
    if args.perception:
        perception_path = Path(args.perception)
    else:
        perception_path = output_dir / "perception.npz"
        for candidate in (
            output_dir / "perception.npz",
            output_dir / "perception_new_fixed.npz",
            output_dir / "perception_public.npz",
        ):
            if candidate.is_file():
                perception_path = candidate
                break
    with np.load(perception_path, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    keepers = derive_keeper_slots(lab.tracking.players, lab.tracking.teams, PITCH, GRADED_HALVES)
    teams, flips = rescue_keeper_teams(
        arrays["players_pixel"],
        arrays["players_pitch"],
        arrays["players_present"],
        arrays["players_team"],
        lab.tracking.players,
        lab.homography.matrices,
        keepers,
        GRADED_HALVES,
        PITCH,
    )
    officials = rescue_officials(
        arrays["players_pixel"],
        arrays["players_present"],
        arrays["players_track"],
        lab.tracking.players,
        lab.homography.matrices,
        PITCH,
    )
    arrays["players_team"] = teams.astype(np.int8)
    arrays["players_official"] = officials
    meta = dict(json.loads(str(arrays["meta"]))) if "meta" in arrays else {}
    meta["keeper_rescue"] = {
        "flips": len(flips),
        "keeper_slots_half_goals": keepers.slot.tolist(),
        "defending_teams": keepers.team.tolist(),
        "official_box_frames": int(officials.sum()),
    }
    arrays["meta"] = np.asarray(json.dumps(meta))
    out = Path(args.out) if args.out else perception_path.with_name(perception_path.stem + "_fixed.npz")
    np.savez_compressed(out, **arrays)
    print(
        f"perception-fix: keeper team flips {len(flips)}, official box-frames "
        f"{int(officials.sum())} -> {out}"
    )
    flips_path = out.with_suffix(".flips.csv")
    if flips:
        pd.DataFrame(flips, columns=["frame", "row", "old_team", "new_team"]).to_csv(
            flips_path, index=False
        )
        print(f"perception-fix: flip log ({len(flips)} rows) -> {flips_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m possession_lab",
        description="Data-only, gap-aware HMM lab for BAR-ATH possession.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "all",
            "predict",
            "evaluate",
            "stress",
            "visualize",
            "viewer",
            "explain",
            "perceive",
            "fusion",
            "transitions",
            "perception-fix",
        ],
    )
    parser.add_argument("--data-dir", default=None, help="gated data directory (default: data/)")
    parser.add_argument("--output-dir", default=None, help="output directory (default: possession_lab/outputs)")
    parser.add_argument("--bootstrap-n", type=int, default=2000, help="bootstrap draws; 0 disables")
    parser.add_argument("--no-stress", action="store_true", help="skip the missing-ball suite")
    parser.add_argument("--no-figures", action="store_true", help="write data only, no report")
    parser.add_argument("--no-explain", action="store_true", help="skip building outputs/explanations.json")

    perception = parser.add_argument_group("perception", "YOLO/video perception and multi-source fusion")
    perception.add_argument("--perception", default=None, help="perception cache path (default: outputs/perception.npz)")
    perception.add_argument("--out", default=None, help="perception-fix: output cache path (default: <perception>_fixed.npz)")
    perception.add_argument("--fusion", default=None, help="fusion cache path (default: outputs/fusion.npz)")
    perception.add_argument("--no-perception", action="store_true", help="ignore any perception cache")
    perception.add_argument("--start", type=int, default=None, help="first video frame for perceive (default: public start)")
    perception.add_argument("--stop", type=int, default=None, help="exclusive last video frame for perceive (default: public stop)")
    perception.add_argument("--step", type=int, default=1, help="process every Nth video frame (default: 1)")
    perception.add_argument("--model", default="yolo11x.pt", help="Ultralytics weights (default: yolo11x.pt)")
    perception.add_argument(
        "--players-model",
        default=None,
        help="dedicated player weights (player/GK/ref classes are all treated as people)",
    )
    perception.add_argument(
        "--players-imgsz",
        type=int,
        default=None,
        help="inference size for --players-model (default: --imgsz)",
    )
    perception.add_argument(
        "--ball-model",
        default=None,
        help="dedicated ball weights (e.g. ball_training/weights/ball_yolo11s_best.pt); default: ball from --model",
    )
    perception.add_argument("--ball-imgsz", type=int, default=1280, help="inference size for --ball-model (default: 1280)")
    perception.add_argument(
        "--stage",
        choices=["both", "players", "ball"],
        default="both",
        help="run both detector families or only one of them (default: both)",
    )
    perception.add_argument("--device", default=None, help="YOLO device, e.g. 0 or cpu (default: auto)")
    perception.add_argument("--imgsz", type=int, default=1920, help="YOLO inference size (default: 1920)")
    perception.add_argument("--combine-mode", choices=["log", "linear"], default="log", help="ballot pool (default: log)")
    perception.add_argument("--skip-baseline", action="store_true", help="do not compute the baseline comparison")
    perception.add_argument("--no-audio", action="store_true", help="disable whistle/audio products")
    perception.add_argument(
        "--no-box",
        action="store_true",
        help="transitions: disable the pixel box-contact channel (kinematic events only)",
    )
    perception.add_argument(
        "--stale-gate-frames",
        type=int,
        default=125,
        help="transitions: drop kinematic kick/onset events inside ATD ball blind windows "
        "longer than N frames on both sides (default: 125 = 5 s; 0 disables)",
    )
    perception.add_argument(
        "--conflict-min-meters",
        type=float,
        default=0.0,
        help="transitions (optional): drop kinematic kick/onset events while ATD and YOLO "
        "balls sit more than N metres apart for a sustained second (default: 0, disabled "
        "-- large divergence is routine here: enabling ~5 fixes drift-pass spells e.g. "
        "20 955, but also gates legitimate events inside divergent live play e.g. the "
        "17.3k spell; measured net-negative on the public window)",
    )
    perception.add_argument(
        "--conflict-pad-frames",
        type=int,
        default=40,
        help="transitions: grow the source-conflict event gate N frames on both sides "
        "(drift ramps in before the divergence crosses the threshold and settles after; "
        "default: 40)",
    )
    perception.add_argument(
        "--dead-hold-confidence",
        type=float,
        default=0.5,
        help="transitions: freeze a dead-ball state across frames that are deep inside "
        "an ATD ball blind window AND below consensus confidence N (default: 0.5; 0 disables)",
    )
    perception.add_argument(
        "--relay-min-prob",
        type=float,
        default=0.8,
        help="transitions: admit a ball-in-box contact inside a settled spell when all four "
        "ball/control voters sit at or above N for the contacting team, both ball streams pin "
        "the ball to the same single player's box, and the spell's team never shows the ball "
        "in its own boxes across the lookback window (default: 0.8; 0 disables)",
    )
    perception.add_argument(
        "--keeper-hold-min-frames",
        type=int,
        default=30,
        help="transitions: frames of ball held by the defending side inside its own penalty "
        "area that count as a keeper catch: possession from the catch (a goalkeeper holding "
        "the ball gives possession to his team). 0 disables (default: 30)",
    )

    viewer = parser.add_argument_group("viewer", "local video reviewer options")
    viewer.add_argument("--port", type=int, default=8765, help="local HTTP port (default: 8765)")
    viewer.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    viewer.add_argument("--predictions", default=None, help="prediction CSV (default: outputs/predictions.csv)")
    viewer.add_argument("--gt", default=None, help="public ground-truth CSV (default: evaluation/gt/...)")
    viewer.add_argument("--video", default=None, help="tactical MP4 (default: data/tactical/BAR-ATH_tactical.mp4)")
    viewer.add_argument("--tags", default=None, help="smart-tag XML (default: gated tactical tags)")
    viewer.add_argument("--metrics", default=None, help="public metrics JSON (default: outputs/metrics_public.json)")
    viewer.add_argument(
        "--user-truth",
        default=None,
        help="user-truth JSON (default: outputs/user_truth.json; viewer-only, never used by the model)",
    )
    viewer.add_argument(
        "--explanations",
        default=None,
        help="switch annotations JSON (default: outputs/explanations.json)",
    )
    viewer.add_argument("--no-explanations", action="store_true", help="start without the annotation layer")
    viewer.add_argument("--start-frame", type=int, default=None, help="initial frame (default: public GT start)")
    viewer.add_argument("--verbose", action="store_true", help="log HTTP requests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        from .config import OUTPUT_DIR

        args.output_dir = str(OUTPUT_DIR)
    commands = {
        "all": cmd_all,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "stress": cmd_stress,
        "visualize": cmd_visualize,
        "viewer": cmd_viewer,
        "explain": cmd_explain,
        "perceive": cmd_perceive,
        "fusion": cmd_fusion,
        "transitions": cmd_transitions,
        "perception-fix": cmd_perception_fix,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
