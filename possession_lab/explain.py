"""Human-readable "why" annotations for every predicted possession switch.

The model is a Viterbi decode over per-frame emissions, so a switch is caused by a *change in the
balance of evidence*, not by a single decision.  This module rebuilds that balance around each
switch from the same feature arrays the model saw and the smart tags, and renders it as short
ordered reasons a human can check against the video:

* ``out_of_bounds`` - the ball's coordinates left the pitch (signed distance to the line);
* ``parked``        - the ball sat near a boundary at walking pace before the switch;
* ``parked``        - the ball sat near a boundary at walking pace before the switch;
* ``restart``       - the ball accelerated from standing to a kick-off/throw within the window;
* ``teams_slow``    - the player group's median speed collapsed relative to the live-play median;
* ``ball_missing``  - the tracker lost the ball for a meaningful stretch;
* ``tag``           - a smart-tag interval (typically ``SET PIECES``) covers or nearly covers f;
* ``geometry``      - who was nearest the ball when the new state was entered;
* ``prior``         - fallback: no strong local signature, the transition prior carried the state.

Every reason carries the measured numbers in ``evidence`` so the viewer can show them, and each
switch carries ``confidence``: the emission probability of the entered state at the switch frame
before temporal smoothing (how strongly that single frame favoured it).

Nothing here feeds back into the model: explanations are built from an existing prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import A, CODE_TO_LABEL, CONTEXT, D, FPS, GRADED_HALVES, H, X
from .data import TagTable, TrackingData
from .features import Features, extract_features
from .model import DEFAULT_PARAMS, ModelParams, emission_log_probs

# Windows around a switch, in seconds: what ended a state leaves its fingerprint *before* the
# boundary, what starts it leaves one just after.
WINDOW_BEFORE_S = 2.5
WINDOW_AFTER_S = 2.5
SWITCH_DEDUPE_FRAMES = int(0.5 * FPS)

TEAM_NAMES = {H: "Barcelona", A: "Athletic Club", D: "balón muerto", X: "sin evaluar"}


def timecode(seconds: float) -> str:
    """mm:ss.s within a match clock of less than two hours."""
    minutes, sec = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(minutes):02d}:{sec:04.1f}"


def find_switches(pred: np.ndarray, halves: tuple[tuple[int, int], ...] = GRADED_HALVES) -> list[tuple[int, int, int]]:
    """State switches ``(frame, from, to)`` inside the graded halves, X included.

    Changes inside the first ``SWITCH_DEDUPE_FRAMES`` of a half are skipped: that is context-pad
    territory, and a switch there is the decode warming up, not a decision about the match.
    """
    switches: list[tuple[int, int, int]] = []
    for start, stop in halves:
        start, stop = int(start), int(stop)
        previous = int(pred[start])
        for frame in range(start + 1, stop):
            current = int(pred[frame])
            if current != previous and current != X and previous != X:
                if not switches or frame - switches[-1][0] >= SWITCH_DEDUPE_FRAMES:
                    switches.append((frame, previous, current))
            previous = current
    return switches


def _coverage(window: np.ndarray) -> float:
    return float(window.mean()) if window.size else 0.0


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _json_num(v: Any) -> Any:
    """Round floats for JSON; never let a NaN/Inf through (browsers reject it)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return round(f, 3) if np.isfinite(f) else None
    return v


def _reason(kind: str, text: str, **evidence: Any) -> dict:
    return {"kind": kind, "text": text, "evidence": {k: _json_num(v) for k, v in evidence.items()}}


def _rule_out_of_bounds(features: Features, lo: int, hi: int, offset: int) -> dict | None:
    oop = features.out_of_play[lo:hi]
    if not oop.any():
        return None
    idx = lo + int(np.argmax(oop))
    depth = float(abs(features.edge[idx])) if np.isfinite(features.edge[idx]) else 0.0
    return _reason("out_of_bounds", f"El balón sale del campo: {depth:.1f} m más allá de la línea (frame {idx + offset}).", frame=idx + offset, depth_m=depth)


def _rule_parked(features: Features, frame: int) -> dict | None:
    lo = max(0, frame - int(3.0 * FPS))
    hi = min(features.present.size, frame + int(1.0 * FPS))
    mask = features.parked_edge[lo:hi]
    if not mask.any() or _coverage(mask.astype(float)) < 0.2:
        return None
    run = _longest_run(mask)
    # A parked run often contains frames where the ATD loses the ball: prefer a frame whose
    # geometry is actually measured, so the evidence carries real numbers, not nulls.
    measured = mask & np.isfinite(features.edge[lo:hi]) & np.isfinite(features.ball_speed[lo:hi])
    idx = lo + int(np.argmax(measured)) if measured.any() else lo + int(np.argmax(mask.astype(np.int8)))
    edge = float(features.edge[idx]) if np.isfinite(features.edge[idx]) else None
    speed = float(features.ball_speed[idx]) if np.isfinite(features.ball_speed[idx]) else None
    edge_txt = f"{edge:.1f} m" if edge is not None else "sin medir"
    speed_txt = f"({speed:.1f} m/s)" if speed is not None else "(velocidad sin medir)"
    return _reason(
        "parked",
        f"Balón parado junto a la línea {run / FPS:.1f} s a {edge_txt} {speed_txt}: pinta a saque de banda/esquina.",
        frames=run,
        seconds=run / FPS,
        edge_m=edge,
        ball_speed_mps=speed,
    )


def _rule_restart(features: Features, local: int, offset: int) -> dict | None:
    hi = min(features.present.size, local + int(2.0 * FPS))
    pre_lo = max(0, local - 25)
    pre = features.ball_speed[pre_lo:local]
    pre = pre[features.present[pre_lo:local] & np.isfinite(pre)]
    if pre.size < 12 or float(np.median(pre)) >= 1.5:
        return None
    med = float(np.median(pre))
    seg = features.ball_speed[local:hi]
    segp = features.present[local:hi]
    peak = med
    crossing = None
    for k in range(seg.size):
        if not segp[k] or not np.isfinite(seg[k]):
            continue
        peak = max(peak, float(seg[k]))
        if crossing is None and seg[k] >= 5.0:
            crossing = local + k
    if crossing is None:
        return None
    elapsed = (crossing - local) / FPS
    return _reason(
        "restart",
        f"Reanudación: el balón acelera de {med:.1f} a {peak:.1f} m/s ({elapsed:.1f} s tras el cambio).",
        frame=crossing + offset,
        speed_from=med,
        speed_to=peak,
        elapsed_s=elapsed,
    )


def _rule_teams_slow(features: Features, frame: int, live_median: float) -> dict | None:
    lo = max(0, frame - int(2.0 * FPS))
    hi = min(features.team_speed.size, frame + int(2.0 * FPS))
    window = features.team_speed[lo:hi]
    window = window[np.isfinite(window)]
    if window.size < FPS or not np.isfinite(live_median) or live_median <= 0:
        return None
    median = float(np.median(window))
    if median >= 0.62 * live_median:
        return None
    return _reason(
        "teams_slow",
        f"El grupo se mueve a {median:.1f} m/s (mediana de la ventana {live_median:.1f}): todo el mundo andando.",
        team_speed_mps=median,
        live_median_mps=live_median,
    )


def _rule_ball_missing(features: Features, frame: int) -> dict | None:
    lo = max(0, frame - int(1.0 * FPS))
    hi = min(features.present.size, frame + int(2.5 * FPS))
    run = _longest_run(~features.present[lo:hi])
    if run < FPS:
        return None
    return _reason(
        "ball_missing",
        f"El tracker pierde el balón {run / FPS:.1f} s alrededor del cambio.",
        seconds=run / FPS,
    )


def _rule_tag(index: tuple[np.ndarray, np.ndarray, list] | None, frame: int) -> dict | None:
    """Set-piece smart tag covering or nearest ``frame`` (±3 s).  Only ``SET PIECES`` intervals
    annotate a switch: ``BALL IN FINAL THIRD`` runs for minutes and explains nothing."""
    if index is None:
        return None
    starts, stops, meta = index
    lo, hi = frame - 3 * FPS, frame + 3 * FPS
    candidates = np.flatnonzero((stops >= lo) & (starts <= hi) & np.array([m["setpiece"] for m in meta]))
    if candidates.size == 0:
        return None
    covering = (starts[candidates] <= frame) & (stops[candidates] > frame)
    pick = int(candidates[int(np.argmax(covering)) if covering.any() else 0])
    event = meta[pick]
    team = TEAM_NAMES.get(event["team"], "N/A")
    label = f"Etiqueta inteligente: {event['code']}"
    if event["type"]:
        label += f" · {event['type']}"
    label += f" ({team}, {timecode(event['start'] / FPS)}–{timecode(event['stop'] / FPS)})"
    return _reason(
        "tag",
        label,
        start=event["start"],
        stop=event["stop"],
        code=event["code"],
        type=event["type"],
        team=event["team"],
    )


def _rule_geometry(features: Features, frame: int) -> dict | None:
    hi = min(features.present.size, frame + int(2.0 * FPS))
    span = features.present[frame:hi]
    if not span.any() or features.near_distance.size == 0:
        return None
    idx = frame + int(np.argmax(span))
    if not np.isfinite(features.near_distance[idx]):
        return None
    team = TEAM_NAMES.get(int(features.near_team[idx]), "N/A")
    return _reason(
        "geometry",
        f"Jugador más cercano al balón: {team} a {features.near_distance[idx]:.1f} m.",
        near_team=int(features.near_team[idx]),
        near_distance_m=float(features.near_distance[idx]),
    )


def build_switch_records(
    pred: np.ndarray,
    features: Features,
    features_offset: int,
    tags: TagTable | None,
    log_emission: np.ndarray,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
) -> list[dict]:
    """Explain every switch inside ``features``' window (absolute frame addressing throughout)."""
    tag_index = None
    if tags is not None and len(tags.events):
        ev_meta = [
            {"start": int(e.start_frame), "stop": int(e.stop_frame), "code": e.code, "type": e.type, "team": int(e.team), "setpiece": e.code == "SET PIECES"}
            for e in tags.events
        ]
        tag_index = (
            np.array([m["start"] for m in ev_meta], dtype=np.int64),
            np.array([m["stop"] for m in ev_meta], dtype=np.int64),
            ev_meta,
        )
    team_speed = features.team_speed
    live_median = float(np.median(team_speed[np.isfinite(team_speed)])) if np.isfinite(team_speed).any() else float("nan")
    records: list[dict] = []
    for frame, source, target in find_switches(pred, halves):
        local = frame - features_offset
        if not (0 <= local < features.present.size):
            continue
        lo = max(0, local - int(WINDOW_BEFORE_S * FPS))
        hi = min(features.present.size, local + int(WINDOW_AFTER_S * FPS))
        reasons = [r for r in (
            _rule_out_of_bounds(features, lo, hi, features_offset),
            _rule_parked(features, local),
            _rule_restart(features, local, features_offset) if target != D else None,
            _rule_teams_slow(features, local, live_median),
            _rule_ball_missing(features, local),
            _rule_tag(tag_index, frame),
            _rule_geometry(features, local) if target in (H, A) else None,
        ) if r]
        if not reasons:
            reasons = [_reason("prior", f"Sin señal local fuerte: el prior de transición del HMM mantiene «{TEAM_NAMES[target]}».", prior=True)]
        band = log_emission[max(0, local - FPS // 2): local + FPS // 2 + 1]
        finite = np.where(np.isfinite(band), band, -1e9)
        softmax = np.exp(finite - finite.max(axis=1, keepdims=True))
        row_prob = softmax[:, target] / softmax.sum(axis=1, keepdims=True)
        confidence = float(np.nan_to_num(row_prob).max()) if row_prob.size else 0.0
        records.append(
            {
                "frame": int(frame),
                "t_s": round(frame / FPS, 3),
                "from": CODE_TO_LABEL[source],
                "to": CODE_TO_LABEL[target],
                "confidence": round(confidence, 3),
                "summary": reasons[0]["text"],
                "reasons": reasons[:4],
            }
        )
    return records


def build_explanations(
    pred: np.ndarray,
    features: Features,
    features_offset: int,
    tags: TagTable | None,
    params: ModelParams = DEFAULT_PARAMS,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
) -> dict:
    """Payload for the viewer: per-switch "why" records over one feature window."""
    log_emission = emission_log_probs(features, params)
    return {
        "fps": FPS,
        "switches": build_switch_records(pred, features, features_offset, tags, log_emission, halves),
    }


def explain_prediction(
    tracking: TrackingData,
    tags: TagTable,
    pred: np.ndarray,
    params: ModelParams = DEFAULT_PARAMS,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
    context: int = CONTEXT,
) -> dict:
    """Explain every switch over the full match, one features window per graded half.

    Each half is explained with the same context-padded feature window the decode used, so the
    annotations describe the evidence the model actually saw.
    """
    n = tracking.n_frames
    switches: list[dict] = []
    for start, stop in halves:
        lo = max(0, int(start) - context)
        hi = min(n, int(stop) + context)
        features = extract_features(tracking, tags, start=lo, stop=hi)
        switches.extend(
            build_switch_records(
                pred, features, lo, tags, emission_log_probs(features, params), halves=[(int(start), int(stop))]
            )
        )
    switches.sort(key=lambda record: record["frame"])
    return {"fps": FPS, "switches": switches}
