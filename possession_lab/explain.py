"""Human-readable "why" annotations for every predicted possession switch.

The model is a Viterbi decode over per-frame emissions, so a switch is caused by a *change in the
balance of evidence*, not by a single decision.  This module rebuilds that balance around each
switch from the same feature arrays the model saw and the smart tags, and renders it as short
ordered reasons a human can check against the video:

* ``out_of_bounds`` - the ball's coordinates left the pitch (signed distance to the line);
* ``parked``        - the ball sat near a boundary at walking pace before the switch;
* ``restart``       - the ball accelerated from standing to a kick-off/throw within the window;
* ``teams_slow``    - the player group's median speed collapsed relative to the live-play median;
* ``ball_missing``  - context only: the tracker lost the ball for a stretch.  A hole in the data
  is not an event in the match, so it is never the leading reason and it never stands alone;
* ``tag``           - a smart-tag interval (typically ``SET PIECES``) covers or nearly covers f;
* ``geometry``      - who was nearest the ball when the new state was entered;
* ``prior``         - fallback: no strong local signature, the transition prior carried the state.

Every reason carries the measured numbers in ``evidence`` so the viewer can show them, and each
switch carries ``confidence``: the emission probability of the entered state at the switch frame
before temporal smoothing (how strongly that single frame favoured it).

Nothing here feeds back into the model: explanations are built from an existing prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CODE_TO_LABEL,
    CONTEXT,
    FPS,
    GRADED_HALVES,
    OUTPUT_DIR,
    PITCH,
    STOP_CONTROL_R,
    STOP_IN_RUN_BALL_SPEED,
    STOP_TEAM_SPEED,
    A,
    D,
    H,
    X,
)
from .data import TagTable, TrackingData
from .features import Features, extract_features
from .model import DEFAULT_PARAMS, ModelParams, emission_log_probs

# Windows around a switch, in seconds: what ended a state leaves its fingerprint *before* the
# boundary, what starts it leaves one just after.
WINDOW_BEFORE_S = 2.5
WINDOW_AFTER_S = 2.5
SWITCH_DEDUPE_FRAMES = int(0.5 * FPS)

TEAM_NAMES = {H: "Barcelona", A: "Athletic Club", D: "balón muerto", X: "sin evaluar"}

TAG_LABELS = {
    "parada_portero": "Parada de portero",
    "control_portero": "Control del portero",
    "falta_arbitral": "Falta arbitral",
    "silbato_arbitro": "Silbato del árbitro",
    "fuera_de_banda": "Fuera de banda",
    "fuera_de_fondo": "Línea de fondo / Córner",
    "fuera_campo": "Balón fuera del campo",
    "preparacion_saque": "Preparación de saque",
    "paron_colectivo": "Parón colectivo",
    "balon_muerto": "Balón muerto",
    "reanudacion_saque": "Reanudación",
    "saque_banda": "Saque de banda",
    "saque_esquina": "Córner",
    "saque_puerta": "Saque de puerta",
    "saque_falta": "Tiro libre",
    "robo_intercepcion": "Robo / Intercepción",
    "contacto_disputa": "Duelo / Disputa",
    "perdida_posesion": "Pérdida de posesión",
    "control_balon": "Control de balón",
    "pase_en_juego": "Pase en juego",
    "etiqueta_inteligente": "Etiqueta inteligente",
    "prior_hmm": "Inferencia HMM",
}

CATEGORY_LABELS = {
    "portero": "Intervención de portero",
    "arbitro_falta": "Decisión arbitral / Falta",
    "balon_fuera": "Fuera de límites",
    "balon_parado": "Balón parado",
    "reanudacion": "Reanudación del juego",
    "cambio_posesion": "Disputa / Recuperación",
    "control_juego": "Posesión activa",
    "evento_oficial": "Evento de partido",
    "prior": "Continuidad del modelo",
}


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
            if (
                current != previous
                and current != X
                and previous != X
                and (not switches or frame - switches[-1][0] >= SWITCH_DEDUPE_FRAMES)
            ):
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


def _reason(kind: str, text: str, tag: str | None = None, **evidence: Any) -> dict:
    d = {"kind": kind, "text": text, "evidence": {k: _json_num(v) for k, v in evidence.items()}}
    if tag is not None:
        d["tag"] = tag
    return d


def _rule_keeper_save(
    features: Features,
    local: int,
    offset: int,
    target: int,
    events_df: Any = None,
    ball_xy: np.ndarray | None = None,
) -> dict | None:
    """Detect goalkeeper save / catch / hold in the defending penalty area."""
    if target not in (H, A):
        return None
    frame = local + offset
    target_label = CODE_TO_LABEL.get(target, "")

    # Check 1: transition_events table hold / area event within window
    if events_df is not None and hasattr(events_df, "empty") and not events_df.empty:
        lo_f, hi_f = frame - int(2.5 * FPS), frame + int(2.5 * FPS)
        subset = events_df[
            (events_df["event_frame"] >= lo_f)
            & (events_df["start_frame"] <= hi_f)
            & (events_df["kind"].isin(["hold", "area"]))
            & (events_df["team"] == target_label)
        ]
        if not subset.empty:
            row = subset.iloc[0]
            team_name = TEAM_NAMES.get(target, "equipo")
            ev_desc = "parada y retención" if str(row["kind"]) == "hold" else "bloqueo en el área"
            return _reason(
                "keeper_save",
                f"Parada del portero: {ev_desc} del balón ({team_name}).",
                tag="parada_portero",
                event_kind=str(row["kind"]),
                event_frame=int(row["event_frame"]),
                team=target,
            )

    # Check 2: ball position inside penalty area and near target team
    if 0 <= local < features.present.size:
        xy = None
        if ball_xy is not None and local < len(ball_xy):
            xy = ball_xy[local]
        elif hasattr(features, "ball_est") and features.ball_est.shape[0] > local:
            xy = features.ball_est[local]
        if xy is not None and np.isfinite(xy[0]) and np.isfinite(xy[1]):
            bx, by = float(xy[0]), float(xy[1])
            # Penalty boxes: length is ~105.3, width is 68.0
            in_box = (bx <= 18.0 or bx >= 87.0) and (12.0 <= by <= 56.0)
            if in_box:
                near_d = float(features.near_distance[local]) if np.isfinite(features.near_distance[local]) else 99.0
                near_t = int(features.near_team[local]) if features.near_team.size > local else -1
                if near_t == target and near_d <= 4.0:
                    team_name = TEAM_NAMES.get(target, "portero")
                    return _reason(
                        "keeper_save",
                        f"Intervención del portero: balón controlado dentro del área de penalti ({team_name}, a {near_d:.1f} m).",
                        tag="parada_portero",
                        box_x=bx,
                        box_y=by,
                        near_distance_m=near_d,
                        team=target,
                    )
    return None


def _rule_foul(
    features: Features,
    local: int,
    offset: int,
    target: int,
    whistle_data: Any = None,
) -> dict | None:
    """Detect referee whistle / foul leading to dead ball."""
    if target != D:
        return None
    frame = local + offset

    # Check 1: whistle_data (e.g. from whistle.npz)
    if whistle_data is not None:
        starts = whistle_data.get("starts") if isinstance(whistle_data, dict) else getattr(whistle_data, "starts", None)
        if starts is None and hasattr(whistle_data, "__getitem__"):
            try:
                starts = whistle_data["starts"]
            except Exception:
                starts = None
        if starts is not None and len(starts) > 0:
            window_f = int(2.0 * FPS)
            near_whistles = starts[(starts >= frame - window_f) & (starts <= frame + window_f)]
            if len(near_whistles) > 0:
                wf = int(near_whistles[0])
                diff_s = (wf - frame) / FPS
                return _reason(
                    "foul",
                    f"Falta / silbato arbitral: el árbitro interrumpe el juego (silbato en frame {wf}, {diff_s:+.1f} s).",
                    tag="falta_arbitral",
                    whistle_frame=wf,
                    offset_s=diff_s,
                )

    # Check 2: features.whistle_score
    if 0 <= local < features.present.size and hasattr(features, "whistle_score") and features.whistle_score.size > local:
        w_lo = max(0, local - int(1.5 * FPS))
        w_hi = min(features.whistle_score.size, local + int(1.5 * FPS))
        w_scores = features.whistle_score[w_lo:w_hi]
        if w_scores.size and np.nanmax(w_scores) > 0.4:
            max_idx = w_lo + int(np.nanargmax(w_scores))
            return _reason(
                "foul",
                f"Falta / silbato arbitral: señal acústica de silbato detectada ({float(np.nanmax(w_scores)):.2f} confianza).",
                tag="falta_arbitral",
                frame=max_idx + offset,
                score=float(np.nanmax(w_scores)),
            )

    return None


def _rule_turnover_interception(
    features: Features,
    local: int,
    offset: int,
    source: int,
    target: int,
    events_df: Any = None,
) -> dict | None:
    """Detect interception, steal, or duel when possession changes directly between H and A."""
    if source not in (H, A) or target not in (H, A):
        return None
    if not (0 <= local < features.present.size):
        return None

    frame = local + offset
    target_label = CODE_TO_LABEL.get(target, "")

    # Check 1: transition_events contact/kicker
    if events_df is not None and hasattr(events_df, "empty") and not events_df.empty:
        lo_f, hi_f = frame - int(1.5 * FPS), frame + int(1.5 * FPS)
        subset = events_df[
            (events_df["event_frame"] >= lo_f)
            & (events_df["start_frame"] <= hi_f)
            & (events_df["team"] == target_label)
        ]
        if not subset.empty:
            ev = subset.iloc[0]
            team_name = TEAM_NAMES.get(target, "equipo rival")
            source_name = TEAM_NAMES.get(source, "rival")
            if str(ev["kind"]) in ("contact", "kick"):
                return _reason(
                    "interception",
                    f"Robo / intercepción: toque directo de {team_name} que arrebata la posesión a {source_name}.",
                    tag="robo_intercepcion",
                    event_kind=str(ev["kind"]),
                    event_frame=int(ev["event_frame"]),
                    team=target,
                )
            elif str(ev["kind"]) == "kicker":
                return _reason(
                    "interception",
                    f"Recuperación y despeje: {team_name} gana la disputa y juega el esférico.",
                    tag="robo_intercepcion",
                    event_kind=str(ev["kind"]),
                    event_frame=int(ev["event_frame"]),
                    team=target,
                )

    # Check 2: Geometry & margin around switch
    hi = min(features.present.size, local + int(1.5 * FPS))
    span = features.present[local:hi]
    if span.any() and features.near_distance.size > local:
        idx = local + int(np.argmax(span))
        near_d = float(features.near_distance[idx]) if np.isfinite(features.near_distance[idx]) else None
        near_t = int(features.near_team[idx]) if features.near_team.size > idx else -1
        margin = float(features.margin[idx]) if features.margin.size > idx and np.isfinite(features.margin[idx]) else None
        team_name = TEAM_NAMES.get(target, "equipo rival")
        source_name = TEAM_NAMES.get(source, "rival")
        if near_t == target and near_d is not None:
            if near_d <= 1.8 and (margin is not None and margin >= 0.8):
                return _reason(
                    "interception",
                    f"Robo / intercepción: {team_name} anticipa y toma control del balón a {near_d:.1f} m de {source_name}.",
                    tag="robo_intercepcion",
                    near_distance_m=near_d,
                    margin_m=margin,
                    team=target,
                )
            elif near_d <= 2.5:
                return _reason(
                    "duel",
                    f"Duelo / disputa dividida: pugna cuerpo a cuerpo resuelta para {team_name} a {near_d:.1f} m.",
                    tag="contacto_disputa",
                    near_distance_m=near_d,
                    margin_m=margin,
                    team=target,
                )
    return None


def _rule_setpiece_specific(
    features: Features,
    local: int,
    offset: int,
    source: int,
    target: int,
    ball_xy: np.ndarray | None = None,
) -> dict | None:
    """Classify set-piece restart subtype (throw-in, corner, goal kick)."""
    if source != D or target not in (H, A):
        return None
    if not (0 <= local < features.present.size):
        return None

    xy = None
    if ball_xy is not None and local < len(ball_xy):
        xy = ball_xy[local]
    elif hasattr(features, "ball_est") and features.ball_est.shape[0] > local:
        xy = features.ball_est[local]

    team_name = TEAM_NAMES.get(target, "equipo atacante")

    if xy is not None and np.isfinite(xy[0]) and np.isfinite(xy[1]):
        bx, by = float(xy[0]), float(xy[1])
        # Corner arcs
        if (bx <= 4.0 or bx >= 101.3) and (by <= 4.0 or by >= 64.0):
            return _reason(
                "corner_restart",
                f"Saque de esquina: córner ejecutado por {team_name} al área.",
                tag="saque_esquina",
                x=bx,
                y=by,
                team=target,
            )
        # Touchline throw-in
        if (4.0 < bx < 101.3) and (by <= 3.0 or by >= 65.0):
            return _reason(
                "throwin_restart",
                f"Saque de banda: reanudación de banda ejecutada para {team_name}.",
                tag="saque_banda",
                x=bx,
                y=by,
                team=target,
            )
        # Goal kick / goalkeeper restart
        if (bx <= 26.0 or bx >= 79.0) and (13.0 <= by <= 55.0):
            return _reason(
                "goalkick_restart",
                f"Saque de portería / reinicio en largo del portero de {team_name} desde el área.",
                tag="saque_puerta",
                x=bx,
                y=by,
                team=target,
            )
        # Free kick restart (out on the field, e.g. after offside or defensive foul)
        if (24.0 < bx < 81.0) and (3.0 < by < 65.0):
            return _reason(
                "freekick_restart",
                f"Tiro libre / fuera de juego: reanudación a balón parado para {team_name}.",
                tag="reanudacion_tiro_libre",
                x=bx,
                y=by,
                team=target,
            )

    edge = float(features.edge[local]) if np.isfinite(features.edge[local]) else None
    if edge is not None and abs(edge) <= 2.5:
        return _reason(
            "throwin_restart",
            f"Saque de banda: balón reanudado desde la línea ({edge:.1f} m) para {team_name}.",
            tag="saque_banda",
            edge_m=edge,
            team=target,
        )

    return None


def assign_decision_tags(
    source: int,
    target: int,
    reasons: list[dict],
    confidence: float,
) -> tuple[str, list[str], str, str]:
    """Map the collected reason signals into (primary_tag, tags, category, justification)."""
    kinds = [r["kind"] for r in reasons]
    source_name = TEAM_NAMES.get(source, "equipo")
    target_name = TEAM_NAMES.get(target, "equipo")

    if "keeper_save" in kinds:
        primary_tag = "parada_portero"
        category = "portero"
        tags = ["parada_portero", "control_portero"]
        justification = (
            f"El guardameta de {target_name} asegura o bloquea el esférico dentro del área de penalti "
            f"tras una aproximación rival, garantizando la posesión directa para su equipo."
        )
    elif "foul" in kinds:
        primary_tag = "falta_arbitral"
        category = "arbitro_falta"
        tags = ["falta_arbitral", "balon_muerto"]
        justification = (
            f"Infracción o detención señalada con silbato por el árbitro, "
            f"interrumpiendo la jugada activa de {source_name} y pasando el estado a balón muerto."
        )
    elif "corner_restart" in kinds:
        primary_tag = "saque_esquina"
        category = "reanudacion"
        tags = ["saque_esquina", "reanudacion_saque"]
        justification = (
            f"Reanudación del juego mediante saque de esquina botado por {target_name} hacia el área."
        )
    elif "throwin_restart" in kinds:
        primary_tag = "saque_banda"
        category = "reanudacion"
        tags = ["saque_banda", "reanudacion_saque"]
        justification = (
            f"Reanudación del partido mediante saque de banda ejecutado a favor de {target_name}."
        )
    elif "goalkick_restart" in kinds:
        primary_tag = "saque_puerta"
        category = "reanudacion"
        tags = ["saque_puerta", "reanudacion_saque"]
        justification = (
            f"Reinicio del partido mediante saque de puerta ejecutado por {target_name} desde su propia área."
        )
    elif "freekick_restart" in kinds:
        primary_tag = "reanudacion_tiro_libre"
        category = "reanudacion"
        tags = ["reanudacion_tiro_libre", "tiro_libre", "reanudacion_saque"]
        justification = (
            f"Reanudación reglamentaria del juego mediante tiro libre ejecutado por {target_name} "
            f"tras detención arbitral por fuera de juego o falta."
        )
    elif "interception" in kinds:
        primary_tag = "robo_intercepcion"
        category = "cambio_posesion"
        tags = ["robo_intercepcion", "recuperacion"]
        justification = (
            f"Cambio de posesión por anticipación o robo defensivo: jugador de {target_name} "
            f"intercepta la trayectoria del esférico y toma el control frente a {source_name}."
        )
    elif "duel" in kinds:
        primary_tag = "contacto_disputa"
        category = "cambio_posesion"
        tags = ["contacto_disputa", "duelo_ganado"]
        justification = (
            f"Duelo o disputa dividida en balón dividido entre jugadores de ambos conjuntos, "
            f"resuelto con posesión efectiva para {target_name}."
        )
    elif "out_of_bounds" in kinds:
        primary_tag = "fuera_de_banda"
        category = "balon_fuera"
        tags = ["fuera_de_banda", "balon_muerto"]
        justification = (
            f"El esférico rebasa los límites del terreno de juego reglamentario tras la acción de {source_name}, "
            f"deteniendo el partido en balón muerto."
        )
    elif "parked" in kinds:
        primary_tag = "preparacion_saque"
        category = "balon_parado"
        tags = ["preparacion_saque", "balon_muerto"]
        justification = (
            f"Balón detenido junto a la línea mientras los jugadores se ubican para el saque correspondiente."
        )
    elif "restart" in kinds:
        primary_tag = "reanudacion_saque"
        category = "reanudacion"
        tags = ["reanudacion_saque", "pase_en_juego"]
        justification = (
            f"El balón acelera con fuerza desde una posición estática, reanudando la posesión en juego activo para {target_name}."
        )
    elif "stopped_field" in kinds or "teams_slow" in kinds:
        if target == D:
            primary_tag = "paron_colectivo"
            category = "balon_parado"
            tags = ["paron_colectivo", "balon_muerto"]
            justification = (
                f"Caída drástica y generalizada de la velocidad media del grupo de jugadores, "
                f"certificando el cese de juego en balón muerto."
            )
        else:
            primary_tag = "control_juego"
            category = "control_juego"
            tags = ["control_juego"]
            justification = (
                f"Posesión activa y pausada de {target_name} con circulación del balón."
            )
    elif "geometry" in kinds:
        if source == D:
            primary_tag = "reanudacion_saque"
            category = "reanudacion"
            tags = ["reanudacion_saque", "control_balon"]
            justification = (
                f"Reanudación del juego bajo control inmediato del jugador más cercano de {target_name}."
            )
        else:
            primary_tag = "robo_intercepcion"
            category = "cambio_posesion"
            tags = ["robo_intercepcion", "control_balon"]
            justification = (
                f"Toma de posesión de {target_name} al encontrarse su jugador en proximidad y control directo del balón."
            )
    elif "tag" in kinds:
        primary_tag = "etiqueta_inteligente"
        category = "evento_oficial"
        tags = ["etiqueta_inteligente"]
        justification = (
            f"Cambio alineado con etiqueta oficial de evento registrada en los metadatos del partido."
        )
    else:
        primary_tag = "prior_hmm"
        category = "prior"
        tags = ["prior_hmm"]
        justification = (
            f"Transición inferida por el prior de continuidad secuencial y emisión probabilística del modelo."
        )

    return primary_tag, tags, category, justification


def _rule_out_of_bounds(features: Features, lo: int, hi: int, offset: int) -> dict | None:
    oop = features.out_of_play[lo:hi]
    if not oop.any():
        return None
    idx = lo + int(np.argmax(oop))
    depth = float(abs(features.edge[idx])) if np.isfinite(features.edge[idx]) else 0.0
    return _reason(
        "out_of_bounds",
        f"El balón sale del campo: {depth:.1f} m más allá de la línea (frame {idx + offset}).",
        tag="fuera_de_banda",
        frame=idx + offset,
        depth_m=depth,
    )


def _rule_parked(features: Features, frame: int) -> dict | None:
    lo = max(0, frame - int(3.0 * FPS))
    hi = min(features.present.size, frame + int(1.0 * FPS))
    mask = features.parked_edge[lo:hi]
    if not mask.any() or _coverage(mask.astype(float)) < 0.2:
        return None
    run = _longest_run(mask)
    measured = mask & np.isfinite(features.edge[lo:hi]) & np.isfinite(features.ball_speed[lo:hi])
    idx = lo + int(np.argmax(measured)) if measured.any() else lo + int(np.argmax(mask.astype(np.int8)))
    edge = float(features.edge[idx]) if np.isfinite(features.edge[idx]) else None
    speed = float(features.ball_speed[idx]) if np.isfinite(features.ball_speed[idx]) else None
    edge_txt = f"{edge:.1f} m" if edge is not None else "sin medir"
    speed_txt = f"({speed:.1f} m/s)" if speed is not None else "(velocidad sin medir)"
    return _reason(
        "parked",
        f"Balón parado junto a la línea {run / FPS:.1f} s a {edge_txt} {speed_txt}: pinta a saque de banda/esquina.",
        tag="preparacion_saque",
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
        tag="reanudacion_saque",
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
        tag="paron_colectivo",
        team_speed_mps=median,
        live_median_mps=live_median,
    )


def _rule_stopped_field(features: Features, frame: int) -> dict | None:
    half = int(1.0 * FPS)
    ahead = int(6.0 * FPS)
    lo = max(0, frame - half)
    hi = min(features.team_speed.size, frame + ahead)
    speed = features.team_speed[lo:hi]
    if speed.size < FPS:
        return None
    walking = speed[np.isfinite(speed)] < STOP_TEAM_SPEED
    run = _longest_run(walking) if walking.size else 0
    if run < 4 * FPS:
        return None
    live = (
        features.present[lo:hi]
        & (features.gap[lo:hi] <= 5)
        & ((features.ball_speed[lo:hi] >= STOP_IN_RUN_BALL_SPEED)
           | (features.near_distance[lo:hi] <= STOP_CONTROL_R))
    )
    idle = 1.0 - float(live.mean())
    if idle < 0.75:
        return None
    median = float(np.median(features.team_speed[lo:hi][np.isfinite(features.team_speed[lo:hi])]))
    return _reason(
        "stopped_field",
        f"Balón parado: el grupo lleva {run / FPS:.0f} s a {median:.1f} m/s y el balón no está "
        f"en juego en el {idle * 100:.0f} % del tramo.",
        tag="balon_muerto",
        walking_seconds=run / FPS,
        team_speed_mps=median,
        idle_fraction=idle,
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
        tag="balon_perdido",
        seconds=run / FPS,
    )


def _rule_tag(index: tuple[np.ndarray, np.ndarray, list] | None, frame: int) -> dict | None:
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
        tag="etiqueta_inteligente",
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
        tag="control_balon",
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
    events: Any | None = None,
    whistle: Any | None = None,
    ball_xy: np.ndarray | None = None,
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

    events_df = events
    whistle_data = whistle

    team_speed = features.team_speed
    live_median = float(np.median(team_speed[np.isfinite(team_speed)])) if np.isfinite(team_speed).any() else float("nan")
    records: list[dict] = []
    for frame, source, target in find_switches(pred, halves):
        local = frame - features_offset
        if not (0 <= local < features.present.size):
            continue
        lo = max(0, local - int(WINDOW_BEFORE_S * FPS))
        hi = min(features.present.size, local + int(WINDOW_AFTER_S * FPS))

        # Specialized domain rules
        keeper_r = _rule_keeper_save(features, local, features_offset, target, events_df, ball_xy)
        foul_r = _rule_foul(features, local, features_offset, target, whistle_data)
        turnover_r = _rule_turnover_interception(features, local, features_offset, source, target, events_df)
        setpiece_r = _rule_setpiece_specific(features, local, features_offset, source, target, ball_xy)

        # Base geometric and kinetic rules
        oob_r = _rule_out_of_bounds(features, lo, hi, features_offset)
        parked_r = _rule_parked(features, local)
        restart_r = _rule_restart(features, local, features_offset) if target != D else None
        slow_r = _rule_teams_slow(features, local, live_median)
        tag_r = _rule_tag(tag_index, frame)
        stopped_r = _rule_stopped_field(features, local) if target == D else None
        geom_r = _rule_geometry(features, local) if target in (H, A) else None
        context = _rule_ball_missing(features, local)

        primary_signals = [r for r in (keeper_r, foul_r, turnover_r, setpiece_r) if r]
        base_signals = [r for r in (oob_r, parked_r, restart_r, slow_r, tag_r, stopped_r, geom_r) if r]

        seen_kinds = set()
        reasons = []
        for r in primary_signals + base_signals:
            if r["kind"] not in seen_kinds:
                seen_kinds.add(r["kind"])
                reasons.append(r)

        if reasons and context:
            reasons.append(context)
        if not reasons:
            reasons = [_reason("prior", f"Sin señal local fuerte: el prior de transición del HMM mantiene «{TEAM_NAMES[target]}».", tag="prior_hmm", prior=True)]

        band = log_emission[max(0, local - FPS // 2): local + FPS // 2 + 1]
        finite = np.where(np.isfinite(band), band, -1e9)
        softmax = np.exp(finite - finite.max(axis=1, keepdims=True))
        row_prob = softmax[:, target] / softmax.sum(axis=1, keepdims=True)
        confidence = float(np.nan_to_num(row_prob).max()) if row_prob.size else 0.0

        primary_tag, tags_list, category, justification = assign_decision_tags(
            source, target, reasons, confidence
        )

        records.append(
            {
                "frame": int(frame),
                "t_s": round(frame / FPS, 3),
                "timecode": timecode(frame / FPS),
                "from": CODE_TO_LABEL[source],
                "to": CODE_TO_LABEL[target],
                "confidence": round(confidence, 3),
                "primary_tag": primary_tag,
                "tags": tags_list,
                "category": category,
                "summary": reasons[0]["text"],
                "justification": justification,
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
    events: Any | None = None,
    whistle: Any | None = None,
    ball_xy: np.ndarray | None = None,
) -> dict:
    """Payload for the viewer: per-switch "why" records over one feature window."""
    log_emission = emission_log_probs(features, params)
    return {
        "fps": FPS,
        "switches": build_switch_records(
            pred, features, features_offset, tags, log_emission, halves, events=events, whistle=whistle, ball_xy=ball_xy
        ),
    }


def explain_prediction(
    tracking: TrackingData,
    tags: TagTable,
    pred: np.ndarray,
    params: ModelParams = DEFAULT_PARAMS,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
    context: int = CONTEXT,
    events: Any | None = None,
    whistle: Any | None = None,
) -> dict:
    """Explain every switch over the full match, one features window per graded half.

    Each half is explained with the same context-padded feature window the decode used, so the
    annotations describe the evidence the model actually saw.
    """
    if events is None:
        ev_file = OUTPUT_DIR / "transition_events.csv"
        if ev_file.is_file():
            try:
                import pandas as pd
                events = pd.read_csv(ev_file)
            except Exception:
                events = None
    if whistle is None:
        wh_file = OUTPUT_DIR / "whistle.npz"
        if wh_file.is_file():
            try:
                whistle = np.load(wh_file)
            except Exception:
                whistle = None

    n = tracking.n_frames
    switches: list[dict] = []
    ball_xy = getattr(tracking, "ball", None)
    fusion_path = OUTPUT_DIR / "fusion.npz"
    if fusion_path.is_file():
        try:
            with np.load(fusion_path) as z:
                if "consensus_ball" in z:
                    ball_xy = np.asarray(z["consensus_ball"], dtype=np.float64)
        except Exception:
            pass
    for start, stop in halves:
        lo = max(0, int(start) - context)
        hi = min(n, int(stop) + context)
        features = extract_features(tracking, tags, start=lo, stop=hi)
        switches.extend(
            build_switch_records(
                pred,
                features,
                lo,
                tags,
                emission_log_probs(features, params),
                halves=[(int(start), int(stop))],
                events=events,
                whistle=whistle,
                ball_xy=ball_xy[lo:hi] if ball_xy is not None else None,
            )
        )
    switches.sort(key=lambda record: record["frame"])
    return {"fps": FPS, "switches": switches}
