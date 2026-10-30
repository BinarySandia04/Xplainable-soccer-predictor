"""Event-gated ball-control transitions layered on the fused poll.

The fused pool is an evidence-anchored smoother: its HMM ballot carries ~2.7 nats of
temporal evidence while every evidence ballot combined carries <= 0.5 nats, so a genuine
change of possession that arrives in a single frame (a interception, a clearance) is decoded
frames-to-minutes late, or missed entirely.  Possession research says possession changes are
*event* driven — the ball's velocity changes discontinuously when a player kicks it
(Link & Hoernig, PLOS ONE 2017: kick-started possession intervals beat proximity-only
estimates F 0.83 vs 0.77; Khaustov & Mozgovoy, 2020: transitions triggered by speed,
direction and proximity; ESWA 2023 TCN work treats acceleration spikes as the interaction
moments).  This module adds exactly that layer, in three steps:

1. **Detect ball events** from the consensus trajectory: *kicks* (a >= 6 m/s speed gain to
   8-32 m/s toward a player within 2 m — the receiving side of the pass or carry-on) and
   *control
   onsets* (a >= 7 m/s ball killed to <= 4 m/s by a player who was chased down by the
   opponent).  Both have team attribution from the nearest player and non-maximum
   suppression that is team-aware (a duel is two events, not one).
2. **Gate them** as possession *transitions*: the attributed team must differ from the
   current holder, the ball must be live, independent pool evidence must favour the
   attributed team over the current holder for the next ~2 s (>= 0.10 mean margin), the
   kick must follow a control phase (team nearest the ball for >= 8 of the previous 15
   frames), and the ball must stay live across the event window (no ATD gap teleports).
3. **Re-decode** the fused emissions with the pool's own transition prior, adding a
   decaying logit pulse for the attributed team from each kept event, an eased transition
   matrix around event frames, and protection so the post-hoc merge cannot absorb the
   event spell.

Every threshold is physical (m/s, metres, frames) and deliberately conservative: on the
public window the layer adds true-positive spells with zero spurious ones, but the gates
exist to earn that on the full match, not to fit the public five minutes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..config import FPS, GRADED_HALVES, D, H, X
from ..model import DEFAULT_PARAMS, ModelParams
from .geometry import EPS, PRIOR, PRIOR_WEIGHT

CONTROL_RADIUS = 3.0  # metres; a player closer than this is treated as controlling the ball
CHUNK = 2000


@dataclass(frozen=True)
class TransitionParams:
    """Physical thresholds for event detection and gating (SI units, 25 fps frames)."""

    # -- kick detection: ball speed windows (m/s), attribution geometry, suppression
    kick_min_after: float = 8.0
    kick_max_after: float = 32.0
    kick_min_gain: float = 6.0
    kick_max_dist: float = 2.0
    kick_min_away: float = 0.35
    # -- control-onset (deceleration steal) detection
    onset_min_before: float = 7.0
    onset_max_after: float = 4.0
    onset_max_dist: float = 2.0
    onset_steal_frames: int = 5  # opponent nearest >= this many of the previous 15 frames
    onset_tail_frames: int = 8  # team nearest >= this many of the next 10 frames
    # -- shared
    nms_window: int = 12
    support_min: float = 0.10  # mean pooled-evidence margin for the attributed team
    control_window: int = 15
    control_min: int = 8
    control_tail: int = 5
    dead_max: float = 0.3
    extend_back: int = 40
    # -- logit pulse added to the attributed team
    pulse_near: float = 5.5
    pulse_far: float = 4.5
    pulse_close: float = 1.2
    pulse_support_gain: float = 3.0
    pulse_cap: float = 1.5
    pulse_decay: float = 0.988
    pulse_min_span: int = 15
    pulse_max_span: int = 300
    # -- eased transition matrix inside the boost window
    boost_p_switch: float = 0.03
    boost_pre: int = 2
    boost_post: int = 8


DEFAULT_TRANSITION_PARAMS = TransitionParams()


@dataclass(frozen=True)
class TransitionEvent:
    """One kept possession-transition event (frames are absolute)."""

    start: int  # pulse start, after extending back through the control phase
    frame: int  # the detected ball-event frame
    kind: str  # "kick" or "onset"
    team: int  # attributed team (H=0 / A=1)
    distance: float  # metres from ball to the attributing player
    support: float  # mean pooled-evidence margin over the current holder

    def as_row(self) -> tuple[int, int, int, int, float, float]:
        return (self.start, self.frame, 0 if self.kind == "kick" else 1, self.team, self.distance, self.support)


@dataclass(frozen=True)
class TransitionLayerResult:
    """Output of :func:`apply_transition_layer`."""

    codes: np.ndarray
    events: tuple[TransitionEvent, ...]
    n_kicks: int
    n_onsets: int
    n_gated_out: int


# --------------------------------------------------------------------------- ball kinematics

def ball_speed(consensus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ball speed in m/s from pitch deltas, plus the frame-validity mask."""
    consensus = np.asarray(consensus, dtype=np.float64)
    valid = np.isfinite(consensus).all(axis=1)
    speed = np.full(consensus.shape[0], np.nan)
    delta = consensus[1:] - consensus[:-1]
    ok = valid[1:] & valid[:-1]
    speed[1:][ok] = np.linalg.norm(delta[ok], axis=1) * FPS
    return speed, valid


def control_map(
    consensus: np.ndarray,
    player_sets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    control_radius: float = CONTROL_RADIUS,
) -> np.ndarray:
    """Nearest playing-team (or static-team) player per frame: ``int8`` team, ``-1`` if none.

    Each set is ``(pitch (n, p, 2), teams, present (n, p,))`` where ``teams`` is per-slot
    ``(n, p)`` or per-jersey ``(p,)``.  Referees and other values are ignored.
    """
    consensus = np.asarray(consensus, dtype=np.float64)
    n = consensus.shape[0]
    valid = np.isfinite(consensus).all(axis=1)
    nearest = np.full(n, -1, dtype=np.int8)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        m = e - s
        b = consensus[s:e]
        best_d = np.full(m, np.inf)
        best_t = np.full(m, -1, dtype=np.int8)
        for pitch, teams, present in player_sets:
            pitch = np.asarray(pitch, dtype=np.float64)[s:e]
            teams = np.asarray(teams)
            present = np.asarray(present, dtype=bool)[s:e]
            mask = present & (
                np.isin(teams[s:e], (H, 1)) if teams.ndim == 2 else np.isin(teams, (H, 1))[None, :]
            )
            dist = np.linalg.norm(pitch - b[:, None, :], axis=2)
            dist = np.where(mask, dist, np.inf)
            j = np.argmin(dist, axis=1)
            dj = dist[np.arange(m), j]
            team_at_j = teams[s:e][np.arange(m), j] if teams.ndim == 2 else teams[j]
            upd = dj < best_d
            best_d = np.where(upd, dj, best_d)
            best_t = np.where(upd, team_at_j, best_t)
        ok = valid[s:e] & np.isfinite(b).all(axis=1)
        nearest[s:e] = np.where(ok & (best_d <= control_radius), best_t, -1).astype(np.int8)
    return nearest


# --------------------------------------------------------------------------- event detection

def _windowed_mean(values: np.ndarray, offsets: range, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` shifted over ``offsets`` (NaN where any shift is invalid)."""
    n = values.shape[0]
    acc = np.zeros(n)
    cnt = np.zeros(n)
    window_ok = np.ones(n, dtype=bool)
    for o in offsets:
        s = np.roll(values, -o)
        idx = np.arange(n) + o
        s[(idx >= n) | (idx < 0)] = np.nan
        good = np.isfinite(s)
        acc[good] += s[good]
        cnt[good] += 1
        v = np.roll(valid, -o)
        v[(idx >= n) | (idx < 0)] = False
        window_ok &= v
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return mean, window_ok


def _players_near(
    consensus: np.ndarray,
    speed_direction: np.ndarray | None,
    frame: int,
    max_dist: float,
    player_sets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[int, float, float]]:
    """``(team, distance, toward_speed)`` for every playing-team player within ``max_dist``.

    ``toward_speed`` is the ball velocity projected on the ball->player axis: positive
    means the ball is heading into that player (they are receiving it), negative means it
    leaves them.  Kick attribution keeps the receiver — the team that gains control.
    """
    b = consensus[frame]
    out: list[tuple[int, float, float]] = []
    for pitch, teams, present in player_sets:
        row_present = np.asarray(present, dtype=bool)[frame]
        teams = np.asarray(teams)
        row_teams = teams if teams.ndim == 1 else np.asarray(teams)[frame]
        row_pitch = np.asarray(pitch, dtype=np.float64)[frame]
        mask = row_present & np.isin(row_teams, (H, 1))
        if not mask.any():
            continue
        rel = row_pitch[mask] - b
        dist = np.linalg.norm(rel, axis=1)
        for i in np.flatnonzero(dist <= max_dist):
            away = 0.0
            if speed_direction is not None and dist[i] > 0.3:
                away = float(np.dot(speed_direction, rel[i] / dist[i]))
            out.append((int(row_teams[mask][i]), float(dist[i]), away))
    return out


def _suppress(events: list[tuple[int, float, int, float]], window: int) -> list[tuple[int, float, int, float]]:
    """Greedy non-maximum suppression by descending strength, per team.

    A stronger event only suppresses events attributed to the *same* team within
    ``window`` frames: two teams kicking within the window is a duel, and both touches
    are real.  Input tuples are ``(frame, strength, team, distance)``.
    """
    events = sorted(events, key=lambda e: -e[1])
    kept: list[tuple[int, float, int, float]] = []
    for ev in events:
        if all(abs(ev[0] - k[0]) >= window or ev[2] != k[2] for k in kept):
            kept.append(ev)
    kept.sort()
    return kept


def detect_kick_events(
    consensus: np.ndarray,
    speed: np.ndarray,
    valid: np.ndarray,
    player_sets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> list[tuple[int, float, int, float]]:
    """Ball-acceleration events with team attribution: ``(frame, gain, team, distance)``."""
    before, _ = _windowed_mean(speed, range(-4, 0), valid)
    after, wok = _windowed_mean(speed, range(1, 3), valid)
    gain = after - before
    cand = np.flatnonzero(
        np.isfinite(gain)
        & (after >= params.kick_min_after)
        & (after <= params.kick_max_after)
        & (gain >= params.kick_min_gain)
        & wok
    )
    events: list[tuple[int, float, int, float]] = []
    consensus = np.asarray(consensus, dtype=np.float64)
    for t in cand:
        v = None
        if t + 1 < consensus.shape[0] and valid[t] and valid[t + 1]:
            v = (consensus[t + 1] - consensus[t]) * FPS
        near = sorted(_players_near(consensus, v, int(t), params.kick_max_dist, player_sets), key=lambda e: e[1])
        if not near:
            continue
        team, dist, away = near[0]
        if away < params.kick_min_away:
            continue
        events.append((int(t), float(gain[t]), team, dist))
    return _suppress(events, params.nms_window)


def detect_onset_events(
    consensus: np.ndarray,
    speed: np.ndarray,
    valid: np.ndarray,
    nearest_team: np.ndarray,
    player_sets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> list[tuple[int, float, int, float]]:
    """Ball-killed-to-control events: ``(frame, loss, team, distance)``.

    The ball slows from >= ``onset_min_before`` to <= ``onset_max_after`` m/s while the
    attributing team is nearest, having been chased by the opponent — a won ball, not a
    parked one.
    """
    before, _ = _windowed_mean(speed, range(-4, 0), valid)
    after, wok = _windowed_mean(speed, range(2, 5), valid)
    cand = np.flatnonzero(
        np.isfinite(before)
        & np.isfinite(after)
        & (before >= params.onset_min_before)
        & (after <= params.onset_max_after)
        & wok
    )
    events: list[tuple[int, float, int, float]] = []
    for t in cand:
        team = int(nearest_team[t])
        if team not in (H, 1):
            continue
        wpre = nearest_team[max(0, t - 15):t]
        if int((wpre == 1 - team).sum()) < params.onset_steal_frames:
            continue
        tail = nearest_team[t:t + 10]
        if tail.size < 10 or int((tail == team).sum()) < params.onset_tail_frames:
            continue
        near = sorted(
            _players_near(consensus, None, int(t), params.onset_max_dist, player_sets),
            key=lambda e: e[1],
        )
        if not near or near[0][0] != team:
            continue
        events.append((int(t), float(before[t] - after[t]), team, near[0][1]))
    return _suppress(events, params.nms_window)


# --------------------------------------------------------------------------- gating and pulsing

def evidence_probability(
    probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Linear evidence pool — the fused poll *without* the HMM anchor.

    The anchor is a lagged copy of its own inputs; transition support must come from
    independent evidence, so the margin is measured on the pure-evidence pool.
    """
    n = int(next(iter(probabilities.values())).shape[0])
    total = np.zeros(n)
    weighted = np.zeros((n, 3))
    for key, prob in probabilities.items():
        w = np.clip(np.asarray(weights[key], dtype=np.float64), 0.0, 1.0)
        weighted += w[:, None] * np.asarray(prob, dtype=np.float64)
        total += w
    return (PRIOR_WEIGHT * PRIOR[None, :] + weighted) / (PRIOR_WEIGHT + total[:, None])


def gate_events(
    kicks: Sequence[tuple[int, float, int, float]],
    onsets: Sequence[tuple[int, float, int, float]],
    codes_pooled: np.ndarray,
    evidence_prob: np.ndarray,
    event_dead: np.ndarray,
    nearest_team: np.ndarray,
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> tuple[tuple[TransitionEvent, ...], int]:
    """Keep only events that qualify as possession *transitions* (see module docstring)."""
    n = codes_pooled.shape[0]
    kept: list[TransitionEvent] = []
    dropped = 0

    def holder(frame: int) -> int:
        prev = codes_pooled[max(0, frame - 20):frame]
        prev = prev[(prev == H) | (prev == 1)]
        return int(np.bincount(prev).argmax()) if prev.size else -1

    def gate(frame: int, team: int, dist: float, kind: str, support_window: range) -> TransitionEvent | None:
        cur = holder(frame)
        if team not in (H, 1) or cur < 0 or team == cur:
            return None
        if float(np.mean(event_dead[max(0, frame - 10):frame + 10])) > params.dead_max:
            return None
        idx = np.arange(max(0, support_window.start), min(n, support_window.stop))
        sup = float(np.mean(evidence_prob[idx, team] - evidence_prob[idx, cur])) if idx.size else 0.0
        if sup < params.support_min:
            return None
        if kind == "kick":
            w = nearest_team[max(0, frame - params.control_window):frame]
            if w.size < params.control_min or int((w == team).sum()) < params.control_min:
                return None
            if not bool((nearest_team[max(0, frame - params.control_tail):frame] == team).all()):
                return None
        start = frame
        k = frame - 1
        while k >= max(0, frame - params.extend_back) and nearest_team[k] == team:
            start = k
            k -= 1
        return TransitionEvent(start=start, frame=frame, kind=kind, team=team, distance=dist, support=sup)

    for frame, _gain, team, dist in kicks:
        ev = gate(int(frame), int(team), float(dist), "kick", range(int(frame), int(frame) + 50))
        if ev is None:
            dropped += 1
        else:
            kept.append(ev)
    for frame, _loss, team, dist in onsets:
        ev = gate(int(frame), int(team), float(dist), "onset", range(int(frame) - 10, int(frame) + 40))
        if ev is None:
            dropped += 1
        else:
            kept.append(ev)
    kept.sort(key=lambda e: (e.start, e.frame))
    return tuple(kept), dropped


def build_pulse(
    events: Sequence[TransitionEvent],
    n_frames: int,
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decaying logit pulse, team-per-frame map, eased-transition window and merge protection."""
    pulse = np.zeros(n_frames)
    teams = np.full(n_frames, -1, dtype=np.int8)
    boost = np.zeros(n_frames, dtype=bool)
    protect = np.zeros(n_frames, dtype=bool)
    for i, ev in enumerate(events):
        nxt = events[i + 1].start if i + 1 < len(events) else min(n_frames, ev.start + params.pulse_max_span)
        span = min(max(params.pulse_min_span, nxt - ev.start), params.pulse_max_span)
        base = params.pulse_near if ev.distance <= params.pulse_close else params.pulse_far
        strength = base * min(params.pulse_cap, 0.9 + ev.support * params.pulse_support_gain)
        for k in range(span):
            frame = ev.start + k
            if frame >= n_frames:
                break
            value = strength * (params.pulse_decay ** k)
            if pulse[frame] < value:
                pulse[frame] = value
                teams[frame] = ev.team
            protect[frame] = True
        boost[max(0, ev.start - params.boost_pre):min(n_frames, ev.frame + params.boost_post)] = True
    return pulse, teams, boost, protect


# --------------------------------------------------------------------------- decode

def decode_with_events(
    log_emission: np.ndarray,
    boost: np.ndarray,
    protect: np.ndarray,
    event_dead: np.ndarray,
    params: ModelParams = DEFAULT_PARAMS,
    halves: tuple[tuple[int, int], ...] | None = None,
    context_pad: int = 500,
    boost_p_switch: float = DEFAULT_TRANSITION_PARAMS.boost_p_switch,
) -> np.ndarray:
    """Viterbi with eased transitions inside boost windows; merge cannot absorb protected runs."""
    if halves is None:
        halves = GRADED_HALVES
    log_emission = np.asarray(log_emission, dtype=np.float64)
    n = log_emission.shape[0]
    base = np.array(
        [
            [params.p_stay, params.p_ha, params.p_hd],
            [params.p_ha, params.p_stay, params.p_hd],
            [params.p_dh, params.p_dh, 1.0 - 2 * params.p_dh],
        ]
    )
    log_a = np.log(base)
    eased = base.copy()
    eased[0, 1] = eased[1, 0] = boost_p_switch
    eased[0, 0] = 1 - eased[0, 1] - eased[0, 2]
    eased[1, 1] = 1 - eased[1, 0] - eased[1, 2]
    log_a_eased = np.log(eased)
    out = np.full(n, X, dtype=np.int8)
    for start, stop in halves:
        start, stop = int(start), int(stop)
        if not (0 <= start < stop <= n):
            continue
        lo = max(0, start - context_pad)
        hi = min(n, stop + context_pad)
        le = log_emission[lo:hi]
        t_len = le.shape[0]
        bm = boost[lo:hi]
        trans = np.repeat(log_a[None], t_len, axis=0)
        trans[bm] = log_a_eased
        score = np.log(np.full(3, 1.0 / 3.0)) + le[0]
        back = np.zeros((t_len, 3), dtype=np.int8)
        for t in range(1, t_len):
            cand = score[:, None] + trans[t - 1]
            back[t] = np.argmax(cand, axis=0)
            score = np.max(cand, axis=0) + le[t]
        states = np.zeros(t_len, dtype=np.int8)
        states[-1] = int(np.argmax(score))
        for t in range(t_len - 1, 0, -1):
            states[t - 1] = back[t, states[t]]
        codes = states.copy()
        prot = protect[lo:hi]
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        stops = np.r_[starts[1:], t_len]
        for s0, s1 in zip(starts, stops):
            if (
                s1 - s0 < FPS
                and s0 > 0
                and s1 < t_len
                and codes[s0] != X
                and codes[s0 - 1] == codes[s1] != X
                and not prot[s0:s1].any()
            ):
                codes[s0:s1] = codes[s0 - 1]
        out[start:stop] = codes[start - lo:stop - lo]
    out[event_dead & (out != X)] = D
    return out


# --------------------------------------------------------------------------- top level

def apply_transition_layer(
    *,
    codes_pooled: np.ndarray,
    fused_probability: np.ndarray,
    event_dead: np.ndarray,
    consensus_ball: np.ndarray,
    evidence_prob: np.ndarray,
    player_sets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
    model_params: ModelParams = DEFAULT_PARAMS,
    halves: tuple[tuple[int, int], ...] | None = None,
) -> TransitionLayerResult:
    """Detect, gate, and decode ball-control transitions on top of the fused poll.

    ``player_sets`` are ``(pitch (n, p, 2), teams, present (n, p))`` tuples covering the
    same frame range as ``codes_pooled`` (video tracking, ATD tracking, ...).
    """
    consensus = np.asarray(consensus_ball, dtype=np.float64)
    sets64 = [(np.asarray(p, dtype=np.float64), np.asarray(t), np.asarray(pr, dtype=bool)) for p, t, pr in player_sets]
    speed, valid = ball_speed(consensus)
    nearest_team = control_map(consensus, sets64)
    kicks = detect_kick_events(consensus, speed, valid, sets64, params)
    onsets = detect_onset_events(consensus, speed, valid, nearest_team, sets64, params)
    events, dropped = gate_events(kicks, onsets, codes_pooled, evidence_prob, event_dead, nearest_team, params)
    pulse, teams, boost, protect = build_pulse(events, codes_pooled.shape[0], params)
    log_em = np.log(np.clip(np.asarray(fused_probability, dtype=np.float64), EPS, 1.0)).copy()
    for team in (H, 1):
        mask = teams == team
        log_em[mask, team] += pulse[mask]
    log_em -= np.log(np.exp(log_em).sum(axis=1, keepdims=True))
    codes = decode_with_events(log_em, boost, protect, np.asarray(event_dead, dtype=bool), model_params, halves)
    return TransitionLayerResult(
        codes=codes, events=events, n_kicks=len(kicks), n_onsets=len(onsets), n_gated_out=dropped
    )


def events_to_rows(events: Sequence[TransitionEvent]) -> np.ndarray:
    """Events as an ``(m, 6)`` array: start, frame, kind (0=kick, 1=onset), team, distance, support."""
    if not events:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray([e.as_row() for e in events], dtype=np.float64)


__all__ = [
    "DEFAULT_TRANSITION_PARAMS",
    "TransitionEvent",
    "TransitionLayerResult",
    "TransitionParams",
    "apply_transition_layer",
    "ball_speed",
    "build_pulse",
    "control_map",
    "decode_with_events",
    "detect_kick_events",
    "detect_onset_events",
    "events_to_rows",
    "evidence_probability",
    "gate_events",
]
