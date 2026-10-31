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
    # -- pixel box-contact channel (dedicated YOLO ball inside YOLO player boxes)
    contact_run: int = 3  # consecutive frames with the ball pixel inside a team's box
    contact_novel: int = 4  # frames before the run during which the team must not have
    # touched: a fresh takeover, not continued juggling
    contact_dominance: int = 2  # contain-frame margin over the other team inside the run
    # (crowds share the image: boxes of far players swallow the ball pixel, but never for
    # more than a frame or two of a genuine take)
    contact_speed_max: float = 12.0  # m/s; control means a slow ball, not a deflection
    kicker_slow_max: float = 7.0  # m/s at the close frame: the ball starts slow ...
    kicker_leave_speed: float = 9.0  # ... and reaches this within 3 frames: released
    kicker_release_max: float = 35.0  # above this the track jumped: artifact, not a kick
    kicker_close_px: float = 20.0  # ball-to-nearest-box-foot distance that counts as feet
    kicker_dominance_px: float = 25.0  # the other team's nearest foot must be this further
    kicker_leave_px: float = 12.0  # growth of ball-to-box distance over 3 frames: it left
    kicker_touch_window: int = 3  # team touch band must fire within [t-2, t+3] frames
    contact_pulse: float = 3.2  # weaker logit than kinematic pulses: pixel events correct
    contact_pulse_solid: float = 3.8  # ...the *timing* of the anchor, not fight it for long
    contact_solid_support: float = 0.6  # support fraction that counts as solid control
    # -- pixel intervals: pixel evidence only corrects timing near possession boundaries;
    # settled spells belong to the pitch geometry (box overlap in depth looks identical)
    pixel_bound_window: int = 130  # fire only this close to a fused-code boundary
    pixel_momentum_horizon: int = 150  # frames in which fused evidence must catch up ...
    pixel_momentum_min: float = 0.5  # ... to this probability for the event's team
    pixel_fingerprint_close: float = 16.0  # win-and-clear kicks pass without catch-up:
    pixel_fingerprint_dom: float = 40.0  # released from feet nobody else was near ...
    pixel_fingerprint_release_max: float = 15.0  # ... a short clearance, not a punt
    pixel_cap: int = 75  # owned interval: end at fused boundary or this cap ...
    pixel_cap_solid: int = 250  # ... whichever first; solid contacts may run longer
    pixel_solid_run: int = 5  # contact run length that counts as solid
    pixel_commit_frames: int = 40  # fused code must stay this team for this many frames ...
    pixel_commit_prob: float = 0.7  # ... at this probability: then extend the interval to the boundary
    pixel_commit_gap: int = 90  # but only when that boundary is this close: the pitch is just slow, not lost
    pixel_bias: float = 3.2  # sustained logit over an owned interval
    # -- cross-stream relay: a camera-attested takeover *inside* a settled spell.
    # the sticky prior is memory, and memory is right until it loses the ball out
    # of sight: it may only be outvoted mid-spell when every independent
    # ball/control voter asserts the other team at once, both ball streams pin the
    # ball to the same single player's box, and the prior team itself never shows
    # the ball on camera across the lookback window
    relay_min_prob: float = 0.8  # each of the four ballot voters' probability for the event team
    relay_sole_frac: float = 0.75  # run frames where both streams' sole container is the same player
    relay_px_max: float = 8.0  # median ATD-to-YOLO ball distance over the run: one agreeing camera
    relay_anchor_window: int = 120  # lookback frames for the possession-dominance comparison
    relay_anchor_max: int = 10  # holder's ball-in-box frames in the window: at most this many ...
    relay_anchor_ratio: int = 4  # ... and this many times fewer than the event team's
    # -- keeper hold: a player of the team that defends a goal keeping the ball inside his
    # own penalty area *is* possession from the catch.  The sticky prior reads a hold as one
    # more loose-ball window and the contact channel needs a fresh takeover, which a held
    # ball never shows (measured 19 855-19 890: Barcelona's keeper held the ball a few tens
    # of pixels inside his box while the spell read Athletic).  0 disables the channel.
    keeper_hold_min_frames: int = 30  # a hold this long is a catch, not a touch
    keeper_hold_px: float = 48.0  # ball pixel to holder box centre: held, not passing by
    keeper_hold_speed_px: float = 8.0  # median ball-pixel speed across the hold (px/frame)
    keeper_hold_depth_m: float = 18.5  # goal line to box edge (16.5 m + homography slack)
    keeper_hold_half_width_m: float = 21.0  # penalty-area half-width 20.16 m + slack
    keeper_hold_move_m: float = 8.0  # holder stays this near its hold median (detector id swaps)
    keeper_hold_opp_frac: float = 0.25  # max share of frames with an opponent box nearer
    keeper_hold_anchor_contain_max: int = 5  # opponent ball-in-box frames allowed across the hold


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
        kind_code = {"kick": 0, "onset": 1, "contact": 2, "kicker": 3, "hold": 4}[self.kind]
        return (self.start, self.frame, kind_code, self.team, self.distance, self.support)


@dataclass(frozen=True)
class TransitionLayerResult:
    """Output of :func:`apply_transition_layer`."""

    codes: np.ndarray
    events: tuple[TransitionEvent, ...]
    n_kicks: int
    n_onsets: int
    n_gated_out: int
    n_contact: int = 0
    n_kicker: int = 0
    n_hold: int = 0


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


# --------------------------------------------------------------------------- pixel contact evidence

@dataclass(frozen=True)
class BoxEvidence:
    """Dedicated-YOLO pixel evidence: player boxes and the observed ball pixel.

    All arrays cover the same frame range as the fusion cache.  ``ball_yolo`` is the
    highest-confidence ball detection (may be NaN when the model saw no ball),
    ``ball_atd`` the tracked pitch-projection pixel; the contact channel trusts the YOLO
    detection first and falls back to the ATD pixel.
    """

    boxes: np.ndarray  # (n, p, 4) float, xyxy in image pixels
    team: np.ndarray  # (n, p) int, H=0 / A=1 (other ids are ignored)
    present: np.ndarray  # (n, p) bool
    ball_yolo: np.ndarray  # (n, 2) float pixel, NaN where absent
    ball_yolo_present: np.ndarray  # (n,) bool
    ball_atd: np.ndarray  # (n, 2) float pixel, NaN where absent
    pitch: np.ndarray | None = None  # (n, p, 2) float metres, keeper-ness for the hold channel
    track: np.ndarray | None = None  # (n, p) int detector track ids (identity across occlusion)


def _contact_bits(
    boxes: np.ndarray,
    teams: np.ndarray,
    present: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit flags (1 = H, 2 = A) for one ball pixel: strict box ``contain`` and leg-band
    ``touch`` — the box's horizontal span widened to ``0.8`` of its width, from the top of
    the shins (``0.5`` of box height above the feet line) to ``0.4`` below the feet line,
    where a ball at a player's feet or just in front of them sits at the same ground line."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    w = np.maximum(x2 - x1, 1.0)
    h = np.maximum(y2 - y1, 1.0)
    ok = present & ((teams == H) | (teams == 1))
    inside = ok & (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
    band = ok & (np.abs(px - (x1 + x2) / 2) <= 0.8 * w) & (py >= y2 - 0.5 * h) & (py <= y2 + 0.4 * h)
    contain = np.zeros(px.shape[:-1], dtype=np.int8)
    touch = np.zeros(px.shape[:-1], dtype=np.int8)
    for team, bit in ((H, 1), (1, 2)):
        sel = teams == team
        contain |= np.where((inside & sel).any(axis=-1), bit, 0).astype(np.int8)
        touch |= np.where((band & sel).any(axis=-1), bit, 0).astype(np.int8)
    return contain, touch


def pixel_contact(evidence: BoxEvidence) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame team bit flags (bit 0 = H, bit 1 = A) for the observed ball pixel.

    The ball pixel is one YOLO-first stream: the dedicated ball detector when it sees the
    ball, the ATD projection otherwise.  A general track's ball pixel rides nearby players
    and invents takeovers that never happened (measured 20 955: the ATD pixel sat 5-7 m
    off the ball on an Athletic player for two seconds while YOLO held the ball on a
    Barcelona player in 69 of 71 frames).  One continuous trajectory keeps a genuine
    ball-in-box accumulating its run, and isolated frames the window rule filters out.
    """
    boxes = np.asarray(evidence.boxes, dtype=np.float64)
    teams = np.asarray(evidence.team)
    present = np.asarray(evidence.present, dtype=bool)
    atd = np.asarray(evidence.ball_atd, dtype=np.float64)
    yolo = np.asarray(evidence.ball_yolo, dtype=np.float64)
    y_seen = np.asarray(evidence.ball_yolo_present, dtype=bool) & np.isfinite(yolo).all(axis=1)
    ball = np.where(y_seen[:, None], yolo, atd)
    contain = np.zeros(boxes.shape[0], dtype=np.int8)
    touch = np.zeros(boxes.shape[0], dtype=np.int8)
    for s in range(0, boxes.shape[0], CHUNK):
        e = min(boxes.shape[0], s + CHUNK)
        b = ball[s:e]
        pc, pt = _contact_bits(
            boxes[s:e], teams[s:e], present[s:e],
            np.nan_to_num(b[:, 0], nan=-1.0e9)[:, None], np.nan_to_num(b[:, 1], nan=-1.0e9)[:, None],
        )
        keep = np.isfinite(b).all(axis=1)
        contain[s:e] = np.where(keep, pc, 0)
        touch[s:e] = np.where(keep, pt, 0)
    return contain, touch


def detect_contact_events(
    contain: np.ndarray,
    touch: np.ndarray,
    speed: np.ndarray,
    event_dead: np.ndarray,
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> list[tuple[int, float, int, float]]:
    """Fresh pixel-control takeovers: ``(run_end_frame, run_length, team, 1.0)``.

    A team claims control when the ball pixel sits *consecutively* inside its boxes for
    ``contact_run`` frames (the run ends: it held the ball that long and then played it),
    it dominates the run (the other team contains the ball in at least
    ``contact_dominance`` frames fewer — in a crowd the ball pixel sits in far players'
    boxes too, but not persistently for the wrong side), it had *no* touch in the
    ``contact_novel`` frames before the run (a fresh takeover, not continued juggling),
    the ball is slow, and play is live.
    """
    n = contain.shape[0]
    own_bits = {1: (np.asarray(contain) & 1) > 0, 2: (np.asarray(contain) & 2) > 0}
    touch_bits = {1: (np.asarray(touch) & 1) > 0, 2: (np.asarray(touch) & 2) > 0}
    events: list[tuple[int, float, int, float]] = []
    idx = np.arange(n)
    for team, bit in ((H, 1), (1, 2)):
        own = own_bits[bit]
        last_off = np.maximum.accumulate(np.where(own, -1, idx))
        run = np.where(own, idx - last_off, 0).astype(np.int32)
        oth = own_bits[3 - bit].astype(np.int32)
        own_t = touch_bits[bit].astype(np.int32)
        cs = np.concatenate(([0], np.cumsum(oth)))
        ts = np.concatenate(([0], np.cumsum(own_t)))
        cand = np.flatnonzero((run >= params.contact_run) & (np.r_[own[1:], False] == 0))
        cand = cand[np.isfinite(speed[cand]) & (speed[cand] <= params.contact_speed_max)]
        for t in cand:
            t = int(t)
            rs = t - int(run[t]) + 1
            if int(run[t]) - int(cs[t + 1] - cs[rs]) < params.contact_dominance:
                continue
            if int(ts[rs] - ts[max(0, rs - params.contact_novel)]) > 0:
                continue
            if float(np.mean(np.asarray(event_dead)[max(0, t - 10):t + 10])) > params.dead_max:
                continue
            events.append((t, float(run[t]), team, 1.0))
    return _suppress(events, params.nms_window)


def _box_foot_distance(evidence: BoxEvidence) -> np.ndarray:
    """Min pixel distance from the ball (YOLO detection preferred while it sees the ball,
    else the ATD track — one continuous trajectory for release direction) to each team's
    box foot-centre.  ``(n, 2)`` float array, column 0 = team H, column 1 = team A."""
    boxes = np.asarray(evidence.boxes, dtype=np.float64)
    teams = np.asarray(evidence.team)
    present = np.asarray(evidence.present, dtype=bool)
    atd = np.asarray(evidence.ball_atd, dtype=np.float64)
    yolo = np.asarray(evidence.ball_yolo, dtype=np.float64)
    y_seen = np.asarray(evidence.ball_yolo_present, dtype=bool) & np.isfinite(yolo).all(axis=1)
    ball = np.where(y_seen[:, None], yolo, atd)
    n = boxes.shape[0]
    out = np.full((n, 2), np.inf)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        b = boxes[s:e]
        cx = (b[..., 0] + b[..., 2]) / 2
        cy = b[..., 3]
        ok = present[s:e] & np.isin(teams[s:e], (H, 1))
        d = np.linalg.norm(ball[s:e, None, :] - np.stack([cx, cy], axis=-1), axis=2)
        for col, team in ((0, H), (1, 1)):
            dm = np.where(ok & (teams[s:e] == team), d, np.inf)
            out[s:e, col] = dm.min(axis=1)
    return out


def detect_kicker_events(
    evidence: BoxEvidence,
    touch: np.ndarray,
    speed: np.ndarray,
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> list[tuple[int, float, int, float, float]]:
    """Win-and-clear kicks: ``(frame, release_speed, team, close_px, dominance_px)``.

    A pixel-only kick signature the receiver-based :func:`detect_kick_events` misses:
    the ball is slow at one team's feet (nearest box foot within ``kicker_close_px``,
    ``speed <= kicker_slow_max``), that team clearly owns the ball there (the other
    team's nearest foot is ``kicker_dominance_px`` further away), the team's touch band
    fires within ``[-2, kicker_touch_window]`` frames of the release, and the ball is
    then put in flight (reaches ``kicker_leave_speed`` within 3 frames while the
    distance to that team's boxes grows by ``kicker_leave_px``).  Attribution goes to
    the side that released the ball even when the homography track shows no player near
    it — the winner touches the ball before any receiver enters the pitch attribution
    radius.
    """
    dist = _box_foot_distance(evidence)
    n = speed.shape[0]
    events: list[tuple[int, float, int, float, float]] = []
    tail_max = np.full(n, np.nan)
    for o in (1, 2, 3):
        s = np.full(n, np.nan)
        s[:-o] = speed[o:]
        tail_max = np.fmax(tail_max, s)
    slow = np.isfinite(speed) & (speed <= params.kicker_slow_max)
    release = np.isfinite(tail_max) & (tail_max >= params.kicker_leave_speed)
    touch_arr = np.asarray(touch)
    for team, col, bit in ((H, 0, 1), (1, 1, 2)):
        d = dist[:, col]
        close = d <= params.kicker_close_px
        dominate = dist[:, 1 - col] >= d + params.kicker_dominance_px
        leaving = np.full(n, -np.inf)
        with np.errstate(invalid="ignore"):
            leaving[:-3] = d[3:] - d[:-3]
        cs = np.concatenate(([0], np.cumsum(((touch_arr & bit) > 0).astype(np.int32))))
        hi = np.minimum(np.arange(n) + params.kicker_touch_window + 1, n)
        lo = np.maximum(np.arange(n) - 2, 0)
        coupled = (cs[hi] - cs[lo]) > 0
        cand = np.flatnonzero(slow & close & dominate & release & (leaving >= params.kicker_leave_px) & coupled)
        for t in cand:
            events.append((int(t), float(tail_max[t]), team, float(d[t]), float(dist[t, 1 - col] - d[t])))
    return _suppress(events, params.nms_window)


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


def _suppress(events: list[tuple], window: int) -> list[tuple]:
    """Greedy non-maximum suppression by descending strength, per team.

    A stronger event only suppresses events attributed to the *same* team within
    ``window`` frames: two teams kicking within the window is a duel, and both touches
    are real.  Input tuples start ``(frame, strength, team, ...)``.
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


RELAY_VOTERS = ("atd_ball", "video_ball", "yolo_ball", "video_players")


def relay_contacts(
    contact: Sequence[tuple],
    contain: np.ndarray,
    evidence: BoxEvidence,
    codes_pooled: np.ndarray,
    ballots: Mapping[str, tuple[np.ndarray, np.ndarray]],
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> frozenset[tuple[int, int]]:
    """Contact events that may relay across a settled spell (``relay_*`` params).

    Returns the ``(frame, team)`` keys of contact candidates that are camera
    testimony rather than depth noise: every independent ballot voter (ATD ball,
    video ball, YOLO ball, video players) puts the event team at
    ``relay_min_prob`` at the run end, both ball streams pin the ball to one and
    the same single containing player across the run with median distance
    ``relay_px_max``, and the spell's team has at most ``relay_anchor_max`` of its
    own ball-in-box frames in the last ``relay_anchor_window`` -- and a
    ``relay_anchor_ratio``-fold minority of the event team's.  The prior is a
    lagged copy of its own inputs: it cannot outvote four agreeing sensors plus a
    camera that repeatedly sees the ball held and never sees *its* team touch it.
    """
    if not all(key in ballots for key in RELAY_VOTERS):
        return frozenset()
    probs = np.stack([np.asarray(ballots[key][0], dtype=np.float64) for key in RELAY_VOTERS])
    weights = np.stack(
        [np.clip(np.asarray(ballots[key][1], dtype=np.float64), 0.0, 1.0) for key in RELAY_VOTERS]
    )
    unanim_team = probs.argmax(axis=2)[0]
    unanim = ((probs.max(axis=2) >= params.relay_min_prob) & (weights > 0)).all(axis=0)
    unanim &= (probs.argmax(axis=2) == unanim_team[None, :]).all(axis=0)

    boxes = np.asarray(evidence.boxes, dtype=np.float64)
    present = np.asarray(evidence.present, dtype=bool)
    atd = np.asarray(evidence.ball_atd, dtype=np.float64)
    yolo = np.asarray(evidence.ball_yolo, dtype=np.float64)
    a_ok = np.isfinite(atd).all(axis=1)
    y_ok = np.asarray(evidence.ball_yolo_present, dtype=bool) & np.isfinite(yolo).all(axis=1)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    n = boxes.shape[0]
    same_player = np.zeros(n, dtype=bool)
    dist = np.full(n, np.inf)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        dist[s:e] = np.linalg.norm(atd[s:e] - yolo[s:e], axis=1)
        sole: dict[str, np.ndarray] = {}
        for name, ball, seen in (("a", atd, a_ok), ("y", yolo, y_ok)):
            px, py = ball[s:e, 0], ball[s:e, 1]
            ins = (present[s:e] & seen[s:e, None]
                   & (px[:, None] >= x1[s:e]) & (px[:, None] <= x2[s:e])
                   & (py[:, None] >= y1[s:e]) & (py[:, None] <= y2[s:e]))
            cnt = ins.sum(axis=1)
            sole[name] = np.where(cnt == 1, ins.argmax(axis=1), -1)
        same_player[s:e] = (sole["a"] >= 0) & (sole["a"] == sole["y"])

    codes = np.asarray(codes_pooled)
    contain_arr = np.asarray(contain)
    bits = {H: 1, 1: 2}
    out: set[tuple[int, int]] = set()
    for ev in contact:
        frame, runlen, team = int(ev[0]), int(ev[1]), int(ev[2])
        if team not in (H, 1) or runlen < params.pixel_solid_run:
            continue
        if not bool(unanim[frame]) or int(unanim_team[frame]) != team:
            continue
        rs = frame - runlen + 1
        if float(same_player[rs:frame + 1].mean()) < params.relay_sole_frac:
            continue
        seen = a_ok[rs:frame + 1] & y_ok[rs:frame + 1]
        if int(seen.sum()) < min(runlen, 5):
            continue
        if float(np.median(dist[rs:frame + 1][seen])) > params.relay_px_max:
            continue
        cur = codes[max(0, frame - 20):frame]
        cur = cur[(cur == H) | (cur == 1)]
        if cur.size == 0:
            continue  # relays correct possession spells, never a dead ball
        anchor = int(np.bincount(cur).argmax())
        window = contain_arr[max(0, frame - params.relay_anchor_window):frame + 1]
        team_c = int(((window & bits[team]) > 0).sum())
        anchor_c = int(((window & bits[anchor]) > 0).sum())
        if anchor_c > params.relay_anchor_max or anchor_c * params.relay_anchor_ratio > team_c:
            continue
        out.add((frame, team))
    return frozenset(out)


@dataclass(frozen=True)
class KeeperHold:
    """One keeper-hold run: ball held by a defending-side player inside his own area."""

    start: int  # first hold frame (the catch)
    frame: int  # last hold frame
    team: int  # holder's team (H=0 / A=1)
    length: int  # frames held


def keeper_holds(
    evidence: BoxEvidence,
    contain: np.ndarray,
    event_dead: np.ndarray,
    defenders: np.ndarray,
    pitch: tuple[float, float],
    halves: tuple[tuple[int, int], ...],
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> list[KeeperHold]:
    """Ball-holds by the side that defends the goal the ball lies at (``keeper_hold_*``).

    ``defenders[h, g]`` (the keeper roster geometry of :mod:`possession_lab.keepers`) says
    which team defends goal ``g`` in half ``h``.  A hold is a run of frames where the
    YOLO-first ball pixel sits within ``keeper_hold_px`` of one present player's box, that
    player is on the defending side and inside its penalty area, and the ball is slow; the
    run then also passes opponent-nearness, opponent-containment and dead-ball limits.
    Detector ids swap during occlusion, so positional steadiness (``keeper_hold_move_m``),
    not track identity, carries the holder.  ``keeper_hold_min_frames <= 0`` or a missing
    ``evidence.pitch`` disables the channel.
    """
    if evidence.pitch is None or defenders is None or params.keeper_hold_min_frames <= 0:
        return []
    boxes = np.asarray(evidence.boxes, dtype=np.float64)
    teams = np.asarray(evidence.team)
    present = np.asarray(evidence.present, dtype=bool)
    pitchp = np.asarray(evidence.pitch, dtype=np.float64)
    n = boxes.shape[0]
    yolo = np.asarray(evidence.ball_yolo, dtype=np.float64)
    atd = np.asarray(evidence.ball_atd, dtype=np.float64)
    y_ok = np.asarray(evidence.ball_yolo_present, dtype=bool) & np.isfinite(yolo).all(axis=1)
    a_ok = np.isfinite(atd).all(axis=1)
    ball = np.where(y_ok[:, None], yolo, atd)
    seen = y_ok | a_ok
    speed = np.full(n, np.nan)
    speed[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=1)
    length, width = float(pitch[0]), float(pitch[1])
    half_of = np.full(n, -1, dtype=np.int8)
    for h, (a, b) in enumerate(halves):
        half_of[int(a):int(b)] = h
    cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
    cy = (boxes[..., 1] + boxes[..., 3]) / 2.0
    holder = np.full(n, -1, dtype=np.int16)
    ball_d = np.full(n, np.inf)
    opp_d = np.full(n, np.inf)  # distance to the nearest box of the other team
    keeper_ok = np.zeros(n, dtype=bool)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        bx, by = ball[s:e, 0], ball[s:e, 1]
        with np.errstate(invalid="ignore"):
            d = np.hypot(cx[s:e] - bx[:, None], cy[s:e] - by[:, None])
        d = np.where(present[s:e], d, np.inf)
        p = np.argmin(d, axis=1)
        dm = np.take_along_axis(d, p[:, None], axis=1)[:, 0]
        for t in np.flatnonzero(seen[s:e] & (dm <= params.keeper_hold_px)):
            j = s + int(t)
            team = int(teams[j, p[t]])
            if team not in (H, 1):
                continue
            px_, py_ = pitchp[j, p[t]]
            if not np.isfinite([px_, py_]).all():
                continue
            side = 0 if px_ < length / 2.0 else 1
            h_i = int(half_of[j])
            if h_i < 0 or int(defenders[h_i, side]) != team:
                continue
            goal_x = 0.0 if side == 0 else length
            if abs(px_ - goal_x) > params.keeper_hold_depth_m:
                continue
            if abs(py_ - width / 2.0) > params.keeper_hold_half_width_m:
                continue
            o_bit = 2 if team == H else 1
            opp_d[j] = np.min(np.where((teams[j] == o_bit) & present[j], d[t], np.inf))
            holder[j] = p[t]
            ball_d[j] = dm[t]
            keeper_ok[j] = True
    out: list[KeeperHold] = []
    contain_arr = np.asarray(contain)
    dead = np.asarray(event_dead, dtype=np.float64)
    rs = -1
    for t in range(n + 1):
        if t < n and keeper_ok[t]:
            if rs < 0:
                rs = t
            continue
        if rs >= 0 and t - rs >= params.keeper_hold_min_frames:
            seg = slice(rs, t)
            team = int(teams[rs, holder[rs]])
            vv = speed[seg]
            vv = vv[np.isfinite(vv)]
            pos = pitchp[np.arange(rs, t), holder[rs:t]]
            if (
                vv.size
                and float(np.median(vv)) <= params.keeper_hold_speed_px
                and np.isfinite(pos).all()
                and float(np.linalg.norm(pos - np.median(pos, axis=0), axis=1).max())
                <= params.keeper_hold_move_m
                and float((opp_d[seg] < ball_d[seg]).mean()) <= params.keeper_hold_opp_frac
                and int(((contain_arr[seg] & (2 if team == H else 1)) > 0).sum())
                <= params.keeper_hold_anchor_contain_max
                and float(dead[seg].mean()) <= params.dead_max
            ):
                out.append(KeeperHold(start=rs, frame=t - 1, team=team, length=t - rs))
        rs = -1
    return out


def gate_pixel_events(
    contact: Sequence[tuple],
    kicker: Sequence[tuple],
    contain: np.ndarray,
    codes_pooled: np.ndarray,
    fused_prob: np.ndarray,
    event_dead: np.ndarray,
    kept_kinematic: Sequence[TransitionEvent],
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
    relay: frozenset[tuple[int, int]] = frozenset(),
    holds: Sequence[KeeperHold] = (),
) -> tuple[list[TransitionEvent], list[tuple[int, int, int]]]:
    """Admit pixel events into a sequential stream and hand out owned intervals.

    Pixel evidence decides a possession only where the camera actually sees what the
    pitch geometry cannot: *near a possession boundary* (within ``pixel_bound_window``
    frames of a change in the fused codes) and only when the fused evidence *catches
    up* — its probability for the event's team reaches ``pixel_momentum_min`` within
    ``pixel_momentum_horizon`` frames — or, for kicks, when the win-and-clear
    fingerprint (``pixel_fingerprint_*``: released slowly from feet the other team was
    nowhere near) says the ball was won and put in flight in one motion.  Contacts
    marked by ``relay_contacts`` pass regardless of boundary distance and momentum:
    four agreeing sensors plus a cross-stream-attested hold that the prior team never
    shows on camera.  A settled spell is otherwise left to the pitch geometry: box
    overlap in depth looks identical to a take and must not puncture the spell.

    Events are processed in frame order.  A team that already owns the stream cannot
    re-anchor, and admitting the other team's flip ends the previous interval.  Each
    kept event owns ``[start, stop)``: the contact extend-back (or one frame before a
    kick) up to the next fused-code boundary, a stronger later pixel event, or the
    length cap (``pixel_cap_solid`` for solid contacts, else ``pixel_cap``), whichever
    comes first.  Returns ``(kept events, owned intervals)``.
    """
    n = codes_pooled.shape[0]
    bnd = np.flatnonzero(np.r_[True, codes_pooled[1:] != codes_pooled[:-1]])
    contact_keys = {(int(e[0]), int(e[2])) for e in contact}
    kicker_map = {(int(e[0]), int(e[2])): e for e in kicker}
    intervals: list[list[int]] = []
    kept: list[TransitionEvent] = []
    for ev in sorted([tuple(e) for e in list(contact) + list(kicker)], key=lambda e: (int(e[0]), int(e[2]))):
        frame, strength, team = int(ev[0]), float(ev[1]), int(ev[2])
        is_contact = (frame, team) in contact_keys
        close_px = dom_px = 0.0
        if not is_contact:
            release = strength
            if release > params.kicker_release_max:
                continue  # a track jump, not a kick
            close_px, dom_px = float(kicker_map[(frame, team)][3]), float(kicker_map[(frame, team)][4])
        if team not in (H, 1):
            continue
        k = int(np.searchsorted(bnd, frame, side="right"))
        prev_b = int(bnd[k - 1]) if k > 0 else -(10**9)
        next_b = int(bnd[k]) if k < len(bnd) else n
        is_relay = (frame, team) in relay
        if not is_relay and min(frame - prev_b, next_b - frame) > params.pixel_bound_window:
            continue  # mid-spell: settled possession, boxes confound depth
        hf = min(n - 1, frame + params.pixel_momentum_horizon)
        momentum = float(fused_prob[hf, team]) >= params.pixel_momentum_min
        fingerprint = (
            not is_contact
            and close_px <= params.pixel_fingerprint_close
            and dom_px >= params.pixel_fingerprint_dom
            and strength <= params.pixel_fingerprint_release_max
        )
        if not (momentum or fingerprint or is_relay):
            continue  # fused evidence never comes around and no winning fingerprint: noise
        cur = codes_pooled[max(0, frame - 20):frame]
        cur = cur[(cur == H) | (cur == 1)]
        holder = int(np.bincount(cur).argmax()) if cur.size else -1
        for a, b, t_ in intervals:
            if a <= frame < b:
                holder = int(t_)
        if holder < 0 or holder == team:
            continue
        if float(np.mean(event_dead[max(0, frame - 10):frame + 10])) > params.dead_max:
            continue
        if any(kev.team == team and abs(kev.frame - frame) < params.nms_window for kev in kept_kinematic):
            continue  # kinematics already attributes this moment
        team_bit = 1 if team == H else 2
        start = frame
        if is_contact:
            k2 = frame - 1
            while k2 >= max(0, frame - params.extend_back) and int(contain[k2]) & team_bit:
                start = k2
                k2 -= 1
            cap = params.pixel_cap_solid if strength >= params.pixel_solid_run else params.pixel_cap
            support = min(1.0, strength / 8.0)
        else:
            start = max(0, frame - 1)
            cap = params.pixel_cap
            support = min(1.0, strength / 12.0)
        stop = min(frame + cap, next_b)
        if next_b - stop <= params.pixel_commit_gap:
            tail = codes_pooled[next_b:next_b + params.pixel_commit_frames]
            if tail.size and np.all(tail == team) and float(fused_prob[min(n - 1, next_b), team]) >= params.pixel_commit_prob:
                stop = next_b  # fused evidence is about to commit to this team: absorb the lag
        for iv in intervals:
            if iv[0] < frame < iv[1]:
                iv[1] = frame
        kept.append(
            TransitionEvent(start=start, frame=frame, kind="contact" if is_contact else "kicker",
                            team=team, distance=close_px if not is_contact else 1.0, support=support)
        )
        intervals.append([start, stop, team])
    # keeper holds own the stretch from the catch to the next fused boundary: the camera
    # watched the ball sit in the defender's box, so this needs no boundary proximity and
    # no momentum, but it never cuts into a frame the pixel channel already decided.
    for hold in sorted(holds, key=lambda h: h.frame):
        frame, team = int(hold.frame), int(hold.team)
        if team not in (H, 1) or hold.length < params.keeper_hold_min_frames:
            continue
        k = int(np.searchsorted(bnd, frame, side="right"))
        next_b = int(bnd[k]) if k < len(bnd) else n
        for a, b, _ in intervals:
            if a < next_b and hold.start < b:
                next_b = a  # hand the stretch over at the pixel-owned front edge
        if next_b <= hold.start:
            continue  # a contact or kicker already owns the whole catch
        cur = codes_pooled[max(0, frame - 20):frame]
        cur = cur[(cur == H) | (cur == 1)]
        holder = int(np.bincount(cur).argmax()) if cur.size else -1
        for a, b, t_ in intervals:
            if a <= frame < b:
                holder = int(t_)
        if holder < 0 or holder == team:
            continue
        if float(np.mean(event_dead[max(0, frame - 10):frame + 10])) > params.dead_max:
            continue
        if any(kev.team == team and abs(kev.frame - frame) < params.nms_window for kev in kept_kinematic):
            continue  # kinematics already attributes this moment
        start = int(hold.start)
        for iv in intervals:
            if iv[0] < start < iv[1]:
                start = iv[1]
        if start >= next_b:
            continue
        kept.append(
            TransitionEvent(start=start, frame=frame, kind="hold", team=team,
                            distance=1.0, support=1.0)
        )
        intervals.append([start, next_b, team])
    return kept, [(a, b, t_) for a, b, t_ in intervals]


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
        if ev.kind == "hold":
            strength = params.contact_pulse
        elif ev.kind in ("contact", "kicker"):
            strength = (
                params.contact_pulse_solid
                if ev.kind == "contact" and ev.support >= params.contact_solid_support
                else params.contact_pulse
            )
        else:
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

def ball_stale_mask(ball: np.ndarray, ball_present: np.ndarray, stale_frames: int = 125) -> np.ndarray:
    """Frames deep inside a long ATD ball blind window (> ``stale_frames`` on both sides).

    During those gaps the consensus ball is video-carried and its speed is a tracker
    artifact, not the ball: kinematic kick/onset detection there invents transitions.
    """
    points = np.asarray(ball, dtype=np.float64)
    obs = np.asarray(ball_present, dtype=bool) & np.isfinite(points).all(axis=1)
    n = obs.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(obs, idx, -10**7))
    nxt = np.minimum.accumulate(np.where(obs, idx, 10**7)[::-1])[::-1]
    return (idx - last > int(stale_frames)) & (nxt - idx > int(stale_frames))


def ball_conflict_mask(
    atd_ball: np.ndarray,
    yolo_ball: np.ndarray,
    min_distance: float = 5.0,
    window: int = 60,
    min_conflicts: int = 8,
    pad: int = 0,
) -> np.ndarray:
    """Frames where the ATD ball and the dedicated-YOLO ball insist on different places.

    Both sources track the same ball; more than ``min_distance`` metres apart *for a
    sustained ``window`` stretch* means one of them is off the ball -- typically the ATD
    slot drifting onto a nearby player while YOLO keeps the ball (measured 20 955: 5.7-6.8
    m divergence for two seconds; the consensus ball rode the drift and a kinematic onset
    handed possession to the player the drift passed -- the dedicated detector kept the
    ball on Barcelona).  Kinematic events computed on those frames ride the wrong
    trajectory and must be dropped.  Single-frame YOLO blips (wrong candidate,
    ``min_conflicts`` guard) never gate.  The mask is grown ``pad`` frames on both sides:
    the drift ramps in before the divergence crosses the distance threshold and settles
    back over after.  The divergence alone decides nothing about who holds the ball (runs
    of 4-25 m happen dozens of times during correctly-decided play) -- it may gate
    trajectory-derived events only, never the pixel contact channel or the decode.
    """
    atd = np.asarray(atd_ball, dtype=np.float64)
    yolo = np.asarray(yolo_ball, dtype=np.float64)
    both = np.isfinite(atd).all(axis=1) & np.isfinite(yolo).all(axis=1)
    if not both.any():
        return np.zeros(atd.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(atd - yolo, axis=1)
    raw = both & (dist > float(min_distance))
    if not raw.any():
        return np.zeros(atd.shape[0], dtype=bool)
    counts = np.convolve(raw.astype(np.int64), np.ones(int(window), dtype=np.int64), mode="full")
    counts = counts[int(window) - 1 : int(window) - 1 + raw.size]  # forward-window sums:
    # a frame is marked while its next ``window`` frames hold >= min_conflicts divergent
    # ones -- gating starts as the drift ramps in, before the sources finish splitting
    out = counts >= int(min_conflicts)
    pad = int(pad)
    if pad > 0 and out.any():
        idx = np.flatnonzero(out)
        for d in range(-pad, pad + 1):
            if d:
                out[np.clip(idx + d, 0, out.size - 1)] = True
    return out


def carry_dead_through_blind(
    codes: np.ndarray,
    blind: np.ndarray,
    halves: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """Carry a dead-ball state through frames where the ball cannot vouch for a restart.

    Inside a deep ATD blind window at low consensus confidence the "restart" the decode
    produces is the video-carried ball resuming the pre-stoppage trajectory -- not the
    game (measured 19 394: inside the window the decode restarted Barcelona *while an
    Athletic player physically held the throw-in ball*).  When the last state seen while
    the ball was still reliable was *dead*, it holds until the window closes; live states
    (H/A) keep decoding, because a takeover genuinely happens while the ball is hidden.
    Applied after the decode so a blind window cannot perturb the rest of the half.
    """
    out = np.asarray(codes).copy()
    mask = np.asarray(blind, dtype=bool)
    if out.size == 0 or not mask.any():
        return out
    if halves is None:
        halves = GRADED_HALVES
    for start, stop in halves:
        start, stop = int(start), int(stop)
        i, n = start, min(stop, out.size)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            state = int(out[i - 1]) if i > 0 else int(out[i])
            j = i
            while j < n and mask[j]:
                if state == D:
                    out[j] = D
                j += 1
            i = j
    return out


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
    box_evidence: BoxEvidence | None = None,
    ball_stale: np.ndarray | None = None,
    blind_dead: np.ndarray | None = None,
    ballots: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    keeper_defenders: np.ndarray | None = None,
    keeper_pitch: tuple[float, float] | None = None,
) -> TransitionLayerResult:
    """Detect, gate, and decode ball-control transitions on top of the fused poll.

    ``player_sets`` are ``(pitch (n, p, 2), teams, present (n, p))`` tuples covering the
    same frame range as ``codes_pooled`` (video tracking, ATD tracking, ...).  With
    ``box_evidence`` the dedicated-YOLO pixel channel joins in: a fresh ball-in-box
    takeover (``contact``) and a win-and-clear kick that leaves the winner (``kicker``)
    are gated against the current holder and the fused evidence: a fresh ball-in-box
    takeover (``contact``) and a win-and-clear kick (``kicker``) fire only near a fused
    boundary where the evidence catches up (or carries the winning fingerprint), hold
    their team with a sustained logit until the boundary or a cap, and leave settled
    spells to the pitch geometry.  With ``ballots`` (voter key -> ``(probability,
    weight)`` for the four ball/control voters) the cross-stream relay
    (``relay_contacts``) additionally admits a camera-attested takeover inside a
    settled spell.
    """
    consensus = np.asarray(consensus_ball, dtype=np.float64)
    sets64 = [(np.asarray(p, dtype=np.float64), np.asarray(t), np.asarray(pr, dtype=bool)) for p, t, pr in player_sets]
    speed, valid = ball_speed(consensus)
    nearest_team = control_map(consensus, sets64)
    kicks = detect_kick_events(consensus, speed, valid, sets64, params)
    onsets = detect_onset_events(consensus, speed, valid, nearest_team, sets64, params)
    if ball_stale is not None:
        # inside long ATD blind windows the consensus speed is a video carry, not the ball
        stale = np.asarray(ball_stale, dtype=bool)
        kicks = [event for event in kicks if not stale[event[0]]]
        onsets = [event for event in onsets if not stale[event[0]]]
    events, dropped = gate_events(kicks, onsets, codes_pooled, evidence_prob, event_dead, nearest_team, params)
    if box_evidence is not None:
        contain, touch = pixel_contact(box_evidence)
        contact = detect_contact_events(contain, touch, speed, event_dead, params)
        relay = (
            relay_contacts(contact, contain, box_evidence, codes_pooled, ballots, params)
            if ballots is not None
            else frozenset()
        )
        kicker = detect_kicker_events(box_evidence, touch, speed, params)
        holds = (
            keeper_holds(box_evidence, contain, event_dead, keeper_defenders, keeper_pitch,
                         halves if halves is not None else GRADED_HALVES, params)
            if keeper_defenders is not None and keeper_pitch is not None
            else []
        )
        pixel_kept, intervals = gate_pixel_events(
            contact, kicker, contain, codes_pooled, np.asarray(fused_probability, dtype=np.float64),
            event_dead, events, params, relay=relay, holds=holds,
        )
        events = tuple(sorted(events + tuple(pixel_kept), key=lambda e: (e.start, e.frame)))
        dropped += len(contact) + len(kicker) + len(holds) - len(pixel_kept)
        n_contact, n_kicker, n_hold = len(contact), len(kicker), sum(1 for e in pixel_kept if e.kind == "hold")
    else:
        n_contact = n_kicker = n_hold = 0
        intervals: list[tuple[int, int, int]] = []
    pulse, teams, boost, protect = build_pulse(events, codes_pooled.shape[0], params)
    log_em = np.log(np.clip(np.asarray(fused_probability, dtype=np.float64), EPS, 1.0)).copy()
    for team in (H, 1):
        mask = teams == team
        log_em[mask, team] += pulse[mask]
    log_em -= np.log(np.exp(log_em).sum(axis=1, keepdims=True))
    for a, b, team in intervals:
        log_em[a:b, team] += params.pixel_bias
        protect[a:b] = True
        boost[max(0, a - 10):min(log_em.shape[0], a + 10)] = True
        boost[max(0, b - 10):min(log_em.shape[0], b + 10)] = True
    if intervals:
        log_em -= np.log(np.exp(log_em).sum(axis=1, keepdims=True))
    codes = decode_with_events(log_em, boost, protect, np.asarray(event_dead, dtype=bool), model_params, halves)
    if blind_dead is not None:
        # a dead ball that went blind stays dead: the "restart" inside the window is the
        # carried trajectory, not play (post-decode so a window cannot disturb the half)
        codes = carry_dead_through_blind(codes, blind_dead, halves)
    return TransitionLayerResult(
        codes=codes,
        events=events,
        n_kicks=len(kicks),
        n_onsets=len(onsets),
        n_gated_out=dropped,
        n_contact=n_contact,
        n_kicker=n_kicker,
        n_hold=n_hold,
    )


def events_to_rows(events: Sequence[TransitionEvent]) -> np.ndarray:
    """Events as an ``(m, 6)`` array: start, frame, kind (0=kick, 1=onset, 2=contact,
    3=kicker, 4=hold), team, distance, support."""
    if not events:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray([e.as_row() for e in events], dtype=np.float64)


__all__ = [
    "DEFAULT_TRANSITION_PARAMS",
    "BoxEvidence",
    "TransitionEvent",
    "TransitionLayerResult",
    "TransitionParams",
    "apply_transition_layer",
    "ball_conflict_mask",
    "ball_speed",
    "ball_stale_mask",
    "build_pulse",
    "carry_dead_through_blind",
    "control_map",
    "decode_with_events",
    "detect_contact_events",
    "detect_kick_events",
    "detect_kicker_events",
    "detect_onset_events",
    "events_to_rows",
    "evidence_probability",
    "gate_events",
    "gate_pixel_events",
    "keeper_holds",
    "KeeperHold",
    "pixel_contact",
]
