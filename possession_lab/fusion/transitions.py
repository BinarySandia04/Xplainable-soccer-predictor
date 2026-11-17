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
    # a ball stopped dead on an opponent's foot outranks the pooled evidence lag: with the
    # chase and tail checks above behind it, the distance threshold alone decides
    onset_touch_dist: float = 0.5  # m: the ball killed within arm's reach of the taker
    onset_touch_min_loss: float = 7.0  # m/s of speed the ball must have shed to count
    # a witnessed takeover survives *decoded* deadness: a keeper gathering a save looks
    # dead to the motion features while play is alive; only the stoppage tag ends the spell
    onset_touch_hold_frames: int = 150  # cap on the hold-through (0 disables it)
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
    # -- box lock: a solid ball-inside-box run may take over a settled spell when the
    # camera never shows the ball in the current holder's box at all during that window
    box_lock_min_run: int = 5  # contact run length that counts as a lock
    box_lock_clear_frames: int = 40  # holder must not contain the ball in this window
    # -- pixel intervals: pixel evidence only corrects timing near possession boundaries;
    # settled spells belong to the pitch geometry (box overlap in depth looks identical)
    # except through the box-lock override above.
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
    # The catch ends a pass: the camera watched the ball travel from a same-team player to
    # the keeper without an opponent touching it, so the *pass itself* belongs to the
    # catcher's team.  Extend the hold back over that flight to the last frame the ball was
    # within ``keeper_hold_pass_m`` of an outfield teammate.  Measured 19 814-19 859: the
    # fused spell read Athletic for the whole pass while Barcelona's player released it at
    # 19 814 and the keeper caught it at 19 859.
    keeper_hold_pass_m: float = 3.0  # ball-to-teammate radius that still counts as control
    keeper_hold_pass_frames: int = 90  # max look-back through the pass (3.6 s)
    keeper_hold_pass_gap: int = 8  # missing-ball frames tolerated inside the flight
    # The catch owns play until the keeper puts the ball back in it, not until the fused poll
    # happens to move: inside a long ball gap the poll is blind, so "to the next fused boundary"
    # stamped one 39-frame hold over 70 s the camera watched Barcelona win (measured 80 651 ->
    # 82 359: the hold ran to 82 359 while the fused codes held H, a live-play tag was on, and a
    # Barcelona player had the ball at 0.4 m at 81 918).  6 s past the release frame.
    keeper_hold_release_frames: int = 150
    # A win-and-clear fingerprint is feet evidence from the dedicated detector; it needs the pitch
    # stream to agree about *where* the ball is.  While the ATD ball is gone the fused poll is
    # carried by its own prior, so "the evidence catches up" cannot be replaced by a fingerprint:
    # a release measured in a blind window is a camera reading with nothing to check it against
    # (measured 86 219: 6.4 px from feet, 131 px clear, the ATD ball blind since 86 186).
    pixel_fingerprint_needs_atd: bool = True
    # -- keeper area catch: the pixel channel goes blind near the goal, where depth
    # compresses the boxes, so the catch is read off the pitch geometry (the pitch
    # twin of the keeper hold for camera-blind moments).  Measured 132 355: the ball
    # decelerated 13.1 -> 1.1 m/s at x = 9.4 m (the Athletic keeper's six-yard box,
    # defending the low side in the second half) and left again at 18.9 m/s, with no
    # opponent closer; the fused prior kept claiming H through catch and punt.
    area_catch_in_min_speed: float = 8.0  # m/s the ball must arrive with
    area_catch_max_speed: float = 3.5  # m/s the stop counts as a catch below
    area_catch_prior_frames: int = 40  # window the arrival speed is read from
    area_catch_depth_m: float = 16.5  # penalty-area depth from the goal line
    area_catch_half_width_m: float = 20.2  # half the area width around the centre
    area_catch_opp_min_m: float = 2.5  # an opponent closer contests the catch
    area_catch_kick_gain: float = 3.0  # m/s jump marking the pass-in kick
    area_catch_extend_max: int = 200  # frames searched back for that kick
    area_catch_extend_margin: float = 0.15  # raw-pool support that anchors the start
    area_catch_extend_dip_max: int = 15  # frames the support may dip for mid-flight
    area_catch_max_span: int = 330  # ownership cap: catch, gather, punt
    area_catch_min_pool_margin: float = 0.25  # the voters must back the catcher
    # -- evidence expiry of a pixel-owned spell: an owned interval cannot outlive
    # the camera's own evidence.  A lone touch that is never followed up must not
    # hold the spell while the ball settles on the other team's feet (measured
    # 132 743: one A contact at 132 619 owned 326 frames while the raw voter pool
    # read 0.80-0.94 H with the ball dribbling at 3-6 m/s under a Barcelona player).
    expire_min_prob: float = 0.80
    expire_min_margin: float = 0.25
    expire_min_frames: int = 25
    expire_max_ball_speed: float = 8.0  # m/s: control, not flight
    # -- pitch contradiction veto on pixel events: boxes overlap at depth, a pitch
    # distance never lies (measured 136 891: four frames of ball-inside-A-box while
    # an H player stood 0.4 m from the ball and his side led the raw pool)
    contact_opp_pitch_margin: float = 2.0  # m the other side must be closer by to veto
    contact_opp_pool_margin: float = 0.15  # and the raw pool must side with that player
    # -- ball out of play without a tag: the camera lost the ball, or it crossed
    # the touchline; the fused confidence says no voter sees it anymore, and the
    # consensus then coasts off the grass (measured 163 282: conf 0.00 for minutes,
    # the Kalman point drifting to y = -59 m while spells kept trading hands)
    lost_min_frames: int = 25  # frames without any ball voter before "lost" (1 s)
    lost_conf_eps: float = 0.02
    oob_margin_m: float = 0.8  # metres past the chalk before "out" (a line ball is in)
    oob_min_frames: int = 25  # frames outside before the stoppage is called
    # -- escape from a falsely locked dead state: the events ballot (weight 1.0)
    # pins the label to D, so no voter can reopen it — but the smart tags can.
    # A live-phase tag naming the team the voters already back, with the ball on
    # the grass and moving, means the stoppage never happened (measured 143 524:
    # an ATD excursion 4 m past the chalk held D for 16 s while Athletic drove
    # forward under a PROGRESSION tag the voters agreed with at 0.85)
    revive_min_frames: int = 25  # frames of live-play evidence before reopening
    revive_gap_min: float = 0.2  # raw pool margin the tagged team must command
    revive_speed_min: float = 1.5  # m/s of the consensus ball proving play is live
    # the ball-state revival: a player at the ball + pool backing reopens the lock
    revive_touch_dist_m: float = 2.0  # how close a player must be to own the ball
    revive_touch_cov: float = 0.7  # share of each second's frame holding that glue
    revive_touch_gap: float = 0.1  # raw pool margin the gluing player's team needs


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
    # witnessed takeover: survives decoded deadness (``onset_touch_hold_frames``) — the
    # spell ends only at a supported stoppage
    hold_through: bool = False

    def as_row(self) -> tuple[int, int, int, int, float, float]:
        kind_code = {"kick": 0, "onset": 1, "contact": 2, "kicker": 3, "hold": 4, "control": 5, "area": 6}[self.kind]
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
    n_control: int = 0
    n_area: int = 0


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


def nearest_team_at(
    ball_xy: np.ndarray,
    players: np.ndarray,
    teams: np.ndarray,
    present: np.ndarray,
    max_distance: float = 2.0,
    min_margin: float = 1.0,
) -> int:
    """Team of the player standing at the ball, when the geometric claim is clear.

    Set-piece smart tags leave throw-ins with team ``N/A``: the player at the ball
    when it starts moving is the one taking the restart.  A contested ball is left
    unresolved (``-1``) so the caller keeps the decoded state instead of guessing.
    """
    ball = np.asarray(ball_xy, dtype=np.float64)
    pitch = np.asarray(players, dtype=np.float64)
    team = np.asarray(teams)
    ok = np.asarray(present, dtype=bool) & np.isin(team, (H, 1))
    if pitch.ndim != 2 or ok.shape != (pitch.shape[0],):
        return -1
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(pitch - ball, axis=1)
    ok &= np.isfinite(dist)
    if not ok.any():
        return -1
    rows = np.flatnonzero(ok)
    closest = rows[int(np.argmin(dist[rows]))]
    if float(dist[closest]) > float(max_distance):
        return -1
    others = rows[team[rows] != team[closest]]
    if others.size and float(dist[others].min()) < float(dist[closest]) + float(min_margin):
        return -1
    return int(team[closest])


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
    ball_pitch: np.ndarray | None = None  # (n, 2) float metres, video ball for the pass extension


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


def _pitch_nearest_m(evidence: BoxEvidence) -> np.ndarray | None:
    """Per-frame metres from the video ball to the nearest player of each team.

    ``(n, 2)`` float array, column 0 = team H, column 1 = team A; ``inf`` where
    that team shows no finite player.  The pitch twin of :func:`_box_foot_distance`
    — pixel boxes confound depth, pitch metres do not.
    """
    if evidence.pitch is None or evidence.ball_pitch is None:
        return None
    p = np.asarray(evidence.pitch, dtype=np.float64)
    teams = np.asarray(evidence.team)
    present = np.asarray(evidence.present, dtype=bool)
    b = np.asarray(evidence.ball_pitch, dtype=np.float64)
    n = min(p.shape[0], b.shape[0])
    out = np.full((n, 2), np.inf)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        ok = present[s:e] & np.isin(teams[s:e], (H, 1)) & np.isfinite(p[s:e]).all(axis=2)
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(p[s:e] - b[s:e, None, :], axis=2)
        for col, team in ((0, H), (1, 1)):
            dm = np.where(ok & (teams[s:e] == team), d, np.inf)
            out[s:e, col] = dm.min(axis=1)
    out[~np.isfinite(b[:n]).all(axis=1)] = np.inf
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


def raw_evidence_pool(
    probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    """The weighted pool of the ball/control voters *without* the flat prior.

    ``evidence_probability`` damps the pool toward the three-way prior; a spell
    expiry must read the voters at full strength (the 0.92 the voters insisted on
    at 132 743 damps below the bar once the prior is mixed in).  Frames where no
    voter carries weight fall back to the uniform three-way split.
    """
    n = int(next(iter(probabilities.values())).shape[0])
    total = np.zeros(n)
    weighted = np.zeros((n, 3))
    for key, prob in probabilities.items():
        w = np.clip(np.asarray(weights[key], dtype=np.float64), 0.0, 1.0)
        weighted += w[:, None] * np.asarray(prob, dtype=np.float64)
        total += w
    return np.where(total[:, None] > 0, weighted / np.maximum(total, 1e-9)[:, None], 1.0 / 3.0)


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``[start, stop)`` runs of True in a boolean mask."""
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def ball_lost_mask(
    confidence: np.ndarray,
    min_frames: int = 25,
    eps: float = 0.02,
) -> np.ndarray:
    """Frames where no ball voter has carried the consensus for ``min_frames`` straight.

    A fused confidence of zero means both ball streams stopped seeing the ball; a
    minute of that is a stoppage the smart tags missed, not a play the system may
    keep attributing (measured 163 265+: conf 0.00 while spells traded hands).  The
    whole qualifying run is marked, never just its tail.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    gone = ~np.isfinite(conf) | (conf <= eps)
    mask = np.zeros(conf.shape[0], dtype=bool)
    for s, e in _bool_runs(gone):
        if e - s >= int(min_frames):
            mask[s:e] = True
    return mask


def out_of_bounds_mask(
    ball_pitch: np.ndarray,
    pitch: tuple[float, float],
    margin_m: float = 0.8,
    min_frames: int = 25,
    confidence: np.ndarray | None = None,
    conf_eps: float = 0.02,
) -> np.ndarray:
    """Frames with the ball centre fully past the touchline for ``min_frames`` straight.

    ``margin_m`` forgives the chalk and calibration jitter (a ball on the line is
    in play); non-finite frames break a run rather than extend it.  The tags miss
    stoppages late in scrambled sequences (measured 163 282: the ball sailed over
    the side with no stoppage tag until the goal-kick restart minutes later), and
    out of bounds nobody possesses the ball: the caller marks those frames dead.
    When ``confidence`` is provided, frames without camera evidence (confidence <= eps)
    are skipped so Kalman coasting does not trigger false out-of-bounds runs.
    """
    b = np.asarray(ball_pitch, dtype=np.float64)
    length, width = float(pitch[0]), float(pitch[1])
    finite = np.isfinite(b).all(axis=1)
    if confidence is not None:
        c = np.asarray(confidence, dtype=np.float64)
        finite = finite & np.isfinite(c) & (c > float(conf_eps))
    out = finite & (
        (b[:, 0] < -margin_m)
        | (b[:, 0] > length + margin_m)
        | (b[:, 1] < -margin_m)
        | (b[:, 1] > width + margin_m)
    )
    mask = np.zeros(b.shape[0], dtype=bool)
    for s, e in _bool_runs(out):
        if e - s >= int(min_frames):
            mask[s:e] = True
    return mask


def dead_revival_mask(
    ball_pitch: np.ndarray,
    evidence_pool: np.ndarray,
    live_tags: Sequence[tuple[int, int, int, bool]],
    pitch: tuple[float, float],
    blocked: np.ndarray,
    min_frames: int = 25,
    gap_min: float = 0.2,
    speed_min: float = 1.5,
    raw_ball: np.ndarray | None = None,
    marginal_max_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Frames where a dead-ball lock broke against evidence that play never stopped.

    The events ballot (weight 1.0) pins the label to ``D``, so while a dead run
    lasts no voter can reopen it — but the smart tags can read the situation the
    out-of-play trigger misfired on.  ``live_tags`` are ``(start, stop, team,
    strong)`` triples: a *strong* live-phase tag (PROGRESSION, CREATING CHANCES)
    naming the team the raw pool already backs by ``gap_min``, with the consensus
    ball on the grass and moving at ``speed_min``, revives the run outright — the
    stoppage is a tracking excursion (measured 143 524: an ATD swing 4 m past the
    chalk held D for 16 s while Athletic drove forward).  A *weak* team tag (ball
    in the box, final third) only revives when the out was geometrically marginal:
    the raw ball crossed the chalk but never passed ``marginal_max_m`` (measured
    116 791: 10 cm of calibration noise at y = 68.1 on a 68.0 m pitch punctured
    live possession).  ``blocked`` frames — setpiece tags, or a ball the cameras
    actually lost — never revive.
    """
    b = np.asarray(ball_pitch, dtype=np.float64)
    n = b.shape[0]
    pool = np.asarray(evidence_pool, dtype=np.float64)
    length, width = float(pitch[0]), float(pitch[1])
    live = np.zeros(n, dtype=bool)
    strong = np.zeros(n, dtype=bool)
    team = np.full(n, -1, dtype=np.int8)
    # two passes, weak tags first: a strong tag always wins the frame's team
    for want_strong in (False, True):
        for s, e, t, st in live_tags:
            if bool(st) != want_strong:
                continue
            s2, e2 = max(0, int(s)), min(n, int(e) + 1)
            if e2 > s2:
                live[s2:e2] = True
                strong[s2:e2] |= bool(st)
                team[s2:e2] = int(t)
    finite = np.isfinite(b).all(axis=1)
    inside = (
        finite
        & (b[:, 0] > -0.3)
        & (b[:, 0] < length + 0.3)
        & (b[:, 1] > -0.3)
        & (b[:, 1] < width + 0.3)
    )
    speed = np.full(n, np.nan, dtype=np.float64)
    ok = finite[1:] & finite[:-1]
    speed[1:][ok] = np.linalg.norm((b[1:] - b[:-1])[ok], axis=1) * FPS
    gap = np.abs(pool[:, 1] - pool[:, 0])
    winner = np.argmax(pool[:, :2], axis=1)
    core = (
        live
        & inside
        & np.isfinite(speed)
        & (speed >= float(speed_min))
        & (winner == team)
        & ~np.asarray(blocked, dtype=bool)
    )
    # how far past the chalk the raw ball strayed, per frame (a marginal cross only)
    if raw_ball is None:
        crossed = np.zeros(n, dtype=bool)
    else:
        rb = np.asarray(raw_ball, dtype=np.float64)
        rfinite = np.isfinite(rb).all(axis=1)
        over = np.maximum(
            np.maximum(np.maximum(-rb[:, 0], rb[:, 0] - length), np.maximum(-rb[:, 1], rb[:, 1] - width)),
            0.0,
        )
        crossed = rfinite & (over > 0.0) & (over <= float(marginal_max_m))
    # a strong live tag reopens when the pool clearly backs it (gap); a weak tag needs
    # the marginal chalk crossing itself as proof — a celebration has no crossing at all,
    # a real throw-in restart is setpiece-blocked
    qualified = core & ((strong & (gap >= float(gap_min))) | crossed)
    mask = np.zeros(n, dtype=bool)
    out_team = np.full(n, -1, dtype=np.int8)
    for s, e in _bool_runs(qualified):
        if e - s >= int(min_frames):
            mask[s:e] = True
            out_team[s:e] = team[s]
    return mask, out_team


def possession_revival_mask(
    ball_pitch: np.ndarray,
    players: np.ndarray,
    player_present: np.ndarray,
    player_teams: np.ndarray,
    evidence_pool: np.ndarray,
    pitch: tuple[float, float],
    blocked: np.ndarray,
    dist_max: float = 2.0,
    hold_cov: float = 0.7,
    gap_min: float = 0.1,
    speed_min: float = 1.5,
    min_frames: int = 25,
    hold_half: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Frames where a dead-ball lock sat on top of a player who was simply playing.

    The events ballot pins ``D`` while a false stoppage lasts and neither the smart
    tags nor any voter can reach the label.  The ball's own state can: a player of
    one team is at the ball (``dist_max``), the ball is on the grass and moving
    (``speed_min``) and the raw pool backs that same player's team (``gap_min``) —
    for a ``hold_cov`` share of every ``hold_half``-frame neighbourhood and at
    least ``min_frames`` in a row.  The pool condition is what keeps goal
    celebrations out (the crowd around the ball belongs to the scoring team, but
    while the play is really dead the pool has collapsed onto D); ``blocked``
    frames — setpiece tags, a ball the cameras lost, a ball out of bounds — never
    revive.  Measured 170 623: Athletic's BUILD UP carried the ball 17 m while the
    label sat dead; 143 488 filled the hole between the false out and the tag's
    start; inside the public window the rule touches zero frames.
    """
    b = np.asarray(ball_pitch, dtype=np.float64)
    pl = np.asarray(players, dtype=np.float64)
    present = np.asarray(player_present, dtype=bool)
    team_of = np.asarray(player_teams, dtype=np.int8)
    pool = np.asarray(evidence_pool, dtype=np.float64)
    n = b.shape[0]
    fin = np.isfinite(b).all(axis=1)
    d = np.linalg.norm(pl - b[:, None, :], axis=2)
    d[~present] = np.inf
    d[~fin] = np.inf
    dmin = d.min(axis=1)
    nearest = team_of[np.clip(np.argmin(d, axis=1), 0, team_of.shape[0] - 1)]
    length, width = float(pitch[0]), float(pitch[1])
    inside = (
        fin
        & (b[:, 0] > -0.3) & (b[:, 0] < length + 0.3)
        & (b[:, 1] > -0.3) & (b[:, 1] < width + 0.3)
    )
    speed = np.full(n, np.nan, dtype=np.float64)
    ok = fin[1:] & fin[:-1]
    speed[1:][ok] = np.linalg.norm((b[1:] - b[:-1])[ok], axis=1) * FPS
    gap = np.abs(pool[:, 1] - pool[:, 0])
    winner = np.argmax(pool[:, :2], axis=1)
    glue = (
        inside
        & np.isfinite(speed)
        & (speed >= float(speed_min))
        & (dmin <= float(dist_max))
        & (nearest >= 0) & (nearest < 2)
        & (winner == nearest)
        & (gap >= float(gap_min))
        & ~np.asarray(blocked, dtype=bool)
    )
    kernel = np.ones(2 * int(hold_half) + 1)
    cov = np.convolve(glue.astype(np.float64), kernel, mode="same") / np.convolve(
        np.ones(n), kernel, mode="same"
    )
    mask = np.zeros(n, dtype=bool)
    out_team = np.full(n, -1, dtype=np.int8)
    for s, e in _bool_runs(cov >= float(hold_cov)):
        if e - s < int(min_frames):
            continue
        teams_in = nearest[s:e][glue[s:e]]
        if not len(teams_in):
            continue
        vals, counts = np.unique(teams_in, return_counts=True)
        mask[s:e] = True
        out_team[s:e] = int(vals[int(np.argmax(counts))])
    return mask, out_team


def stitch_keeper_saves(
    codes: np.ndarray,
    consensus_ball: np.ndarray,
    players: np.ndarray,
    keeper_defenders: np.ndarray,
    keeper_slots: np.ndarray,
    halves: Sequence[tuple[int, int]],
    pitch: tuple[float, float],
    raw_ball: np.ndarray | None = None,
    setpiece_tags: np.ndarray | None = None,
    max_gap_frames: int = 300,
    min_gap_frames: int = 10,
    keeper_near_dist: float = 12.0,
) -> np.ndarray:
    """Stitch goalkeeper saves/catches that were falsely broken by ball occlusion into D.

    When a goalkeeper catches or parries the ball into gloves/chest, cameras lose sight of the
    ball, tripping lost-ball or coasted out-of-bounds deadness. If the goalkeeper is inside his
    defending penalty area, secured the ball, and releases it to his team within 10 seconds
    (without the ball having crossed the end line out of bounds for a goal-kick restart),
    the preceding dead spell is stitched back to the goalkeeper's team possession.
    """
    out_codes = np.asarray(codes, dtype=np.int8).copy()
    length, width = float(pitch[0]), float(pitch[1])
    b = np.asarray(consensus_ball, dtype=np.float64)
    pl = np.asarray(players, dtype=np.float64)

    for h_idx, (h_start, h_stop) in enumerate(halves):
        h_start, h_stop = max(0, int(h_start)), min(len(out_codes), int(h_stop))
        for g_side in (0, 1):
            k_team = int(keeper_defenders[h_idx, g_side])
            k_slot = (
                int(keeper_slots[h_idx, g_side])
                if np.ndim(keeper_slots) == 2
                else int(keeper_slots[h_idx][g_side])
            )
            if k_team < 0 or k_slot < 0:
                continue
            k_pos = pl[:, k_slot]
            goal_x = 0.0 if g_side == 0 else length

            spells = []
            cur_l, cur_s = None, None
            for f in range(h_start, h_stop):
                l = int(out_codes[f])
                if l != cur_l:
                    if cur_l is not None:
                        spells.append((cur_s, f - 1, cur_l))
                    cur_l, cur_s = l, f
            if cur_l is not None:
                spells.append((cur_s, h_stop - 1, cur_l))

            for i, (s_start, s_end, s_lbl) in enumerate(spells):
                if s_lbl == k_team and i > 0:
                    p_start, p_end, p_lbl = spells[i - 1]
                    gap_len = p_end - p_start + 1
                    if p_lbl == D and min_gap_frames <= gap_len <= max_gap_frames:
                        kp_end = k_pos[p_end]
                        in_box = (
                            np.isfinite(kp_end[0])
                            and abs(kp_end[0] - goal_x) <= 20.0
                            and abs(kp_end[1] - width / 2.0) <= 22.0
                        )
                        ball_window = b[max(0, p_start - 35) : min(len(b), p_end + 35)]
                        fin = np.isfinite(ball_window).all(axis=1)
                        if fin.any() and np.isfinite(kp_end).all():
                            dist_to_keeper = float(np.min(np.linalg.norm(ball_window[fin] - kp_end, axis=1)))
                        elif np.isfinite(kp_end).all():
                            # Ball occluded in gloves/chest of keeper inside his penalty box
                            dist_to_keeper = 5.0
                        else:
                            dist_to_keeper = 999.0

                        crossed_end = False
                        if raw_ball is not None:
                            lookback = max(0, p_start - 50)
                            rb = raw_ball[lookback:p_start]
                            if np.any(np.isfinite(rb)):
                                if g_side == 0 and np.nanmin(rb[:, 0]) < -0.5:
                                    crossed_end = True
                                elif g_side == 1 and np.nanmax(rb[:, 0]) > length + 0.5:
                                    crossed_end = True

                        is_setpiece = False
                        if setpiece_tags is not None:
                            is_setpiece = bool(np.any(setpiece_tags[p_start : p_end + 1]))

                        if in_box and dist_to_keeper <= float(keeper_near_dist) and not crossed_end and not is_setpiece:
                            out_codes[p_start : p_end + 1] = k_team

    return out_codes


def apply_goalkeeper_restarts(
    codes: np.ndarray,
    consensus_ball: np.ndarray,
    players: np.ndarray,
    player_teams: np.ndarray,
    player_present: np.ndarray,
    keeper_defenders: np.ndarray,
    keeper_slots: np.ndarray,
    halves: Sequence[tuple[int, int]],
    pitch: tuple[float, float],
    fps: float = FPS,
    speed_threshold: float = 7.0,
    flight_lookahead: int = 120,
) -> np.ndarray:
    """Assign possession to the defending team on goalkeeper long restarts / goal kicks from dead balls.

    When the game restarts from a stoppage or dead ball inside the defending penalty area via a long kick
    by the goalkeeper or defender, the kick initiates possession for that defending team. During the ball's
    aerial flight, the possession belongs to the kicking team until the ball is received, contested or
    controlled downfield by another player, or goes out of play.
    """
    out_codes = np.asarray(codes, dtype=np.int8).copy()
    length, width = float(pitch[0]), float(pitch[1])
    b = np.asarray(consensus_ball, dtype=np.float64)
    pl = np.asarray(players, dtype=np.float64)
    pr = np.asarray(player_present, dtype=bool)
    tm = np.asarray(player_teams, dtype=np.int8)

    # Ball speed
    b_diff = np.zeros_like(b)
    b_diff[1:] = (b[1:] - b[:-1]) * fps
    b_spd = np.linalg.norm(b_diff, axis=1)
    b_pres = np.isfinite(b).all(axis=1)

    for h_idx, (h_start, h_stop) in enumerate(halves):
        h_start, h_stop = max(0, int(h_start)), min(len(out_codes), int(h_stop))
        for g_side in (0, 1):
            k_team = int(keeper_defenders[h_idx, g_side])
            k_slot = (
                int(keeper_slots[h_idx, g_side])
                if np.ndim(keeper_slots) == 2
                else int(keeper_slots[h_idx][g_side])
            )
            if k_team < 0 or k_slot < 0:
                continue
            goal_x = 0.0 if g_side == 0 else length

            for f in range(h_start + 10, h_stop - 100):
                if not b_pres[f]:
                    continue
                bx, by = b[f]
                # Within defensive penalty area / goal area (length <= 26m, width centered)
                if abs(bx - goal_x) > 26.0 or abs(by - width / 2.0) > 22.0:
                    continue

                # Follows a dead ball or unobserved ball within the last 25 frames
                was_dead = (out_codes[f - 1] == D) or np.any(out_codes[max(0, f - 25):f] == D) or not b_pres[f - 1]
                if not was_dead:
                    continue

                # The restart frame f must be the release point: ball accelerates into flight (speed >= 5.0 m/s)
                # while previous frames before release were stationary/resting (must remain D)
                if b_spd[f] < 5.0:
                    continue

                kp = pl[f, k_slot]
                if not np.isfinite(kp).all():
                    continue
                # Keeper is within 2.5m of the ball at launch
                if np.linalg.norm(kp - b[f]) > 2.5:
                    continue

                future_f = min(h_stop, f + 25)
                if future_f - f < 10:
                    continue
                valid_future = b_pres[f + 1:future_f]
                if not np.any(valid_future):
                    continue
                f_spd = float(np.mean(b_spd[f + 1:future_f][valid_future]))
                dx_future = b[future_f - 1, 0] - bx if b_pres[future_f - 1] else 0.0
                downfield = (dx_future > 8.0) if g_side == 0 else (dx_future < -8.0)

                if f_spd > speed_threshold and downfield:
                    # Trace flight until opponent control, tracking gap, or out of play
                    flight_end = min(h_stop, f + flight_lookahead)
                    for t in range(f + 25, flight_end):
                        if not b_pres[t]:
                            flight_end = t
                            break
                        # Opponent controls downfield (< 2.0m and ball slows)
                        pt = pl[t]
                        present_t = pr[t]
                        opp_mask = present_t & (tm == (1 - k_team))
                        if np.any(opp_mask):
                            opp_dists = np.linalg.norm(pt[opp_mask] - b[t], axis=1)
                            if np.min(opp_dists) <= 2.0 and b_spd[t] < 5.0:
                                flight_end = t
                                break
                    out_codes[f:flight_end] = k_team

    return out_codes


def fill_blind_continuity(
    codes: np.ndarray,
    confidence: np.ndarray,
    dead_support: np.ndarray | None = None,
    out_of_bounds: np.ndarray | None = None,
    max_frames: int = 75,
    max_confidence: float = 0.5,
) -> np.ndarray:
    """Keep a live possession through a short tracker blind spot.

    A high or fast ball leaves the optical tracker for a second or two without
    leaving play.  When no event or set-piece stoppage covers the gap, the ball
    confidence stays near zero, and the same team is on both sides of it, treating
    the gap as a transfer invents a turnover the evidence does not support
    (measured 51 646: 2.8 s of conf 0.00 while Barcelona kept the ball in play).
    Gaps bracketed by different teams, covered by a dead-ball window, crossing the
    touchline, or longer than ``max_frames`` are left untouched.
    """
    out = np.asarray(codes, dtype=np.int8).copy()
    n = out.size
    if n == 0:
        return out
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.size != n:
        raise ValueError("confidence must cover every code")
    blocked = np.zeros(n, dtype=bool)
    if dead_support is not None:
        blocked |= np.asarray(dead_support, dtype=bool)
    if out_of_bounds is not None:
        blocked |= np.asarray(out_of_bounds, dtype=bool)
    for start, stop in _bool_runs(out == D):
        if start == 0 or stop >= n or stop - start > max_frames:
            continue
        before, after = int(out[start - 1]), int(out[stop])
        if before != after or before not in (H, 1):
            continue
        window = conf[start:stop]
        if not np.isfinite(window).all() or float(window.max()) >= max_confidence:
            continue
        if blocked[start:stop].any():
            continue
        out[start:stop] = before
    return out


def honor_clean_interceptions(
    codes: np.ndarray,
    consensus_ball: np.ndarray,
    players: np.ndarray,
    teams: np.ndarray,
    present: np.ndarray,
    atd_ball_probability: np.ndarray,
    event_dead: np.ndarray,
    setpiece_dead: np.ndarray,
    pitch: tuple[float, float],
    touch_dist_m: float = 0.20,
    ball_min_speed: float = 8.0,
    ball_max_speed: float = 32.0,
    min_frames: int = 12,
    prior_win_frames: int = 20,
    prior_dist_m: float = 1.0,
    min_margin_m: float = 0.70,
    min_prob: float = 0.30,
    pitch_tol_m: float = 0.5,
    fps: float = FPS,
) -> np.ndarray:
    """Reclaim a live spell that a clean in-motion interception should have ended.

    The poll keeps the spell's team when the interception is silent: the ball is poked
    away at speed and never slows to the onset channel's floor, and at the re-acceleration
    no player sits inside the kick channel's attribution radius — so no event fires, and
    the smoothed baseline never sees a turnover.  The provided tracking still shows the
    whole affair at ball level; when it does with margin to rule out a loose-ball chase,
    the label follows the ball:

    * the stealing team is the nearest provided-tracking player for a sustained run of
      at least ``min_frames`` during which the whole run is live (on the pitch, tracked,
      outside dead and set-piece windows);
    * inside that run one of its players is at the ball's feet (``touch_dist_m``) while
      the ball is in firm motion — ``ball_min_speed`` to ``ball_max_speed`` m/s, the same
      physical ceiling the kick channel applies to discard tracking artifacts;
    * from that touch to the end of the run the stealing team stays nearest by at least
      ``min_margin_m`` — a clear, sustained take, not a marginal contest;
    * the ATD ball probability favours the stealing team over the spell's team by at
      least ``min_prob`` on average over the same stretch — the direct evidence, not the
      smoothed baseline, backs the switch;
    * the spell's team was at the ball within the last ``prior_win_frames`` (no closer
      than ``prior_dist_m``) — a steal from live opponent possession, not a pickup of a
      ball nobody owned.

    The label changes from the touch frame to the end of the run.  Measured 178 806:
    a poke at 0.14 m with the ball at 10 m/s that the onset floor (4 m/s) can never see.
    """
    out = np.asarray(codes, dtype=np.int8).copy()
    n = out.size
    if n == 0:
        return out
    players = np.asarray(players, dtype=np.float64)
    teams = np.asarray(teams, dtype=np.int8)
    present = np.asarray(present, dtype=bool)
    cball = np.asarray(consensus_ball, dtype=np.float64)
    prob = np.asarray(atd_ball_probability, dtype=np.float64)
    dead = np.asarray(event_dead, dtype=bool) | np.asarray(setpiece_dead, dtype=bool)

    onball = np.isfinite(cball).all(axis=1)
    onpitch = (
        (cball[:, 0] >= -pitch_tol_m)
        & (cball[:, 0] <= pitch[0] + pitch_tol_m)
        & (cball[:, 1] >= -pitch_tol_m)
        & (cball[:, 1] <= pitch[1] + pitch_tol_m)
    )
    valid = onball & onpitch & ~dead

    # consensus-ball speed: displacement between two consecutive valid frames
    speed = np.full(n, np.nan)
    pair = valid[1:] & valid[:-1]
    speed[1:][pair] = np.hypot(cball[1:, 0] - cball[:-1, 0], cball[1:, 1] - cball[:-1, 1])[pair] * fps

    # per-team closest distance and the nearest playing team's margin
    dH = np.full(n, np.inf)
    dA = np.full(n, np.inf)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        d = np.linalg.norm(players[s:e] - cball[s:e][:, None, :], axis=2)
        d = np.where(np.isfinite(d) & present[s:e], d, np.inf)
        if teams.ndim == 2:
            h = teams[s:e] == H
            a = teams[s:e] == 1
        else:
            h = (teams == H)[None, :]
            a = (teams == 1)[None, :]
        dH[s:e] = np.where(h.any(axis=1), np.where(h, d, np.inf).min(axis=1), np.inf)
        dA[s:e] = np.where(a.any(axis=1), np.where(a, d, np.inf).min(axis=1), np.inf)

    margin = np.where(dH <= dA, dA - dH, dH - dA)
    nearest = np.where(dH <= dA, H, 1)
    nearest[~np.isfinite(dH) & ~np.isfinite(dA)] = -1
    nearest[~valid] = -1

    for o in (H, 1):
        T = 1 - o
        d_o = dH if o == H else dA
        d_T = dA if o == H else dH
        for start, stop in _bool_runs(nearest == o):
            if stop - start < min_frames:
                continue
            if not np.all(out[start:stop] == T):
                continue
            if not np.all(valid[start:stop]):
                continue
            tf = start + int(np.argmin(d_o[start:stop]))
            if not np.isfinite(d_o[tf]) or d_o[tf] > touch_dist_m:
                continue
            v = speed[tf]
            if not np.isfinite(v) or not (ball_min_speed <= v <= ball_max_speed):
                continue
            if margin[tf:stop].min() < min_margin_m:
                continue
            if prob[tf:stop, o].mean() - prob[tf:stop, T].mean() < min_prob:
                continue
            pre = slice(max(0, start - prior_win_frames), start)
            if not np.isfinite(d_T[pre]).any() or float(d_T[pre].min()) > prior_dist_m:
                continue
            out[tf:stop] = o
    return out


def stitch_aerial_excursions(
    codes: np.ndarray,
    consensus_ball: np.ndarray,
    pitch: tuple[float, float],
    consensus_confidence: np.ndarray | None = None,
    setpiece_tags: np.ndarray | None = None,
    max_excursion_frames: int = 175,
    max_return_dist_m: float = 25.0,
    max_speed_ms: float = 12.0,
    fps: float = FPS,
) -> np.ndarray:
    """Stitch continuous possession across high aerial / vertical balloon kicks (balón alto/globo).

    In soccer, high aerial clearances or vertical balloons cause planar homography projections
    (assuming Z=0) to project the ball outside the pitch touchline (e.g. y < 0 or y > width)
    or cause camera pan/zoom to temporarily lose the ball from the field of view.
    When a ball was in play under possession of team T and undergoes an apparent dead-ball or
    out-of-bounds excursion, but returns back onto the pitch within a short window (<= max_excursion_frames)
    near the same vicinity (<= max_return_dist_m) with no active set-piece stoppage (SET PIECES)
    or referee whistle, the ball never actually left the field of play.
    This function bridges possession for team T through the aerial excursion up to the start
    of any subsequent genuine set-piece stoppage.
    """
    out_codes = np.asarray(codes, dtype=np.int8).copy()
    n = len(out_codes)
    length, width = float(pitch[0]), float(pitch[1])
    cb = np.asarray(consensus_ball, dtype=np.float64)
    conf = np.asarray(consensus_confidence, dtype=np.float64) if consensus_confidence is not None else None

    in_pitch = (cb[:, 0] >= -0.5) & (cb[:, 0] <= length + 0.5) & (cb[:, 1] >= -0.5) & (cb[:, 1] <= width + 0.5)
    valid_ball = np.isfinite(cb).all(axis=1) & in_pitch
    if conf is not None:
        valid_ball = valid_ball & (conf > 0.05)

    is_dead = (out_codes == D)
    for s, e in _bool_runs(is_dead):
        if s == 0:
            continue
        prev_team = out_codes[s - 1]
        if prev_team not in (H, 1):
            continue

        sp_start = None
        if setpiece_tags is not None:
            sp_slice = np.asarray(setpiece_tags[s:e], dtype=bool)
            if sp_slice.any():
                sp_start = s + int(np.argmax(sp_slice))

        check_e = sp_start if sp_start is not None else e
        if check_e <= s:
            continue

        p_start = None
        for k in range(s - 1, max(-1, s - 50), -1):
            if valid_ball[k]:
                p_start = cb[k]
                break
        if p_start is None:
            continue

        p_end = None
        return_frame = None
        for k in range(s, min(n, check_e + 75)):
            if valid_ball[k]:
                p_end = cb[k]
                return_frame = k
                break
        if p_end is None:
            continue

        dist = float(np.linalg.norm(p_end - p_start))
        flight_dur = return_frame - s
        if flight_dur <= 0:
            continue
        speed = dist / (flight_dur / fps)

        if flight_dur <= max_excursion_frames and (dist <= max_return_dist_m or speed <= max_speed_ms):
            out_codes[s:check_e] = prev_team

    return out_codes


def apply_offside_freekick_restarts(
    codes: np.ndarray,
    consensus_ball: np.ndarray,
    pitch: tuple[float, float],
    fps: float = FPS,
    min_dist_from_box: float = 12.0,
    min_still_frames: int = 20,
    still_speed_max: float = 0.8,
    kick_speed_min: float = 4.5,
    max_setup_frames: int = 600,
    max_touchline_m: float = 8.0,
) -> np.ndarray:
    """Delay possession assignment on offside / defensive free-kick restarts until the kick is struck.

    When play is stopped inside or near a defensive penalty area (e.g. goalkeeper catch / offside whistle),
    the defending player carries or walks the ball far out of the penalty area (>= 12 m from the goal/box)
    to the spot of the infringement (e.g. first quarter of the pitch). The ball is then placed stationary on
    the turf (speed < 0.8 m/s for >= 20 frames) while players set up for the indirect free kick.
    The ball remains dead (D) during this entire transit and placement phase, and only re-enters play
    when the free kick is actually struck (speed >= 4.5 m/s with significant displacement).

    The pattern must start *on the pitch*.  A throw-in or corner restart also looks like
    "ball travels >= 12 m, slows, gets struck again" -- but its D->team boundary sits on the
    chalk with the ball already in flight (measured 66 030: a throw-in at 0.05 m from the
    touchline was delayed to D for 3.2 s; 156 953: a corner-side throw-in at 1.81 m lost
    19 s of live play to a late slow patch).  A setup that walks the ball >= 12 m begins
    well inside the pitch (measured 22.7-26.4 m from the chalk); boundaries closer to
    ``max_touchline_m`` are side-restarts, not free-kick setups.
    """
    out_codes = np.asarray(codes, dtype=np.int8).copy()
    n = len(out_codes)
    length, width = float(pitch[0]), float(pitch[1])
    cb = np.asarray(consensus_ball, dtype=np.float64)
    b_diff = np.zeros_like(cb)
    b_diff[1:] = (cb[1:] - cb[:-1]) * fps
    b_spd = np.linalg.norm(b_diff, axis=1)

    diff = (out_codes != np.roll(out_codes, 1))
    diff[0] = False

    for s in np.flatnonzero(diff):
        if out_codes[s - 1] != D or out_codes[s] not in (H, 1):
            continue
        team = out_codes[s]
        e = s
        while e < n and out_codes[e] == team:
            e += 1

        p_start = cb[s]
        in_box = (p_start[0] <= 24.0) or (p_start[0] >= length - 24.0)
        if not in_box:
            continue
        # side-restarts start at the chalk: never model a throw-in as a free-kick setup
        touchline_m = min(float(p_start[1]), float(width - p_start[1]))
        if touchline_m < max_touchline_m:
            continue

        look_e = min(e, s + max_setup_frames)
        still_run = 0
        had_still_at_dist = False
        restart_k = None

        for k in range(s, look_e):
            dist_from_start = np.linalg.norm(cb[k] - p_start)
            if dist_from_start >= min_dist_from_box:
                if b_spd[k] < still_speed_max:
                    still_run += 1
                    if still_run >= min_still_frames:
                        had_still_at_dist = True
                else:
                    if had_still_at_dist:
                        if b_spd[k] >= kick_speed_min and np.nanmean(b_spd[k:min(n, k + 4)]) >= 3.0:
                            disp = float(np.linalg.norm(cb[min(n - 1, k + 25)] - cb[k]))
                            if disp >= 4.0:
                                restart_k = k
                                break
                    still_run = 0

        if restart_k is not None:
            out_codes[s:restart_k] = D

    return out_codes


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

    def gate(frame: int, team: int, dist: float, kind: str, support_window: range, loss: float = 0.0) -> TransitionEvent | None:
        cur = holder(frame)
        if team not in (H, 1) or cur < 0 or team == cur:
            return None
        if float(np.mean(event_dead[max(0, frame - 10):frame + 10])) > params.dead_max:
            return None
        idx = np.arange(max(0, support_window.start), min(n, support_window.stop))
        sup = float(np.mean(evidence_prob[idx, team] - evidence_prob[idx, cur])) if idx.size else 0.0
        # a ball killed onto an opponent's foot is a save or a tackle, not a lagging pool:
        # a stop at half a metre with the chase and tail checks behind it outranks support
        touch = kind == "onset" and dist <= params.onset_touch_dist and loss >= params.onset_touch_min_loss
        if sup < params.support_min and not touch:
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
        return TransitionEvent(
            start=start, frame=frame, kind=kind, team=team, distance=dist, support=sup,
            hold_through=touch and sup < params.support_min,
        )

    for frame, _gain, team, dist in kicks:
        ev = gate(int(frame), int(team), float(dist), "kick", range(int(frame), int(frame) + 50))
        if ev is None:
            dropped += 1
        else:
            kept.append(ev)
    for frame, loss, team, dist in onsets:
        ev = gate(int(frame), int(team), float(dist), "onset", range(int(frame) - 10, int(frame) + 40), float(loss))
        if ev is None:
            dropped += 1
        else:
            kept.append(ev)
    kept.sort(key=lambda e: (e.start, e.frame))
    return tuple(kept), dropped


RELAY_VOTERS = ("atd_ball", "yolo_ball", "video_players")


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


def _hold_pass_start(
    catch: int,
    team: int,
    side: int,
    ball_pitch: np.ndarray | None,
    pitchp: np.ndarray,
    teams: np.ndarray,
    present: np.ndarray,
    seen: np.ndarray,
    pitch: tuple[float, float],
    params: TransitionParams,
) -> int:
    """Walk a keeper catch back to the release of the pass that produced it.

    The fused spell can flip to the other team while the ball is travelling to the keeper
    (measured 19 814-19 858): the passer is beyond the possession boundary the gate already
    passed, so the pass itself is never admitted as a contact.  The camera, however, saw a
    clean flight: no opponent came within ``keeper_hold_pass_m`` of the ball between the
    release and the catch.  Return the frame where the ball was last under a same-team
    outfielder's control, or ``catch`` when there is no such release (a punt, a contested
    ball, or no video ball position).
    """
    if (
        ball_pitch is None
        or params.keeper_hold_pass_frames <= 0
        or params.keeper_hold_pass_m <= 0
    ):
        return catch
    length, width = float(pitch[0]), float(pitch[1])
    goal_x = 0.0 if side == 0 else length
    lo = max(0, catch - int(params.keeper_hold_pass_frames))
    anchor = catch
    in_control = False
    gap = 0
    for k in range(catch - 1, lo - 1, -1):
        point = ball_pitch[k]
        if not bool(seen[k]) or not np.isfinite(point).all():
            gap += 1
            if gap > int(params.keeper_hold_pass_gap):
                break
            continue
        gap = 0
        d_team = np.inf
        d_opp = np.inf
        for i in range(pitchp.shape[1]):
            if not bool(present[k, i]):
                continue
            px, py = float(pitchp[k, i, 0]), float(pitchp[k, i, 1])
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            other = int(teams[k, i])
            if other not in (H, 1):
                continue
            distance = float(np.hypot(px - float(point[0]), py - float(point[1])))
            if other == team:
                # the catcher's own area is the receiving end, not the passer
                if (
                    abs(px - goal_x) <= params.keeper_hold_depth_m
                    and abs(py - width / 2.0) <= params.keeper_hold_half_width_m
                ):
                    continue
                d_team = min(d_team, distance)
            else:
                d_opp = min(d_opp, distance)
        if d_opp <= params.keeper_hold_pass_m:
            break  # an opponent touched the ball: not a clean same-team pass
        if d_team <= params.keeper_hold_pass_m:
            anchor = k
            in_control = True
        elif in_control:
            break  # the control run ended here: that next frame is the release
    return anchor if in_control else catch


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
    ball_pitch = (
        np.asarray(evidence.ball_pitch, dtype=np.float64)
        if evidence.ball_pitch is not None
        else None
    )
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
                hx = float(pitchp[rs, int(holder[rs]), 0])
                side = 0 if hx < length / 2.0 else 1
                start = _hold_pass_start(
                    rs, team, side, ball_pitch, pitchp, teams, present, seen,
                    (length, width), params,
                )
                out.append(KeeperHold(start=start, frame=t - 1, team=team, length=t - rs))
        rs = -1
    return out


def keeper_area_catches(
    consensus: np.ndarray,
    speed: np.ndarray,
    valid: np.ndarray,
    event_dead: np.ndarray,
    player_sets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    defenders: np.ndarray,
    pitch: tuple[float, float],
    halves: tuple[tuple[int, int], ...],
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
    evidence_pool: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Ball stops inside the penalty area of the goal its defending side protects.

    The pixel twin (``keeper_holds``) goes blind right by the goal, where depth
    compresses the boxes and a gathered ball never trips the contain bits; the pitch
    geometry reads the same story: the ball decelerates from flight speed to a stop
    inside the defended area (``area_catch_*``) with no opponent near, then
    leaves again.  The catch belongs to the defending side (``defenders[h, g]``, the
    keeper-roster geometry of :mod:`possession_lab.keepers`) from the kick that sent
    the ball in — the nearest velocity jump searched back over
    ``area_catch_extend_max`` frames, so the pass to the keeper is the keeper's ball
    the whole way — through the clearance.  A keeper gathering the ball keeps his
    keeper's possession, not the side the lagging prior last saw it with (measured
    132 355: the Athletic keeper took the ball at x = 9.4 m and punted at 18.9 m/s
    while the fused prior kept the spell H).  Where ``evidence_pool`` (the raw
    voter pool) is given, its mean margin for the defending side over the catch
    must reach ``area_catch_min_pool_margin``: a ball that dies inside the area
    while the voters are neutral belongs to whatever the rest of the system
    already read (measured 11 598: the ball rolled out at the goal mouth with the
    pool at 0.07 -- a conceded goal, not a keeper's catch; without the guard it
    cost the correct A spell 11 552-11 627).  With the pool given the spell start
    anchors on the earliest frame whose support runs up to the catch (dips of at
    most ``area_catch_extend_dip_max`` frames tolerated through the flight);
    without it the start is the nearest velocity jump (``area_catch_kick_gain``)
    searched back over ``area_catch_extend_max`` frames.  Returns
    ``(start, catch, team)``.
    """
    length, width = float(pitch[0]), float(pitch[1])
    pool = None if evidence_pool is None else np.asarray(evidence_pool, dtype=np.float64)
    ball = np.asarray(consensus, dtype=np.float64)
    sp = np.asarray(speed, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    dead = np.asarray(event_dead, dtype=bool)
    n = ball.shape[0]
    half_of = np.full(n, -1, dtype=np.int8)
    for h, (a, b) in enumerate(halves):
        half_of[int(a) : int(b)] = h
    sets = [
        (np.asarray(p, dtype=np.float64), np.asarray(t), np.asarray(pr, dtype=bool))
        for p, t, pr in player_sets
    ]
    trigger = np.zeros(n, dtype=bool)
    team_at = np.full(n, -1, dtype=np.int8)
    for t in range(1, n):
        if not ok[t] or dead[t]:
            continue
        if float(np.mean(dead[max(0, t - 10): min(n, t + 10)])) > params.dead_max:
            continue
        if not np.isfinite(sp[t]) or sp[t] > params.area_catch_max_speed:
            continue
        w = sp[max(0, t - int(params.area_catch_prior_frames)) : t]
        if w.size == 0 or not np.isfinite(w).any() or float(np.nanmax(w)) < params.area_catch_in_min_speed:
            continue
        x, y = ball[t]
        if not np.isfinite([x, y]).all():
            continue
        side = 0 if x < length / 2.0 else 1
        goal_x = 0.0 if side == 0 else length
        if abs(x - goal_x) > params.area_catch_depth_m or abs(y - width / 2.0) > params.area_catch_half_width_m:
            continue
        h_i = int(half_of[t])
        if h_i < 0 or h_i >= defenders.shape[0]:
            continue
        team = int(defenders[h_i, side])
        if team not in (H, 1):
            continue
        d_min = np.inf
        for pitch_p, teams_p, present_p in sets:
            if t >= pitch_p.shape[0]:
                continue
            teams_t = teams_p[t] if teams_p.ndim == 2 else teams_p
            m = present_p[t] & (teams_t == (1 if team == H else H))
            if not np.any(m):
                continue
            with np.errstate(invalid="ignore"):
                dist = np.linalg.norm(pitch_p[t][m] - ball[t], axis=1)
            finite = dist[np.isfinite(dist)]
            if finite.size:
                d_min = min(d_min, float(finite.min()))
        if d_min < params.area_catch_opp_min_m:
            continue
        trigger[t] = True
        team_at[t] = team
    out: list[tuple[int, int, int]] = []
    for s, e in _bool_runs(trigger):
        catch = s
        team = int(team_at[catch])
        if pool is not None:
            seg = pool[s:e, team] - pool[s:e, 1 - team]
            if seg.size == 0 or float(np.mean(seg)) < params.area_catch_min_pool_margin:
                continue  # voters neutral while the ball lay there — not their keeper's catch
        start = catch
        lo = max(1, catch - int(params.area_catch_extend_max))
        if pool is not None:
            # the pass-in belongs to the catcher's side from the touch the voters
            # first backed, walking back over the flight's dips (measured 132 272:
            # the pool turned A when his player won the ball at x = 37 m, dipped
            # to 0.02 only while the ball was in the air towards the keeper)
            best, dips = catch, 0
            for k in range(catch - 1, lo - 1, -1):
                if pool[k, team] - pool[k, 1 - team] >= params.area_catch_extend_margin:
                    best, dips = k, 0
                else:
                    dips += 1
                    if dips > int(params.area_catch_extend_dip_max):
                        break
            start = best
        else:
            for k in range(catch, lo, -1):
                if (
                    np.isfinite(sp[k])
                    and np.isfinite(sp[k - 1])
                    and sp[k] - sp[k - 1] >= params.area_catch_kick_gain
                    and sp[k] >= params.area_catch_in_min_speed
                ):
                    start = k
                    break
        out.append((int(start), int(catch), team))
    return out


def expire_owned_spells(
    intervals: Sequence[Sequence[int]],
    evidence_pool: np.ndarray,
    event_dead: np.ndarray,
    speed: np.ndarray,
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
    protected: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[list[TransitionEvent], list[tuple[int, int, int]]]:
    """Cut pixel-owned spells the camera's own evidence no longer supports.

    An owned interval (contact/kicker/area) runs to the next fused boundary — made
    for the touch that opens play, not the touch that dies out: measured 132 743, a
    lone A touch at 132 619 owned 326 frames while the ball settled under a
    Barcelona player and the raw voter pool (``raw_evidence_pool``) insisted on H at
    0.80-0.94.  Where the pool dominates for the other team at probability ≥
    ``expire_min_prob`` and margin ≥ ``expire_min_margin`` with a slow ball
    (≤ ``expire_max_ball_speed`` m/s — control, not flight) for ≥ ``expire_min_frames``
    live frames fully inside an interval, the interval ends at the run's start and a
    ``control`` event marks the handover.  Keeper holds (``protected`` by
    ``(start, team)``) are never expired: there the camera literally watched the ball
    in the keeper's box.  Returns ``(events, intervals)``.
    """
    pool = np.asarray(evidence_pool, dtype=np.float64)
    dead = np.asarray(event_dead, dtype=bool)
    sp = np.asarray(speed, dtype=np.float64)
    ivs = [list(iv) for iv in intervals]
    found: list[tuple[int, int, int, int]] = []  # (interval index, run start, run end, taker)
    for i, (a, b, team) in enumerate(ivs):
        if (int(a), int(team)) in protected:
            continue
        if int(team) not in (H, 1):
            continue
        other = 1 if int(team) == H else H
        ok = (
            (pool[:, other] >= params.expire_min_prob)
            & (pool[:, other] - pool[:, int(team)] >= params.expire_min_margin)
            & (sp <= params.expire_max_ball_speed)
            & (~dead)
        )
        for s, e in _bool_runs(ok):
            if e - s < int(params.expire_min_frames):
                continue
            if a <= s and e <= b:
                found.append((i, s, e, other))
    events: list[TransitionEvent] = []
    for i, s, e, other in sorted(found, key=lambda r: (r[0], r[1])):
        iv = ivs[i]
        if iv[1] <= s:
            continue  # an earlier run already ended this interval
        margin = float(np.mean(pool[s:e, other] - pool[s:e, int(iv[2])]))
        iv[1] = s
        events.append(
            TransitionEvent(start=s, frame=s, kind="control", team=other,
                            distance=1.0, support=min(1.0, margin))
        )
    return events, [(a, b, int(t_)) for a, b, t_ in ivs if b > a]


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
    atd_ball: np.ndarray | None = None,
    pitch_close: np.ndarray | None = None,
    pool: np.ndarray | None = None,
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
    And a contact never beats a player standing on the ball: where ``pitch_close``
    gives it, if the other side's nearest player is ``contact_opp_pitch_margin``
    metres closer to the ball than the containing side's, the box swallowed the
    pixel at depth and the contact is dropped.

    Events are processed in frame order.  A team that already owns the stream cannot
    re-anchor, and admitting the other team's flip ends the previous interval.  Each
    kept event owns ``[start, stop)``: the contact extend-back (or one frame before a
    kick) up to the next fused-code boundary, a stronger later pixel event, or the
    length cap (``pixel_cap_solid`` for solid contacts, else ``pixel_cap``), whichever
    comes first.  Returns ``(kept events, owned intervals)``.
    """
    n = codes_pooled.shape[0]
    bnd = np.flatnonzero(np.r_[True, codes_pooled[1:] != codes_pooled[:-1]])
    atd_seen = np.ones(n, dtype=bool) if atd_ball is None else np.asarray(atd_ball, dtype=bool)
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
        cur = codes_pooled[max(0, frame - 20):frame]
        cur = cur[(cur == H) | (cur == 1)]
        holder = int(np.bincount(cur).argmax()) if cur.size else -1
        for a, b, t_ in intervals:
            if a <= frame < b:
                holder = int(t_)
        if is_relay:
            pass  # cross-stream attested: boundary and momentum do not apply
        elif min(frame - prev_b, next_b - frame) > params.pixel_bound_window:
            holder_bit = 1 if holder == H else 2
            lo = max(0, frame - params.box_lock_clear_frames)
            if not (
                holder in (H, 1)
                and holder != team
                and is_contact
                and strength >= params.box_lock_min_run
                and not (np.asarray(contain)[lo:frame] & holder_bit).any()
            ):
                continue  # mid-spell: settled possession, boxes confound depth
        else:
            hf = min(n - 1, frame + params.pixel_momentum_horizon)
            momentum = float(fused_prob[hf, team]) >= params.pixel_momentum_min
            fingerprint = (
                not is_contact
                and close_px <= params.pixel_fingerprint_close
                and dom_px >= params.pixel_fingerprint_dom
                and strength <= params.pixel_fingerprint_release_max
                and (not params.pixel_fingerprint_needs_atd or bool(atd_seen[frame]))
            )
            if not (momentum or fingerprint):
                continue  # fused evidence never comes around and no winning fingerprint: noise
        if holder < 0 or holder == team:
            continue
        if float(np.mean(event_dead[max(0, frame - 10):frame + 10])) > params.dead_max:
            continue
        if is_contact and pitch_close is not None and not is_relay:
            d = pitch_close[min(frame, pitch_close.shape[0] - 1)]
            if np.isfinite(d).all():
                if float(d[team]) > 4.5:
                    continue  # depth perspective artifact: player is >4.5m from the ball on the grass
                if d[1 - team] + params.contact_opp_pitch_margin <= d[team]:
                    contradicted = True
                    if pool is not None and params.contact_opp_pool_margin > 0:
                        pm = pool[min(frame, pool.shape[0] - 1)]
                        contradicted = float(pm[1 - team]) - float(pm[team]) >= params.contact_opp_pool_margin
                    if contradicted:
                        continue  # the other side stands on the ball and the voters agree
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
    # keeper holds own the stretch from the catch to the next fused boundary or to the release
    # cap: the camera watched the ball sit in the defender's box, so this needs no boundary
    # proximity and no momentum, but it never cuts into a frame the pixel channel already decided.
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
        cap = int(params.keeper_hold_release_frames)
        stop = min(next_b, frame + cap) if cap > 0 else next_b
        if stop <= start:
            continue
        kept.append(
            TransitionEvent(start=start, frame=frame, kind="hold", team=team,
                            distance=1.0, support=1.0)
        )
        intervals.append([start, stop, team])
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
        if ev.kind in ("hold", "area"):
            strength = params.contact_pulse_solid if ev.kind == "area" else params.contact_pulse
        elif ev.kind == "control":
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


def dead_needs_support(
    codes: np.ndarray,
    blind: np.ndarray,
    support: np.ndarray,
    halves: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """Inside a ball-blind window, ``D`` has to come from an event, never from the gap itself.

    The deeper the ATD ball gap, the more the decode reads the pitch shape around the *last*
    known ball position: a long enough hole eventually looks like a ball parked at a touchline,
    whatever the match was doing (measured 86 307: the boundary crossing 120 blind frames is
    what flipped the state, and the only reason attached to it was "the tracker lost the ball").
    A missing ball is not a stoppage.  Keep the dead state only where independent evidence says
    the game was stopped -- a confirmed whistle/out-of-play stoppage from
    :mod:`possession_lab.events`, or a ``SET PIECES`` smart tag -- and hand every other blind
    ``D`` back to the last live team the decode held in the same half.  Frames before that half
    ever held a live team have nothing to go back to, and are left as they were.
    """
    out = np.asarray(codes).copy()
    mask = np.asarray(blind, dtype=bool)
    supported = np.asarray(support, dtype=bool)
    if out.size == 0 or not mask.any() or supported.size != out.size:
        return out
    if halves is None:
        halves = GRADED_HALVES
    for start, stop in halves:
        start, stop = int(start), int(min(stop, out.size))
        if start >= stop:
            continue
        segment = out[start:stop]
        # last index of a live (H/A) state at or before each frame; -1 while the half has only
        # ever been D/X, where there is no team to hand the frames back to
        live_index = np.maximum.accumulate(np.where((segment == H) | (segment == 1), np.arange(segment.size), -1))
        bad = mask[start:stop] & (segment == D) & ~supported[start:stop] & (live_index >= 0)
        if not bad.any():
            continue
        segment[bad] = segment[live_index[bad]]
        out[start:stop] = segment
    return out


def hold_touch_through(
    codes: np.ndarray,
    events: Sequence[TransitionEvent],
    event_dead: np.ndarray,
    dead_support: np.ndarray,
    params: TransitionParams = DEFAULT_TRANSITION_PARAMS,
) -> np.ndarray:
    """A witnessed takeover survives *decoded* deadness until a supported stoppage.

    A keeper gathering a save kills the ball's speed for a long stretch: the motion
    features read that as a dead ball while the game is still playing (measured 17 273:
    the save, ~70 frames of crawl, the stoppage tag only at 17 377).  Where a
    ``hold_through`` event (an onset that passed the gate on the ball-killed-at-the-
    taker's-foot evidence alone) owns the spell, hand unsupported ``D`` back to the
    event team until the first stoppage ``dead_support`` frame, capped at
    ``onset_touch_hold_frames``.  A tagged stoppage, another event, or the ball living
    with the other team all end the hold first.
    """
    out = np.asarray(codes).copy()
    n = out.size
    if n == 0 or params.onset_touch_hold_frames <= 0:
        return out
    tagged = np.asarray(event_dead, dtype=bool)
    supported = np.asarray(dead_support, dtype=bool)
    if supported.size != n or tagged.size != n:
        return out
    holders = [e for e in events if e.hold_through]
    if not holders:
        return out
    starts = sorted(e.start for e in events)
    for event in holders:
        start = max(0, min(int(event.start), int(event.frame)))
        end = min(n, int(event.frame) + int(params.onset_touch_hold_frames))
        after = np.flatnonzero(supported[int(event.frame) + 1:end])
        if after.size:
            end = int(event.frame) + int(after[0]) + 1
        end = min(end, min((s for s in starts if s > event.start), default=end))
        for t in range(start, end):
            if tagged[t]:
                break  # the tag is the authority: play is stopped from here
            current = out[t]
            if current in (H, 1) and current != event.team:
                break  # the other team owns the live ball already
            if current == D:
                out[t] = event.team
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
    atd_ball_present: np.ndarray | None = None,
    dead_support: np.ndarray | None = None,
    evidence_pool: np.ndarray | None = None,
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
    settled spell.  ``atd_ball_present`` tells the pixel channel which frames the pitch
    stream actually saw the ball on; ``dead_support`` (a confirmed stoppage or a
    ``SET PIECES`` smart tag) is what a ``D`` inside a ball-blind window must be backed by.
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
            event_dead, events, params, relay=relay, holds=holds, atd_ball=atd_ball_present,
            pitch_close=_pitch_nearest_m(box_evidence),
            pool=None if evidence_pool is None else np.asarray(evidence_pool, dtype=np.float64),
        )
        intervals = [list(iv) for iv in intervals]
        events = tuple(sorted(events + tuple(pixel_kept), key=lambda e: (e.start, e.frame)))
        dropped += len(contact) + len(kicker) + len(holds) - len(pixel_kept)
        n_contact, n_kicker, n_hold = len(contact), len(kicker), sum(1 for e in pixel_kept if e.kind == "hold")
        hold_keys = frozenset((int(e.start), int(e.team)) for e in pixel_kept if e.kind == "hold")
    else:
        n_contact = n_kicker = n_hold = 0
        intervals: list[tuple[int, int, int]] = []
        hold_keys: frozenset[tuple[int, int]] = frozenset()
    n_area = 0
    if keeper_defenders is not None and keeper_pitch is not None:
        # the pitch twin of the keeper hold: a stop inside the defended area that the
        # blind near-goal pixel channel never saw belongs to the side that owns goal
        bnd = np.flatnonzero(np.r_[True, codes_pooled[1:] != codes_pooled[:-1]])
        for start, frame, team in keeper_area_catches(
            consensus, speed, valid, np.asarray(event_dead, dtype=bool), sets64,
            keeper_defenders, keeper_pitch, halves if halves is not None else GRADED_HALVES, params,
            evidence_pool=None if evidence_pool is None else np.asarray(evidence_pool, dtype=np.float64),
        ):
            if any(a <= frame < b and t_ == team for a, b, t_ in intervals):
                continue  # the pixel channel already owns the catch with the same team
            if any((a <= frame < b) and (a, int(t_)) in hold_keys for a, b, t_ in intervals):
                continue  # a keeper hold literally watched the ball in the box
            k = int(np.searchsorted(bnd, start, side="right"))
            next_b = int(bnd[k]) if k < bnd.size else int(codes_pooled.shape[0])
            stop = min(next_b, start + int(params.area_catch_max_span))
            for iv in intervals:
                if int(iv[2]) == team:
                    continue
                if iv[1] > start and iv[0] < stop and (iv[0], int(iv[2])) not in hold_keys:
                    iv[1] = start  # the catch takes the spell over from its front edge
            intervals = [iv for iv in intervals if iv[1] > iv[0]]
            if not any(a <= frame < b and t_ != team for a, b, t_ in intervals):
                intervals.append([start, stop, team])
                events = tuple(sorted(
                    (*events,
                     TransitionEvent(start=start, frame=frame, kind="area", team=team,
                                     distance=1.0, support=1.0)),
                    key=lambda e: (e.start, e.frame),
                ))
                n_area += 1
    n_control = 0
    if evidence_pool is not None:
        control, intervals = expire_owned_spells(
            intervals, np.asarray(evidence_pool, dtype=np.float64),
            np.asarray(event_dead, dtype=bool), speed, params, hold_keys,
        )
        n_control = len(control)
        if control:
            events = tuple(sorted(events + tuple(control), key=lambda e: (e.start, e.frame)))
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
    if dead_support is not None and ball_stale is not None:
        # a ball the tracker cannot see never *becomes* a dead ball on the strength of the gap
        codes = dead_needs_support(codes, np.asarray(ball_stale, dtype=bool), dead_support, halves)
    if blind_dead is not None:
        # a dead ball that went blind stays dead: the "restart" inside the window is the
        # carried trajectory, not play (post-decode so a window cannot disturb the half)
        codes = carry_dead_through_blind(codes, blind_dead, halves)
    if params.onset_touch_hold_frames > 0:
        # a witnessed takeover survives decoded deadness until the stoppage tag supports it
        codes = hold_touch_through(
            codes, events, np.asarray(event_dead, dtype=bool),
            np.asarray(dead_support if dead_support is not None else event_dead, dtype=bool),
            params,
        )
    from ..pipeline import fill_match_breaks

    codes = fill_match_breaks(codes, halves if halves is not None else GRADED_HALVES)
    return TransitionLayerResult(
        codes=codes,
        events=events,
        n_kicks=len(kicks),
        n_onsets=len(onsets),
        n_gated_out=dropped,
        n_contact=n_contact,
        n_kicker=n_kicker,
        n_hold=n_hold,
        n_control=n_control,
        n_area=n_area,
    )


def events_to_rows(events: Sequence[TransitionEvent]) -> np.ndarray:
    """Events as an ``(m, 6)`` array: start, frame, kind (0=kick, 1=onset, 2=contact,
    3=kicker, 4=hold, 5=control, 6=area), team, distance, support."""
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
    "ball_lost_mask",
    "ball_speed",
    "ball_stale_mask",
    "build_pulse",
    "carry_dead_through_blind",
    "control_map",
    "dead_needs_support",
    "dead_revival_mask",
    "decode_with_events",
    "detect_contact_events",
    "detect_kick_events",
    "detect_kicker_events",
    "detect_onset_events",
    "events_to_rows",
    "evidence_probability",
    "expire_owned_spells",
    "fill_blind_continuity",
    "gate_events",
    "gate_pixel_events",
    "honor_clean_interceptions",
    "hold_touch_through",
    "keeper_area_catches",
    "keeper_holds",
    "KeeperHold",
    "apply_goalkeeper_restarts",
    "apply_offside_freekick_restarts",
    "stitch_keeper_saves",
    "stitch_aerial_excursions",
    "out_of_bounds_mask",
    "possession_revival_mask",
    "pixel_contact",
    "raw_evidence_pool",
]
