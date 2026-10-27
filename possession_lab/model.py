"""A three-state, gap-aware hidden Markov model for possession.

States: H (home), A (away), D (dead ball / out of play).  The emissions are deliberately simple:

* when the ball is visible, a Gaussian in the distance to each team's nearest player;
* D gets a sigmoid on the nearest-player distance plus an out-of-play bonus;
* when the ball is missing, short gaps are uniform (the transition prior carries the state);
* long gaps use the team shape around the last known ball position, the last known edge, and a
  weak smart-tag vote;
* a ball far from every player is treated as loose/in flight: uniform, or the last-touch team when
  the ball is also slow, so distance alone cannot vote a live ball dead.

The Viterbi pass is CPU-only.  It is a 199k-step, 3-state dynamic program; a GPU would spend more
time copying arrays than computing, so the package does not pretend otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FPS, MODEL_PARAMS, X


@dataclass(frozen=True)
class ModelParams:
    sigma: float = 3.5
    d_dead: float = 7.0
    dead_scale: float = 1.5
    edge0: float = 5.0
    edge_scale: float = 1.5
    w_edge: float = 0.0
    w_gap: float = 0.0
    gap0: float = 100.0
    gap_scale: float = 25.0
    w_out: float = 1.0
    w_dead: float = 0.0
    w_motion: float = 0.0
    motion0: float = 1.6
    motion_scale: float = 0.35
    motion_control_R: float = 4.0
    w_whistle: float = 0.0
    whistle_control_R: float = 3.0
    w_est_out: float = 0.0
    est_blend: float = 0.0
    est_sigma: float = 4.0
    flight_R: float = 0.0
    flight_speed: float = 10.0
    flight_p: float = 0.55
    w_air: float = 0.0
    w_tag: float = 0.0
    w_sp: float = 0.0
    w_lt: float = 0.0
    lt_R: float = 0.8
    p_stay: float = 0.995
    p_ha: float = 0.0005
    p_hd: float = 0.001
    p_dh: float = 0.01
    gate: float = 0.0
    loose_R: float = 7.0
    loose_speed: float = 2.0
    touch_age: int = 125
    touch_p: float = 0.7
    miss_mode: str = "hybrid"
    sigma_miss: float = 6.0
    gap_thresh: float = 120.0
    min_players: int = 6
    w_parked_edge: float = 0.0
    loose_stop_gate: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None = None) -> ModelParams:
        if values is None:
            return cls(**MODEL_PARAMS)
        merged = dict(MODEL_PARAMS)
        merged.update(values)
        return cls(**merged)


DEFAULT_PARAMS = ModelParams.from_mapping()


@dataclass(frozen=True)
class DecodeResult:
    codes: np.ndarray
    posterior: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _log_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    normalizer = values.sum(axis=1, keepdims=True)
    normalizer = np.where(normalizer > 0, normalizer, 1.0)
    return np.log(values / normalizer)


def dead_ball_evidence(features: Any, params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    """Additive support for the D state from stoppage evidence.

    This is deliberately separate from the team-distance emission.  Players walk or stand still
    during a stoppage, and the ball sits near a touchline for throw-ins, corners and goal kicks.
    ``w_dead`` is a base-rate offset: it keeps the dead-ball prior independent of the scale of the
    team Gaussians.

    Team motion is only evidence when no player is actually controlling the ball.  A player
    standing still with the ball is live possession, and the old ungated term called that "Balón
    muerto" (the 18,389 false D is the clearest example).
    """
    edge = np.where(np.isfinite(features.edge), features.edge, features.last_edge)
    edge = np.where(np.isfinite(edge), edge, 20.0)
    present = np.asarray(features.present, dtype=bool)
    near_distance = np.asarray(features.near_distance, dtype=np.float64)
    edge_now = np.where(np.isfinite(features.edge), features.edge, np.inf)
    stop = np.asarray(getattr(features, "stop_dead", np.zeros(present.size, dtype=bool)), dtype=bool)
    controlled = (
        present
        & np.isfinite(near_distance)
        & (near_distance <= params.motion_control_R)
        & (edge_now > 2.5)
        & ~stop
    )
    motion = _sigmoid(
        (params.motion0 - np.asarray(features.team_speed, dtype=np.float64)) / params.motion_scale
    )
    motion = np.where(controlled, 0.0, motion)
    return (
        params.w_dead
        + params.w_motion * motion
        + params.w_edge * _sigmoid((params.edge0 - edge) / params.edge_scale)
    )


def emission_log_probs(features: Any, params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    """Return an ``(n, 3)`` array of log emission probabilities."""
    n = len(np.asarray(features.present))
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    d_home = np.asarray(features.d_home, dtype=np.float64)
    d_away = np.asarray(features.d_away, dtype=np.float64)
    present = np.asarray(features.present, dtype=bool)
    last_edge = np.asarray(features.last_edge, dtype=np.float64)
    gap = np.asarray(features.gap, dtype=np.float64)
    out_of_play = np.asarray(features.out_of_play, dtype=bool)

    finite_home = np.isfinite(d_home)
    finite_away = np.isfinite(d_away)
    q_home = np.where(
        finite_home,
        np.exp(-(d_home * d_home) / (2.0 * params.sigma * params.sigma)),
        1e-12,
    ) + 1e-12
    q_away = np.where(
        finite_away,
        np.exp(-(d_away * d_away) / (2.0 * params.sigma * params.sigma)),
        1e-12,
    ) + 1e-12

    d_min = np.minimum(d_home, d_away)
    finite_min = np.isfinite(d_min)
    q_dead = np.where(
        finite_min,
        _sigmoid((d_min - params.d_dead) / params.dead_scale),
        0.0,
    ) + 1e-12

    q_dead += dead_ball_evidence(features, params)
    if params.w_gap:
        q_dead += params.w_gap * _sigmoid((gap - params.gap0) / params.gap_scale)
    q_dead += params.w_out * out_of_play

    parked = np.asarray(getattr(features, "parked_edge", False), dtype=bool)
    if params.w_parked_edge:
        q_dead = q_dead + params.w_parked_edge * parked

    log_emission = _log_normalize(np.stack([q_home, q_away, q_dead], axis=1))

    # Speed gate.  The reference model applies this even when ``gate`` is zero: a stationary or
    # unknown ball speed then gets a 50/50 blend with the uniform emission.  That is not a no-op,
    # so keep the operation unconditional to reproduce the validated configuration.
    speed = np.nan_to_num(np.asarray(features.ball_speed, dtype=np.float64), nan=0.0)
    alpha = 1.0 - _sigmoid((speed - params.gate) / 2.0)
    alpha = np.where(present, alpha, 0.0)
    log_emission = alpha[:, None] * log_emission + (1.0 - alpha[:, None]) * np.log(1.0 / 3.0)

    log_emission[~present] = np.log(1.0 / 3.0)
    log_emission[out_of_play] = np.log(np.array([1e-6, 1e-6, 1.0 - 2e-6]))

    if params.w_tag:
        vote = np.asarray(features.tag_vote, dtype=np.float64)
        log_emission[:, 0] += params.w_tag * np.maximum(vote, 0.0)
        log_emission[:, 1] += params.w_tag * np.maximum(-vote, 0.0)
    if params.w_sp:
        log_emission[:, 2] += params.w_sp * np.asarray(features.tag_setpiece, dtype=np.float64)

    if params.w_lt:
        touch = (
            present
            & (np.asarray(features.last_touch) >= 0)
            & (np.asarray(features.near_distance, dtype=np.float64) <= params.lt_R)
        )
        touch_team = np.asarray(features.last_touch, dtype=np.int8)
        log_emission[:, 0] += params.w_lt * touch * (touch_team == 0)
        log_emission[:, 1] += params.w_lt * touch * (touch_team == 1)

    # Missing-ball emissions.  The hybrid mode is the recommendation: keep short holes uniform so
    # the transition prior does the work, and switch to the geometric prior only for long holes.
    missing = ~present
    if missing.any():
        if params.miss_mode == "persist":
            log_emission[missing] = np.log(1.0 / 3.0)
        else:
            missing_q = np.ones((n, 3), dtype=np.float64)
            if params.miss_mode in {"prox", "both", "hybrid", "hybrid_prox"}:
                d_home_last = np.nan_to_num(
                    np.asarray(features.d_home_last, dtype=np.float64), nan=1e6
                )
                d_away_last = np.nan_to_num(
                    np.asarray(features.d_away_last, dtype=np.float64), nan=1e6
                )
                missing_q[:, 0] = (
                    np.exp(-(d_home_last * d_home_last) / (2.0 * params.sigma_miss * params.sigma_miss))
                    + 1e-6
                )
                missing_q[:, 1] = (
                    np.exp(-(d_away_last * d_away_last) / (2.0 * params.sigma_miss * params.sigma_miss))
                    + 1e-6
                )
                last_edge_safe = np.where(np.isfinite(last_edge), last_edge, 20.0)
                missing_q[:, 2] = 0.15 + _sigmoid((params.edge0 - last_edge_safe) / 1.5)
            if params.miss_mode in {"tag", "both", "hybrid"}:
                vote = np.asarray(features.tag_vote, dtype=np.float64)
                missing_q[:, 0] *= 1.0 + np.maximum(vote, 0.0)
                missing_q[:, 1] *= 1.0 + np.maximum(-vote, 0.0)
                missing_q[:, 2] *= 1.0 + 0.2 * np.asarray(features.tag_setpiece, dtype=np.float64)
            missing_q = missing_q / missing_q.sum(axis=1, keepdims=True)
            log_emission[missing] = np.log(missing_q[missing])
            if params.miss_mode in {"hybrid", "hybrid_prox"}:
                short = missing & (gap <= params.gap_thresh)
                log_emission[short] = np.log(1.0 / 3.0)

    # If the tracking feed itself is almost empty, do not read team shape from three lonely
    # away slots.  Treat the frame as missing evidence and let the transition prior persist.
    if params.min_players > 0 and hasattr(features, "player_count"):
        sparse = missing & (np.asarray(features.player_count) < params.min_players)
        log_emission[sparse] = np.log(1.0 / 3.0)

    # A live-play smart tag is evidence that the game is continuing, even if the ball tracker has
    # lost the ball.  Using the last position here is what changed possession in the 17,230 hole:
    # the ball was last seen near Barcelona players but Atletic still had it.  Persist the previous
    # state instead of reading team shape through a live-play gap.  Confirmed stoppages are applied
    # as hard overrides after decoding, so this does not hide a real dead ball.
    if hasattr(features, "tag_live"):
        live_missing = missing & np.asarray(features.tag_live, dtype=bool)
        log_emission[live_missing] = np.log(1.0 / 3.0)

    # A ball that is far from every player is loose or in flight, not evidence of a dead ball.
    # Let the prior carry the possession instead of letting the distance term vote `D`.  When
    # that loose ball is also slow, its only possession evidence is the team that touched it
    # last, so give that team the mass and leave the rest for the prior.
    if params.loose_R > 0:
        loose = present & (d_min > params.loose_R) & ~out_of_play
        if params.loose_stop_gate:
            # A ball that sits at a boundary is a restart waiting to be taken, not a live loose
            # ball.  Do not hand it to the prior or to last-touch.
            loose &= ~parked
        log_emission[loose] = np.log(1.0 / 3.0)
        if params.touch_p > 0:
            last_touch = np.asarray(features.last_touch, dtype=np.int8)
            recent = loose & (speed < params.loose_speed) & (last_touch >= 0)
            recent = recent & (
                np.asarray(features.last_touch_age, dtype=np.int64) <= params.touch_age
            )
            if recent.any():
                touched = last_touch[recent]
                q = np.full((touched.size, 3), (1.0 - params.touch_p) / 2.0)
                q[np.arange(touched.size), touched] = params.touch_p
                log_emission[recent] = np.log(q)

    # A fast ball away from every player is in flight.  Distance-to-player is then not evidence
    # that the *other* team has possession: a 20 m/s pass spends most of its travel time closer to
    # the receiving team than the passer, and distance-only emissions flip possession in mid-air.
    # Keep the last-touch team (or the prior when the touch is unknown) instead.
    if params.flight_R > 0:
        flight = (
            present
            & ~out_of_play
            & (speed >= params.flight_speed)
            & (d_min >= params.flight_R)
        )
        if flight.any():
            last_touch = np.asarray(features.last_touch, dtype=np.int8)
            touch_age = np.asarray(features.last_touch_age, dtype=np.int64)
            touched = flight & (last_touch >= 0) & (touch_age <= params.touch_age)
            q = np.full((int(flight.sum()), 3), 0.5)
            q[:, 2] = 1e-3
            local_touched = touched[flight]
            if local_touched.any():
                q[local_touched, 2] = 1e-3
                q[local_touched, :2] = (1.0 - params.flight_p) / 2.0
                q[np.flatnonzero(local_touched), last_touch[touched]] = params.flight_p
            q /= q.sum(axis=1, keepdims=True)
            log_emission[flight] = np.log(q)

    # Aerial duels: causal flight frames where ``aerial.launch_track`` froze the launching
    # team.  The ATD ball is a ground projection, so while the ball is in the air the nearest
    # player is often one *under* the flight and the proximity Gaussian actively votes for the
    # wrong team; the speed gate above has already erased most of that vote, and this restores
    # it to the last toucher.  Support only the launch team and leave D's mass untouched;
    # out-of-play frames are a hard override and get no bonus.
    if params.w_air and hasattr(features, "air"):
        air = np.asarray(features.air, dtype=bool)
        if air.size == n and air.any():
            launch_team = np.asarray(features.air_team, dtype=np.int8)
            dead = (
                np.asarray(getattr(features, "stop_dead", False), dtype=bool)
                | np.asarray(getattr(features, "goal_dead", False), dtype=bool)
                | np.asarray(getattr(features, "parked_edge", False), dtype=bool)
            )
            hit = air & present & ~out_of_play & ~dead & (launch_team >= 0)
            if hit.any():
                rows = np.flatnonzero(hit)
                log_emission[rows, launch_team[hit]] += params.w_air

    # Tracker reconstruction for short holes.  The tracker only fills up to a bounded time and
    # its confidence decays with age, so this cannot invent possession across a long dead-ball gap.
    if params.est_blend > 0 and missing.any() and hasattr(features, "est_confidence"):
        est_conf = np.nan_to_num(
            np.asarray(features.est_confidence, dtype=np.float64), nan=0.0
        )
        blend = np.clip(params.est_blend * est_conf, 0.0, 0.95)
        valid = missing & (blend > 0.0)
        if valid.any():
            est_home = np.nan_to_num(
                np.asarray(features.est_d_home, dtype=np.float64), nan=1e6
            )
            est_away = np.nan_to_num(
                np.asarray(features.est_d_away, dtype=np.float64), nan=1e6
            )
            est_out = np.asarray(features.est_out, dtype=bool)
            est_q = np.empty((int(valid.sum()), 3), dtype=np.float64)
            est_q[:, 0] = np.exp(
                -(est_home[valid] ** 2) / (2.0 * params.est_sigma * params.est_sigma)
            ) + 1e-9
            est_q[:, 1] = np.exp(
                -(est_away[valid] ** 2) / (2.0 * params.est_sigma * params.est_sigma)
            ) + 1e-9
            est_q[:, 2] = 1e-4 + 4.0 * est_out[valid] + 0.02
            est_q /= est_q.sum(axis=1, keepdims=True)
            base = np.exp(log_emission[valid])
            base /= fb_sum(base)
            weight = blend[valid, None]
            log_emission[valid] = np.log(weight * est_q + (1.0 - weight) * base)

    # Whistle support, only where no player is controlling a visible ball and only as an additive
    # D nudge.  The confirmed stoppage intervals are applied as a hard override in the pipeline;
    # this term keeps the Viterbi state sensible around the edges of those intervals.
    if params.w_whistle > 0 and hasattr(features, "whistle_env"):
        env = np.asarray(features.whistle_env, dtype=np.float64)
        controlled = (
            present
            & np.isfinite(np.asarray(features.near_distance, dtype=np.float64))
            & (np.asarray(features.near_distance, dtype=np.float64) <= params.whistle_control_R)
            & ~out_of_play
        )
        low_motion = _sigmoid((1.8 - np.asarray(features.team_speed, dtype=np.float64)) / 0.35)
        live = np.asarray(getattr(features, "tag_live", np.zeros(n, dtype=bool)), dtype=bool)
        log_emission[:, 2] += params.w_whistle * env * (~controlled) * low_motion * (1.0 - 0.5 * live)

    if params.w_est_out > 0 and hasattr(features, "est_out"):
        est_out = np.asarray(features.est_out, dtype=bool)
        est_conf = np.asarray(features.est_confidence, dtype=np.float64)
        log_emission[:, 2] += params.w_est_out * est_out * (est_conf > 0.05)

    return log_emission


def fb_sum(values: np.ndarray) -> np.ndarray:
    """Row sums with a floor, kept local so model.py has no forward-backward dependency."""
    total = values.sum(axis=1, keepdims=True)
    return np.where(total > 0, total, 1.0)


def transition_matrix(params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    return np.array(
        [
            [params.p_stay, params.p_ha, params.p_hd],
            [params.p_ha, params.p_stay, params.p_hd],
            [params.p_dh, params.p_dh, 1.0 - 2.0 * params.p_dh],
        ],
        dtype=np.float64,
    )


def viterbi(log_emission: np.ndarray, params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    """Standard Viterbi for three states, using log-space arithmetic."""
    log_emission = np.asarray(log_emission, dtype=np.float64)
    n = log_emission.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    log_transition = np.log(transition_matrix(params))
    back = np.zeros((n, 3), dtype=np.int8)
    score = np.log(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])) + log_emission[0]
    for t in range(1, n):
        candidates = score[:, None] + log_transition
        back[t] = np.argmax(candidates, axis=0)
        score = np.max(candidates, axis=0) + log_emission[t]
    states = np.zeros(n, dtype=np.int8)
    states[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def merge_isolated_runs(codes: np.ndarray, min_frames: int = FPS) -> np.ndarray:
    """Absorb brief label flips that return to the same label.

    Viterbi can drop a possession for a few frames when the ball is between players, in flight,
    or briefly untracked.  A run shorter than ``min_frames`` whose neighbours agree is a flicker,
    not a new possession, so the neighbours take it back.  ``X`` frames are never changed.
    """
    out = np.asarray(codes, dtype=np.int8).copy()
    if min_frames <= 1 or out.size < 3:
        return out
    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    stops = np.r_[starts[1:], out.size]
    for start, stop in zip(starts, stops):
        if (
            stop - start < min_frames
            and start > 0
            and stop < out.size
            and out[start] != X
            and out[start - 1] == out[stop] != X
        ):
            out[start:stop] = out[start - 1]
    return out


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))


def forward_backward(
    log_emission: np.ndarray, params: ModelParams = DEFAULT_PARAMS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior, per-frame confidence, and entropy for diagnostics.

    These are not used for the submission labels; they are how the report highlights uncertain
    gaps.  The forward-backward recursion is in log space so long missing runs stay stable.
    """
    log_emission = np.asarray(log_emission, dtype=np.float64)
    n = log_emission.shape[0]
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    log_transition = np.log(transition_matrix(params))
    log_start = np.log(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))

    alpha = np.empty((n, 3), dtype=np.float64)
    alpha[0] = log_start + log_emission[0]
    for t in range(1, n):
        alpha[t] = log_emission[t] + _logsumexp(alpha[t - 1][:, None] + log_transition, axis=0)[0]

    beta = np.zeros((n, 3), dtype=np.float64)
    for t in range(n - 2, -1, -1):
        beta[t] = _logsumexp(
            log_transition + log_emission[t + 1][None, :] + beta[t + 1][None, :],
            axis=1,
        )[:, 0]

    log_posterior = alpha + beta
    log_posterior -= _logsumexp(log_posterior, axis=1)
    posterior = np.exp(log_posterior)
    confidence = np.max(posterior, axis=1)
    entropy = -np.sum(np.where(posterior > 0, posterior * np.log(posterior), 0.0), axis=1)
    return posterior, confidence, entropy


def decode(
    features: Any,
    params: ModelParams = DEFAULT_PARAMS,
    *,
    with_confidence: bool = True,
) -> DecodeResult:
    log_emission = emission_log_probs(features, params)
    codes = viterbi(log_emission, params)
    if with_confidence:
        posterior, confidence, entropy = forward_backward(log_emission, params)
    else:
        posterior = np.zeros((len(features), 3), dtype=np.float64)
        confidence = np.zeros(len(features), dtype=np.float64)
        entropy = np.zeros(len(features), dtype=np.float64)
    return DecodeResult(codes=codes, posterior=posterior, confidence=confidence, entropy=entropy)
