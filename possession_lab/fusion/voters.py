"""The voters: one named unit per evidence source, behind one contract.

Every source is a class with three identity attributes (``key``, ``label``, ``role``)
and a single method ``cast(ctx)`` that reads the shared :class:`VotingContext` and
returns a :class:`~possession_lab.fusion.ballot.SourceVote`.  Nothing votes twice,
nothing knows about the pool, and no voter can reach another voter's inputs.

The sources are (module docstring of the poll, kept here where the registry lives):

``atd_ball``
    The provided ATD ball and the 46 ATD player slots.
``atd_shape``
    ATD team shape around the last known ball during a tracking gap.
``video_players``
    YOLO person detections projected to the pitch, with team labels.
``video_ball``
    The own video ball tracker.  It is a gap-filling source: while ATD is present it is
    anchored to ATD, and while ATD is missing it uses YOLO ball detections plus a
    white-blob fallback.
``yolo_ball``
    YOLO sports-ball detections, associated in time under a physical speed gate.
``hmm_atd``
    The posterior of the existing validated ATD/HMM.  Keeping it as one voter means the
    new evidence has to earn a disagreement instead of replacing a strong baseline by
    fiat.
``tags`` and ``audio``
    Smart tags and the whistle envelope, deliberately weak votes.
``events``
    Confirmed goals and stoppages: a hard ``D``, pooled in no mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import A, D, H, X
from ..model import decode
from .ballot import SourceVote, VoterRole
from .geometry import EPS, PRIOR, geometry_ballot, normalise_rows, sigmoid

SOURCE_LABELS = {
    "atd_ball": "ATD balón",
    "atd_shape": "ATD forma",
    "video_players": "Jugadores YOLO",
    "video_ball": "Balón vídeo",
    "yolo_ball": "YOLO balón",
    "hmm_atd": "HMM ATD",
    "tags": "Smart tags",
    "audio": "Silbato",
    "events": "Eventos",
}

# Evidence strengths for the log-opinion pool.  The validated HMM is the prior, not one more
# equally-weighted ballot; these weights say how much a source may move that prior.  A value of
# 0.10 is a moderate update, 0.03 is a weak hint, and the HMM prior itself is not counted here.
DEFAULT_WEIGHTS = {
    # ATD geometry, tags, and audio are already inside the HMM emission.  They stay visible as
    # ballots but have almost no marginal weight; otherwise the same evidence is counted twice.
    "atd_ball": 0.02,
    "atd_shape": 0.01,
    "video_players": 0.10,
    "video_ball": 0.14,
    "yolo_ball": 0.04,
    "hmm_atd": 1.00,
    "tags": 0.01,
    "audio": 0.02,
    "events": 1.00,
}


@dataclass(frozen=True)
class AtdInputs:
    """The provided tracking feed, normalised once for every voter."""

    ball: np.ndarray  # (N, 2) pitch metres
    present: np.ndarray  # (N,) bool
    players: np.ndarray  # (N, P, 2) pitch metres
    teams: np.ndarray  # (P,) 0 home, 1 away, -1 unknown
    player_present: np.ndarray  # (N, P) bool


@dataclass(frozen=True)
class VotingContext:
    """Everything a voter may read, assembled once by ``build_fusion``.

    A voter must not need anything outside this object; if a new source wants a new
    input, the input is added here so the dependency stays visible in one place.
    """

    n_frames: int
    tracking: Any
    features: Any
    video: dict[str, np.ndarray]  # resample_perception() output, full frame length
    events: Any | None
    params: Any
    weights: Mapping[str, float]  # resolved per-voter base weights
    atd: AtdInputs
    baseline_codes: np.ndarray | None = None


@runtime_checkable
class Voter(Protocol):
    """The one contract every evidence source implements."""

    key: str
    label: str
    role: VoterRole

    def cast(self, ctx: VotingContext) -> SourceVote:
        """Read the context, emit this source's ballot.  Pure: no hidden state."""
        ...


# --------------------------------------------------------------------------- helpers


def _downweight_ballot(vote: SourceVote, scale: np.ndarray, status: np.ndarray) -> SourceVote:
    """Rescale a geometry ballot while a stronger source owns the ball.

    ``video_ball`` and ``yolo_ball`` share this pattern: their weight is scaled down
    where ATD already provides the ball, they drop below the noise floor into
    inactivity, and every inactive frame reverts to prior probability and abstain
    status.
    """
    weight = vote.weight * scale
    active = vote.active & (weight > 1e-5)
    probability = vote.probability.copy()
    probability[~active] = PRIOR
    return SourceVote(
        key=vote.key,
        label=vote.label,
        role=vote.role,
        probability=probability,
        weight=weight,
        active=active,
        status=np.where(active, status, 2).astype(np.int8),
        ball=vote.ball,
        details=vote.details,
    )


def _fit_event_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """Return ``mask`` as a boolean array of exactly ``n`` frames, padding or truncating.

    All three event masks have to be fitted the same way: a shorter ``goal``/``stop`` would make
    ``kind`` shorter than the payload, and the per-frame explanation looks ``kind`` up by frame
    number, so a window that runs past the cached events would raise instead of answering.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size >= n:
        return mask[:n]
    return np.pad(mask, (0, n - mask.size))


def _off_pitch(point: np.ndarray, pitch: tuple[float, float], margin: float) -> bool:
    """True when a pitch point lies clearly beyond a touch/goal line."""
    if not np.isfinite(point).all():
        return False
    length, width = float(pitch[0]), float(pitch[1])
    return bool(
        point[0] < -margin
        or point[0] > length + margin
        or point[1] < -margin
        or point[1] > width + margin
    )


def video_gap_dead(
    atd_present: np.ndarray,
    atd_ball: np.ndarray,
    video_ball: np.ndarray,
    video_confidence: np.ndarray,
    pitch: tuple[float, float],
    *,
    min_gap_frames: int = 60,
    min_confidence: float = 0.02,
    out_margin: float = 0.05,
) -> np.ndarray:
    """Dead-ball gaps proven by the ball being out on *both* sides of a blind window.

    The ATD event rule only fires when the last *ATD* observation was beyond the line, so a
    ball lost while still inside — the camera panned away as it crossed — is never dead.
    The video ball closes that hole: when the most recent observation from either stream is
    beyond the line, every stream then loses the ball for at least ``min_gap_frames``, and
    the first observation afterwards is still beyond the line, the ball was out of play for
    the whole gap.  Requiring the re-detection to be out too keeps a camera cut while the
    ball is live (last out, next in) from becoming a false dead interval.
    """
    n = int(np.asarray(atd_present).shape[0])
    dead = np.zeros(n, dtype=bool)
    if n == 0:
        return dead
    atd = np.asarray(atd_present, dtype=bool)
    atd_xy = np.asarray(atd_ball, dtype=np.float64)
    video_xy = np.asarray(video_ball, dtype=np.float64)
    conf = np.asarray(video_confidence, dtype=np.float64)
    seen_atd = atd & np.isfinite(atd_xy).all(axis=1)
    seen_video = np.isfinite(video_xy).all(axis=1) & (conf > float(min_confidence))
    seen = seen_atd | seen_video
    atd_idx = np.flatnonzero(seen_atd)
    video_idx = np.flatnonzero(seen_video)
    t = 0
    while t < n:
        if seen[t]:
            t += 1
            continue
        start = t
        while t < n and not seen[t]:
            t += 1
        stop = t
        if stop - start < int(min_gap_frames):
            continue
        a_pos = int(np.searchsorted(atd_idx, start))
        v_pos = int(np.searchsorted(video_idx, start))
        last_atd = int(atd_idx[a_pos - 1]) if a_pos > 0 else -1
        last_video = int(video_idx[v_pos - 1]) if v_pos > 0 else -1
        last = max(last_atd, last_video)
        if last < 0:
            continue
        last_point = atd_xy[last] if last == last_atd else video_xy[last]
        a_next = int(atd_idx[a_pos]) if a_pos < atd_idx.size else n
        v_next = int(video_idx[v_pos]) if v_pos < video_idx.size else n
        nxt = min(a_next, v_next)
        if nxt >= n:
            continue
        next_point = atd_xy[nxt] if nxt == a_next else video_xy[nxt]
        if _off_pitch(last_point, pitch, out_margin) and _off_pitch(next_point, pitch, out_margin):
            dead[start:nxt] = True
    return dead


# ------------------------------------------------------------------------ the voters


class AtdBallVoter:
    """Geometry of the provided ATD ball against the 46 ATD player slots."""

    key = "atd_ball"
    label = SOURCE_LABELS[key]
    role = VoterRole.REDUNDANT

    def cast(self, ctx: VotingContext) -> SourceVote:
        atd = ctx.atd
        return geometry_ballot(
            key=self.key,
            label=self.label,
            role=self.role,
            ball=atd.ball,
            ball_confidence=atd.present.astype(np.float64),
            players=atd.players,
            teams=np.tile(atd.teams, (ctx.n_frames, 1)),
            present=atd.player_present,
            base_weight=ctx.weights[self.key],
            ball_status=np.where(atd.present, 0, 2).astype(np.int8),
            sigma=3.8,
            dead_r=7.5,
        )


class AtdShapeVoter:
    """Team shape around the last known ball position while the ball is in a gap."""

    key = "atd_shape"
    label = SOURCE_LABELS[key]
    role = VoterRole.REDUNDANT

    def cast(self, ctx: VotingContext) -> SourceVote:
        n = ctx.n_frames
        features = ctx.features
        present = np.asarray(features.present, dtype=bool)
        gap = np.asarray(features.gap, dtype=np.float64)
        d_home = np.asarray(features.d_home_last, dtype=np.float64)
        d_away = np.asarray(features.d_away_last, dtype=np.float64)
        edge = np.asarray(features.last_edge, dtype=np.float64)
        vote = np.asarray(features.tag_vote, dtype=np.float64)
        probability = np.tile(PRIOR, (n, 1))
        active = (~present) & (gap >= 5.0) & (np.asarray(features.player_count) >= 6)
        if active.any():
            q_home = np.where(np.isfinite(d_home), np.exp(-(d_home**2) / (2.0 * 6.0**2)), 1e-6) + EPS
            q_away = np.where(np.isfinite(d_away), np.exp(-(d_away**2) / (2.0 * 6.0**2)), 1e-6) + EPS
            q_dead = 0.08 + sigmoid((5.0 - np.nan_to_num(edge, nan=20.0)) / 1.5)
            q_home *= 1.0 + 0.2 * np.maximum(vote, 0.0)
            q_away *= 1.0 + 0.2 * np.maximum(-vote, 0.0)
            probability[active] = normalise_rows(np.stack([q_home, q_away, q_dead], axis=1)[active])
        weight = ctx.weights[self.key] * np.clip(gap / 25.0, 0.0, 1.0) * active
        weight[~active] = 0.0
        status = np.where(active, 1, 2).astype(np.int8)
        return SourceVote(
            key=self.key,
            label=self.label,
            role=self.role,
            probability=probability,
            weight=weight,
            active=active,
            status=status,
            ball=np.full((n, 2), np.nan, dtype=np.float64),
            details={
                "gap": gap.astype(np.float32),
                "edge": edge.astype(np.float32),
                "d_home": d_home.astype(np.float32),
                "d_away": d_away.astype(np.float32),
            },
        )


class VideoPlayersVoter:
    """Ballot from YOLO player positions, read against the best available ball.

    Owning the ball selection is deliberate: this source measures *who is around the
    ball*, so it decides for itself which ball to use — the own video measurement when
    confident, otherwise ATD, and nothing when both are absent.
    """

    key = "video_players"
    label = SOURCE_LABELS[key]
    role = VoterRole.EVIDENCE

    def _best_available_ball(self, ctx: VotingContext) -> tuple[np.ndarray, np.ndarray]:
        ball = np.asarray(ctx.video["ball_pitch"], dtype=np.float64).copy()
        confidence = np.asarray(ctx.video["ball_confidence"], dtype=np.float64).copy()
        use_video = (ctx.video["ball_source"] == 0) & (confidence >= 0.22)
        use_atd = ctx.atd.present & ~use_video
        ball[use_atd] = ctx.atd.ball[use_atd]
        confidence[use_atd] = 1.0
        valid = np.isfinite(ball).all(axis=1)
        confidence[~valid] = 0.0
        return ball, confidence

    def cast(self, ctx: VotingContext) -> SourceVote:
        ball, confidence = self._best_available_ball(ctx)
        return geometry_ballot(
            key=self.key,
            label=self.label,
            role=self.role,
            ball=ball,
            ball_confidence=confidence,
            players=ctx.video["player_pitch"],
            teams=ctx.video["player_team"],
            present=ctx.video["player_present"],
            base_weight=ctx.weights[self.key],
            sigma=3.8,
            dead_r=7.5,
            min_players=6,
        )


class VideoBallVoter:
    """The own video ball tracker, anchored to ATD whenever ATD has the ball."""

    key = "video_ball"
    label = SOURCE_LABELS[key]
    role = VoterRole.EVIDENCE

    def cast(self, ctx: VotingContext) -> SourceVote:
        ball = ctx.video["ball_pitch"]
        confidence = ctx.video["ball_confidence"].astype(np.float64)
        source = ctx.video["ball_source"]
        # The source is authoritative only in ATD gaps.  While ATD is present, a video
        # measurement is a useful cross-check but not a second independent ballot.
        weight_scale = np.where(ctx.atd.present, 0.08, 1.0)
        status = np.where(source == 2, 2, np.where(source == 1, 1, 0)).astype(np.int8)
        geometry = geometry_ballot(
            key=self.key,
            label=self.label,
            role=self.role,
            ball=ball,
            ball_confidence=confidence,
            players=ctx.atd.players,
            teams=np.tile(ctx.atd.teams, (ctx.n_frames, 1)),
            present=np.isfinite(ctx.atd.players).all(axis=2),
            base_weight=ctx.weights[self.key],
            ball_status=status,
            sigma=3.8,
            dead_r=7.5,
        )
        vote = _downweight_ballot(geometry, weight_scale, status)
        return SourceVote(
            key=vote.key,
            label=vote.label,
            role=vote.role,
            probability=vote.probability,
            weight=vote.weight,
            active=vote.active,
            status=vote.status,
            ball=vote.ball,
            details={
                **vote.details,
                "source": source.astype(np.int8),
                "age": ctx.video["age"].astype(np.int32),
            },
        )


class YoloBallVoter:
    """Raw YOLO ball candidates, already associated under the physical speed gate."""

    key = "yolo_ball"
    label = SOURCE_LABELS[key]
    role = VoterRole.EVIDENCE

    def _best_candidate(self, ctx: VotingContext) -> tuple[np.ndarray, np.ndarray]:
        n = ctx.n_frames
        present = np.asarray(ctx.video["yolo_ball_present"], dtype=bool)
        confidence = np.asarray(ctx.video["yolo_ball_confidence"], dtype=np.float64)
        points = np.asarray(ctx.video["yolo_ball"], dtype=np.float64)
        # The resampler has already associated candidates under the physical speed gate, so
        # choosing the best remaining candidate cannot teleport to a far false positive.
        safe = np.where(present & np.isfinite(points).all(axis=2), confidence, -np.inf)
        best = np.argmax(safe, axis=1)
        has = np.isfinite(safe[np.arange(n), best]) & (safe[np.arange(n), best] > 0)
        ball = np.full((n, 2), np.nan, dtype=np.float64)
        ball_conf = np.zeros(n, dtype=np.float64)
        ball[has] = points[np.arange(n)[has], best[has]]
        ball_conf[has] = confidence[np.arange(n)[has], best[has]]
        return ball, ball_conf

    def cast(self, ctx: VotingContext) -> SourceVote:
        ball, ball_conf = self._best_candidate(ctx)
        status = np.where(ball_conf >= 0.25, 0, 1).astype(np.int8)
        geometry = geometry_ballot(
            key=self.key,
            label=self.label,
            role=self.role,
            ball=ball,
            ball_confidence=ball_conf,
            players=ctx.atd.players,
            teams=np.tile(ctx.atd.teams, (ctx.n_frames, 1)),
            present=np.isfinite(ctx.atd.players).all(axis=2),
            base_weight=ctx.weights[self.key],
            ball_status=status,
            sigma=3.8,
            dead_r=7.5,
        )
        # The YOLO ballot is more valuable when ATD has lost the ball, but it should not win a
        # fight with ATD while the provided ball is present.
        vote = _downweight_ballot(geometry, np.where(ctx.atd.present, 0.35, 1.0), status)
        return SourceVote(
            key=vote.key,
            label=vote.label,
            role=vote.role,
            probability=vote.probability,
            weight=vote.weight,
            active=vote.active,
            status=vote.status,
            ball=ball,
            details={**vote.details, "age": ctx.video["age"].astype(np.int32)},
        )


class HmmVoter:
    """The validated ATD/HMM result as the poll's soft prior.

    The Viterbi code is the anchor.  Its probability is modulated by the
    forward-backward confidence: a confident baseline is hard to overturn, while an
    uncertain frame leaves room for the independent video ballots.  We do not use the
    HMM posterior argmax as the anchor because the Viterbi code is the validated,
    temporally coherent output.
    """

    key = "hmm_atd"
    label = SOURCE_LABELS[key]
    role = VoterRole.PRIOR

    def cast(self, ctx: VotingContext) -> SourceVote:
        decoded = decode(ctx.features, ctx.params, with_confidence=True)
        n = ctx.n_frames
        codes = np.asarray(decoded.codes if ctx.baseline_codes is None else ctx.baseline_codes, dtype=np.int8)
        confidence = np.asarray(decoded.confidence, dtype=np.float64)
        probability = np.tile(PRIOR, (n, 1))
        graded = (codes != X) & np.isin(codes, (H, A, D))
        if graded.any():
            anchor = np.clip(0.60 + 0.25 * np.nan_to_num(confidence, nan=0.0), 0.55, 0.88)
            other = (1.0 - anchor) / 2.0
            probability[graded, 0] = other[graded]
            probability[graded, 1] = other[graded]
            probability[graded, 2] = other[graded]
            rows = np.flatnonzero(graded)
            probability[rows, codes[rows]] = anchor[rows]
        probability = normalise_rows(probability)
        active = graded
        weight = np.where(active, ctx.weights[self.key], 0.0)
        status = np.where(active, 0, 2).astype(np.int8)
        posterior = np.asarray(decoded.posterior, dtype=np.float64)
        return SourceVote(
            key=self.key,
            label=self.label,
            role=self.role,
            probability=probability,
            weight=weight,
            active=active,
            status=status,
            ball=np.full((n, 2), np.nan, dtype=np.float64),
            details={"code": codes, "confidence": confidence.astype(np.float32)},
            diagnostics={
                "confidence_mean": float(np.mean(confidence)) if confidence.size else 0.0,
                "posterior_mean": float(np.mean(posterior)) if posterior.size else 0.0,
            },
        )


class TagsVoter:
    """Smart-tag team hints and set-piece dead-ball hints, deliberately weak."""

    key = "tags"
    label = SOURCE_LABELS[key]
    role = VoterRole.REDUNDANT

    def cast(self, ctx: VotingContext) -> SourceVote:
        n = ctx.n_frames
        features = ctx.features
        vote = np.asarray(features.tag_vote, dtype=np.float64)
        coverage = np.asarray(features.tag_coverage, dtype=bool)
        setpiece = np.asarray(features.tag_setpiece, dtype=np.float64)
        live = np.asarray(features.tag_live, dtype=bool)
        team = np.tanh(vote / 2.0)
        q_home = 0.50 + 0.32 * team
        q_away = 0.50 - 0.32 * team
        q_dead = 0.03 + 0.32 * setpiece
        q_dead = np.where(live, q_dead * 0.25, q_dead)
        probability = normalise_rows(np.stack([q_home, q_away, q_dead], axis=1))
        active = coverage & np.isfinite(probability).all(axis=1)
        probability[~active] = PRIOR
        weight = np.where(
            active,
            ctx.weights[self.key] * (0.35 + 0.65 * np.minimum(1.0, np.abs(team) + setpiece)),
            0.0,
        )
        status = np.where(active, 0, 2).astype(np.int8)
        return SourceVote(
            key=self.key,
            label=self.label,
            role=self.role,
            probability=probability,
            weight=weight,
            active=active,
            status=status,
            ball=np.full((n, 2), np.nan, dtype=np.float64),
            details={
                "vote": vote.astype(np.int16),
                "setpiece": setpiece.astype(np.int8),
                "coverage": coverage.astype(np.int8),
            },
        )


class AudioVoter:
    """The whistle envelope: a weak dead-ball hint near whistle peaks."""

    key = "audio"
    label = SOURCE_LABELS[key]
    role = VoterRole.REDUNDANT

    def cast(self, ctx: VotingContext) -> SourceVote:
        n = ctx.n_frames
        env = np.asarray(getattr(ctx.features, "whistle_env", np.zeros(n)), dtype=np.float64)
        env = np.clip(np.nan_to_num(env, nan=0.0), 0.0, 1.0)
        active = env > 0.025
        probability = np.tile(PRIOR, (n, 1))
        probability[active] = np.array([0.08, 0.08, 0.84])
        weight = np.where(active, ctx.weights[self.key] * np.clip(env * 2.5, 0.0, 1.0), 0.0)
        status = np.where(active, 0, 2).astype(np.int8)
        return SourceVote(
            key=self.key,
            label=self.label,
            role=self.role,
            probability=probability,
            weight=weight,
            active=active,
            status=status,
            ball=np.full((n, 2), np.nan, dtype=np.float64),
            details={"env": env.astype(np.float32)},
        )


class EventsVoter:
    """Confirmed goals and stoppages: a hard ``D`` override, not a pooled opinion."""

    key = "events"
    label = SOURCE_LABELS[key]
    role = VoterRole.OVERRIDE

    def cast(self, ctx: VotingContext) -> SourceVote:
        n = ctx.n_frames
        features = ctx.features
        events = ctx.events
        goal = np.asarray(getattr(features, "goal_dead", np.zeros(n)), dtype=bool)
        stop = np.asarray(getattr(features, "stop_dead", np.zeros(n)), dtype=bool)
        if events is not None and getattr(events, "dead_mask", None) is not None:
            dead = _fit_event_mask(events.dead_mask, n)
            goal = _fit_event_mask(getattr(events, "goal_dead", goal), n)
            stop = _fit_event_mask(getattr(events, "stop_dead", stop), n)
        else:
            dead = goal | stop
        video_dead = video_gap_dead(
            ctx.atd.present,
            ctx.atd.ball,
            np.asarray(ctx.video.get("ball_pitch", np.full((n, 2), np.nan))),
            np.asarray(ctx.video.get("ball_confidence", np.zeros(n))),
            tuple(ctx.tracking.pitch),
        )
        dead = dead | video_dead
        kind = np.where(goal, 1, np.where(stop | video_dead, 2, 0)).astype(np.int8)
        probability = np.tile(PRIOR, (n, 1))
        probability[dead] = np.array([0.0, 0.0, 1.0])
        weight = np.where(dead, 1.0, 0.0)
        status = np.where(dead, 3, 2).astype(np.int8)
        return SourceVote(
            key=self.key,
            label=self.label,
            role=self.role,
            probability=probability,
            weight=weight,
            active=dead,
            status=status,
            ball=np.full((n, 2), np.nan, dtype=np.float64),
            details={"kind": kind},
        )


# The poll order is load-bearing, not cosmetic: the pool sums ballots in float arithmetic,
# and float addition is order-dependent.  ``VOTERS`` order equals the validated run's
# accumulation order, the ``sources`` dict order, the saved-npz metadata order and the
# viewer ballot order.  Reorder only together with a revalidated output hash.
VOTERS: tuple[Voter, ...] = (
    AtdBallVoter(),
    AtdShapeVoter(),
    VideoPlayersVoter(),
    VideoBallVoter(),
    YoloBallVoter(),
    HmmVoter(),
    TagsVoter(),
    AudioVoter(),
    EventsVoter(),
)

ROLE_BY_KEY: dict[str, VoterRole] = {voter.key: voter.role for voter in VOTERS}


def role_for_key(key: str) -> VoterRole:
    """The registered role of ``key``; unknown sources are treated as plain evidence."""
    return ROLE_BY_KEY.get(str(key), VoterRole.EVIDENCE)
