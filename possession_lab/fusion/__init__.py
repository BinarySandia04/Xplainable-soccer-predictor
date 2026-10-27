"""Transparent multi-source possession polling.

The validated ATD/HMM is no longer the only opinion.  Each independent source emits a
ballot over ``H`` (Barcelona), ``A`` (Athletic), and ``D`` (dead ball), plus a
confidence/weight.  The ballots are combined with a weighted pool anchored on the
validated HMM, the result is decoded by a second, small HMM with the same transition
prior as the baseline, and confirmed goals/stoppages are hard ``D`` overrides after the
decode.  No public ground truth is read here.

Package map (each concern in exactly one module):

``ballot``
    Data contracts: :class:`SourceVote`, :class:`FusionResult`, :class:`VoterRole`,
    and the viewer-facing rendering (:func:`ballot_reason`, ``STATUS_NAMES``).
``geometry``
    Shared primitives every position-based ballot is built from (``PRIOR``,
    :func:`geometry_ballot`).
``resample``
    Adapts the sampled perception cache to full-length voter inputs.
``voters``
    One named class per evidence source behind the :class:`Voter` contract, plus the
    ``VOTERS`` registry whose order is the pool's float accumulation order.
``pool``
    The tally: log/linear pooling filtered by role, and the consensus-ball blend.
``build``
    :func:`build_fusion` — assemble, poll, pool, decode; no arithmetic of its own.
``cache``
    npz persistence (:func:`save_fusion` / :func:`load_fusion`).

Adding a source means adding one voter class, registering it in ``VOTERS`` (at the
point in the order its accumulation belongs), and giving it a weight in
``DEFAULT_WEIGHTS``; nothing else in the poll changes.
"""

from __future__ import annotations

from .ballot import STATUS_NAMES, FusionResult, SourceVote, VoterRole, ballot_reason
from .build import build_fusion
from .cache import load_fusion, save_fusion
from .geometry import EPS, PRIOR, PRIOR_WEIGHT, geometry_ballot
from .pool import blend_ball_positions, pool_ballots
from .resample import resample_perception
from .voters import (
    DEFAULT_WEIGHTS,
    ROLE_BY_KEY,
    SOURCE_LABELS,
    VOTERS,
    AtdBallVoter,
    AtdInputs,
    AtdShapeVoter,
    AudioVoter,
    EventsVoter,
    HmmVoter,
    TagsVoter,
    VideoBallVoter,
    VideoPlayersVoter,
    VotingContext,
    YoloBallVoter,
    role_for_key,
)

__all__ = [
    "AtdBallVoter",
    "AtdInputs",
    "AtdShapeVoter",
    "AudioVoter",
    "DEFAULT_WEIGHTS",
    "EPS",
    "EventsVoter",
    "FusionResult",
    "HmmVoter",
    "PRIOR",
    "PRIOR_WEIGHT",
    "ROLE_BY_KEY",
    "SOURCE_LABELS",
    "STATUS_NAMES",
    "SourceVote",
    "TagsVoter",
    "VideoBallVoter",
    "VideoPlayersVoter",
    "VoterRole",
    "VOTERS",
    "VotingContext",
    "YoloBallVoter",
    "ballot_reason",
    "blend_ball_positions",
    "build_fusion",
    "geometry_ballot",
    "load_fusion",
    "pool_ballots",
    "resample_perception",
    "role_for_key",
    "save_fusion",
]
