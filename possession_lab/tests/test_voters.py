"""Contract tests for the voter architecture.

These lock the invariants the poll relies on: the registry is the single source of
truth for keys/labels/roles, its order is the pool's accumulation order, every voter
obeys the ``Voter`` protocol and reads only the ``VotingContext``, and diagnostics
survive the cache round trip.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from possession_lab.config import A, H
from possession_lab.data import TagTable
from possession_lab.features import extract_features
from possession_lab.fusion import (
    DEFAULT_WEIGHTS,
    SOURCE_LABELS,
    VOTERS,
    VoterRole,
    build_fusion,
    load_fusion,
    save_fusion,
)
from possession_lab.fusion.voters import Voter, role_for_key

from .test_fusion import graded, make_tracking


class VoterRegistryTest(unittest.TestCase):
    def test_every_voter_satisfies_the_protocol_and_declares_its_identity(self):
        for voter in VOTERS:
            self.assertIsInstance(voter, Voter)
            self.assertEqual(voter.label, SOURCE_LABELS[voter.key])
            self.assertIsInstance(voter.role, VoterRole)
            self.assertIn(voter.key, DEFAULT_WEIGHTS)

    def test_registry_keys_and_weights_cover_exactly_the_same_sources(self):
        self.assertEqual(len({voter.key for voter in VOTERS}), len(VOTERS))
        self.assertEqual({voter.key for voter in VOTERS}, set(DEFAULT_WEIGHTS))

    def test_registry_order_is_the_documented_accumulation_order(self):
        # The pool sums ballots in float arithmetic; this exact sequence is pinned to the
        # validated fusion output hash.  A reorder must be a deliberate, revalidated act.
        self.assertEqual(
            [voter.key for voter in VOTERS],
            [
                "atd_ball",
                "atd_shape",
                "video_players",
                "yolo_ball",
                "hmm_atd",
                "tags",
                "audio",
                "events",
            ],
        )

    def test_roles_are_the_registry_default_for_every_key(self):
        self.assertEqual(role_for_key("hmm_atd"), VoterRole.PRIOR)
        self.assertEqual(role_for_key("events"), VoterRole.OVERRIDE)
        self.assertEqual(role_for_key("atd_ball"), VoterRole.REDUNDANT)
        self.assertEqual(role_for_key("yolo_ball"), VoterRole.EVIDENCE)
        self.assertEqual(role_for_key("who-dis-this"), VoterRole.EVIDENCE)


class BallotContractTest(unittest.TestCase):
    def _sources(self):
        tracking = make_tracking(40)
        tags = TagTable.empty(40)
        features = extract_features(tracking, tags)
        baseline = np.full(40, H, dtype=np.int8)
        baseline[20:] = A
        result = build_fusion(
            tracking,
            tags,
            features=features,
            baseline_codes=baseline,
            halves=graded(40),
        )
        return result, tracking

    def test_every_ballot_matches_its_voter_identity_and_frame_length(self):
        result, tracking = self._sources()
        for voter in VOTERS:
            vote = result.sources[voter.key]
            self.assertEqual(vote.key, voter.key)
            self.assertEqual(vote.label, voter.label)
            self.assertEqual(vote.role, voter.role)
            self.assertEqual(vote.probability.shape, (tracking.n_frames, 3))
            self.assertEqual(vote.active.shape, (tracking.n_frames,))
            self.assertEqual(vote.status.shape, (tracking.n_frames,))
            np.testing.assert_allclose(vote.probability.sum(axis=1), 1.0, rtol=0, atol=1e-8)

    def test_prior_and_override_weights_follow_the_role_contract(self):
        result, _ = self._sources()
        hmm = result.sources["hmm_atd"]
        events = result.sources["events"]
        # The prior anchors the pool wherever the baseline graded a state.
        self.assertTrue(hmm.active.any())
        self.assertTrue(np.all(hmm.weight[hmm.active] == DEFAULT_WEIGHTS["hmm_atd"]))
        # The override carries all-or-nothing weight and never an opinion elsewhere.
        self.assertTrue(np.all(events.weight[~events.active] == 0.0))
        self.assertTrue(np.all(events.weight[events.active] == 1.0))

    def test_diagnostics_round_trip_through_the_cache(self):
        result, _ = self._sources()
        self.assertIn("confidence_mean", result.sources["hmm_atd"].diagnostics)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "fusion.npz"
            save_fusion(path, result)
            loaded = load_fusion(path)
            assert loaded is not None
            self.assertEqual(
                loaded.sources["hmm_atd"].diagnostics["confidence_mean"],
                result.sources["hmm_atd"].diagnostics["confidence_mean"],
            )
            self.assertEqual(
                loaded.sources["hmm_atd"].diagnostics["posterior_mean"],
                result.sources["hmm_atd"].diagnostics["posterior_mean"],
            )
            self.assertEqual(
                {source["role"] for source in loaded.sample(0)["sources"]},
                {source["role"] for source in result.sample(0)["sources"]},
            )


if __name__ == "__main__":
    unittest.main()
