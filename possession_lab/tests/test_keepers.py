"""Tests for keeper-slot derivation and the keeper/referee team rescue."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.config import PITCH
from possession_lab.keepers import (
    derive_keeper_slots,
    half_index,
    rescue_keeper_teams,
    rescue_officials,
)

LENGTH, WIDTH = PITCH
HALVES = ((0, 1000), (1000, 2000))


def scaling_matrix(scale: float = 10.0) -> np.ndarray:
    """Homography with ``pixel = scale * metres`` so distances stay comparable.

    ``pitch_to_pixel`` normalises pitch coordinates by the pitch size before applying the
    matrix, so the matrix itself must carry ``scale * pitch`` on its diagonal.
    """
    return np.diag([scale * LENGTH, scale * WIDTH, 1.0])


def _roster() -> tuple[np.ndarray, np.ndarray]:
    """Four-slot roster: slot 0 team-keeper at goal 0, slot 1 team-keeper at goal 1,
    two outfielders near midfield."""
    players = np.array(
        [
            [1.5, WIDTH / 2],  # keeper of the goal-0 side
            [LENGTH - 1.5, WIDTH / 2],  # keeper of the goal-1 side
            [LENGTH / 2, 20.0],
            [LENGTH / 2, 48.0],
        ],
        dtype=np.float64,
    )
    teams = np.array([0, 1, 0, 1], dtype=np.int8)
    return players, teams


class DeriveKeeperSlotsTest(unittest.TestCase):
    def setUp(self):
        self.players, self.teams = _roster()

    def test_identifies_keepers_and_defended_goals(self):
        atd = np.repeat(self.players[None], 2000, axis=0)
        keepers = derive_keeper_slots(atd, self.teams, PITCH, HALVES)
        self.assertEqual(keepers.slot.tolist(), [[0, 1], [0, 1]])
        self.assertEqual(keepers.team.tolist(), [[0, 1], [0, 1]])

    def test_side_swaps_with_second_half(self):
        atd = np.repeat(self.players[None], 2000, axis=0)
        atd[1000:, 0] = (LENGTH - 1.5, WIDTH / 2)  # team-0 keeper moved to goal 1
        atd[1000:, 1] = (1.5, WIDTH / 2)  # team-1 keeper moved to goal 0
        keepers = derive_keeper_slots(atd, self.teams, PITCH, HALVES)
        self.assertEqual(keepers.slot[1].tolist(), [1, 0])
        self.assertEqual(keepers.team[1].tolist(), [1, 0])

    def test_roaming_slot_never_wins(self):
        atd = np.repeat(self.players[None], 2000, axis=0)
        rows = np.linspace(5.0, LENGTH - 5.0, 2000)
        atd[:, 2, 0] = rows  # outfielder roams the whole length
        atd[:, 2, 1] = WIDTH / 2
        keepers = derive_keeper_slots(atd, self.teams, PITCH, HALVES)
        self.assertNotIn(2, keepers.slot[0].tolist())

    def test_short_half_is_skipped(self):
        atd = np.repeat(self.players[None], 2000, axis=0)
        keepers = derive_keeper_slots(atd, self.teams, PITCH, ((0, 1000), (1000, 1100)))
        self.assertEqual(keepers.slot[1].tolist(), [-1, -1])


class RescueKeeperTeamsTest(unittest.TestCase):
    def setUp(self):
        self.matrix = scaling_matrix()
        self.atd, self.teams = _roster()
        self.atd = np.repeat(self.atd[None], 300, axis=0)  # static roster for 300 frames
        self.keepers = derive_keeper_slots(self.atd, self.teams, PITCH, ((0, 1000),))

    def _store(self, boxes_team):
        n, p = 10, len(boxes_team)
        pixel = np.zeros((n, p, 2), dtype=np.float64)
        pitch_xy = np.zeros((n, p, 2), dtype=np.float64)
        present = np.ones((n, p), dtype=bool)
        teams = np.full((n, p), -1, dtype=np.int8)
        for j, team in enumerate(boxes_team):
            teams[:, j] = team
        return pixel, pitch_xy, present, teams

    def test_keeper_slot_claimer_gets_goal_defenders_team(self):
        # A box sits on the goal-0 keeper projection (15px, 340px) but is mislabelled
        # as the attacking team; the rescue hands it to the goal-0 defender (team 0).
        pixel, pitch_xy, present, teams = self._store([1])
        pixel[:, 0] = (1.5 * 10, WIDTH / 2 * 10)
        pitch_xy[:, 0] = (1.5, WIDTH / 2)
        fixed, flips = rescue_keeper_teams(
            pixel, pitch_xy, present, teams, self.atd,
            np.repeat(self.matrix[None], 10, axis=0), self.keepers, ((0, 1000),), PITCH,
        )
        self.assertTrue((fixed[:, 0] == 0).all())
        self.assertEqual(len(flips), 10)

    def test_box_far_from_slots_is_untouched(self):
        pixel, pitch_xy, present, teams = self._store([-1])
        pixel[:, 0] = (LENGTH / 2 * 10, 20.0 * 10)
        pitch_xy[:, 0] = (LENGTH / 2, 20.0)
        fixed, flips = rescue_keeper_teams(
            pixel, pitch_xy, present, teams, self.atd,
            np.repeat(self.matrix[None], 10, axis=0), self.keepers, ((0, 1000),), PITCH,
        )
        self.assertEqual(flips, [])
        self.assertTrue((fixed == -1).all())

    def test_frames_outside_halves_are_skipped(self):
        pixel, pitch_xy, present, teams = self._store([1])
        pixel[:, 0] = (1.5 * 10, WIDTH / 2 * 10)
        pitch_xy[:, 0] = (1.5, WIDTH / 2)
        fixed, flips = rescue_keeper_teams(
            pixel, pitch_xy, present, teams, self.atd,
            np.repeat(self.matrix[None], 10, axis=0), self.keepers, ((100, 200),), PITCH,
        )
        self.assertEqual(flips, [])
        self.assertTrue((fixed[:, 0] == 1).all())

    def test_half_index(self):
        self.assertEqual(half_index(5, HALVES), 0)
        self.assertEqual(half_index(1500, HALVES), 1)
        self.assertEqual(half_index(2500, HALVES), -1)


class RescueOfficialsTest(unittest.TestCase):
    def test_slot_free_track_is_flagged(self):
        matrix = scaling_matrix()
        atd, _teams = _roster()
        atd = np.repeat(atd[None], 40, axis=0)
        n, p = 40, 2
        pixel = np.zeros((n, p, 2), dtype=np.float64)
        present = np.ones((n, p), dtype=bool)
        tracks = np.tile(np.array([7, 8], dtype=np.int64), (n, 1))
        # track 7 sits on a keeper projection (player); track 8 always roams far away
        pixel[:, 0] = (1.5 * 10, WIDTH / 2 * 10)
        pixel[:, 1] = (30.0 * 10, 60.0 * 10)
        official = rescue_officials(
            pixel, present, tracks, atd, np.repeat(matrix[None], n, axis=0), PITCH,
            min_appearances=25,
        )
        self.assertFalse(official[:, 0].any())
        self.assertTrue(official[:, 1].all())

    def test_short_tracks_are_ignored(self):
        matrix = scaling_matrix()
        atd, _teams = _roster()
        atd = np.repeat(atd[None], 40, axis=0)
        n, p = 40, 1
        pixel = np.full((n, p, 2), (LENGTH / 2 * 10, 20.0 * 10), dtype=np.float64)
        present = np.zeros((n, p), dtype=bool)
        present[:10, 0] = True  # only ten appearances, below min_appearances
        tracks = np.tile(np.array([9], dtype=np.int64), (n, 1))
        official = rescue_officials(
            pixel, present, tracks, atd, np.repeat(matrix[None], n, axis=0), PITCH,
            min_appearances=25,
        )
        self.assertFalse(official.any())


if __name__ == "__main__":
    unittest.main()
