"""Tests for keeper-slot derivation and the keeper/referee team rescue."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.config import PITCH
from possession_lab.keepers import (
    derive_keeper_slots,
    half_index,
    keeper_area_team_map,
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


class KeeperAreaTeamMapTest(unittest.TestCase):
    """A keeper slot standing in a penalty area wears the team that defends that goal."""

    def setUp(self):
        players, teams = _roster()
        self.players = np.repeat(players[None], 300, axis=0)
        self.teams = teams
        self.present = np.ones(self.players.shape[:2], dtype=bool)
        self.keepers = derive_keeper_slots(self.players, teams, PITCH, ((0, 300),))

    def test_other_keeper_slot_is_relabelled_by_the_area_it_stands_in(self):
        # measured 19 863: Athletic's keeper slot carried the Barcelona keeper
        # holding the ball in Barcelona's penalty area.
        frames = slice(100, 120)
        self.players[frames, 1] = (11.3, 29.5)  # inside goal 0's penalty area
        out = keeper_area_team_map(
            self.players, self.teams, self.present, self.keepers, ((0, 300),), PITCH
        )
        self.assertTrue((out[frames, 1] == 0).all())  # goal 0 is defended by team 0
        self.assertTrue((out[frames, 0] == 0).all())  # team 0's own keeper stays team 0
        self.assertTrue((out[:100, 1] == 1).all())  # away from the area: roster team stands

    def test_non_keepers_and_players_outside_an_area_are_untouched(self):
        out = keeper_area_team_map(
            self.players, self.teams, self.present, self.keepers, ((0, 300),), PITCH
        )
        self.assertTrue((out[:, 2] == self.teams[2]).all())
        self.assertTrue((out[:, 3] == self.teams[3]).all())
        self.assertTrue((out[:, 0] == self.teams[0]).all())

    def test_per_frame_teams_are_preserved(self):
        teams = np.tile(self.teams[None, :], (300, 1))
        teams[:, 3] = 0  # a roster change the caller already resolved
        out = keeper_area_team_map(
            self.players, teams, self.present, self.keepers, ((0, 300),), PITCH
        )
        self.assertTrue((out[:, 3] == 0).all())
        self.assertEqual(out.shape, (300, 4))

    def test_absent_keeper_slot_is_not_relabelled(self):
        frames = slice(100, 120)
        self.players[frames, 1] = (11.3, 29.5)
        self.present[frames, 1] = False
        out = keeper_area_team_map(
            self.players, self.teams, self.present, self.keepers, ((0, 300),), PITCH
        )
        self.assertTrue((out[frames, 1] == 1).all())


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

    def test_both_keeper_slots_on_one_side_resolves_to_nearest_goal(self):
        # measured 19 863: during the stoppage the Athletic keeper slot froze away
        # from goal 1 while gluing onto the holding Barcelona keeper, so both keeper
        # slots sat on goal 0's side. The box nearest goal 0's goal mouth is the
        # keeper standing there (Barça), whatever slot was projecting onto it.
        atd = np.repeat(self.atd, 1, axis=0)
        atd = self.atd.copy()
        confusion = slice(100, 110)
        atd[confusion, 0] = (21.6, 31.7)  # Barça's slot frozen midfield-ish
        atd[confusion, 1] = (11.3, 29.5)  # Athletic's slot glued to the holding Barça keeper
        keepers = derive_keeper_slots(atd, self.teams, PITCH, ((0, 300),))
        self.assertEqual(keepers.slot[0].tolist(), [0, 1])
        self.assertEqual(keepers.team[0].tolist(), [0, 1])
        n, p = 120, 2
        pixel = np.zeros((n, p, 2), dtype=np.float64)
        pitch_xy = np.zeros((n, p, 2), dtype=np.float64)
        pixel[:, 0] = (22.0 * 10, 31.9 * 10)  # claims the frozen slot 0 projection
        pitch_xy[:, 0] = (22.0, 31.9)
        pixel[:, 1] = (11.3 * 10, 29.5 * 10)  # claims the teleporting slot 1 projection
        pitch_xy[:, 1] = (11.1, 29.1)
        present = np.ones((n, p), dtype=bool)
        teams = np.stack([np.full(n, 0), np.full(n, 1)], axis=1).astype(np.int8)
        fixed, flips = rescue_keeper_teams(
            pixel, pitch_xy, present, teams, atd,
            np.repeat(self.matrix[None], n, axis=0), keepers, ((0, 300),), PITCH,
        )
        # the box standing in goal 0's mouth flips to its defender, team 0
        self.assertTrue((fixed[confusion, 1] == 0).all())
        self.assertTrue((fixed[:100, 1] == 1).all())  # untouched outside the confusion
        self.assertEqual(len(flips), 10)
        self.assertTrue((fixed[:, 0] == 0).all())  # other box keeps its team

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
