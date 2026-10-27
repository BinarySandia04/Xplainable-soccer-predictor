"""Tests for the per-switch "why" annotations."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from possession_lab.config import A, D, H, X
from possession_lab.explain import (
    build_explanations,
    build_switch_records,
    find_switches,
)
from possession_lab.model import ModelParams, emission_log_probs


def make_features(n: int = 200) -> SimpleNamespace:
    """Synthetic features with one stoppage: ball parked at the line (95-145), lost by the
    tracker (120-130), out of play (108-111), teams walking (95-135), restart at 145."""
    present = np.ones(n, dtype=bool)
    present[120:130] = False
    edge = np.full(n, 20.0)
    edge[95:145] = 1.2
    edge[120:130] = np.nan
    ball_speed = np.full(n, 3.0)
    ball_speed[95:145] = 0.4
    ball_speed[145:] = 6.5
    ball_speed[120:130] = np.nan
    return SimpleNamespace(
        present=present,
        d_home=np.where(present, 5.0, np.nan),
        d_away=np.where(present, 9.0, np.nan),
        near_distance=np.where(present, 5.0, np.nan),
        near_team=np.where(present, 0, -1).astype(np.int8),
        near_player=np.full(n, -1, dtype=np.int16),
        second_distance=np.full(n, 12.0),
        margin=np.full(n, 4.0),
        near_speed=np.ones(n),
        team_speed=np.full(n, 2.0),
        ball_speed=ball_speed,
        ball_acceleration=np.zeros(n),
        gap=np.zeros(n, dtype=np.int32),
        edge=edge,
        last_edge=np.full(n, 1.2),
        out_of_play=np.zeros(n, dtype=bool),
        last_touch=np.zeros(n, dtype=np.int8),
        last_touch_age=np.zeros(n, dtype=np.int32),
        d_home_last=np.full(n, 5.0),
        d_away_last=np.full(n, 9.0),
        tag_vote=np.zeros(n, dtype=np.int16),
        tag_setpiece=np.zeros(n, dtype=np.int8),
        tag_coverage=np.zeros(n, dtype=bool),
        player_count=np.full(n, 20, dtype=np.int16),
        parked_edge=np.zeros(n, dtype=bool),
    )


def make_prediction() -> np.ndarray:
    pred = np.array([H] * 110 + [D] * 30 + [A] * 40 + [X] * 20, dtype=np.int8)
    return pred


class FindSwitchesTest(unittest.TestCase):
    def test_switches_inside_halves_excluding_x(self):
        switches = find_switches(make_prediction(), halves=((0, 180),))
        self.assertEqual(switches, [(110, H, D), (140, D, A)])

    def test_flickers_collapse_into_the_first_switch(self):
        pred = np.array([H] * 100 + [D] * 5 + [H] * 5 + [A] * 100, dtype=np.int8)
        switches = find_switches(pred, halves=((0, 210),))
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0], (100, H, D))


class SwitchReasonsTest(unittest.TestCase):
    def records(self, features, prediction=None):
        prediction = prediction if prediction is not None else make_prediction()
        log_emission = emission_log_probs(features, ModelParams())
        return build_switch_records(
            prediction, features, 0, None, log_emission, halves=((0, 180),)
        )

    def test_stoppage_switch_collects_every_signal(self):
        features = make_features()
        features.out_of_play[108:111] = True
        features.parked_edge[100:120] = True
        features.team_speed[80:135] = 0.5
        records = self.records(features)
        kinds = {r["kind"] for r in records[0]["reasons"]}
        self.assertTrue({"out_of_bounds", "parked", "teams_slow"} <= kinds)
        self.assertEqual(records[0]["from"], "H")
        self.assertEqual(records[0]["to"], "D")

    def test_restart_and_geometry_explain_the_exit(self):
        features = make_features()
        records = self.records(features)
        exit_record = records[1]
        kinds = {r["kind"] for r in exit_record["reasons"]}
        self.assertTrue({"restart", "geometry"} <= kinds)
        restart = next(r for r in exit_record["reasons"] if r["kind"] == "restart")
        self.assertEqual(restart["evidence"]["frame"], 145)
        self.assertLess(restart["evidence"]["speed_from"], 1.5)

    def test_silent_switch_falls_back_to_the_prior(self):
        features = make_features()
        features.team_speed[:] = 2.0
        features.ball_speed[:] = 3.0
        features.parked_edge[:] = False
        features.out_of_play[:] = False
        features.present[:] = True
        records = self.records(features)
        self.assertEqual(records[0]["reasons"][0]["kind"], "prior")

    def test_confidence_is_a_probability(self):
        features = make_features()
        records = self.records(features)
        for record in records:
            self.assertGreaterEqual(record["confidence"], 0.0)
            self.assertLessEqual(record["confidence"], 1.0)

    def test_evidence_is_finite_json(self):
        features = make_features()
        features.parked_edge[100:120] = True
        features.ball_speed[100:120] = np.nan  # ATD loses the ball during the parked run
        records = self.records(features)
        parked = next(r for r in records[0]["reasons"] if r["kind"] == "parked")
        for value in parked["evidence"].values():
            self.assertTrue(
                value is None or not (isinstance(value, (float, np.floating)) and np.isnan(value)),
                f"NaN in {parked['evidence']}",
            )
        json.loads(json.dumps(build_explanations(  # strict round-trip: no NaN/Infinity tokens
            np.array([H] * 110 + [D] * 70, dtype=np.int8), features, 0, None, halves=((0, 180),)
        )))


if __name__ == "__main__":
    unittest.main()
