"""Tests for the event-gated ball-control transition layer."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.config import FPS, A, D, H, X
from possession_lab.fusion.transitions import (
    TransitionParams,
    apply_transition_layer,
    ball_speed,
    build_pulse,
    control_map,
    decode_with_events,
    detect_kick_events,
    detect_onset_events,
    events_to_rows,
    evidence_probability,
    gate_events,
)

N = 120


def ramp_ball(n: int = N, kick_at: int = 80, kick_speed: float = 12.0) -> np.ndarray:
    """Ball dries slowly in +x, then accelerates to ``kick_speed`` m/s at ``kick_at``."""
    ball = np.zeros((n, 2))
    for t in range(n):
        if t < kick_at:
            ball[t] = [20.0 + 0.04 * t, 20.0]  # 1 m/s drift
        else:
            base = 20.0 + 0.04 * (kick_at - 1)
            ball[t] = [base + kick_speed / FPS * (t - kick_at + 1), 20.0]
    return ball


def player_set(pitch: np.ndarray, teams: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = pitch.shape[0], pitch.shape[1]
    return pitch, np.asarray(teams, dtype=np.int8), np.ones((n, p), dtype=bool)


def standing_players(hold_at: tuple[float, float], chase_at: tuple[float, float], n: int = N):
    """One H player at ``hold_at``, one A player at ``chase_at``, both present everywhere."""
    pitch = np.zeros((n, 2, 2))
    pitch[:, 0] = hold_at
    pitch[:, 1] = chase_at
    return player_set(pitch, [H, A])


class KinematicsTests(unittest.TestCase):
    def test_ball_speed_matches_displacement(self):
        ball = ramp_ball()
        speed, valid = ball_speed(ball)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(np.nanmean(speed[10:40])), 1.0, places=3)
        self.assertAlmostEqual(float(np.nanmean(speed[90:110])), 12.0, places=3)

    def test_control_map_picks_nearest_playing_team(self):
        ball = np.full((20, 2), [30.0, 20.0])
        sets = [standing_players((31.0, 20.0), (27.0, 20.0), n=20)]
        nearest = control_map(ball, sets)
        self.assertTrue((nearest == H).all())

    def test_control_map_ignores_out_of_range_and_referees(self):
        ball = np.full((10, 2), [30.0, 20.0])
        sets = [standing_players((60.0, 20.0), (80.0, 20.0), n=10)]
        self.assertTrue((control_map(ball, sets) == -1).all())
        pitch = np.zeros((10, 1, 2))
        pitch[:, 0] = [30.5, 20.0]
        referee = player_set(pitch, [2])  # referee team is not a control candidate
        self.assertTrue((control_map(ball, [referee]) == -1).all())


class KickDetectionTests(unittest.TestCase):
    def test_detects_kick_to_receiver(self):
        kick_at = 80
        ball = ramp_ball(kick_at=kick_at)
        # H player sits 0.8 m in front of the ball at the kick frame; A trails behind.
        hx = float(ball[kick_at, 0]) + 0.8
        sets = [standing_players((hx, 20.0), (float(ball[kick_at, 0]) - 25.0, 20.0))]
        speed, valid = ball_speed(ball)
        events = detect_kick_events(ball, speed, valid, sets)
        self.assertEqual(len(events), 1)
        frame, gain, team, dist = events[0]
        self.assertIn(frame, {kick_at - 1, kick_at})  # the speed step is caught on the ramp edge
        self.assertEqual(team, H)
        self.assertGreater(gain, 8.0)
        self.assertLessEqual(dist, 2.0)

    def test_rejects_acceleration_without_receiver(self):
        ball = ramp_ball()
        sets = [standing_players((5.0, 20.0), (2.0, 40.0))]  # nobody near the ball
        speed, valid = ball_speed(ball)
        self.assertEqual(detect_kick_events(ball, speed, valid, sets), [])

    def test_suppression_is_team_aware(self):
        same = [(100, 10.0, H, 1.0), (104, 20.0, H, 1.0)]
        duel = [(100, 10.0, H, 1.0), (104, 20.0, A, 1.0)]
        from possession_lab.fusion.transitions import _suppress

        self.assertEqual(len(_suppress(same, 12)), 1)
        self.assertEqual(len(_suppress(duel, 12)), 2)


class OnsetDetectionTests(unittest.TestCase):
    def test_detects_control_onset_after_steal(self):
        n = 120
        ball = np.zeros((n, 2))
        for t in range(n):
            if t < 60:
                ball[t] = [10.0 + 10.0 / FPS * t, 20.0]  # 10 m/s toward the receiver
            else:
                base = 10.0 + 10.0 / FPS * 59
                ball[t] = [base + 1.0 / FPS * (t - 59), 20.0]  # killed to 1 m/s
        receiver = (float(ball[60, 0]) + 0.6, 20.0)
        pitch = np.zeros((n, 2, 2))
        pitch[:, 0] = receiver  # H waits and receives
        pitch[:, 1] = [27.5, 21.5]  # A chases the ball down from nearby behind
        teams = np.asarray([H, A], dtype=np.int8)
        sets = [(pitch, teams, np.ones((n, 2), dtype=bool))]
        speed, valid = ball_speed(ball)
        nearest = control_map(ball, sets)
        events = detect_onset_events(ball, speed, valid, nearest, sets)
        self.assertEqual(len(events), 1)
        frame, _loss, team, _dist = events[0]
        self.assertIn(frame, {59, 60})  # the kill is caught on the deceleration edge
        self.assertEqual(team, H)


class GatingTests(unittest.TestCase):
    def base_arrays(self):
        codes = np.full(N, A, dtype=np.int8)
        prob = np.zeros((N, 3))
        prob[:, A] = 0.9
        prob[:, H] = 0.1
        dead = np.zeros(N, dtype=bool)
        return codes, prob, dead

    def nearest_with_h_control(self, frame: int, span: int = 12):
        nearest = np.full(N, A, dtype=np.int8)
        nearest[frame - span:frame] = H
        return nearest

    def test_keeps_transition_to_new_controller(self):
        codes, prob, dead = self.base_arrays()
        frame = 80
        prob[frame : frame + 50, H] = 0.6
        prob[frame : frame + 50, A] = 0.4
        events, _dropped = gate_events(
            [(frame, 11.0, H, 0.8)], [], codes, prob, dead, self.nearest_with_h_control(frame)
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].team, H)
        self.assertEqual(events[0].kind, "kick")
        self.assertLess(events[0].start, frame)  # control phase extended back
        self.assertGreaterEqual(events[0].support, 0.10)

    def test_drops_same_team_and_weak_support(self):
        codes, prob, dead = self.base_arrays()
        same, _ = gate_events([(80, 11.0, A, 0.8)], [], codes, prob, dead, self.nearest_with_h_control(80))
        self.assertEqual(same, ())
        weak, _ = gate_events([(80, 11.0, H, 0.8)], [], codes, prob, dead, self.nearest_with_h_control(80))
        self.assertEqual(weak, ())  # evidence still strongly favours A

    def test_drops_kick_without_control_phase(self):
        codes, prob, dead = self.base_arrays()
        frame = 80
        prob[frame : frame + 50] = [0.6, 0.4, 0.0]
        nearest = np.full(N, A, dtype=np.int8)  # H never near the ball before the kick
        events, _ = gate_events([(frame, 11.0, H, 0.8)], [], codes, prob, dead, nearest)
        self.assertEqual(events, ())

    def test_drops_event_in_dead_ball_window(self):
        codes, prob, dead = self.base_arrays()
        frame = 80
        prob[frame : frame + 50] = [0.6, 0.4, 0.0]
        dead[frame - 5 : frame + 5] = True
        events, _ = gate_events([(frame, 11.0, H, 0.8)], [], codes, prob, dead, self.nearest_with_h_control(frame))
        self.assertEqual(events, ())


class PulseAndDecodeTests(unittest.TestCase):
    def test_pulse_decays_and_protects(self):
        from possession_lab.fusion.transitions import TransitionEvent

        events = [TransitionEvent(start=50, frame=60, kind="kick", team=H, distance=1.0, support=0.2)]
        pulse, teams, boost, protect = build_pulse(events, 200)
        self.assertEqual(teams[50], H)
        self.assertGreater(pulse[50], pulse[100])
        self.assertTrue(protect[50:80].all())
        self.assertTrue(boost[48:68].all())
        self.assertFalse(boost[100])

    def test_decode_protects_short_event_spell(self):
        log_em = np.full((200, 3), np.log(0.01))
        log_em[:, A] = np.log(0.98)
        log_em[60:70, H] = np.log(0.8)
        log_em[60:70, A] = np.log(0.1)
        log_em[60:70, D] = np.log(0.1)
        boost = np.zeros(200, dtype=bool)
        boost[58:72] = True
        protect = np.zeros(200, dtype=bool)
        dead = np.zeros(200, dtype=bool)
        absorbed = decode_with_events(log_em, boost, protect, dead, halves=((0, 200),))
        self.assertTrue((absorbed[60:70] == A).all())  # < FPS frames, merged away
        self.assertTrue((absorbed[:60] == A).all())
        protected = decode_with_events(log_em, boost, np.ones(200, dtype=bool), dead, halves=((0, 200),))
        self.assertTrue((protected[60:70] == H).all())

    def test_dead_frames_forced(self):
        log_em = np.zeros((100, 3))
        log_em[:, H] = 1.0
        out = decode_with_events(log_em, np.zeros(100, bool), np.zeros(100, bool), np.zeros(100, bool), halves=((0, 100),))
        self.assertTrue((out == H).all())
        dead = np.zeros(100, dtype=bool)
        dead[40:50] = True
        out = decode_with_events(log_em, np.zeros(100, bool), np.zeros(100, bool), dead, halves=((0, 100),))
        self.assertTrue((out[40:50] == D).all())
        self.assertTrue((out[:40] == H).all())


class EndToEndTests(unittest.TestCase):
    def test_layer_recovers_missed_possession(self):
        kick_at = 80
        ball = ramp_ball(n=N, kick_at=kick_at)
        n = N
        pitch = np.zeros((n, 2, 2))
        pitch[:kick_at, 0] = [float(ball[kick_at, 0]) + 0.8, 20.0]  # H stands in front, receives
        pitch[kick_at:, 0] = ball[kick_at:] + np.array([0.8, 0.0])  # H carries the ball on
        pitch[:, 1] = [5.0, 20.0]
        teams = np.asarray([H, A], dtype=np.int8)
        sets = [(pitch, teams, np.ones((n, 2), dtype=bool))]

        codes_pooled = np.full(n, A, dtype=np.int8)
        fused = np.zeros((n, 3))
        fused[:, A] = 0.97
        fused[:, H] = 0.02
        fused[:, D] = 0.01
        fused[kick_at - 12 :, H] = 0.60  # evidence turns to H around the event
        fused[kick_at - 12 :, A] = 0.35
        fused[kick_at - 12 :, D] = 0.05
        evidence = np.zeros((n, 3))
        evidence[:, A] = 0.8
        evidence[:, H] = 0.2
        evidence[:, D] = 0.001
        evidence[kick_at - 12 :, H] = 0.6
        evidence[kick_at - 12 :, A] = 0.35
        evidence[kick_at - 12 :, D] = 0.05

        result = apply_transition_layer(
            codes_pooled=codes_pooled,
            fused_probability=fused,
            event_dead=np.zeros(n, dtype=bool),
            consensus_ball=ball,
            evidence_prob=evidence,
            player_sets=sets,
            halves=((0, n),),
        )
        self.assertGreaterEqual(len(result.events), 1)
        self.assertTrue(any(e.kind == "kick" and e.team == H for e in result.events))
        self.assertTrue((result.codes[kick_at : kick_at + 25] == H).all())
        self.assertTrue((result.codes[:30] == A).all())  # before the control phase begins
        rows = events_to_rows(result.events)
        self.assertEqual(rows.shape[1], 6)

    def test_evidence_probability_matches_linear_pool(self):
        prob = {"x": np.tile([0.8, 0.1, 0.1], (10, 1)), "y": np.tile([0.2, 0.7, 0.1], (10, 1))}
        weight = {"x": np.ones(10), "y": np.zeros(10)}
        out = evidence_probability(prob, weight)
        from possession_lab.fusion.geometry import PRIOR, PRIOR_WEIGHT

        expected = (PRIOR_WEIGHT * PRIOR + np.array([0.8, 0.1, 0.1])) / (PRIOR_WEIGHT + 1.0)
        np.testing.assert_allclose(out, np.tile(expected, (10, 1)), rtol=1e-9)


class ParamsTests(unittest.TestCase):
    def test_defaults_are_physical(self):
        p = TransitionParams()
        self.assertGreater(p.kick_min_gain, 0.0)
        self.assertLess(p.kick_min_away, 1.0)
        self.assertGreater(p.pulse_decay, 0.9)
        self.assertLess(p.pulse_decay, 1.0)
        self.assertLess(X, 4)


if __name__ == "__main__":
    unittest.main()
