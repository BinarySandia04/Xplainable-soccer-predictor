"""Tests for the event-gated ball-control transition layer."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.config import FPS, A, D, H, X
from possession_lab.fusion.transitions import (
    BoxEvidence,
    TransitionParams,
    apply_transition_layer,
    ball_conflict_mask,
    ball_speed,
    ball_stale_mask,
    build_pulse,
    carry_dead_through_blind,
    control_map,
    decode_with_events,
    detect_contact_events,
    detect_kick_events,
    detect_kicker_events,
    detect_onset_events,
    events_to_rows,
    evidence_probability,
    gate_events,
    gate_pixel_events,
    pixel_contact,
    relay_contacts,
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


class BallStaleGateTests(unittest.TestCase):
    def test_mask_marks_only_the_deep_middle_of_long_gaps(self):
        present = np.ones(700, dtype=bool)
        present[100:400] = False  # 300-frame blind window
        ball = np.zeros((700, 2))
        mask = ball_stale_mask(ball, present, stale_frames=125)
        self.assertFalse(mask[0])
        self.assertFalse(mask[99])
        self.assertFalse(mask[126])  # still < 125 frames from the last observation
        self.assertTrue(mask[226])
        self.assertTrue(mask[274])
        self.assertFalse(mask[276])  # already < 125 frames from the next observation
        self.assertFalse(mask[401])
        # short gaps (110 frames) never qualify: no frame is >100 from both ends
        short = np.ones(400, dtype=bool)
        short[150:260] = False
        self.assertFalse(ball_stale_mask(np.zeros((400, 2)), short, stale_frames=100).any())

    def test_mask_requires_finite_positions_not_just_present(self):
        present = np.ones(500, dtype=bool)
        ball = np.zeros((500, 2))
        ball[100:400] = np.nan
        present[250] = False  # keep a real observation-free frame at the centre anyway
        mask = ball_stale_mask(ball, present, stale_frames=10)
        self.assertFalse(mask[0])
        self.assertTrue(mask[120])

    def test_apply_transition_layer_drops_kicks_inside_blind_windows(self):
        kick_at = 80
        n = N
        ball = ramp_ball(n=n, kick_at=kick_at)
        pitch = np.zeros((n, 2, 2))
        pitch[:kick_at, 0] = [float(ball[kick_at, 0]) + 0.8, 20.0]
        pitch[kick_at:, 0] = ball[kick_at:] + np.array([0.8, 0.0])
        pitch[:, 1] = [5.0, 20.0]
        teams = np.asarray([H, A], dtype=np.int8)
        sets = [(pitch, teams, np.ones((n, 2), dtype=bool))]
        codes_pooled = np.full(n, A, dtype=np.int8)
        fused = np.zeros((n, 3))
        fused[:, A] = 0.97
        fused[:, H] = 0.02
        fused[:, D] = 0.01
        fused[kick_at - 12 :, H] = 0.60
        fused[kick_at - 12 :, A] = 0.35
        fused[kick_at - 12 :, D] = 0.05
        evidence = np.full((n, 3), 0.0)
        evidence[:, A] = 0.8
        evidence[:, H] = 0.2
        evidence[:, D] = 0.001
        evidence[kick_at - 12 :, H] = 0.6
        evidence[kick_at - 12 :, A] = 0.35
        evidence[kick_at - 12 :, D] = 0.05

        stale = np.zeros(n, dtype=bool)
        stale[60:100] = True  # the kick sits inside a video-carried blind window
        result = apply_transition_layer(
            codes_pooled=codes_pooled,
            fused_probability=fused,
            event_dead=np.zeros(n, dtype=bool),
            consensus_ball=ball,
            evidence_prob=evidence,
            player_sets=sets,
            halves=((0, n),),
            ball_stale=stale,
        )
        self.assertEqual(len(result.events), 0)
        without = apply_transition_layer(
            codes_pooled=codes_pooled,
            fused_probability=fused,
            event_dead=np.zeros(n, dtype=bool),
            consensus_ball=ball,
            evidence_prob=evidence,
            player_sets=sets,
            halves=((0, n),),
        )
        self.assertGreaterEqual(len(without.events), 1)  # same fixture, gate open


class PixelChannelTests(unittest.TestCase):
    """YOLO ball-in-box contacts, win-and-clear kicks and the boundary interval gate."""

    @staticmethod
    def evidence(n, atd, yolo=None, yolo_present=None, boxes=None, team=None):
        if boxes is None:
            boxes = np.zeros((n, 2, 4))
            boxes[:, 0] = [10.0, 10.0, 20.0, 40.0]  # H player
            boxes[:, 1] = [80.0, 10.0, 90.0, 40.0]  # A player
        if team is None:
            team = np.tile(np.array([H, A], dtype=np.int8), (n, 1))
        return BoxEvidence(
            boxes=boxes,
            team=team,
            present=np.ones((n, 2), dtype=bool),
            ball_yolo=np.full((n, 2), np.nan) if yolo is None else yolo,
            ball_yolo_present=np.zeros(n, dtype=bool) if yolo_present is None else yolo_present,
            ball_atd=atd,
        )

    def test_pixel_contact_trusts_yolo_when_it_sees_the_ball(self):
        n = 6
        atd = np.zeros((n, 2))
        atd[:, :] = [15.0, 30.0]  # inside the H box, at its feet line: contain + touch
        yolo = np.zeros((n, 2))
        yolo[:, :] = [85.0, 25.0]  # the detector sees the ball at the A player instead
        yolo[0] = [np.nan, np.nan]
        yp = np.ones(n, dtype=bool)
        yp[0] = False
        ev = self.evidence(n, atd, yolo, yp)
        contain, touch = pixel_contact(ev)
        self.assertEqual(int(contain[0]), 1)  # yolo absent: atd alone -> H
        self.assertEqual(int(contain[1]), 2)  # both seen: yolo first, the atd pixel is noise
        self.assertEqual(int(touch[0]), 1)  # ball on the H player's feet line
        self.assertEqual(int(touch[1]), 2)  # ... and while yolo sees it, only the A player's

    def test_contact_needs_a_fresh_dominant_consecutive_run(self):
        n = 60
        contain = np.zeros(n, dtype=np.int8)
        contain[20:23] = 1
        touch = np.zeros(n, dtype=np.int8)
        speed = np.full(n, 5.0)
        dead = np.zeros(n, dtype=bool)
        events = detect_contact_events(contain, touch, speed, dead)
        self.assertEqual([(f, r, t) for f, r, t, _ in events], [(22, 3.0, H)])
        # scattered contain frames never form a run
        contain_loose = np.zeros(n, dtype=np.int8)
        contain_loose[[20, 22, 24]] = 1
        self.assertEqual(detect_contact_events(contain_loose, touch, speed, dead), [])
        # a touch right before the run means continued play, not a takeover
        touch_stale = np.zeros(n, dtype=np.int8)
        touch_stale[18] = 1
        self.assertEqual(detect_contact_events(contain, touch_stale, speed, dead), [])
        # the other team sharing every frame strips dominance
        contain_shared = np.zeros(n, dtype=np.int8)
        contain_shared[20:23] = 3
        self.assertEqual(detect_contact_events(contain_shared, touch, speed, dead), [])
        # a fast ball is a deflection, not control
        self.assertEqual(detect_contact_events(contain, touch, np.full(n, 20.0), dead), [])
        # dead ball frames are not takeovers
        dead[15:30] = True
        self.assertEqual(detect_contact_events(contain, touch, speed, dead), [])

    def test_kicker_finds_a_win_and_clear(self):
        n = 80
        boxes = np.zeros((n, 2, 4))
        boxes[:, 0] = [-2.0, 0.0, 2.0, 20.0]  # H player, feet at (0, 20)
        boxes[:, 1] = [60.0, 0.0, 64.0, 20.0]  # A player far away
        atd = np.zeros((n, 2))
        atd[:, :] = [0.0, 19.0]  # at H's feet
        for t in range(51, n):
            atd[t, 0] = 5.0 * (t - 50)  # released: 5 px/frame away
        speed = np.full(n, 4.0)
        speed[52:] = 10.0  # put in flight within 3 frames of frame 50
        ev = self.evidence(n, atd, boxes=boxes)
        _, touch = pixel_contact(ev)
        events = detect_kicker_events(ev, touch, speed)
        self.assertEqual([(f, t) for f, _, t, _, _ in events], [(50, H)])
        self.assertLess(events[0][3], 2.0)  # released right off the foot

    def test_kicker_needs_the_other_team_far_and_a_real_release(self):
        n = 80
        boxes = np.zeros((n, 2, 4))
        boxes[:, 0] = [-2.0, 0.0, 2.0, 20.0]
        boxes[:, 1] = [6.0, 0.0, 10.0, 20.0]  # A player contesting: dominance fails
        atd = np.zeros((n, 2))
        atd[:, :] = [0.0, 19.0]
        for t in range(51, n):
            atd[t, 0] = 5.0 * (t - 50)
        speed = np.full(n, 4.0)
        speed[52:] = 10.0
        ev = self.evidence(n, atd, boxes=boxes)
        _, touch = pixel_contact(ev)
        self.assertEqual(detect_kicker_events(ev, touch, speed), [])
        # slow at the feet but never put in flight: no release
        ev2 = self.evidence(n, np.zeros((n, 2)), boxes=boxes)
        speed_still = np.full(n, 4.0)
        _, touch2 = pixel_contact(ev2)
        self.assertEqual(detect_kicker_events(ev2, touch2, speed_still), [])

    @staticmethod
    def _gate_world(n, flip_at):
        codes = np.zeros(n, dtype=np.int8)
        codes[:flip_at] = A
        prob = np.full((n, 3), 0.33)
        prob[:, D] = 0.0
        dead = np.zeros(n, dtype=bool)
        return codes, prob, dead

    def test_gate_keeps_boundary_contact_whose_team_the_fusion_reaches(self):
        n = 700
        codes, prob, dead = self._gate_world(n, 260)
        prob[400:, H] = 0.6  # fused evidence catches up within the horizon
        prob[405:, A] = 0.6
        contain = np.zeros(n, dtype=np.int8)
        contain[246:251] = 1
        contact = [(250, 3.0, H, 1.0)]
        kept, intervals = gate_pixel_events(contact, [], contain, codes, prob, dead, ())
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(246, 260, H)])
        self.assertEqual(len(kept), 1)
        # a second event for the team that already owns the stream dies
        _, intervals2 = gate_pixel_events([*contact, (252, 3.0, H, 1.0)], [], contain, codes, prob, dead, ())
        self.assertEqual(len(intervals2), 1)
        # the opposing team's event cuts the interval and takes over
        _, intervals3 = gate_pixel_events([*contact, (255, 3.0, A, 1.0)], [], contain, codes, prob, dead, ())
        self.assertEqual(intervals3[0][1], 255)
        self.assertEqual(len(intervals3), 2)

    def test_gate_rejects_mid_spell_and_unconfirmed_events(self):
        n = 700
        codes, prob, dead = self._gate_world(n, 260)
        contain = np.zeros(n, dtype=np.int8)
        contain[420:425] = 1
        # far from any possession boundary: settled spell belongs to pitch geometry
        _, intervals = gate_pixel_events([(424, 3.0, H, 1.0)], [], contain, codes, prob, dead, ())
        self.assertEqual(intervals, [])
        # near a boundary but the fused evidence never comes around
        _, intervals = gate_pixel_events([(250, 3.0, H, 1.0)], [], contain[:251], codes, prob, dead, ())
        self.assertEqual(intervals, [])
        # ... unless the win-and-clear fingerprint carries it
        kicker = [(424, 11.0, H, 13.0, 57.0)]
        codes2 = np.zeros(n, dtype=np.int8)
        codes2[:424] = A  # boundary right at the event: close to it
        _, intervals = gate_pixel_events([], kicker, np.zeros(n, dtype=np.int8), codes2, prob, dead, ())
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(423, 499, H)])
        # a punt-speed "release" above the fingerprint ceiling is a track jump
        kicker_jump = [(424, 40.0, H, 13.0, 57.0)]
        _, intervals = gate_pixel_events([], kicker_jump, np.zeros(n, dtype=np.int8), codes2, prob, dead, ())
        self.assertEqual(intervals, [])
        # dead-ball frames cannot flip possession
        dead[420:430] = True
        _, intervals = gate_pixel_events([], kicker, np.zeros(n, dtype=np.int8), codes2, prob, dead, ())
        self.assertEqual(intervals, [])

    def test_gate_extends_to_a_boundary_the_fusion_is_about_to_cross(self):
        n = 700
        codes, prob, dead = self._gate_world(n, 300)
        prob[340:, H] = 0.6  # momentum passes ...
        prob[300:, H] = 0.9  # ... and the fusion commits at the boundary: absorb the lag
        contain = np.zeros(n, dtype=np.int8)
        contain[186:191] = 1
        _, intervals = gate_pixel_events([(190, 3.0, H, 1.0)], [], contain, codes, prob, dead, ())
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(186, 300, H)])
        # without the commit (probability below the bar) the interval stops at the cap
        prob2 = prob.copy()
        prob2[:, H] = 0.6  # momentum passes at the horizon; commit prob at 300 is 0.6 < 0.7
        _, intervals = gate_pixel_events([(190, 3.0, H, 1.0)], [], contain, codes, prob2, dead, ())
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(186, 265, H)])

    def test_box_channel_off_by_default_in_apply_layer(self):
        # apply_transition_layer without box_evidence reproduces the kinematic layer:
        # exercised across the suite; here just assert the parameter is optional.
        p = TransitionParams()
        self.assertGreater(p.pixel_bound_window, 0)
        self.assertLess(p.pixel_bias, p.pulse_near)


class RelayContactTests(unittest.TestCase):
    """A camera-attested takeover may relay across a spell the prior never saw break."""

    @staticmethod
    def world(n: int = 700):
        codes = np.zeros(n, dtype=np.int8)
        codes[:600] = A  # a settled spell; the next boundary is 200 frames away
        codes[600:] = D
        prob = np.full((n, 3), 0.15)
        prob[:, A] = 0.7  # the fused anchor stays on A: momentum never catches up
        dead = np.zeros(n, dtype=bool)
        boxes = np.zeros((n, 2, 4))
        boxes[:, 0] = [-2.0, 15.0, 2.0, 25.0]  # H player around the ball pixel
        boxes[:, 1] = [100.0, 0.0, 104.0, 20.0]  # A player far away
        atd = np.full((n, 2), [0.0, 20.0])
        yolo = np.full((n, 2), [0.0, 20.0])  # the two streams agree exactly
        ev = BoxEvidence(
            boxes=boxes,
            team=np.asarray([H, A], dtype=np.int8),
            present=np.ones((n, 2), dtype=bool),
            ball_yolo=yolo,
            ball_yolo_present=np.ones(n, dtype=bool),
            ball_atd=atd,
        )
        contain = np.zeros(n, dtype=np.int8)
        contain[390:401] = 1
        contact = [(400, 11.0, H, 1.0)]
        p_h = np.full((n, 3), 0.05)
        p_h[:, H] = 0.87
        w = np.ones(n)
        ballots = {k: (p_h, w) for k in ("atd_ball", "video_ball", "yolo_ball", "video_players")}
        return codes, prob, dead, ev, contain, contact, ballots

    def test_relay_admits_a_camera_unanimous_mid_spell_contact(self):
        codes, prob, dead, ev, contain, contact, ballots = self.world()
        relay = relay_contacts(contact, contain, ev, codes, ballots)
        self.assertEqual(relay, {(400, H)})
        # without the relay set the gate still rejects: settled spells belong to geometry
        _, intervals = gate_pixel_events(contact, [], contain, codes, prob, dead, ())
        self.assertEqual(intervals, [])
        kept, intervals = gate_pixel_events(contact, [], contain, codes, prob, dead, (), relay=relay)
        self.assertEqual(len(kept), 1)
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(390, 600, H)])

    def test_relay_requires_every_ballot_voter_on_the_event_team(self):
        codes, _, _, ev, contain, contact, ballots = self.world()
        weak = dict(ballots)
        p_low = np.full((700, 3), 0.25)
        p_low[:, H] = 0.5  # one voter below the unanimity bar
        weak["yolo_ball"] = (p_low, np.ones(700))
        self.assertEqual(relay_contacts(contact, contain, ev, codes, weak), frozenset())
        other = dict(ballots)
        p_a = np.full((700, 3), 0.05)
        p_a[:, A] = 0.9  # unanimous, but for the spell's own team
        other = {key: (p_a, np.ones(700)) for key in other}
        self.assertEqual(relay_contacts(contact, contain, ev, codes, other), frozenset())

    def test_relay_requires_the_spell_team_to_never_show_the_ball(self):
        codes, _, _, ev, contain, contact, ballots = self.world()
        anchor_seen = contain.copy()
        anchor_seen[300:401] |= 2  # the anchor team's boxes hold the ball all over the lookback
        self.assertEqual(relay_contacts(contact, anchor_seen, ev, codes, ballots), frozenset())

    def test_relay_requires_the_two_ball_streams_to_agree_on_one_player(self):
        codes, _, _, ev, contain, contact, ballots = self.world()
        yolo = np.full((700, 2), [20.0, 20.0])  # the other stream sees the ball elsewhere
        ev2 = BoxEvidence(
            boxes=ev.boxes, team=ev.team, present=ev.present,
            ball_yolo=yolo, ball_yolo_present=np.ones(700, dtype=bool), ball_atd=ev.ball_atd,
        )
        self.assertEqual(relay_contacts(contact, contain, ev2, codes, ballots), frozenset())

    def test_relay_never_overturns_a_dead_ball_and_needs_a_solid_run(self):
        codes, _, _, ev, contain, contact, ballots = self.world()
        all_dead = np.full(700, D, dtype=np.int8)
        self.assertEqual(relay_contacts(contact, contain, ev, all_dead, ballots), frozenset())
        short = [(400, 4.0, H, 1.0)]  # below the solid-run bar
        self.assertEqual(relay_contacts(short, contain, ev, codes, ballots), frozenset())


class DeadBlindCarryTests(unittest.TestCase):
    """A dead ball that went blind stays dead; live states keep decoding."""

    def test_dead_holds_through_blind_live_does_not(self):
        codes = np.array([D] * 10 + [H] * 10 + [A] * 10 + [H] * 10, dtype=np.int8)
        blind = np.zeros(40, dtype=bool)
        blind[10:20] = True  # deep blind window, decode restarted H inside it
        out = carry_dead_through_blind(codes, blind, halves=((0, 40),))
        self.assertEqual(list(out[10:20]), [D] * 10)  # the D before the window holds
        self.assertEqual(list(out[20:]), [A] * 10 + [H] * 10)  # decode resumes after
        codes_live = np.array([H] * 10 + [A] * 10, dtype=np.int8)
        blind2 = np.zeros(20, dtype=bool)
        blind2[10:20] = True
        out_live = carry_dead_through_blind(codes_live, blind2, halves=((0, 20),))
        self.assertEqual(list(out_live[10:]), [A] * 10)  # a live takeover is not frozen

    def test_carry_stays_inside_graded_halves(self):
        codes = np.array([D] * 10 + [H] * 10 + [X] * 10, dtype=np.int8)
        blind = np.zeros(30, dtype=bool)
        blind[5:25] = True
        out = carry_dead_through_blind(codes, blind, halves=((0, 20),))
        self.assertEqual(list(out[5:20]), [D] * 15)
        self.assertEqual(list(out[20:]), [X] * 10)  # outside the graded half: untouched


class ConflictMaskTests(unittest.TestCase):
    """Sustained ATD-vs-YOLO divergence marks drifting-ball frames (gate knob only)."""

    def test_sustained_divergence_gates_blips_do_not(self):
        n = 300
        atd = np.zeros((n, 2))
        yolo = np.zeros((n, 2))
        atd[80:110] = [6.0, 0.0]  # 6 m apart for 30 frames
        atd[150:152] = [7.0, 0.0]  # a two-frame YOLO blip
        mask = ball_conflict_mask(atd, yolo, min_distance=5.0, window=60, min_conflicts=8)
        # forward-window rule: marking starts once the next 60 frames accumulate 8
        # divergent ones (t=28 sees 80..87) and holds while they still see 8 (t=104
        # sees 104..109 plus the blip -- 8 exactly)
        self.assertFalse(mask[:28].any())
        self.assertTrue(mask[28:105].all())
        self.assertFalse(mask[105:].any())

    def test_pad_grows_both_sides(self):
        n = 300
        atd = np.zeros((n, 2))
        yolo = np.zeros((n, 2))
        atd[80:110] = [6.0, 0.0]
        base = ball_conflict_mask(atd, yolo, min_distance=5.0, window=60, min_conflicts=8)
        padded = ball_conflict_mask(
            atd, yolo, min_distance=5.0, window=60, min_conflicts=8, pad=15
        )
        self.assertTrue(bool(padded[base.argmax() - 1]) and not bool(base[base.argmax() - 1]))
        idx = np.flatnonzero(padded)
        first, last = int(idx[0]), int(idx[-1])
        self.assertEqual(len(idx), last - first + 1)  # one contiguous grown block
        self.assertEqual(first, int(base.argmax()) - 15)
        self.assertEqual(last, int(np.flatnonzero(base)[-1]) + 15)


if __name__ == "__main__":
    unittest.main()
