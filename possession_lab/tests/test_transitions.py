"""Tests for the event-gated ball-control transition layer."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from possession_lab.config import FPS, A, D, H, X
from possession_lab.fusion.transitions import (
    DEFAULT_TRANSITION_PARAMS,
    BoxEvidence,
    TransitionEvent,
    TransitionParams,
    apply_transition_layer,
    ball_conflict_mask,
    ball_release_mask,
    ball_speed,
    ball_stale_mask,
    blind_keeper_clearances,
    build_pulse,
    carry_dead_through_blind,
    control_map,
    dead_needs_support,
    decode_with_events,
    detect_contact_events,
    detect_kick_events,
    detect_kicker_events,
    detect_onset_events,
    events_to_rows,
    evidence_probability,
    expire_owned_spells,
    gate_events,
    gate_pixel_events,
    hold_through_flight,
    hold_touch_through,
    keeper_area_catches,
    keeper_holds,
    pixel_contact,
    raw_evidence_pool,
    relay_contacts,
    restart_ownership,
    rest_ownership,
    team_distances,
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


class OnsetTouchBypassTests(unittest.TestCase):
    """A ball killed on the taker's foot is a save or a tackle, not a lagging pool."""

    def base_arrays(self):
        codes = np.full(N, A, dtype=np.int8)
        prob = np.zeros((N, 3))
        prob[:, A] = 0.9
        prob[:, H] = 0.1
        dead = np.zeros(N, dtype=bool)
        return codes, prob, dead

    def nearest_with_h_control(self, frame: int, span: int = 12):
        nearest = np.full(N, A, dtype=np.int8)
        nearest[frame - span : frame] = H
        return nearest

    def test_touch_onset_survives_weak_pool_and_is_flagged(self):
        codes, prob, dead = self.base_arrays()
        events, _ = gate_events([], [(80, 12.0, H, 0.4)], codes, prob, dead, np.full(N, A, dtype=np.int8))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "onset")
        self.assertTrue(events[0].hold_through)  # the gate leaned on the touch alone

    def test_far_or_gentle_onset_still_needs_support(self):
        codes, prob, dead = self.base_arrays()
        nearest = np.full(N, A, dtype=np.int8)
        far, _ = gate_events([], [(80, 12.0, H, 1.2)], codes, prob, dead, nearest)
        self.assertEqual(far, ())  # 1.2 m is not a kill at the foot
        gentle, _ = gate_events([], [(80, 5.0, H, 0.4)], codes, prob, dead, nearest)
        self.assertEqual(gentle, ())  # the ball was barely moving already

    def test_kick_gets_no_touch_bypass(self):
        codes, prob, dead = self.base_arrays()
        events, _ = gate_events([(80, 11.0, H, 0.4)], [], codes, prob, dead, self.nearest_with_h_control(80))
        self.assertEqual(events, ())  # kicks need real support, however close the ball lands

    def test_supported_onset_is_not_flagged(self):
        codes, prob, dead = self.base_arrays()
        prob[70:120] = [0.6, 0.4, 0.0]  # the onset support window is [frame-10, frame+40)
        events, _ = gate_events([], [(80, 12.0, H, 0.4)], codes, prob, dead, np.full(N, A, dtype=np.int8))
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].hold_through)  # the pool agreed; nothing special to carry


class HoldTouchThroughTests(unittest.TestCase):
    """A witnessed takeover survives *decoded* deadness until a supported stoppage."""

    def setUp(self):
        self.n = 300
        self.codes = np.zeros(self.n, dtype=np.int8)
        self.codes[150:250] = D
        self.event = TransitionEvent(
            start=170, frame=175, kind="onset", team=H, distance=0.4, support=0.02, hold_through=True
        )

    def runs(self, codes):
        out, s = [], 0
        for i in range(1, len(codes)):
            if codes[i] != codes[s]:
                out.append((s, i, int(codes[s])))
                s = i
        out.append((s, len(codes), int(codes[s])))
        return out

    def test_fills_unsupported_dead_to_the_first_supported_stoppage(self):
        support = np.zeros(self.n, dtype=bool)
        support[220:250] = True
        out = hold_touch_through(self.codes, [self.event], np.zeros(self.n, bool), support)
        self.assertEqual(self.runs(out), [(0, 150, H), (150, 170, D), (170, 220, H), (220, 250, D), (250, 300, H)])

    def test_tagged_dead_ends_the_hold(self):
        tagged = np.zeros(self.n, dtype=bool)
        tagged[200:250] = True
        out = hold_touch_through(self.codes, [self.event], tagged, tagged)
        self.assertTrue((out[170:200] == H).all())
        self.assertTrue((out[200:250] == D).all())  # the tag is the authority

    def test_other_team_live_possession_ends_the_hold(self):
        codes = self.codes.copy()
        codes[190:220] = A
        support = np.zeros(self.n, dtype=bool)
        support[220:250] = True
        out = hold_touch_through(codes, [self.event], np.zeros(self.n, bool), support)
        self.assertTrue((out[170:190] == H).all())
        self.assertTrue((out[190:220] == A).all())  # the other team's live ball is untouched
        self.assertTrue((out[220:250] == D).all())

    def test_flags_and_cap_bound_the_hold(self):
        params = DEFAULT_TRANSITION_PARAMS
        off = params.onset_touch_hold_frames
        support = np.zeros(self.n, dtype=bool)
        support[220:250] = True
        plain = hold_touch_through(self.codes, [replace(self.event, hold_through=False)], np.zeros(self.n, bool), support)
        self.assertEqual(self.runs(plain), [(0, 150, H), (150, 250, D), (250, 300, H)])
        capped = hold_touch_through(
            self.codes, [replace(self.event, start=30, frame=35)], np.zeros(self.n, bool), support
        )
        self.assertTrue((capped[150 : 35 + off] == H).all())
        self.assertTrue((capped[35 + off : 250] == D).all())
        disabled = hold_touch_through(
            self.codes, [self.event], np.zeros(self.n, bool), support, replace(params, onset_touch_hold_frames=0)
        )
        self.assertEqual(self.runs(disabled), [(0, 150, H), (150, 250, D), (250, 300, H)])

    def test_next_event_start_cuts_the_hold(self):
        later = TransitionEvent(start=200, frame=205, kind="kick", team=A, distance=1.0, support=0.2)
        support = np.zeros(self.n, dtype=bool)
        support[220:250] = True
        out = hold_touch_through(self.codes, [self.event, later], np.zeros(self.n, bool), support)
        self.assertTrue((out[170:200] == H).all())
        self.assertTrue((out[200:220] == D).all())  # the next event owns the boundary, not the hold


class HoldThroughFlightTests(unittest.TestCase):
    """A spell is the team that last played the ball, frame by frame.

    A play is an anchor, strongest first: a witnessed event, a release (the ball
    leaves a slow start to live speed with its player at the feet), or a trap
    from rest (a player takes on a ball that was (nearly) at rest).  Frames are
    re-labelled only while the latest anchor in the spell's own window governs
    them; with no play in the window the decode's verdict stands.
    """

    def setUp(self):
        self.n = 300
        self.codes = np.full(self.n, H, dtype=np.int8)
        self.codes[100:200] = A
        self.dist = np.full((self.n, 2), 10.0)  # ball ten metres from everyone
        self.valid = np.ones(self.n, dtype=bool)
        self.speed = np.zeros(self.n)  # a resting ball

    def run_rule(self, codes=None, events=(), dead=None, params=None):
        return hold_through_flight(
            np.asarray(self.codes if codes is None else codes, dtype=np.int8),
            self.dist,
            self.valid,
            self.speed,
            list(events),
            None,
            dead,
            params=params or DEFAULT_TRANSITION_PARAMS,
        )

    def near_contact(self, team, f0, f1, d=1.0):
        """A player of ``team`` stands at the ball's feet on ``[f0, f1)``: with the
        ball at rest, three frames of this is a trap from rest, the play's anchor."""
        self.dist[f0:f1, team] = d

    def test_no_anchor_leaves_everything_alone(self):
        out, changed = self.run_rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == self.codes).all())  # nothing was played: no verdict

    def test_witnessed_event_anchors_the_spell_that_follows(self):
        event = TransitionEvent(start=190, frame=190, kind="kick", team=A, distance=1.0, support=0.2)
        out, changed = self.run_rule(events=[event])
        self.assertTrue((out[:100] == H).all())
        self.assertTrue((out[100:] == A).all())  # the A spell and the H spell after it
        self.assertEqual(changed, 100)  # the H spell 200-299 re-labelled to A

    def test_play_just_before_the_boundary_counts(self):
        self.near_contact(A, 195, 198)  # a trap 5 frames inside the lookback window
        out, changed = self.run_rule()
        self.assertTrue((out[:100] == H).all())
        self.assertTrue((out[100:] == A).all())
        self.assertEqual(changed, 100)

    def test_play_too_far_before_the_boundary_does_not_count(self):
        self.near_contact(A, 120, 123)  # 80 frames before the boundary: too old
        out, changed = self.run_rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == self.codes).all())  # the spell after has no play: it stands

    def test_release_anchors_the_spell_when_nobody_witnessed_the_kick(self):
        self.speed[191:] = 15.0  # the ball leaves the foot at 190
        self.dist[190, A] = 1.0  # the releasing player at its feet
        out, changed = self.run_rule()
        self.assertTrue((out[:100] == H).all())
        self.assertTrue((out[100:] == A).all())
        self.assertEqual(changed, 100)

    def test_release_needs_the_player_clearly_at_the_feet(self):
        self.speed[191:] = 15.0
        self.dist[190, A] = 3.0  # three metres: not at the feet
        out, changed = self.run_rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == self.codes).all())

    def test_a_live_speed_ball_is_not_a_trap(self):
        self.speed[195:] = 15.0  # the ball is rolling at live speed
        self.near_contact(A, 195, 205)  # a player shadows it, but it was never at rest
        out, changed = self.run_rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == self.codes).all())

    def test_unseen_ball_is_not_judged(self):
        codes = self.codes.copy()
        codes[200:] = A  # one A spell to 299
        self.valid[200:] = False
        self.near_contact(A, 195, 198)
        out, changed = self.run_rule(codes)
        self.assertEqual(changed, 0)
        self.assertTrue((out == codes).all())

    def test_spell_after_deadness_inherits_the_play_before_it(self):
        codes = self.codes.copy()
        codes[100:200] = D
        self.near_contact(A, 95, 98)  # Athletic took the ball just before it went blind
        out, changed = self.run_rule(codes)
        self.assertTrue((out[:100] == H).all())
        self.assertTrue((out[100:200] == D).all())  # the stoppage itself is not judged
        self.assertTrue((out[200:] == A).all())
        self.assertEqual(changed, 100)

    def test_spell_after_deadness_without_recent_play_is_left_alone(self):
        codes = np.full(self.n, D, dtype=np.int8)
        codes[:50] = A
        codes[100:] = H
        self.near_contact(A, 10, 13)  # played, but older than the dead run plus lookback
        out, changed = self.run_rule(codes)
        self.assertTrue((out[:50] == A).all())
        self.assertTrue((out[50:100] == D).all())
        self.assertTrue((out[100:] == H).all())  # no play in reach: the decode stands
        self.assertEqual(changed, 0)

    def test_deadness_suppresses_kinematic_anchors_but_not_witnessed_events(self):
        codes = np.full(self.n, D, dtype=np.int8)
        codes[:50] = A
        codes[100:] = H
        dead = np.zeros(self.n, dtype=bool)
        dead[50:100] = True
        self.near_contact(A, 70, 73)  # inside the stoppage: not play
        out, changed = self.run_rule(codes, dead=dead)
        self.assertEqual(changed, 0)  # the suppressed contact leaves no anchor at all
        event = TransitionEvent(start=99, frame=99, kind="kick", team=A, distance=1.0, support=0.2)
        out, changed = self.run_rule(codes, dead=dead, events=[event])  # the taken restart
        self.assertTrue((out[100:] == A).all())
        self.assertEqual(changed, 200)

    def test_anchor_older_than_the_rollback_window_does_not_count(self):
        n = 1000
        codes = np.full(n, D, dtype=np.int8)
        codes[:50] = A
        codes[900:] = H
        dist = np.full((n, 2), 10.0)
        dist[100:103, A] = 1.0  # inside the stoppage, 800 frames before the spell
        speed = np.zeros(n)
        valid = np.ones(n, dtype=bool)
        out, changed = hold_through_flight(codes, dist, valid, speed)
        self.assertTrue((out[900:] == H).all())  # the old play is out of reach: the decode stands
        self.assertEqual(changed, 0)
        dist = np.full((n, 2), 10.0)
        dist[200:203, A] = 1.0  # inside the window: the spell follows it
        out, changed = hold_through_flight(codes, dist, valid, speed)
        self.assertTrue((out[900:] == A).all())
        self.assertEqual(changed, 100)

    def test_half_clips_the_lookback_window(self):
        codes = np.full(self.n, D, dtype=np.int8)
        codes[:100] = A
        codes[120:] = A
        self.near_contact(A, 90, 93)  # before the second half begins
        halves = ((0, 100), (120, self.n))
        out, changed = hold_through_flight(codes, self.dist, self.valid, self.speed, (), halves)
        self.assertEqual(changed, 0)  # the play before the restart cannot cross it
        self.assertTrue((out == codes).all())

    def test_rule_can_be_disabled(self):
        self.near_contact(A, 195, 198)
        out, changed = self.run_rule(params=replace(DEFAULT_TRANSITION_PARAMS, flight_max_rollback_frames=0))
        self.assertEqual(changed, 0)
        self.assertTrue((out == self.codes).all())


class ReleaseTests(unittest.TestCase):
    """A release: the ball leaves the foot — a kick, a punt, a clearance."""

    def test_release_needs_a_slow_start_and_a_real_speed(self):
        speed = np.array([0.0, 5.0, 15.0, 20.0, 18.0, 18.0])
        mask = ball_release_mask(speed, gain=6.0, min_speed=8.0, max_speed=32.0, window_frames=3)
        self.assertTrue(mask[0] and mask[1])
        self.assertFalse(mask[2:] .any())

    def test_release_rejects_detector_jumps_and_gentle_moves(self):
        jump = ball_release_mask(np.array([0.0, 40.0, 40.0]), gain=6.0, min_speed=8.0, max_speed=32.0, window_frames=3)
        self.assertFalse(jump.any())  # 40 m/s is a jump, not a foot
        gentle = ball_release_mask(np.array([5.0, 6.0, 6.5, 10.0]), gain=6.0, min_speed=8.0, max_speed=32.0, window_frames=3)
        self.assertFalse(gentle.any())  # the peak stays under live speed: the ball rolling on

    def test_release_never_fires_on_a_lost_ball(self):
        speed = np.array([np.nan, 15.0, 15.0])
        self.assertFalse(ball_release_mask(speed, gain=6.0, min_speed=8.0, max_speed=32.0, window_frames=3).any())
class RestOwnershipTests(unittest.TestCase):
    """A stopped, seen ball belongs to the player clearly nearest to it."""

    def setUp(self):
        self.n = 100
        self.codes = np.full(self.n, D, dtype=np.int8)
        self.dist = np.full((self.n, 2), 10.0)
        self.speed = np.zeros(self.n)
        self.valid = np.ones(self.n, dtype=bool)
        self.dead = np.zeros(self.n, dtype=bool)
        self.consensus = np.full((self.n, 2), 50.0)

    def rule(self, **kw):
        return rest_ownership(
            self.codes, self.dist, self.speed, self.valid, self.dead, self.consensus,
            halves=kw.get("halves"), pitch=kw.get("pitch"),
            params=kw.get("params", DEFAULT_TRANSITION_PARAMS),
        )

    def test_resting_ball_goes_to_the_nearest_team(self):
        self.dist[:, H] = 2.0  # a home player two metres from the dead ball
        out, changed = self.rule()
        self.assertTrue((out == H).all())
        self.assertEqual(changed, self.n)

    def test_player_on_the_ball_is_ownership_even_in_a_scramble(self):
        # on-ball clause: a foot within half a metre and a metre ahead of the
        # opponent — the clear-stop clause (8 m of margin) does not apply here
        self.dist[:, H] = 0.4
        self.dist[:, A] = 1.6
        out, changed = self.rule()
        self.assertTrue((out == H).all())
        self.assertEqual(changed, self.n)

    def test_on_the_ball_needs_a_clear_foot(self):
        self.dist[:, H] = 0.4
        self.dist[:, A] = 1.2  # eighty centimetres ahead: contested, not a foot
        out, changed = self.rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == D).all())

    def test_contested_ball_is_left_alone(self):
        self.dist[:, H] = 2.0
        self.dist[:, A] = 3.0  # one metre apart: nobody clearly nearest
        out, changed = self.rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == D).all())

    def test_player_passing_by_is_not_ownership(self):
        self.dist[50:58, H] = 2.0  # eight frames: a walk-past, not a setup
        out, changed = self.rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == D).all())

    def test_speed_blip_inside_the_rest_run_is_bridged(self):
        self.dist[:, H] = 2.0
        self.speed[60:63] = 5.0  # three frames of tracker noise
        out, changed = self.rule()
        self.assertTrue((out == H).all())
        self.assertEqual(changed, self.n)

    def test_speed_run_is_not_bridged(self):
        self.dist[:, H] = 2.0
        self.speed[60:65] = 5.0  # five frames: the ball really moved
        out, changed = self.rule()
        self.assertTrue((out[:60] == H).all())
        self.assertTrue((out[60:65] == D).all())
        self.assertTrue((out[65:] == H).all())
        self.assertEqual(changed, self.n - 5)

    def test_dead_or_unseen_ball_is_not_judged(self):
        self.dist[:, H] = 2.0
        self.dead[:] = True
        out, changed = self.rule()
        self.assertEqual(changed, 0)
        self.dead[:] = False
        self.valid[:] = False
        out, changed = self.rule()
        self.assertEqual(changed, 0)
        self.assertTrue((out == D).all())

    def test_out_of_bounds_ball_is_not_judged(self):
        self.dist[:, H] = 2.0
        self.consensus[:, 0] = -5.0  # well past the touchline
        out, changed = self.rule(pitch=(105.0, 68.0))
        self.assertEqual(changed, 0)
        self.assertTrue((out == D).all())
class TeamDistancesTests(unittest.TestCase):
    def test_per_team_minimum_distance(self):
        n = 4
        ball = np.zeros((n, 2))
        ball[:, 0] = np.arange(n)
        pitch = np.full((n, 2, 2), 100.0)
        pitch[:, 0] = (0.0, 0.0)  # home player on every frame
        pitch[:, 1] = (0.5, 0.0)
        teams = np.array([H, A], dtype=np.int8)
        present = np.ones((n, 2), dtype=bool)
        d = team_distances(ball, [(pitch, teams, present)])
        self.assertTrue(np.allclose(d[:, H], ball[:, 0]))
        self.assertTrue(np.allclose(d[:, A], np.abs(ball[:, 0] - 0.5)))

    def test_unseen_ball_and_absent_team_are_infinite(self):
        ball = np.array([[np.nan, np.nan], [1.0, 1.0]])
        pitch = np.full((2, 1, 2), 5.0)
        teams = np.array([A], dtype=np.int8)
        present = np.array([[True], [True]])
        d = team_distances(ball, [(pitch, teams, present)])
        self.assertTrue(np.isinf(d[0]).all())  # the ball is unseen
        self.assertTrue(np.isinf(d[:, H]).all())  # no home player at all
        self.assertTrue(np.isfinite(d[1, A]))
        present[1] = False
        d = team_distances(ball, [(pitch, teams, present)])
        self.assertTrue(np.isinf(d[1, A]))  # the only away player is gone


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
        # the ball dries at 1 m/s straight into the home player while the chase is
        # 15 m away: a dead ball this close is his even before the fused pool
        # catches up (a direct observation outranks the decode's memory)
        self.assertEqual(result.codes[0], A)  # frame zero has no ball speed: no verdict
        self.assertTrue((result.codes[1:kick_at] == H).all())
        rows = events_to_rows(result.events)
        self.assertEqual(rows.shape[1], 6)

    def test_evidence_probability_matches_linear_pool(self):
        prob = {"x": np.tile([0.8, 0.1, 0.1], (10, 1)), "y": np.tile([0.2, 0.7, 0.1], (10, 1))}
        weight = {"x": np.ones(10), "y": np.zeros(10)}
        out = evidence_probability(prob, weight)
        from possession_lab.fusion.geometry import PRIOR, PRIOR_WEIGHT

        expected = (PRIOR_WEIGHT * PRIOR + np.array([0.8, 0.1, 0.1])) / (PRIOR_WEIGHT + 1.0)
        np.testing.assert_allclose(out, np.tile(expected, (10, 1)), rtol=1e-9)

    def test_gap_between_halves_is_dead_not_forward_filled(self):
        n = N
        ball = np.tile(np.array([20.0, 20.0]), (n, 1))
        pitch = np.zeros((n, 1, 2))
        pitch[:, 0] = ball
        sets = [player_set(pitch, [H])]
        codes = np.full(n, H, dtype=np.int8)
        fused = np.zeros((n, 3))
        fused[:, H] = 0.9
        fused[:, A] = 0.05
        fused[:, D] = 0.05
        result = apply_transition_layer(
            codes_pooled=codes,
            fused_probability=fused,
            event_dead=np.zeros(n, dtype=bool),
            consensus_ball=ball,
            evidence_prob=fused,
            player_sets=sets,
            halves=((0, 60), (80, n)),
        )
        np.testing.assert_array_equal(result.codes[60:80], np.full(20, D, dtype=np.int8))


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

    def test_kicker_catches_a_cushioned_first_touch_release(self):
        n = 80
        boxes = np.zeros((n, 2, 4))
        boxes[:, 0] = [-2.0, 0.0, 2.0, 20.0]  # H player, feet at (0, 20)
        boxes[:, 1] = [60.0, 0.0, 64.0, 20.0]  # A player far away
        atd = np.zeros((n, 2))
        atd[:, :] = [0.0, 19.0]  # at H's feet
        for t in range(51, n):
            atd[t, 0] = 5.0 * (t - 50)  # released: 5 px/frame away
        speed = np.full(n, 4.0)
        speed[54:] = 10.0  # cushioned first touch: in flight 4 frames after the close
        ev = self.evidence(n, atd, boxes=boxes)
        _, touch = pixel_contact(ev)
        events = detect_kicker_events(ev, touch, speed)
        self.assertEqual([(f, t) for f, _, t, _, _ in events], [(50, H)])
        self.assertEqual(events[0][1], 10.0)  # release speed read from the 4th frame
        # the 0.16 s ceiling is relative to the ball leaving the feet: once it rolls
        # out of reach of both feet, a later acceleration is a second touch, not this
        atd_gap = atd.copy()
        atd_gap[51:56] = [25.0, 19.0]  # out of reach of both feet (20 px close window)
        ev_gap = self.evidence(n, atd_gap, boxes=boxes)
        _, touch_gap = pixel_contact(ev_gap)
        speed_late = np.full(n, 4.0)
        speed_late[56:] = 10.0
        self.assertEqual(detect_kicker_events(ev_gap, touch_gap, speed_late), [])

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

    def test_a_fingerprint_needs_the_pitch_stream_to_have_seen_the_ball(self):
        # feet evidence in a window the ATD ball is blind in has no ball position to check the
        # release against, and the poll there is carried by its own prior: no takeover
        n = 700
        _codes, prob, dead = self._gate_world(n, 260)
        codes2 = np.zeros(n, dtype=np.int8)
        codes2[:424] = A
        kicker = [(424, 11.0, H, 13.0, 57.0)]
        blind = np.ones(n, dtype=bool)
        blind[380:520] = False
        contain = np.zeros(n, dtype=np.int8)
        _kept, intervals = gate_pixel_events(
            [], kicker, contain, codes2, prob, dead, (), atd_ball=blind
        )
        self.assertEqual(intervals, [])
        _kept, intervals = gate_pixel_events(
            [], kicker, contain, codes2, prob, dead, (), atd_ball=np.ones(n, dtype=bool)
        )
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(423, 499, H)])
        # the requirement is a parameter, and momentum still carries a kick in a blind window
        params = TransitionParams(pixel_fingerprint_needs_atd=False)
        _kept, intervals = gate_pixel_events(
            [], kicker, contain, codes2, prob, dead, (), params, atd_ball=blind
        )
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(423, 499, H)])
        prob_momentum = prob.copy()
        prob_momentum[440:, H] = 0.8
        _kept, intervals = gate_pixel_events(
            [], kicker, contain, codes2, prob_momentum, dead, (), atd_ball=blind
        )
        self.assertTrue(intervals, "a blind kick the evidence agrees with still counts")

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
        ballots = {k: (p_h, w) for k in ("atd_ball", "yolo_ball", "video_players")}
        return codes, prob, dead, ev, contain, contact, ballots

    def test_relay_admits_a_camera_unanimous_mid_spell_contact(self):
        codes, prob, dead, ev, contain, contact, ballots = self.world()
        relay = relay_contacts(contact, contain, ev, codes, ballots)
        self.assertEqual(relay, {(400, H)})
        # box lock: a solid in-box run the holder never shows is accepted even without relay
        _, intervals = gate_pixel_events(contact, [], contain, codes, prob, dead, ())
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(390, 600, H)])
        kept, intervals = gate_pixel_events(contact, [], contain, codes, prob, dead, (), relay=relay)
        self.assertEqual(len(kept), 1)
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(390, 600, H)])

    def test_box_lock_needs_a_solid_run(self):
        codes, prob, dead, _ev, contain, _contact, _ballots = self.world()
        weak_contact = [(400, 4.0, H, 1.0)]  # below box_lock_min_run
        _, intervals = gate_pixel_events(weak_contact, [], contain, codes, prob, dead, ())
        self.assertEqual(intervals, [])

    def test_box_lock_needs_a_holder_that_never_shows_the_ball(self):
        codes, prob, dead, _ev, contain, contact, _ballots = self.world()
        holder_contain = contain.copy()
        holder_contain[380] = 2  # the ball sits inside the holder's (A) box before the take
        _, intervals = gate_pixel_events(contact, [], holder_contain, codes, prob, dead, ())
        self.assertEqual(intervals, [])

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


class KeeperHoldTests(unittest.TestCase):
    """A goalkeeper holding the ball inside his own area is possession from the catch."""

    @staticmethod
    def world(
        n: int = 120,
        hold: tuple[int, int] = (40, 80),
        speed_px: float = 3.0,
        pitch_x: float = 10.0,
        defenders: tuple = ((H, A), (H, A)),
        dead=None,
    ):
        codes = np.full(n, A, dtype=np.int8)  # the spell reads the *other* team
        codes[100:] = D  # a dead-ball boundary the owned interval must stop at
        prob = np.full((n, 3), 0.15)
        prob[:, A] = 0.7
        dead = np.zeros(n, dtype=bool) if dead is None else dead
        boxes = np.zeros((n, 2, 4))
        boxes[:, 0] = [0.0, 0.0, 20.0, 40.0]  # holder box, centre (10, 20)
        boxes[:, 1] = [200.0, 0.0, 220.0, 40.0]  # opponent far away, centre (210, 20)
        pitch = np.zeros((n, 2, 2))
        pitch[:, 0] = [pitch_x, 34.0]  # holder stands on the goal line's area, x=0 side
        pitch[:, 1] = [100.0, 34.0]
        ball = np.full((n, 2), [500.0, 500.0])  # off scene: nobody holds the ball
        seg = slice(hold[0], hold[1])
        ball[seg] = [40.0, 20.0]  # 30 px from the holder box centre
        ball[seg, 0] += (np.arange(hold[1] - hold[0]) % 2) * speed_px  # jitter sets pixel speed
        ev = BoxEvidence(
            boxes=boxes,
            team=np.tile(np.array([H, A], dtype=np.int8), (n, 1)),
            present=np.ones((n, 2), dtype=bool),
            ball_yolo=ball,
            ball_yolo_present=np.ones(n, dtype=bool),
            ball_atd=ball.copy(),
            pitch=pitch,
        )
        contain = np.zeros(n, dtype=np.int8)
        defs = np.asarray(defenders, dtype=np.int8)
        return codes, prob, dead, ev, contain, defs, ((0, n),)

    def test_a_keeper_hold_inside_its_own_area_fires_and_owns_the_spell(self):
        codes, prob, dead, ev, contain, defs, halves = self.world()
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves)
        self.assertEqual(len(holds), 1)
        self.assertEqual((holds[0].start, holds[0].frame, holds[0].team), (40, 79, H))
        kept, intervals = gate_pixel_events([], [], contain, codes, prob, dead, (), holds=holds)
        self.assertEqual([e.kind for e in kept], ["hold"])
        # owned from the catch to the next fused code boundary, never into the dead ball
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(40, 100, H)])

    def test_a_hold_never_outruns_the_camera_that_watched_it(self):
        # the fused poll going blind is not the keeper holding the ball for another 300 frames: the
        # owned interval stops at the release cap, not wherever the poll happened to wake up
        codes, prob, dead, ev, contain, defs, halves = self.world(n=400)
        codes[100:380] = A
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves)
        self.assertEqual(len(holds), 1)
        _kept, intervals = gate_pixel_events([], [], contain, codes, prob, dead, (), holds=holds)
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(40, 79 + 150, H)])
        # the cap is a parameter: off, the interval runs to the fused boundary again
        params = TransitionParams(keeper_hold_release_frames=0)
        _kept, intervals = gate_pixel_events([], [], contain, codes, prob, dead, (), params, holds=holds)
        self.assertEqual([(a, b, t) for a, b, t in intervals], [(40, 380, H)])

    def test_a_fast_ball_past_the_area_is_not_a_hold(self):
        _codes, _prob, dead, ev, contain, defs, halves = self.world(speed_px=12.0)
        self.assertEqual(keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves), [])

    def test_a_hold_needs_the_defending_side_of_that_goal(self):
        _codes, _prob, dead, ev, contain, defs, halves = self.world(defenders=((A, H), (A, H)))
        self.assertEqual(keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves), [])

    def test_a_hold_needs_the_penalty_area(self):
        _codes, _prob, dead, ev, contain, defs, halves = self.world(pitch_x=40.0)
        self.assertEqual(keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves), [])

    def test_dead_frames_do_not_count_into_a_hold(self):
        dead = np.zeros(120, dtype=bool)
        dead[40:80] = True
        _codes, _prob, dead, ev, contain, defs, halves = self.world(dead=dead)
        self.assertEqual(keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves), [])

    def test_opponent_containment_breaks_the_hold(self):
        _codes, _prob, dead, ev, contain, defs, halves = self.world()
        contain[50:56] = 2  # six opponent ball-in-box frames across the hold
        self.assertEqual(keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves), [])

    def test_holder_may_change_detector_id_inside_the_hold(self):
        # identity is positional, not the detector id: the ball sits still while two
        # overlapping boxes of the same keeper alternate as the nearest one
        n = 80
        dead = np.zeros(n, dtype=bool)
        boxes = np.zeros((n, 3, 4))
        boxes[:, 1] = [200.0, 0.0, 220.0, 40.0]  # opponent far away
        for t in range(n):
            if t % 2 == 0:
                boxes[t, 0] = [24.0, 0.0, 32.0, 40.0]  # keeper box A, centre (28, 20)
                boxes[t, 2] = [46.0, 0.0, 54.0, 40.0]  # keeper box B, centre (50, 20)
            else:
                boxes[t, 2] = [25.0, 0.0, 33.0, 40.0]  # now B is nearest (29, 20)
                boxes[t, 0] = [46.0, 0.0, 54.0, 40.0]
        pitch = np.zeros((n, 3, 2))
        pitch[:, 0] = [10.0, 34.0]  # both boxes report the keeper's own-area pitch
        pitch[:, 2] = [11.0, 34.0]
        pitch[:, 1] = [100.0, 34.0]
        ball = np.full((n, 2), [30.0, 20.0])  # the held ball: not moving
        ev = BoxEvidence(
            boxes=boxes,
            team=np.tile(np.array([H, A, H], dtype=np.int8), (n, 1)),
            present=np.ones((n, 3), dtype=bool),
            ball_yolo=ball,
            ball_yolo_present=np.ones(n, dtype=bool),
            ball_atd=ball.copy(),
            pitch=pitch,
        )
        holds = keeper_holds(ev, np.zeros(n, dtype=np.int8), dead,
                             np.array([[H, A], [H, A]], dtype=np.int8), (105.0, 68.0), ((0, n),))
        self.assertEqual(len(holds), 1)
        self.assertEqual((holds[0].start, holds[0].frame, holds[0].team), (0, 79, H))

    def test_gate_skips_a_hold_when_the_spell_is_already_dead(self):
        codes, _prob, dead, ev, contain, defs, halves = self.world()
        codes[:] = D
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves)
        _, intervals = gate_pixel_events([], [], contain, codes, np.full((120, 3), 0.4), dead, (), holds=holds)
        self.assertEqual(intervals, [])

    def test_a_hold_hands_over_at_the_pixel_owned_front_edge(self):
        # a relay contact decides the boundary itself: the hold keeps the catch up to
        # that interval's front edge instead of losing the whole stretch to overlap
        codes, prob, dead, ev, contain, defs, halves = self.world()
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves)
        kept, intervals = gate_pixel_events(
            [(99, 5.0, H)], [], contain, codes, prob, dead, (),
            relay=frozenset({(99, H)}), holds=holds,
        )
        self.assertEqual([e.kind for e in kept], ["contact", "hold"])
        self.assertEqual(sorted(intervals), [(40, 99, H), (99, 100, H)])

    def test_gate_skips_a_hold_the_pixel_channel_already_owns_whole(self):
        codes, prob, dead, ev, contain, defs, halves = self.world()
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves)
        kept, _intervals = gate_pixel_events(
            [(35, 5.0, H)], [], contain, codes, prob, dead, (),
            relay=frozenset({(35, H)}), holds=holds,
        )
        self.assertEqual([e.kind for e in kept], ["contact"])

    def test_channel_disabled_without_defenders_or_frames(self):
        _codes, _prob, dead, ev, contain, defs, halves = self.world()
        holds = keeper_holds(ev, contain, dead, None, (105.0, 68.0), halves)
        self.assertEqual(holds, [])
        params = TransitionParams(keeper_hold_min_frames=0)
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves, params)
        self.assertEqual(holds, [])

    @staticmethod
    def pass_world(interrupt: bool = False, n: int = 140, catch: int = 90):
        """A keeper catch at 90, released by a teammate at 50 after a clean flight."""
        dead = np.zeros(n, dtype=bool)
        boxes = np.zeros((n, 3, 4))
        boxes[:, 0] = [0.0, 0.0, 20.0, 40.0]  # keeper box, centre (10, 20)
        boxes[:, 1] = [300.0, 0.0, 320.0, 40.0]  # passer, far from the held pixel
        boxes[:, 2] = [600.0, 0.0, 620.0, 40.0]  # opponent, farther still
        pitch = np.zeros((n, 3, 2))
        pitch[:, 0] = [10.0, 34.0]  # keeper, own area
        pitch[:, 1] = [30.0, 34.0]  # passer
        pitch[:, 2] = [20.0, 34.0] if interrupt else [100.0, 34.0]
        ball = np.full((n, 2), [500.0, 500.0])
        ball[catch:120] = [40.0, 20.0]  # 30 px from the keeper box centre
        ball_pitch = np.full((n, 2), np.nan)
        ball_pitch[50:70] = [30.0, 34.0]  # control at the passer's feet
        ball_pitch[70:catch, 0] = np.linspace(30.0, 10.0, catch - 70)
        ball_pitch[70:catch, 1] = 34.0  # flight to the keeper
        ball_pitch[catch:120] = [10.0, 34.0]  # held
        ev = BoxEvidence(
            boxes=boxes,
            team=np.tile(np.array([H, H, A], dtype=np.int8), (n, 1)),
            present=np.ones((n, 3), dtype=bool),
            ball_yolo=ball,
            ball_yolo_present=np.ones(n, dtype=bool),
            ball_atd=ball.copy(),
            pitch=pitch,
            ball_pitch=ball_pitch,
        )
        contain = np.zeros(n, dtype=np.int8)
        return dead, ev, contain, np.array([[H, A], [H, A]], dtype=np.int8), ((0, n),)

    def test_hold_extends_back_through_the_pass_to_the_release(self):
        dead, ev, contain, defs, halves = self.pass_world()
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves)
        self.assertEqual(len(holds), 1)
        self.assertEqual((holds[0].start, holds[0].frame, holds[0].team), (50, 119, H))

    def test_an_opponent_touch_blocks_the_pass_extension(self):
        dead, ev, contain, defs, halves = self.pass_world(interrupt=True)
        holds = keeper_holds(ev, contain, dead, defs, (105.0, 68.0), halves)
        self.assertEqual(len(holds), 1)
        self.assertEqual(holds[0].start, 90)  # the catch, not the pass


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


class DeadNeedsSupportTests(unittest.TestCase):
    """Inside a ball-blind window a dead state has to come from an event, not from the gap."""

    def world(self, n: int = 400):
        codes = np.zeros(n, dtype=np.int8)
        codes[:100] = H
        codes[100:250] = D  # the decode went dead inside a ball gap
        blind = np.zeros(n, dtype=bool)
        blind[120:250] = True
        return codes, blind

    def test_an_unsupported_blind_dead_ball_goes_back_to_the_live_team(self):
        codes, blind = self.world()
        support = np.zeros(codes.size, dtype=bool)
        out = dead_needs_support(codes, blind, support, halves=((0, codes.size),))
        self.assertEqual(list(out[100:120]), [D] * 20)  # the ball was still visible: untouched
        self.assertEqual(list(out[120:250]), [H] * 130)  # the gap alone never stops a match

    def test_a_supported_blind_dead_ball_stands(self):
        codes, blind = self.world()
        support = np.zeros(codes.size, dtype=bool)
        support[150:250] = True  # a confirmed stoppage / SET PIECES tag from frame 150
        out = dead_needs_support(codes, blind, support, halves=((0, codes.size),))
        self.assertEqual(list(out[120:150]), [H] * 30)
        self.assertEqual(list(out[150:250]), [D] * 100)

    def test_live_states_are_not_touched(self):
        codes, blind = self.world()
        codes[100:250] = A
        out = dead_needs_support(codes, blind, np.zeros(codes.size, dtype=bool), halves=((0, codes.size),))
        np.testing.assert_array_equal(out, codes)

    def test_a_half_with_no_live_team_is_left_alone(self):
        codes = np.full(60, D, dtype=np.int8)
        blind = np.ones(60, dtype=bool)
        out = dead_needs_support(codes, blind, np.zeros(60, dtype=bool), halves=((0, 60),))
        np.testing.assert_array_equal(out, codes)  # nothing to hand the frames back to

    def test_support_never_leaks_across_a_half_boundary(self):
        codes = np.zeros(200, dtype=np.int8)
        codes[:100] = H
        codes[100:] = D
        blind = np.ones(200, dtype=bool)
        out = dead_needs_support(codes, blind, np.zeros(200, dtype=bool), halves=((0, 100), (100, 200)))
        self.assertEqual(list(out[:100]), [H] * 100)
        self.assertEqual(list(out[100:]), [D] * 100)  # the second half never held a live team


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


class KeeperAreaCatchTests(unittest.TestCase):
    """The pitch-side twin of the keeper hold where the pixel channel goes blind at the goal.

    Measured 132 355: the ball decelerated 13.1 -> 1.1 m/s at x = 9.4 m (the Athletic
    keeper's six-yard box, second half) and left again at 18.9 m/s; the boxes near the
    goal compress in depth, the pixel channel saw nothing, and the fused prior kept
    claiming H through the catch and the punt.
    """

    def catch_world(self, n=240, opp_far=True, stop_x=9.6, dead=None):
        ramp = np.array([0.36, 0.24, 0.12, 0.06])  # 12 -> 0 m/s over frames 94..97
        off = np.zeros(n)
        for t in range(n):
            if t < 94:
                off[t] = 0.48 * (94 - max(t, 60)) + ramp.sum()
            elif t < 98:
                off[t] = ramp[t - 94 :].sum()
        punt = np.maximum(0.0, np.arange(n) - 105) * 0.72  # the punt away, 18 m/s
        x = stop_x + off + punt
        ball = np.stack([x, np.full(n, 34.0)], axis=1)
        pitch = np.zeros((n, 2, 2))
        pitch[:, 0] = [9.0, 34.4]  # the keeper of the side that defends goal 0
        pitch[:, 1] = [45.0, 34.0] if opp_far else [10.8, 33.0]
        teams = np.asarray([1, H], dtype=np.int8)
        sets = [(pitch, teams, np.ones((n, 2), dtype=bool))]
        defenders = np.array([[1, H]], dtype=np.int8)  # this half: A defends goal 0
        ev_dead = np.zeros(n, dtype=bool)
        if dead is not None:
            ev_dead[92:102] = dead
        return ball, sets, defenders, ev_dead

    def test_catch_in_own_area_is_won_by_the_defending_side(self):
        ball, sets, defenders, ev_dead = self.catch_world()
        from possession_lab.fusion.transitions import ball_speed

        speed, valid = ball_speed(ball)
        catches = keeper_area_catches(
            ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0), ((0, 240),)
        )
        self.assertEqual(len(catches), 1)
        start, frame, team = catches[0]
        self.assertEqual(team, 1)
        self.assertTrue(96 <= frame <= 100)
        self.assertTrue(60 <= start < 96)  # the pass-in kick, not the standstill

    def test_contested_catch_is_not_given(self):
        ball, sets, defenders, ev_dead = self.catch_world(opp_far=False)
        from possession_lab.fusion.transitions import ball_speed

        speed, valid = ball_speed(ball)
        self.assertEqual(
            keeper_area_catches(ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0), ((0, 240),)),
            [],
        )

    def test_midfield_stopper_is_not_an_area_catch(self):
        ball, sets, defenders, ev_dead = self.catch_world(stop_x=60.0)
        ball[:, 0] = np.where(np.arange(len(ball)) < 99, 60.0, ball[:, 0])
        from possession_lab.fusion.transitions import ball_speed

        speed, valid = ball_speed(ball)
        self.assertEqual(
            keeper_area_catches(ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0), ((0, 240),)),
            [],
        )

    def test_dead_catch_is_not_given(self):
        ball, sets, defenders, ev_dead = self.catch_world(dead=True)
        from possession_lab.fusion.transitions import ball_speed

        speed, valid = ball_speed(ball)
        self.assertEqual(
            keeper_area_catches(ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0), ((0, 240),)),
            [],
        )

    def test_neutral_pool_is_not_a_keeper_catch(self):
        """Measured 11 598: ball dead at the goal mouth with the voter pool flat (0.07) —
        a conceded goal, not a keeper gathering his side's ball.  The raw-pool margin
        must back the catcher or the channel stays quiet."""
        ball, sets, defenders, ev_dead = self.catch_world()
        from possession_lab.fusion.transitions import ball_speed

        speed, valid = ball_speed(ball)
        flat = np.full((len(ball), 3), 1.0 / 3.0)
        self.assertEqual(
            keeper_area_catches(ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0),
                                ((0, 240),), evidence_pool=flat),
            [],
        )
        backing = np.tile([0.20, 0.70, 0.10], (len(ball), 1))
        self.assertEqual(
            len(keeper_area_catches(ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0),
                                    ((0, 240),), evidence_pool=backing)),
            1,
        )


class AreaCatchArrivalTests(unittest.TestCase):
    """The arrival into the area must be a pass, not a tracker spike.

    One frame at flight speed inside the 40-frame arrival window is the tracker
    hiccuping over a slow roll into the box (measured 129 415: the ball rolled in
    at 7.3 m/s with a single 8.0 m/s frame); a real pass is at least two frames
    at or above ``area_catch_in_min_speed`` (``area_catch_in_frames``).
    """

    def arriving_world(self, fast_frames, n=240):
        # the ball parks, is fed in with `fast_frames` frames at 12 m/s, then
        # rolls and stops at (9.6, 34) in the defended left area
        delta = np.zeros(n)
        for t in range(60, 60 + fast_frames):
            delta[t] = 0.48  # 12 m/s
        for t in range(60 + fast_frames, 96):
            delta[t] = 0.06  # 1.5 m/s
        for t in range(96, 99):
            delta[t] = 0.02  # 0.5 m/s
        x = np.full(n, np.nan)
        x[0] = 9.6 + delta.sum()
        for t in range(1, n):
            x[t] = x[t - 1] - delta[t]
        ball = np.stack([x, np.full(n, 34.0)], axis=1)
        pitch = np.zeros((n, 2, 2))
        pitch[:, 0] = [9.0, 34.4]  # the keeper of the side that defends goal 0
        pitch[:, 1] = [45.0, 34.0]
        teams = np.asarray([1, H], dtype=np.int8)
        sets = [(pitch, teams, np.ones((n, 2), dtype=bool))]
        defenders = np.array([[1, H]], dtype=np.int8)  # this half: A defends goal 0
        ev_dead = np.zeros(n, dtype=bool)
        return ball, sets, defenders, ev_dead

    def test_single_fast_frame_is_not_a_pass_in(self):
        ball, sets, defenders, ev_dead = self.arriving_world(1)
        speed, valid = ball_speed(ball)
        # the old gate (one fast frame) read this as a catch...
        old = replace(DEFAULT_TRANSITION_PARAMS, area_catch_in_frames=1)
        legacy = keeper_area_catches(
            ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0), ((0, 240),), params=old
        )
        self.assertEqual(legacy, [(60, 61, 1)])  # anchored on the 12 m/s kick frame
        # ...the arrival gate rejects it: a slow roll with a one-frame hiccup
        self.assertEqual(
            keeper_area_catches(ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0), ((0, 240),)),
            [],
        )

    def test_two_fast_frames_are_the_pass_in(self):
        ball, sets, defenders, ev_dead = self.arriving_world(2)
        speed, valid = ball_speed(ball)
        catches = keeper_area_catches(
            ball, speed, valid, ev_dead, sets, defenders, (105.0, 68.0), ((0, 240),)
        )
        self.assertEqual(len(catches), 1)
        start, frame, team = catches[0]
        self.assertEqual(team, 1)
        self.assertEqual(frame, 62)
        self.assertEqual(start, 60)  # anchored on the kick that fed the keeper


class BlindKeeperClearanceTests(unittest.TestCase):
    """The blind twin of the area catch: the fused ball goes dark for a second or
    more while the keeper holds the ball, then returns as a downfield punt.

    Measured 69 405: blind 69 116-69 404 (confidence below 0.5), the ball back in
    the defended area at (7.4, 35.3) with the keeper 0.4 m from it, the punt
    building to 20.6 m/s, and the fused prior still claiming the other side
    through the clearance.
    """

    def clearance_world(
        self,
        n=200,
        ret=141,
        blind_frames=41,
        blind_conf=0.2,
        opp_dist=6.0,
        def_dist=0.4,
        reappear=(9.6, 34.0),
        punt_mps=14.0,
        dead=False,
        goals=(),
        invalid_blind=False,
    ):
        # parked until the blind run starts, blind for `blind_frames` frames, back
        # in the defended left area at `ret`, punted downfield at `punt_mps` m/s
        blind0 = ret - blind_frames
        ball = np.full((n, 2), np.nan)
        ball[:blind0, 0] = 30.0
        ball[:blind0, 1] = 34.0
        ball[blind0:ret, 0] = 30.0 + 0.1 * np.arange(ret - blind0)  # low-confidence ghosts
        ball[blind0:ret, 1] = 34.0
        step = punt_mps / FPS  # m per frame
        for t in range(ret, n):
            ball[t, 0] = reappear[0] + step * (t - ret)
            ball[t, 1] = reappear[1]
        conf = np.ones(n)
        conf[blind0:ret] = blind_conf
        if invalid_blind:
            ball[blind0:ret] = np.nan  # and the fused ball is simply absent
        speed, valid = ball_speed(ball)
        dead_mask = np.zeros(n, dtype=bool)
        if dead:
            dead_mask[ret] = True
        dist = np.full((n, 2), 6.0)
        dist[ret] = [opp_dist, def_dist]  # column 0 home, column 1 away
        defenders = np.array([[1, H]], dtype=np.int8)  # this half: A defends goal 0
        return ball, conf, speed, valid, dead_mask, dist, defenders, ret

    def _clears(self, world, goals=()):
        ball, conf, speed, valid, dead, dist, defenders, ret = world
        return blind_keeper_clearances(
            ball, conf, speed, valid, dead, dist, defenders, (105.0, 68.0), ((0, 200),),
            goals_windows=tuple(goals),
        )

    def test_blind_hold_and_punt_is_won_by_the_defending_side(self):
        world = self.clearance_world()
        _, _, _, _, _, _, _, ret = world
        self.assertEqual(self._clears(world), [(ret, ret, 1)])

    def test_short_blind_run_is_not_a_clearance(self):
        self.assertEqual(self._clears(self.clearance_world(blind_frames=15, ret=115)), [])

    def test_sighted_blind_run_is_not_a_clearance(self):
        # confidence stays above clearance_conf_min: the channel never went blind
        self.assertEqual(self._clears(self.clearance_world(blind_conf=0.9)), [])

    def test_invalid_frames_count_as_blind(self):
        world = self.clearance_world(blind_conf=1.0, invalid_blind=True)
        _, _, _, _, _, _, _, ret = world
        self.assertEqual(self._clears(world), [(ret, ret, 1)])

    def test_contested_reappearance_is_not_a_clearance(self):
        self.assertEqual(self._clears(self.clearance_world(opp_dist=1.5)), [])

    def test_defender_not_on_the_ball_is_not_a_clearance(self):
        self.assertEqual(self._clears(self.clearance_world(def_dist=2.5)), [])

    def test_shot_into_the_goal_is_not_a_clearance(self):
        # a shot leaves at flight speed too, but gains ground into the goal, not away
        self.assertEqual(self._clears(self.clearance_world(reappear=(9.6, 34.0), punt_mps=-14.0)), [])

    def test_slow_reappearance_is_not_a_clearance(self):
        # a hand pass gains ground but does not leave at punt speed
        self.assertEqual(self._clears(self.clearance_world(punt_mps=7.5)), [])

    def test_reappearance_outside_the_area_is_not_a_clearance(self):
        self.assertEqual(self._clears(self.clearance_world(reappear=(60.0, 34.0))), [])

    def test_reappearance_off_the_pitch_is_not_a_clearance(self):
        self.assertEqual(self._clears(self.clearance_world(reappear=(-1.0, 34.0))), [])

    def test_dead_reappearance_is_not_a_clearance(self):
        self.assertEqual(self._clears(self.clearance_world(dead=True)), [])

    def test_goals_window_suppresses_the_clearance(self):
        world = self.clearance_world()
        ret = 141
        self.assertEqual(self._clears(world, goals=((ret - 5, ret + 5),)), [])


class ExpireOwnedSpellsTests(unittest.TestCase):
    """A pixel-owned interval cannot outlive the camera's own evidence."""

    def world(self, n=300, prob_a=0.05, prob_h=0.90, speed_val=4.0):
        pool = np.full((n, 3), 1.0 / 3.0)
        pool[100:150] = [prob_h, prob_a, 1.0 - prob_h - prob_a]
        dead = np.zeros(n, dtype=bool)
        speed = np.full(n, speed_val)
        return pool, dead, speed

    def test_sustained_contrary_pool_cuts_the_interval(self):
        pool, dead, speed = self.world()
        intervals = [[60, 200, 1]]  # an A contact owns the stretch
        events, ivs = expire_owned_spells(intervals, pool, dead, speed)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].kind, events[0].team, events[0].start), ("control", H, 100))
        self.assertIn((60, 100, 1), ivs)

    def test_short_burst_does_not_expire_the_interval(self):
        pool, dead, speed = self.world()
        pool[100:130] = 1.0 / 3.0  # only 20 contrary frames
        events, ivs = expire_owned_spells([[60, 200, 1]], pool, dead, speed)
        self.assertEqual(events, [])
        self.assertEqual(list(ivs), [(60, 200, 1)])

    def test_weak_contrary_pool_does_not_expire_the_interval(self):
        pool, dead, speed = self.world(prob_a=0.25, prob_h=0.71)
        events, _ = expire_owned_spells([[60, 200, 1]], pool, dead, speed)
        self.assertEqual(events, [])

    def test_ball_in_flight_does_not_expire_the_interval(self):
        pool, dead, speed = self.world(speed_val=14.0)
        events, _ = expire_owned_spells([[60, 200, 1]], pool, dead, speed)
        self.assertEqual(events, [])

    def test_keeper_hold_is_never_expired(self):
        pool, dead, speed = self.world()
        events, ivs = expire_owned_spells([[60, 200, 1]], pool, dead, speed, protected=frozenset({(60, 1)}))
        self.assertEqual(events, [])
        self.assertEqual(list(ivs), [(60, 200, 1)])

    def test_run_must_lie_fully_inside_the_owned_interval(self):
        pool, dead, speed = self.world()
        pool[150:170] = [0.90, 0.05, 0.05]  # the contrary run runs past the interval end
        events, ivs = expire_owned_spells([[60, 160, 1]], pool, dead, speed)
        self.assertEqual(events, [])
        self.assertEqual(list(ivs), [(60, 160, 1)])


class ControlRowTests(unittest.TestCase):
    def test_control_and_area_kind_rows_round_trip(self):
        rows = events_to_rows(
            [
                TransitionEvent(start=10, frame=12, kind="control", team=H, distance=1.0, support=0.8),
                TransitionEvent(start=20, frame=24, kind="area", team=1, distance=1.0, support=1.0),
                TransitionEvent(start=30, frame=30, kind="clearance", team=H, distance=1.0, support=1.0),
            ]
        )
        self.assertEqual(rows[0, 2], 5)
        self.assertEqual(rows[1, 2], 6)
        self.assertEqual(rows[2, 2], 9)


class RawPoolTests(unittest.TestCase):
    def test_single_voter_pool_is_exact_and_unprior(self):
        prob = {"x": np.tile([0.9, 0.05, 0.05], (4, 1))}
        weight = {"x": np.ones(4)}
        np.testing.assert_allclose(raw_evidence_pool(prob, weight), np.tile([0.9, 0.05, 0.05], (4, 1)))

    def test_zero_weight_frames_fall_back_to_uniform(self):
        prob = {"x": np.tile([0.9, 0.05, 0.05], (4, 1))}
        weight = {"x": np.zeros(4)}
        np.testing.assert_allclose(raw_evidence_pool(prob, weight), np.full((4, 3), 1.0 / 3.0))

    def test_raw_pool_beats_the_flat_prior_dilution(self):
        prob = {"x": np.tile([0.9, 0.05, 0.05], (4, 1))}
        weight = {"x": np.ones(4)}
        self.assertGreater(raw_evidence_pool(prob, weight)[:, 0].min(), evidence_probability(prob, weight)[:, 0].min())


class BallLostMaskTests(unittest.TestCase):
    """A consensus with no active voter for a second is a ball the system lost."""

    def test_sustained_zero_confidence_marks_the_whole_run(self):
        from possession_lab.fusion.transitions import ball_lost_mask

        conf = np.ones(200)
        conf[100:130] = 0.0
        m = ball_lost_mask(conf, min_frames=25)
        self.assertTrue(bool(m[100:130].all()))
        self.assertFalse(bool(m[:100].any() or m[130:].any()))

    def test_short_blip_is_not_lost(self):
        from possession_lab.fusion.transitions import ball_lost_mask

        conf = np.ones(200)
        conf[100:115] = 0.0
        self.assertFalse(bool(ball_lost_mask(conf, min_frames=25).any()))


class OutOfBoundsMaskTests(unittest.TestCase):
    """A ball that stays past the touchline is out of play, tag or no tag."""

    @staticmethod
    def _ball(n=200):
        y = np.full(n, 30.0)
        y[100:135] = -4.0  # over the side line for 35 frames
        return np.stack([np.full(n, 50.0), y], axis=1)

    def test_outside_run_marks_dead(self):
        from possession_lab.fusion.transitions import out_of_bounds_mask

        m = out_of_bounds_mask(self._ball(), (105.0, 68.0), margin_m=0.8, min_frames=25)
        self.assertTrue(bool(m[100:135].all()))
        self.assertFalse(bool(m[:100].any() or m[135:].any()))

    def test_ball_on_the_line_stays_live(self):
        from possession_lab.fusion.transitions import out_of_bounds_mask

        ball = self._ball()
        ball[:, 1] = 68.4  # half over the chalk, inside the margin
        m = out_of_bounds_mask(ball, (105.0, 68.0), margin_m=0.8, min_frames=25)
        self.assertFalse(bool(m.any()))

    def test_short_excursion_is_live(self):
        from possession_lab.fusion.transitions import out_of_bounds_mask

        ball = self._ball()
        ball[100:112, 1] = -4.0
        ball[112:135, 1] = 20.0
        m = out_of_bounds_mask(ball, (105.0, 68.0), margin_m=0.8, min_frames=25)
        self.assertFalse(bool(m.any()))

    def test_nan_breaks_the_run(self):
        from possession_lab.fusion.transitions import out_of_bounds_mask

        ball = self._ball()
        ball[115:120, 1] = np.nan
        m = out_of_bounds_mask(ball, (105.0, 68.0), margin_m=0.8, min_frames=25)
        self.assertFalse(bool(m.any()))


class PitchContradictionGateTests(unittest.TestCase):
    """Box overlap at depth must not claim a ball a player is standing on.

    Measured 136 891: the ball pixel sat inside Barcelona players' boxes for four
    frames while an Athletic player was 0.4 m from the ball and his side led the
    raw pool -1 (the pixel side 4.5 m away); the contact punctured a settled H
    spell.  The pitch distance between the nearest players of each side to the
    ball decides the tie the pixel geometry cannot.
    """

    @staticmethod
    def world(n=220):
        codes = np.where(np.arange(n) < 108, H, A).astype(np.int8)  # boundary near the contact
        prob = np.zeros((n, 3))
        prob[:, 1] = 1.0  # the fused poll catches up on A: the momentum gate is open
        contain = np.zeros(n, dtype=np.int8)
        contain[96:101] = 2  # A boxes contain the ball pixel: run of 5, ends at frame 100
        contact = [(100, 5.0, 1, 1.0)]
        dead = np.zeros(n, dtype=bool)
        return codes, prob, contain, contact, dead

    def test_contact_dropped_when_the_other_side_is_on_the_ball(self):
        codes, prob, contain, contact, dead = self.world()
        pitch_close = np.full((len(codes), 2), 3.0)
        pitch_close[100] = [0.5, 4.6]  # H on the ball, the containing side 4 m away
        kept, ivs = gate_pixel_events(contact, [], contain, codes, prob, dead, [], pitch_close=pitch_close)
        self.assertEqual(kept, [])
        self.assertEqual(ivs, [])

    def test_contact_kept_when_its_side_is_closer(self):
        codes, prob, contain, contact, dead = self.world()
        pitch_close = np.full((len(codes), 2), 3.0)
        pitch_close[100] = [4.6, 0.5]
        kept, ivs = gate_pixel_events(contact, [], contain, codes, prob, dead, [], pitch_close=pitch_close)
        self.assertEqual(len(kept), 1)
        self.assertTrue(ivs)

    def test_no_pitch_data_no_veto(self):
        codes, prob, contain, contact, dead = self.world()
        kept, _ = gate_pixel_events(contact, [], contain, codes, prob, dead, [])
        self.assertEqual(len(kept), 1)


class RelabelTeamsFromTrackingTests(unittest.TestCase):
    """The tracking file carries each slot's roster team — the authority the jersey
    classifier loses when the light fails (measured 155 945: every visible player
    labelled Barcelona, eight more unknown)."""

    def test_matched_detections_adopt_the_roster_team(self):
        from possession_lab.perception import relabel_teams_from_tracking

        pitch = np.array([[[10.0, 20.0], [50.0, 10.0], [70.0, 40.0]]])
        present = np.array([[True, True, True]])
        teams = np.array([[0, 0, 0]], dtype=np.int8)
        track_xy = np.array([[[10.2, 20.1], [49.5, 10.5], [30.0, 30.0]]])
        track_teams = np.array([1, 0, 1], dtype=np.int8)
        track_present = np.array([[True, True, False]])
        out = relabel_teams_from_tracking(
            pitch, present, teams, track_xy, track_teams, track_present, (105.0, 68.0)
        )
        self.assertEqual(out[0].tolist(), [1, 0, 0])  # slot 3 not present: raw label stands

    def test_off_pitch_ghost_slot_is_never_a_match(self):
        from possession_lab.perception import relabel_teams_from_tracking

        pitch = np.array([[[10.0, 70.9]]])
        present = np.array([[True]])
        teams = np.array([[0]], dtype=np.int8)
        track_xy = np.array([[[10.0, 71.0]]])
        track_teams = np.array([1], dtype=np.int8)
        track_present = np.array([[True]])
        out = relabel_teams_from_tracking(
            pitch, present, teams, track_xy, track_teams, track_present, (105.0, 68.0)
        )
        self.assertEqual(out[0].tolist(), [0])

    def test_far_detection_keeps_its_own_label(self):
        from possession_lab.perception import relabel_teams_from_tracking

        pitch = np.array([[[40.0, 20.0]]])
        present = np.array([[True]])
        teams = np.array([[-1]], dtype=np.int8)
        track_xy = np.array([[[45.0, 20.0]]])
        track_teams = np.array([1], dtype=np.int8)
        track_present = np.array([[True]])
        out = relabel_teams_from_tracking(
            pitch, present, teams, track_xy, track_teams, track_present, (105.0, 68.0)
        )
        self.assertEqual(out[0].tolist(), [-1])


class DeadRevivalMaskTests(unittest.TestCase):
    """A falsely locked D reopens only to a live tag the voters back."""

    @staticmethod
    def scene(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        ball = np.zeros((n, 2), dtype=np.float64)
        ball[:, 0] = np.arange(n) * (2.0 / 25.0)  # 2 m/s along x, on the grass
        ball[:, 1] = 10.0
        pool = np.tile([0.05, 0.75, 0.02], (n, 1))  # the pool backs A by 0.70
        return ball, pool

    def run_case(self, tags, blocked=None, **kw):
        from possession_lab.fusion.transitions import dead_revival_mask

        ball, pool = self.scene()
        blocked = np.zeros(200, dtype=bool) if blocked is None else blocked
        return dead_revival_mask(ball, pool, tags, (105.0, 68.0), blocked, **kw)

    def test_tagged_live_run_revives_with_the_tagged_team(self):
        mask, team = self.run_case([(50, 149, 1, True)])
        self.assertTrue(mask[50:150].all())
        self.assertFalse(mask[:50].any() or mask[150:].any())
        self.assertEqual(int(team[60]), 1)

    def test_short_run_does_not_revive(self):
        mask, _ = self.run_case([(50, 60, 1, True)], min_frames=25)
        self.assertFalse(mask.any())

    def test_stationary_ball_does_not_revive(self):
        from possession_lab.fusion.transitions import dead_revival_mask

        ball, pool = self.scene()
        ball[:, 0] = 50.0  # parked: play is not live
        mask, _ = dead_revival_mask(ball, pool, [(50, 149, 1, True)], (105.0, 68.0), np.zeros(200, bool))
        self.assertFalse(mask.any())

    def test_blocked_frames_never_revive(self):
        blocked = np.zeros(200, dtype=bool)
        blocked[50:150] = True  # a real setpiece covers the window
        mask, _ = self.run_case([(50, 149, 1, True)], blocked=blocked)
        self.assertFalse(mask.any())

    def test_pool_against_the_tag_never_revives(self):
        from possession_lab.fusion.transitions import dead_revival_mask

        ball, _ = self.scene()
        pool = np.tile([0.75, 0.05, 0.02], (200, 1))  # voters back H, tag says A
        mask, _ = dead_revival_mask(ball, pool, [(50, 149, 1, True)], (105.0, 68.0), np.zeros(200, bool))
        self.assertFalse(mask.any())

    def test_off_pitch_ball_does_not_revive(self):
        from possession_lab.fusion.transitions import dead_revival_mask

        ball, pool = self.scene()
        ball[:, 1] = 75.0  # 7 m past the touchline: this one is genuinely out
        mask, _ = dead_revival_mask(ball, pool, [(50, 149, 1, True)], (105.0, 68.0), np.zeros(200, bool))
        self.assertFalse(mask.any())

    def test_weak_tag_without_crossing_does_not_revive(self):
        # a celebration: zone tag over a parked-the-game scene, ball never left the grass
        from possession_lab.fusion.transitions import dead_revival_mask

        ball, pool = self.scene()
        pool = np.tile([0.46, 0.52, 0.02], (200, 1))  # winner backs the tag, but no gap
        mask, _ = dead_revival_mask(
            ball, pool, [(50, 149, 1, False)], (105.0, 68.0), np.zeros(200, bool),
            raw_ball=ball.copy(),
        )
        self.assertFalse(mask.any())

    def test_weak_tag_revives_a_marginal_chalk_crossing(self):
        # measured 116 791: consensus grazes 0.1 m over the chalk, pool gap is tiny,
        # a zone tag names the winner — the out never happened
        from possession_lab.fusion.transitions import dead_revival_mask

        ball, pool = self.scene()
        ball[:, 1] = 67.95  # consensus grazes the touchline but stays inside the pad
        pool = np.tile([0.46, 0.52, 0.02], (200, 1))  # winner A by 0.06 only
        raw_ball = ball.copy()
        raw_ball[:, 1] = 68.1  # raw ball 0.1 m past the 68.0 chalk: a calibration graze
        mask, team = dead_revival_mask(
            ball, pool, [(50, 149, 1, False)], (105.0, 68.0), np.zeros(200, bool),
            raw_ball=raw_ball,
        )
        self.assertTrue(mask[50:150].all())
        self.assertEqual(int(team[60]), 1)

    def test_weak_tag_not_guaranteed_for_full_gap_needs_no_crossing_never_opens(self):
        # strong tags still demand the gap: no crossing, thin gap -> stays locked
        from possession_lab.fusion.transitions import dead_revival_mask

        ball, _ = self.scene()
        pool = np.tile([0.30, 0.40, 0.02], (200, 1))  # A wins by 0.10 < gap_min
        mask, _ = dead_revival_mask(
            ball, pool, [(50, 149, 1, True)], (105.0, 68.0), np.zeros(200, bool),
            raw_ball=ball.copy(),
        )
        self.assertFalse(mask.any())


class PossessionRevivalMaskTests(unittest.TestCase):
    """A dead lock reopens when a player is simply playing the ball."""

    @staticmethod
    def scene(n: int = 200):
        ball = np.zeros((n, 2), dtype=np.float64)
        ball[:, 0] = np.arange(n) * (2.0 / 25.0)  # 2 m/s along x, on the grass
        ball[:, 1] = 10.0
        players = np.full((n, 2, 2), np.nan, dtype=np.float64)
        players[:, 0] = ball + 0.5  # an A player at the ball all along
        players[:, 1] = [100.0, 40.0]  # an H player far away
        present = np.ones((n, 2), dtype=bool)
        teams = np.array([1, 0], dtype=np.int8)
        pool = np.tile([0.05, 0.75, 0.02], (n, 1))  # the pool backs A by 0.70
        return ball, players, present, teams, pool

    def run_case(self, **kw):
        from possession_lab.fusion.transitions import possession_revival_mask

        ball, players, present, teams, pool = self.scene()
        return possession_revival_mask(
            ball, players, present, teams, pool, (105.0, 68.0), np.zeros(len(ball), bool), **kw
        )

    def test_player_at_the_ball_reopens_the_lock(self):
        mask, team = self.run_case()
        # the coverage kernel normalises by the frames actually present, so glue
        # that holds from the first second owns its edges
        self.assertTrue(mask.all())
        self.assertTrue((team == 1).all())

    def test_pool_against_the_player_never_reopens(self):
        from possession_lab.fusion.transitions import possession_revival_mask

        ball, players, present, teams, _ = self.scene()
        pool = np.tile([0.75, 0.05, 0.02], (len(ball), 1))  # pool backs H, player is A
        mask, _ = possession_revival_mask(
            ball, players, present, teams, pool, (105.0, 68.0), np.zeros(len(ball), bool)
        )
        self.assertFalse(mask.any())

    def test_stationary_ball_never_reopens(self):
        from possession_lab.fusion.transitions import possession_revival_mask

        ball, players, present, teams, pool = self.scene()
        ball[:, 0] = 50.0  # parked: not play
        players[:, 0, 0] = 50.5
        mask, _ = possession_revival_mask(
            ball, players, present, teams, pool, (105.0, 68.0), np.zeros(len(ball), bool)
        )
        self.assertFalse(mask.any())

    def test_blocked_frames_never_reopen(self):
        from possession_lab.fusion.transitions import possession_revival_mask

        ball, players, present, teams, pool = self.scene()
        blocked = np.zeros(len(ball), dtype=bool)
        blocked[:] = True  # a real setpiece covers the window
        mask, _ = possession_revival_mask(
            ball, players, present, teams, pool, (105.0, 68.0), blocked
        )
        self.assertFalse(mask.any())

    def test_no_player_near_the_ball_never_reopens(self):
        from possession_lab.fusion.transitions import possession_revival_mask

        ball, players, present, teams, pool = self.scene()
        players[:, 0] = [100.0, 40.0]  # nobody at the ball
        mask, _ = possession_revival_mask(
            ball, players, present, teams, pool, (105.0, 68.0), np.zeros(len(ball), bool)
        )
        self.assertFalse(mask.any())

    def test_short_glue_run_does_not_reopen(self):
        from possession_lab.fusion.transitions import possession_revival_mask

        ball, players, present, teams, pool = self.scene()
        players[:170, 0] = [100.0, 40.0]  # the glue lasts barely 30 frames
        mask, _ = possession_revival_mask(
            ball, players, present, teams, pool, (105.0, 68.0), np.zeros(len(ball), bool)
        )
        self.assertFalse(mask.any())


class RestartOwnershipTests(unittest.TestCase):
    """A smart-tagged restart paints only when the ball's own streams witness it
    and a player stands at the ball; it may only fill D/X gaps."""

    PITCH = (105.0, 68.0)
    N = 100

    def rule(self, codes, consensus, speed, yconf, players, oop, windows):
        return restart_ownership(
            codes, consensus, speed, yconf, players, oop, windows, self.PITCH
        )

    def standing(self, h_xy, a_xy):
        """One H player at ``h_xy``, one A player at ``a_xy``, present everywhere."""
        pitch = np.zeros((self.N, 2, 2))
        pitch[:, 0] = h_xy
        pitch[:, 1] = a_xy
        return (
            pitch,
            np.tile(np.array([H, A], dtype=np.int8), (self.N, 1)),
            np.ones((self.N, 2), dtype=bool),
        )

    def thrown_ball(self, cross=50):
        """The ball rides past the touchline until ``cross``, then is back in bounds."""
        ball = np.full((self.N, 2), 50.0)
        ball[:, 1] = 70.0  # two metres out
        ball[cross:, 1] = 67.0  # in bounds from the crossing
        return ball

    def throw_scene(self, cross=50, h_xy=(50.0, 68.0), a_xy=(70.0, 68.0), speed_peak=7.0):
        """A throw-in: crossing at ``cross``, the ball at throw speed just after."""
        speed = np.zeros(self.N)
        if speed_peak > 0:
            speed[cross : cross + 6] = speed_peak
        players = self.standing(h_xy, a_xy)
        return self.thrown_ball(cross), speed, players

    def test_witnessed_throw_in_gives_play_to_the_thrower(self):
        codes = np.full(self.N, D, dtype=np.int8)
        ball, speed, players = self.throw_scene()
        oop = np.zeros(self.N, dtype=bool)
        out, events, painted = self.rule(codes, ball, speed, None, players, oop, [(30, 80, "Throw-in")])
        self.assertTrue((out[30:50] == D).all())
        self.assertTrue((out[50:81] == H).all())  # the thrower's team owns play from the crossing
        self.assertEqual(painted, 31)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "restart")
        self.assertEqual(events[0].frame, 50)
        self.assertEqual(events[0].team, H)
        self.assertAlmostEqual(events[0].distance, 2.0)

    def test_crossing_below_throw_speed_is_not_a_throw(self):
        codes = np.full(self.N, D, dtype=np.int8)
        ball, speed, players = self.throw_scene(speed_peak=0.0)  # a slow drift back in
        out, events, painted = self.rule(codes, ball, speed, None, players, np.zeros(self.N, bool), [(30, 80, "Throw-in")])
        self.assertEqual(painted, 0)
        self.assertEqual(events, ())

    def test_contested_throw_is_not_attributed(self):
        codes = np.full(self.N, D, dtype=np.int8)
        # H 2.1 m from the ball, A 2.24 m: nobody a clear metre ahead
        ball, speed, players = self.throw_scene(h_xy=(50.0, 67.9), a_xy=(51.0, 68.0))
        out, events, painted = self.rule(codes, ball, speed, None, players, np.zeros(self.N, bool), [(30, 80, "Throw-in")])
        self.assertEqual(painted, 0)
        self.assertEqual(events, ())
        self.assertTrue((out == D).all())

    def test_no_thrower_in_reach_is_not_a_restart(self):
        codes = np.full(self.N, D, dtype=np.int8)
        ball, speed, players = self.throw_scene(h_xy=(50.0, 60.0), a_xy=(70.0, 68.0))  # H 8 m away
        out, events, painted = self.rule(codes, ball, speed, None, players, np.zeros(self.N, bool), [(30, 80, "Throw-in")])
        self.assertEqual(painted, 0)
        self.assertEqual(events, ())

    def test_witnessed_kickoff_deadens_the_pre_kick_frames(self):
        codes = np.full(self.N, D, dtype=np.int8)
        codes[45:50] = H  # the decode's memory of the last spell, before the kick
        ball = np.full((self.N, 2), 50.0)
        ball[:, 1] = 30.0
        yconf = np.full(self.N, 0.3)
        yconf[50:53] = 0.6  # the detector locks the kicked ball for three frames
        speed = np.zeros(self.N)
        players = self.standing((50.0, 30.0), (55.0, 30.0))
        out, events, painted = self.rule(
            codes, ball, speed, yconf, players, np.zeros(self.N, bool),
            [(30, 80, "Kick Off (Start)")],
        )
        self.assertTrue((out[45:50] == D).all())  # pre-kick memory is dead
        self.assertTrue((out[50:81] == H).all())  # the kicker's team from the lock
        self.assertEqual(painted, 5 + 31)
        self.assertEqual(events[0].frame, 50)
        self.assertEqual(events[0].team, H)
        self.assertAlmostEqual(events[0].distance, 0.0)

    def test_unlocked_kickoff_is_not_a_restart(self):
        codes = np.full(self.N, D, dtype=np.int8)
        codes[45:50] = H
        ball = np.full((self.N, 2), 50.0)
        ball[:, 1] = 30.0
        yconf = np.full(self.N, 0.4)  # the detector never locks
        players = self.standing((50.0, 30.0), (55.0, 30.0))
        out, events, painted = self.rule(
            codes, ball, np.zeros(self.N), yconf, players, np.zeros(self.N, bool),
            [(30, 80, "Kick Off (Start)")],
        )
        self.assertEqual(painted, 0)
        self.assertEqual(events, ())
        self.assertTrue((out[45:50] == H).all())  # untouched: no witnessed restart

    def test_fill_never_overpaints_a_decoded_possession(self):
        codes = np.full(self.N, D, dtype=np.int8)
        codes[60:80] = A  # the decode already moved play to Athletic
        ball, speed, players = self.throw_scene()
        out, events, painted = self.rule(codes, ball, speed, None, players, np.zeros(self.N, bool), [(30, 80, "Throw-in")])
        self.assertTrue((out[50:60] == H).all())
        self.assertTrue((out[60:80] == A).all())  # the decode's change stands
        self.assertEqual(painted, 10)

    def test_fill_stops_where_the_ball_is_lost(self):
        codes = np.full(self.N, D, dtype=np.int8)
        ball, speed, players = self.throw_scene()
        oop = np.zeros(self.N, dtype=bool)
        oop[65] = True  # the camera loses the ball again
        out, events, painted = self.rule(codes, ball, speed, None, players, oop, [(30, 80, "Throw-in")])
        self.assertTrue((out[50:65] == H).all())
        self.assertEqual(out[65], D)
        self.assertEqual(painted, 15)

    def test_a_short_fill_is_not_painted_but_the_restart_is_witnessed(self):
        codes = np.full(self.N, D, dtype=np.int8)
        ball, speed, players = self.throw_scene()
        out, events, painted = self.rule(codes, ball, speed, None, players, np.zeros(self.N, bool), [(30, 51, "Throw-in")])
        self.assertEqual(painted, 0)  # a two-frame spell is not paintable
        self.assertEqual(len(events), 1)  # the restart itself is still logged
        self.assertTrue((out == D).all())

    def test_absent_players_are_not_attributed(self):
        codes = np.full(self.N, D, dtype=np.int8)
        ball, speed, players = self.throw_scene()
        present = np.zeros((self.N, 2), dtype=bool)
        out, events, painted = self.rule(codes, ball, speed, None, (players[0], players[1], present), np.zeros(self.N, bool), [(30, 80, "Throw-in")])
        self.assertEqual(painted, 0)
        self.assertEqual(events, ())

    def test_restart_kind_code_is_eight(self):
        row = TransitionEvent(start=50, frame=50, kind="restart", team=H, distance=2.0, support=0.0).as_row()
        self.assertEqual(row[2], 8)


if __name__ == "__main__":
    unittest.main()
