"""Tests for the command-line entry points."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from possession_lab.cli import cmd_evaluate, cmd_visualize, pick_setpiece_release


class MissingInputsTest(unittest.TestCase):
    """A fresh checkout has no ``outputs/`` yet, and a missing file must not traceback."""

    def run_command(self, command, output_dir=None, **args):
        errors = io.StringIO()
        with TemporaryDirectory() as tmp, redirect_stderr(errors):
            status = command(
                SimpleNamespace(output_dir=output_dir or Path(tmp), bootstrap_n=0, **args)
            )
        return status, errors.getvalue()

    def test_evaluate_reports_a_missing_predictions_file(self):
        status, message = self.run_command(cmd_evaluate)
        self.assertEqual(status, 1)
        self.assertIn("predictions.csv", message)
        self.assertIn("predict", message)

    def test_visualize_reports_both_missing_inputs(self):
        status, message = self.run_command(cmd_visualize)
        self.assertEqual(status, 1)
        self.assertIn("predictions.csv", message)
        self.assertIn("metrics_public.json", message)

    def test_visualize_names_only_the_file_that_is_actually_missing(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "predictions.csv").write_text("frame,label\n0,H\n", encoding="utf-8")
            status, message = self.run_command(cmd_visualize, output_dir=Path(tmp))
        self.assertEqual(status, 1)
        self.assertNotIn("predictions.csv", message)
        self.assertIn("metrics_public.json", message)


class PickSetpieceReleaseTests(unittest.TestCase):
    """The release frame is a ball in flight, not a tracker jump.

    The 65 558 throw-in tag carried the ball out of bounds, jumped 248 m back to
    the touchline (frame 65 727), and only threw it properly at 66 026.  The
    248 m jump is a valid speed spike on the line, so the continuity guard must
    prefer the later *continuous* release over it.
    """

    PITCH_WIDTH = 68.0
    PITCH_LENGTH = 105.0
    FPS = 25.0

    def _arrays(self, ball):
        """Build speed/jump/near/touchline_m exactly as ``cmd_transitions`` does."""
        n = len(ball)
        diff = np.zeros_like(ball)
        diff[1:] = (ball[1:] - ball[:-1]) * self.FPS
        speed = np.linalg.norm(diff, axis=1)
        jump = np.zeros(n)
        jump[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=1)
        margin = 2.0
        near = (
            (ball[:, 0] >= -margin)
            & (ball[:, 0] <= self.PITCH_LENGTH + margin)
            & (ball[:, 1] >= -margin)
            & (ball[:, 1] <= self.PITCH_WIDTH + margin)
        )
        touchline_m = np.minimum(np.abs(ball[:, 1]), np.abs(ball[:, 1] - self.PITCH_WIDTH))
        return speed, jump, near, touchline_m

    def test_continuous_release_wins_over_earlier_tracker_jump(self):
        n = 120
        ball = np.tile((10.0, -10.0), (n, 1))          # carried out of bounds
        for f in range(30, 41):                         # 248 m jump back to the line
            ball[f] = (10.0, 0.3 + (f - 30) * 0.15)
        ball[41:60] = (10.0, 1.8)                       # paused
        for f in range(60, 71):                         # the true continuous throw
            ball[f] = (10.0, 2.2 + (f - 60) * 0.4)
        speed, jump, near, touchline_m = self._arrays(ball)

        restart_f, strict = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, True, self.PITCH_WIDTH
        )
        # frame 30 (the jump) is skipped; the continuous throw at 60 is taken
        self.assertEqual(restart_f, 60)
        self.assertTrue(strict)

    def test_first_strict_pick_kept_when_no_continuous_release(self):
        n = 120
        ball = np.tile((10.0, -10.0), (n, 1))
        for f in range(30, 41):
            ball[f] = (10.0, 0.3 + (f - 30) * 0.15)
        ball[41:] = (10.0, 1.8)                         # never a clean continuous throw
        speed, jump, near, touchline_m = self._arrays(ball)

        restart_f, strict = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, True, self.PITCH_WIDTH
        )
        # the jump is the only strict candidate: keep it, do not degrade to the
        # first-spike fallback (which would report strict=False)
        self.assertEqual(restart_f, 30)
        self.assertTrue(strict)

    def test_plain_continuous_release_is_unchanged(self):
        n = 120
        ball = np.tile((50.0, 34.0), (n, 1))
        for f in range(60, 71):
            ball[f] = (50.0, 34.0 + (f - 59) * 0.4)   # moving from frame 60
        speed, jump, near, touchline_m = self._arrays(ball)

        restart_f, strict = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, False, self.PITCH_WIDTH
        )
        self.assertEqual(restart_f, 60)
        self.assertTrue(strict)

    def test_no_clean_release_falls_back_to_first_spike(self):
        n = 120
        ball = np.tile((10.0, 40.0), (n, 1))
        ball[50:] = (10.0, 80.0)                        # spike off the pitch: not a release
        speed, jump, near, touchline_m = self._arrays(ball)

        restart_f, strict = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, False, self.PITCH_WIDTH
        )
        self.assertEqual(restart_f, 50)
        self.assertFalse(strict)

    def test_line_spike_without_inward_motion_is_not_a_release(self):
        n = 120
        ball = np.tile((10.0, -10.0), (n, 1))
        for f in range(30, 41):
            ball[f] = (10.0, 0.3)                       # at the line but never sent in
        speed, jump, near, touchline_m = self._arrays(ball)

        restart_f, strict = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, True, self.PITCH_WIDTH
        )
        # the throw-in line test rejects it, so this degrades to the spike fallback
        self.assertEqual(restart_f, 30)
        self.assertFalse(strict)

    def test_carry_across_the_line_without_a_player_at_the_ball_is_not_a_release(self):
        # the tracker carries the ball around behind the touchline to the throw
        # spot at running speed and crosses the line with nobody at the ball;
        # the real throw comes later, with the thrower at the ball (measured
        # 164 964 carry at 13.5 m/s, thrower 3.4 m away; throw 164 993 at 0.4 m)
        n = 200
        ball = np.tile((15.0, -5.0), (n, 1))
        for f in range(30, 49):                          # the carry, crossing the line
            ball[f] = (15.0 + (f - 30) * 0.2, -5.0 + (f - 30) * 0.35)
        ball[49:90] = (19.0, 1.2)                        # parked at the throw spot
        for f in range(90, 121):                         # the true throw, inward for >10 frames
            ball[f] = (19.0, 1.2 + (f - 90) * 0.4)
        speed, jump, near, touchline_m = self._arrays(ball)
        claim = np.full(n, -1)
        claim[90:121] = 0                                # thrower at the ball, only at the throw

        r0, s0 = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, True, self.PITCH_WIDTH
        )
        self.assertEqual(r0, 39)                          # without claims: the carry wins
        r1, s1 = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, True, self.PITCH_WIDTH, claim=claim
        )
        self.assertEqual(r1, 91)                          # with claims: the carry is skipped
        self.assertTrue(s1)

    def test_no_claimed_release_keeps_the_pre_gate_fallback(self):
        # nobody ever stands at the ball: the gate must not degrade the tag,
        # it must return exactly the pre-gate pick
        n = 120
        ball = np.tile((10.0, -10.0), (n, 1))
        for f in range(30, 41):
            ball[f] = (10.0, 0.3 + (f - 30) * 0.15)
        ball[41:] = (10.0, 1.8)
        speed, jump, near, touchline_m = self._arrays(ball)
        claim = np.full(n, -1)

        r0, s0 = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, True, self.PITCH_WIDTH
        )
        r1, s1 = pick_setpiece_release(
            speed, jump, near, touchline_m, ball, 0, n, True, self.PITCH_WIDTH, claim=claim
        )
        self.assertEqual((r0, s0), (r1, s1))
        self.assertTrue(s1)


if __name__ == "__main__":
    unittest.main()
