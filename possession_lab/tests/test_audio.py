"""Synthetic-audio tests for the whistle extractor.

These never touch the gated MP4: they exercise the spectral score and the event builder with a
tone that has the same band shape as a referee whistle.
"""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.audio import (
    band_score,
    detect_events,
    envelope,
    score_per_video_frame,
)
from possession_lab.config import AUDIO_SR, FPS


def tone_audio(seconds: float = 2.0, frequency: float = 3_000.0) -> np.ndarray:
    t = np.arange(int(seconds * AUDIO_SR), dtype=np.float32) / AUDIO_SR
    return (0.05 * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


class AudioScoreTest(unittest.TestCase):
    def test_whistle_band_score_peaks_on_a_tonal_whistle(self):
        audio = tone_audio()
        score = band_score(audio, AUDIO_SR)
        self.assertGreater(score.size, 0)
        self.assertGreater(float(np.median(score)), 0.0)

        silence = np.zeros(audio.size, dtype=np.float32)
        quiet = band_score(silence, AUDIO_SR)
        self.assertLess(float(np.max(quiet)), 1e-6)

    def test_score_is_aligned_to_video_frames(self):
        audio = tone_audio()
        score = score_per_video_frame(audio, n_frames=50)
        self.assertEqual(score.shape, (50,))
        self.assertGreater(float(score.max()), 0.0)
        self.assertTrue(np.all(score >= 0.0))

    def test_short_high_score_blip_becomes_one_event(self):
        n = 200
        score = np.zeros(n, dtype=np.float32)
        score[40:48] = 6.0
        events = detect_events(score, n)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["frame"], 40)
        self.assertEqual(events[0]["start"], 40)
        self.assertEqual(events[0]["stop"], 48)
        self.assertAlmostEqual(events[0]["width_s"], 8 / FPS)

    def test_close_blips_merge_and_distant_blips_stay_separate(self):
        n = 400
        score = np.zeros(n, dtype=np.float32)
        score[40:44] = 6.0
        score[60:64] = 6.0  # 0.8 s later, inside the 1.0 s merge gap
        score[300:304] = 6.0
        events = detect_events(score, n)
        self.assertEqual(len(events), 2)
        self.assertLess(events[0]["stop"], 100)
        self.assertGreater(events[1]["frame"], 250)

    def test_envelope_peaks_at_the_event_and_decays_afterward(self):
        n = 250
        event = {"start": 100, "stop": 110, "frame": 105, "peak": 8.0, "width_s": 0.4}
        env = envelope([event], n)
        self.assertEqual(env.shape, (n,))
        self.assertGreater(float(env[105]), 0.9)
        self.assertGreater(float(env[120]), 0.0)
        self.assertLess(float(env[200]), float(env[120]))
        self.assertLess(float(env[0]), 0.2)


if __name__ == "__main__":
    unittest.main()
