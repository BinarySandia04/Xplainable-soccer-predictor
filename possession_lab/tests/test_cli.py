"""Tests for the command-line entry points."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from possession_lab.cli import _default_perception_path, cmd_evaluate, cmd_visualize


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


class PerceptionCacheSelectionTest(unittest.TestCase):
    def test_prefers_full_match_cache_over_public_window(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for name, start, stop in (
                ("perception.npz", 11_185, 18_685),
                ("perception_new.npz", 0, 199_319),
                ("perception_public.npz", 11_185, 18_685),
            ):
                path = output_dir / name
                path.touch()
                path.with_suffix(path.suffix + ".json").write_text(
                    json.dumps({"start": start, "stop": stop, "n_frames": 199_319}),
                    encoding="utf-8",
                )

            self.assertEqual(
                _default_perception_path(output_dir),
                output_dir / "perception_new.npz",
            )

    def test_perceive_defaults_to_full_match_range(self):
        from possession_lab.cli import cmd_perceive

        args = SimpleNamespace(
            output_dir=Path("/tmp"),
            data_dir=None,
            start=None,
            stop=None,
            perception="/tmp/perception.npz",
            step=1,
            stage="both",
            model="yolo11x.pt",
            players_model=None,
            ball_model=None,
            players_imgsz=None,
            ball_imgsz=1280,
            device=None,
            imgsz=1920,
        )
        from possession_lab.config import N_FRAMES

        with (
            patch("possession_lab.pipeline.load_lab"),
            patch("possession_lab.perception.perceive_video") as perceive,
            patch(
                "possession_lab.perception.save_perception",
                return_value=Path("/tmp/perception.npz"),
            ),
        ):
            perceive.return_value.samples = N_FRAMES
            perceive.return_value.ball_source = np.zeros(N_FRAMES, dtype=np.int8)
            cmd_perceive(args)

        self.assertEqual(perceive.call_args.kwargs["start"], 0)
        self.assertEqual(perceive.call_args.kwargs["stop"], N_FRAMES)


if __name__ == "__main__":
    unittest.main()
