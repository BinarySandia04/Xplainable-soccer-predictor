"""Tests for the command-line entry points."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from possession_lab.cli import cmd_evaluate, cmd_visualize


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


if __name__ == "__main__":
    unittest.main()
