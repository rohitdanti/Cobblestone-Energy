import subprocess
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import ensure_pipeline_outputs


class FlowAndE2ETests(unittest.TestCase):
    def test_end_to_end_cli_run(self):
        result = subprocess.run(
            ["./.venv/bin/python", "run_pipeline.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def test_submission_shape_and_columns(self):
        ensure_pipeline_outputs()
        sub = pd.read_csv("outputs/submission.csv")
        self.assertEqual(list(sub.columns), ["id", "y_pred"])
        self.assertGreaterEqual(len(sub), 6000)

    def test_required_output_files_exist(self):
        ensure_pipeline_outputs()
        required = [
            "outputs/qa_report.md",
            "outputs/qa_summary.json",
            "outputs/metrics.json",
            "outputs/submission.csv",
            "outputs/curve_view.json",
            "outputs/daily_commentary.md",
            "outputs/figures/forecast_vs_actual.png",
            "outputs/figures/feature_importance.png",
        ]
        for path in required:
            p = Path(path)
            self.assertTrue(p.exists(), msg=f"Missing: {path}")
            self.assertGreater(p.stat().st_size, 0, msg=f"Empty: {path}")


if __name__ == "__main__":
    unittest.main()
