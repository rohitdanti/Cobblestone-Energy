from __future__ import annotations

import subprocess
from pathlib import Path


_ENSURED = False


REQUIRED_OUTPUTS = [
    Path("outputs/qa_report.md"),
    Path("outputs/qa_summary.json"),
    Path("outputs/metrics.json"),
    Path("outputs/submission.csv"),
    Path("outputs/curve_view.json"),
    Path("outputs/daily_commentary.md"),
    Path("outputs/figures/forecast_vs_actual.png"),
    Path("outputs/figures/feature_importance.png"),
]


def ensure_pipeline_outputs() -> None:
    global _ENSURED
    if _ENSURED and all(p.exists() and p.stat().st_size > 0 for p in REQUIRED_OUTPUTS):
        return

    result = subprocess.run(
        ["./.venv/bin/python", "run_pipeline.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Pipeline failed during test setup.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    missing = [str(p) for p in REQUIRED_OUTPUTS if not (p.exists() and p.stat().st_size > 0)]
    if missing:
        raise RuntimeError(f"Pipeline completed but required outputs missing: {missing}")

    _ENSURED = True
