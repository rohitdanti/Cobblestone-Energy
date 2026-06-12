import json
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import ensure_pipeline_outputs


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_pipeline_outputs()

    def test_feature_artifact_schema(self):
        df = pd.read_parquet("processed/features.parquet")
        required = {
            "target",
            "load_forecast_mw",
            "wind_forecast_mw",
            "solar_forecast_mw",
            "residual_load_mw",
            "da_price_lag_24h",
            "da_price_lag_168h",
        }
        self.assertTrue(required.issubset(df.columns))
        self.assertGreater(len(df), 1000)

    def test_model_and_metrics_artifacts(self):
        self.assertTrue(Path("models/extra_trees_model.pkl").exists())
        self.assertTrue(Path("models/residual_bands.pkl").exists())

        metrics = json.loads(Path("outputs/metrics.json").read_text())
        self.assertIn("naive_validation", metrics)
        self.assertIn("extra_trees_validation", metrics)
        self.assertIn("naive_holdout", metrics)
        self.assertIn("extra_trees_holdout", metrics)
        self.assertIn("rolling_validation_summary", metrics)

    def test_curve_view_contract(self):
        curve = json.loads(Path("outputs/curve_view.json").read_text())
        self.assertEqual(curve["production"]["output_type"], "PRODUCTION")
        self.assertEqual(curve["scenario"]["output_type"], "SCENARIO_FAIR_VALUE")
        self.assertIn("invalidation_flags", curve)


if __name__ == "__main__":
    unittest.main()
