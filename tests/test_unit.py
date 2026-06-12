import unittest

import pandas as pd

from src.ai.commentary import verify_numbers
from src.features.feature_engineering import impute_feature_gaps
from src.models.baseline import compute_metrics, seasonal_naive_predict


class UnitTests(unittest.TestCase):
    def test_impute_feature_gaps_limit(self):
        df = pd.DataFrame({"x": [1.0, None, None, None, 5.0]})
        out = impute_feature_gaps(df.copy(), "x", max_fill=2)
        self.assertEqual(out.loc[1, "x"], 1.0)
        self.assertEqual(out.loc[2, "x"], 1.0)
        self.assertTrue(pd.isna(out.loc[3, "x"]))

    def test_seasonal_naive_predict_shift(self):
        idx = pd.date_range("2023-01-01", periods=200, freq="h", tz="UTC")
        df = pd.DataFrame({"target": range(200)}, index=idx)
        pred = seasonal_naive_predict(df)
        self.assertTrue(pred.iloc[:168].isna().all())
        self.assertEqual(pred.iloc[168], 0)
        self.assertEqual(pred.iloc[199], 31)

    def test_compute_metrics_nonempty(self):
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0])
        y_pred = pd.Series([12.0, 18.0, 33.0, 39.0])
        m = compute_metrics(y_true, y_pred, label="x")
        self.assertEqual(m["n_samples"], 4)
        self.assertAlmostEqual(m["mae"], 2.0, places=4)
        self.assertIn("rmse", m)

    def test_verify_numbers(self):
        context = {"a": 10.5, "b": 2.0}
        good = "Forecast is 10.5 and spread is 2.0."
        bad = "Forecast is 11.2 and spread is 2.0."
        self.assertTrue(verify_numbers(good, context, tol=0.05))
        self.assertFalse(verify_numbers(bad, context, tol=0.05))


if __name__ == "__main__":
    unittest.main()
