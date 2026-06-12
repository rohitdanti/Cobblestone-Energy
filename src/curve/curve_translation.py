"""
Convert holdout hourly forecasts into DA fair-value products and scenario rollups.
Writes outputs/curve_view.json.
"""
import json
import sys
from pathlib import Path

# Allow direct script execution: python src/curve/curve_translation.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml


def load_config() -> dict:
	return yaml.safe_load(Path("config/config.yaml").read_text())


def compute_da_products(forecast_df: pd.DataFrame) -> dict:
	"""DST-safe DA baseload, peak, off-peak, and peak-base spread."""
	col = "y_pred"
	local = forecast_df["delivery_start_local"]

	da_baseload = float(forecast_df.groupby("delivery_date_local")[col].mean().mean())
	peak_mask = local.dt.hour.between(9, 19)
	da_peak = float(forecast_df.loc[peak_mask, col].mean())
	da_offpeak = float(forecast_df.loc[~peak_mask, col].mean())
	spread = round(da_peak - da_baseload, 4)

	return {
		"da_baseload_eur_mwh": round(da_baseload, 4),
		"da_peak_eur_mwh": round(da_peak, 4),
		"da_offpeak_eur_mwh": round(da_offpeak, 4),
		"peak_base_spread_eur_mwh": spread,
	}


def compute_scenario_rollup(model, feature_template: pd.DataFrame, days: int = 7) -> float:
	"""Directional scenario fair value using repeated D+1 fundamental pattern."""
	from src.models.extra_trees_model import FEATURE_COLS

	last_row = feature_template.tail(24).copy()
	preds = []
	for d in range(days):
		row = last_row.copy()
		row["day_of_week"] = (row["day_of_week"] + d + 1) % 7
		row["day_sin"] = np.sin(2 * np.pi * row["day_of_week"] / 7)
		row["day_cos"] = np.cos(2 * np.pi * row["day_of_week"] / 7)
		row["is_weekend"] = (row["day_of_week"] >= 5).astype(int)
		day_preds = model.predict(row[FEATURE_COLS])
		preds.extend(day_preds.tolist())

	return round(float(np.mean(preds)), 4)


def compute_invalidation_flags(qa_summary: dict, p10: float, p90: float, cfg: dict) -> dict:
	threshold = cfg["curve_translation"]["confidence_band_threshold_eur_mwh"]
	qa_failed = not qa_summary.get("all_checks_passed", True)
	wide_confidence_band = (p90 - p10) > threshold
	manual_review = qa_failed or wide_confidence_band
	return {
		"qa_failed": qa_failed,
		"wide_confidence_band": wide_confidence_band,
		"recent_model_deterioration": False,
		"product_mismatch": False,
		"manual_review_required": manual_review,
	}


def compute_signal(fair_value: float, quote: dict | None, flags: dict, val_rmse: float, cfg: dict) -> tuple[float | None, str]:
	if quote is None:
		return None, "NO_QUOTE_PROVIDED"

	if flags["manual_review_required"]:
		return None, "MANUAL_REVIEW"

	if quote.get("product") != "DA_BASELOAD":
		flags["product_mismatch"] = True
		flags["manual_review_required"] = True
		return None, "PRODUCT_MISMATCH"

	threshold_eur = val_rmse * cfg["curve_translation"]["signal_threshold_factor"]
	edge = fair_value - quote["price_eur_mwh"]
	if edge > threshold_eur:
		return round(edge, 4), "BUY"
	if edge < -threshold_eur:
		return round(edge, 4), "SELL"
	return round(edge, 4), "NEUTRAL"


def run_curve_translation(forecast_date: str = None) -> dict:
	cfg = load_config()
	val_end = pd.Timestamp(f"{cfg['splits']['validation_end']} 23:00", tz="UTC")
	hold_start = val_end + pd.Timedelta(hours=1)
	qa_summary = json.loads(Path("outputs/qa_summary.json").read_text())
	metrics = json.loads(Path("outputs/metrics.json").read_text())
	model = joblib.load("models/extra_trees_model.pkl")
	band_df = joblib.load("models/residual_bands.pkl")
	features = pd.read_parquet("processed/features.parquet")
	common_end = qa_summary.get("common_end")

	hold_df = features[features.index >= hold_start].copy()
	if common_end:
		hold_df = hold_df[hold_df.index <= pd.Timestamp(common_end)].copy()
	hold_df = hold_df.dropna(subset=["da_price_lag_24h", "da_price_lag_168h"])

	from src.models.extra_trees_model import FEATURE_COLS

	hold_df["y_pred"] = model.predict(hold_df[FEATURE_COLS])
	hold_df["p10_band"] = hold_df["y_pred"] + hold_df["hour_of_day"].map(band_df["p10_residual"])
	hold_df["p90_band"] = hold_df["y_pred"] + hold_df["hour_of_day"].map(band_df["p90_residual"])

	first_date = pd.Timestamp(hold_df["delivery_date_local"].iloc[0])
	delivery_dates = pd.to_datetime(hold_df["delivery_date_local"]).dt.date
	day_df = hold_df[delivery_dates == first_date.date()]

	if day_df.empty:
		raise RuntimeError("Curve translation failed: no rows found for selected forecast day.")

	if forecast_date is None:
		forecast_date = str(first_date.date())

	products = compute_da_products(day_df)
	p10 = float(hold_df["p10_band"].mean())
	p90 = float(hold_df["p90_band"].mean())

	products["confidence_p10_eur_mwh"] = round(p10, 4)
	products["confidence_p90_eur_mwh"] = round(p90, 4)
	products["output_type"] = "PRODUCTION"

	week_fv = compute_scenario_rollup(model, hold_df, days=7)
	month_fv = compute_scenario_rollup(model, hold_df, days=30)

	flags = compute_invalidation_flags(qa_summary, p10, p90, cfg)

	val_rmse = metrics.get("extra_trees_validation", {}).get("rmse", 10.0)
	quote = cfg["curve_translation"].get("market_forward_quote")
	edge, signal = compute_signal(products["da_baseload_eur_mwh"], quote, flags, val_rmse, cfg)

	origin_local = pd.Timestamp(forecast_date, tz="Europe/Berlin") - pd.Timedelta(days=1)
	origin_local = origin_local.replace(hour=12)
	origin_utc = origin_local.tz_convert("UTC")

	output = {
		"forecast_date": forecast_date,
		"forecast_origin_utc": origin_utc.isoformat(),
		"production": products,
		"scenario": {
			"output_type": "SCENARIO_FAIR_VALUE",
			"note": "Uses repeated D+1 fundamentals as proxy. Directional only.",
			"week_ahead_baseload_eur_mwh": week_fv,
			"month_ahead_baseload_eur_mwh": month_fv,
		},
		"market_forward_quote": quote,
		"edge_eur_mwh": edge,
		"signal": signal,
		"invalidation_flags": flags,
	}

	Path("outputs/curve_view.json").write_text(json.dumps(output, indent=2))
	print("Saved -> outputs/curve_view.json")
	print(f"Signal: {signal}")
	return output


if __name__ == "__main__":
	run_curve_translation()
