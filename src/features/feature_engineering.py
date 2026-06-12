"""
Build the ML feature matrix from the four cleaned data series.
Saves to processed/features.parquet.

Forecast origin: D-1 at 12:00 Europe/Berlin local time.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("processed")
PROCESSED.mkdir(exist_ok=True)


def _coerce_value_column(df: pd.DataFrame, preferred: str) -> pd.DataFrame:
	"""Normalize expected value column names across ingestion variants."""
	if preferred in df.columns:
		return df
	if "value" in df.columns:
		return df.rename(columns={"value": preferred})
	raise KeyError(f"Missing expected column '{preferred}' (and fallback 'value').")


def load_raw() -> dict[str, pd.DataFrame]:
	prices = pd.read_parquet("raw_data/da_prices_DELU.parquet")
	load = pd.read_parquet("raw_data/load_DELU.parquet")
	wind = pd.read_parquet("raw_data/wind_DELU.parquet")
	solar = pd.read_parquet("raw_data/solar_DELU.parquet")

	return {
		"prices": _coerce_value_column(prices, "da_price_eur_mwh"),
		"load": _coerce_value_column(load, "load_mw"),
		"wind": _coerce_value_column(wind, "wind_mw"),
		"solar": _coerce_value_column(solar, "solar_mw"),
	}


def impute_feature_gaps(df: pd.DataFrame, col: str, max_fill: int = 2) -> pd.DataFrame:
	"""
	Forward-fill gaps of up to max_fill hours in feature series.
	Gaps longer than max_fill are left as NaN.
	"""
	df[col] = df[col].ffill(limit=max_fill)
	return df


def build_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
	prices = raw["prices"].rename(columns={"da_price_eur_mwh": "target"})
	load = raw["load"].rename(columns={"load_mw": "load_forecast_mw"})
	wind = raw["wind"].rename(columns={"wind_mw": "wind_forecast_mw"})
	solar = raw["solar"].rename(columns={"solar_mw": "solar_forecast_mw"})

	for frame, col in [
		(load, "load_forecast_mw"),
		(wind, "wind_forecast_mw"),
		(solar, "solar_forecast_mw"),
	]:
		impute_feature_gaps(frame, col)

	df = (
		prices[["delivery_start_local", "delivery_date_local", "target"]]
		.join(load[["load_forecast_mw"]], how="left")
		.join(wind[["wind_forecast_mw"]], how="left")
		.join(solar[["solar_forecast_mw"]], how="left")
	)

	local_hour = df["delivery_start_local"].dt.hour.astype(float)
	local_dow = df["delivery_start_local"].dt.dayofweek.astype(float)
	local_mon = df["delivery_start_local"].dt.month.astype(float)

	df["hour_of_day"] = local_hour.astype(int)
	df["hour_sin"] = np.sin(2 * np.pi * local_hour / 24)
	df["hour_cos"] = np.cos(2 * np.pi * local_hour / 24)
	df["day_of_week"] = local_dow.astype(int)
	df["day_sin"] = np.sin(2 * np.pi * local_dow / 7)
	df["day_cos"] = np.cos(2 * np.pi * local_dow / 7)
	df["month"] = local_mon.astype(int)
	df["month_sin"] = np.sin(2 * np.pi * local_mon / 12)
	df["month_cos"] = np.cos(2 * np.pi * local_mon / 12)
	df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

	df["residual_load_mw"] = (
		df["load_forecast_mw"] - df["wind_forecast_mw"] - df["solar_forecast_mw"]
	)
	df["renewable_share_pct"] = (
		(df["wind_forecast_mw"] + df["solar_forecast_mw"])
		/ df["load_forecast_mw"].replace(0, np.nan)
		* 100
	)

	df["da_price_lag_24h"] = df["target"].shift(24)
	df["da_price_lag_168h"] = df["target"].shift(168)

	delivery_date = pd.to_datetime(df["delivery_date_local"])
	d_minus_1 = delivery_date - pd.Timedelta(days=1)
	origin_local = (
		pd.to_datetime(d_minus_1.dt.strftime("%Y-%m-%d") + " 12:00")
		.dt.tz_localize("Europe/Berlin")
		.dt.tz_convert("UTC")
	)
	df["forecast_origin_utc"] = origin_local

	n_before = len(df)
	df = df.dropna(subset=["target"])
	n_dropped = n_before - len(df)
	if n_dropped:
		print(f"  Dropped {n_dropped} rows with missing DA price target.")

	# Guardrail placeholder column kept for downstream auditability.
	df["feature_available_at_utc"] = df["forecast_origin_utc"]
	assert (df["feature_available_at_utc"] <= df["forecast_origin_utc"]).all(), (
		"LEAKAGE DETECTED: some features are timestamped after the forecast origin."
	)

	print(f"  Feature matrix: {len(df)} rows x {df.shape[1]} columns")
	return df


def run_feature_pipeline() -> pd.DataFrame:
	print("Building features...")
	raw = load_raw()
	df = build_features(raw)
	out = PROCESSED / "features.parquet"
	df.to_parquet(out)
	print(f"  Saved -> {out}")
	return df


if __name__ == "__main__":
	run_feature_pipeline()
