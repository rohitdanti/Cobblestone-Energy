"""
Fetch Wind (B19) and Solar (B16) generation forecasts for DE-LU from ENTSO-E (A69).
Saves to raw_data/wind_DELU.parquet and raw_data/solar_DELU.parquet
"""
import sys
import pandas as pd
from pathlib import Path

# Allow direct script execution: python src/ingestion/fetch_generation.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.entsoe_client import (
    get_client,
    fetch_range,
    normalise_to_dataframe,
    has_entsoe_token,
    load_stale_data_or_raise,
)

COUNTRY_CODE = "DE_LU"
TARGET_START_UTC = pd.Timestamp("2019-01-01 00:00", tz="UTC")
TARGET_END_UTC = pd.Timestamp("2025-09-30 23:00", tz="UTC")


def _get_existing_cached(out_path: Path) -> pd.DataFrame | None:
    if not out_path.exists():
        return None
    df = pd.read_parquet(out_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def _compute_missing_window(
    existing_df: pd.DataFrame | None,
    target_start_utc: pd.Timestamp,
    target_end_utc: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if existing_df is None or existing_df.empty:
        return target_start_utc, target_end_utc
    existing_max = existing_df.index.max()
    if existing_max >= target_end_utc:
        return None
    start = max(target_start_utc, existing_max + pd.Timedelta(hours=1))
    if start > target_end_utc:
        return None
    return start, target_end_utc


def _merge_and_save(
    existing_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
    out_path: Path,
    target_start_utc: pd.Timestamp,
    target_end_utc: pd.Timestamp,
) -> pd.DataFrame:
    combined = new_df if existing_df is None else pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined[(combined.index >= target_start_utc) & (combined.index <= target_end_utc)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    return combined


def _fetch_generation_type(psr_type: str, label: str, out_path: Path,
                            start_year: int, end_year: int, end_month: int = 9) -> pd.DataFrame:
    print(f"Fetching {label} generation forecast (A69, psrType={psr_type})...")
    target_start = pd.Timestamp(f"{start_year}-01-01 00:00", tz="UTC")
    target_end = pd.Timestamp(f"{end_year}-{end_month:02d}-01", tz="UTC") + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)

    existing_df = _get_existing_cached(out_path)
    missing_window = _compute_missing_window(existing_df, target_start, target_end)
    if missing_window is None:
        print("  Cache already covers target range. No live pull needed.")
        return existing_df

    if not has_entsoe_token():
        return load_stale_data_or_raise(
            out_path,
            "ENTSOE_API_TOKEN missing, skipping live pull.",
        )

    print("[INGESTION MODE] LIVE_PULL — ENTSOE_API_TOKEN found")

    try:
        client = get_client()
        start_utc, end_utc = missing_window
        print(f"  Incremental pull window: {start_utc} -> {end_utc}")

        series = fetch_range(
            client.query_wind_and_solar_forecast,
            start_year=start_utc.year,
            end_year=end_utc.year,
            country_code=COUNTRY_CODE,
            psr_type=psr_type,
        )

        # The result may be a DataFrame with one column per psr_type
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        # ENTSO-E may return quarter-hourly forecasts; normalize to hourly.
        if series.index.tz is None:
            series.index = series.index.tz_localize("Europe/Berlin", ambiguous="infer")
        else:
            series.index = series.index.tz_convert("Europe/Berlin")
        series = series.resample("h").mean()

        df = normalise_to_dataframe(series, value_col=f"{label}_mw")
        df = df[(df.index >= start_utc) & (df.index <= end_utc)]
        merged = _merge_and_save(existing_df, df, out_path, target_start, target_end)
        print(f"  Saved {len(merged)} total rows → {out_path}")
        return merged
    except Exception as exc:
        return load_stale_data_or_raise(
            out_path,
            f"Live pull failed ({exc}). Falling back to cached stale data.",
        )


def fetch_wind(start_year: int = 2019, end_year: int = 2025, end_month: int = 9) -> pd.DataFrame:
    return _fetch_generation_type(
        psr_type="B19", label="wind",
        out_path=Path("raw_data/wind_DELU.parquet"),
        start_year=start_year, end_year=end_year, end_month=end_month,
    )


def fetch_solar(start_year: int = 2019, end_year: int = 2025, end_month: int = 9) -> pd.DataFrame:
    return _fetch_generation_type(
        psr_type="B16", label="solar",
        out_path=Path("raw_data/solar_DELU.parquet"),
        start_year=start_year, end_year=end_year, end_month=end_month,
    )


if __name__ == "__main__":
    fetch_wind()
    fetch_solar()