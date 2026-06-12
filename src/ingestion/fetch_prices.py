"""
Fetch Day-Ahead prices for DE-LU from ENTSO-E (document type A44).
Saves to raw_data/da_prices_DELU.parquet
"""
import sys
import pandas as pd
from pathlib import Path

# Allow direct script execution: python src/ingestion/fetch_prices.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.entsoe_client import (
    get_client,
    normalise_to_dataframe,
    has_entsoe_token,
    load_stale_data_or_raise,
)

COUNTRY_CODE = "DE_LU"
OUT_PATH = Path("raw_data/da_prices_DELU.parquet")
TARGET_START_UTC = pd.Timestamp("2019-01-01 00:00", tz="UTC")
TARGET_END_UTC = pd.Timestamp("2025-09-30 23:00", tz="UTC")
_NO_DATA = object()


def _get_existing_cached() -> pd.DataFrame | None:
    """Load cached parquet if present, normalized to UTC index."""
    if not OUT_PATH.exists():
        return None

    df = pd.read_parquet(OUT_PATH)
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
    """Return UTC [start, end] for incremental pull, or None if already complete."""
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
    target_start_utc: pd.Timestamp,
    target_end_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Merge incremental rows into cache, de-dupe on UTC index, and persist."""
    combined = new_df if existing_df is None else pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined[(combined.index >= target_start_utc) & (combined.index <= target_end_utc)]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH)
    return combined


def _fetch_month_with_retry(client, month_start: pd.Timestamp, month_end: pd.Timestamp,
                            max_retries: int = 3):
    """Fetch one month of A44 prices with exponential-backoff retries."""
    for attempt in range(max_retries):
        try:
            return client.query_day_ahead_prices(
                country_code=COUNTRY_CODE,
                start=month_start,
                end=month_end,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "NoMatchingDataError":
                print(f"    No data available for {month_start.strftime('%Y-%m')} — stopping incremental pull.")
                return _NO_DATA
            wait = 2 ** attempt
            print(
                f"    Attempt {attempt + 1} failed for {month_start.strftime('%Y-%m')}: "
                f"{type(exc).__name__}: {repr(exc)}. Retrying in {wait}s..."
            )
            import time
            time.sleep(wait)

    print(f"    All retries failed for {month_start.strftime('%Y-%m')}. Skipping month.")
    return None


def _fetch_prices_monthly(client, start_ts_utc: pd.Timestamp, end_ts_utc: pd.Timestamp) -> pd.Series:
    """
    Fetch A44 prices month-by-month.

    A44 can return HTTP 400 when requested with wide yearly windows;
    monthly chunks are API-safe and still cover full history.
    """
    parts = []
    start_local = start_ts_utc.tz_convert("Europe/Berlin")
    end_local = end_ts_utc.tz_convert("Europe/Berlin")
    end_local_inclusive = end_local - pd.Timedelta(hours=1)

    month_starts = pd.date_range(
        start=start_local.normalize().replace(day=1),
        end=end_local_inclusive.normalize().replace(day=1),
        freq="MS",
        tz="Europe/Berlin",
    )

    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthBegin(1) + pd.Timedelta(hours=1)
        print(f"  Fetching {month_start.strftime('%Y-%m')}...")

        result = _fetch_month_with_retry(client, month_start, month_end)
        if result is _NO_DATA:
            break
        if result is None:
            continue

        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        parts.append(result)

    if not parts:
        raise RuntimeError("No data fetched for any month. Check API token and market code.")

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def fetch_da_prices(start_year: int = 2019, end_year: int = 2025, end_month: int = 9) -> pd.DataFrame:
    print("Fetching DA prices (A44)...")
    target_start = pd.Timestamp(f"{start_year}-01-01 00:00", tz="UTC")
    target_end = pd.Timestamp(f"{end_year}-{end_month:02d}-01", tz="UTC") + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)

    existing_df = _get_existing_cached()
    missing_window = _compute_missing_window(existing_df, target_start, target_end)
    if missing_window is None:
        print("  Cache already covers target range. No live pull needed.")
        return existing_df

    if not has_entsoe_token():
        return load_stale_data_or_raise(
            OUT_PATH,
            "ENTSOE_API_TOKEN missing, skipping live pull.",
        )

    print("[STAGE 1: INGESTION] LIVE_PULL — ENTSOE_API_TOKEN found")

    try:
        client = get_client()
        start_utc, end_utc = missing_window
        print(f"  Incremental pull window: {start_utc} -> {end_utc}")

        series = _fetch_prices_monthly(client, start_ts_utc=start_utc, end_ts_utc=end_utc)

        df = normalise_to_dataframe(series, value_col="da_price_eur_mwh")

        # Keep only incremental rows, then merge to full cache.
        df = df[(df.index >= start_utc) & (df.index <= end_utc)]
        merged = _merge_and_save(existing_df, df, target_start, target_end)
        print(f"  Saved {len(merged)} total rows → {OUT_PATH}")
        return merged
    except Exception as exc:
        return load_stale_data_or_raise(
            OUT_PATH,
            f"Live pull failed ({exc}). Falling back to cached stale data.",
        )


if __name__ == "__main__":
    fetch_da_prices()