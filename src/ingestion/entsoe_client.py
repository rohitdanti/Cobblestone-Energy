"""
ENTSO-E API client.
Authenticated with token from environment variable.
Handles year-by-year pagination (API cap = 1 year per request).
"""
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

load_dotenv()


def has_entsoe_token() -> bool:
    """Return True if ENTSOE_API_TOKEN is set and non-empty."""
    token = os.environ.get("ENTSOE_API_TOKEN", "")
    token = token.strip()
    if not token:
        return False

    # Treat common placeholder values as missing.
    placeholders = {
        "PASTE_YOUR_ENTSOE_TOKEN_HERE",
        "your_entsoe_token_here",
        "changeme",
        "token_here",
        "ENTSOE_API_TOKEN",
    }
    return token not in placeholders


def get_client() -> EntsoePandasClient:
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token or not has_entsoe_token():
        raise EnvironmentError(
            "ENTSOE_API_TOKEN not found. "
            "Copy .env.example to .env and add your real token."
        )
    return EntsoePandasClient(api_key=token)


def load_stale_data_or_raise(out_path: Path, reason: str) -> pd.DataFrame:
    """
    Fallback helper used when live ingestion cannot run.

    If cached parquet exists, load it and continue in STALE mode.
    Otherwise raise a RuntimeError with a clear action message.
    """
    if out_path.exists():
        print(f"[INGESTION MODE] STALE_CACHE — {reason}")
        print(f"[INGESTION MODE] Using cached file: {out_path}")
        return pd.read_parquet(out_path)

    print(f"[INGESTION MODE] NO_LIVE_OR_CACHE — {reason}")
    raise RuntimeError(
        "ENTSO-E live pull unavailable and no cached stale data found at "
        f"'{out_path}'. Add ENTSOE_API_TOKEN to .env or place cached parquet files in raw_data/."
    )


def fetch_year_with_retry(func, year: int, max_retries: int = 3, **kwargs):
    """
    Call an entsoe-py function for a single calendar year.
    Retries up to max_retries times on transient errors.
    Returns None if no data is available for that year.
    """
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
    # Add a small overlap into next year to avoid endpoint-specific end-boundary drops.
    end   = pd.Timestamp(f"{year + 1}-01-01 01:00", tz="Europe/Berlin")

    for attempt in range(max_retries):
        try:
            return func(start=start, end=end, **kwargs)
        except NoMatchingDataError:
            print(f"  No data for {year} (NoMatchingDataError) — skipping.")
            return None
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1} failed for {year}: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  All retries failed for {year}. Skipping.")
    return None


def fetch_range(func, start_year: int, end_year: int, **kwargs) -> pd.Series:
    """
    Fetch data year-by-year from start_year to end_year inclusive.
    Concatenates all yearly results into one Series.
    """
    parts = []
    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year}...")
        result = fetch_year_with_retry(func, year, **kwargs)
        if result is not None:
            parts.append(result)

    if not parts:
        raise RuntimeError("No data fetched for any year. Check your API token and market code.")

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def normalise_to_dataframe(series: pd.Series, value_col: str = "value") -> pd.DataFrame:
    """
    Convert an entsoe-py Series to a clean DataFrame with 3 timestamp columns.

    DST rule:
    - delivery_start_utc  → primary key, always UTC, always unique
    - delivery_start_local → Europe/Berlin (for display only)
    - delivery_date_local  → local calendar date (for peak/off-peak/baseload groupby)

    Autumn DST: two rows share local clock label 02:00 but are distinct in UTC.
    They are NEVER averaged or collapsed.
    """
    # Ensure index is timezone-aware in Europe/Berlin
    if series.index.tz is None:
        series.index = series.index.tz_localize("Europe/Berlin", ambiguous="infer")
    elif str(series.index.tz) != "Europe/Berlin":
        series.index = series.index.tz_convert("Europe/Berlin")

    df = pd.DataFrame({value_col: series.values}, index=series.index)
    df.index.name = "delivery_start_local_raw"

    # Build all three timestamp columns
    df["delivery_start_utc"]   = series.index.tz_convert("UTC")
    df["delivery_start_local"] = series.index
    df["delivery_date_local"]  = pd.to_datetime(series.index.date)

    # Set UTC as primary index
    df = df.set_index("delivery_start_utc")
    df.index = df.index.tz_localize(None).tz_localize("UTC")  # ensure UTC dtype

    # Hard check: UTC index must be unique
    assert not df.index.duplicated().any(), (
        f"Duplicate delivery_start_utc detected in '{value_col}' series. "
        "This is a data error, not a DST event. Investigate the source."
    )

    df.rename(columns={value_col: "value"}, inplace=True)
    return df[["delivery_start_local", "delivery_date_local", "value"]]