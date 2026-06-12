"""
Data quality checks for all four series.
Outputs:
  - outputs/qa_report.md     (human-readable)
  - outputs/qa_summary.json  (machine-readable, fed to LLM)
"""
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import yaml

RAW_FILES = {
    "da_prices": Path("raw_data/da_prices_DELU.parquet"),
    "load":      Path("raw_data/load_DELU.parquet"),
    "wind":      Path("raw_data/wind_DELU.parquet"),
    "solar":     Path("raw_data/solar_DELU.parquet"),
}

VALUE_COLS = {
    "da_prices": "da_price_eur_mwh",
    "load":      "load_mw",
    "wind":      "wind_mw",
    "solar":     "solar_mw",
}

OUTLIER_BOUNDS = {
    "da_prices": (-200, 1000),
    "load":      (0, None),     # uses rolling-sigma check instead
    "wind":      (0, 90_000),   # MW — 90 GW cap
    "solar":     (0, 90_000),
}

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)


def _expected_index() -> pd.DatetimeIndex:
    cfg = yaml.safe_load(Path("config/config.yaml").read_text())
    start = pd.Timestamp(f"{cfg['date_range']['start']} 00:00", tz="UTC")
    end_local = pd.Timestamp(f"{cfg['date_range']['end']} 23:00", tz="Europe/Berlin")
    end = end_local.tz_convert("UTC")
    return pd.date_range(start=start, end=end, freq="h", tz="UTC")


def _clip_to_expected_window(df: pd.DataFrame) -> pd.DataFrame:
    expected_idx = _expected_index()
    return df.loc[df.index.intersection(expected_idx)].sort_index()


def load_series(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_FILES[name])
    # Hard-fail on duplicate UTC
    assert not df.index.duplicated().any(), (
        f"FATAL: Duplicate delivery_start_utc in '{name}'. "
        "Ingestion error — do not proceed."
    )
    return df


def check_coverage(df: pd.DataFrame, name: str) -> dict:
    """Check 1: what % of expected UTC hourly slots have data."""
    expected_idx = _expected_index()
    hours_expected = len(expected_idx)
    hours_present  = df.loc[df.index.intersection(expected_idx), "value"].notna().sum()
    coverage_pct   = round(hours_present / hours_expected * 100, 3)
    return {
        "hours_expected": hours_expected,
        "hours_present":  int(hours_present),
        "coverage_pct":   coverage_pct,
        "flag":           coverage_pct < 99.0,
    }


def check_gaps(df: pd.DataFrame, name: str, is_target: bool = False) -> dict:
    """Check 2: detect and characterise consecutive missing UTC slots."""
    expected_idx = _expected_index()
    missing = expected_idx.difference(df.index.intersection(expected_idx))
    gaps = []

    # Group consecutive missing timestamps into runs
    if len(missing) > 0:
        gaps_series = pd.Series(missing)
        diff = gaps_series.diff()
        gap_starts = gaps_series[diff != pd.Timedelta("1h")].tolist()
        for start_ts in gap_starts:
            run = [t for t in missing if start_ts <= t < start_ts + pd.Timedelta("200h")]
            run = sorted(run)
            # find contiguous run
            contiguous = [run[0]]
            for i in range(1, len(run)):
                if run[i] - run[i-1] == pd.Timedelta("1h"):
                    contiguous.append(run[i])
                else:
                    break
            gaps.append({
                "start": str(contiguous[0]),
                "end":   str(contiguous[-1]),
                "hours": len(contiguous),
            })

    gaps_gt2h = [g for g in gaps if g["hours"] > 2]
    gaps_le2h = [g for g in gaps if g["hours"] <= 2]

    result = {
        "total_missing_hours": len(missing),
        "gaps_le2h_count": len(gaps_le2h),
        "gaps_gt2h_count": len(gaps_gt2h),
        "gaps_gt2h":       gaps_gt2h[:10],  # show first 10 in report
        "is_target":       is_target,
        "impute_note":     (
            "Target (DA prices): missing rows DROPPED, never imputed."
            if is_target else
            "Feature series: gaps ≤2h forward-filled; gaps >2h excluded from training."
        ),
    }
    return result


def check_duplicates(df: pd.DataFrame, name: str) -> dict:
    """Check 3: UTC index must be strictly unique (already asserted in load_series)."""
    n_dupes = df.index.duplicated().sum()
    return {
        "duplicate_utc_count": int(n_dupes),
        "passed":              n_dupes == 0,
    }


def check_outliers(df: pd.DataFrame, name: str) -> dict:
    """Check 4: flag statistically extreme values."""
    col = "value"
    lo, hi = OUTLIER_BOUNDS.get(name, (None, None))
    outlier_mask = pd.Series(False, index=df.index)

    if lo is not None:
        outlier_mask |= df[col] < lo
    if hi is not None:
        outlier_mask |= df[col] > hi

    # Rolling sigma check for load
    if name == "load":
        rolling_mean = df[col].rolling(28 * 24, min_periods=24).mean()
        rolling_std  = df[col].rolling(28 * 24, min_periods=24).std()
        outlier_mask |= (df[col] - rolling_mean).abs() > 3 * rolling_std

    outlier_count = int(outlier_mask.sum())
    sample = (df[outlier_mask]["value"].head(5).to_dict()
              if outlier_count > 0 else {})
    return {
        "outlier_count":   outlier_count,
        "sample_outliers": {str(k): v for k, v in sample.items()},
        "note":            "Negative DA prices are valid in Germany — flagged, not removed.",
    }


def check_distribution(df: pd.DataFrame, name: str) -> dict:
    """Check 6: basic distribution statistics."""
    col = "value"
    s = df[col].dropna()
    return {
        "mean": round(s.mean(), 3),
        "std":  round(s.std(),  3),
        "min":  round(s.min(),  3),
        "p5":   round(s.quantile(0.05), 3),
        "p25":  round(s.quantile(0.25), 3),
        "p50":  round(s.quantile(0.50), 3),
        "p75":  round(s.quantile(0.75), 3),
        "p95":  round(s.quantile(0.95), 3),
        "max":  round(s.max(),  3),
    }


def check_alignment(series_dict: dict) -> dict:
    """Check 5: all series must cover the same UTC index after individual checks."""
    ranges = {n: (df.index.min(), df.index.max()) for n, df in series_dict.items()}
    all_indexes = [set(df.index) for df in series_dict.values()]
    common = set.intersection(*all_indexes)
    union  = set.union(*all_indexes)
    return {
        "date_ranges":          {n: (str(s), str(e)) for n, (s, e) in ranges.items()},
        "common_timestamps":    len(common),
        "union_timestamps":     len(union),
        "missing_from_any":     len(union) - len(common),
        "aligned":              len(union) == len(common),
        "common_start":         str(min(common)) if common else None,
        "common_end":           str(max(common)) if common else None,
    }


def run_all_checks() -> dict:
    print("Running QA checks...")
    loaded = {}
    results = {}

    for name in RAW_FILES:
        print(f"  Loading {name}...")
        df = load_series(name)
        # Rename value column to generic 'value' for checks
        value_col = VALUE_COLS[name]
        df = df.rename(columns={value_col: "value"})
        loaded[name] = _clip_to_expected_window(df)

    for name, df in loaded.items():
        is_target = name == "da_prices"
        results[name] = {
            "coverage":     check_coverage(df, name),
            "gaps":         check_gaps(df, name, is_target=is_target),
            "duplicates":   check_duplicates(df, name),
            "outliers":     check_outliers(df, name),
            "distribution": check_distribution(df, name),
        }

    common_indexes = [set(df.index) for df in loaded.values()]
    common = set.intersection(*common_indexes)
    aligned_loaded = {name: df.loc[df.index.intersection(common)].sort_index() for name, df in loaded.items()}
    alignment = check_alignment(aligned_loaded)

    all_checks_passed = (
        all(r["coverage"]["coverage_pct"] >= 99.0 for r in results.values()) and
        all(r["duplicates"]["passed"] for r in results.values()) and
        alignment["aligned"]
    )

    return {
        "results":          results,
        "alignment":        alignment,
        "all_checks_passed": all_checks_passed,
    }


def write_qa_report(qa: dict) -> None:
    """Write human-readable QA report to outputs/qa_report.md."""
    lines = [
        "# Data Quality Report",
        f"**Run date:** {date.today()}",
        f"**All checks passed:** {qa['all_checks_passed']}",
        "",
        "---",
    ]

    for name, r in qa["results"].items():
        lines += [
            f"\n## {name}",
            f"\n### Coverage",
            f"- Expected hours: {r['coverage']['hours_expected']}",
            f"- Present hours: {r['coverage']['hours_present']}",
            f"- Coverage: {r['coverage']['coverage_pct']}%",
            f"- Flag: {r['coverage']['flag']}",
            f"\n### Gaps",
            f"- Total missing hours: {r['gaps']['total_missing_hours']}",
            f"- Gaps ≤2h (forward-fillable): {r['gaps']['gaps_le2h_count']}",
            f"- Gaps >2h (excluded from training): {r['gaps']['gaps_gt2h_count']}",
            f"- Note: {r['gaps']['impute_note']}",
            f"\n### Outliers",
            f"- Count: {r['outliers']['outlier_count']}",
            f"- Note: {r['outliers']['note']}",
            f"\n### Distribution",
            "| Stat | Value |",
            "|---|---|",
        ]
        for stat, val in r["distribution"].items():
            lines.append(f"| {stat} | {val} |")

    lines += [
        "\n---",
        "\n## Cross-Series Alignment",
        f"- Common timestamps: {qa['alignment']['common_timestamps']}",
        f"- Missing from any series: {qa['alignment']['missing_from_any']}",
        f"- Aligned: {qa['alignment']['aligned']}",
    ]

    Path("outputs/qa_report.md").write_text("\n".join(lines))
    print("  Written → outputs/qa_report.md")


def write_qa_summary(qa: dict) -> None:
    """Write machine-readable summary to outputs/qa_summary.json (for LLM)."""
    summary = {
        "run_date": str(date.today()),
        "series": {
            name: {
                "coverage_pct":   r["coverage"]["coverage_pct"],
                "gaps_gt2h":      r["gaps"]["gaps_gt2h_count"],
                "outliers":       r["outliers"]["outlier_count"],
                "duplicate_utc":  not r["duplicates"]["passed"],
            }
            for name, r in qa["results"].items()
        },
        "aligned": qa["alignment"]["aligned"],
        "common_start": qa["alignment"].get("common_start"),
        "common_end": qa["alignment"].get("common_end"),
        "all_checks_passed": qa["all_checks_passed"],
    }
    Path("outputs/qa_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("  Written → outputs/qa_summary.json")


def run_qa_pipeline():
    qa = run_all_checks()
    write_qa_report(qa)
    write_qa_summary(qa)
    print(f"\nQA complete. all_checks_passed = {qa['all_checks_passed']}")
    return qa


if __name__ == "__main__":
    run_qa_pipeline()