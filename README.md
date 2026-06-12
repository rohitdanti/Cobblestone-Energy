# ROHITH DANTI - rdanti1@asu.edu
# Cobblestone Power - Take Home Assessment
----------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------
## DE-LU DA Price Forecasting Pipeline

## Overview
This repository contains the pipeline for DE-LU day-ahead (DA) price forecasting and fair-value translation.

For evaluators:
- `SUBMISSION.md` is the concise case-study writeup aligned to the assignment requirements.
- `README.md` is the technical project documentation with setup, run instructions, and implementation details.

The pipeline runs in six stages:
1. Ingestion (incremental sync from ENTSO-E or stale-cache fallback)
2. QA checks
3. Feature engineering
4. Modelling (seasonal naive baseline + global Extra Trees)
5. Curve translation
6. AI commentary

## Final Scope
- Market: DE-LU (EIC `10Y1001A1001A82H`)
- Frequency: Hourly
- Data horizon: `2019-01-01` to `2025-09-30` (frozen)
- Splits:
  - Train: through `2023-12-31`
  - Validation: `2024-01-01` to `2024-12-31`
  - Holdout: `2025-01-01` to `2025-09-30`

## Setup
### Prerequisites
- Python 3.11
- ENTSO-E API token (optional if cached parquet files exist)
- OpenAI API key (optional; commentary degrades gracefully)

### Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure secrets
```bash
cp .env.example .env
# Add ENTSOE_API_TOKEN and OPENAI_API_KEY
```

## Run
```bash
./.venv/bin/python run_pipeline.py
```


## 1. Objective
I have chosen to DE-LU market to Forecast hourly DA prices and translate them into actionable fair-value products with robust data quality controls.

## 2. Data and Windowing
- Market: DE-LU, EIC `10Y1001A1001A82H`
- Horizon: `2019-01-01 00:00 UTC` to `2025-09-30 23:00` (Berlin-local day-end converted to UTC in QA)
- Forecast target: DA hourly price (`da_price_eur_mwh`)

## 3. Ingestion
### Files
- `src/ingestion/fetch_prices.py`
- `src/ingestion/fetch_load.py`
- `src/ingestion/fetch_generation.py`
- `src/ingestion/entsoe_client.py`

### Public source endpoints
- Source provider: ENTSO-E Transparency Platform
- API base endpoint: `https://web-api.tp.entsoe.eu/api`
- Python client used in code: `entsoe-py` over the same public ENTSO-E API
- Market / bidding zone: DE-LU (`10Y1001A1001A82H`)
- Datasets pulled:
  - Day-Ahead prices: ENTSO-E document type `A44`
  - Total load forecast: ENTSO-E document type `A65`
  - Wind forecast: ENTSO-E document type `A69`, `psrType=B19`
  - Solar forecast: ENTSO-E document type `A69`, `psrType=B16`

### Key behavior
- Ingestion pulls only publicly accessible ENTSO-E data; no synthetic or paid data is used.
- For faster reruns, raw data is cached into parquet files and reused when present.
- Incremental pulls only for missing windows.
- Cached parquet history is merged and deduplicated by UTC index.
- DA prices use month-by-month chunking to avoid A44 query failures on wider windows.
- Load/wind/solar are resampled to hourly when ENTSO-E returns sub-hourly resolution.

### Timezone and DST handling
- UTC is the primary timestamp key used for storage, joins, deduplication, and QA.
- `delivery_start_local` is preserved in `Europe/Berlin` for market-calendar logic such as hour-of-day, delivery date, peak/off-peak grouping, and DST-aware reporting.
- Autumn DST is handled by keeping both repeated local `02:00` hours as distinct UTC timestamps; they are never collapsed or averaged away.
- QA builds the expected coverage window from the configured start date and the Berlin-local `23:00` end-of-day converted to UTC.

### Output schema
Each raw parquet is UTC-indexed with:
- `delivery_start_local`
- `delivery_date_local`
- value column (`da_price_eur_mwh`, `load_mw`, `wind_mw`, `solar_mw`)

### Output files
 - `raw_data/da_prices_DELU.parquet`
 - `raw_data/load_DELU.parquet`
 - `raw_data/solar_DELU.parquet`
 - `raw_data/wind_DELU.parquet`

## 4. QA checks
### Source Code
- `src/quality/qa_checks.py`

### Checks
1. Coverage against expected hourly index
2. Gap characterization (`<=2h` vs `>2h`)
3. Duplicate UTC timestamps
4. Outliers (bounds + rolling sigma for load)
5. Cross-series alignment
6. Distribution summary stats

### Window and alignment logic
- Expected index is built from config date range.
- End timestamp uses Berlin-local 23:00 converted to UTC.
- Series are clipped to expected window before scoring.
- Alignment is computed on common intersection.
- QA summary publishes:
  - `aligned`
  - `common_start`
  - `common_end`
  - `all_checks_passed`

### Reported QA outputs
- Coverage by field over the expected hourly window
- Missingness as total missing hours plus counts of gaps `<=2h` and `>2h`
- Duplicate UTC timestamp checks
- Obvious outlier flags using hard bounds for price / wind / solar and rolling sigma checks for load
- Cross-series alignment over the common timestamp intersection

### Output File
 - `outputs/qa_report.md`
 - `outputs/qa_summary.json`

## 5. Feature Engineering
### File
- `src/features/feature_engineering.py`

### Core features
- Calendar: hour/day/month with sin/cos encodings, weekend flag
- Fundamentals: load/wind/solar forecasts, residual load, renewable share
- Lags: `da_price_lag_24h`, `da_price_lag_168h`

Output: `processed/features.parquet`

## 6. Modeling
### File
- `src/models/extra_trees_model.py`

### Model setup
- Target choice: **Option A** from the case study. The model forecasts next-day hourly Day-Ahead prices in a historical backtest setting, then reuses those hourly forecasts for curve-relevant translation.

- Primary model: `ExtraTreesRegressor`
- Baseline: seasonal naive
- Split windows from config:
  - Train Set :through 2023-12-31
  - Validation Set: 2024
  - Holdout/Testing Set 2025-01 through 2025-09
- Rolling validation summary: quarter-sized folds within validation window

### Forecast output scope
- The model produces hourly DA price predictions for the full holdout window, not just a single day.
- In the current frozen backtest setup, that means hourly predictions are generated from `2025-01-01` through `2025-09-30`.
- These full-horizon hourly predictions are used for model evaluation and are written to `outputs/submission.csv`.
- The first holdout day, `2025-01-01`, is later used as the worked daily example for the curve-translation stage.

### Validation and metrics
- Validation is time-series appropriate: all splits are chronological, and no future data is used in training.
- The main evaluation windows are:
  - Train: through `2023-12-31`
  - Validation: `2024-01-01` to `2024-12-31`
  - Holdout: `2025-01-01` to `2025-09-30`
- A rolling blocked validation diagnostic is also run inside the validation year using quarter-sized folds.
- Reported metrics include:
  - `MAE`
  - `RMSE`
  - `tail_mae_high`
  - `tail_mae_low`
- This gives both price-level accuracy and a simple view of performance in the upper and lower tails.

NOTE: To finalize a model, I ran trials with 6 models and found out that ExtraTreesRegressor model had the best performance. The code for the model comparision is in /experiments/model_comparision and the experiment results are in outputs/model_comparison_summary.json and outputs/model_comparison.csv


### Artifacts
- `models/extra_trees_model.pkl`
- `models/residual_bands.pkl`
- `outputs/metrics.json`
- `outputs/submission.csv`
- `outputs/figures/forecast_vs_actual.png`
- `outputs/figures/feature_importance.png`

## 7. Curve Translation
### File
- `src/curve/curve_translation.py`

### Outputs
- `outputs/curve_view.json` with:
  - Production DA products (baseload, peak, off-peak, spread)
  - Scenario fair values (week/month)
  - Confidence bands
  - Signal and invalidation flags

Uses `extra_trees_validation.rmse` from metrics for edge thresholding.

### Concrete translation from forecast to tradable view
- Hourly DA price forecasts are first converted into product-level fair values:
  - DA baseload
  - DA peak
  - DA off-peak
  - peak-base spread
- The pipeline also computes confidence bands from validation residuals, producing a simple uncertainty range around the hourly forecast.
- For prompt-curve relevance, the hourly model is additionally rolled into:
  - week-ahead baseload scenario fair value
  - month-ahead baseload scenario fair value
- If a prompt market quote is provided in `config/config.yaml`, the pipeline computes:
  - `edge = model fair value - market quote`
  - `BUY` if edge is sufficiently positive
  - `SELL` if edge is sufficiently negative
  - `NEUTRAL` otherwise
- The trade trigger is confidence-aware: the edge must exceed a threshold based on validation RMSE times the configured `signal_threshold_factor`.

### How to interpret the forecast horizon
- The curve translation stage does not display the full holdout panel.
- Instead, it selects only the first holdout delivery date, `2025-01-01`, and converts that day's hourly predictions into DA baseload, peak, off-peak, and spread products.
- This is the explicit daily forecast view shown in `outputs/curve_view.json` and used by `outputs/daily_commentary.md`.
- The week-ahead and month-ahead values in `outputs/curve_view.json` are not separate direct delivery-period forecasts.
- They are scenario rollups: the pipeline takes the latest 24-hour feature pattern, repeats it forward for 7 or 30 days, re-scores the hourly model, and averages the resulting predicted hours.
- These week/month numbers should therefore be described precisely as **directional scenario fair values derived from the hourly DA model**, not as a full probabilistic forecast distribution or a standalone forward-curve model.

### What the desk would do with it
- If model baseload fair value is above the prompt quote, the desk would lean long prompt baseload exposure.
- If model baseload fair value is below the prompt quote, the desk would lean short prompt baseload exposure.
- If the peak-base spread implied by the hourly forecast is attractive, the desk could express the view via shape rather than outright flat price.
- The week-ahead and month-ahead scenario fair values provide a directional prompt-curve anchor even when a live forward quote is not injected into the prototype.

### What invalidates the signal
- Data QA failure: if upstream coverage / alignment checks fail, the signal should not be trusted.
- Wide confidence band: if forecast uncertainty is too large, the desk should reduce conviction or stand down.
- Product mismatch: if the market quote does not correspond to the same product definition as the model fair value, the edge is invalid.
- Manual review flag: any of the above conditions escalates the output from automatic signal to analyst review.

## 8. AI Commentary
### File
- `src/ai/commentary.py`

### Behavior
- This is the programmatic AI / LLM component used in the pipeline.
- It automates a short daily analyst-style commentary from computed pipeline outputs, reducing the need for manual writeups.
- Builds compact context from curve + metrics + QA + recent features
- Calls the LLM from code through `src/ai/llm_client.py`
- Verifies numeric traceability before accepting response
- Writes `outputs/daily_commentary.md`

### AI workflow compliance
- The LLM is called programmatically from pipeline code, not used as a manual chat tool.
- The LLM input is grounded only in computed pipeline outputs, not free-form market text.
- The prompt instructs the model not to invent numbers, and the pipeline applies a post-generation numeric verification check before accepting the response.
- Prompts, responses, token usage, latency, and success / failure status are logged to `outputs/llm_logs/`.
- Failure modes are handled explicitly:
  - missing API key
  - API call failure
  - response rejection due to unverifiable numbers
- If the LLM is unavailable, the pipeline degrades gracefully and still completes.
- Secrets are not committed: `OPENAI_API_KEY` is read from environment variables / `.env`, not hardcoded in the repository.

## 9. Entrypoints
- Full pipeline: `./.venv/bin/python run_pipeline.py`
- Tests: `./.venv/bin/python -m unittest discover -s tests -v`

## 9.1 Latest Measured Run (2026-06-11)
- Feature matrix rows written: `59158`
- Split sample counts: train `43612`, validation `8784`, holdout `6550`
- QA pass: `true` with aligned overlap window through `2025-09-30 21:00:00+00:00`
- Holdout error: Extra Trees MAE `17.5564` vs Naive MAE `31.9634`
- Validation error: Extra Trees MAE `18.0585` vs Naive MAE `32.4001`
- Submission size: `6550` predictions
- Curve output signal: `NO_QUOTE_PROVIDED`

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

## Data sources
- ENTSO-E Transparency Platform API
- Base endpoint: `https://web-api.tp.entsoe.eu/api`
- Datasets:
  - A44 DA prices
  - A65 load forecast
  - A69 wind forecast (`B19`)
  - A69 solar forecast (`B16`)

## Main outputs
- `outputs/qa_report.md`
- `outputs/qa_summary.json`
- `processed/features.parquet`
- `models/extra_trees_model.pkl`
- `models/residual_bands.pkl`
- `outputs/metrics.json`
- `outputs/submission.csv`
- `outputs/curve_view.json`
- `outputs/daily_commentary.md`
- `outputs/figures/forecast_vs_actual.png`
- `outputs/figures/feature_importance.png`

## Tests
Run full test suite:
```bash
./.venv/bin/python -m unittest discover -s tests -v
```

## Latest Verified Run (2026-06-11)
- Pipeline command: `./.venv/bin/python run_pipeline.py`
- QA status: `all_checks_passed = true`, `aligned = true`
- QA common window: `2019-01-01 00:00:00+00:00` to `2025-09-30 21:00:00+00:00`
- Coverage: DA `100.0%`, Load `99.915%`, Wind `99.997%`, Solar `99.997%`
- Validation MAE: Naive `32.4001`, Extra Trees `18.0585`
- Holdout MAE: Naive `31.9634`, Extra Trees `17.5564`
- Submission rows: `6550`
- Curve signal: `NO_QUOTE_PROVIDED` (forward quote is null)
