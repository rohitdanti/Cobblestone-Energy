# Full Name: Rohith Danti
# Email: rdanti1@asu.edu

# European Power Fair Value: DE-LU Day-Ahead Forecasting and Prompt Curve Translation

## Summary
This prototype builds a daily fair-value pipeline for the DE-LU power market using only public ENTSO-E data. It ingests hourly day-ahead prices plus hourly load, wind, and solar forecasts, applies data-quality controls with explicit UTC / Berlin-local handling, trains and validates an hourly DA price model, translates forecast output into prompt-curve views, and adds a programmatic LLM commentary layer with logging and safeguards.

## 1. Public Data Ingestion and Data Quality
The chosen market is DE-LU (`10Y1001A1001A82H`). All inputs come from the public ENTSO-E Transparency Platform API (`https://web-api.tp.entsoe.eu/api`) via `entsoe-py`. The datasets used are hourly Day-Ahead prices (`A44`), Total Load Forecast (`A65`), Wind Forecast (`A69`, `psrType=B19`), and Solar Forecast (`A69`, `psrType=B16`), which satisfies the requirement for hourly DA prices plus at least two matching-granularity fundamentals.

Timezone handling is explicit. UTC is the canonical storage, join, and QA key, while Berlin-local timestamps are preserved for delivery-date logic, hourly seasonality, and peak/off-peak aggregation. Autumn DST is handled by keeping repeated local hours as distinct UTC timestamps rather than collapsing them.

QA checks are implemented in `src/quality/qa_checks.py` and reported in `outputs/qa_report.md` and `outputs/qa_summary.json`. They cover coverage against the expected hourly index, missingness and gap characterization, duplicate UTC timestamps, obvious outliers, distribution summaries, and cross-series timestamp alignment. In the latest verified run, all checks passed, alignment was true, and coverage was `100.0%` for DA prices, `99.915%` for load, and `99.997%` for both wind and solar.

## 2. Forecasting and Model Validation
I choose **Option A**: forecast next-day hourly Day-Ahead prices and then translate the hourly forecast into curve-relevant products. The target is hourly DE-LU DA price. The baseline is a seasonal naive forecast using `t-168h`, and the improved model is a global `ExtraTreesRegressor` using engineered calendar, lag, and fundamental features.

Validation is time-series appropriate. The pipeline uses chronological train / validation / holdout windows:
- Train: through `2023-12-31`
- Validation: `2024-01-01` to `2024-12-31`
- Holdout: `2025-01-01` to `2025-09-30`

The validation year is also stress-tested using quarter-sized rolling blocked folds. Metrics reported include MAE, RMSE, and tail MAE for both upper and lower tails. In the latest verified run, validation MAE improved from `32.4001` for the naive baseline to `18.0585` for Extra Trees; holdout MAE improved from `31.9634` to `17.5564`.

The model produces hourly predictions across the full holdout panel, which are written to `outputs/submission.csv`. For presentation, the curve-translation stage uses the first holdout day (`2025-01-01`) as the worked daily example.

## 3. Prompt Curve Translation
Hourly DA forecasts are converted into product-level fair values: DA baseload, DA peak, DA off-peak, and peak-base spread. Validation residuals are also used to produce confidence bands. This creates a concrete DA-to-curve translation framework even when live forward quotes are not injected into the prototype.

For prompt-curve relevance, the pipeline also produces week-ahead and month-ahead **directional scenario fair values**. These are not separate direct delivery-period models or full probabilistic curve forecasts. Instead, the latest 24-hour feature pattern is repeated forward for 7 or 30 days, the hourly model is re-scored, and the predicted hours are averaged.

If a market quote is supplied in configuration, the pipeline computes an edge as:

`edge = model fair value - market quote`

and emits `BUY`, `SELL`, or `NEUTRAL` when the edge exceeds a threshold scaled to validation RMSE. In desk terms, the view can be expressed through prompt baseload exposure or via shape if the peak-base spread looks attractive. The signal is invalidated when QA fails, confidence bands are too wide, or the product being compared does not match the model fair value definition.

## 4. AI-Accelerated Workflow
The AI component is an automated daily commentary generator implemented in `src/ai/commentary.py` and `src/ai/llm_client.py`. It calls the LLM programmatically from pipeline code using `OPENAI_API_KEY` from environment variables, not as a manual chat tool. The LLM receives only structured, computed pipeline outputs: curve values, QA summary, recent feature aggregates, and model metrics.

To keep the workflow auditable and controlled, prompts, responses, token usage, latency, and failures are logged to `outputs/llm_logs/`. The prompt explicitly tells the model not to invent numbers, and a post-generation verification step rejects any commentary containing numbers that cannot be traced back to the pipeline context. If the API key is missing or the call fails, the pipeline degrades gracefully and still completes.

This reduces manual analyst effort by automatically producing a short daily drivers-style note grounded entirely in the model outputs.

## Deliverables
- Technical README with setup and run instructions: `README.md`
- Full reproducible pipeline entrypoint: `run_pipeline.py`
- Dependency file: `requirements.txt`
- QA outputs: `outputs/qa_report.md`, `outputs/qa_summary.json`
- Figures: `outputs/figures/forecast_vs_actual.png`, `outputs/figures/feature_importance.png`
- Model metrics: `outputs/metrics.json`
- Out-of-sample predictions: `outputs/submission.csv`
- Curve translation output: `outputs/curve_view.json`
- AI prompts / code / logs: `src/ai/`, `outputs/llm_logs/`

## Notes
- The README is intended as technical project documentation.
- This file is intended as the concise submission writeup.
