"""
Single entry point. Runs the full pipeline in sequence.
Idempotent: re-running overwrites outputs.
"""
from pathlib import Path


def main() -> None:
	print("=" * 60)
	print("Cobblestone Power Pipeline")
	print("=" * 60)

	files = [
		"raw_data/da_prices_DELU.parquet",
		"raw_data/load_DELU.parquet",
		"raw_data/wind_DELU.parquet",
		"raw_data/solar_DELU.parquet",
	]

	print("\n[1/6] Ingestion - syncing raw data (incremental live pull or cache)...")
	from src.ingestion.fetch_generation import fetch_solar, fetch_wind
	from src.ingestion.fetch_load import fetch_load
	from src.ingestion.fetch_prices import fetch_da_prices

	fetch_da_prices()
	fetch_load()
	fetch_wind()
	fetch_solar()

	print("\n[2/6] QA checks...")
	from src.quality.qa_checks import run_qa_pipeline

	qa = run_qa_pipeline()
	if not qa["all_checks_passed"]:
		print("  WARNING: QA checks did not all pass. Review outputs/qa_report.md before proceeding.")

	print("\n[3/6] Feature engineering...")
	from src.features.feature_engineering import run_feature_pipeline

	run_feature_pipeline()

	print("\n[4/6] Training model and generating forecasts...")
	from src.models.extra_trees_model import generate_figures, train_and_evaluate

	model, val_df, hold_df, band_df, metrics = train_and_evaluate()
	generate_figures(val_df, hold_df, model)

	print("\n[5/6] Curve translation...")
	from src.curve.curve_translation import run_curve_translation

	run_curve_translation()

	print("\n[6/6] AI commentary...")
	from src.ai.commentary import run_commentary_pipeline

	run_commentary_pipeline()

	print("\n" + "=" * 60)
	print("Pipeline complete. Required outputs:")
	outputs = [
		"outputs/qa_report.md",
		"outputs/qa_summary.json",
		"outputs/metrics.json",
		"outputs/submission.csv",
		"outputs/curve_view.json",
		"outputs/daily_commentary.md",
		"outputs/figures/forecast_vs_actual.png",
		"outputs/figures/feature_importance.png",
	]
	for path in outputs:
		exists = "OK" if Path(path).exists() else "MISSING"
		print(f"  {exists}  {path}")
	print("=" * 60)


if __name__ == "__main__":
	main()
