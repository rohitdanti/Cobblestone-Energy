"""
Build LLM context JSON from pipeline outputs, call LLM, verify numbers, and write commentary.
"""
import json
import re
import sys
from pathlib import Path

# Allow direct script execution: python src/ai/commentary.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.llm_client import call_llm

SYSTEM_PROMPT = """You are an automated energy market analyst for a European power trading desk.
Write a 3-5 sentence commentary on the German DA power price forecast shown in the input data.
Only use numbers from the input JSON. Do not invent or extrapolate any values.
Reference: (1) the forecast level, (2) the primary fundamental driver, (3) the confidence band width, (4) data quality.
Keep under 120 words.
If qa_all_passed is false, start with [DATA QUALITY WARNING]."""


def build_context(curve_view: dict, metrics: dict, qa_summary: dict, features_path: str = "processed/features.parquet") -> dict:
	"""Assemble compact JSON context strictly from computed pipeline outputs."""
	import pandas as pd

	fv = curve_view["production"]
	forecast_date = curve_view["forecast_date"]
	features = pd.read_parquet(features_path)
	recent = features.tail(7 * 25)

	context = {
		"forecast_date": forecast_date,
		"da_baseload_forecast_eur_mwh": fv["da_baseload_eur_mwh"],
		"da_peak_forecast_eur_mwh": fv["da_peak_eur_mwh"],
		"confidence_p10": fv["confidence_p10_eur_mwh"],
		"confidence_p90": fv["confidence_p90_eur_mwh"],
		"residual_load_gw": round(float(recent["residual_load_mw"].tail(24).mean()) / 1000, 2),
		"residual_load_7d_avg_gw": round(float(recent["residual_load_mw"].mean()) / 1000, 2),
		"wind_forecast_gw": round(float(recent["wind_forecast_mw"].tail(24).mean()) / 1000, 2),
		"wind_7d_avg_gw": round(float(recent["wind_forecast_mw"].mean()) / 1000, 2),
		"solar_forecast_gw": round(float(recent["solar_forecast_mw"].tail(24).mean()) / 1000, 2),
		"load_forecast_gw": round(float(recent["load_forecast_mw"].tail(24).mean()) / 1000, 2),
		"recent_mae_eur_mwh": metrics.get("extra_trees_validation", {}).get("mae", None),
		"qa_all_passed": qa_summary.get("all_checks_passed", True),
		"outliers_flagged": sum(v.get("outliers", 0) for v in qa_summary.get("series", {}).values()),
	}
	return context


def verify_numbers(commentary: str, context: dict, tol: float = 0.05) -> bool:
	"""Reject responses containing non-date numbers not traceable to context within tolerance."""

	def extract_floats(text: str) -> list[float]:
		# Ignore horizon/count tokens like "7-day" that are descriptive rather than forecast values.
		text = re.sub(r"\b\d+\s*-\s*day\b", "", text, flags=re.IGNORECASE)
		candidates = re.findall(r"\b\d+\.?\d*\b", text)
		return [float(c) for c in candidates if not (len(c) == 4 and c.startswith("20"))]

	context_nums = extract_floats(json.dumps(context))
	for num in extract_floats(commentary):
		if not any(abs(num - ctx) <= tol for ctx in context_nums):
			return False
	return True


def run_commentary_pipeline() -> str:
	curve_view = json.loads(Path("outputs/curve_view.json").read_text())
	metrics = json.loads(Path("outputs/metrics.json").read_text())
	qa_summary = json.loads(Path("outputs/qa_summary.json").read_text())

	forecast_date = curve_view["forecast_date"]
	context = build_context(curve_view, metrics, qa_summary)
	user_prompt = (
		f"Write the daily commentary for {forecast_date}.\n\n"
		f"Input data:\n{json.dumps(context, indent=2)}"
	)

	log = call_llm(SYSTEM_PROMPT, user_prompt, forecast_date=forecast_date)

	if log["success"] and log["response_text"]:
		verified = verify_numbers(log["response_text"], context)
		log["number_verification_passed"] = verified
		commentary = log["response_text"] if verified else "[LLM_RESPONSE_REJECTED: unverifiable number in response]"
		if not verified:
			print("  WARNING: LLM response rejected - number not traceable to input context.")
	else:
		commentary = "[LLM_UNAVAILABLE]"
		print("  WARNING: LLM call failed - see outputs/llm_logs/ for details.")

	# Persist post-verification status to the per-date log file.
	log_path = Path("outputs/llm_logs") / f"{forecast_date}.json"
	if log_path.exists():
		log_path.write_text(json.dumps(log, indent=2))

	Path("outputs/daily_commentary.md").write_text(
		f"# Daily Commentary - {forecast_date}\n\n{commentary}\n"
	)
	print("Commentary written -> outputs/daily_commentary.md")
	return commentary


if __name__ == "__main__":
	print(run_commentary_pipeline())
