"""
OpenAI API wrapper with temperature=0, full logging, graceful failure.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("outputs/llm_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def call_llm(system_prompt: str, user_prompt: str, forecast_date: str = None, model: str = "gpt-4o-mini") -> dict:
	"""Call the LLM and return a structured log dictionary. Never raises."""
	from openai import OpenAI

	api_key = os.environ.get("OPENAI_API_KEY")
	token = (api_key or "").strip()
	placeholders = {
		"PASTE_YOUR_OPENAI_API_KEY_HERE",
		"your_openai_api_key_here",
		"changeme",
		"api_key_here",
	}
	if (not token) or (token in placeholders):
		log = _error_result("OPENAI_API_KEY not set", system_prompt, user_prompt, model, forecast_date)
		_write_log(log, forecast_date)
		return log

	t_start = time.time()

	try:
		client = OpenAI(api_key=token)
		response = client.chat.completions.create(
			model=model,
			temperature=0,
			max_tokens=200,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		)
		latency_ms = int((time.time() - t_start) * 1000)
		response_text = (response.choices[0].message.content or "").strip()
		log = {
			"timestamp_utc": datetime.now(timezone.utc).isoformat(),
			"forecast_date": forecast_date or "unknown",
			"model": model,
			"temperature": 0,
			"system_prompt": system_prompt,
			"user_prompt": user_prompt,
			"response_text": response_text,
			"number_verification_passed": None,
			"prompt_tokens": response.usage.prompt_tokens,
			"completion_tokens": response.usage.completion_tokens,
			"total_tokens": response.usage.total_tokens,
			"latency_ms": latency_ms,
			"success": True,
			"error": None,
		}
	except Exception as exc:
		log = _error_result(str(exc), system_prompt, user_prompt, model, forecast_date)

	_write_log(log, forecast_date)
	return log


def _error_result(error_msg: str, system_prompt: str, user_prompt: str, model: str, forecast_date: str = None) -> dict:
	return {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"forecast_date": forecast_date or "unknown",
		"model": model,
		"temperature": 0,
		"system_prompt": system_prompt,
		"user_prompt": user_prompt,
		"response_text": None,
		"number_verification_passed": None,
		"prompt_tokens": None,
		"completion_tokens": None,
		"total_tokens": None,
		"latency_ms": None,
		"success": False,
		"error": error_msg,
	}


def _write_log(log: dict, forecast_date: str = None) -> None:
	date_str = forecast_date or datetime.now().strftime("%Y-%m-%d")
	log_path = LOG_DIR / f"{date_str}.json"
	log_path.write_text(json.dumps(log, indent=2))

	sample_path = LOG_DIR / "sample_log.json"
	if not sample_path.exists():
		sample_path.write_text(json.dumps(log, indent=2))
