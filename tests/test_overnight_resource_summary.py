"""Sample and ledger accounting; no GPU or network required."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("resource_summary", Path(__file__).resolve().parents[1] / "scripts/overnight_resource_summary.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_minute_samples_are_separate_from_continuous_busy_time():
    rows = [dict(utc=f"2026-10-21T18:{minute}:00Z", values=values,
            fields=["memory_MiB", "utilization_percent", "power_W", "temperature_C"])
        for minute, values in [("20", "1024, 0, 80, 30\n2048, 50, 100, 40"),
                                ("21", "2048, 100, 200, 50\n3072, 100, 200, 60")]]
    gpus, cadence = summary.gpu_summary(rows, summary.utc("2026-10-21T18:21:00Z"), 10)
    assert gpus[0]["mean_sampled_utilization_percent"] == 50
    assert gpus[0]["loaded_sample_fraction"] == .5
    assert gpus[0]["mean_utilization_loaded_samples_percent"] == 100
    assert gpus[1]["peak_memory_GiB"] == 3
    assert cadence["median_interval_seconds"] == 60
    with pytest.raises(ValueError, match="duplicate telemetry"):
        summary.gpu_summary(rows+rows, summary.utc("2026-10-21T18:21:00Z"), 10)


def test_usage_counts_each_request_and_missing_usage_explicitly():
    usage = summary.request_summary([
        dict(usage=dict(prompt_tokens=20, completion_tokens=30, total_tokens=50), generation_seconds=2),
        dict(usage=dict(prompt_tokens=10, completion_tokens=40), generation_seconds=4),
        dict(usage={}, generation_seconds=180)])
    assert usage["persisted_requests"] == 3 and usage["requests_missing_token_usage"] == 1
    assert usage["prompt_tokens"] == 30 and usage["completion_tokens"] == 70 and usage["total_tokens"] == 100
    assert usage["max_actual_prompt_plus_completion_tokens"] == 50
    assert usage["max_request_seconds"] == 180
    with pytest.raises(ValueError, match="Malformed token usage"):
        summary.request_summary([dict(usage=dict(prompt_tokens=-1))])


def test_cost_is_total_pod_rate_and_incomplete_append_is_not_counted():
    result = summary.elapsed_cost(summary.utc("2026-10-21T18:16:28Z"), summary.utc("2026-10-22T02:16:28Z"), 3.58)
    assert result["estimated_elapsed_compute_usd"] == pytest.approx(28.64)
    warnings = []
    assert list(summary.json_lines(b'{"round":0}\n{"round":', "ledger", warnings)) == [dict(round=0)]
    assert warnings
    with pytest.raises(ValueError):
        summary.elapsed_cost(summary.utc("2026-10-21T18:16:28Z"), summary.utc("2026-10-21T18:00:00Z"), 3.58)
