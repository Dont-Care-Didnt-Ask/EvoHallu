import datetime as dt
import argparse
import json
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--current-only", action="store_true")
args = ap.parse_args()
now = dt.datetime.now(dt.timezone.utc)
result = {"utc": now.isoformat(), "allocated_hours_since_connection":
          (now-dt.datetime.fromisoformat("2026-10-14T23:15:00+00:00")).total_seconds()/3600}
state = Path("out/campaign/state.json")
if state.exists():
    data = json.loads(state.read_text())
    result["stage"] = data.get("current")
    result["completed_actions"] = data.get("completed")
phase2 = Path("out/campaign/state_phase2.json")
if phase2.exists():
    data = json.loads(phase2.read_text())
    result["phase1_stage"] = result.get("stage")
    result["stage"] = data.get("current")
    result["completed_phase2_actions"] = data.get("completed")
for action, suffix in [("optional_small_writer", "phase3"), ("optional_devstral_writer", "phase5"),
                       ("devstral_native_request_retry", "phase5_retry"), ("optional_4b_structured", "phase4")]:
    nested = Path("out/campaign/state_"+suffix+".json")
    if nested.exists() and (result.get("stage") or {}).get("action") == action:
        data = json.loads(nested.read_text())
        result["phase2_stage"] = result.get("stage")
        result["stage"] = data.get("current")
        result["completed_"+suffix+"_actions"] = data.get("completed")
result["feature_cache_progress"] = {}
for kind, directory in [("source", "/dev/shm/mod1-eval-cache"), ("target_1b", "/dev/shm/mod1-transfer-cache")]:
    files = list(Path(directory).glob("*.pt"))
    result["feature_cache_progress"][kind] = {"complete_dataset_files": len(files),
                                              "raw_GiB": round(sum(p.stat().st_size for p in files)/2**30,3)}
result["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()
result["runs"] = {}
total_requests = total_valid = total_reported_tokens = 0
latest = None
for path in sorted(Path("out/logs/mod1").glob("*/summary.json")):
    summary = json.loads(path.read_text())
    total_requests += int(summary.get("requests", 0))
    total_valid += int(summary.get("valid_candidates", 0))
    total_reported_tokens += int(summary.get("total_tokens", 0))
    result["runs"][path.parent.name] = {k: v for k, v in summary.items() if k not in {"baseline_train", "baseline_val"}}
    if latest is None or path.stat().st_mtime > latest.stat().st_mtime:
        latest = path
result["totals"] = {"requests": total_requests, "valid_candidates": total_valid, "reported_tokens": total_reported_tokens}
if args.current_only and latest is not None:
    name = latest.parent.name
    result["runs"] = {name: result["runs"][name]}
if not result["runs"]:
    precompute = Path("out/campaign/precompute_train.log")
    if precompute.exists():
        result["precompute_tail"] = precompute.read_text(errors="replace")[-350:]
print(json.dumps(result, indent=2))
