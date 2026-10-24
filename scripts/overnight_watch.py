"""Read-only progress snapshots for the authorized overnight campaign."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/overnight20261021"
START = dt.datetime.fromisoformat("2026-10-21T18:16:28+00:00")


def snapshot():
    manifest = json.loads((OUT / "analysis_manifest.json").read_text())
    completed, running, issued = [], [], 0
    for job in manifest["jobs"]:
        path = ROOT / job["expected_logs"] / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        count = summary.get("requests", 0)
        issued += count
        if summary.get("status") == "finished":
            completed.append(job["id"])
        else:
            running.append(dict(job=job["id"], requests=count,
                valid=summary.get("valid_candidates"),
                best_validation=summary.get("best_validation_auc_percent")))
    now = dt.datetime.now(dt.timezone.utc)
    return dict(utc=now.isoformat(), hours=(now-START).total_seconds()/3600,
        approximate_compute_usd=round((now-START).total_seconds()/3600*3.58, 2),
        issued_requests=issued, planned_requests=sum(j["requests"] for j in manifest["jobs"]),
        finished_jobs=len(completed), planned_jobs=len(manifest["jobs"]), running=running,
        generation_complete=(OUT / "generation_complete.json").exists(),
        late_writer_decision=(OUT / "late_writer_decision.json").exists(),
        cpu_complete=(OUT / "cpu_complete.json").exists(),
        late_extensions_complete=(OUT / "late_extensions_complete.json").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not 5 <= args.interval <= 60:
        parser.error("interval must be between 5 and 60 seconds")
    while True:
        result = snapshot()
        print(json.dumps(result), flush=True)
        if not args.watch or result["late_extensions_complete"]:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
