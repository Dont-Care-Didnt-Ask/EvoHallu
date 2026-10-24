"""Execute the explicit CPU component-refit plan and retain stage exit receipts."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import sys
import threading

from overnight_late_extensions import Runner

OUT = Path("results/overnight20261021/screen_component_refit")


def main():
    if sys.argv[1:] != ["--execute"]:
        raise SystemExit("Explicit --execute is required; this fits source training data and scores CPU caches")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def event(name, **details):
        row = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), event=name, **details)
        with lock, (OUT / "execution_events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    runner = Runner(event)
    outcomes = dict(fit=runner.run("component_refit_fit",
        [sys.executable, "scripts/fit_screen_component_logistic.py", "--execute"], max_seconds=1800))
    if outcomes["fit"]["exit_code"] == 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(runner.run, "component_refit_" + name,
                ["bash", str(OUT / ("run_" + name + ".sh"))], max_seconds=3600): name
                for name in ("source_nine", "target_six")}
            for future in concurrent.futures.as_completed(futures):
                outcomes[futures[future]] = future.result()
    outcomes["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (OUT / "execution_receipt.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    print(json.dumps(outcomes), flush=True)


if __name__ == "__main__":
    main()
