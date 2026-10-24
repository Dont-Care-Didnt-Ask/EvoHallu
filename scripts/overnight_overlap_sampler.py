"""Fill CPU evaluation gaps with independent, disjoint rank-sampler jobs.

Two controllers share each local writer replica after the prompt-screen barrier.
Sampling and candidate budgets are unchanged; latency comparisons are descriptive.
"""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/overnight20261021"


def main():
    os.chdir(ROOT)
    cutoff = dt.datetime.fromisoformat("2026-10-22T01:46:28+00:00").timestamp()
    source = ROOT / "results/next_experiments/parallel_execution_state.json"
    while time.time() < cutoff - 3600:
        if source.exists():
            state = json.loads(source.read_text())
            if any(e.get("event") == "controller_stopped" for e in state.get("events", [])):
                return
            if any(e.get("event") == "batch_started" and e.get("batch") == "evolution_four_policies"
                   for e in state.get("events", [])):
                record = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                              independent_controllers_share_writer_replicas=True,
                              max_active_search_jobs=4, writer_replicas=2,
                              evaluation_workers_per_job=1, requests_per_round_per_job=2,
                              scientific_source_and_candidate_budget_changed=False,
                              latency_comparisons="Descriptive; concurrent jobs share each writer queue",
                              overlap_scope="Disjoint rank-parent jobs; source controller never launches these IDs")
                (OUT / "overlap_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
                os.execv(sys.executable, [sys.executable, "scripts/run_overnight_parallel.py", "--execute",
                    "--manifest", "results/next_experiments/sampler_manifest.json", "--phase", "all",
                    "--state", "results/next_experiments/sampler_execution_state.json", "--ports", "18000", "18001"])
        time.sleep(30)


if __name__ == "__main__":
    main()
