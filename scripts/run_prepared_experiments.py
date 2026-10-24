"""List or execute a prepared phase against an already running local model.

Default: list only. Does not provision a GPU, download a model, or stop a pod.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/next_experiments/run_manifest.json")
    ap.add_argument("--phase", choices=["prompt_screen", "evolution_screen", "feedback_control", "compositional_control", "all"], required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-hours", type=float, default=8)
    args = ap.parse_args()
    if not 0 < args.max_hours <= 12:
        ap.error("max-hours must be in (0, 12]")
    manifest = json.loads(args.manifest.read_text())
    jobs = [j for j in manifest["jobs"] if args.phase == "all" or j["phase"] == args.phase]
    for job in jobs:
        print(job["id"], "requests="+str(job["requests"]), shlex.join(job["command"]), flush=True)
    if not args.execute:
        print("PREPARED ONLY. Add --execute after starting the pinned local writer server.")
        return
    readiness = json.loads((args.manifest.parent / "readiness.json").read_text())
    if not readiness["cpu_gate_passed"] or not readiness["all_ten_supplied_csvs_identical"]:
        raise ValueError("Run the CPU readiness audit successfully before generation")
    from audit_next_experiment_readiness import sha
    if not manifest.get("training_raw_sha256"):
        raise ValueError("Run readiness audit, then regenerate the prepared manifest to pin the training cache")
    if sha(ROOT / manifest["training_raw_path"]) != manifest["training_raw_sha256"]:
        raise ValueError("Training raw cache differs from the audited artifact")
    if sha(ROOT / "data/claims/popqa/popqa_train_filtered_10shot.csv") != manifest["training_csv_sha256"]:
        raise ValueError("Training CSV differs from the audited artifact")
    for name, expected in manifest["study_source_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Study implementation changed; inspect and regenerate the manifest: "+name)
    for job in jobs:
        path = ROOT / "configs" / (job["config"] + ".yaml")
        if hashlib.sha256(path.read_bytes()).hexdigest() != job["config_sha256"]:
            raise ValueError("Prepared config changed: "+job["id"])
    # Client endpoint is deliberately fixed to loopback; no paid hosted API.
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
        served = json.load(response)
    if manifest["writer"] not in {m["id"] for m in served["data"]}:
        raise ValueError("Local server does not expose the planned writer ID")
    env = dict(os.environ, PYTHONPATH="src:.", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", WANDB_DISABLED="true")
    deadline = time.monotonic()+args.max_hours*3600
    state_path = args.manifest.parent / "execution_state.json"
    events = json.loads(state_path.read_text()) if state_path.exists() else []
    for job in jobs:
        logs = ROOT / job["expected_logs"]
        summary = logs / "summary.json"
        if summary.exists():
            previous = json.loads(summary.read_text())
            if previous.get("status") == "finished" and previous.get("requests") == job["requests"]:
                print("Skipping completed job:",job["id"],flush=True)
                continue
            raise ValueError("Incomplete job exists; inspect its recorded budget before explicitly resuming: "+job["id"])
        if (logs / "candidates.jsonl").exists():
            raise ValueError("Existing interrupted ledger requires inspection: "+job["id"])
        remaining = deadline-time.monotonic()
        if remaining < 900:
            print("Stopping at the total time budget before starting another job",flush=True)
            break
        log_path = args.manifest.parent / "execution_logs" / (job["id"]+".log")
        log_path.parent.mkdir(parents=True,exist_ok=True)
        command = job["command"] + ["trial.max_seconds="+str(min(5400,int(remaining)-15))]
        started = time.monotonic()
        with log_path.open("a") as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = child.wait(timeout=max(1,remaining))
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                raise
        events.append(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),job=job["id"],
                           exit_code=code,elapsed_seconds=time.monotonic()-started))
        state_path.write_text(json.dumps(events,indent=2)+"\n")
        if code:
            raise RuntimeError("Study job failed; see "+str(log_path))
        result = json.loads(summary.read_text())
        if result["requests"] != job["requests"]:
            raise RuntimeError("Realized budget differs from the plan; inspect before comparing: "+job["id"])


if __name__ == "__main__":
    main()
