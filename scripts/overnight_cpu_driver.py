"""Score completed study jobs on CPU while independent GPU searches continue."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/overnight20261021"
MANIFEST = OUT / "analysis_manifest.json"
CACHE = ROOT / "data/restored/mod1-eval-cache"
CUTOFF = dt.datetime.fromisoformat("2026-10-22T02:06:28+00:00").timestamp()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(command, label):
    remaining = CUTOFF - time.time()
    if remaining < 300:
        return False
    log = OUT / f"cpu_{label}.log"
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log.open("a") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=min(7200, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = -1
    print(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), event="cpu_finished", label=label,
                          exit_code=code, log=str(log))), flush=True)
    return code == 0


def main():
    os.chdir(ROOT)
    state_path = OUT / "cpu_execution_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(done=[], failures=[], supervised=False)
    files = [p for p in sorted((ROOT / "data/claims").rglob("*.csv")) if "popqa_train" not in p.name]
    names = [p.parent.name + "__" + p.stem for p in files]
    if len(names) != 9:
        raise ValueError("Expected exactly nine evaluation datasets")
    evaluator = [sys.executable, "scripts/overnight_eval.py", "--manifest", str(MANIFEST),
                 "--cache-dir", str(CACHE), "--execute", "--workers", "3", "--bootstrap", "500"]
    while time.time() < CUTOFF:
        if not all((CACHE / (name + ".pt")).exists() for name in names):
            time.sleep(30)
            continue
        if not state["supervised"]:
            if run(evaluator + ["--kind", "supervised", "--roles", "primary"], "supervised"):
                state["supervised"] = True
            else:
                state["failures"].append("supervised")
                dump(state_path, state)
                raise RuntimeError("Supervised CPU evaluation failed; inspect before retrying")
            dump(state_path, state)
        manifest = json.loads(MANIFEST.read_text())
        ready = []
        for job in manifest["jobs"]:
            if job["phase"] == "prompt_screen" or job["id"] in state["done"] + state["failures"]:
                continue
            path = ROOT / job["expected_logs"] / "summary.json"
            if path.exists():
                summary = json.loads(path.read_text())
                if summary.get("status") == "finished" and summary.get("requests") == job["requests"]:
                    ready.append(job["id"])
        if ready:
            batch = ready[:4]
            label = "batch_" + str(len(state["done"]))
            success = run(evaluator + ["--kind", "new", "--include-train-selected", "--roles", "primary,trainbest",
                                       "--jobs", ",".join(batch)], label)
            state["done" if success else "failures"].extend(batch)
            dump(state_path, state)
            run([sys.executable, "scripts/overnight_report.py", "--manifest", str(MANIFEST)], "report")
            continue
        if (OUT / "generation_complete.json").exists() and (OUT / "late_writer_decision.json").exists():
            run([sys.executable, "scripts/overnight_report.py", "--manifest", str(MANIFEST)], "final_report")
            dump(OUT / "cpu_complete.json", dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), **state))
            return
        time.sleep(30)
    dump(OUT / "cpu_deadline.json", dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), **state))


if __name__ == "__main__":
    main()
