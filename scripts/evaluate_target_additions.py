"""Add source-frozen late controls to the completed 1B transfer cohort on CPU."""
import ast
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from audit_frozen_estimators import dotted
from merge_evaluation_cohorts import merge_cohorts
from verify_augmented_evaluation import verify_cohort


def main():
    deadline = dt.datetime.fromisoformat("2026-10-15T10:40:00+00:00").timestamp()
    original = Path("out/campaign/transfer_llama32_1b")
    writer = Path("out/logs/mod1/uncertainty_quantification_devstral_evo_native_1048")
    while time.time() < deadline-180:
        complete = (original / "all_datasets.json").exists()
        finished_writer = (writer / "summary.json").exists() and json.loads((writer / "summary.json").read_text()).get("status") == "finished"
        state = Path("out/campaign/state_phase2.json")
        retry_processed = state.exists() and "devstral_native_request_retry" in json.loads(state.read_text()).get("completed", [])
        if complete and (finished_writer or retry_processed):
            break
        time.sleep(15)
    else:
        print("Late transfer additions skipped to preserve the shutdown budget", flush=True)
        return
    if len(json.loads((original / "all_datasets.json").read_text())) != 9:
        raise ValueError("The original transfer cohort must complete all nine datasets")
    paths = sorted(Path("out/campaign/large_linear_budget_frozen").glob("*/frozen_estimator.py"))
    if (writer / "frozen_estimator.py").exists() and not (writer / "exclude_from_evaluation.json").exists():
        paths.append(writer / "frozen_estimator.py")
    snapshot = Path("out/campaign/target_additions_frozen")
    provenance = []
    for path in paths:
        text = path.read_text()
        tree = ast.parse(text)
        calls = {dotted(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if any(c in {"open", "eval", "exec", "__import__"} or c.startswith(("np.random.", "numpy.random.", "random.", "os.", "subprocess.", "requests.", "urllib.")) for c in calls):
            raise ValueError("Late frozen method needs a reproducibility review before target scoring: " + str(path))
        destination = snapshot / path.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination / path.name)
        shutil.copy2(path.parent / "summary.json", destination / "summary.json")
        provenance.append({"source": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()})
    delta = Path("out/campaign/transfer_llama32_1b_additions_delta")
    output = Path("out/campaign/transfer_llama32_1b_augmented")
    command = [sys.executable, "scripts/evaluate_parallel.py", "--workers", "3", "--bootstrap", "500",
               "--cache-dir", "/dev/shm/mod1-transfer-cache", "--model-id", "/dev/shm/mod1-models/Llama3.2-1B",
               "--tokenizer-id", "/workspace/mod1-models/llama31", "--output", str(delta),
               "--frozen-glob", str(snapshot / "*" / "frozen_estimator.py"), "--method-prefix", "new/"]
    env = os.environ.copy()
    env.update(PYTHONPATH="src:.", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = child.wait(timeout=min(600, deadline-time.time()))
        if code:
            raise RuntimeError("Supplementary target scoring failed")
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    merge_cohorts(original, delta, output)
    result = verify_cohort(original, output, new_prefix="new/")
    (output / "late_addition_protocol.json").write_text(json.dumps({
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_frozen_programs": provenance,
        "target_labels_used_for_selection": False, "reason": "Prespecified late source-model writers and both higher-budget fixed-search controls",
        "original_target_predictions_preserved": result["common_method_predictions_identical"]}, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
