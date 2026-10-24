"""Freeze the current reference cohort and score datasets as GPU caches finish."""
import concurrent.futures
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    output = Path("out/campaign/evaluation_before_9b")
    snapshot = Path("out/campaign/reference_frozen")
    output.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    if not (snapshot / "snapshot_complete.json").exists():
        for path in sorted(Path("out/logs/mod1").glob("*/frozen_estimator.py")):
            if (path.parent / "exclude_from_evaluation.json").exists():
                continue
            if not json.loads((path.parent / "summary.json").read_text())["status"] == "finished":
                raise ValueError("Reference cohort contains an unfinished run")
            destination = snapshot / path.parent.name
            destination.mkdir(parents=True, exist_ok=True)
            for name in ["frozen_estimator.py", "summary.json", "trial_config.json"]:
                if (path.parent / name).exists():
                    shutil.copy2(path.parent / name, destination / name)
        (snapshot / "snapshot_complete.json").write_text(json.dumps({
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "cohort_selected_before_held_out_scoring": True,
            "methods": [p.parent.name for p in sorted(snapshot.glob("*/frozen_estimator.py"))]}, indent=2)+"\n")
    files = [p for p in sorted(Path("data/claims").rglob("*.csv")) if "popqa_train" not in p.name]
    if len(files) != 9:
        raise ValueError("Expected nine datasets")
    names = [p.parent.name+"__"+p.stem for p in files]
    cache = Path("/dev/shm/mod1-eval-cache")
    deadline = dt.datetime.fromisoformat("2026-10-15T10:40:00+00:00").timestamp()
    env = os.environ.copy()
    env.update(PYTHONPATH="src:.", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def evaluate(index):
        name = names[index]
        if (output / (name+".json")).exists():
            return name
        command = [sys.executable, "scripts/evaluate_cached_claims.py", "--mode", "evaluate",
                   "--dataset-index", str(index), "--cpu-threads", "1", "--bootstrap", "500",
                   "--output", str(output), "--frozen-glob", str(snapshot/"*/frozen_estimator.py")]
        with (output / (name+".log")).open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=max(1, deadline-time.time()))
        return name

    pending = set(range(9))
    active = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while pending or active:
            if time.time() >= deadline:
                raise TimeoutError("Streaming reference evaluation deadline reached")
            ready = [i for i in pending if (cache / (names[i]+".pt")).exists()]
            ready.sort(key=lambda i:(cache/(names[i]+".pt")).stat().st_size, reverse=True)
            for i in ready[:max(0,3-len(active))]:
                active[pool.submit(evaluate,i)] = i
                pending.remove(i)
                print("Started reference dataset",names[i],flush=True)
            for future in list(active):
                if future.done():
                    print("Completed reference dataset",future.result(),flush=True)
                    del active[future]
            time.sleep(2)
    aggregate = {name: json.loads((output/(name+".json")).read_text()) for name in names}
    temporary = output / "all_datasets.tmp"
    temporary.write_text(json.dumps(aggregate,indent=2,allow_nan=False)+"\n")
    temporary.replace(output/"all_datasets.json")
    print("All nine reference datasets complete",flush=True)


if __name__ == "__main__":
    main()
