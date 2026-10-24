"""Start the pinned controller once cache transfer and both servers are ready."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    timeout = time.monotonic() + 1800
    manifest = json.loads((ROOT / "results/next_experiments/run_manifest.json").read_text())
    raw = ROOT / manifest["training_raw_path"]
    expected_size = 0
    archive_manifest = json.loads((ROOT / "data/cache_backup_manifest.json").read_text())
    for name, info in archive_manifest.items():
        if name.endswith("train/popqa_train_filtered_10shot_raw_stats_logits.pt.gz"):
            expected_size = info["raw_bytes"]
    while time.monotonic() < timeout:
        ready = raw.exists() and (not expected_size or raw.stat().st_size == expected_size)
        for port in [18000, 18001]:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as f:
                    model_ids = {x["id"] for x in json.load(f)["data"]}
                ready &= manifest["writer"] in model_ids
            except Exception:
                ready = False
        if ready:
            print("READY", dt.datetime.now(dt.timezone.utc).isoformat(), flush=True)
            os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/run_overnight_parallel.py"),
                                      "--phase", "all", "--execute", "--ports", "18000", "18001"])
        time.sleep(10)
    raise TimeoutError("Training cache or writer servers did not become ready")


if __name__ == "__main__":
    main()
