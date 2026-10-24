import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--deadline", required=True)
ap.add_argument("--output", type=Path, default=Path("out/campaign/gpu_telemetry.jsonl"))
args = ap.parse_args()
deadline = dt.datetime.fromisoformat(args.deadline).timestamp()
args.output.parent.mkdir(parents=True, exist_ok=True)
while time.time() < deadline:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    row = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "values": result.stdout.strip(),
           "fields": ["memory_MiB", "utilization_percent", "power_W", "temperature_C"]}
    with args.output.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    time.sleep(min(60, max(0, deadline - time.time())))
