"""Independent pod-side deadline guard for the user's authorized eight hours."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = dt.datetime.fromisoformat("2026-10-22T02:16:28+00:00").timestamp()


def main():
    record = ROOT / "results/overnight20261021/remote_guard.json"
    record.write_text(json.dumps(dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                     deadline_utc="2026-10-22T02:16:28Z",
                                     pod_id="p4ta9o85on4ojb", action="waiting"), indent=2) + "\n")
    while time.time() < DEADLINE:
        time.sleep(min(30, max(.1, DEADLINE - time.time())))
    subprocess.run([sys.executable, str(ROOT / "scripts/stop_own_pod.py"),
                    "--expected-id", "p4ta9o85on4ojb", "--expected-name", "qualified_fuchsia_sole", "--execute"],
                   check=True, timeout=60)


if __name__ == "__main__":
    main()
