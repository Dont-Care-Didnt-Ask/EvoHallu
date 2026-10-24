"""Back up this night's outputs and enforce the user's eight-hour pod limit.

Only manages the explicitly authorized pod. No credential is logged or saved.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/overnight20261021"
SSH = "ssh -i /root/.ssh/processbench_pod_ed25519 -p 5933 -o BatchMode=yes -o ConnectTimeout=15"
HOST = "root@204.12.201.57"
DEADLINE = dt.datetime.fromisoformat("2026-10-22T02:16:28+00:00").timestamp()


def record(**event):
    event["utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with (OUTPUT / "resource_guard.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")


def backup():
    ok = True
    for source in ["out", "results/next_experiments", "results/overnight20261021", "configs/next_experiments"]:
        destination = ROOT / source
        destination.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["rsync", "-a", "--update", "--partial", "--exclude=resource_guard*", "--exclude=watchdog.cancel",
                 "-e", SSH, f"{HOST}:/workspace/mod1/{source}/", str(destination) + "/"],
                capture_output=True, text=True, timeout=100)
            ok &= result.returncode == 0
            if result.returncode:
                record(action="backup_error", source=source, exit_code=result.returncode,
                       error=result.stderr[-1500:])
        except Exception as exc:
            ok = False
            record(action="backup_exception", source=source, error=type(exc).__name__)
    record(action="backup", complete=ok)
    return ok


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record(action="started", deadline_utc="2026-10-22T02:16:28Z")
    while time.time() < DEADLINE - 150:
        if (OUTPUT / "watchdog.cancel").exists():
            record(action="cancelled_after_manual_management")
            return
        backup()
        time.sleep(min(120, max(1, DEADLINE - 150 - time.time())))
    backup()
    # Authorized shutdown at the budget limit; storage remains with RunPod.
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/stop_remote_pod.py"),
               "--host", HOST, "--port", "5933", "--identity", "/root/.ssh/processbench_pod_ed25519",
               "--expected-id", "p4ta9o85on4ojb", "--expected-name", "qualified_fuchsia_sole", "--execute"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=200)
        record(action="deadline_stop", exit_code=result.returncode,
               response=result.stdout, error=result.stderr[-1500:])
    except Exception as exc:
        record(action="deadline_stop_exception", error=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
