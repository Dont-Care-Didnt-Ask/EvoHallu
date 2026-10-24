"""Resumable, bounded local-GPU campaign. The plan is re-read after each action."""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=Path("out/campaign/state.json"))
    ap.add_argument("--require-cohort-verified", type=Path)
    args = ap.parse_args()
    if args.require_cohort_verified:
        marker = args.require_cohort_verified
        if not marker.exists() or not json.loads(marker.read_text()).get("common_method_predictions_identical", False):
            print("Optional plan skipped: its prerequisite cohort is not verified", flush=True)
            return
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.state.read_text()) if args.state.exists() else {"completed": [], "events": []}
    server = None
    server_stream = None
    child = None
    env = os.environ.copy()
    env.update(PYTHONPATH="src:.", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", WANDB_DISABLED="true")

    def event(status, **details):
        row = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": status, **details}
        state["events"].append(row)
        state["current"] = row
        write(args.state, state)
        print(json.dumps(row), flush=True)

    def remaining():
        plan = json.loads(args.plan.read_text())
        return dt.datetime.fromisoformat(plan["hard_deadline_utc"]).timestamp() - time.time()

    def stop_server():
        nonlocal server, server_stream
        if server is not None:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            server = None
        if server_stream:
            server_stream.close()
            server_stream = None

    try:
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Campaign interrupted by signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        while remaining() > 0:
            plan = json.loads(args.plan.read_text())
            action = next((a for a in plan["actions"] if a["id"] not in state["completed"]), None)
            if action is None:
                event("plan_finished")
                break
            if remaining() < float(action.get("minimum_remaining_seconds", 0)):
                state["completed"].append(action["id"])
                event("skipped_to_preserve_evaluation_budget", action=action["id"])
                continue
            event("starting", action=action["id"], kind=action["kind"])
            kind = action["kind"]
            if kind == "wait_file":
                deadline = time.time() + min(action.get("timeout", 1800), remaining())
                while not Path(action["path"]).is_file():
                    if time.time() >= deadline:
                        raise TimeoutError(f"Waiting for {action['path']}")
                    time.sleep(5)
                # torch.save writes atomically only at close; wait for the
                # precompute's completion message as well as the output file.
                marker = action.get("log_contains")
                while marker and marker not in Path(action["log"]).read_text(errors="replace"):
                    if time.time() >= deadline:
                        raise TimeoutError("Precompute did not report completion")
                    time.sleep(2)
            elif kind == "serve":
                stop_server()
                server_stream = open(action["log"], "a")
                server = subprocess.Popen(action["command"], env=env, stdout=server_stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                state["server_pid"] = server.pid
                deadline = time.time() + min(action.get("timeout", 900), remaining())
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited ({server.returncode}); see {action['log']}")
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        pass
                    if time.time() >= deadline:
                        raise TimeoutError("Model server did not become ready")
                    time.sleep(5)
                event("server_ready", pid=server.pid)
            elif kind == "stop_server":
                stop_server()
            elif kind == "command":
                timeout = min(float(action.get("timeout", 3600)), remaining())
                with open(action["log"], "a") as log:
                    child = subprocess.Popen(action["command"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    state["child_pid"] = child.pid
                    write(args.state, state)
                    try:
                        code = child.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        raise TimeoutError(f"Action {action['id']} exceeded {timeout:.0f}s")
                if code:
                    event("command_failed", action=action["id"], returncode=code, log=action["log"])
                    if not action.get("continue_on_failure", False):
                        raise RuntimeError(f"Action failed: {action['id']}")
            else:
                raise ValueError(f"Unknown action kind: {kind}")
            state["completed"].append(action["id"])
            event("completed", action=action["id"])
        else:
            event("hard_deadline")
    except BaseException as exc:
        event("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        stop_server()


if __name__ == "__main__":
    main()
