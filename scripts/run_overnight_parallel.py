"""Run the pinned study on independent local writer replicas.

Dry-run by default. This controller neither starts model servers nor provisions or
stops a pod. The absolute deadline includes a reserved backup/evaluation window.
"""
from __future__ import annotations

import argparse
from collections import deque
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
HARD_DEADLINE = dt.datetime(2026, 10, 22, 2, 16, 28, tzinfo=UTC)
PHASES = ("prompt_screen", "evolution_screen", "compositional_control", "feedback_control")
POLICIES = ("original", "neutral", "diverse", "compositional")


def utcnow():
    return dt.datetime.now(UTC)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def parse_deadline(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("deadline must include the UTC timezone (Z or +00:00)")
    if parsed > HARD_DEADLINE:
        raise ValueError("deadline cannot extend the authorized 2026-10-22T02:16:28Z cutoff")
    return parsed.astimezone(UTC)


def ordered_batches(manifest, phase="all"):
    """Finish the prompt screen, compare all four policies per seed, then feedback."""
    jobs = manifest["jobs"]
    if phase != "all":
        jobs = [j for j in jobs if j["phase"] == phase]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs in selected manifest jobs")
    if len({j["expected_logs"] for j in jobs}) != len(jobs):
        raise ValueError("Selected jobs would write to the same experiment log directory")
    order = {seed: i for i, seed in enumerate(manifest.get("seeds", []))}
    key = lambda j: (order.get(j["seed"], len(order)), j["seed"], POLICIES.index(j["prompt_policy"]), j["id"])
    batches = []
    screens = sorted((j for j in jobs if j["phase"] == "prompt_screen"), key=key)
    evolution = sorted((j for j in jobs if j["phase"] in {"evolution_screen", "compositional_control"}), key=key)
    feedback = sorted((j for j in jobs if j["phase"] == "feedback_control"), key=key)
    for label, selected in [("prompt_screen", screens), ("evolution_four_policies", evolution), ("feedback_control", feedback)]:
        if selected:
            batches.append((label, selected))
    if sum(len(selected) for _, selected in batches) != len(jobs):
        raise ValueError("Unrecognized phase in selected manifest jobs")
    return batches


def verify_inputs(manifest, manifest_path, jobs, root=ROOT):
    readiness = json.loads((manifest_path.parent / "readiness.json").read_text())
    if not readiness.get("cpu_gate_passed") or not readiness.get("all_ten_supplied_csvs_identical"):
        raise ValueError("CPU readiness and supplied-CSV verification must pass before generation")
    if not manifest.get("training_raw_sha256"):
        raise ValueError("Prepared manifest must pin the audited training raw cache")
    pinned = {
        manifest["training_raw_path"]: manifest["training_raw_sha256"],
        "data/claims/popqa/popqa_train_filtered_10shot.csv": manifest["training_csv_sha256"],
        **manifest["study_source_sha256"],
    }
    for name, expected in pinned.items():
        if sha(root / name) != expected:
            raise ValueError("Audited input or study source changed: " + name)
    for job in jobs:
        if sha(root / "configs" / (job["config"] + ".yaml")) != job["config_sha256"]:
            raise ValueError("Prepared config changed: " + job["id"])
    if manifest.get("contexts_sha256") and sha(manifest_path.parent / "frozen_contexts.json") != manifest["contexts_sha256"]:
        raise ValueError("Prepared frozen contexts changed")


def verify_servers(ports, writer):
    for port in ports:
        url = f"http://127.0.0.1:{port}/v1/models"
        with urllib.request.urlopen(url, timeout=10) as response:
            served = json.load(response)
        if writer not in {m["id"] for m in served.get("data", [])}:
            raise ValueError(f"Local server at port {port} does not expose the pinned writer ID {writer}")


def inspect_job(job, root=ROOT):
    """Return complete/pending; refuse to reuse any partial experiment directory."""
    folder = root / job["expected_logs"]
    summary = folder / "summary.json"
    if summary.exists():
        result = json.loads(summary.read_text())
        if result.get("status") == "finished" and result.get("requests") == job["requests"]:
            ledger = folder / "candidates.jsonl"
            if not ledger.is_file() or len([json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]) != job["requests"]:
                raise ValueError("Completed summary does not match its candidate ledger: " + job["id"])
            if not (folder / "frozen_estimator.py").is_file():
                raise ValueError("Completed job is missing its selected estimator: " + job["id"])
            return "complete"
        raise ValueError("Partial job requires explicit inspection; refusing to redo: " + job["id"])
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Interrupted job directory requires inspection; refusing to redo: " + job["id"])
    return "pending"


def child_command(job, port, max_seconds):
    return job["command"] + [f"cluster.api_base_url=http://127.0.0.1:{port}/v1", f"trial.max_seconds={int(max_seconds)}"]


def terminate(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def execute(manifest, manifest_path, batches, args):
    """The filesystem lock covers verification, execution, and final state write."""
    state_path = args.state or manifest_path.parent / "parallel_execution_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(state_path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another parallel controller holds this study lock") from exc
        jobs = [job for _, selected in batches for job in selected]
        verify_inputs(manifest, manifest_path, jobs)
        statuses = {job["id"]: inspect_job(job) for job in jobs}
        state = json.loads(state_path.read_text()) if state_path.exists() else {"events": [], "jobs": {}}
        manifest_sha = sha(manifest_path)
        if state.get("manifest_sha256", manifest_sha) != manifest_sha:
            raise ValueError("Execution state belongs to a different manifest")
        for previous in state.get("running", []):
            try:
                os.kill(previous["pid"], 0)
            except ProcessLookupError:
                continue
            raise RuntimeError("A previous controller child may still be alive; inspect PID " + str(previous["pid"]))
        cutoff = args.deadline_utc - dt.timedelta(minutes=args.reserve_minutes)
        state.update(manifest_sha256=manifest_sha, deadline_utc=args.deadline_utc.isoformat(),
                     experiment_cutoff_utc=cutoff.isoformat(), reserve_minutes=args.reserve_minutes,
                     controller_pid=os.getpid(), ports=args.ports, running=[])
        active = {}

        def persist(event=None, **details):
            if event:
                item = dict(utc=utcnow().isoformat(), event=event, **details)
                state["events"].append(item)
                print(json.dumps(item), flush=True)
            state["updated_utc"] = utcnow().isoformat()
            state["running"] = [dict(job=r["job"]["id"], port=port, pid=r["child"].pid,
                                     started_utc=r["started_utc"]) for port, r in active.items()]
            atomic_json(state_path, state)

        for job in jobs:
            if statuses[job["id"]] == "complete":
                state["jobs"][job["id"]] = dict(status="complete", resumed_skip=True)
        persist("controller_started", selected_jobs=len(jobs), completed_skips=sum(s == "complete" for s in statuses.values()))
        env = dict(os.environ, PYTHONPATH="src:.", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WANDB_DISABLED="true")
        failures, budget_stop = [], False
        old_handlers = {}

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        try:
            pending = [job for job in jobs if statuses[job["id"]] == "pending"]
            if pending and utcnow() < cutoff:
                verify_servers(args.ports, manifest["writer"])
            for label, selected in batches:
                queue = deque(j for j in selected if statuses[j["id"]] == "pending")
                persist("batch_started", batch=label, pending=len(queue))
                while queue or active:
                    now = utcnow()
                    remaining = (cutoff - now).total_seconds()
                    if remaining <= 0:
                        budget_stop = True
                        persist("experiment_cutoff_reached", pending=len(queue))
                        break
                    for port in args.ports:
                        if port in active or not queue:
                            continue
                        job = queue[0]
                        minimum = args.min_job_minutes * 60 * min(1, job["requests"] / 60)
                        if remaining < minimum + 60:
                            budget_stop = True
                            continue
                        queue.popleft()
                        # Recheck just before launch, including external/previous-run writes.
                        if inspect_job(job) == "complete":
                            statuses[job["id"]] = "complete"
                            persist("completed_skip", job=job["id"])
                            continue
                        allowance = min(args.max_job_minutes * 60, int(remaining) - 45)
                        command = child_command(job, port, allowance)
                        log_path = manifest_path.parent / "execution_logs" / (job["id"] + ".parallel.log")
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        stream = log_path.open("a")
                        try:
                            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                                     stderr=subprocess.STDOUT, start_new_session=True)
                        except OSError as exc:
                            stream.close()
                            failures.append(job["id"])
                            state["jobs"][job["id"]] = dict(status="failed_to_start", error=str(exc))
                            persist("job_failed_to_start", job=job["id"], error=str(exc))
                            continue
                        active[port] = dict(child=child, stream=stream, job=job, started=time.monotonic(),
                                            started_utc=utcnow().isoformat(), allowance=allowance)
                        state["jobs"][job["id"]] = dict(status="running", port=port, pid=child.pid,
                            command=command, log=str(log_path), max_seconds=allowance)
                        persist("job_started", job=job["id"], port=port, pid=child.pid, max_seconds=allowance)
                    for port, running in list(active.items()):
                        child, job = running["child"], running["job"]
                        elapsed = time.monotonic() - running["started"]
                        timeout = elapsed > running["allowance"] + 30
                        if timeout:
                            terminate(child)
                        code = child.poll()
                        if code is None:
                            continue
                        running["stream"].close()
                        del active[port]
                        error = None
                        try:
                            complete = code == 0 and not timeout and inspect_job(job) == "complete"
                        except (ValueError, OSError) as exc:
                            complete, error = False, str(exc)
                        status = "complete" if complete else "failed_or_partial"
                        if not complete:
                            failures.append(job["id"])
                        statuses[job["id"]] = status
                        state["jobs"][job["id"]].update(status=status, exit_code=code,
                            elapsed_seconds=elapsed, timed_out=timeout, error=error)
                        persist("job_finished", job=job["id"], port=port, status=status,
                                exit_code=code, elapsed_seconds=elapsed, error=error)
                    if budget_stop and not active:
                        break
                    if queue or active:
                        time.sleep(2)
                if budget_stop:
                    break
        except BaseException as exc:
            persist("controller_interrupted", error=repr(exc))
            raise
        finally:
            for port, running in list(active.items()):
                terminate(running["child"])
                running["stream"].close()
                job_id = running["job"]["id"]
                state["jobs"][job_id].update(status="interrupted", exit_code=running["child"].returncode)
                del active[port]
                persist("job_interrupted", job=job_id, port=port)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            persist("controller_stopped", failed_jobs=failures, budget_stop=budget_stop,
                    unfinished_jobs=[j["id"] for j in jobs if statuses[j["id"]] != "complete"])
        return 1 if failures else 2 if budget_stop else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/next_experiments/run_manifest.json")
    parser.add_argument("--phase", choices=["all", *PHASES], default="all")
    parser.add_argument("--ports", nargs="+", type=int, default=[8000, 8001])
    parser.add_argument("--deadline-utc", default="2026-10-22T02:16:28Z")
    parser.add_argument("--reserve-minutes", type=float, default=30)
    parser.add_argument("--max-job-minutes", type=float, default=90)
    parser.add_argument("--min-job-minutes", type=float, default=30,
                        help="Minimum remaining time to start a 60-request job; scaled for screens")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        args.deadline_utc = parse_deadline(args.deadline_utc)
    except ValueError as exc:
        parser.error(str(exc))
    if args.reserve_minutes < 25:
        parser.error("At least 25 minutes must remain for evaluation and backup")
    if not 0 < args.min_job_minutes <= args.max_job_minutes <= 180:
        parser.error("Require 0 < min-job-minutes <= max-job-minutes <= 180")
    if not args.ports or len(set(args.ports)) != len(args.ports) or any(not 1 <= p <= 65535 for p in args.ports):
        parser.error("Ports must be unique TCP port numbers")
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text())
    batches = ordered_batches(manifest, args.phase)
    cutoff = args.deadline_utc - dt.timedelta(minutes=args.reserve_minutes)
    print(f"Absolute deadline: {args.deadline_utc.isoformat()}; generation cutoff: {cutoff.isoformat()}", flush=True)
    for label, jobs in batches:
        print(f"Batch {label}: {len(jobs)} jobs, {sum(j['requests'] for j in jobs)} requests", flush=True)
        for index, job in enumerate(jobs):
            port = args.ports[index % len(args.ports)]
            print(job["id"], shlex.join(child_command(job, port, args.max_job_minutes * 60)), flush=True)
    if not args.execute:
        print("DRY RUN ONLY. Displayed port assignments are illustrative; execution assigns the next free replica.", flush=True)
        return 0
    return execute(manifest, args.manifest, batches, args)


if __name__ == "__main__":
    raise SystemExit(main())
