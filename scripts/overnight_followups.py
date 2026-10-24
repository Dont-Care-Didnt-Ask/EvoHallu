"""Finish the authorized night: 9B controls, then conditional 27B confirmation.

Read-only dry run by default. --execute runs on the GPU pod itself. This script
never stops a pod; the independent resource guards enforce the rental deadline.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/overnight20261021"
DEADLINE = dt.datetime.fromisoformat("2026-10-22T02:16:28+00:00")
CUTOFF = dt.datetime.fromisoformat("2026-10-22T01:46:28+00:00")
PORTS = [18000, 18001]
REV9 = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
REV27 = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE = ROOT / "results/next_experiments/run_manifest.json"
SAMPLER = ROOT / "results/next_experiments/sampler_manifest.json"
CONFIRMATION = ROOT / "results/next_experiments/confirmation27b_manifest.json"
SOURCE_STATE = ROOT / "results/next_experiments/parallel_execution_state.json"
TERMINAL_OBSERVATION_GRACE_SECONDS = 90


def now():
    return dt.datetime.now(dt.timezone.utc)


def minutes_remaining():
    return (CUTOFF - now()).total_seconds() / 60


def confirmation_cutoff(reserve_transfer_minutes=45):
    return CUTOFF - dt.timedelta(minutes=reserve_transfer_minutes)


def confirmation_minutes_remaining(reserve_transfer_minutes=45):
    return (confirmation_cutoff(reserve_transfer_minutes) - now()).total_seconds() / 60


def transfer_reservation(reserve_transfer_minutes=45):
    return dict(reserved_gpu_feature_transfer_minutes=reserve_transfer_minutes,
                confirmation_generation_cutoff_utc=confirmation_cutoff(reserve_transfer_minutes).isoformat(),
                planned_gpu_feature_window=dict(start_utc=confirmation_cutoff(reserve_transfer_minutes).isoformat(),
                    end_utc=CUTOFF.isoformat(), purpose="Full nine-dataset feature transfer after writer generation"),
                scope="27B confirmation only; source 9B and sampler retain the global experiment cutoff")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def full_budget(job, root=ROOT):
    folder = root / job["expected_logs"]
    summary = folder / "summary.json"
    ledger = folder / "candidates.jsonl"
    if not summary.is_file() or not ledger.is_file() or not (folder / "frozen_estimator.py").is_file():
        return False
    result = read(summary)
    return (result.get("status") == "finished" and result.get("requests") == job["requests"]
            and len([json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]) == job["requests"])


def controller_stopped(state):
    events = state.get("events", [])
    starts = [i for i, item in enumerate(events) if item.get("event") == "controller_started"]
    stops = [i for i, item in enumerate(events) if item.get("event") == "controller_stopped"]
    return bool(starts and stops and stops[-1] > starts[-1] and not state.get("running"))


def terminal_evidence(state):
    if controller_stopped(state):
        return "recorded_controller_stopped_event"
    pid = state.get("controller_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    children = [row.get("pid") for row in state.get("running", [])]
    if any(isinstance(child, bool) or not isinstance(child, int) or child <= 1 for child in children):
        return None
    if not process_alive(pid) and not any(process_alive(child) for child in children):
        return "recorded_controller_and_children_no_longer_alive"
    return None


def wait_controller_terminal(state_path, manifest_path, event, poll_seconds, optional=False):
    """Observe shutdown beyond the launch cutoff without extending generation.

    An overlapping sampler owns its separate controller lock. Never launch a
    second controller or rotate its model while that first owner is still live.
    """
    if optional and not state_path.exists():
        return None
    expected = sha(manifest_path)
    observation_deadline = CUTOFF + dt.timedelta(seconds=TERMINAL_OBSERVATION_GRACE_SECONDS)
    event("waiting_for_controller_terminal", state=str(state_path), manifest_sha256=expected,
          observation_deadline_utc=observation_deadline.isoformat(), no_generation_budget_extension=True)
    while now() < observation_deadline:
        if state_path.exists():
            state = read(state_path)
            if state.get("manifest_sha256") != expected:
                raise ValueError("Controller state identifies a different manifest: " + str(state_path))
            evidence = terminal_evidence(state)
            if evidence:
                event("controller_terminal_observed", state=str(state_path), evidence=evidence)
                return state
        time.sleep(min(poll_seconds, max(.1, (observation_deadline - now()).total_seconds())))
    raise TimeoutError("Controller shutdown was not certified before the observation deadline: " + str(state_path))


def existing_sampler_outcome(event, poll_seconds):
    state_path = ROOT / "results/next_experiments/sampler_execution_state.json"
    state = wait_controller_terminal(state_path, SAMPLER, event, poll_seconds, optional=True)
    if state is None:
        return None
    complete, errors = [], []
    for job in read(SAMPLER)["jobs"]:
        try:
            if full_budget(job):
                complete.append(job["id"])
        except (OSError, ValueError, KeyError) as exc:
            errors.append(dict(job=job["id"], error=repr(exc)))
    outcome = dict(existing_controller_terminal=True, complete_jobs=complete,
                   expected_jobs=len(read(SAMPLER)["jobs"]), artifact_errors=errors)
    event("existing_sampler_outcome", **outcome)
    return outcome


def confirmation_job_count(remaining_minutes, maximum_jobs=4):
    if maximum_jobs not in {2, 4}:
        raise ValueError("Confirmation cap must be two or four jobs")
    if remaining_minutes >= 170:
        return min(4, maximum_jobs)
    if remaining_minutes >= 100:
        return 2
    return 0


def novel_writer_priority_due(after_utc, at=None):
    """A prospective wall-clock allocation rule, independent of model quality."""
    if after_utc is None:
        return False
    threshold = dt.datetime.fromisoformat(after_utc.replace("Z", "+00:00"))
    if threshold.utcoffset() != dt.timedelta(0):
        raise ValueError("Novel-writer priority threshold must be UTC")
    return (at or now()) >= threshold


def confirmation_controller_command(manifest, reserve_transfer_minutes=45):
    if confirmation_minutes_remaining(reserve_transfer_minutes) < 100:
        raise TimeoutError("Insufficient time before the 27B cutoff; no confirmation job may be launched")
    # The controller subtracts its own 30-minute reserve from this deadline.
    deadline = confirmation_cutoff(reserve_transfer_minutes) + dt.timedelta(minutes=30)
    return [sys.executable, "scripts/run_overnight_parallel.py", "--execute", "--manifest", str(manifest),
            "--phase", "all", "--state", str(ROOT / "results/next_experiments/confirmation27b_execution_state.json"),
            "--deadline-utc", deadline.isoformat(), "--reserve-minutes", "30", "--ports", *map(str, PORTS)]


def update_analysis(manifest_paths, root=ROOT):
    """Normalize reporting phases without editing any execution manifest."""
    path = root / "results/overnight20261021/analysis_manifest.json"
    analysis = read(path) if path.exists() else copy.deepcopy(read(manifest_paths[0]))
    jobs = {item["id"]: item for item in analysis.get("jobs", [])}
    provenance = {item["path"]: item for item in analysis.get("execution_manifests", [])}
    for manifest_path in manifest_paths:
        manifest = read(manifest_path)
        relative = str(manifest_path.relative_to(root))
        digest = sha(manifest_path)
        if relative in provenance and provenance[relative]["sha256"] != digest:
            raise ValueError("An execution manifest changed after analysis provenance was recorded: " + relative)
        provenance[relative] = dict(path=relative, sha256=digest)
        for original in manifest["jobs"]:
            item = copy.deepcopy(original)
            item["execution_phase"] = original["phase"]
            item["execution_manifest"] = relative
            item["execution_manifest_sha256"] = digest
            item["writer_model"] = manifest["writer"]
            if original.get("analysis_phase") == "interface_control":
                item["phase"] = "interface_control"
            elif manifest.get("writer") == "Qwen/Qwen3.8-27B":
                item["phase"] = "writer_confirmation"
            elif original.get("parent_sampling") == "rank_uniform":
                item["phase"] = "sampler_control"
            elif original["phase"] == "compositional_control":
                item["phase"] = "evolution_screen"
            jobs[item["id"]] = item
    # Normalize already recorded 9B jobs too, including an initial combined manifest.
    for item in jobs.values():
        if item["phase"] == "compositional_control":
            item.setdefault("execution_phase", "compositional_control")
            item["phase"] = "evolution_screen"
    totals = {}
    for item in jobs.values():
        totals[item["phase"]] = totals.get(item["phase"], 0) + item["requests"]
    analysis.update(status="analysis_only_not_an_execution_manifest", jobs=list(jobs.values()),
                    execution_manifests=list(provenance.values()), requests_by_phase=totals,
                    updated_utc=now().isoformat())
    atomic(path, analysis)
    return path


def process_alive(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")") + 2:].split()[0] != "Z"
    except FileNotFoundError:
        return False


def validated_servers(metadata_path, model, revision, root=ROOT):
    metadata = read(metadata_path)
    if metadata.get("model") != model or metadata.get("revision") != revision:
        raise ValueError("Server metadata model/revision mismatch: " + str(metadata_path))
    records = metadata.get("servers", [])
    if len(records) != 2 or {r["port"] for r in records} != set(PORTS):
        raise ValueError("Server metadata must identify the two authorized ports")
    for record in records:
        pid = record["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise ValueError("Invalid managed server PID")
        command = record["command"]
        if "serve" not in command or "--served-model-name" not in command:
            raise ValueError("Metadata does not identify a vLLM server command")
        if command[command.index("--served-model-name") + 1] != model:
            raise ValueError("Server metadata command has the wrong served model")
        if command[command.index("--port") + 1] != str(record["port"]):
            raise ValueError("Server metadata command has the wrong port")
        if not process_alive(pid):
            continue
        actual = [s.decode() for s in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if s]
        # A Python shebang can prepend the interpreter before the recorded script.
        if len(actual) < len(command) - 1 or actual[-(len(command)-1):] != command[1:]:
            raise ValueError(f"PID {pid} no longer matches the managed vLLM command")
        if Path(f"/proc/{pid}/cwd").resolve() != root.resolve() or os.getpgid(pid) != pid:
            raise ValueError(f"PID {pid} has the wrong working directory or process group")
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if f"CUDA_VISIBLE_DEVICES={record['gpu']}".encode() not in environment:
            raise ValueError(f"PID {pid} has the wrong GPU assignment")
    return records


def stop_servers(metadata_path, model, revision, event):
    if not metadata_path.exists():
        event("server_metadata_absent", path=str(metadata_path), model=model)
        return
    records = validated_servers(metadata_path, model, revision)
    live = [r for r in records if process_alive(r["pid"])]
    for record in live:
        try:
            os.killpg(record["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.monotonic() + 25
    while any(process_alive(r["pid"]) for r in live) and time.monotonic() < until:
        time.sleep(1)
    for record in live:
        if process_alive(record["pid"]):
            # Validate identity again before escalating; never kill a reused PID.
            validated_servers(metadata_path, model, revision)
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    event("managed_servers_stopped", model=model, metadata_sha256=sha(metadata_path),
          pids=[r["pid"] for r in live])


def require_ports_free():
    until = time.monotonic() + 20
    occupied = []
    while time.monotonic() < until:
        occupied = []
        for port in PORTS:
            with socket.socket() as connection:
                connection.settimeout(.5)
                if connection.connect_ex(("127.0.0.1", port)) == 0:
                    occupied.append(port)
        if not occupied:
            return
        time.sleep(1)
    raise RuntimeError("Refusing to start 27B while a server still owns ports " + str(occupied))


def servers_ready(model):
    for port in PORTS:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as stream:
                if model not in {item["id"] for item in json.load(stream).get("data", [])}:
                    return False
        except (OSError, ValueError):
            return False
    return True


def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run_stage(name, command, event, env, timeout=None):
    if minutes_remaining() <= 0:
        raise TimeoutError("Experiment cutoff reached before " + name)
    log = OUT / "followup_logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    event("stage_started", stage=name, command=command, log=str(log))
    with log.open("a") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            limit = max(1, (CUTOFF - now()).total_seconds())
            code = child.wait(timeout=min(limit, timeout) if timeout else limit)
        except BaseException:
            terminate(child)
            raise
    event("stage_finished", stage=name, exit_code=code)
    return code


def wait_source(event, poll_seconds):
    if len(read(SOURCE)["jobs"]) != 19:
        raise ValueError("Expected the complete 19-job source suite")
    state = wait_controller_terminal(SOURCE_STATE, SOURCE, event, poll_seconds)
    # Feedback may have been moved earlier with unchanged original identities.
    # Observe its independent owner before checking ledgers or rotating writers.
    existing_feedback_overlap_outcome(event,poll_seconds)
    unfinished, errors = [], []
    for job in read(SOURCE)["jobs"]:
        try:
            if full_budget(job):
                continue
        except (OSError, ValueError, KeyError) as exc:
            errors.append(dict(job=job["id"], error=repr(exc)))
        unfinished.append(job["id"])
    event("source_controller_stopped", unfinished_jobs=unfinished, artifact_errors=errors,
          terminal_evidence=terminal_evidence(state))
    return not unfinished


def validate_feedback_overlap_manifest(manifest,source,parent_sha256):
    expected=[job for job in source.get("jobs",[]) if job.get("phase")=="feedback_control"]
    if (len(source.get("jobs",[]))!=19 or len(expected)!=3 or
        {job.get("seed") for job in expected}!={2107,4319,8741} or
        any(job.get("requests")!=60 or job.get("mode")!="baseline_seeded" for job in expected)):
        raise ValueError("Expected the three original full-budget fixed-feedback jobs")
    if manifest.get("jobs")!=expected or manifest.get("parent_source_manifest_sha256")!=parent_sha256:
        raise ValueError("Feedback overlap is not the exact original three-job subset/parent hash")
    keys=("writer","writer_revision","labels","feature_model","training_raw_path","training_raw_sha256",
          "training_csv_sha256","study_source_sha256")
    if any(manifest.get(key)!=source.get(key) for key in keys) or manifest.get("requests_by_phase")!={"feedback_control":180}:
        raise ValueError("Feedback overlap changed a pinned source setting or request budget")
    return expected


def existing_feedback_overlap_outcome(event,poll_seconds):
    manifest_path=ROOT/"results/next_experiments/feedback_overlap_manifest.json"
    state_path=ROOT/"results/next_experiments/feedback_overlap_state.json"
    receipt_path=ROOT/"results/overnight20261021/feedback_overlap_receipt.json"
    if not state_path.exists():
        receipt=read(receipt_path) if receipt_path.exists() else {}
        if receipt.get("status") not in {"reserved","running"}:
            return None  # A prepared-but-unlaunched manifest must not block.
    if not manifest_path.exists():
        raise ValueError("Feedback overlap owner/state exists without its immutable manifest")
    manifest=read(manifest_path)
    expected=validate_feedback_overlap_manifest(manifest,read(SOURCE),sha(SOURCE))
    state=wait_controller_terminal(state_path,manifest_path,event,poll_seconds)
    complete,errors=[],[]
    for job in expected:
        try:
            if full_budget(job):complete.append(job["id"])
        except (OSError,ValueError,KeyError) as exc:
            errors.append(dict(job=job["id"],error=repr(exc)))
    outcome=dict(existing_controller_terminal=True,terminal_evidence=terminal_evidence(state),
                 manifest_sha256=sha(manifest_path),parent_source_manifest_sha256=sha(SOURCE),
                 complete_jobs=complete,expected_jobs=3,artifact_errors=errors,
                 analysis_identities="Unchanged original source feedback_control jobs")
    event("existing_feedback_overlap_outcome",**outcome)
    return outcome


def subset_confirmation(path):
    manifest = read(path)
    subset = copy.deepcopy(manifest)
    subset["jobs"] = [j for j in manifest["jobs"] if j["seed"] == 2107]
    if len(subset["jobs"]) != 2:
        raise ValueError("Expected exactly the predeclared 2107 confirmation pair")
    subset.update(seeds=[2107], requests_by_phase={"evolution_screen": 120},
                  budget_subset_rule="Prespecified seed2107 original/chosen pair: 100..170 remaining minutes or prospective two-job breadth cap",
                  parent_confirmation_manifest=str(path.relative_to(ROOT)), parent_confirmation_manifest_sha256=sha(path))
    output = path.with_name("confirmation27b_seed2107_manifest.json")
    if output.exists():
        if read(output) != subset:
            raise ValueError("Existing reduced confirmation manifest differs; refusing to overwrite")
    else:
        atomic(output, subset)
    return output


def start_27b(event, env, timeout_minutes, reserve_transfer_minutes=45):
    if confirmation_minutes_remaining(reserve_transfer_minutes) < 100:
        event("confirmation_skipped", reason="less_than_100_minutes_before_reserved_confirmation_cutoff")
        return False
    metadata = ROOT / "out/overnight20261021/qwen27b_servers.json"
    if metadata.exists() and servers_ready("Qwen/Qwen3.8-27B"):
        validated_servers(metadata, "Qwen/Qwen3.8-27B", REV27)
        event("reusing_managed_27b_servers")
        return True
    command = [sys.executable, "scripts/overnight_start_servers.py", "--model-path", "/workspace/models/Qwen3.8-27B",
               "--model-id", "Qwen/Qwen3.8-27B", "--revision", REV27, "--tag", "qwen27b",
               "--gpu-memory-utilization", ".90", "--ports", *map(str, PORTS)]
    for eager in (False, True):
        if confirmation_minutes_remaining(reserve_transfer_minutes) < 100:
            event("confirmation_skipped", reason="less_than_100_minutes_before_reserved_confirmation_cutoff")
            return False
        if metadata.exists():
            stop_servers(metadata, "Qwen/Qwen3.8-27B", REV27, event)
        if eager and any((ROOT / j["expected_logs"]).exists() and any((ROOT / j["expected_logs"]).iterdir())
                         for j in read(CONFIRMATION)["jobs"]):
            raise RuntimeError("Eager fallback is forbidden after any confirmation job artifacts exist")
        require_ports_free()
        code = run_stage("start_27b_eager" if eager else "start_27b_compiled",
                         command + (["--enforce-eager"] if eager else []), event, env, timeout=120)
        if code:
            event("server_startup_failed", eager=eager, reason="launcher_nonzero")
        else:
            until = min(confirmation_cutoff(reserve_transfer_minutes).timestamp(), time.time() + timeout_minutes * 60)
            while time.time() < until:
                if servers_ready("Qwen/Qwen3.8-27B"):
                    validated_servers(metadata, "Qwen/Qwen3.8-27B", REV27)
                    event("27b_servers_ready", eager=eager)
                    return True
                if metadata.exists() and any(not process_alive(r["pid"]) for r in read(metadata)["servers"]):
                    break
                time.sleep(10)
            event("server_startup_failed", eager=eager, reason="server_died_or_readiness_timeout")
        # No inference job has been started. Archive compiled diagnostics before
        # the launcher truncates its standard server log filenames on retry.
        stop_servers(metadata, "Qwen/Qwen3.8-27B", REV27, event)
        if not eager:
            for path in (ROOT / "out/overnight20261021").glob("qwen27b_server_gpu*.log"):
                shutil.copy2(path, path.with_name(path.name.replace("qwen27b_", "qwen27b_compiled_failed_")))
            if metadata.exists():
                shutil.copy2(metadata, metadata.with_name("qwen27b_compiled_failed_servers.json"))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--server-ready-minutes", type=float, default=25)
    parser.add_argument("--reserve-transfer-minutes", type=float, default=45,
                        help="Reserve 0..60 GPU minutes after 27B generation for full-dataset feature transfer")
    parser.add_argument("--prioritize-novel-writer-after-utc",
                        help="After this UTC time, leave the optional writer window to the previously untested GLM")
    parser.add_argument("--max-confirmation-jobs", type=int, choices=[2, 4], default=4,
                        help="Prospective request allocation cap; two jobs retain exactly the seed2107 policy pair")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or not 1 <= args.server_ready_minutes <= 30:
        parser.error("poll-seconds must be 1..60 and server-ready-minutes 1..30")
    if not 0 <= args.reserve_transfer_minutes <= 60:
        parser.error("reserve-transfer-minutes must be in [0, 60]")
    if args.prioritize_novel_writer_after_utc:
        try:
            novel_writer_priority_due(args.prioritize_novel_writer_after_utc)
        except (ValueError, TypeError):
            parser.error("prioritize-novel-writer-after-utc must be an aware UTC timestamp")
    reservation = transfer_reservation(args.reserve_transfer_minutes)
    if not args.execute:
        print(json.dumps(dict(event="dry_run", stages=["wait_19_full_budget_9b_jobs_and_controller_stop",
            "rank_sampler_3_seeds_on_existing_9b_replicas", "prepare_validation_selected_27b_confirmation",
            "stop_validated_owned_9b_servers", "start_27b_compiled_then_eager_only_if_startup_fails",
            "run_4_jobs_if_170_minutes_or_2107_pair_if_100_minutes_else_skip", "generation_complete_marker"],
            ports=PORTS, experiment_cutoff_utc=CUTOFF.isoformat(), hard_deadline_utc=DEADLINE.isoformat(),
            no_pod_stop=True, novel_writer_priority_after_utc=args.prioritize_novel_writer_after_utc,
            maximum_confirmation_jobs=args.max_confirmation_jobs,
            **reservation)), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "followups.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = OUT / "generation_complete.json"
        if marker.exists():
            print(json.dumps(dict(event="generation_already_complete", marker=str(marker))), flush=True)
            return 0
        state = dict(started_utc=now().isoformat(), pid=os.getpid(), events=[],
                     experiment_cutoff_utc=CUTOFF.isoformat(), hard_deadline_utc=DEADLINE.isoformat(),
                     novel_writer_priority_after_utc=args.prioritize_novel_writer_after_utc,
                     maximum_confirmation_jobs=args.max_confirmation_jobs, **reservation)

        def event(name, **details):
            item = dict(utc=now().isoformat(), event=name, **details)
            state["events"].append(item)
            atomic(OUT / "followups_state.json", state)
            print(json.dumps(item), flush=True)

        env = dict(os.environ, PYTHONPATH="src:.", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WANDB_DISABLED="true")
        python = sys.executable
        source_stopped = False
        sampler_terminal = False
        outcome = "failed"
        confirmation_run = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            analysis_path = update_analysis([SOURCE, SAMPLER])
            analysis = read(analysis_path)
            analysis["gpu_feature_transfer_reservation"] = reservation
            atomic(analysis_path, analysis)
            event("gpu_feature_window_reserved", **reservation)
            source_ok = wait_source(event, args.poll_seconds)
            source_stopped = True
            existing_sampler = existing_sampler_outcome(event, args.poll_seconds)
            sampler_terminal = True
            if not source_ok:
                outcome = "aborted_incomplete_source_suite"
                return 1
            if minutes_remaining() < 31:
                outcome = "completed_source_no_time_for_followups"
                event("followups_skipped", reason="insufficient_time_for_sampler")
                return 0
            sampler_state = ROOT / "results/next_experiments/sampler_execution_state.json"
            if existing_sampler is None:
                sampler_terminal = False
                sampler_code = run_stage("rank_sampler", [python, "scripts/run_overnight_parallel.py", "--execute",
                    "--manifest", str(SAMPLER), "--phase", "all", "--state", str(sampler_state),
                    "--ports", *map(str, PORTS)], event, env)
                existing_sampler = existing_sampler_outcome(event, args.poll_seconds)
                sampler_terminal = True
            else:
                sampler_code = 0 if len(existing_sampler["complete_jobs"]) == existing_sampler["expected_jobs"] else 1
                event("sampler_controller_not_relaunched", exit_code=sampler_code,
                      reason="existing_terminal_full_or_partial_results_preserved")
            event("sampler_control_completed", exit_code=sampler_code,
                  full_budget_jobs=len(existing_sampler["complete_jobs"]) if existing_sampler else 0)
            if novel_writer_priority_due(args.prioritize_novel_writer_after_utc):
                outcome = "completed_9b_novel_glm_prioritized_for_remaining_writer_window"
                event("confirmation_skipped", reason="prospective_wall_clock_priority_for_previously_untested_glm",
                      threshold_utc=args.prioritize_novel_writer_after_utc,
                      source_and_sampler_preserved=True, held_out_scores_used=False)
                return 0
            if confirmation_minutes_remaining(args.reserve_transfer_minutes) < 100:
                outcome = "completed_9b_confirmation_skipped_for_time"
                event("confirmation_skipped", reason="less_than_100_minutes_before_reserved_confirmation_cutoff")
                return 0
            if not CONFIRMATION.exists():
                code = run_stage("prepare_27b_confirmation", [python, "scripts/overnight_prepare_confirmation.py", "--prepare"], event, env, timeout=180)
                if code:
                    raise RuntimeError("Conditional confirmation preparation failed")
            else:
                if read(CONFIRMATION)["selection_provenance"]["source_manifest_sha256"] != sha(SOURCE):
                    raise ValueError("Existing confirmation provenance differs from source manifest")
                event("reusing_immutable_confirmation_manifest", sha256=sha(CONFIRMATION))
            stop_servers(ROOT / "out/overnight20261021/qwen9b_ready_servers.json", "Qwen/Qwen3.5-9B", REV9, event)
            if not start_27b(event, env, args.server_ready_minutes, args.reserve_transfer_minutes):
                outcome = "completed_9b_27b_startup_failed_or_timed_out"
                return 0
            count = confirmation_job_count(confirmation_minutes_remaining(args.reserve_transfer_minutes), args.max_confirmation_jobs)
            if count == 0:
                event("confirmation_skipped", reason="startup_left_less_than_100_minutes")
                outcome = "completed_9b_confirmation_skipped_after_startup"
                return 0
            confirmation_run = CONFIRMATION if count == 4 else subset_confirmation(CONFIRMATION)
            analysis = update_analysis([SOURCE, SAMPLER, confirmation_run])
            event("confirmation_budget_selected", jobs=count,
                  maximum_confirmation_jobs=args.max_confirmation_jobs,
                  remaining_minutes=confirmation_minutes_remaining(args.reserve_transfer_minutes),
                  manifest=str(confirmation_run), manifest_sha256=sha(confirmation_run), analysis_manifest=str(analysis),
                  **reservation)
            code = run_stage("27b_confirmation", confirmation_controller_command(confirmation_run, args.reserve_transfer_minutes),
                             event, env)
            outcome = "completed" if code == 0 else "completed_with_partial_or_failed_confirmation"
            return 0 if code == 0 else 1
        except BaseException as exc:
            event("followups_error", error=repr(exc))
            raise
        finally:
            if source_stopped and not sampler_terminal:
                try:
                    existing_sampler_outcome(event, min(args.poll_seconds, 5))
                    sampler_terminal = True
                except Exception as exc:
                    event("sampler_shutdown_not_certified", error=repr(exc))
            if source_stopped and sampler_terminal:
                for name, model, revision in [("qwen9b_ready", "Qwen/Qwen3.5-9B", REV9),
                                               ("qwen27b", "Qwen/Qwen3.8-27B", REV27)]:
                    try:
                        stop_servers(ROOT / f"out/overnight20261021/{name}_servers.json", model, revision, event)
                    except Exception as exc:
                        event("managed_server_cleanup_failed", model=model, error=repr(exc))
                payload = dict(completed_utc=now().isoformat(), generation_complete=True, outcome=outcome,
                    source_manifest_sha256=sha(SOURCE), sampler_manifest_sha256=sha(SAMPLER),
                    confirmation_execution_manifest=str(confirmation_run) if confirmation_run else None,
                    experiment_cutoff_utc=CUTOFF.isoformat(), hard_deadline_utc=DEADLINE.isoformat(),
                    analysis_manifest="results/overnight20261021/analysis_manifest.json", pod_stopped=False, **reservation)
                atomic(marker, payload)
                event("generation_complete", outcome=outcome, marker=str(marker))
            else:
                event("generation_marker_withheld", reason="source_or_sampler_controller_not_confirmed_stopped",
                      source_stopped=source_stopped, sampler_terminal=sampler_terminal)


if __name__ == "__main__":
    raise SystemExit(main())
