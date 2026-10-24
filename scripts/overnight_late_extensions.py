"""Optional late writer and feature-transfer work, bounded by this night's budget.

Dry run by default. Runs only on the authorized pod; never stops a pod, creates
an archive, or compresses caches. The independent guards retain the hard rental
deadline. All child work stops by the earlier CPU finish deadline.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import overnight_followups as common
import overnight_start_glm as glm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/overnight20261021"
GLM_CUTOFF = dt.datetime.fromisoformat("2026-10-22T01:10:00+00:00")
TRANSFER_CUTOFF = dt.datetime.fromisoformat("2026-10-22T02:01:28+00:00")
CPU_FINISH = dt.datetime.fromisoformat("2026-10-22T02:06:28+00:00")
HARD_DEADLINE = dt.datetime.fromisoformat("2026-10-22T02:16:28+00:00")
PORTS = [18000, 18001]
DEVSTRAL_WRITER = "mistralai/Devstral-Small-2-24B-Instruct-2512"
DEVSTRAL_REVISION = "55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128"
DEVSTRAL_ENTRY_MINUTES = 75


def now():
    return dt.datetime.now(dt.timezone.utc)


def remaining(deadline):
    return (deadline - now()).total_seconds()


def can_start_glm():
    return remaining(GLM_CUTOFF) >= 55 * 60


def pilot_results(manifest, root=ROOT):
    rows = []
    for job in manifest["jobs"]:
        folder = root / job["expected_logs"]
        summary = common.read(folder / "summary.json") if (folder / "summary.json").exists() else {}
        full = common.full_budget(job, root)
        records = [json.loads(line) for line in (folder / "candidates.jsonl").read_text().splitlines() if line.strip()] if full else []
        elapsed = summary.get("elapsed_seconds")
        valid_time = isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed > 0
        rows.append(dict(job=job["id"], full_budget=full,
            valid_candidates=sum("score" in row.get("val", {}) for row in records),
            elapsed_seconds=elapsed if valid_time else None, requests=summary.get("requests"),
            summary_sha256=common.sha(folder / "summary.json") if summary else None))
    return rows


def extension_gate(pilot, remaining_minutes):
    eligible = (len(pilot) == 2 and all(row["full_budget"] and row["valid_candidates"] > 0
                and row["elapsed_seconds"] is not None for row in pilot))
    required = max(45., 1.3 * max(row["elapsed_seconds"] for row in pilot) / 60 + 5) if eligible else None
    return dict(eligible=eligible, required_minutes=required, remaining_minutes=remaining_minutes,
                run=bool(eligible and remaining_minutes >= required))


def devstral_breadth_gate(enabled, root=None):
    """Reserve a different writer using identity/readiness and time, never quality.

    The Devstral launcher independently verifies checkpoint contents, idle GPUs,
    free ports, and the time left after startup. This gate does not certify that
    inference has run or that a prefetched checkpoint can be served successfully.
    """
    root = ROOT if root is None else root
    minutes = remaining(GLM_CUTOFF) / 60
    gate = dict(run=False, enabled=bool(enabled), remaining_minutes=minutes,
                required_minutes=DEVSTRAL_ENTRY_MINUTES,
                writer_cutoff_utc=GLM_CUTOFF.isoformat(),
                model=DEVSTRAL_WRITER, revision=DEVSTRAL_REVISION)
    if not enabled:
        return dict(gate, reason="breadth_flag_disabled")
    path = root / "results/overnight20261021/devstral/prefetch_receipt.json"
    if not path.exists():
        return dict(gate, reason="native_prefetch_receipt_absent")
    try:
        receipt = common.read(path)
        identity_ready = (isinstance(receipt, dict)
            and receipt.get("status") == "ready_native_checkpoint"
            and receipt.get("model") == DEVSTRAL_WRITER
            and receipt.get("revision") == DEVSTRAL_REVISION
            and isinstance(receipt.get("path"), str)
            and Path(receipt["path"]).is_absolute())
    except (OSError, ValueError, TypeError) as exc:
        return dict(gate, reason="native_prefetch_receipt_unreadable", error=repr(exc))
    gate["prefetch_receipt_sha256"] = common.sha(path)
    if not identity_ready:
        return dict(gate, reason="native_prefetch_identity_not_ready")
    if not (root / "scripts/overnight_devstral.py").is_file():
        return dict(gate, reason="devstral_launcher_unavailable")
    if minutes < DEVSTRAL_ENTRY_MINUTES:
        return dict(gate, reason="less_than_75_minutes_before_writer_cutoff")
    return dict(gate, run=True, reason="ready_native_new_writer_and_capacity",
                model_path=receipt["path"])


def merge_glm_pair(path, root=ROOT):
    """Add only the pair about to launch, preserving execution identities."""
    target = root / "results/overnight20261021/analysis_manifest.json"
    analysis, manifest = common.read(target), common.read(path)
    if manifest.get("writer") != glm.WRITER or len(manifest["jobs"]) != 2:
        raise ValueError("Expected an immutable two-job GLM pair manifest")
    jobs = {row["id"]: row for row in analysis["jobs"]}
    relative, digest = str(path.relative_to(root)), common.sha(path)
    provenance = {row["path"]: row for row in analysis.get("execution_manifests", [])}
    if relative in provenance and provenance[relative]["sha256"] != digest:
        raise ValueError("A previously recorded GLM execution manifest changed")
    provenance[relative] = dict(path=relative, sha256=digest)
    for original in manifest["jobs"]:
        row = copy.deepcopy(original)
        row.update(phase="writer_generalization", execution_phase=original["phase"],
            execution_manifest=relative, execution_manifest_sha256=digest, writer_model=glm.WRITER)
        if row["id"] in jobs and jobs[row["id"]] != row:
            raise ValueError("Refusing to change an existing GLM analysis job identity")
        jobs[row["id"]] = row
    totals = {}
    for row in jobs.values():
        totals[row["phase"]] = totals.get(row["phase"], 0) + row["requests"]
    analysis.update(jobs=list(jobs.values()), execution_manifests=list(provenance.values()),
                    requests_by_phase=totals, updated_utc=now().isoformat())
    common.atomic(target, analysis)


def child_groups(pid):
    """Collect only process groups descended from a subprocess we launched."""
    result = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid="], capture_output=True, text=True,
                            check=True, timeout=5)
    rows = [tuple(map(int, row.split())) for row in result.stdout.splitlines() if row.strip()]
    owned = {pid}
    while True:
        descendants = {child for child, parent, _ in rows if parent in owned}
        expanded = owned | descendants
        if expanded == owned:
            break
        owned = expanded
    return {group for child, _, group in rows if child in owned and group in owned} | {pid}


def terminate_tree(child):
    groups = child_groups(child.pid) if child.poll() is None else {child.pid}
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


class Runner:
    def __init__(self, event):
        self.event = event
        self.env = dict(os.environ, PYTHONPATH="src:.", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WANDB_DISABLED="true")

    def run(self, name, command, deadline=CPU_FINISH, max_seconds=None):
        # Leave time to discover nested sessions and terminate/reap them before
        # the CPU finish deadline, rather than beginning cleanup at that deadline.
        deadline = min(deadline, CPU_FINISH - dt.timedelta(seconds=20))
        available = remaining(deadline)
        if available <= 0:
            self.event("stage_skipped", stage=name, reason="deadline_reached")
            return dict(exit_code=None, timed_out=True, skipped=True)
        timeout = min(available, max_seconds) if max_seconds else available
        log = OUT / "late_logs" / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.event("stage_started", stage=name, command=command, deadline_utc=deadline.isoformat(), log=str(log))
        started = time.monotonic()
        with log.open("a") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
                result = dict(exit_code=code, timed_out=False, skipped=False)
            except subprocess.TimeoutExpired:
                terminate_tree(child)
                result = dict(exit_code=child.returncode, timed_out=True, skipped=False)
            except BaseException:
                terminate_tree(child)
                raise
        result["elapsed_seconds"] = time.monotonic() - started
        self.event("stage_finished", stage=name, **result)
        return result


def pair_command(seed, max_job_minutes=45):
    return [sys.executable, "scripts/run_overnight_parallel.py", "--execute", "--phase", "all",
            "--manifest", f"results/next_experiments/glm47_seed{seed}_manifest.json",
            "--state", f"results/next_experiments/glm47_{'pilot' if seed == 2107 else 'extension'}_state.json",
            "--ports", *map(str, PORTS), "--max-job-minutes", str(max_job_minutes), "--min-job-minutes", "30",
            "--deadline-utc", "2026-10-22T01:40:00+00:00", "--reserve-minutes", "30"]


def optional_glm(runner, event, prefer_devstral_breadth=False, glm_max_job_minutes=45):
    decision = dict(model=glm.WRITER, revision=glm.REVISION, status="skipped", pilot=None, extension=None)
    if glm_max_job_minutes != 45:
        decision["max_job_minutes"] = glm_max_job_minutes
    if not can_start_glm():
        decision["reason"] = "less_than_55_minutes_before_01_10_writer_cutoff"
        return decision
    full = ROOT / "results/next_experiments/glm47_manifest.json"
    if not full.exists():
        result = runner.run("prepare_glm", [sys.executable, "scripts/overnight_prepare_glm.py", "--prepare"],
                            deadline=GLM_CUTOFF, max_seconds=180)
        if result["exit_code"] != 0:
            decision.update(status="preparation_failed", preparation=result)
            return decision
    result = runner.run("start_glm", [sys.executable, "scripts/overnight_start_glm.py", "--execute",
        "--model-path", "/dev/shm/overnight-models/GLM-4.7-Flash", "--ports", *map(str, PORTS),
        "--ready-timeout-minutes", "10"], deadline=GLM_CUTOFF, max_seconds=720)
    decision["startup"] = result
    if result["exit_code"] != 0:
        decision.update(status="startup_failed", reason="GLM servers did not become ready")
        return decision
    if remaining(GLM_CUTOFF) < 45 * 60:
        decision["reason"] = "startup_left_less_than_45_minutes_before_writer_cutoff"
        return decision
    path = ROOT / "results/next_experiments/glm47_seed2107_manifest.json"
    merge_glm_pair(path)
    decision["pilot"] = runner.run("glm_pilot_2107", pair_command(2107, glm_max_job_minutes), deadline=GLM_CUTOFF)
    pilot = pilot_results(common.read(path))
    decision["pilot_jobs"] = pilot
    gate = extension_gate(pilot, remaining(GLM_CUTOFF) / 60)
    decision["extension_gate"] = gate
    event("glm_extension_gate", **gate)
    decision["status"] = "pilot_finished" if decision["pilot"]["exit_code"] == 0 else "pilot_partial_or_failed"
    if prefer_devstral_breadth:
        reservation = devstral_breadth_gate(True)
        decision["devstral_reservation"] = reservation
        event("devstral_breadth_reservation", **reservation)
        if reservation["run"]:
            decision["extension_skipped_reason"] = "reserved_capacity_for_new_devstral_writer"
            return decision
    if decision["pilot"]["exit_code"] == 0 and gate["run"]:
        path = ROOT / "results/next_experiments/glm47_seed4319_manifest.json"
        merge_glm_pair(path)
        decision["extension"] = runner.run("glm_extension_4319", pair_command(4319, glm_max_job_minutes), deadline=GLM_CUTOFF)
        decision["extension_jobs"] = pilot_results(common.read(path))
        decision["status"] = "both_pairs_finished" if decision["extension"]["exit_code"] == 0 else "extension_partial_or_failed"
    return decision


def optional_devstral(runner, event):
    """Run the separately owned launcher only while its fixed capacity gate holds."""
    gate = devstral_breadth_gate(True)
    event("devstral_breadth_entry_gate", **gate)
    decision = dict(model=DEVSTRAL_WRITER, revision=DEVSTRAL_REVISION,
                    status="skipped", entry_gate=gate)
    if not gate["run"]:
        decision["reason"] = gate["reason"]
        return decision
    result = runner.run("devstral_breadth_2107", [sys.executable,
        "scripts/overnight_devstral.py", "--execute", "--deadline-utc", GLM_CUTOFF.isoformat()],
        deadline=GLM_CUTOFF)
    decision.update(status="launcher_finished" if result["exit_code"] == 0 else "launcher_partial_or_failed",
                    execution=result)
    receipt = OUT / "devstral/decision.json"
    if receipt.exists():
        decision["launcher_decision"] = common.read(receipt)
        decision["launcher_decision_sha256"] = common.sha(receipt)
    return decision


def complete_writer_decision(runner, event, observed, prefer_devstral_breadth=False, glm_max_job_minutes=45):
    """Publish the CPU-driver marker only after both optional writers clean up."""
    result = dict(writer=None, errors=[])
    if prefer_devstral_breadth:
        result["devstral"] = dict(model=DEVSTRAL_WRITER, revision=DEVSTRAL_REVISION,
                                  status="skipped", reason="not_started" if observed else "source_marker_not_observed")
    glm_cleaned = False
    try:
        try:
            if observed:
                result["writer"] = (optional_glm(runner, event,
                    prefer_devstral_breadth=prefer_devstral_breadth, glm_max_job_minutes=glm_max_job_minutes)
                    if prefer_devstral_breadth or glm_max_job_minutes != 45 else optional_glm(runner, event))
            else:
                result["writer"] = dict(status="skipped", reason="source_marker_wait_expired")
        except Exception as exc:
            result["writer"] = dict(status="failed", error=repr(exc))
            result["errors"].append(dict(stage="optional_glm", error=repr(exc)))
            event("optional_writer_error", error=repr(exc))
        except BaseException as exc:
            result["writer"] = dict(status="interrupted", error=repr(exc))
            raise
        finally:
            try:
                common.stop_servers(ROOT / "out/overnight20261021/glm47_servers.json",
                                    glm.WRITER, glm.REVISION, event)
                glm_cleaned = True
            except Exception as exc:
                result["errors"].append(dict(stage="GLM_cleanup", error=repr(exc)))
                event("glm_cleanup_failed", error=repr(exc))
        if prefer_devstral_breadth and observed:
            try:
                if glm_cleaned:
                    # This covers GLM preparation/startup/pilot failures too. The
                    # gate uses only the ready identity and time left, not scores.
                    result["devstral"] = optional_devstral(runner, event)
                else:
                    result["devstral"].update(reason="GLM_cleanup_not_confirmed")
            except Exception as exc:
                result["devstral"] = dict(model=DEVSTRAL_WRITER, revision=DEVSTRAL_REVISION,
                                          status="failed", error=repr(exc))
                result["errors"].append(dict(stage="optional_devstral", error=repr(exc)))
                event("optional_devstral_error", error=repr(exc))
            except BaseException as exc:
                result["devstral"] = dict(model=DEVSTRAL_WRITER, revision=DEVSTRAL_REVISION,
                                          status="interrupted", error=repr(exc))
                raise
            finally:
                try:
                    common.stop_servers(ROOT / "out/overnight20261021/devstral_servers.json",
                                        DEVSTRAL_WRITER, DEVSTRAL_REVISION, event)
                except Exception as exc:
                    result["errors"].append(dict(stage="Devstral_cleanup", error=repr(exc)))
                    event("devstral_cleanup_failed", error=repr(exc))
    finally:
        marker = dict(completed_utc=now().isoformat(), decision_complete=True,
                      writer=result["writer"], errors=result["errors"],
                      writer_cutoff_utc=GLM_CUTOFF.isoformat())
        if prefer_devstral_breadth:
            marker["devstral"] = result["devstral"]
            marker["prefer_devstral_breadth"] = True
        decision_path = OUT / "late_writer_decision.json"
        common.atomic(decision_path, marker)
        event("late_writer_decision_complete", marker=str(decision_path))
    return result


def coverage_from_receipt(path):
    if not path.exists():
        return "six"
    receipt = common.read(path)
    complete = (receipt.get("status") == "complete_three_missing_caches"
                and receipt.get("all_six_historical_caches_unchanged") is True
                and len(receipt.get("new_caches", [])) == 3
                and len(receipt.get("preserved_historical_caches", [])) == 6)
    return "nine" if complete else "six"


def wait_marker(path, deadline, poll_seconds, event):
    event("waiting_for_marker", path=str(path), deadline_utc=deadline.isoformat())
    while remaining(deadline) > 0:
        if path.exists():
            event("marker_observed", path=str(path), sha256=common.sha(path))
            return True
        time.sleep(min(poll_seconds, max(.1, remaining(deadline))))
    event("marker_wait_expired", path=str(path))
    return False


def final_analyses(runner, event):
    results = {}
    commands = [
        ("final_report", "scripts/overnight_report.py", ["--manifest", str(OUT / "analysis_manifest.json")]),
        ("prompt_screen_analysis", "scripts/overnight_prompt_screen_analysis.py", []),
        ("failure_audit", "scripts/overnight_failure_audit.py", ["--manifest", str(OUT / "analysis_manifest.json")]),
    ]
    for name, script, extra in commands:
        if not (ROOT / script).exists() or remaining(CPU_FINISH) < 10:
            event("analysis_skipped", stage=name, reason="unavailable_script_or_deadline")
            continue
        results[name] = runner.run(name, [sys.executable, script, *extra], max_seconds=min(120, remaining(CPU_FINISH)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--prefer-devstral-breadth", action="store_true",
                        help="Prefer one ready new Devstral writer pair over a second GLM seed when >=75 minutes remain")
    parser.add_argument("--glm-max-job-minutes", type=int, choices=(45, 60), default=45,
                        help="Prospective GLM job time cap; 60 improves full-request completion chances without extending the writer cutoff")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be in [1, 60]")
    if not args.execute:
        stages = ["wait_source_generation", "optional_GLM_pilot_then_conditional_extension",
            "late_writer_decision_marker", "optional_three_missing_transfer_caches", "wait_source_CPU_completion",
            "CPU_transfer_scoring_nine_only_with_complete_receipt_otherwise_six", "final_analyses"]
        if args.prefer_devstral_breadth:
            stages[2:2] = ["conditional_ready_Devstral_pair_after_GLM_cleanup"]
        print(json.dumps(dict(event="dry_run", stages=stages,
            prefer_devstral_breadth=args.prefer_devstral_breadth,
            glm_max_job_minutes=args.glm_max_job_minutes,
            glm_cutoff_utc=GLM_CUTOFF.isoformat(), transfer_cutoff_utc=TRANSFER_CUTOFF.isoformat(),
            cpu_finish_utc=CPU_FINISH.isoformat(), hard_deadline_utc=HARD_DEADLINE.isoformat(), no_pod_stop=True)), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "late_extensions.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "late_extensions_complete.json").exists():
            print(json.dumps(dict(event="late_extensions_already_complete")), flush=True)
            return 0
        state = dict(started_utc=now().isoformat(), pid=os.getpid(), events=[],
                     prefer_devstral_breadth=args.prefer_devstral_breadth,
                     glm_max_job_minutes=args.glm_max_job_minutes,
                     glm_cutoff_utc=GLM_CUTOFF.isoformat(), transfer_cutoff_utc=TRANSFER_CUTOFF.isoformat(),
                     cpu_finish_utc=CPU_FINISH.isoformat(), hard_deadline_utc=HARD_DEADLINE.isoformat())

        def event(name, **details):
            item = dict(utc=now().isoformat(), event=name, **details)
            state["events"].append(item)
            common.atomic(OUT / "late_extensions_state.json", state)
            print(json.dumps(item), flush=True)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        runner = Runner(event)
        outcome = dict(status="running", writer=None, feature_completion=None, transfer_evaluation=None,
                       final_analyses={}, errors=[], pod_stopped=False)
        decision_path = OUT / "late_writer_decision.json"
        try:
            observed = wait_marker(OUT / "generation_complete.json", CPU_FINISH - dt.timedelta(minutes=20),
                                   args.poll_seconds, event)
            outcome["source_generation_marker_observed"] = observed
            writers = complete_writer_decision(runner, event, observed,
                args.prefer_devstral_breadth, args.glm_max_job_minutes)
            outcome["writer"] = writers["writer"]
            outcome["errors"].extend(writers["errors"])
            if args.prefer_devstral_breadth:
                outcome["devstral"] = writers["devstral"]
            receipt = OUT / "remaining_transfer_completion.json"
            if coverage_from_receipt(receipt) == "nine":
                outcome["feature_completion"] = dict(status="already_complete_nine", receipt_sha256=common.sha(receipt))
            elif remaining(HARD_DEADLINE) >= 50 * 60 and remaining(TRANSFER_CUTOFF) > 0:
                try:
                    glm.require_idle(glm.telemetry(), [0, 1])
                    outcome["feature_completion"] = runner.run("complete_transfer_features",
                        [sys.executable, "scripts/prepare_remaining_transfer.py", "--execute", "--gpu-indices", "0", "1",
                         "--max-minutes", "35", "--deadline-utc", TRANSFER_CUTOFF.isoformat(),
                         "--cache-dir", "data/restored/mod1-transfer-cache"], deadline=TRANSFER_CUTOFF, max_seconds=2400)
                except Exception as exc:
                    outcome["feature_completion"] = dict(status="failed_or_skipped_fallback_six", error=repr(exc))
                    event("feature_completion_failed_fallback_six", error=repr(exc))
            else:
                outcome["feature_completion"] = dict(status="skipped_fallback_six", reason="less_than_50_minutes_before_hard_deadline")
                event("feature_completion_skipped", **outcome["feature_completion"])
            coverage = coverage_from_receipt(receipt)
            outcome["target_cache_coverage"] = coverage
            cpu_done = wait_marker(OUT / "cpu_complete.json", CPU_FINISH - dt.timedelta(minutes=3), args.poll_seconds, event)
            outcome["source_cpu_complete_observed"] = cpu_done
            scoring_seconds = min(1200, int(remaining(CPU_FINISH)) - 150)
            if cpu_done and scoring_seconds >= 60:
                result = runner.run("transfer_evaluation", [sys.executable, "scripts/overnight_transfer_eval.py",
                    "--execute", "--coverage", coverage, "--workers", "4", "--bootstrap", "500",
                    "--max-seconds", str(scoring_seconds), "--cache-dir", "data/restored/mod1-transfer-cache"],
                    deadline=CPU_FINISH - dt.timedelta(seconds=120), max_seconds=scoring_seconds + 15)
                outcome["transfer_evaluation"] = dict(coverage=coverage, **result)
                if coverage == "nine" and result["exit_code"] != 0 and remaining(CPU_FINISH) >= 240:
                    event("nine_transfer_scoring_failed_trying_prespecified_six")
                    budget = min(1200, int(remaining(CPU_FINISH)) - 150)
                    outcome["transfer_evaluation_fallback_six"] = runner.run("transfer_evaluation_fallback_six",
                        [sys.executable, "scripts/overnight_transfer_eval.py", "--execute", "--coverage", "six",
                         "--workers", "4", "--bootstrap", "500", "--max-seconds", str(budget)],
                        deadline=CPU_FINISH - dt.timedelta(seconds=120), max_seconds=budget + 15)
            else:
                outcome["transfer_evaluation"] = dict(status="skipped_or_pending", coverage=coverage,
                    reason="source_CPU_not_complete_or_scoring_deadline")
            outcome["final_analyses"] = final_analyses(runner, event)
            outcome["status"] = "finished_with_recorded_stage_outcomes"
        except BaseException as exc:
            outcome.update(status="interrupted_or_failed", error=repr(exc))
            event("late_extensions_error", error=repr(exc))
            raise
        finally:
            # Even a launcher failure or interruption must unblock the persistent
            # CPU driver. The independent source-generation marker is still required.
            if not decision_path.exists():
                common.atomic(decision_path, dict(completed_utc=now().isoformat(), decision_complete=True,
                    writer=outcome["writer"], error=outcome.get("error", "interrupted_before_writer_decision")))
            outcome["completed_utc"] = now().isoformat()
            common.atomic(OUT / "late_extensions_complete.json", outcome)
            event("late_extensions_complete", status=outcome["status"])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
