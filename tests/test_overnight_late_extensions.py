"""Scheduling/deadline protocol checks; no GPU or inference work is performed."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("late_extensions", SCRIPTS / "overnight_late_extensions.py")
late = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(late)


def test_glm_requires_55_minutes_before_writer_cutoff(monkeypatch):
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=55))
    assert late.can_start_glm()
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=55) + dt.timedelta(seconds=1))
    assert not late.can_start_glm()


def test_extension_requires_valid_full_budget_pairs_and_measured_runtime():
    pilot = [dict(full_budget=True, valid_candidates=10, elapsed_seconds=2400)] * 2
    gate = late.extension_gate(pilot, 56)
    assert gate["required_minutes"] == 57
    assert not gate["run"]
    assert late.extension_gate(pilot, 57)["run"]
    for field, value in [("full_budget", False), ("valid_candidates", 0), ("elapsed_seconds", None)]:
        bad = [dict(pilot[0]), dict(pilot[1], **{field: value})]
        assert not late.extension_gate(bad, 200)["run"]
    assert not late.extension_gate(pilot[:1], 200)["run"]


def test_pair_commands_have_distinct_states_and_01_10_cutoff():
    commands = [late.pair_command(seed) for seed in (2107, 4319)]
    assert commands[0][commands[0].index("--state") + 1] != commands[1][commands[1].index("--state") + 1]
    for command in commands:
        assert command[command.index("--max-job-minutes") + 1] == "45"
        assert command[command.index("--min-job-minutes") + 1] == "30"
        deadline = dt.datetime.fromisoformat(command[command.index("--deadline-utc") + 1])
        reserve = float(command[command.index("--reserve-minutes") + 1])
        assert deadline - dt.timedelta(minutes=reserve) == late.GLM_CUTOFF


def test_opt_in_60_minute_glm_cap_keeps_cutoff_reserve_and_job_identity():
    for seed in (2107, 4319):
        default, extended = late.pair_command(seed), late.pair_command(seed, 60)
        cap_index = default.index("--max-job-minutes") + 1
        assert default[cap_index] == "45" and extended[cap_index] == "60"
        assert default[:cap_index] == extended[:cap_index]
        assert default[cap_index + 1:] == extended[cap_index + 1:]
        deadline = dt.datetime.fromisoformat(extended[extended.index("--deadline-utc") + 1])
        reserve = float(extended[extended.index("--reserve-minutes") + 1])
        assert deadline - dt.timedelta(minutes=reserve) == late.GLM_CUTOFF


@pytest.mark.parametrize("extra,expected", [([], 45), (["--glm-max-job-minutes", "60", "--prefer-devstral-breadth"], 60)])
def test_glm_cap_dry_run_is_default_45_and_opt_in_60(monkeypatch, capsys, extra, expected):
    monkeypatch.setattr(sys, "argv", ["overnight_late_extensions.py", *extra])
    assert late.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["event"] == "dry_run" and plan["glm_max_job_minutes"] == expected
    assert plan["glm_cutoff_utc"] == late.GLM_CUTOFF.isoformat()
    assert plan["transfer_cutoff_utc"] == late.TRANSFER_CUTOFF.isoformat()
    assert plan["hard_deadline_utc"] == late.HARD_DEADLINE.isoformat()
    if expected == 60:
        assert plan["prefer_devstral_breadth"] is True


@pytest.mark.parametrize("value", ["30", "90", "61"])
def test_glm_cap_rejects_unplanned_values(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["overnight_late_extensions.py", "--glm-max-job-minutes", value])
    with pytest.raises(SystemExit) as exc:
        late.main()
    assert exc.value.code == 2


def test_merge_adds_only_launched_pair_with_generalization_phase(tmp_path):
    analysis = tmp_path / "results/overnight20261021/analysis_manifest.json"
    analysis.parent.mkdir(parents=True)
    analysis.write_text(json.dumps(dict(jobs=[dict(id="existing", phase="evolution_screen", requests=60)], execution_manifests=[])))
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps(dict(writer=late.glm.WRITER, jobs=[dict(id=f"2107_{p}", phase="evolution_screen", requests=60)
        for p in ["original", "neutral"]])))
    late.merge_glm_pair(pair, tmp_path)
    result = json.loads(analysis.read_text())
    assert len(result["jobs"]) == 3
    assert all(row["phase"] == "writer_generalization" for row in result["jobs"] if row["id"] != "existing")
    assert all("4319" not in row["id"] for row in result["jobs"])
    assert result["execution_manifests"][0]["sha256"] == late.common.sha(pair)


def test_nine_requires_complete_receipt_and_all_six_preserved(tmp_path):
    path = tmp_path / "receipt.json"
    assert late.coverage_from_receipt(path) == "six"
    receipt = dict(status="partial_or_failed", new_caches=[{}, {}, {}], preserved_historical_caches=[{}] * 6,
                   all_six_historical_caches_unchanged=True)
    path.write_text(json.dumps(receipt))
    assert late.coverage_from_receipt(path) == "six"
    receipt["status"] = "complete_three_missing_caches"
    path.write_text(json.dumps(receipt))
    assert late.coverage_from_receipt(path) == "nine"
    receipt["all_six_historical_caches_unchanged"] = False
    path.write_text(json.dumps(receipt))
    assert late.coverage_from_receipt(path) == "six"


def test_no_inference_when_startup_consumes_pilot_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    (tmp_path / "results/next_experiments").mkdir(parents=True)
    (tmp_path / "results/next_experiments/glm47_manifest.json").write_text("{}")
    current = [late.GLM_CUTOFF - dt.timedelta(minutes=56)]
    monkeypatch.setattr(late, "now", lambda: current[0])
    calls = []
    class Runner:
        def run(self, name, command, **kwargs):
            calls.append(name)
            current[0] += dt.timedelta(minutes=12)
            return dict(exit_code=0)
    result = late.optional_glm(Runner(), lambda *a, **k: None)
    assert calls == ["start_glm"]
    assert "less_than_45_minutes" in result["reason"]


def test_startup_failure_unblocks_cpu_driver_and_preserves_six_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    monkeypatch.setattr(late, "OUT", tmp_path / "results/overnight20261021")
    monkeypatch.setattr(sys, "argv", ["overnight_late_extensions.py", "--execute"])
    monkeypatch.setattr(late.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(late, "wait_marker", lambda *a, **k: True)
    def fail_writer(*a, **k):
        raise RuntimeError("synthetic startup failure")
    monkeypatch.setattr(late, "optional_glm", fail_writer)
    monkeypatch.setattr(late.common, "stop_servers", lambda *a, **k: None)
    monkeypatch.setattr(late.glm, "telemetry", lambda: {})
    def busy(*a, **k):
        raise RuntimeError("synthetic occupied GPU")
    monkeypatch.setattr(late.glm, "require_idle", busy)
    calls = []
    class Runner:
        def __init__(self, event):
            pass
        def run(self, name, command, **kwargs):
            calls.append((name, command))
            return dict(exit_code=0, timed_out=False)
    monkeypatch.setattr(late, "Runner", Runner)
    monkeypatch.setattr(late, "final_analyses", lambda *a, **k: {})
    assert late.main() == 0
    decision = json.loads((late.OUT / "late_writer_decision.json").read_text())
    outcome = json.loads((late.OUT / "late_extensions_complete.json").read_text())
    assert decision["decision_complete"] and decision["writer"]["status"] == "failed"
    assert outcome["target_cache_coverage"] == "six"
    command = next(command for name, command in calls if name == "transfer_evaluation")
    assert command[command.index("--coverage") + 1] == "six"


def test_termination_reaps_nested_process_group(tmp_path):
    pid_path = tmp_path / "nested.pid"
    source = ("import pathlib,subprocess,sys,time; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
              f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(30)")
    child = subprocess.Popen([sys.executable, "-c", source], start_new_session=True)
    try:
        until = time.monotonic() + 3
        # File creation and writing are separate operations in the child.
        while (not pid_path.exists() or not pid_path.read_text()) and time.monotonic() < until:
            time.sleep(.01)
        assert pid_path.exists() and pid_path.read_text()
        nested = int(pid_path.read_text())
        assert nested in late.child_groups(child.pid)
        late.terminate_tree(child)
        until = time.monotonic() + 2
        while late.common.process_alive(nested) and time.monotonic() < until:
            time.sleep(.01)
        assert not late.common.process_alive(nested)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            late.terminate_tree(child)


def ready_devstral(tmp_path):
    receipt = tmp_path / "results/overnight20261021/devstral/prefetch_receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(status="ready_native_checkpoint", model=late.DEVSTRAL_WRITER,
        revision=late.DEVSTRAL_REVISION, path="/workspace/models/Devstral-Small-2-24B")))
    helper = tmp_path / "scripts/overnight_devstral.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("# Test fixture only; no inference.\n")
    return receipt


def test_devstral_reservation_is_opt_in_and_requires_ready_native_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=100))
    assert late.devstral_breadth_gate(False, tmp_path)["reason"] == "breadth_flag_disabled"
    assert not late.devstral_breadth_gate(True, tmp_path)["run"]
    receipt = ready_devstral(tmp_path)
    gate = late.devstral_breadth_gate(True, tmp_path)
    assert gate["run"] and gate["prefetch_receipt_sha256"] == late.common.sha(receipt)
    (tmp_path / "scripts/overnight_devstral.py").unlink()
    assert late.devstral_breadth_gate(True, tmp_path)["reason"] == "devstral_launcher_unavailable"


@pytest.mark.parametrize("field,value", [
    ("status", "download_in_progress"), ("model", "another/model"),
    ("revision", "unverified"), ("path", "relative/checkpoint"), ("path", None),
])
def test_devstral_reservation_rejects_bad_receipts(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=100))
    path = ready_devstral(tmp_path)
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    gate = late.devstral_breadth_gate(True, tmp_path)
    assert not gate["run"] and gate["reason"] == "native_prefetch_identity_not_ready"


def test_devstral_reservation_rejects_unreadable_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=100))
    ready_devstral(tmp_path).write_text("{invalid")
    assert late.devstral_breadth_gate(True, tmp_path)["reason"] == "native_prefetch_receipt_unreadable"


def test_devstral_reservation_75_minute_boundary(tmp_path, monkeypatch):
    ready_devstral(tmp_path)
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=75))
    assert late.devstral_breadth_gate(True, tmp_path)["run"]
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=75) + dt.timedelta(seconds=1))
    assert late.devstral_breadth_gate(True, tmp_path)["reason"] == "less_than_75_minutes_before_writer_cutoff"


def stub_glm(tmp_path, monkeypatch, pilot_exit=0, valid=True):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    folder = tmp_path / "results/next_experiments"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("glm47_manifest.json", "glm47_seed2107_manifest.json", "glm47_seed4319_manifest.json"):
        (folder / name).write_text("{}")
    monkeypatch.setattr(late, "merge_glm_pair", lambda *a, **k: None)
    rows = [dict(full_budget=valid, valid_candidates=10 if valid else 0, elapsed_seconds=600)] * 2
    monkeypatch.setattr(late, "pilot_results", lambda *a, **k: rows)
    calls = []
    class Runner:
        def run(self, name, command, **kwargs):
            calls.append(name)
            return dict(exit_code=pilot_exit if name == "glm_pilot_2107" else 0)
    return Runner(), calls


@pytest.mark.parametrize("pilot_exit,valid", [(0, True), (1, False)])
def test_ready_new_writer_reservation_skips_glm_extension_independently_of_pilot_quality(
        tmp_path, monkeypatch, pilot_exit, valid):
    ready_devstral(tmp_path)
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=100))
    runner, calls = stub_glm(tmp_path, monkeypatch, pilot_exit, valid)
    result = late.optional_glm(runner, lambda *a, **k: None, prefer_devstral_breadth=True)
    assert calls == ["start_glm", "glm_pilot_2107"]
    assert result["devstral_reservation"]["run"]
    assert result["extension_skipped_reason"] == "reserved_capacity_for_new_devstral_writer"


@pytest.mark.parametrize("enabled,ready,minutes", [(False, True, 100), (True, False, 100), (True, True, 74)])
def test_no_devstral_reservation_preserves_existing_glm_extension_rule(
        tmp_path, monkeypatch, enabled, ready, minutes):
    if ready:
        ready_devstral(tmp_path)
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=minutes))
    runner, calls = stub_glm(tmp_path, monkeypatch)
    result = late.optional_glm(runner, lambda *a, **k: None, prefer_devstral_breadth=enabled)
    assert calls == ["start_glm", "glm_pilot_2107", "glm_extension_4319"]
    assert result["status"] == "both_pairs_finished"
    if not enabled:
        assert "devstral_reservation" not in result


def test_glm_60_minute_cap_reaches_both_pairs_without_changing_extension_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=100))
    runner, calls = stub_glm(tmp_path, monkeypatch)
    original_pair_command, commands = late.pair_command, []
    def record_pair(seed, max_job_minutes):
        command = original_pair_command(seed, max_job_minutes)
        commands.append(command)
        return command
    monkeypatch.setattr(late, "pair_command", record_pair)
    result = late.optional_glm(runner, lambda *a, **k: None, glm_max_job_minutes=60)
    assert calls == ["start_glm", "glm_pilot_2107", "glm_extension_4319"]
    assert result["max_job_minutes"] == 60 and result["extension_gate"]["required_minutes"] == 45
    assert len(commands) == 2
    assert all(command[command.index("--max-job-minutes") + 1] == "60" for command in commands)


def test_writer_orchestration_passes_cap_and_keeps_breadth_option(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    monkeypatch.setattr(late, "OUT", tmp_path / "results/overnight20261021")
    calls = []
    def glm_stage(runner, event, **kwargs):
        calls.append(kwargs)
        return dict(status="pilot_finished", max_job_minutes=kwargs["glm_max_job_minutes"])
    monkeypatch.setattr(late, "optional_glm", glm_stage)
    monkeypatch.setattr(late, "optional_devstral", lambda *a, **k: dict(status="skipped", reason="capacity"))
    monkeypatch.setattr(late.common, "stop_servers", lambda *a, **k: None)
    result = late.complete_writer_decision(None, lambda *a, **k: None, True,
        prefer_devstral_breadth=True, glm_max_job_minutes=60)
    assert calls == [dict(prefer_devstral_breadth=True, glm_max_job_minutes=60)]
    marker = json.loads((late.OUT / "late_writer_decision.json").read_text())
    assert marker["writer"]["max_job_minutes"] == result["writer"]["max_job_minutes"] == 60
    assert marker["prefer_devstral_breadth"] is True


def test_devstral_launch_uses_same_writer_cutoff_and_rechecks_capacity(tmp_path, monkeypatch):
    ready_devstral(tmp_path)
    monkeypatch.setattr(late, "ROOT", tmp_path)
    monkeypatch.setattr(late, "OUT", tmp_path / "results/overnight20261021")
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=100))
    calls = []
    class Runner:
        def run(self, name, command, **kwargs):
            calls.append((name, command, kwargs))
            return dict(exit_code=0)
    decision = late.optional_devstral(Runner(), lambda *a, **k: None)
    assert decision["status"] == "launcher_finished"
    name, command, kwargs = calls[0]
    assert name == "devstral_breadth_2107"
    assert command[command.index("--deadline-utc") + 1] == late.GLM_CUTOFF.isoformat()
    assert kwargs["deadline"] == late.GLM_CUTOFF
    monkeypatch.setattr(late, "now", lambda: late.GLM_CUTOFF - dt.timedelta(minutes=74))
    assert late.optional_devstral(Runner(), lambda *a, **k: None)["status"] == "skipped"
    assert len(calls) == 1


@pytest.mark.parametrize("glm_fails,devstral_fails", [(False, False), (True, False), (False, True)])
def test_late_marker_follows_both_writer_cleanups_including_failures(
        tmp_path, monkeypatch, glm_fails, devstral_fails):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    monkeypatch.setattr(late, "OUT", tmp_path / "results/overnight20261021")
    marker = late.OUT / "late_writer_decision.json"
    calls = []
    def glm_stage(*a, **k):
        assert not marker.exists()
        calls.append("glm")
        if glm_fails:
            raise RuntimeError("synthetic GLM startup failure")
        return dict(status="pilot_finished")
    def devstral_stage(*a, **k):
        assert not marker.exists() and calls[-1] == "cleanup_glm"
        calls.append("devstral")
        if devstral_fails:
            raise RuntimeError("synthetic Devstral failure")
        return dict(status="launcher_finished")
    def cleanup(path, model, revision, event):
        assert not marker.exists()
        calls.append("cleanup_glm" if model == late.glm.WRITER else "cleanup_devstral")
    monkeypatch.setattr(late, "optional_glm", glm_stage)
    monkeypatch.setattr(late, "optional_devstral", devstral_stage)
    monkeypatch.setattr(late.common, "stop_servers", cleanup)
    def event(name, **details):
        if name == "late_writer_decision_complete":
            assert marker.exists() and calls[-1] == "cleanup_devstral"
    result = late.complete_writer_decision(None, event, True, prefer_devstral_breadth=True)
    receipt = json.loads(marker.read_text())
    assert calls == ["glm", "cleanup_glm", "devstral", "cleanup_devstral"]
    assert receipt["decision_complete"] and receipt["prefer_devstral_breadth"]
    assert receipt["devstral"] == result["devstral"]
    assert (result["writer"]["status"] == "failed") == glm_fails
    assert (result["devstral"]["status"] == "failed") == devstral_fails


def test_no_new_writer_when_glm_cleanup_is_not_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    monkeypatch.setattr(late, "OUT", tmp_path / "results/overnight20261021")
    monkeypatch.setattr(late, "optional_glm", lambda *a, **k: dict(status="pilot_finished"))
    def no_devstral(*a, **k):
        raise AssertionError("Must not launch a second writer while cleanup failed")
    monkeypatch.setattr(late, "optional_devstral", no_devstral)
    def cleanup(path, model, revision, event):
        if model == late.glm.WRITER:
            raise RuntimeError("synthetic identity mismatch")
    monkeypatch.setattr(late.common, "stop_servers", cleanup)
    result = late.complete_writer_decision(None, lambda *a, **k: None, True, prefer_devstral_breadth=True)
    assert result["devstral"]["reason"] == "GLM_cleanup_not_confirmed"
    assert result["errors"][0]["stage"] == "GLM_cleanup"


def test_default_writer_marker_retains_glm_schema_without_devstral(tmp_path, monkeypatch):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    monkeypatch.setattr(late, "OUT", tmp_path / "results/overnight20261021")
    monkeypatch.setattr(late, "optional_glm", lambda *a, **k: dict(status="pilot_finished"))
    models = []
    monkeypatch.setattr(late.common, "stop_servers", lambda path, model, *a: models.append(model))
    result = late.complete_writer_decision(None, lambda *a, **k: None, True)
    receipt = json.loads((late.OUT / "late_writer_decision.json").read_text())
    assert "devstral" not in result and "devstral" not in receipt
    assert "prefer_devstral_breadth" not in receipt
    assert models == [late.glm.WRITER]


@pytest.mark.parametrize("interrupted_stage", ["glm", "devstral"])
def test_interrupted_writer_cleans_up_before_cpu_marker(tmp_path, monkeypatch, interrupted_stage):
    monkeypatch.setattr(late, "ROOT", tmp_path)
    monkeypatch.setattr(late, "OUT", tmp_path / "results/overnight20261021")
    marker = late.OUT / "late_writer_decision.json"
    calls = []
    def glm_stage(*a, **k):
        calls.append("glm")
        if interrupted_stage == "glm":
            raise KeyboardInterrupt("synthetic interruption")
        return dict(status="pilot_finished")
    def devstral_stage(*a, **k):
        calls.append("devstral")
        raise KeyboardInterrupt("synthetic interruption")
    def cleanup(path, model, revision, event):
        assert not marker.exists()
        calls.append("cleanup_glm" if model == late.glm.WRITER else "cleanup_devstral")
    monkeypatch.setattr(late, "optional_glm", glm_stage)
    monkeypatch.setattr(late, "optional_devstral", devstral_stage)
    monkeypatch.setattr(late.common, "stop_servers", cleanup)
    with pytest.raises(KeyboardInterrupt):
        late.complete_writer_decision(None, lambda *a, **k: None, True, prefer_devstral_breadth=True)
    assert marker.exists()
    expected = ["glm", "cleanup_glm"]
    if interrupted_stage == "devstral":
        expected += ["devstral", "cleanup_devstral"]
    assert calls == expected
    receipt = json.loads(marker.read_text())
    interrupted = receipt["writer"] if interrupted_stage == "glm" else receipt["devstral"]
    assert interrupted["status"] == "interrupted"
