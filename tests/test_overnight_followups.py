"""Synthetic followup scheduling checks; never starts a GPU server."""
import importlib.util
import datetime as dt
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/overnight_followups.py"
SPEC = importlib.util.spec_from_file_location("overnight_followups", SCRIPT)
followups = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(followups)


def test_controller_stop_must_follow_latest_start_and_have_no_running_jobs():
    events = [dict(event="controller_started"), dict(event="controller_stopped")]
    assert followups.controller_stopped(dict(events=events, running=[]))
    assert not followups.controller_stopped(dict(events=events + [dict(event="controller_started")], running=[]))
    assert not followups.controller_stopped(dict(events=events, running=[dict(pid=1)]))
    assert not followups.controller_stopped(dict(events=[dict(event="controller_stopped")]))


@pytest.mark.parametrize("remaining,expected", [(200, 4), (170, 4), (169.999, 2), (100, 2), (99.999, 0), (-1, 0)])
def test_confirmation_budget_thresholds(remaining, expected):
    assert followups.confirmation_job_count(remaining) == expected


def test_prospective_breadth_cap_keeps_fixed_pair_and_time_floor():
    assert followups.confirmation_job_count(200, 2) == 2
    assert followups.confirmation_job_count(100, 2) == 2
    assert followups.confirmation_job_count(99, 2) == 0
    with pytest.raises(ValueError, match="cap"):
        followups.confirmation_job_count(200, 3)


def test_novel_writer_priority_is_opt_in_and_clock_only():
    threshold = "2026-10-21T22:45:00Z"
    before = dt.datetime.fromisoformat("2026-10-21T22:44:59+00:00")
    after = dt.datetime.fromisoformat("2026-10-21T22:45:00+00:00")
    assert not followups.novel_writer_priority_due(None, after)
    assert not followups.novel_writer_priority_due(threshold, before)
    assert followups.novel_writer_priority_due(threshold, after)
    with pytest.raises(ValueError, match="UTC"):
        followups.novel_writer_priority_due("2026-10-21T22:45:00+03:00", after)
    with pytest.raises(ValueError, match="UTC"):
        followups.novel_writer_priority_due("2026-10-21T22:45:00", after)


def test_full_budget_requires_matching_ledger_and_estimator(tmp_path):
    item = dict(expected_logs="logs/run", requests=2)
    folder = tmp_path / item["expected_logs"]
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(dict(status="finished", requests=2)))
    assert not followups.full_budget(item, tmp_path)
    (folder / "candidates.jsonl").write_text("{}\n{}\n")
    (folder / "frozen_estimator.py").write_text("# selected")
    assert followups.full_budget(item, tmp_path)
    (folder / "candidates.jsonl").write_text("{}\n")
    assert not followups.full_budget(item, tmp_path)


def test_analysis_phase_mapping_preserves_original_manifest_hashes(tmp_path):
    source = tmp_path / "source.json"
    sampler = tmp_path / "sampler.json"
    confirmation = tmp_path / "confirmation.json"
    original = dict(id="compositional", phase="compositional_control", requests=60)
    rank = dict(id="rank", phase="evolution_screen", requests=60, parent_sampling="rank_uniform")
    confirm = dict(id="27b", phase="evolution_screen", requests=60)
    for path, writer, jobs in [(source, "Qwen/Qwen3.5-9B", [original]),
                               (sampler, "Qwen/Qwen3.5-9B", [rank]),
                               (confirmation, "Qwen/Qwen3.8-27B", [confirm])]:
        path.write_text(json.dumps(dict(writer=writer, jobs=jobs)))
    original_hashes = {path: followups.sha(path) for path in [source, sampler, confirmation]}
    path = followups.update_analysis([source, sampler, confirmation], tmp_path)
    analysis = json.loads(path.read_text())
    rows = {row["id"]: row for row in analysis["jobs"]}
    assert rows["compositional"]["phase"] == "evolution_screen"
    assert rows["compositional"]["execution_phase"] == "compositional_control"
    assert rows["rank"]["phase"] == "sampler_control"
    assert rows["27b"]["phase"] == "writer_confirmation"
    assert rows["27b"]["writer_model"] == "Qwen/Qwen3.8-27B"
    assert analysis["requests_by_phase"] == dict(evolution_screen=60, sampler_control=60, writer_confirmation=60)
    assert all(followups.sha(path) == digest for path, digest in original_hashes.items())
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="execution manifest changed"):
        followups.update_analysis([source], tmp_path)


def test_reduced_confirmation_is_exact_2107_pair_and_pins_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(followups, "ROOT", tmp_path)
    path = tmp_path / "confirmation27b_manifest.json"
    manifest = dict(jobs=[dict(id=f"{s}_{p}", seed=s, phase="evolution_screen", requests=60)
                         for s in [2107, 4319] for p in ["original", "neutral"]])
    path.write_text(json.dumps(manifest))
    output = followups.subset_confirmation(path)
    subset = json.loads(output.read_text())
    assert len(subset["jobs"]) == 2
    assert {j["seed"] for j in subset["jobs"]} == {2107}
    assert subset["parent_confirmation_manifest_sha256"] == followups.sha(path)
    assert followups.subset_confirmation(path) == output
    output.write_text("{}")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        followups.subset_confirmation(path)


def test_interface_analysis_stays_separate_and_survives_later_merge(tmp_path):
    source = tmp_path / "source.json"
    interface = tmp_path / "interface.json"
    writer = "Qwen/Qwen3.5-9B"
    source.write_text(json.dumps(dict(writer=writer, jobs=[dict(
        id="neutral", phase="evolution_screen", prompt_policy="neutral", requests=60)])))
    interface.write_text(json.dumps(dict(writer=writer, jobs=[dict(
        id="neutral_contract", phase="evolution_screen", analysis_phase="interface_control",
        prompt_policy="neutral", requests=60)])))
    result = followups.update_analysis([source, interface], tmp_path)
    followups.update_analysis([source], tmp_path)
    value = json.loads(result.read_text())
    rows = {row["id"]: row for row in value["jobs"]}
    assert rows["neutral"]["phase"] == "evolution_screen"
    assert rows["neutral_contract"]["phase"] == "interface_control"
    assert rows["neutral_contract"]["execution_phase"] == "evolution_screen"
    assert value["requests_by_phase"] == dict(evolution_screen=60, interface_control=60)


def test_server_metadata_wrong_model_refused_before_signals(tmp_path):
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(dict(model="wrong", revision=followups.REV9)))
    with pytest.raises(ValueError, match="model/revision mismatch"):
        followups.validated_servers(path, "Qwen/Qwen3.5-9B", followups.REV9, tmp_path)


def test_server_metadata_invalid_pid_refused(tmp_path):
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(dict(model="Qwen/Qwen3.5-9B", revision=followups.REV9,
        servers=[dict(pid=1, port=18000), dict(pid=1, port=18001)])))
    with pytest.raises(ValueError, match="Invalid managed server PID"):
        followups.validated_servers(path, "Qwen/Qwen3.5-9B", followups.REV9, tmp_path)


def test_transfer_reserve_moves_only_confirmation_cutoff():
    assert followups.confirmation_cutoff().isoformat() == "2026-10-22T01:01:28+00:00"
    assert followups.confirmation_cutoff(0) == followups.CUTOFF
    assert followups.confirmation_cutoff(60).isoformat() == "2026-10-22T00:46:28+00:00"
    assert followups.CUTOFF.isoformat() == "2026-10-22T01:46:28+00:00"
    reservation = followups.transfer_reservation()
    assert reservation["reserved_gpu_feature_transfer_minutes"] == 45
    assert reservation["planned_gpu_feature_window"]["end_utc"] == followups.CUTOFF.isoformat()


def test_confirmation_cli_deadline_subtracts_controller_reserve(monkeypatch):
    monkeypatch.setattr(followups, "now", lambda: dt.datetime(2026, 10, 21, 21, tzinfo=dt.timezone.utc))
    for reserve in (0, 45, 60):
        command = followups.confirmation_controller_command(Path("confirmation.json"), reserve)
        deadline = dt.datetime.fromisoformat(command[command.index("--deadline-utc") + 1])
        controller_reserve = float(command[command.index("--reserve-minutes") + 1])
        assert deadline - dt.timedelta(minutes=controller_reserve) == followups.confirmation_cutoff(reserve)
        assert deadline <= followups.DEADLINE
        assert command[-3:] == ["--ports", "18000", "18001"]


def test_reserved_time_changes_confirmation_job_selection(monkeypatch):
    # 200 min until global cutoff, but only 155 min until the default 27B cutoff.
    monkeypatch.setattr(followups, "now", lambda: followups.CUTOFF - dt.timedelta(minutes=200))
    assert followups.confirmation_job_count(followups.minutes_remaining()) == 4
    assert followups.confirmation_job_count(followups.confirmation_minutes_remaining()) == 2


def test_no_confirmation_launch_after_reserved_cutoff(monkeypatch):
    monkeypatch.setattr(followups, "now", lambda: followups.confirmation_cutoff() + dt.timedelta(seconds=1))
    monkeypatch.setattr(followups.subprocess, "Popen", lambda *a, **k: pytest.fail("No child may start after cutoff"))
    with pytest.raises(TimeoutError, match="no confirmation job may be launched"):
        followups.confirmation_controller_command(Path("confirmation.json"))
    events = []
    assert not followups.start_27b(lambda name, **kwargs: events.append((name, kwargs)), {}, 25)
    assert any(name == "confirmation_skipped" for name, _ in events)


def test_terminal_inference_requires_dead_controller_and_all_recorded_children(monkeypatch):
    state = dict(controller_pid=10, running=[dict(pid=20)], events=[dict(event="controller_started")])
    monkeypatch.setattr(followups, "process_alive", lambda pid: pid == 20)
    assert followups.terminal_evidence(state) is None
    monkeypatch.setattr(followups, "process_alive", lambda pid: False)
    assert followups.terminal_evidence(state) == "recorded_controller_and_children_no_longer_alive"
    assert followups.terminal_evidence(dict(events=[])) is None


def test_controller_stop_after_cutoff_is_observed_without_extending_generation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    path = tmp_path / "state.json"
    state = dict(controller_pid=10, running=[dict(pid=20)], manifest_sha256=followups.sha(manifest),
                 events=[dict(event="controller_started")])
    path.write_text(json.dumps(state))
    current = [followups.CUTOFF + dt.timedelta(seconds=30)]
    monkeypatch.setattr(followups, "now", lambda: current[0])
    monkeypatch.setattr(followups, "process_alive", lambda pid: True)
    def stop_after_poll(seconds):
        current[0] += dt.timedelta(seconds=seconds)
        state["running"] = []
        state["events"].append(dict(event="controller_stopped"))
        path.write_text(json.dumps(state))
    monkeypatch.setattr(followups.time, "sleep", stop_after_poll)
    events = []
    result = followups.wait_controller_terminal(path, manifest, lambda name, **kw: events.append((name, kw)), 1)
    assert followups.controller_stopped(result)
    assert current[0] > followups.CUTOFF
    assert events[0][1]["no_generation_budget_extension"] is True


def test_existing_overlap_sampler_is_observed_to_stop_and_partial_results_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(followups, "ROOT", tmp_path)
    manifest = tmp_path / "sampler.json"
    manifest.write_text(json.dumps(dict(jobs=[dict(id="good"), dict(id="partial"), dict(id="bad_ledger")])))
    monkeypatch.setattr(followups, "SAMPLER", manifest)
    path = tmp_path / "results/next_experiments/sampler_execution_state.json"
    path.parent.mkdir(parents=True)
    state = dict(controller_pid=10, running=[dict(pid=20)], manifest_sha256=followups.sha(manifest),
                 events=[dict(event="controller_started")])
    path.write_text(json.dumps(state))
    monkeypatch.setattr(followups, "now", lambda: followups.CUTOFF - dt.timedelta(minutes=1))
    monkeypatch.setattr(followups, "process_alive", lambda pid: True)
    def stop(seconds):
        state["running"] = []
        state["events"].append(dict(event="controller_stopped"))
        path.write_text(json.dumps(state))
    monkeypatch.setattr(followups.time, "sleep", stop)
    def full(job):
        if job["id"] == "bad_ledger":
            raise ValueError("synthetic malformed ledger")
        return job["id"] == "good"
    monkeypatch.setattr(followups, "full_budget", full)
    result = followups.existing_sampler_outcome(lambda *a, **k: None, 1)
    assert result["existing_controller_terminal"]
    assert result["complete_jobs"] == ["good"]
    assert len(result["artifact_errors"]) == 1


def test_active_sampler_prevents_writer_shutdown_and_generation_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(followups, "ROOT", tmp_path)
    monkeypatch.setattr(followups, "OUT", tmp_path / "outcomes")
    monkeypatch.setattr(followups.sys, "argv", ["overnight_followups.py", "--execute"])
    monkeypatch.setattr(followups.signal, "signal", lambda *a, **k: None)
    analysis = tmp_path / "analysis.json"
    analysis.write_text("{}")
    monkeypatch.setattr(followups, "update_analysis", lambda *a, **k: analysis)
    monkeypatch.setattr(followups, "wait_source", lambda *a, **k: False)
    def still_active(*a, **k):
        raise TimeoutError("synthetic sampler still owns writer replicas")
    monkeypatch.setattr(followups, "existing_sampler_outcome", still_active)
    monkeypatch.setattr(followups, "stop_servers", lambda *a, **k: pytest.fail("Must not stop an active sampler's model"))
    with pytest.raises(TimeoutError, match="still owns"):
        followups.main()
    assert not (followups.OUT / "generation_complete.json").exists()
    events = json.loads((followups.OUT / "followups_state.json").read_text())["events"]
    assert events[-1]["event"] == "generation_marker_withheld"


def test_terminal_source_and_sampler_publish_marker_even_after_generation_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(followups, "ROOT", tmp_path)
    monkeypatch.setattr(followups, "OUT", tmp_path / "outcomes")
    monkeypatch.setattr(followups.sys, "argv", ["overnight_followups.py", "--execute"])
    monkeypatch.setattr(followups.signal, "signal", lambda *a, **k: None)
    analysis = tmp_path / "analysis.json"
    analysis.write_text("{}")
    source, sampler = tmp_path / "source.json", tmp_path / "sampler.json"
    source.write_text("{}")
    sampler.write_text("{}")
    monkeypatch.setattr(followups, "SOURCE", source)
    monkeypatch.setattr(followups, "SAMPLER", sampler)
    monkeypatch.setattr(followups, "update_analysis", lambda *a, **k: analysis)
    monkeypatch.setattr(followups, "wait_source", lambda *a, **k: False)
    monkeypatch.setattr(followups, "existing_sampler_outcome", lambda *a, **k: dict(complete_jobs=[], expected_jobs=3))
    monkeypatch.setattr(followups, "stop_servers", lambda *a, **k: None)
    monkeypatch.setattr(followups, "now", lambda: followups.CUTOFF + dt.timedelta(seconds=35))
    assert followups.main() == 1
    marker = json.loads((followups.OUT / "generation_complete.json").read_text())
    assert marker["generation_complete"] is True
    assert marker["outcome"] == "aborted_incomplete_source_suite"
