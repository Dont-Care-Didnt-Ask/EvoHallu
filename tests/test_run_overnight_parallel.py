"""Controller protocol checks; no writer server or language model is invoked."""
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_overnight_parallel.py"
SPEC = importlib.util.spec_from_file_location("overnight_controller", SCRIPT)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def job(name, phase="evolution_screen", policy="original", seed=2107, requests=2):
    return dict(id=name, phase=phase, prompt_policy=policy, seed=seed, requests=requests,
                expected_logs="logs/" + name, command=[sys.executable, "unused.py"])


def completed(root, item):
    folder = root / item["expected_logs"]
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(dict(status="finished", requests=item["requests"])))
    (folder / "candidates.jsonl").write_text("{}\n" * item["requests"])
    (folder / "frozen_estimator.py").write_text("# selected estimator\n")


def test_four_policies_are_balanced_per_seed_before_feedback():
    jobs = [job("feedback", "feedback_control", "neutral", 1)]
    for seed in (2, 1):
        for policy in reversed(controller.POLICIES):
            phase = "compositional_control" if policy == "compositional" else "evolution_screen"
            jobs.append(job(f"{seed}-{policy}", phase, policy, seed))
    jobs.append(job("screen", "prompt_screen"))
    batches = controller.ordered_batches(dict(jobs=jobs, seeds=[1, 2]))
    assert [label for label, _ in batches] == ["prompt_screen", "evolution_four_policies", "feedback_control"]
    assert [j["id"] for j in batches[1][1]] == [f"{s}-{p}" for s in (1, 2) for p in controller.POLICIES]
    assert [j["id"] for _, selected in controller.ordered_batches(dict(jobs=jobs), "prompt_screen") for j in selected] == ["screen"]


def test_duplicate_output_directories_are_rejected():
    first, second = job("a"), job("b")
    second["expected_logs"] = first["expected_logs"]
    with pytest.raises(ValueError, match="same experiment log"):
        controller.ordered_batches(dict(jobs=[first, second]))


def test_deadline_cannot_be_extended_and_requires_utc():
    assert controller.parse_deadline("2026-10-22T02:16:28Z") == controller.HARD_DEADLINE
    for value in ["2026-10-22T02:16:29Z", "2026-10-22T02:16:28", "2026-10-22T02:16:28+03:00"]:
        with pytest.raises(ValueError):
            controller.parse_deadline(value)


def test_completed_full_budget_skips_but_partial_or_corrupt_ledger_fails(tmp_path):
    item = job("complete")
    assert controller.inspect_job(item, tmp_path) == "pending"
    completed(tmp_path, item)
    assert controller.inspect_job(item, tmp_path) == "complete"
    folder = tmp_path / item["expected_logs"]
    (folder / "candidates.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="candidate ledger"):
        controller.inspect_job(item, tmp_path)
    (folder / "summary.json").write_text(json.dumps(dict(status="finished", requests=1)))
    with pytest.raises(ValueError, match="Partial job"):
        controller.inspect_job(item, tmp_path)


def test_partial_directory_without_summary_fails(tmp_path):
    item = job("partial")
    folder = tmp_path / item["expected_logs"]
    folder.mkdir(parents=True)
    (folder / "trial_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Interrupted job directory"):
        controller.inspect_job(item, tmp_path)


def test_verify_inputs_checks_audited_bytes_and_frozen_contexts(tmp_path):
    directory = tmp_path / "study"
    directory.mkdir()
    (directory / "readiness.json").write_text(json.dumps(dict(cpu_gate_passed=True, all_ten_supplied_csvs_identical=True)))
    names = ["raw.pt", "data/claims/popqa/popqa_train_filtered_10shot.csv", "source.py", "configs/example.yaml"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    contexts = directory / "frozen_contexts.json"
    contexts.write_text("{}")
    manifest = dict(training_raw_path="raw.pt", training_raw_sha256=controller.sha(tmp_path / names[0]),
                    training_csv_sha256=controller.sha(tmp_path / names[1]),
                    study_source_sha256={"source.py": controller.sha(tmp_path / "source.py")},
                    contexts_sha256=controller.sha(contexts))
    item = dict(id="example", config="example", config_sha256=controller.sha(tmp_path / "configs/example.yaml"))
    controller.verify_inputs(manifest, directory / "run_manifest.json", [item], tmp_path)
    (tmp_path / "source.py").write_text("changed")
    with pytest.raises(ValueError, match="source.py"):
        controller.verify_inputs(manifest, directory / "run_manifest.json", [item], tmp_path)


def test_commands_use_replica_endpoint_without_mutating_manifest():
    item = job("a")
    before = list(item["command"])
    command = controller.child_command(item, 8001, 5400)
    assert command[-2:] == ["cluster.api_base_url=http://127.0.0.1:8001/v1", "trial.max_seconds=5400"]
    assert item["command"] == before


def test_execute_records_failure_and_continues_other_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "verify_inputs", lambda *a, **k: None)
    monkeypatch.setattr(controller, "verify_servers", lambda *a, **k: None)
    original_inspect = controller.inspect_job
    monkeypatch.setattr(controller, "inspect_job", lambda item: original_inspect(item, tmp_path))
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    items = [job("bad"), job("good"), job("later")]
    for item in items:
        folder = tmp_path / item["expected_logs"]
        if item["id"] == "bad":
            source = "raise SystemExit(7)"
        else:
            source = ("from pathlib import Path; import json; "
                      f"p=Path({str(folder)!r}); p.mkdir(parents=True); "
                      "(p/'summary.json').write_text(json.dumps(dict(status='finished', requests=2))); "
                      "(p/'candidates.jsonl').write_text('{}\\n{}\\n'); "
                      "(p/'frozen_estimator.py').write_text('# selected')")
        item["command"] = [sys.executable, "-c", source]
    manifest = dict(jobs=items, writer="test-writer")
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    args = argparse.Namespace(state=None, deadline_utc=controller.utcnow() + dt.timedelta(hours=2),
                              reserve_minutes=30, ports=[8000, 8001], min_job_minutes=30, max_job_minutes=90)
    code = controller.execute(manifest, path, controller.ordered_batches(manifest), args)
    state = json.loads((tmp_path / "parallel_execution_state.json").read_text())
    assert code == 1
    assert state["jobs"]["bad"]["status"] == "failed_or_partial"
    assert all(state["jobs"][name]["status"] == "complete" for name in ["good", "later"])
    assert state["running"] == []
    assert len([e for e in state["events"] if e["event"] == "job_started"]) == 3


def test_atomic_json_replaces_complete_document(tmp_path):
    path = tmp_path / "nested/state.json"
    controller.atomic_json(path, {"events": [1]})
    controller.atomic_json(path, {"events": [1, 2]})
    assert json.loads(path.read_text()) == {"events": [1, 2]}
    assert list(path.parent.iterdir()) == [path]
