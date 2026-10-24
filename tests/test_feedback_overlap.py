import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import overnight_followups as common
from scripts import run_feedback_overlap as overlap

ROOT=Path(__file__).resolve().parents[1]


def prepared(tmp_path):
    source_path=tmp_path/"results/next_experiments/run_manifest.json"
    source_path.parent.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/"results/next_experiments/run_manifest.json").read_text())
    source_path.write_text(json.dumps(source))
    path=source_path.with_name("feedback_overlap_manifest.json")
    manifest=overlap.prepare_manifest(source_path,path,tmp_path)
    return source_path,path,source,manifest


def test_manifest_is_exact_original_subset_immutable_and_does_not_rewrite_configs(tmp_path):
    source_path,path,source,manifest=prepared(tmp_path)
    expected=[job for job in source["jobs"] if job["phase"]=="feedback_control"]
    assert manifest["jobs"]==expected and manifest["additional_requests"]==0
    assert manifest["parent_source_manifest_sha256"]==common.sha(source_path)
    original=path.read_bytes()
    assert overlap.prepare_manifest(source_path,path,tmp_path)==manifest and path.read_bytes()==original
    changed=copy.deepcopy(manifest);changed["jobs"][0]["prompt_policy"]="diverse"
    with pytest.raises(ValueError,match="exact original"):
        common.validate_feedback_overlap_manifest(changed,source,common.sha(source_path))
    assert not (tmp_path/"configs").exists()


def test_all_queued_jobs_reserved_main_inspector_refuses_and_owned_hook_keeps_partial_protections(tmp_path):
    _,path,_,manifest=prepared(tmp_path)
    original=overlap.controller.inspect_job
    with overlap.owned_feedback_directories(manifest,common.sha(path),tmp_path):
        for job in manifest["jobs"]:
            assert overlap.controller.inspect_job(job,tmp_path)=="pending"
            # Model the separate original controller, whose inspect hook is not patched.
            with pytest.raises(ValueError,match="Interrupted job"):
                original(job,tmp_path)
        queued=manifest["jobs"][-1]
        folder=tmp_path/queued["expected_logs"]
        # The third job is protected while queued, before its child exists.
        assert (folder/overlap.MARKER).exists()
        (folder/"trial_config.json").write_text("{}")
        with pytest.raises(ValueError,match="Interrupted job"):
            overlap.controller.inspect_job(queued,tmp_path)
        (folder/"summary.json").write_text(json.dumps(dict(status="finished",requests=60)))
        (folder/"candidates.jsonl").write_text("{}\n"*60)
        (folder/"frozen_estimator.py").write_text("# selected synthetic detector")
        assert original(queued,tmp_path)==overlap.controller.inspect_job(queued,tmp_path)=="complete"
    assert overlap.controller.inspect_job is original
    # A failed-before-start reservation leaves the first two directories empty.
    assert all(not any((tmp_path/job["expected_logs"]).iterdir()) for job in manifest["jobs"][:2])
    assert (folder/overlap.MARKER).exists()


def test_wait_source_waits_for_overlap_before_checking_all_original_nineteen_ledgers(tmp_path,monkeypatch):
    source_path,path,source,manifest=prepared(tmp_path)
    monkeypatch.setattr(common,"ROOT",tmp_path);monkeypatch.setattr(common,"SOURCE",source_path)
    main_state=source_path.with_name("parallel_execution_state.json")
    main_state.write_text(json.dumps(dict(manifest_sha256=common.sha(source_path),running=[],controller_pid=100,
        events=[dict(event="controller_started"),dict(event="controller_stopped")],unfinished_jobs=[j["id"] for j in manifest["jobs"]])))
    monkeypatch.setattr(common,"SOURCE_STATE",main_state)
    aux=source_path.with_name("feedback_overlap_state.json")
    state=dict(manifest_sha256=common.sha(path),controller_pid=200,running=[dict(pid=201)],events=[dict(event="controller_started")])
    aux.write_text(json.dumps(state))
    monkeypatch.setattr(common,"process_alive",lambda pid:True)
    monkeypatch.setattr(common,"now",lambda:common.CUTOFF-common.dt.timedelta(minutes=10))
    completed=False
    calls=[]
    def finish_overlap(seconds):
        nonlocal completed
        state["running"]=[];state["events"].append(dict(event="controller_stopped"))
        aux.write_text(json.dumps(state));completed=True
    def full(job):
        assert completed,"No original final budget check may precede overlap terminal"
        calls.append(job["id"])
        return True
    monkeypatch.setattr(common.time,"sleep",finish_overlap)
    monkeypatch.setattr(common,"full_budget",full)
    events=[]
    assert common.wait_source(lambda name,**kw:events.append((name,kw)),1)
    assert set(calls)=={j["id"] for j in source["jobs"]}
    overlap_event=next(v for name,v in events if name=="existing_feedback_overlap_outcome")
    assert overlap_event["expected_jobs"]==3 and len(overlap_event["complete_jobs"])==3


def test_prepared_unlaunched_overlap_does_not_block_and_changed_parent_is_refused(tmp_path,monkeypatch):
    source_path,path,source,manifest=prepared(tmp_path)
    monkeypatch.setattr(common,"ROOT",tmp_path);monkeypatch.setattr(common,"SOURCE",source_path)
    assert common.existing_feedback_overlap_outcome(lambda *a,**kw:None,1) is None
    path.with_name("feedback_overlap_state.json").write_text("{}")
    changed=copy.deepcopy(manifest);changed["parent_source_manifest_sha256"]="0"*64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="exact original"):
        common.existing_feedback_overlap_outcome(lambda *a,**kw:None,1)


def test_launch_requires_adaptive_slack_and_prior_rank_interface_owners_terminal(tmp_path,monkeypatch):
    source_path,path,source,manifest=prepared(tmp_path)
    monkeypatch.setattr(common,"process_alive",lambda pid:True)
    state=dict(manifest_sha256=common.sha(source_path),controller_pid=10,events=[dict(event="controller_started")],running=[],ports=[18000,18001])
    assert overlap.verify_early_source_window(source,state,tmp_path)["unstarted_adaptive_jobs"]==12
    state["events"].append(dict(event="batch_started",batch="feedback_control"))
    with pytest.raises(ValueError,match="already reached"):
        overlap.verify_early_source_window(source,state,tmp_path)
    state["events"].pop()
    rank_manifest=path.with_name("sampler_manifest.json");rank_manifest.write_text("{}")
    rank_state=path.with_name("sampler_execution_state.json")
    rank=dict(manifest_sha256=common.sha(rank_manifest),controller_pid=20,running=[dict(pid=21)],events=[dict(event="controller_started")])
    rank_state.write_text(json.dumps(rank))
    with pytest.raises(RuntimeError,match="still owns"):
        overlap.verify_prior_overlap_owners_terminal(tmp_path)
    rank.update(running=[],events=[dict(event="controller_started"),dict(event="controller_stopped")])
    rank_state.write_text(json.dumps(rank))
    assert overlap.verify_prior_overlap_owners_terminal(tmp_path)["sampler_execution_state"]["terminal_evidence"]=="recorded_controller_stopped_event"


def test_reserved_subset_runs_real_controller_with_three_synthetic_children_and_no_duplicate_launch(tmp_path,monkeypatch):
    """Exercise the actual controller's inspect/start/complete path without ML."""
    source_path,path,source,manifest=prepared(tmp_path)
    child=tmp_path/"write_synthetic_result.py"
    child.write_text('''import json,pathlib,sys,time
folder=pathlib.Path(sys.argv[1]);time.sleep(.1)
(folder/"candidates.jsonl").write_text("{}\\n"*60)
(folder/"frozen_estimator.py").write_text("# synthetic; never inferred")
(folder/"summary.json").write_text(json.dumps(dict(status="finished",requests=60)))
''')
    # Synthetic source and subset still have byte-identical job dictionaries.
    for job in source["jobs"]:
        if job["phase"]=="feedback_control":job["command"]=[sys.executable,str(child),str(tmp_path/job["expected_logs"])]
    source_path.write_text(json.dumps(source));path.unlink()
    manifest=overlap.prepare_manifest(source_path,path,tmp_path)
    monkeypatch.setattr(overlap.controller,"ROOT",tmp_path)
    monkeypatch.setattr(overlap.controller,"verify_inputs",lambda *a,**k:None)
    monkeypatch.setattr(overlap.controller,"verify_servers",lambda *a,**k:None)
    args=SimpleNamespace(state=path.with_name("feedback_overlap_state.json"),
        deadline_utc=overlap.controller.utcnow()+common.dt.timedelta(minutes=10),reserve_minutes=0,
        min_job_minutes=.01,max_job_minutes=1,ports=[18000,18001])
    original=overlap.controller.inspect_job
    with overlap.owned_feedback_directories(manifest,common.sha(path),tmp_path):
        for job in manifest["jobs"]:
            with pytest.raises(ValueError,match="Interrupted job"):
                original(job,tmp_path)
        assert overlap.controller.execute(manifest,path,overlap.controller.ordered_batches(manifest),args)==0
    state=json.loads(args.state.read_text())
    assert common.controller_stopped(state)
    assert sum(e["event"]=="job_started" for e in state["events"])==3
    assert all(common.full_budget(job,tmp_path) for job in manifest["jobs"])
    assert overlap.controller.inspect_job is original
