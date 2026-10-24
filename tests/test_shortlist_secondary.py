import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import overnight_shortlist_secondary as shortlist


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(shortlist, "ROOT", tmp_path)
    monkeypatch.setattr(shortlist.evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(shortlist.reporter, "ROOT", tmp_path)
    jobs = []
    baseline = "\nclass HallucinationDetector:\n    pass\n"
    other = "\nclass HallucinationDetector:\n    value = 3\n"
    for policy in shortlist.POLICIES:
        for seed in shortlist.SEEDS:
            job_id = f"next_evo_{policy}_{seed}"
            job = dict(id=job_id, phase="compositional_control" if policy == "compositional" else "evolution_screen",
                       prompt_policy=policy, seed=seed, requests=60, mode="evolution", expected_logs=f"out/{job_id}")
            jobs.append(job)
            run = tmp_path / job["expected_logs"]
            run.mkdir(parents=True)
            (run/"frozen_estimator.py").write_text(baseline)
            rows=[]
            for rank, candidate, code, val in ((1,"baseline",baseline,75.),(2,"same_code",baseline,75.),(3,"other",other,74.)):
                path=run/"validation_shortlist"/f"rank_{rank:02d}.py"
                path.parent.mkdir(exist_ok=True)
                path.write_text(code)
                rows.append(dict(rank=rank,candidate=candidate,sha256=shortlist.sha(path),file=str(path),validation_auc_percent=val))
            dump(run/"summary.json",dict(status="finished",requests=60,valid_candidates=2,selected_candidate="baseline",
                selected_validation_auc_percent=75.,validation_shortlist=rows))
            dump(run/"trial_config.json",dict(model=dict(hf_model_id="Qwen/Qwen3.5-9B"),trial=dict(shortlist_size=3)))
    source=dict(jobs=jobs,writer="Qwen/Qwen3.5-9B",writer_revision="fixed")
    source_path=tmp_path/"source.json"
    dump(source_path,source)
    primary=tmp_path/"primary"
    frozen=shortlist.evaluator.stage(SimpleNamespace(manifest=source_path,directory=primary,
        include_prompt_screen=False,include_train_selected=False,shortlist=1))
    predictions={r["method"]:np.array([.1,.9]) for r in frozen["methods"]}
    predictions.update(labels=np.array([0,1]),claim_group_ids=np.array([0,1]),
        duplicate_disjoint_mask=np.ones(2,dtype=bool),**{shortlist.BASELINE:np.array([.1,.9])})
    folder=primary/"evaluations"/"fixed"
    folder.mkdir(parents=True)
    for index in range(9):
        np.savez(folder/f"dataset{index}_scores.npz",**predictions)
    secondary=tmp_path/"secondary"
    dump(secondary/"source_manifest.json",source)
    args=SimpleNamespace(directory=secondary,primary_directory=primary)
    return args,jobs


def test_all_twelve_full_budgets_before_freeze_and_isolated_alias_slots(tmp_path,monkeypatch):
    args,jobs=fixture(tmp_path,monkeypatch)
    (args.primary_directory/"evaluations"/"fixed"/"writing_scores.npz").write_bytes(b"incomplete zip")
    original=args.primary_directory/"frozen_manifest.json"
    original_bytes=original.read_bytes()
    summary_path=tmp_path/jobs[-1]["expected_logs"]/"summary.json"
    summary=json.loads(summary_path.read_text())
    summary["requests"]=58
    dump(summary_path,summary)
    assert not shortlist.readiness(jobs,args.primary_directory)["ready"]
    with pytest.raises(RuntimeError,match="incomplete twelve-run"):
        shortlist.freeze(args,jobs)
    assert not (args.directory/"frozen_manifest.json").exists()
    summary["requests"]=60
    dump(summary_path,summary)
    slots=shortlist.freeze(args,jobs)
    assert len(slots)==36
    assert original.read_bytes()==original_bytes
    rank_two=[r for r in slots if r["validation_rank"]==2]
    assert len(rank_two)==12 and all(r["aliases_earlier_source"] for r in rank_two)
    assert all(r["staged_role"]=="primary" and r["candidate"]=="same_code" for r in rank_two)
    rank_three=[r for r in slots if r["validation_rank"]==3]
    assert all(r["staged_role"]=="valrank03" for r in rank_three)
    assert len(json.loads((args.directory/"frozen_manifest.json").read_text())["methods"])==24


def test_baseline_tie_order_and_export_integrity_are_selection_guards(tmp_path,monkeypatch):
    _,jobs=fixture(tmp_path,monkeypatch)
    run=tmp_path/jobs[0]["expected_logs"]
    summary=json.loads((run/"summary.json").read_text())
    # A proposal tied with baseline cannot be promoted ahead of baseline.
    rows=summary["validation_shortlist"]
    rows[0]["candidate"],rows[1]["candidate"]="same_code","baseline"
    summary["selected_candidate"]="same_code"
    with pytest.raises(ValueError,match="Baseline tie priority"):
        shortlist.validation_rows(run,summary)
    rows.pop()
    with pytest.raises(ValueError,match="Missing exported valid"):
        shortlist.validation_rows(run,summary)


def test_nine_dataset_macro_requires_complete_saved_predictions():
    names=[f"dataset{i}" for i in range(9)]
    saved={name:dict(labels=np.array([0,1,0,1]),duplicate_disjoint_mask=np.ones(4,dtype=bool),
                     method=np.array([.1,.8,.3,.9])) for name in names}
    del saved[names[-1]]["method"]
    partial=shortlist.metrics("method",saved,names)
    assert partial["status"]=="incomplete" and "paper" not in partial and "directed" not in partial
    saved[names[-1]]["method"]=np.array([.1,.8,.3,.9])
    complete=shortlist.metrics("method",saved,names)
    assert complete["paper"]==100 and complete["strict_paper"]==100


def test_waiting_does_not_consume_active_budget_and_utc_hard_gate(monkeypatch,tmp_path):
    clock=[1000.]
    answers=iter([dict(ready=False),dict(ready=False),dict(ready=True)])
    monkeypatch.setattr(shortlist,"readiness",lambda *args:next(answers))
    monkeypatch.setattr(shortlist.time,"time",lambda:clock[0])
    monkeypatch.setattr(shortlist.time,"sleep",lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    status,started,deadline=shortlist.await_ready([],tmp_path,9000.,1800,[])
    assert status["ready"] and started==1040. and deadline==2840.
    monkeypatch.setattr(shortlist,"readiness",lambda *args:dict(ready=True))
    assert shortlist.await_ready([],tmp_path,2000.,1800,[])[2]==2000.
    assert shortlist.validated_deadline("2026-10-22T02:06:00Z")==shortlist.HARD_DEADLINE.timestamp()
    for invalid in ("2026-10-22T02:06:01Z","2026-10-22T02:00:00","2026-10-22T03:00:00+01:00"):
        with pytest.raises(ValueError):
            shortlist.validated_deadline(invalid)
    cancel=tmp_path/"cancel"
    cancel.touch()
    with pytest.raises(KeyboardInterrupt,match="cancelled by marker"):
        shortlist.await_ready([],tmp_path,9000.,1800,[cancel])


def test_cancel_and_sigterm_unwind_owned_subprocess_group(tmp_path):
    pid_file=tmp_path/"child.pid"
    cancel=tmp_path/"cancel"
    command=[sys.executable,"-c",f"import os,time;open({str(pid_file)!r},'w').write(str(os.getpid()));time.sleep(60)"]
    def request_cancel():
        until=time.time()+5
        while not pid_file.exists() and time.time()<until:
            time.sleep(.01)
        cancel.touch()
    thread=threading.Thread(target=request_cancel,daemon=True)
    thread.start()
    with pytest.raises(KeyboardInterrupt,match="cancelled by marker"):
        shortlist.bounded(command,time.time()+45,tmp_path/"log",[cancel])
    thread.join(timeout=2)
    assert pid_file.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()),0)
    prior=signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt,match="received signal"):
        with shortlist.execution_guard(time.time()+30):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    assert signal.getsignal(signal.SIGTERM)==prior
