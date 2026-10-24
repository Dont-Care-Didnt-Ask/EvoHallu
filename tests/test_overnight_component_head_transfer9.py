import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("overnight_component_head_transfer9",ROOT/"scripts/overnight_component_head_transfer9.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def numeric_completion():
    return dict(status="complete_three_missing_caches",numerical_reference_preflight=dict(
        status="passed",rows=427,historical_cache_unchanged=True,quality_labels_used_for_numerical_guard=False,quality_auc_computed=False,
        comparison=dict(status="passed",all_rows_compared=427,
            all_claims_labels_ids_input_ids_attention_masks_shapes_dtypes_exact=True,
            all_compared_numeric_values_finite=True,labels_used_for_guard=False,quality_auc_computed=False,
            guard_checks=dict(observed_logprob_gross_drift=True,canonical_nll_rank_preservation=True),
            absolute_difference_summaries=dict(sequence_logprobs=dict(max_abs_difference=.0078125)),
            canonical_nll=dict(spearman_rho=.99999))))


def identity():
    return dict(labels=np.array([0,1,0,1]),claim_group_ids=np.array([0,1,2,3]),duplicate_disjoint_mask=np.array([True,True,False,True]))


def arrays():
    return dict(**identity(),**{m.METHOD:np.array([.1,.8,.3,.7]),m.BASELINE:np.array([.2,.6,.4,.8])})


def bootstrap():
    return {m.METHOD:np.full(500,.8),m.BASELINE:np.full(500,.7)}


def test_complete_must_include_passed427_current_runtime_guard():
    good=numeric_completion()
    assert m.completion_ready(good)
    assert not m.completion_ready({"status":"running"})
    assert not m.completion_ready(None)
    for field,value in [("rows",426),("historical_cache_unchanged",False),("quality_auc_computed",True)]:
        bad=copy.deepcopy(good);bad["numerical_reference_preflight"][field]=value
        with pytest.raises(ValueError,match="427-row"):m.completion_ready(bad)
    bad=copy.deepcopy(good);bad["numerical_reference_preflight"]["comparison"]["canonical_nll"]["spearman_rho"]=.998
    with pytest.raises(ValueError,match="427-row"):m.completion_ready(bad)


def test_commands_score_only_three_additions_with_frozen_wrapper():
    plan=dict(frozen_wrapper="results/overnight20261021/screen_component_refit/frozen/screen_component_logistic/frozen_estimator.py")
    for i in [6,7,8]:
        command=m.evaluator_command(plan,i,ROOT/"results/test_transfer9"/str(i))
        assert command[command.index("--mode")+1]=="evaluate"
        assert command[command.index("--dataset-index")+1]==str(i)
        assert command[command.index("--frozen-glob")+1]==plan["frozen_wrapper"]
        assert command[command.index("--bootstrap")+1]=="500"
        assert command[command.index("--bootstrap-unit")+1]=="claim"
        assert command[command.index("--tokenizer-id")+1]=="/workspace/mod1-models/llama31-tokenizer"
        assert "--primary-only" not in command and "--execute" not in command
    for i in [0,5,9]:
        with pytest.raises(ValueError):m.evaluator_command(plan,i,ROOT/"results/test_transfer9")


def test_copy_six_is_byte_identical_idempotent_and_preserves_originals(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path);monkeypatch.setattr(m,"HEAD",tmp_path/"head")
    source=m.HEAD/"evaluations/target_six";source.mkdir(parents=True)
    plan=dict(datasets=[dict(name=f"dataset{i}") for i in range(9)],protected_sha256={})
    paths=[source/"manifest.json"]+[source/(f"dataset{i}"+suffix) for i in range(6) for suffix in ["_scores.npz","_bootstrap.npz",".json"]]
    for i,path in enumerate(paths):
        path.write_bytes((f"synthetic immutable artifact {i}\n"*3).encode())
        plan["protected_sha256"][m.rel(path)]=m.report.sha(path)
    original={path:path.read_bytes() for path in paths}
    out=tmp_path/"new";m.reuse_six(plan,out);m.reuse_six(plan,out)
    assert all(path.read_bytes()==content for path,content in original.items())
    assert all((out/"target_nine"/path.name).read_bytes()==content for path,content in original.items())
    receipt=json.loads((out/"six_reuse_receipt.json").read_text())
    assert len(receipt["files"])==18 and receipt["original_six_unchanged"]
    (out/"target_nine"/paths[1].name).write_bytes(b"conflict")
    with pytest.raises(ValueError,match="overwrite"):m.reuse_six(plan,out)
    assert all(path.read_bytes()==content for path,content in original.items())


def test_protected_head_or_source_change_fails_before_any_work(tmp_path):
    source=tmp_path/"weights.json";source.write_text("immutable")
    plan=dict(protected_sha256={"weights.json":m.report.sha(source)})
    m.assert_unchanged(plan,tmp_path)
    source.write_text("different")
    with pytest.raises(ValueError,match="Protected"):m.assert_unchanged(plan,tmp_path)


def test_target_cache_cohort_cannot_change_under_reused_predictions(tmp_path):
    rows=[dict(dataset=f"dataset{i}",raw_sha256=str(i)*64) for i in range(9)]
    original=m.pin_cache_identity(tmp_path,"a"*64,rows)
    assert m.pin_cache_identity(tmp_path,"a"*64,rows)==original
    with pytest.raises(ValueError,match="never reuse"):
        m.pin_cache_identity(tmp_path,"b"*64,rows)
    changed=copy.deepcopy(rows);changed[0]["raw_sha256"]="f"*64
    with pytest.raises(ValueError,match="never reuse"):
        m.pin_cache_identity(tmp_path,"a"*64,changed)
    assert json.loads((tmp_path/"TARGET_CACHE_COHORT.json").read_text())==original


def test_raw_cache_audit_never_runs_without_numerical_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path);monkeypatch.setattr(m,"assert_unchanged",lambda *a:None)
    receipt=tmp_path/"completion.json";receipt.write_text(json.dumps(dict(status="complete_three_missing_caches")))
    monkeypatch.setattr(m.provenance,"audit_target_caches",lambda *a:pytest.fail("raw caches opened before numeric gate"))
    with pytest.raises(ValueError,match="427-row"):
        m.cache_audit(dict(completion_receipt="completion.json"),tmp_path/"out")


def test_main_reference_pending_cannot_be_inferred_as_a_match(tmp_path):
    names=[f"dataset{i}" for i in range(9)]
    scores={name:arrays() for name in names};boots={name:bootstrap() for name in names}
    result=m.main_nine_parity(tmp_path,names,scores,boots)
    assert result["status"]=="pending_main_full_nine_reference"
    assert len(result["per_dataset"])==9


def reference_fixture(tmp_path):
    names=[f"dataset{i}" for i in range(9)]
    cohort=tmp_path/"transfer_llama32_1b/cohort";folder=cohort/"new";folder.mkdir(parents=True)
    (cohort/"transfer_protocol.json").write_text(json.dumps(dict(coverage="nine",prespecified_transfer_subset=names,
        target_cache_audit=dict(status="passed",numerical_reference_preflight_verified=True))))
    for name in names:
        np.savez(folder/(name+"_scores.npz"),**arrays())
        np.savez(folder/(name+"_bootstrap.npz"),**bootstrap())
    return names,folder,{name:arrays() for name in names},{name:bootstrap() for name in names}


def test_main_fullnine_identity_nll_and500_draw_parity_verified(tmp_path):
    names,folder,scores,boots=reference_fixture(tmp_path)
    assert m.main_nine_parity(tmp_path,names,scores,boots)["status"]=="exact_match"
    changed=bootstrap();changed[m.BASELINE][0]+=.001
    np.savez(folder/(names[0]+"_bootstrap.npz"),**changed)
    assert m.main_nine_parity(tmp_path,names,scores,boots)["status"]=="MISMATCH"


def test_reference_being_written_remains_pending_not_fabricated(tmp_path):
    names,folder,scores,boots=reference_fixture(tmp_path)
    (folder/(names[0]+"_scores.npz")).write_bytes(b"partial")
    assert m.main_nine_parity(tmp_path,names,scores,boots)["status"]=="pending_main_full_nine_reference"


def test_partial_eight_datasets_never_create_nine_dataset_macro(tmp_path):
    names=[f"dataset{i}" for i in range(9)]
    for name in names[:8]:
        np.savez(tmp_path/(name+"_scores.npz"),**arrays())
        np.savez(tmp_path/(name+"_bootstrap.npz"),**bootstrap())
        (tmp_path/(name+".json")).write_text(json.dumps(dict(failed={})))
    errors=[]
    scores,boots,_=m.load_checked(tmp_path,names,{n:identity() for n in names},dict(status="passed"),errors)
    assert not errors and len(scores)==8
    result=m.report.scope_summary(scores,boots,names)
    assert result["expected_datasets"]==9 and result["verified_datasets"]==8 and "macro" not in result


def test_wait_once_is_cpu_idle_and_never_launches_a_child(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"ROOT",tmp_path)
    monkeypatch.setattr(m,"run_bounded",lambda *a:pytest.fail("child launched before receipt complete"))
    args=SimpleNamespace(once=True,poll_seconds=1,max_seconds=1200,workers=2)
    m.execute(dict(completion_receipt="absent.json"),tmp_path/"out",args,m.Bounds(m.now()+dt.timedelta(seconds=5)))
    assert json.loads((tmp_path/"out/execution_receipt.json").read_text())["status"]=="waiting_for_completed_numeric_guard"


def test_deadline_terminates_only_the_dummy_cpu_child(tmp_path):
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        m.run_bounded([sys.executable,"-c","import time; time.sleep(30)"],tmp_path/"log",m.Bounds(m.now()+dt.timedelta(seconds=.1)))
    assert time.monotonic()-started<3


def test_cancellation_and_0206_cap():
    stop=m.threading.Event();stop.set()
    with pytest.raises(InterruptedError):m.Bounds(m.now()+dt.timedelta(minutes=1),stop=stop).check()
    with pytest.raises(Exception):m.parse_deadline("2026-10-22T02:06:01Z")


def test_default_plan_path_does_not_launch_or_open_raw_caches(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,"ROOT",tmp_path);monkeypatch.setattr(m,"HEAD",tmp_path/"head")
    monkeypatch.setattr(m,"prepare",lambda directory:dict(protected_sha256={},completion_receipt="absent.json"))
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**k:pytest.fail("default mode started a process"))
    monkeypatch.setattr(m.provenance,"audit_target_caches",lambda *a:pytest.fail("default opened raw caches"))
    monkeypatch.setattr(sys,"argv",["transfer9","--directory",str(tmp_path/"plan")])
    m.main()
    assert json.loads(capsys.readouterr().out)["status"]=="prepared_not_executed"


def test_execute_rejected_outside_authorized_pod_before_planning(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,"ROOT",tmp_path);monkeypatch.setattr(m,"HEAD",tmp_path/"head")
    monkeypatch.setattr(m,"prepare",lambda *a:pytest.fail("should reject host before preparing"))
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**k:pytest.fail("nonpod execution started a process"))
    monkeypatch.setattr(sys,"argv",["transfer9","--execute","--directory",str(tmp_path/"plan")])
    with pytest.raises(SystemExit):m.main()
    assert "authorized pod" in capsys.readouterr().err
