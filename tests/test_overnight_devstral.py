import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scripts import overnight_devstral as dev


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(root,monkeypatch):
    path=root/"model";path.mkdir()
    config=dict(architectures=["Mistral3ForConditionalGeneration"],dtype="bfloat16",quantization_config=dict(quant_method="fp8"))
    content={name:b"data" for name in dev.NATIVE_SHARDS}
    content.update({"config.json":json.dumps(config).encode(),"params.json":b"{}","tekken.json":b"{}",
        "tokenizer_config.json":b"{}","chat_template.jinja":b"native template",
        "consolidated.safetensors.index.json":json.dumps(dict(weight_map=dict(zip(["a","b"],sorted(dev.NATIVE_SHARDS))))).encode()})
    metadata=[];receipts=[]
    for name,data in content.items():
        (path/name).write_bytes(data)
        row=dict(rfilename=name,size=len(data))
        if name.endswith(".safetensors"):row["lfs"]=dict(sha256=hashlib.sha256(data).hexdigest())
        else:row["blobId"]=hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
        metadata.append(row);receipts.append(dict(file=name,bytes=len(data)))
    dump(path/"download_metadata.json",dict(id=dev.MODEL,sha=dev.REVISION,private=False,gated=False,siblings=metadata))
    dump(path/"download_manifest.json",dict(model=dev.MODEL,revision=dev.REVISION,files=receipts))
    model=root/"configs/model/devstral_small2.yaml";model.parent.mkdir(parents=True)
    model.write_text(OmegaConf.to_yaml(OmegaConf.create(dict(hf_model_id=dev.MODEL,revision=dev.REVISION,
        supports_chat_template_kwargs=False,temperature=.7,topk=20,topp=.8,max_tokens=4096,repetition_penalty=1.))))
    monkeypatch.setattr(dev,"WEIGHT_BYTES",8)
    return path


def source_study(root):
    jobs=[]
    for seed in dev.discovery.DISCOVERY_SEEDS:
        for policy,score in [("original",95),("neutral",80),("diverse",82),("compositional",81)]:
            name=f"{policy}_{seed}";folder=root/"logs"/name;folder.mkdir(parents=True)
            dump(folder/"summary.json",dict(status="finished",requests=60,selected_validation_auc_percent=score))
            (folder/"candidates.jsonl").write_text("{}\n"*60);(folder/"frozen_estimator.py").write_text("# frozen")
            jobs.append(dict(id=name,mode="evolution",phase="evolution_screen",prompt_policy=policy,seed=seed,
                requests=60,expected_logs="logs/"+name,config=name,config_sha256="source-config-pin"))
    raw=root/"raw.pt";raw.write_bytes(b"tiny raw")
    csv=root/"data/claims/popqa/popqa_train_filtered_10shot.csv";csv.parent.mkdir(parents=True)
    csv.write_text("claim,is_hallu\nx,0\n")
    for file in ["source.py","scripts/overnight_devstral.py","scripts/overnight_prepare_confirmation.py","configs/model/devstral_small2.yaml"]:
        p=root/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("frozen bytes")
    source=dict(writer=dev.discovery.SOURCE_WRITER,writer_revision=dev.discovery.SOURCE_REVISION,jobs=jobs,
        feature_model="Llama3.1-8b-instruct",labels="is_hallu",training_raw_path="raw.pt",training_raw_sha256=dev.sha(raw),
        training_csv_sha256=dev.sha(csv),study_source_sha256={"source.py":dev.sha(root/"source.py")})
    path=root/"results/next_experiments/run_manifest.json";dump(path,source)
    dump(path.parent/"readiness.json",dict(cpu_gate_passed=True,all_ten_supplied_csvs_identical=True))
    return path,source


def test_public_native_files_require_both_published_hash_schemes(tmp_path,monkeypatch):
    path=checkpoint(tmp_path,monkeypatch)
    proof=dev.verify_native(path,root=tmp_path)
    assert proof["status"]=="ready_native_checkpoint" and proof["weight_bytes"]==8
    assert all(len(r["sha256"])==64 for r in proof["native_files"])
    assert {r["verification"] for r in proof["native_files"]}=={"published_lfs_sha256","published_git_blob_sha1_plus_recorded_sha256"}
    shard=path/sorted(dev.NATIVE_SHARDS)[0];shard.write_bytes(b"evil")
    with pytest.raises(ValueError,match="Published LFS SHA256"):
        dev.verify_native(path,root=tmp_path)
    shard.write_bytes(b"data");(path/"params.json").write_bytes(b"[]")
    with pytest.raises(ValueError,match="Git blob SHA1"):
        dev.verify_native(path,root=tmp_path)


def test_reject_gating_wrong_revision_and_duplicate_weight_format(tmp_path,monkeypatch):
    path=checkpoint(tmp_path,monkeypatch);p=path/"download_metadata.json";m=json.loads(p.read_text())
    for key,bad in [("gated",True),("sha","wrong")]:
        broken=dict(m);broken[key]=bad;dump(p,broken)
        with pytest.raises(ValueError,match="public, ungated, pinned"):
            dev.verify_native(path,root=tmp_path)
    dump(p,m);(path/"model-duplicate.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError,match="duplicate HF"):
        dev.verify_native(path,root=tmp_path)


def test_fixed_two_arm_manifest_uses_all_source_validation_and_preserves_pins(tmp_path):
    path,source=source_study(tmp_path);receipt=tmp_path/"results/overnight20261021/devstral/prefetch_receipt.json";dump(receipt,{"ready":True})
    broken=tmp_path/source["jobs"][-1]["expected_logs"]/"summary.json";saved=broken.read_bytes()
    dump(broken,dict(status="finished",requests=58,selected_validation_auc_percent=99))
    with pytest.raises(ValueError,match="all twelve"):
        dev.prepare(path,receipt,root=tmp_path)
    broken.write_bytes(saved)
    output,manifest=dev.prepare(path,receipt,root=tmp_path)
    assert len(manifest["selection_provenance"]["observations"])==12
    assert manifest["selection_provenance"]["selected_policy"]=="diverse"
    assert manifest["selection_provenance"]["held_out_test_scores_read"] is False
    assert {(r["prompt_policy"],r["seed"],r["requests"]) for r in manifest["jobs"]}=={("original",2107,60),("diverse",2107,60)}
    assert manifest["study_source_sha256"]["source.py"]==source["study_source_sha256"]["source.py"]
    for row in manifest["jobs"]:
        config=tmp_path/"configs"/(row["config"]+".yaml");cfg=OmegaConf.load(config)
        assert dev.sha(config)==row["config_sha256"]
        assert cfg.num_rounds==30 and cfg.num_outputs_per_prompt==2 and cfg.model.max_tokens==4096
        assert cfg.trial.max_seconds==3600 and cfg.trial.max_consecutive_failed_rounds==31 and cfg.trial.shortlist_size==3
    assert json.loads(path.read_text())==source
    with pytest.raises(FileExistsError,match="repeat/replace"):
        dev.prepare(path,receipt,root=tmp_path)
    assert output.exists()


def test_wall_clock_and_native_command_contract():
    cutoff=dev.parse_cutoff("2026-10-22T01:10:00Z")
    assert dev.time_gate(cutoff,75,cutoff-dt.timedelta(minutes=75))==75
    with pytest.raises(TimeoutError):dev.time_gate(cutoff,75,cutoff-dt.timedelta(minutes=74))
    with pytest.raises(TimeoutError):dev.time_gate(cutoff,65,cutoff-dt.timedelta(minutes=64))
    for value in ["2026-10-22T01:10:01Z","2026-10-22T01:00:00","2026-10-22T02:00:00+01:00"]:
        with pytest.raises(ValueError):dev.parse_cutoff(value)
    command=dev.server_command(Path("/model"),18000)
    for flag,value in [("--tokenizer-mode","mistral"),("--config-format","mistral"),("--load-format","mistral"),("--linear-backend","marlin"),("--max-num-seqs","2")]:
        assert command[command.index(flag)+1]==value
    assert "--enforce-eager" in command and not any("chat-template" in s for s in command)
    command=dev.controller_command(Path("manifest"),Path("state"),cutoff)
    assert command[command.index("--max-job-minutes")+1]=="60"
    assert command[command.index("--min-job-minutes")+1]=="60"
    assert dev.controller.parse_deadline(command[command.index("--deadline-utc")+1])-dt.timedelta(minutes=30)==cutoff


def test_analysis_merge_adds_only_two_new_generalization_identities(tmp_path):
    path,source=source_study(tmp_path);receipt=tmp_path/"receipt.json";dump(receipt,{})
    output,manifest=dev.prepare(path,receipt,root=tmp_path)
    target=tmp_path/"results/overnight20261021/analysis_manifest.json"
    original=dict(id="untouched",phase="evolution_screen",requests=60)
    dump(target,dict(jobs=[original],execution_manifests=[]))
    dev.merge_analysis(output,root=tmp_path)
    merged=json.loads(target.read_text())
    assert merged["jobs"][0]==original
    assert len(merged["jobs"])==3 and merged["requests_by_phase"]["writer_generalization"]==120
    assert all(r["writer_model"]==dev.MODEL and r["phase"]=="writer_generalization" for r in merged["jobs"][1:])
    assert merged["execution_manifests"][0]["sha256"]==dev.sha(output)


def test_failed_second_replica_stops_only_owned_first_process(tmp_path,monkeypatch):
    out=tmp_path/"results/overnight20261021/devstral";out.mkdir(parents=True)
    dump(out/"prefetch_receipt.json",{})
    monkeypatch.setattr(dev,"ROOT",tmp_path);monkeypatch.setattr(dev,"OUT",out)
    monkeypatch.setattr(dev,"SERVER_METADATA",tmp_path/"out/overnight20261021/devstral_servers.json")
    monkeypatch.setattr(dev,"now",lambda:dt.datetime(2026,10,21,22,tzinfo=dt.timezone.utc))
    monkeypatch.setattr(dev,"prefetch_receipt",lambda path:{})
    monkeypatch.setattr(dev.importlib.metadata,"version",lambda name:"0.31.0")
    monkeypatch.setattr(dev.hardware,"telemetry",lambda:{})
    monkeypatch.setattr(dev.hardware,"require_idle",lambda *args:None)
    monkeypatch.setattr(dev.hardware,"require_free_ports",lambda *args:None)
    manifest=tmp_path/"manifest.json";dump(manifest,dict(jobs=[]))
    monkeypatch.setattr(dev,"prepare",lambda *args:(manifest,dict(jobs=[])))
    real_popen=subprocess.Popen;children=[]
    def fake_server(*args,**kwargs):
        if children:raise OSError("synthetic second-replica start failure")
        p=real_popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True)
        children.append(p);return p
    monkeypatch.setattr(dev.subprocess,"Popen",fake_server)
    result=dev.execute(SimpleNamespace(deadline_utc="2026-10-22T01:10:00Z",model_path=Path("/model"),source_manifest=Path("/source")))
    assert result["status"]=="skipped_or_failed" and result["owned_process_cleanup"] is True
    assert "second-replica" in result["error"]
    assert children[0].poll() is not None
    with pytest.raises(ProcessLookupError):os.kill(children[0].pid,0)
    assert json.loads(dev.SERVER_METADATA.read_text())["status"]=="stopped"


def test_orphan_trial_cleanup_refuses_unrelated_state_or_changed_command(tmp_path,monkeypatch):
    manifest_path=tmp_path/"manifest.json";manifest=dict(jobs=[dict(id="ours",command=["python","main.py","--config-name","ours"])])
    dump(manifest_path,manifest)
    state=tmp_path/"state.json"
    row=dict(job="ours",port=18000,pid=999999)
    data=dict(manifest_sha256="unrelated",running=[row],jobs={"ours":dict(max_seconds=3600,command=["unrelated"])})
    dump(state,data)
    monkeypatch.setattr(dev.os,"killpg",lambda *args:pytest.fail("Must not signal an unrelated process"))
    with pytest.raises(ValueError,match="unrelated controller state"):
        dev.cleanup_trial_groups(state,manifest_path,manifest)
    data["manifest_sha256"]=dev.sha(manifest_path);dump(state,data)
    monkeypatch.setattr(dev.common,"process_alive",lambda pid:True)
    with pytest.raises(ValueError,match="Changed owned trial command"):
        dev.cleanup_trial_groups(state,manifest_path,manifest)
