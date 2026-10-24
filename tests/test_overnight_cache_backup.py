import copy
import datetime as dt
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("overnight_cache_backup",ROOT/"scripts/overnight_cache_backup.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def guard():
    return m.Guard(m.utcnow()+dt.timedelta(minutes=3))


def receipts():
    completion=dict(status="complete_three_missing_caches",
        target=dict(model="unsloth/Llama-3.2-1B-Instruct",revision=m.TARGET_REVISION,all_files_match_pinned_historical_hashes=True),
        tokenizer=dict(model="VityaVitalich/Llama3.1-8b-instruct",revision=m.TOKENIZER_REVISION,all_files_match_pinned_historical_hashes=True),
        tokenizer_reference_parity=dict(exact_chat_input_ids_match=True,exact_raw_claim_input_ids_match=True),
        all_six_historical_caches_unchanged=True,new_caches=[])
    payloads={}
    for i,(name,count) in enumerate(m.EXPECTED.items()):
        raw=(b"synthetic lossless cache"+bytes([i]))*1000
        completion["new_caches"].append(dict(path="/dev/shm/mod1-transfer-cache/"+name,sample_count=count,
            raw_bytes=len(raw),raw_sha256=digest(raw),claims_labels_match_csv_in_order=True,
            all_topk_widths_512=True,all_topk_logprob_length_offset_1=True,
            meta=dict(hidden_size=2048,num_layers=16,num_attention_heads=32,vocab_size=128256,modalities="logits",model_dtype="torch.float32")))
        payloads[name+".xz"]=lzma.compress(raw)
    text=json.dumps(completion,indent=2)+"\n"
    archive=dict(status="complete_three_new_cache_archives",source_receipt_sha256=digest(text.encode()),
        original_raw_files_removed=False,archives=[])
    for row in completion["new_caches"]:
        name=Path(row["path"]).name+".xz";data=payloads[name]
        archive["archives"].append(dict(source=row["path"],archive=m.REMOTE_ARCHIVES+"/"+name,
            archive_bytes=len(data),archive_sha256=digest(data),raw_bytes=row["raw_bytes"],raw_sha256=row["raw_sha256"],
            source_unchanged=True,roundtrip_raw_sha256_verified=True))
    archive["total_archive_bytes"]=sum(r["archive_bytes"] for r in archive["archives"])
    return completion,archive,payloads,text


def test_complete_and_archive_guards_are_read_only_and_require_all_three():
    completion,archive,_,_=receipts();original=copy.deepcopy((completion,archive))
    assert m.complete(completion)
    assert m.archive_complete(archive,completion,archive["source_receipt_sha256"])
    assert (completion,archive)==original
    assert not m.complete({"status":"running"})
    partial=copy.deepcopy(archive);partial["status"]="running"
    assert not m.archive_complete(partial,completion,archive["source_receipt_sha256"])


@pytest.mark.parametrize("change",["traversal","revision","dtype","duplicate","row_order"])
def test_rejects_unsafe_or_unpinned_completion(change):
    completion,_,_,_=receipts()
    if change=="traversal":completion["new_caches"][0]["path"]="/dev/shm/mod1-transfer-cache/../geometry__counterfact_true_false.pt"
    if change=="revision":completion["target"]["revision"]="wrong"
    if change=="dtype":completion["new_caches"][0]["meta"]["model_dtype"]="torch.float16"
    if change=="duplicate":completion["new_caches"][1]=completion["new_caches"][0]
    if change=="row_order":completion["new_caches"][0]["claims_labels_match_csv_in_order"]=False
    with pytest.raises(ValueError):m.complete(completion)


@pytest.mark.parametrize("change",["source_hash","archive_path","roundtrip","raw_deleted","total"])
def test_rejects_conflicting_or_unsafe_archive_receipts(change):
    completion,archive,_,_=receipts()
    if change=="source_hash":archive["source_receipt_sha256"]="f"*64
    if change=="archive_path":archive["archives"][0]["archive"]="/tmp/geometry__counterfact_true_false.pt.xz"
    if change=="roundtrip":archive["archives"][0]["roundtrip_raw_sha256_verified"]=False
    if change=="raw_deleted":archive["original_raw_files_removed"]=True
    if change=="total":archive["total_archive_bytes"]+=1
    with pytest.raises(ValueError):m.archive_complete(archive,completion,digest((json.dumps(completion,indent=2)+"\n").encode()))


def test_space_admission_counts_all_remaining_bytes_and_reserve(tmp_path):
    _,archive,payloads,_=receipts();directory=tmp_path/"archives";directory.mkdir()
    rows=archive["archives"];total=archive["total_archive_bytes"]
    plan=m.space_plan(rows,directory,guard(),free=total+m.RESERVE-1)
    assert not plan["space_admitted"] and plan["remaining_archive_bytes"]==total
    target,partial=m.local_paths(directory,rows[0]);target.write_bytes(payloads[target.name])
    _,partial=m.local_paths(directory,rows[1]);partial.write_bytes(payloads[Path(rows[1]["archive"]).name][:20])
    plan=m.space_plan(rows,directory,guard(),free=total+m.RESERVE)
    assert plan["space_admitted"]
    assert plan["remaining_archive_bytes"]==total-rows[0]["archive_bytes"]-20
    assert plan["already_verified"]==[target.name]


def test_sparse_partial_does_not_count_unallocated_logical_bytes(tmp_path):
    _,archive,_,_=receipts();rows=archive["archives"]
    _,partial=m.local_paths(tmp_path,rows[0])
    with partial.open("wb") as stream:stream.truncate(rows[0]["archive_bytes"])
    plan=m.space_plan(rows,tmp_path,guard(),free=archive["total_archive_bytes"]+m.RESERVE)
    assert partial.stat().st_blocks==0
    assert plan["remaining_archive_bytes"]==archive["total_archive_bytes"]


def test_insufficient_total_space_prevents_first_download(tmp_path,monkeypatch):
    _,archive,_,_=receipts();events=[]
    class Usage:
        free=archive["total_archive_bytes"]+m.RESERVE-1
    monkeypatch.setattr(m.shutil,"disk_usage",lambda path:Usage())
    with pytest.raises(m.SpaceError,match="no download started"):
        m.fetch_archives(archive,tmp_path,guard(),lambda status,**kw:events.append((status,kw)),
                         lambda *a,**k:pytest.fail("space check must precede every download"))
    assert events[-1][0]=="insufficient_local_space"
    assert not list(tmp_path.iterdir())


def test_verified_downloads_are_atomic_resumable_and_idempotent(tmp_path):
    _,archive,payloads,_=receipts();directory=tmp_path/"archives";calls=[];records=[]
    def runner(command,*args,**kwargs):
        assert command[0]=="rsync" and "--partial" in command and "--inplace" in command and "--checksum" in command
        assert str(command[-2]).startswith(m.HOST+":"+m.REMOTE_ARCHIVES+"/")
        destination=Path(command[-1]);destination.write_bytes(payloads[destination.name.removesuffix(".download.part")]);calls.append(command)
        return ""
    result=m.fetch_archives(archive,directory,guard(),lambda status,**kw:records.append(status),runner)
    assert len(result)==len(calls)==3 and all(r["compressed_sha256_verified"] for r in result)
    assert not list(directory.glob("*.part"))
    original={p:p.read_bytes() for p in directory.iterdir()}
    again=m.fetch_archives(archive,directory,guard(),lambda *a,**k:None,lambda *a,**k:pytest.fail("already verified must not transfer"))
    assert len(again)==3 and all(p.read_bytes()==b for p,b in original.items())


def test_hash_mismatch_retains_partial_and_never_publishes(tmp_path):
    _,archive,_,_=receipts()
    def wrong(command,*args,**kwargs):Path(command[-1]).write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        m.fetch_archives(archive,tmp_path,guard(),lambda *a,**k:None,wrong)
    assert len(list(tmp_path.glob("*.part")))==1 and not list(tmp_path.glob("*.pt.xz"))


def test_existing_conflict_and_symlink_are_never_overwritten(tmp_path):
    _,archive,_,_=receipts();row=archive["archives"][0];target,partial=m.local_paths(tmp_path,row)
    target.write_bytes(b"important existing input")
    with pytest.raises(ValueError,match="refusing overwrite"):
        m.fetch_archives(archive,tmp_path,guard(),lambda *a,**k:None,lambda *a,**k:pytest.fail("must not transfer"))
    assert target.read_bytes()==b"important existing input"
    target.unlink();outside=tmp_path/"outside";outside.write_bytes(b"untouched");partial.symlink_to(outside)
    with pytest.raises(ValueError,match="symlink"):
        m.fetch_archives(archive,tmp_path,guard(),lambda *a,**k:None,lambda *a,**k:pytest.fail("must not transfer"))
    assert outside.read_bytes()==b"untouched"


def test_cancel_and_deadline_prevent_every_network_action(tmp_path):
    cancel=tmp_path/"watchdog.cancel";cancel.touch()
    watcher=m.Watcher(m.Guard(m.utcnow()+dt.timedelta(minutes=1),[cancel]),output=tmp_path,destination=tmp_path/"archives",runner=lambda *a,**k:pytest.fail("must not SSH"))
    with pytest.raises(m.Cancelled):watcher.cycle()
    watcher.guard=m.Guard(m.utcnow()-dt.timedelta(seconds=1))
    with pytest.raises(TimeoutError):watcher.cycle()
    assert not (tmp_path/"archives").exists()


def test_inflight_local_process_is_terminated_at_deadline():
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        m.monitored([sys.executable,"-c","import time; time.sleep(30)"],m.Guard(m.utcnow()+dt.timedelta(seconds=.1)),timeout=30)
    assert time.monotonic()-started<3


def test_inflight_process_stops_when_protected_disk_reserve_is_lost(tmp_path,monkeypatch):
    class Usage:
        free=m.RESERVE-1
    monkeypatch.setattr(m.shutil,"disk_usage",lambda path:Usage())
    with pytest.raises(m.SpaceError,match="protected reserve"):
        m.monitored([sys.executable,"-c","import time; time.sleep(30)"],guard(),space_path=tmp_path)


def test_default_dry_run_never_connects_or_writes(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,"OUTPUT",tmp_path/"absent")
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**k:pytest.fail("dry-run attempted a process"))
    monkeypatch.setattr(sys,"argv",["backup"])
    m.main()
    assert json.loads(capsys.readouterr().out)["status"]=="dry_run_no_ssh_or_launch"
    assert not (tmp_path/"absent").exists()


def test_explicit_execution_is_rejected_outside_the_local_checkout(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,"ROOT",tmp_path/"remote_checkout")
    monkeypatch.setattr(m,"OUTPUT",tmp_path/"absent")
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**k:pytest.fail("nonlocal execution attempted a process"))
    monkeypatch.setattr(sys,"argv",["backup","--execute"])
    with pytest.raises(SystemExit):m.main()
    assert "never on the pod" in capsys.readouterr().err
    assert not (tmp_path/"absent").exists()


def test_local_completed_receipt_avoids_remote_launch_and_fetch(tmp_path):
    completion,archive,payloads,text=receipts();output=tmp_path/"results";destination=tmp_path/"archives"
    output.mkdir();destination.mkdir()
    (output/"remaining_transfer_completion.json").write_text(text)
    (output/"new_feature_cache_archives.json").write_text(json.dumps(archive))
    for name,data in payloads.items():(destination/name).write_bytes(data)
    watcher=m.Watcher(guard(),output,destination,lambda *a,**k:pytest.fail("complete local archive must not SSH or fetch"))
    assert watcher.cycle()
    receipt=json.loads((output/"cache_backup_receipt.json").read_text())
    assert receipt["status"]=="complete_three_local_cache_archives" and len(receipt["files"])==3


def test_remote_transaction_holds_flock_across_detached_child(tmp_path):
    # Emulate the fixed transaction on local temporary files only; the child is
    # a synthetic sleep/receipt writer, never the real archiver or an SSH call.
    output=tmp_path/"results/overnight20261021";output.mkdir(parents=True)
    completion,_,_,text=receipts();(output/"remaining_transfer_completion.json").write_text(text)
    script=tmp_path/"scripts/overnight_archive_transfer.py";script.parent.mkdir()
    script.write_text("import time\ntime.sleep(.6)\n")
    interpreter=tmp_path/".venv/bin/python";interpreter.parent.mkdir(parents=True);interpreter.symlink_to(sys.executable)
    code=m.REMOTE_LAUNCH_CODE.replace("ROOT = Path('/workspace/mod1')","ROOT = Path("+repr(str(tmp_path))+")")
    code=code.replace("cutoff=dt.datetime(2026,10,22,2,6,tzinfo=dt.timezone.utc)","cutoff=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=1)")
    command=[sys.executable,"-c",code,digest(text.encode()),m.hash_file(script)]
    rejected=json.loads(subprocess.check_output(command[:-1]+["0"*64],text=True))
    assert rejected["status"]=="archive_script_sha_mismatch"
    assert not (output/"cache_archive_backup_launch.json").exists()
    first=json.loads(subprocess.check_output(command,text=True))
    second=json.loads(subprocess.check_output(command,text=True))
    assert first["status"]=="launched_detached_cpu_archiver"
    assert second["status"]=="archive_lock_busy"
    assert (output/"cache_archive_backup_launch.json").exists()
    time.sleep(.7)
    third=json.loads(subprocess.check_output(command,text=True))
    assert third["status"]=="archive_launch_died_requires_review"
    assert "--workers" in first["command"] and "2" in first["command"]
    assert first["command"][-1]=="2026-10-22T02:06:00Z"


def test_launch_command_is_fixed_quoted_and_hash_guarded():
    command=m.ssh_command(m.REMOTE_LAUNCH_CODE,"a"*64,"b"*64)
    assert command[:len(m.SSH)]==m.SSH and command[len(m.SSH)]==m.HOST
    assert "/root/.ssh/processbench_pod_ed25519" in command
    assert "pass_fds=(lock.fileno(),)" in m.REMOTE_LAUNCH_CODE
    assert "archive_script_sha_mismatch" in m.REMOTE_LAUNCH_CODE
    assert "prepare_remaining_transfer.py" not in m.REMOTE_LAUNCH_CODE
    with pytest.raises(Exception):m.parse_deadline("2026-10-22T02:14:00Z")
