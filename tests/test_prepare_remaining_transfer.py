import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_remaining_transfer", ROOT / "scripts/prepare_remaining_transfer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_checkpoint_pins_fetch_1b_weights_and_only_source_tokenizer(tmp_path):
    target = module.pinned_assets("Llama3.2-1B", module.TARGET_ID, module.TARGET_REVISION, tmp_path / "target")
    tokenizer = module.pinned_assets("llama31", module.TOKENIZER_ID, module.TOKENIZER_REVISION,
                                     tmp_path / "tokenizer", tokenizer_only=True)
    assert [r["file"] for r in target["files"] if r["file"].endswith(".safetensors")] == ["model.safetensors"]
    assert all(not r["file"].endswith(".safetensors") for r in tokenizer["files"])
    assert sum(r["bytes"] for r in tokenizer["files"]) < 10 * 1024**2
    assert set(module.MISSING) == {6, 7, 8}
    assert sum(row[2] for row in module.MISSING.values()) == 41253


def test_commands_only_missing_indices_and_preserve_historical_forward_settings(tmp_path):
    args = SimpleNamespace(data_root=ROOT / "data/claims", cache_dir=tmp_path / "caches",
        output=tmp_path / "output", model_dir=tmp_path / "model", tokenizer_dir=tmp_path / "tokenizer")
    for index in module.MISSING:
        command = module.cache_command(args, index)
        expected = {"--mode": "cache", "--dataset-index": str(index), "--dtype": "float32",
                    "--batch-size": "16", "--max-batch-tokens": "16384", "--cpu-threads": "12",
                    "--gc-interval": "32", "--cuda-cache-interval": "32"}
        assert all(command[command.index(key) + 1] == value for key, value in expected.items())


def test_existing_asset_hash_guard_never_replaces_file(tmp_path, monkeypatch):
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b"different_existing_file")
    before = path.read_bytes()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    assets = dict(destination=str(tmp_path), model="public/pinned", revision="abc",
                  files=[dict(file=path.name, bytes=3, sha256=hashlib.sha256(b"abc").hexdigest())])
    with pytest.raises(ValueError, match="Refusing to replace"):
        module.download_assets(assets)
    assert path.read_bytes() == before


def test_cache_inspection_checks_rows_labels_shapes_and_metadata(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame(dict(claim=["sample zero", "sample one"], is_hallu=[0, 1])).to_csv(csv, index=False)
    samples = [dict(claim=f"sample {name}", label=i,
        sequence_logprobs=np.full(5, -1., dtype=np.float16), top_k_logits=torch.zeros(6, 512, dtype=torch.float16))
        for i, name in enumerate(["zero", "one"])]
    cache = tmp_path / "cache.pt"
    raw = dict(meta={**module.EXPECTED_META, "model_id": "/new/path/to/the/same/pinned/model"}, samples=samples)
    torch.save(raw, cache)
    result = module.inspect_cache(cache, csv, 2)
    assert result["sample_count"] == 2 and result["claims_labels_match_csv_in_order"]
    assert result["raw_sha256"] == module.sha(cache)
    raw["meta"]["model_dtype"] = "torch.bfloat16"
    torch.save(raw, cache)
    with pytest.raises(ValueError, match="metadata mismatch"):
        module.inspect_cache(cache, csv, 2)


def test_default_plan_never_downloads_or_starts_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(module, "download_assets", lambda *args, **kwargs: pytest.fail("download called"))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("child started"))
    monkeypatch.setattr("sys.argv", ["prepare_remaining_transfer", "--cache-dir", str(tmp_path / "caches"),
        "--model-dir", str(tmp_path / "model"), "--tokenizer-dir", str(tmp_path / "tokenizer"),
        "--output", str(tmp_path / "output")])
    module.main()
    capsys.readouterr()
    plan = json.loads((tmp_path / "output/remaining_transfer_plan.json").read_text())
    assert plan["status"] == "prepared_no_inference"
    assert len(plan["historical_caches_must_be_preserved"]) == 6
    assert [j["dataset_index"] for j in plan["missing_jobs"]] == [6, 7, 8]
    assert not (tmp_path / "model").exists() and not (tmp_path / "tokenizer").exists()


def synthetic_process_jobs(tmp_path, durations, failures=None):
    """Real subprocesses record CUDA routing without importing a GPU library."""
    failures = failures or {}
    script = tmp_path / "synthetic_cache.py"
    script.write_text('''import json, os, pathlib, sys, time
cache, pidfile, duration, exitcode, prerequisite = sys.argv[1:]
pathlib.Path(pidfile).write_text(str(os.getpid()))
started = time.time()
if prerequisite:
    while not pathlib.Path(prerequisite).exists():
        if time.time() - started > 3:
            raise SystemExit(9)
        time.sleep(.01)
time.sleep(float(duration))
if int(exitcode):
    raise SystemExit(int(exitcode))
pathlib.Path(cache).write_text(json.dumps(dict(
    gpu=os.environ["CUDA_VISIBLE_DEVICES"], started=started, finished=time.time())))
''')
    jobs = []
    for index in (6, 7, 8):
        cache, pidfile = tmp_path / f"cache_{index}.pt", tmp_path / f"pid_{index}"
        prerequisite = str(tmp_path / "pid_6") if index == 7 and failures.get(7) else ""
        jobs.append(dict(dataset_index=index, cache_path=str(cache), dataset_csv="unused.csv", rows=2,
            command=[sys.executable, str(script), str(cache), str(pidfile), str(durations[index]),
                     str(failures.get(index, 0)), prerequisite]))
    return jobs


def assert_recorded_children_reaped(tmp_path):
    for marker in tmp_path.glob("pid_*"):
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)


def test_parallel_routing_one_process_per_gpu_and_main_thread_receipts(tmp_path, monkeypatch):
    import datetime as dt
    import threading

    jobs = synthetic_process_jobs(tmp_path, {6: .8, 7: .2, 8: .2})
    monkeypatch.setattr(module, "inspect_cache", lambda path, csv, rows:
        dict(path=str(path), **json.loads(path.read_text())))
    records, threads = [], []

    def receive(entry, event):
        records.append((entry, event))
        threads.append(threading.get_ident())

    cutoff = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5)
    module.execute_cache_jobs(jobs, [3, 7], cutoff, tmp_path, receive, min_launch_seconds=0)
    by_index = {event["dataset_index"]: entry for entry, event in records}
    assert by_index[6]["gpu"] == "3"
    assert by_index[7]["gpu"] == by_index[8]["gpu"] == "7"
    assert by_index[7]["started"] < by_index[6]["finished"]
    assert by_index[6]["started"] < by_index[7]["finished"]
    assert by_index[8]["started"] >= by_index[7]["finished"]
    assert threads == [threading.get_ident()] * 3
    assert_recorded_children_reaped(tmp_path)


def test_parallel_shared_deadline_reaps_both_children_and_never_starts_xfact(tmp_path, monkeypatch):
    import datetime as dt

    jobs = synthetic_process_jobs(tmp_path, {6: 30, 7: 30, 8: .1})
    monkeypatch.setattr(module, "inspect_cache", lambda *args: pytest.fail("timed out cache inspected"))
    started = time.monotonic()
    cutoff = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=1)
    with pytest.raises(TimeoutError, match="deadline"):
        module.execute_cache_jobs(jobs, [0, 1], cutoff, tmp_path,
                                  lambda *args: pytest.fail("unexpected success"), min_launch_seconds=0)
    assert time.monotonic() - started < 3
    assert not (tmp_path / "pid_8").exists()
    assert_recorded_children_reaped(tmp_path)


def test_failed_parallel_lane_cancels_other_child_and_queued_xfact(tmp_path, monkeypatch):
    import datetime as dt

    jobs = synthetic_process_jobs(tmp_path, {6: 30, 7: .1, 8: .1}, failures={7: 3})
    monkeypatch.setattr(module, "inspect_cache", lambda *args: pytest.fail("failed cache inspected"))
    started = time.monotonic()
    cutoff = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5)
    with pytest.raises(RuntimeError, match="process failed"):
        module.execute_cache_jobs(jobs, [0, 1], cutoff, tmp_path,
                                  lambda *args: pytest.fail("unexpected success"), min_launch_seconds=0)
    assert time.monotonic() - started < 3
    assert not (tmp_path / "pid_8").exists()
    assert_recorded_children_reaped(tmp_path)


@pytest.mark.parametrize("extra", [
    ["--gpu-index", "0", "--gpu-indices", "0", "1"],
    ["--gpu-indices", "0", "0"],
    ["--gpu-indices", "0", "1", "--max-minutes", "36"],
])
def test_invalid_parallel_gpu_options_rejected(extra, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prepare_remaining_transfer", *extra])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_two_gpu_plan_has_fixed_routes_shared_35_minute_cap(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["prepare_remaining_transfer", "--gpu-indices", "0", "1",
                                    "--output", str(tmp_path / "output")])
    module.main()
    capsys.readouterr()
    plan = json.loads((tmp_path / "output/remaining_transfer_plan.json").read_text())
    assert plan["execution_mode"] == "two_gpu_dataset_lanes"
    assert plan["gpu_assignment"] == {"0": [6], "1": [7, 8]}
    assert plan["maximum_child_runtime_minutes"] == 35
    assert plan["numerical_reference_preflight"]["rows"]==427
    assert plan["numerical_reference_preflight"]["required_before_missing_jobs"] is True


def numerical_cache_pair(tmp_path):
    import copy
    samples=[]
    for i in range(6):
        n=7+i
        lp=np.linspace(-1-i*.2,-2-i*.2,n-1).astype(np.float16)
        samples.append(dict(claim=f"numerical sample {i}",label=float(i%2),id=None,
            input_ids=torch.arange(n,dtype=torch.int32),orig_input_ids=torch.arange(n-2,dtype=torch.int32),
            attention_mask=torch.ones(n,dtype=torch.int8),sequence_logprobs=lp,
            greedy_logprobs=np.full(n,-.5,dtype=np.float16),token_entropies=np.full(n,3.,dtype=np.float16),
            top_k_logits=torch.tensor(np.tile(np.linspace(6,-6,512),(n,1)),dtype=torch.float16),
            top_k_indices=torch.arange(512,dtype=torch.int64).repeat(n,1)))
    original=dict(meta=dict(module.EXPECTED_META),samples=samples)
    changed=copy.deepcopy(original)
    for sample in changed["samples"]:
        sample["sequence_logprobs"]+=np.float16(.0078125)
        sample["top_k_logits"]+=.0078125
        # Rank-slot ID changes are descriptive, not a failure condition.
        sample["top_k_indices"][:,[-2,-1]]=sample["top_k_indices"][:,[-1,-2]]
    old,new=tmp_path/"old.pt",tmp_path/"recomputed.pt"
    torch.save(original,old);torch.save(changed,new)
    return old,new,changed


def test_numerical_reference_reports_small_quantization_drift_and_rejects_gross_drift(tmp_path):
    old,new,changed=numerical_cache_pair(tmp_path)
    before=module.sha(old)
    result=module.compare_numerical_reference(old,new,expected_rows=6)
    assert result["status"]=="passed" and result["all_rows_compared"]==6
    assert result["canonical_nll"]["spearman_rho"]>=.999
    assert 0<result["absolute_difference_summaries"]["sequence_logprobs"]["max_abs_difference"]<=.1
    assert result["topk_ids_descriptive"]["rank_slot_match_fraction"]<1
    assert result["topk_ids_descriptive"]["top1_id_match_fraction"]==1
    assert result["labels_used_for_guard"] is False and result["quality_auc_computed"] is False
    assert "top512_conditional_entropy" in result["absolute_difference_summaries"]
    changed["samples"][0]["sequence_logprobs"][0]+=.25
    torch.save(changed,new)
    result=module.compare_numerical_reference(old,new,expected_rows=6)
    assert result["status"]=="failed_gross_drift_guard"
    assert result["guard_checks"]["observed_logprob_gross_drift"] is False
    assert module.sha(old)==before


@pytest.mark.parametrize("problem",["identity","nonfinite","dtype"])
def test_numerical_reference_missing_nonfinite_or_misaligned_features_fail(tmp_path,problem):
    old,new,changed=numerical_cache_pair(tmp_path)
    sample=changed["samples"][0]
    if problem=="identity":sample["input_ids"][0]+=1
    if problem=="nonfinite":sample["sequence_logprobs"][0]=np.nan
    if problem=="dtype":sample["sequence_logprobs"]=sample["sequence_logprobs"].astype(np.float32)
    torch.save(changed,new)
    with pytest.raises(ValueError,match="identity mismatch|Nonfinite|shape/dtype mismatch"):
        module.compare_numerical_reference(old,new,expected_rows=6)


def test_reference_gpu_job_uses_separate_scratch_first_gpu_shared_deadline_and_failure_receipt(tmp_path,monkeypatch):
    import datetime as dt
    args=SimpleNamespace(cache_dir=tmp_path/"preserved",data_root=ROOT/"data/claims",output=tmp_path/"receipts",
                         model_dir=tmp_path/"model",tokenizer_dir=tmp_path/"tokenizer")
    args.cache_dir.mkdir()
    historical=args.cache_dir/module.REFERENCE_NAME
    historical.write_bytes(b"preserved historical cache")
    entry=dict(path=str(historical),raw_sha256=module.sha(historical))
    cutoff=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=35)
    calls=[]
    def execute(jobs,gpus,reference_cutoff,output,completed):
        assert gpus==[3] and len(jobs)==1 and jobs[0]["dataset_index"]==0 and jobs[0]["rows"]==427
        assert reference_cutoff<=cutoff and (reference_cutoff-dt.datetime.now(dt.timezone.utc)).total_seconds()<=180
        cache=Path(jobs[0]["cache_path"])
        assert not cache.is_relative_to(args.cache_dir) and cache.parent.name!=args.cache_dir.name
        assert jobs[0]["command"][jobs[0]["command"].index("--dtype")+1]=="float32"
        calls.append(cache)
        cache.write_bytes(b"new disposable reference")
        completed(dict(path=str(cache)),dict(dataset_index=0,gpu_index=3,exit_code=0))
    monkeypatch.setattr(module,"execute_cache_jobs",execute)
    monkeypatch.setattr(module,"compare_numerical_reference",lambda *args,**kwargs:dict(status="failed_gross_drift_guard"))
    with pytest.raises(ValueError,match="gross-drift guard"):
        module.run_numerical_reference(args,3,cutoff,entry)
    receipt=json.loads((args.output/"remaining_transfer_numerical_reference.json").read_text())
    assert receipt["status"]=="failed" and receipt["historical_cache_unchanged"] is True
    assert historical.read_bytes()==b"preserved historical cache" and len(calls)==1
    monkeypatch.setattr(module,"compare_numerical_reference",lambda *args,**kwargs:dict(status="passed"))
    passed=module.run_numerical_reference(args,3,cutoff,entry)
    assert passed["status"]=="passed" and passed["rows"]==427
    assert passed["receipt_sha256"]==module.sha(args.output/"remaining_transfer_numerical_reference.json")
    assert calls[0]!=calls[1]
