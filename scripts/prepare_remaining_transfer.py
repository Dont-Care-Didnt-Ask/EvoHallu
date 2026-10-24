"""Plan or complete the three missing, pinned Llama-3.2-1B feature caches.

Dry run by default. --download fetches only pinned public 1B model files and
the original 8B tokenizer files; no 8B weights are fetched. --execute additionally
requires an explicit GPU and deadline, and preserves all six historical caches.
"""
from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "unsloth/Llama-3.2-1B-Instruct"
TARGET_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
TOKENIZER_ID = "VityaVitalich/Llama3.1-8b-instruct"
TOKENIZER_REVISION = "247cb7c3baebabc46cff1fb97a3a5913e4c4511d"
HARD_DEADLINE = dt.datetime(2026, 10, 22, 2, 16, 28, tzinfo=dt.timezone.utc)
MISSING = {
    6: ("geometry/counterfact_true_false.csv", "geometry__counterfact_true_false.pt", 31964),
    7: ("popqa/popqa_test_filtered_10shot.csv", "popqa__popqa_test_filtered_10shot.pt", 6974),
    8: ("x-fact/test_filtered_10shot.csv", "x-fact__test_filtered_10shot.pt", 2315),
}
EXPECTED_META = dict(hidden_size=2048, num_layers=16, num_attention_heads=32,
    vocab_size=128256, modalities="logits", model_dtype="torch.float32",
    requested_forward_batch_size=16, max_padded_forward_tokens=16384,
    cpu_feature_threads=12, cpu_gc_interval_batches=32,
    cuda_cache_cleanup_interval_batches=32)
REFERENCE_NAME = "AVeriTeC__dev_filtered_10shot.pt"
REFERENCE_ROWS = 427
NUMERICAL_GUARD = dict(observed_logprob_max_abs_difference_at_most=.1,
                       canonical_nll_spearman_at_least=.999,
                       require_exact_identity_shapes_dtypes=True,
                       require_all_numeric_values_finite=True)
REFERENCE_MAX_SECONDS = 180


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pinned_assets(folder, expected_id, expected_revision, destination, tokenizer_only=False):
    source = ROOT / "out/campaign/model_provenance" / folder / "download_manifest.json"
    manifest = json.loads(source.read_text())
    if manifest["model"] != expected_id or manifest["revision"] != expected_revision:
        raise ValueError("Historical download manifest does not match the required checkpoint")
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    rows = [row for row in manifest["files"] if not tokenizer_only or row["file"] in names]
    if tokenizer_only and not {"tokenizer.json", "tokenizer_config.json"}.issubset({r["file"] for r in rows}):
        raise ValueError("Pinned source tokenizer files are incomplete")
    return dict(model=expected_id, revision=expected_revision, tokenizer_only=tokenizer_only,
                historical_manifest=str(source.relative_to(ROOT)), historical_manifest_sha256=sha(source),
                destination=str(destination), files=rows)


def download_assets(assets):
    directory = Path(assets["destination"])
    directory.mkdir(parents=True, exist_ok=True)
    checked = []
    for entry in assets["files"]:
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("Only checkpoint-root assets are allowed")
        path = directory / name
        if path.exists():
            if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise ValueError("Refusing to replace an existing checkpoint asset with a different hash: " + str(path))
        else:
            url = (f"https://huggingface.co/{assets['model']}/resolve/{assets['revision']}/"
                   + urllib.parse.quote(name))
            temporary = directory / (name + ".part")
            try:
                digest = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                    for block in iter(lambda: response.read(8 * 1024**2), b""):
                        stream.write(block)
                        digest.update(block)
                if temporary.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError("Pinned checkpoint asset failed size/SHA-256 verification: " + name)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        checked.append(dict(file=name, bytes=path.stat().st_size, sha256=sha(path)))
    return {**assets, "files": checked, "all_files_match_pinned_historical_hashes": True}


def tokenizer_parity(tokenizer_path, reference_path):
    sys.path.insert(0, str(ROOT / "src"))
    from transformers import AutoTokenizer
    from packing.evaluate.uncertainty_quantification.precompute_raw_stats_separated import _apply_chat_template

    reference = json.loads(reference_path.read_text())["sample_0"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    rendered = _apply_chat_template(tokenizer, [reference["claim"]], "generate_true_statement")
    chat_ids = tokenizer(rendered, padding=True, truncation=True)["input_ids"][0]
    claim_ids = tokenizer([reference["claim"]], padding=True, truncation=True,
                          add_special_tokens=False)["input_ids"][0]
    if chat_ids != reference["input_ids"] or claim_ids != reference["orig_input_ids"]:
        raise ValueError("Pinned source tokenizer does not reproduce historical transfer-cache tokenization")
    return dict(historical_sample="AVeriTeC sample 0", exact_chat_input_ids_match=True,
                exact_raw_claim_input_ids_match=True, historical_template="generate_true_statement",
                add_generation_prompt=False, template_render_then_default_tokenization=True,
                preserves_historical_double_bos=True, reference_metadata_sha256=sha(reference_path))


def inspect_cache(path, csv_path, expected_rows):
    import numpy as np
    import pandas as pd
    import torch

    raw = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
    if len(raw["samples"]) != expected_rows:
        raise ValueError("Unexpected cached sample count: " + str(path))
    for key, expected in EXPECTED_META.items():
        if raw["meta"].get(key) != expected:
            raise ValueError(f"Transfer-cache metadata mismatch for {path}: {key}")
    frame = pd.read_csv(csv_path, usecols=["claim", "is_hallu"])
    if len(frame) != expected_rows:
        raise ValueError("Unexpected source CSV row count")
    claims = frame.claim.astype(str).str.strip().tolist()
    labels = frame.is_hallu.astype(float).tolist()
    for index, sample in enumerate(raw["samples"]):
        if sample["claim"] != claims[index] or float(sample["label"]) != labels[index]:
            raise ValueError("Cached claims/labels do not match the CSV in order")
        topk = sample["top_k_logits"]
        if tuple(topk.shape) != (len(sample["sequence_logprobs"]) + 1, 512):
            raise ValueError("Unexpected cached top-k / observed-logprob shape")
        if str(np.asarray(sample["sequence_logprobs"]).dtype) != "float16" or str(topk.dtype) != "torch.float16":
            raise ValueError("Historical stored feature dtypes must remain float16")
    return dict(path=str(path), raw_sha256=sha(path), raw_bytes=path.stat().st_size,
                sample_count=expected_rows, dataset_csv=str(csv_path), dataset_csv_sha256=sha(csv_path),
                meta=raw["meta"], claims_labels_match_csv_in_order=True,
                all_topk_widths_512=True, all_topk_logprob_length_offset_1=True,
                stored_sequence_logprobs_dtype="float16", stored_topk_logits_dtype="torch.float16")


def cache_command(args, index):
    return [sys.executable, "scripts/evaluate_cached_claims.py", "--mode", "cache",
            "--data-root", str(args.data_root), "--cache-dir", str(args.cache_dir),
            "--output", str(args.output / "cache_metadata" / f"dataset_{index}"),
            "--model-id", str(args.model_dir), "--tokenizer-id", str(args.tokenizer_dir),
            "--dataset-index", str(index), "--dtype", "float32", "--batch-size", "16",
            "--max-batch-tokens", "16384", "--cpu-threads", "12",
            "--gc-interval", "32", "--cuda-cache-interval", "32",
            "--frozen-glob", "out/overnight_no_selected_programs/*/frozen_estimator.py"]


def compare_numerical_reference(historical_path, recomputed_path, expected_rows=REFERENCE_ROWS):
    """Compare cache numerics without computing quality AUC or fitting anything.

    Labels are checked only for exact row identity, never used in a numerical
    guard. Top-k logits are compared by sorted rank, not assumed token identity.
    """
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from scipy.stats import spearmanr

    previous = torch.load(historical_path,mmap=True,weights_only=False,map_location="cpu")
    current = torch.load(recomputed_path,mmap=True,weights_only=False,map_location="cpu")
    if len(previous["samples"]) != expected_rows or len(current["samples"]) != expected_rows:
        raise ValueError("Numerical reference must include all prescribed rows")
    for raw in [previous,current]:
        if any(raw["meta"].get(key) != value for key,value in EXPECTED_META.items()):
            raise ValueError("Numerical reference forward metadata mismatch")
    identity_keys = ["input_ids","orig_input_ids","attention_mask"]
    numeric_keys = ["sequence_logprobs","greedy_logprobs","token_entropies","top_k_logits","top_k_indices"]
    differences = {name:[] for name in numeric_keys if name!="top_k_indices"}
    differences.update({name:[] for name in ["top512_conditional_top1_mass","top512_conditional_top10_mass","top512_conditional_entropy"]})
    nll_before,nll_after=[],[]
    slots,slot_matches,token_rows,top1_matches,exact_topk_rows=0,0,0,0,0
    for index,(old,new) in enumerate(zip(previous["samples"],current["samples"])):
        if old["claim"] != new["claim"] or old["label"] != new["label"] or old.get("id") != new.get("id"):
            raise ValueError("Numerical reference claim/label/id identity mismatch at row "+str(index))
        for key in identity_keys+numeric_keys:
            if key not in old or key not in new:
                raise ValueError("Missing numerical-reference field: "+key)
            a,b=np.asarray(old[key]),np.asarray(new[key])
            if a.shape!=b.shape or a.dtype!=b.dtype:
                raise ValueError("Numerical reference shape/dtype mismatch: "+key)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Nonfinite numerical-reference field: "+key)
            if key in identity_keys and not np.array_equal(a,b):
                raise ValueError("Numerical reference input/attention identity mismatch: "+key)
            if key in differences:
                differences[key].append(np.abs(a.astype(np.float64)-b.astype(np.float64)).ravel())
        n=len(old["input_ids"])
        if (np.asarray(old["sequence_logprobs"]).shape!=(n-1,) or
            np.asarray(old["greedy_logprobs"]).shape!=(n,) or
            np.asarray(old["token_entropies"]).shape!=(n,) or
            np.asarray(old["top_k_logits"]).shape!=(n,512) or
            np.asarray(old["top_k_indices"]).shape!=(n,512)):
            raise ValueError("Numerical reference field alignment mismatch")
        if (np.asarray(old["sequence_logprobs"]).dtype!=np.float16 or
            np.asarray(old["top_k_logits"]).dtype!=np.float16):
            raise ValueError("Numerical reference must retain historical float16 feature storage")
        # Match the actual SequenceLogProb implementation: float16 sum followed
        # by a float32 output. No promotion before this canonical NLL sum.
        nll_before.append(np.float32(-np.sum(old["sequence_logprobs"])))
        nll_after.append(np.float32(-np.sum(new["sequence_logprobs"])))
        a,b=[np.asarray(sample["top_k_logits"],dtype=np.float64) for sample in [old,new]]
        conditional=[]
        for logits in [a,b]:
            logp=logits-logsumexp(logits,axis=1,keepdims=True)
            probabilities=np.exp(logp)
            conditional.append((probabilities[:,0],probabilities[:,:10].sum(axis=1),-(probabilities*logp).sum(axis=1)))
        for j,key in enumerate(["top512_conditional_top1_mass","top512_conditional_top10_mass","top512_conditional_entropy"]):
            differences[key].append(np.abs(conditional[0][j]-conditional[1][j]))
        ids_a,ids_b=[np.asarray(sample["top_k_indices"]) for sample in [old,new]]
        slots+=ids_a.size
        slot_matches+=int(np.count_nonzero(ids_a==ids_b))
        token_rows+=len(ids_a)
        top1_matches+=int(np.count_nonzero(ids_a[:,0]==ids_b[:,0]))
        exact_topk_rows+=int(np.count_nonzero(np.all(ids_a==ids_b,axis=1)))
    numeric={}
    for key,parts in differences.items():
        values=np.concatenate(parts)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Missing/nonfinite numerical comparison: "+key)
        numeric[key]=dict(elements=len(values),max_abs_difference=float(np.max(values)),
                          mean_abs_difference=float(np.mean(values)),p99_abs_difference=float(np.quantile(values,.99)),
                          p999_abs_difference=float(np.quantile(values,.999)),exact_value_fraction=float(np.mean(values==0)))
    before,after=np.asarray(nll_before),np.asarray(nll_after)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Canonical NLL nonfinite in reference comparison")
    rho=float(spearmanr(before,after).statistic)
    checks=dict(observed_logprob_gross_drift=numeric["sequence_logprobs"]["max_abs_difference"]<=NUMERICAL_GUARD["observed_logprob_max_abs_difference_at_most"],
                canonical_nll_rank_preservation=bool(np.isfinite(rho) and rho>=NUMERICAL_GUARD["canonical_nll_spearman_at_least"]))
    return dict(status="passed" if all(checks.values()) else "failed_gross_drift_guard",all_rows_compared=expected_rows,
                all_claims_labels_ids_input_ids_attention_masks_shapes_dtypes_exact=True,
                all_compared_numeric_values_finite=True,predeclared_guard=NUMERICAL_GUARD,guard_checks=checks,
                absolute_difference_summaries=numeric,
                canonical_nll=dict(spearman_rho=rho if np.isfinite(rho) else None,
                    max_abs_difference=float(np.max(np.abs(before-after))),mean_abs_difference=float(np.mean(np.abs(before-after))),
                    historical_float16_sum_then_float32_output=True),
                topk_ids_descriptive=dict(rank_slot_match_fraction=slot_matches/slots,top1_id_match_fraction=top1_matches/token_rows,
                    exact_512_rank_id_row_fraction=exact_topk_rows/token_rows,logit_comparison="rank-aligned sorted values; token IDs may differ around ties; not an equality requirement"),
                quality_auc_computed=False,labels_used_for_guard=False,bitwise_parity_claimed=False)


def run_numerical_reference(args,gpu_index,cutoff,historical_entry):
    """Recompute all 427 existing AVeriTeC rows in disposable isolated storage."""
    import importlib.metadata
    started=dt.datetime.now(dt.timezone.utc)
    reference_cutoff=min(cutoff,started+dt.timedelta(seconds=REFERENCE_MAX_SECONDS))
    attempt=started.strftime("%Y%m%dT%H%M%S%fZ")+"_"+str(os.getpid())
    scratch=args.cache_dir.parent/"mod1-transfer-numerical-reference"/attempt
    scratch.mkdir(parents=True,exist_ok=False)
    clone=SimpleNamespace(**vars(args))
    clone.cache_dir=scratch
    clone.output=args.output/"numerical_reference"/attempt
    path=scratch/REFERENCE_NAME
    job=dict(dataset_index=0,dataset_csv=str(args.data_root/"AVeriTeC/dev_filtered_10shot.csv"),
             cache_path=str(path),rows=REFERENCE_ROWS,command=cache_command(clone,0))
    receipt_path=args.output/"remaining_transfer_numerical_reference.json"
    versions={}
    for package in ["torch","transformers","numpy","scipy"]:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    receipt=dict(status="running",started_utc=started.isoformat(),cutoff_utc=reference_cutoff.isoformat(),
                 dataset="AVeriTeC",rows=REFERENCE_ROWS,gpu_index=gpu_index,predeclared_guard=NUMERICAL_GUARD,
                 maximum_reference_gpu_seconds=REFERENCE_MAX_SECONDS,shared_missing_cache_cutoff_utc=cutoff.isoformat(),
                 original_cache=historical_entry,scratch_cache=str(path),command=job["command"],current_package_versions=versions,
                 quality_labels_used_for_numerical_guard=False,quality_auc_computed=False,
                 rationale="Gross-drift check, not bitwise certification: float16 feature quantization and backend/kernel variance permit small differences; require LP max abs <=0.1 and canonical NLL rank rho >=0.999 over all 427 rows. No test-quality AUC or label-based quality chooses these thresholds.")
    atomic_json(receipt_path,receipt)
    def completed(entry,event):
        receipt.update(scratch_cache_inspection=entry,execution=event)
        atomic_json(receipt_path,receipt)
    try:
        execute_cache_jobs([job],[gpu_index],reference_cutoff,clone.output,completed)
        if dt.datetime.now(dt.timezone.utc)>=cutoff:
            raise TimeoutError("Shared transfer deadline reached before numerical comparison")
        receipt["comparison"]=compare_numerical_reference(Path(historical_entry["path"]),path)
        if receipt["comparison"]["status"]!="passed":
            raise ValueError("Historical/current backend numerical gross-drift guard failed")
        if dt.datetime.now(dt.timezone.utc)>=cutoff:
            raise TimeoutError("Shared transfer deadline reached during numerical comparison")
        receipt["status"]="passed"
    except BaseException as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["historical_cache_unchanged"]=sha(Path(historical_entry["path"]))==historical_entry["raw_sha256"]
        receipt["finished_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(receipt_path,receipt)
    if not receipt["historical_cache_unchanged"]:
        raise ValueError("Numerical reference historical cache changed")
    return {**receipt,"receipt_file":str(receipt_path),"receipt_sha256":sha(receipt_path)}


def terminate(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def assign_jobs(jobs, gpu_indices):
    """Keep CounterFact on the first GPU; PopQA then X-Fact on the second."""
    if len(gpu_indices) not in (1, 2) or len(set(gpu_indices)) != len(gpu_indices):
        raise ValueError("Use one GPU or two distinct GPUs")
    queues = {gpu: deque() for gpu in gpu_indices}
    for job in sorted(jobs, key=lambda row: row["dataset_index"]):
        gpu = gpu_indices[0] if len(gpu_indices) == 1 or job["dataset_index"] == 6 else gpu_indices[1]
        queues[gpu].append(job)
    return queues


def execute_cache_jobs(jobs, gpu_indices, cutoff, output, on_success,
                       min_launch_seconds=120):
    """Run one child per GPU, with a shared cutoff and main-thread receipts.

    Workers own and always reap their subprocesses. Main-thread completion
    callbacks are the only receipt writers; atomic_json publishes each updated
    receipt to readers. A failure cancels both lanes before another job starts.
    """
    queues = assign_jobs(jobs, gpu_indices)
    abort = threading.Event()
    completed = set()
    active = {}
    all_futures = []

    def run_one(job, gpu):
        remaining = (cutoff - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if abort.is_set() or remaining < min_launch_seconds:
            raise TimeoutError("Insufficient time before the shared transfer deadline")
        cache = Path(job["cache_path"])
        if cache.exists() or cache.with_suffix(".part").exists():
            raise ValueError("Refusing to overwrite an existing or partial target cache: " + str(cache))
        log = Path(output) / "transfer_completion_logs" / f"dataset_{job['dataset_index']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": "src:.",
               "OMP_NUM_THREADS": "12", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        started = time.monotonic()
        with log.open("a") as stream:
            child = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None:
                    if abort.is_set():
                        raise concurrent.futures.CancelledError("Another transfer lane failed")
                    if dt.datetime.now(dt.timezone.utc) >= cutoff:
                        raise TimeoutError("Shared transfer deadline reached")
                    time.sleep(.05)
                code = child.returncode
            finally:
                terminate(child)
        if code:
            raise RuntimeError("Transfer cache process failed; inspect " + str(log))
        entry = inspect_cache(cache, Path(job["dataset_csv"]), job["rows"])
        event = dict(dataset_index=job["dataset_index"], elapsed_seconds=time.monotonic()-started,
                     command=job["command"], gpu_index=gpu, exit_code=code)
        return entry, event

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_indices))

    def launch(gpu):
        if queues[gpu] and not abort.is_set():
            future = pool.submit(run_one, queues[gpu].popleft(), gpu)
            active[future] = gpu
            all_futures.append(future)

    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Transfer interrupted by signal {signum}")
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
    failure = None
    try:
        for gpu in gpu_indices:
            launch(gpu)
        while active:
            ready, _ = concurrent.futures.wait(active, timeout=.2,
                return_when=concurrent.futures.FIRST_COMPLETED)
            # Check all returned failures before filling a newly free GPU.
            results = []
            for future in ready:
                gpu = active.pop(future)
                results.append((future, gpu, future.result()))
            for future, gpu, (entry, event) in results:
                on_success(entry, event)
                completed.add(future)
            for _, gpu, _ in results:
                launch(gpu)
    except BaseException as exc:
        failure = exc
        abort.set()
        raise
    finally:
        # Each running worker observes abort, terminates its process group and
        # waits for it. No feature child survives a deadline/failure/signal.
        if failure is not None:
            abort.set()
        pool.shutdown(wait=True, cancel_futures=True)
        # Preserve a cache that finished successfully just as the other lane
        # failed. Its identity must be recorded before a possible future retry.
        if failure is not None:
            for future in all_futures:
                if future in completed or future.cancelled():
                    continue
                try:
                    entry, event = future.result()
                    on_success(entry, event)
                    completed.add(future)
                except BaseException:
                    pass
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=Path("/dev/shm/mod1-transfer-cache"))
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/claims")
    ap.add_argument("--model-dir", type=Path, default=Path("/workspace/mod1-models/Llama3.2-1B"))
    ap.add_argument("--tokenizer-dir", type=Path, default=Path("/workspace/mod1-models/llama31-tokenizer"))
    ap.add_argument("--output", type=Path, default=ROOT / "results/overnight20261021")
    ap.add_argument("--download", action="store_true", help="Only fetch/check the pinned public assets; no GPU inference")
    ap.add_argument("--execute", action="store_true", help="Explicitly run the three missing caches on the assigned GPU")
    gpu_args = ap.add_mutually_exclusive_group()
    gpu_args.add_argument("--gpu-index", type=int, help="Run the three caches serially on this GPU")
    gpu_args.add_argument("--gpu-indices", nargs=2, type=int, help="CounterFact on GPU one; PopQA then X-Fact on GPU two")
    ap.add_argument("--max-minutes", type=float, help="Shared runtime cap: default/max 35 for two GPUs; default/max 45 for serial")
    ap.add_argument("--deadline-utc", help="Required for execution; must not extend the authorized pod cutoff")
    args = ap.parse_args()
    os.chdir(ROOT)
    for name in ("cache_dir", "data_root", "model_dir", "tokenizer_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    gpu_indices = args.gpu_indices or ([] if args.gpu_index is None else [args.gpu_index])
    parallel = len(gpu_indices) == 2
    maximum = 35 if parallel else 45
    if args.max_minutes is None:
        args.max_minutes = maximum
    if not 0 < args.max_minutes <= maximum:
        ap.error(f"max-minutes must be in (0,{maximum}] for this GPU mode")
    if any(gpu < 0 for gpu in gpu_indices) or len(set(gpu_indices)) != len(gpu_indices):
        ap.error("GPU indices must be nonnegative and distinct")
    deadline = None
    if args.execute:
        if not gpu_indices or not args.deadline_utc:
            ap.error("Execution requires --gpu-index or --gpu-indices, and --deadline-utc")
        deadline = dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        if deadline.tzinfo is None or deadline.utcoffset() != dt.timedelta(0) or deadline > HARD_DEADLINE:
            ap.error("deadline must be UTC and cannot exceed 2026-10-22T02:16:28Z")
    args.output.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(args.data_root.rglob("*.csv")) if "popqa_train" not in p.name]
    old_manifest = json.loads((ROOT / "out/campaign/transfer_llama32_1b/manifest.json").read_text())
    if len(files) != 9 or {str(p.relative_to(args.data_root)): sha(p) for p in files} != old_manifest["data"]:
        raise ValueError("The nine evaluation CSVs differ from the historical transfer protocol")
    for index, (relative, _, _) in MISSING.items():
        if files[index] != args.data_root / relative:
            raise ValueError("The required dataset-index mapping changed")
    archives = json.loads((ROOT / "results/feature_cache_validation.json").read_text())["archives"]
    historical = [{"path": str(args.cache_dir / Path(name).name.removesuffix(".gz")),
                   "raw_sha256": row["raw_sha256"], "raw_bytes": row["raw_bytes"]}
                  for name, row in archives.items() if "/mod1-transfer-cache/" in name]
    if len(historical) != 6:
        raise ValueError("Exactly six verified historical target caches are required")
    target = pinned_assets("Llama3.2-1B", TARGET_ID, TARGET_REVISION, args.model_dir)
    tokenizer = pinned_assets("llama31", TOKENIZER_ID, TOKENIZER_REVISION, args.tokenizer_dir, tokenizer_only=True)
    jobs = [dict(dataset_index=i, relative_csv=relative, dataset_csv=str(files[i]), cache_path=str(args.cache_dir / name), rows=rows,
                 command=cache_command(args, i)) for i, (relative, name, rows) in MISSING.items()]
    plan = dict(status="prepared_no_inference", target=target, tokenizer=tokenizer,
        model_forward_dtype="float32", attention_implementation="sdpa", expected_embedded_meta=EXPECTED_META,
        source_chat_template="generate_true_statement", feature_storage_dtype="float16",
        no_header_suffix_trim=True, native_target_tokenizer_used=False,
        historical_caches_must_be_preserved=historical, missing_jobs=jobs, missing_rows=41253,
        gpu_indices=gpu_indices, execution_mode="two_gpu_dataset_lanes" if parallel else "serial",
        gpu_assignment=({str(gpu): [j["dataset_index"] for j in queue]
                         for gpu, queue in assign_jobs(jobs, gpu_indices).items()} if gpu_indices else {}),
        estimated_compute_minutes=[11, 20] if parallel else [15, 30], maximum_child_runtime_minutes=args.max_minutes,
        estimate_basis="Historical CounterFact 28896 rows / 590 s; short-training probe 42.07 rows/s; longer X-Fact claims may be slower",
        additional_uncompressed_cache_bytes_approx=6750471644,
        target_model_download_bytes=sum(row["bytes"] for row in target["files"]),
        no_source_8b_weights_downloaded=True, no_model_or_cache_rotation=True,
        numerical_reference_preflight=dict(required_before_missing_jobs=True,dataset_index=0,
            dataset="AVeriTeC",rows=REFERENCE_ROWS,maximum_gpu_seconds=REFERENCE_MAX_SECONDS,
            first_assigned_gpu=True,separate_scratch_cache=True,shared_runtime_cap=True,
            guard=NUMERICAL_GUARD,quality_auc_or_label_based_thresholds=False),
        labels="is_hallu", quality_labels_used_for_model_selection=False,
        protected_source_sha256={name: sha(ROOT / name) for name in (
            "scripts/prepare_remaining_transfer.py", "scripts/evaluate_cached_claims.py",
            "src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py")})
    atomic_json(args.output / "remaining_transfer_plan.json", plan)
    if not (args.download or args.execute):
        print(json.dumps(plan, indent=2))
        return
    verified_assets = dict(target=download_assets(target), tokenizer=download_assets(tokenizer))
    reference_path = ROOT / "results/overnight20261021/remaining_transfer_reference_metadata.json"
    parity = tokenizer_parity(args.tokenizer_dir, reference_path)
    download_receipt = dict(status="downloaded_verified_no_inference", **verified_assets,
                            tokenizer_reference_parity=parity)
    atomic_json(args.output / "remaining_transfer_download.json", download_receipt)
    if not args.execute:
        print(json.dumps(download_receipt, indent=2))
        return
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = min(deadline, now + dt.timedelta(minutes=args.max_minutes))
    protected = []
    for row in historical:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["raw_bytes"] or sha(path) != row["raw_sha256"]:
            raise ValueError("Restore and verify all six historical target caches before completing transfer: " + str(path))
        protected.append({**row, "verified_before": True})
    receipt_path = args.output / "remaining_transfer_completion.json"
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt = dict(status="running", started_utc=now.isoformat(), cutoff_utc=cutoff.isoformat(),
        target=verified_assets["target"], tokenizer=verified_assets["tokenizer"],
        tokenizer_reference_parity=parity, preserved_historical_caches=protected,
        protected_source_sha256=plan["protected_source_sha256"], new_caches=[], events=[],
        gpu_indices=gpu_indices, execution_mode=plan["execution_mode"],
        numerical_reference_preflight=dict(status="not_started",predeclared_guard=NUMERICAL_GUARD),
        labels="is_hallu", model_selection_uses_target_labels=False,
        note="Revision/tokenizer identity is externally pinned; historical embedded meta lacks those identifiers")
    prior = {row["path"]: row for row in old.get("new_caches", [])}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    try:
        reference_entry=next(row for row in protected if Path(row["path"]).name==REFERENCE_NAME)
        try:
            receipt["numerical_reference_preflight"]=run_numerical_reference(args,gpu_indices[0],cutoff,reference_entry)
        finally:
            numerical_path=args.output/"remaining_transfer_numerical_reference.json"
            if numerical_path.exists():
                receipt["numerical_reference_preflight"]={**json.loads(numerical_path.read_text()),
                    "receipt_file":str(numerical_path),"receipt_sha256":sha(numerical_path)}
            atomic_json(receipt_path,receipt)
        pending = []
        for job in jobs:
            cache = Path(job["cache_path"])
            if cache.exists():
                if str(cache) not in prior or sha(cache) != prior[str(cache)]["raw_sha256"]:
                    raise ValueError("Existing new target cache lacks a matching pinned completion receipt; refusing replacement: " + str(cache))
                receipt["new_caches"].append(inspect_cache(cache, files[job["dataset_index"]], job["rows"]))
                atomic_json(receipt_path, receipt)
                continue
            if cache.with_suffix(".part").exists():
                raise ValueError("Unreviewed partial cache already exists: " + str(cache.with_suffix(".part")))
            pending.append(job)

        def record(entry, event):
            receipt["new_caches"].append(entry)
            receipt["events"].append(event)
            atomic_json(receipt_path, receipt)

        execute_cache_jobs(pending, gpu_indices, cutoff, args.output, record)
        receipt["status"] = "complete_three_missing_caches"
    except BaseException as exc:
        receipt.update(status="partial_or_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for row in receipt["preserved_historical_caches"]:
            row["verified_after"] = Path(row["path"]).is_file() and sha(row["path"]) == row["raw_sha256"]
        receipt["all_six_historical_caches_unchanged"] = all(row["verified_after"] for row in receipt["preserved_historical_caches"])
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(receipt_path, receipt)
    if not receipt["all_six_historical_caches_unchanged"]:
        raise ValueError("A protected historical target cache changed")
    print(json.dumps(dict(status=receipt["status"], new_caches=len(receipt["new_caches"]), receipt=str(receipt_path))))


if __name__ == "__main__":
    main()
