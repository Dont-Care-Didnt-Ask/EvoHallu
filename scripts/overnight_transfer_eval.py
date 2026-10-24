"""CPU-only transfer of immutable source-validation winners to cached datasets.

No target features are extracted and no target label is used for selection.
The default six-dataset control remains separate from optional full-nine
coverage. Source/target comparisons always use the same prespecified datasets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "original/SequenceLogProb"
NON_METHOD = {"labels", "duplicate_disjoint_mask", "claim_group_ids"}
GROUPED14 = {"supervised/logistic_documented_claim_group", "supervised/qda_documented_claim_group"}
TARGET_ID = "unsloth/Llama-3.2-1B-Instruct"
TARGET_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
TOKENIZER_ID = "VityaVitalich/Llama3.1-8b-instruct"
TOKENIZER_REVISION = "247cb7c3baebabc46cff1fb97a3a5913e4c4511d"
EXPECTED_META = dict(hidden_size=2048, num_layers=16, num_attention_heads=32,
                     vocab_size=128256, modalities="logits", model_dtype="torch.float32")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def completion_numerical_reference_passed(completion):
    """Full-nine coverage requires the predeclared mixed-runtime numeric check."""
    receipt=completion.get("numerical_reference_preflight",{})
    comparison=receipt.get("comparison",{})
    lp=comparison.get("absolute_difference_summaries",{}).get("sequence_logprobs",{})
    nll=comparison.get("canonical_nll",{})
    maximum=lp.get("max_abs_difference")
    rho=nll.get("spearman_rho")
    return bool(receipt.get("status")=="passed" and receipt.get("rows")==427 and
        receipt.get("historical_cache_unchanged") is True and comparison.get("status")=="passed" and
        comparison.get("all_rows_compared")==427 and
        comparison.get("all_claims_labels_ids_input_ids_attention_masks_shapes_dtypes_exact") is True and
        comparison.get("all_compared_numeric_values_finite") is True and
        receipt.get("quality_labels_used_for_numerical_guard") is False and
        receipt.get("quality_auc_computed") is False and
        comparison.get("labels_used_for_guard") is False and
        comparison.get("quality_auc_computed") is False and
        comparison.get("guard_checks",{}).get("observed_logprob_gross_drift") is True and
        comparison.get("guard_checks",{}).get("canonical_nll_rank_preservation") is True and
        isinstance(maximum,(int,float)) and np.isfinite(maximum) and 0<=maximum<=.1 and
        isinstance(rho,(int,float)) and np.isfinite(rho) and .999<=rho<=1.)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def load_outputs(folder):
    outputs, bootstrap = defaultdict(dict), defaultdict(dict)
    hashes = {}
    for path in sorted(folder.glob("*/*_scores.npz")):
        name = path.name.removesuffix("_scores.npz")
        for key, values in dict(np.load(path)).items():
            if key in outputs[name] and not np.array_equal(outputs[name][key], values):
                raise ValueError(f"Conflicting repeated predictions or row identity: {path}, {key}")
            outputs[name][key] = values
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in sorted(folder.glob("*/*_bootstrap.npz")):
        name = path.name.removesuffix("_bootstrap.npz")
        for key, values in dict(np.load(path)).items():
            old = bootstrap[name].get(key)
            if old is None or len(values) > len(old):
                bootstrap[name][key] = values
    return outputs, bootstrap, hashes


def audit_target_caches(args, names, files, indices):
    """Check embedded metadata, row/input identity and external pinned provenance.

    Historical embedded metadata records architecture and forward dtype but
    not model revision or tokenizer identity. Those are verified externally.
    """
    import torch
    torch.set_num_threads(1)
    completion = json.loads(args.completion_receipt.read_text()) if args.completion_receipt.exists() else None
    historical_path = ROOT / "out/campaign/transfer_llama32_1b/protocol.json"
    historical = json.loads(historical_path.read_text()) if historical_path.exists() else {}
    historical_pin = (historical.get("target_model") == TARGET_ID and
        historical.get("target_revision") == TARGET_REVISION and historical.get("dtype") == "float32" and
        "llama31" in historical.get("tokenizer", "").lower())
    completion_pin = False
    numerical_reference_pin = False
    expected_sha = {}
    if completion is not None:
        target, tokenizer = completion.get("target", {}), completion.get("tokenizer", {})
        parity = completion.get("tokenizer_reference_parity", {})
        completion_pin = (completion.get("status") == "complete_three_missing_caches" and
            target.get("model") == TARGET_ID and target.get("revision") == TARGET_REVISION and
            target.get("all_files_match_pinned_historical_hashes") is True and
            tokenizer.get("model") == TOKENIZER_ID and tokenizer.get("revision") == TOKENIZER_REVISION and
            tokenizer.get("all_files_match_pinned_historical_hashes") is True and
            parity.get("exact_chat_input_ids_match") is True and parity.get("exact_raw_claim_input_ids_match") is True and
            completion.get("all_six_historical_caches_unchanged") is True)
        numerical_reference_pin=completion_numerical_reference_passed(completion)
        if args.coverage=="nine":
            completion_pin=completion_pin and numerical_reference_pin
        if completion_pin:
            for row in completion.get("preserved_historical_caches", []) + completion.get("new_caches", []):
                expected_sha[Path(row["path"]).name] = row["raw_sha256"]
    restored_path = args.directory / "restored_caches.json"
    if restored_path.exists():
        for row in json.loads(restored_path.read_text()):
            if Path(row["target"]).parent.name == "mod1-transfer-cache":
                name = Path(row["target"]).name
                if name in expected_sha and expected_sha[name] != row["raw_sha256"]:
                    raise ValueError(f"Conflicting restored/completion target-cache hashes: {name}")
                expected_sha[name] = row["raw_sha256"]
    if args.coverage == "nine" and not completion_pin:
        raise ValueError("Full-nine transfer requires the completed pinned model/tokenizer receipt, all six historical caches preserved, and passed 427-row mixed-runtime numerical reference")
    if args.coverage == "six" and not (historical_pin or completion_pin):
        raise ValueError("Historical target revision/float32/shared-tokenizer provenance could not be verified")
    rows = []
    for name in names:
        path = args.cache_dir / (name+".pt")
        raw = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
        meta = raw.get("meta", {})
        if any(meta.get(key) != value for key,value in EXPECTED_META.items()):
            raise ValueError(f"Target cache architecture, modality or model dtype differs: {name}: {meta}")
        csv_path = files[indices[name]]
        frame = pd.read_csv(csv_path, usecols=["claim", "is_hallu"])
        samples = raw["samples"]
        if (len(samples) != len(frame) or
            [s["claim"] for s in samples] != frame.claim.astype(str).str.strip().tolist() or
            not np.array_equal([s["label"] for s in samples], frame.is_hallu.to_numpy())):
            raise ValueError(f"Target cache claims/labels do not match CSV order: {name}")
        fingerprint = sha(path)
        reference = expected_sha.get(path.name)
        if reference is not None and reference != fingerprint:
            raise ValueError(f"Target cache SHA differs from pinned restoration/completion receipt: {name}")
        if args.coverage == "nine" and reference is None:
            raise ValueError(f"No raw SHA reference for full-nine target cache: {name}")
        source_path = args.source_cache_dir / path.name
        input_parity = None
        if source_path.exists():
            source = torch.load(source_path,mmap=True,weights_only=False,map_location="cpu")
            if len(source["samples"]) != len(samples):
                raise ValueError(f"Source/target raw-cache row counts differ: {name}")
            input_parity = all(a["claim"] == b["claim"] and a["label"] == b["label"] and
                np.array_equal(a["input_ids"], b["input_ids"]) and
                np.array_equal(a["orig_input_ids"], b["orig_input_ids"])
                for a,b in zip(source["samples"],samples))
            if not input_parity:
                raise ValueError(f"Source/target full input token IDs or row identity differ: {name}")
            del source
        rows.append(dict(dataset=name, cache=str(path), raw_sha256=fingerprint,
            hash_matches_restoration_or_pinned_completion=reference == fingerprint if reference else None,
            meta=meta, samples=len(samples), claims_labels_match_csv_in_order=True,
            all_source_chat_and_raw_token_ids_identical=input_parity, source_csv_sha256=sha(csv_path)))
        del raw, samples
        gc.collect()
    return dict(status="passed", target_model=TARGET_ID, target_revision=TARGET_REVISION,
        model_forward_dtype="float32", stored_features_may_be_float16=True,
        shared_tokenizer=TOKENIZER_ID, tokenizer_revision=TOKENIZER_REVISION,
        embedded_revision_and_tokenizer_available=False,
        revision_tokenizer_evidence="External historical pinned extraction protocol; full-nine completion verifies pinned model/tokenizer file hashes and historical tokenization parity",
        historical_protocol_sha256=sha(historical_path) if historical_path.exists() else None,
        historical_protocol_pin_verified=historical_pin,
        completion_receipt_sha256=sha(args.completion_receipt) if completion is not None else None,
        completion_pin_verified=completion_pin,numerical_reference_preflight_verified=numerical_reference_pin,
        numerical_reference_receipt_sha256=(completion.get("numerical_reference_preflight",{}).get("receipt_sha256") if completion else None),per_dataset=rows)


def summarize(args, protocol, output, identities, failures):
    source, source_boot, source_hashes = load_outputs(args.directory / "evaluations")
    target, target_boot, target_hashes = load_outputs(output)
    names = protocol["prespecified_transfer_subset"]
    count_expected = len(names)
    coverage_word = "nine" if count_expected == 9 else "six"
    complete_status = coverage_word + "_dataset_transfer_complete"
    estimator_failures = []
    for path in sorted(output.glob("*/*.json")):
        if path.stem not in names:
            continue
        result = json.loads(path.read_text())
        estimator_failures.extend(dict(dataset=path.stem, method=method, error=error)
                                  for method,error in result.get("failed", {}).items()
                                  if method in identities or method in GROUPED14 or method == BASELINE)
    methods = [BASELINE] + sorted(identities) + (sorted(GROUPED14) if args.fitted_models.exists() else [])
    results = []
    for method in methods:
        per_dataset = []
        for name in names:
            if name not in target or method not in target[name]:
                continue
            z = target[name]
            target_auc = 100 * float(roc_auc_score(z["labels"], z[method]))
            row = dict(dataset=name, rows=len(z["labels"]), target_directed_auc_percent=target_auc,
                       target_paper_auc_percent=max(target_auc, 100-target_auc), source_available=False)
            if name in source and method in source[name]:
                before = source[name]
                for key in NON_METHOD:
                    if key not in before or key not in z or not np.array_equal(before[key], z[key]):
                        raise ValueError(f"Source/target row order, labels or claim grouping differ: {name}, {key}")
                source_auc = 100 * float(roc_auc_score(before["labels"], before[method]))
                row.update(source_available=True, row_identity_verified=True, source_directed_auc_percent=source_auc,
                           source_paper_auc_percent=max(source_auc, 100-source_auc),
                           target_minus_source_directed_pp=target_auc-source_auc,
                           target_minus_source_paper_pp=max(target_auc, 100-target_auc)-max(source_auc, 100-source_auc))
            per_dataset.append(row)
        result = dict(method=method, identity=identities.get(method), expected_subset_datasets=len(names),
                      target_datasets_scored=len(per_dataset), matched_source_target_datasets=sum(r["source_available"] for r in per_dataset),
                      full_nine_dataset_macro=count_expected == 9 and len(per_dataset) == 9, per_dataset=per_dataset)
        complete = count_expected in {6, 9} and len(per_dataset) == count_expected
        result["status"] = complete_status if complete and all(r["source_available"] for r in per_dataset) else "pending_or_failed"
        if complete:
            result["target_matched_subset_directed_macro_auc_percent"] = float(np.mean([r["target_directed_auc_percent"] for r in per_dataset]))
            result["target_matched_subset_paper_macro_auc_percent"] = float(np.mean([r["target_paper_auc_percent"] for r in per_dataset]))
            for measure in ("directed", "paper"):
                result[f"target_{coverage_word}_dataset_{measure}_macro_auc_percent"] = result[f"target_matched_subset_{measure}_macro_auc_percent"]
        if result["status"] == complete_status:
            result["source_matched_subset_directed_macro_auc_percent"] = float(np.mean([r["source_directed_auc_percent"] for r in per_dataset]))
            result["source_matched_subset_paper_macro_auc_percent"] = float(np.mean([r["source_paper_auc_percent"] for r in per_dataset]))
            for measure in ("directed", "paper"):
                result[f"source_same_{coverage_word}_{measure}_macro_auc_percent"] = result[f"source_matched_subset_{measure}_macro_auc_percent"]
            result["target_minus_source_directed_pp"] = float(np.mean([r["target_minus_source_directed_pp"] for r in per_dataset]))
            result["target_minus_source_paper_pp"] = float(np.mean([r["target_minus_source_paper_pp"] for r in per_dataset]))
            if all(method in source_boot[n] and method in target_boot[n] and len(source_boot[n][method]) and len(target_boot[n][method]) for n in names):
                count = min(min(len(source_boot[n][method]), len(target_boot[n][method])) for n in names)
                s = np.asarray([source_boot[n][method][:count] for n in names]) * 100
                t = np.asarray([target_boot[n][method][:count] for n in names]) * 100
                result["paired_claim_bootstrap"] = dict(samples=count,
                    target_minus_source_directed_ci95_pp=np.nanpercentile((t-s).mean(axis=0), [2.5, 97.5]).tolist(),
                    target_minus_source_paper_ci95_pp=np.nanpercentile((np.maximum(t,100-t)-np.maximum(s,100-s)).mean(axis=0), [2.5,97.5]).tolist())
        results.append(result)
    groups = defaultdict(list)
    for row in results:
        if row["identity"] and row["status"] == complete_status:
            identity = row["identity"]
            groups[(identity["phase"], identity["policy"], identity["writer_model"])].append(row)
    grouped = []
    for (phase, policy, writer), rows in sorted(groups.items()):
        scores = [r["target_matched_subset_paper_macro_auc_percent"] for r in rows]
        deltas = [r["target_minus_source_paper_pp"] for r in rows]
        group = dict(phase=phase, policy=policy, writer_model=writer,
            seeds=[r["identity"]["seed"] for r in rows],
            subset_size=count_expected,
            target_matched_subset_paper_macro_mean=float(np.mean(scores)),
            target_matched_subset_paper_macro_seed_std=float(np.std(scores,ddof=1)) if len(scores)>1 else None,
            target_minus_source_paper_pp_mean=float(np.mean(deltas)))
        group[f"target_{coverage_word}_dataset_paper_macro_mean"] = group["target_matched_subset_paper_macro_mean"]
        group[f"target_{coverage_word}_dataset_paper_macro_seed_std"] = group["target_matched_subset_paper_macro_seed_std"]
        grouped.append(group)
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_feature_model="Llama3.1-8b-instruct", target_feature_model="Llama3.2-1B-Instruct",
        tokenizer="Original shared Llama3.1 tokenizer retained by preexisting transfer extraction",
        labels="is_hallu", target_labels_used_for_selection=False, fitting_on_target=False,
        selection="Same immutable source-validation primary winners and same source-fitted documented14 heads",
        reporting=f"Per-dataset max(AUC,100-AUC), followed by equal-weight average over the prespecified {count_expected} datasets",
        coverage=coverage_word, prespecified_transfer_subset=names, subset_size=count_expected, nine_dataset_benchmark_size=9,
        full_nine_dataset_macro_available=any(r["full_nine_dataset_macro"] for r in results),
        source_prediction_sha256=source_hashes,
        target_prediction_sha256=target_hashes, failures=failures, estimator_failures=estimator_failures,
        target_cache_audit=protocol.get("target_cache_audit"),
        primary_budget_rule=protocol["primary_budget_rule"],
        excluded_partial_primary_identities=protocol["excluded_partial_primary_identities"],
        methods=results, all_seed_primary_groups=grouped,
        uncertainty="Paired claim-bootstrap differences reuse matching deterministic resamples after verifying source/target row identity; intervals condition on frozen programs")
    dump(output / "transfer_report.json", report)
    description = ("This is a **six-of-nine dataset transfer control**, not the full paper benchmark."
                   if count_expected == 6 else "This is a **nine-of-nine dataset feature-model transfer control**. The target feature model differs from the paper's source feature model.")
    lines = ["# Frozen-program transfer to Llama3.2-1B", "",
             description + f" Source and target averages use exactly the same {count_expected} datasets. No target labels are used for program selection or fitting.", "",
             "Source features: Llama3.1-8b-instruct. Target features: Llama3.2-1B-Instruct caches, using the original shared tokenizer.", "",
             f"Only full-budget source-selected primary identities are transferred. Excluded partial/unverified primary identities: {len(protocol['excluded_partial_primary_identities'])}; details remain in the protocol.", "",
             f"| Frozen source-selected method | Target datasets | Matched source datasets | Source same-{count_expected} paper AUROC | Target same-{count_expected} paper AUROC | Target − source |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value):
        return "pending" if value is None else f"{value:.3f}"
    for row in results:
        lines.append(f"| {row['method']} | {row['target_datasets_scored']}/{count_expected} | {row['matched_source_target_datasets']}/{count_expected} | {fmt(row.get('source_matched_subset_paper_macro_auc_percent'))} | {fmt(row.get('target_matched_subset_paper_macro_auc_percent'))} | {fmt(row.get('target_minus_source_paper_pp'))} |")
    lines += ["", "Prespecified datasets:", ""] + ["- `" + n + "`" for n in names]
    if failures:
        lines += ["", "Execution failures:", ""] + ["- " + str(f) for f in failures]
    if estimator_failures:
        lines += ["", "Estimator failures on target datasets:", ""] + ["- " + str(f) for f in estimator_failures]
    (output / "TRANSFER_REPORT.md").write_text("\n".join(lines)+"\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, default=ROOT / "results/overnight20261021")
    ap.add_argument("--registry", type=Path)
    ap.add_argument("--coverage", choices=["six", "nine"], default="six")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/claims")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/restored/mod1-transfer-cache")
    ap.add_argument("--source-cache-dir", type=Path, default=ROOT / "data/restored/mod1-eval-cache")
    ap.add_argument("--tokenizer-dir", type=Path, default=Path("/workspace/mod1-models/llama31-tokenizer"),
                    help="Actual shared tokenizer directory recorded in evaluation provenance; cached evaluation does not load it")
    ap.add_argument("--completion-receipt", type=Path, default=ROOT / "results/overnight20261021/remaining_transfer_completion.json")
    ap.add_argument("--cache-manifest", type=Path, default=ROOT / "data/cache_backup_manifest.json")
    ap.add_argument("--fitted-models", type=Path, default=ROOT / "out/research20261021/supervised_grouped/models.joblib")
    ap.add_argument("--kind", choices=["new", "supervised", "all"], default="all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--max-seconds", type=float, default=1800)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    args.directory = args.directory.resolve()
    args.registry = args.registry or args.directory / "frozen_manifest.json"
    registry = json.loads(args.registry.read_text())
    all_primary = [row for row in registry["methods"] if row["role"] == "primary"]
    selected = [row for row in all_primary if row.get("full_budget") is True]
    excluded_partial = [row for row in all_primary if row.get("full_budget") is not True]
    identities = {row["method"]:row for row in selected}
    backup = json.loads(args.cache_manifest.read_text())
    archived_names = sorted(Path(name).name.removesuffix(".pt.gz") for name in backup if Path(name).parent.name == "mod1-transfer-cache")
    if len(archived_names) != 6:
        raise ValueError(f"Expected the prespecified six archived target caches, found {len(archived_names)}")
    files = [p for p in sorted(args.data_root.rglob("*.csv")) if "popqa_train" not in p.name]
    if len(files) != 9:
        raise ValueError("Expected nine source CSVs")
    indices = {p.parent.name + "__" + p.stem:i for i,p in enumerate(files)}
    if not set(archived_names) <= set(indices):
        raise ValueError("Archived target cache names do not match the benchmark datasets")
    names = sorted(indices) if args.coverage == "nine" else archived_names
    available = [name for name in names if (args.cache_dir / (name+".pt")).exists()]
    if args.execute and args.coverage == "nine" and len(available) != 9:
        raise ValueError(f"Full-nine transfer requires all nine target caches; missing {sorted(set(names)-set(available))}")
    audit = audit_target_caches(args, available, files, indices) if args.execute else dict(status="not_run_dry_plan")
    identity = dict(methods=[{k:r[k] for k in ("method","source_sha256")} for r in selected],
                    documented14_sha256=sha(args.fitted_models) if args.fitted_models.exists() else None,
                    transfer_datasets=names)
    cohort = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    output = args.directory / "transfer_llama32_1b" / cohort
    snapshot = args.directory / "transfer_frozen" / cohort
    for row in selected:
        source = ROOT / row["staged_file"]
        if sha(source) != row["source_sha256"]:
            raise ValueError(f"Frozen source identity changed: {source}")
        dest = snapshot / row["method"].removeprefix("new/") / "frozen_estimator.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != row["source_sha256"]:
            raise ValueError(f"Transfer snapshot identity changed: {dest}")
        if not dest.exists():
            dest.write_bytes(source.read_bytes())
    protocol = dict(cohort_id=cohort, frozen_registry_sha256=sha(args.registry),
        selection_uses_target_labels=False, primary_only_flag_used=False, all_source_seeds_preserved=True,
        source_method_identities=selected, grouped14_models_sha256=identity["documented14_sha256"],
        primary_budget_rule="Only full_budget=true source-validation primary identities; partial/unverified budgets remain exploratory and are not transferred",
        excluded_partial_primary_identities=excluded_partial,
        cache_archive_manifest_sha256=sha(args.cache_manifest),
        target_cache_audit=audit,
        shared_tokenizer_directory=str(args.tokenizer_dir), shared_tokenizer_model=TOKENIZER_ID,
        shared_tokenizer_revision=TOKENIZER_REVISION,
        coverage=args.coverage, prespecified_transfer_subset=names, target_caches_available=available,
        absent_target_cache_names=[n for n in names if n not in available],
        intentionally_unavailable_benchmark_datasets=sorted(set(indices)-set(names)),
        data_sha256={str(p.relative_to(args.data_root)):sha(p) for p in files}, bootstrap_samples=args.bootstrap,
        bootstrap_unit="claim", full_nine_dataset_macro=args.coverage == "nine")
    commands = []
    for kind in (["new","supervised"] if args.kind == "all" else [args.kind]):
        if kind == "new" and not selected:
            continue
        if kind == "supervised" and not args.fitted_models.exists():
            continue
        for name in available:
            dest = output / kind
            command = [sys.executable, "scripts/evaluate_cached_claims.py", "--mode", "evaluate",
                "--dataset-index", str(indices[name]), "--data-root", str(args.data_root),
                "--cache-dir", str(args.cache_dir), "--output", str(dest), "--cpu-threads", "1",
                "--bootstrap", str(args.bootstrap), "--bootstrap-unit", "claim",
                "--model-id", "unsloth/Llama-3.2-1B-Instruct", "--tokenizer-id", str(args.tokenizer_dir),
                "--frozen-glob", str((snapshot / "*/frozen_estimator.py").relative_to(ROOT)),
                "--method-prefix", kind+"/"]
            if kind == "supervised":
                command += ["--extra-fitted-models", str(args.fitted_models)]
            commands.append(dict(dataset=name,kind=kind,command=command,output=str(dest.relative_to(ROOT))))
    dump(output / "transfer_protocol.json", protocol)
    dump(output / "commands.json", dict(execute=args.execute,commands=commands))
    failures = []
    deadline = time.monotonic() + args.max_seconds
    if args.execute:
        def evaluate(job):
            dest = ROOT / job["output"]
            dest.mkdir(parents=True,exist_ok=True)
            result = dest / (job["dataset"]+".json")
            expected = set(identities) if job["kind"] == "new" else GROUPED14
            if result.exists():
                existing = json.loads(result.read_text())
                if expected <= set(existing["methods"]) | set(existing["failed"]):
                    return
                raise ValueError(f"Existing target dataset has a different frozen cohort: {result}")
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Transfer scoring deadline reached")
            with (dest / (job["dataset"]+".log")).open("w") as stream:
                child = subprocess.Popen(job["command"],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,"PYTHONPATH":"src:.","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
                try:
                    code = child.wait(timeout=remaining)
                    if code:
                        raise RuntimeError(f"Target evaluator exited {code}; see {job['output']}/{job['dataset']}.log")
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL)
                            child.wait()
                    raise
            print(json.dumps(dict(completed=job["dataset"],kind=job["kind"])),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(evaluate,job):job for job in commands}
            for future in concurrent.futures.as_completed(pending):
                try:
                    future.result()
                except Exception as exc:
                    job = pending[future]
                    failures.append(dict(dataset=job["dataset"],kind=job["kind"],error=f"{type(exc).__name__}: {exc}"))
    report = summarize(args,protocol,output,identities,failures)
    print(json.dumps(dict(executed=args.execute,cohort_id=cohort,frozen_primary_methods=len(selected),
        available_target_caches=len(available),planned_target_subset=len(names),benchmark_datasets=9,
        coverage=args.coverage,
        completed_transfer_methods=sum(r["status"] in {"six_dataset_transfer_complete","nine_dataset_transfer_complete"} for r in report["methods"]),
        failures=failures,output=str(output.relative_to(ROOT)),commands=[] if args.execute else commands),indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
