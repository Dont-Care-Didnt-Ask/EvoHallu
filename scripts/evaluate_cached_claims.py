"""Cache held-out features once and evaluate frozen formulas on all nine datasets.

GPU microbatches and estimator scoring chunks are separate: scoring always uses
512 samples because several supplied estimators normalize across the batch.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import inspect
import json
import logging
import resource
import signal
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.metrics import roc_auc_score


def weighted_auc_bootstrap(labels, scores, counts, order=None, starts=None):
    """AUC with bootstrap multiplicities, including exact handling of ties."""
    if order is None:
        order = np.argsort(scores, kind="stable")
    if starts is None:
        values = scores[order]
        starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    positives = np.add.reduceat(counts[order] * labels[order], starts)
    negatives = np.add.reduceat(counts[order] * (1 - labels[order]), starts)
    denom = positives.sum() * negatives.sum()
    if not denom:
        return float("nan")
    return float(np.sum(positives * (np.cumsum(negatives) - negatives * 0.5)) / denom)


def bootstrap_counts(group_ids, rng, unit="claim"):
    if unit == "row":
        n = len(group_ids)
        return np.bincount(rng.integers(0, n, n), minlength=n)
    n = int(np.max(group_ids)) + 1
    groups = np.bincount(rng.integers(0, n, n), minlength=n)
    return groups[group_ids]


def load_estimators(paths, eval_meta=None, extra_fitted_paths=()):
    from packing.evaluate.uncertainty_quantification import found_formulas
    from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
    from experiments.extension_trial import parse_candidate
    estimators = {}
    provenance = {}
    for name, cls in inspect.getmembers(found_formulas, inspect.isclass):
        if cls.__module__ == found_formulas.__name__:
            estimators["original/" + name] = cls({"supervised": False, "meta": {"eval_meta": eval_meta or {}}})
    from packing.evaluate.uncertainty_quantification import alignment_baselines
    for name, cls in inspect.getmembers(alignment_baselines, inspect.isclass):
        if cls.__module__ == alignment_baselines.__name__:
            estimators["alignment/" + name] = cls()
    for path in paths:
        name = "new/" + path.parent.name
        text = path.read_text()
        code, imports = parse_candidate(text)
        namespace = {"Estimator": Estimator, "np": np, "scipy": scipy,
                     "Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple, "Optional": Optional}
        exec(imports + "\n" + code, namespace)
        estimators[name] = namespace["HallucinationDetector"]({"supervised": False, "meta": {"eval_meta": eval_meta or {}}})
        provenance[name] = {"file": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    import joblib
    from packing.evaluate.uncertainty_quantification.supervised_baselines import FittedDetector
    for fitted_path in sorted(set(Path("out/campaign").glob("supervised*/models.joblib")) | set(extra_fitted_paths)):
        for name, model in joblib.load(fitted_path).items():
            estimators["supervised/" + name] = FittedDetector(model)
            provenance["supervised/" + name] = {"file": str(fitted_path),
                "sha256": hashlib.sha256(fitted_path.read_bytes()).hexdigest()}
    return estimators, provenance


def primary_paths(paths):
    """Select writer seeds using source-model validation scores only."""
    selected = {}
    for path in paths:
        summary = json.loads((path.parent / "summary.json").read_text())
        config_path = path.parent / "trial_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config["search_mode"] != "evolution" or config["task"].get("split_strategy", "stratified") != "stratified" or "pilot" in path.parent.name:
                continue
            group = config["model"]["hf_model_id"]
        elif summary.get("generator") == "fixed_feature_linear_evolution":
            group = "fixed_linear_features"
        else:
            continue
        score = summary["selected_validation_auc_percent"]
        if group not in selected or score > selected[group][0]:
            selected[group] = score, path
    return sorted(value[1] for value in selected.values())


def trim_header_suffix(samples, count):
    if not count:
        return samples
    result = []
    for sample in samples:
        if len(sample["input_ids"]) <= count + 1:
            raise ValueError("Suffix trim would remove the complete input")
        item = dict(sample)
        for key in ["input_ids", "attention_mask", "sequence_logprobs", "greedy_logprobs", "token_entropies", "top_k_logits", "top_k_indices"]:
            if key in item:
                item[key] = item[key][:-count]
        result.append(item)
    return result


def score_with_timeout(estimator, stats, seconds):
    def expired(signum, frame):
        raise TimeoutError(f"Estimator exceeded {seconds:g}s on one scoring chunk")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return np.asarray(estimator(copy.deepcopy(stats)), dtype=np.float64)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/claims"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/dev/shm/mod1-eval-cache"))
    ap.add_argument("--output", type=Path, default=Path("out/campaign/evaluation"))
    ap.add_argument("--model-id", default="/workspace/mod1-models/llama31")
    ap.add_argument("--tokenizer-id", help="Explicit shared tokenizer for a feature-model transfer control")
    ap.add_argument("--mode", choices=["cache", "evaluate", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--bootstrap-unit", choices=["claim", "row"], default="claim")
    ap.add_argument("--bootstrap-original", choices=["all", "shortlist"], default="shortlist")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--frozen-glob", default="out/logs/mod1/*/frozen_estimator.py")
    ap.add_argument("--method-prefix", help="Score this method prefix and the shared NLL reference only")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--dataset-index", type=int)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--gc-interval", type=int, default=1)
    ap.add_argument("--cuda-cache-interval", type=int, default=1)
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--extra-frozen-glob", action="append", default=[], help="Explicit supplementary frozen methods after primary validation selection")
    ap.add_argument("--extra-fitted-models", action="append", type=Path, default=[])
    ap.add_argument("--trim-header-suffix", type=int, default=0)
    ap.add_argument("--chunk-timeout", type=float, default=120)
    ap.add_argument("--memory-gb", type=float, default=32)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(args.cpu_threads)
    if args.mode == "evaluate" and args.memory_gb:
        # CPU workers can share mapped caches while limiting temporary arrays.
        # Never impose a virtual-address cap on CUDA feature-cache processes.
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = int(args.memory_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (min(limit, hard) if hard > 0 else limit, hard))
    files = [p for p in sorted(args.data_root.rglob("*.csv")) if "popqa_train" not in p.name]
    if len(files) != 9:
        raise ValueError(f"Expected exactly 9 held-out datasets, found {len(files)}")
    selected_files = files if args.dataset_index is None else [files[args.dataset_index]]
    frozen_paths = [p for p in sorted(Path(".").glob(args.frozen_glob))
                    if not (p.parent / "exclude_from_evaluation.json").exists()]
    if args.primary_only:
        frozen_paths = primary_paths(frozen_paths)
    for pattern in args.extra_frozen_glob:
        frozen_paths.extend(sorted(Path(".").glob(pattern)))
    frozen_paths = sorted(set(frozen_paths))
    provenance = {"new/" + path.parent.name: {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in frozen_paths}
    for path in sorted(set(Path("out/campaign").glob("supervised*/models.joblib")) | set(args.extra_fitted_models)):
        provenance[str(path)] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"frozen_estimators": provenance, "feature_model": args.model_id, "dtype": args.dtype,
                "tokenizer": args.tokenizer_id or args.model_id, "primary_only": args.primary_only,
                "trimmed_header_suffix_tokens": args.trim_header_suffix,
                "scoring_chunk_timeout_seconds": args.chunk_timeout,
                "cpu_worker_address_space_GiB": args.memory_gb if args.mode == "evaluate" else None,
                "scoring_chunk_size": 512, "bootstrap_samples": args.bootstrap,
                "bootstrap_unit": args.bootstrap_unit,
                "original_bootstrap_policy": args.bootstrap_original,
                "method_prefix": args.method_prefix,
                "data": {str(p.relative_to(args.data_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = tokenizer = None
    if args.mode in {"cache", "both"}:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from packing.evaluate.uncertainty_quantification.precompute_raw_stats_separated import _precompute_split
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=getattr(torch, args.dtype),
                                                   attn_implementation="sdpa").eval().to("cuda:0")
        for path in selected_files:
            name = path.parent.name + "__" + path.stem
            dest = args.cache_dir / (name + ".pt")
            if dest.exists():
                continue
            df = pd.read_csv(path, usecols=["claim", "is_hallu"])
            if args.max_samples:
                df = df.head(args.max_samples)
            df.claim = df.claim.astype(str).str.strip()
            if not df.claim.ne("").all():
                raise ValueError(f"Empty claims: {path}")
            print(f"Caching {name}: {len(df)} samples", flush=True)
            raw = _precompute_split(df, tokenizer, model, "claim", "is_hallu", None, args.batch_size,
                                    "generate_true_statement", 512, 1, modalities="logits",
                                    max_batch_tokens=args.max_batch_tokens, gc_interval=args.gc_interval,
                                    cuda_cache_interval=args.cuda_cache_interval)["logits_struct"]
            tmp = dest.with_suffix(".part")
            torch.save(raw, tmp)
            tmp.replace(dest)
            del raw
            print(f"Cached {dest} ({dest.stat().st_size / 1024**3:.2f} GiB)", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()
    if args.mode == "cache":
        return
    from packing.evaluate.uncertainty_quantification.features import make_all_stats
    from packing.evaluate.uncertainty_quantification.supervised_baselines import (
        FittedDetector, compact_features, DOCUMENTED_FEATURE_INDICES,
    )
    training = pd.read_csv(args.data_root / "popqa/popqa_train_filtered_10shot.csv", usecols=["claim"])
    train_claims = set(training.claim.fillna("").astype(str).str.split().str.join(" ").str.casefold())
    aggregate = {}
    for path in selected_files:
        name = path.parent.name + "__" + path.stem
        raw = torch.load(args.cache_dir / (name + ".pt"), mmap=True, weights_only=False, map_location="cpu")
        # Match separate benchmark invocations: start every estimator fresh for
        # each dataset, retaining only the separately fitted supervised models.
        estimators, _ = load_estimators(frozen_paths, raw["meta"], args.extra_fitted_models)
        if args.method_prefix:
            estimators = {key: estimator for key, estimator in estimators.items()
                          if key == "original/SequenceLogProb" or key.startswith(args.method_prefix)}
            if len(estimators) < 2:
                raise ValueError("Requested new method is missing")
        if args.primary_only:
            original_file = Path("results/validation_controls/original_formulas.json")
            originals = {"SequenceLogProb", "Perplexity", "MeanTokenEntropy"}
            if original_file.exists():
                originals.add(json.loads(original_file.read_text())["protocols"]["stratified"]["frozen_top_three_by_validation"][0])
            estimators = {name: estimator for name, estimator in estimators.items() if
                          name.startswith("new/") or name in {"supervised/logistic_stratified", "supervised/qda_stratified",
                          "supervised/logistic_documented_stratified", "supervised/qda_documented_stratified",
                          "alignment/ClaimOnlySequenceLogProb", "alignment/LegacyTailCropSequenceLogProb"} or
                          name in {"original/"+n for n in originals}}
        scores = {key: [] for key in estimators}
        failed = {}
        labels, strict, claim_names = [], [], []
        for start in range(0, len(raw["samples"]), 512):
            samples = trim_header_suffix(raw["samples"][start:start+512], args.trim_header_suffix)
            stats, y, _, _ = make_all_stats({"samples": samples, "meta": raw["meta"]}, input_regime="logits")
            stats["input_tokens"] = [s["input_ids"].tolist() for s in samples]
            labels.extend(y)
            normalized = [" ".join(s["claim"].casefold().split()) for s in samples]
            claim_names.extend(normalized)
            strict.extend(claim not in train_claims for claim in normalized)
            fitted_features = {}
            for key, estimator in estimators.items():
                if key in failed:
                    continue
                try:
                    # Candidate programs may use in-place NumPy operations.
                    # Each estimator receives its own input copy.
                    if isinstance(estimator, FittedDetector):
                        feature_set = estimator.feature_set
                        if feature_set not in fitted_features:
                            fitted_features[feature_set] = compact_features(stats, feature_set)
                            if feature_set == "full":
                                fitted_features["documented"] = fitted_features["full"][:, DOCUMENTED_FEATURE_INDICES]
                        # Trusted fitted heads share feature calculations while
                        # retaining a private array for each classifier call.
                        features = fitted_features[feature_set]
                        result = score_with_timeout(lambda unused: estimator.score_features(features.copy(order="K")),
                                                    {}, args.chunk_timeout)
                    else:
                        result = score_with_timeout(estimator, stats, args.chunk_timeout)
                    if result.shape != (len(samples),) or not np.isfinite(result).all():
                        raise ValueError("nonfinite scores or wrong output shape")
                    scores[key].extend(result)
                except Exception as exc:
                    failed[key] = f"{type(exc).__name__}: {exc}"
        y = np.asarray(labels, dtype=int)
        strict = np.asarray(strict, dtype=bool)
        group_ids, unique_claims = pd.factorize(np.asarray(claim_names, dtype=object), sort=False)
        good = {k: np.asarray(v) for k, v in scores.items() if k not in failed}
        np.savez_compressed(args.output / (name + "_scores.npz"), labels=y, duplicate_disjoint_mask=strict,
                            claim_group_ids=group_ids, **good)
        baseline = "original/SequenceLogProb"
        result = {"rows": len(y), "duplicate_disjoint_rows": int(strict.sum()), "failed": failed, "methods": {},
                  "feature_model_metadata": raw.get("meta", {})}
        result.update(unique_normalized_claims=len(unique_claims), bootstrap_unit=args.bootstrap_unit)
        # Shared bootstrap samples across all methods enable paired deltas.
        shortlist_file = Path("results/validation_controls/original_formulas.json")
        original_shortlist = {"original/SequenceLogProb"}
        if shortlist_file.exists():
            shortlist = json.loads(shortlist_file.read_text())
            for protocol in shortlist["protocols"].values():
                original_shortlist.update("original/" + key for key in protocol["frozen_top_three_by_validation"])
        bootstrap_keys = [key for key in good if args.bootstrap_original == "all" or
                          not key.startswith("original/") or key in original_shortlist]
        boot = {key: [] for key in bootstrap_keys}
        prepared = {}
        for key in bootstrap_keys:
            pred = good[key]
            order = np.argsort(pred, kind="stable")
            prepared[key] = order, np.r_[0, np.flatnonzero(np.diff(pred[order])) + 1]
        dataset_seed = 441 + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(dataset_seed)
        for _ in range(args.bootstrap):
            counts = bootstrap_counts(group_ids, rng, args.bootstrap_unit)
            for key in bootstrap_keys:
                boot[key].append(weighted_auc_bootstrap(y, good[key], counts, *prepared[key]))
        baseline_auc = roc_auc_score(y, good[baseline])
        for key, pred in good.items():
            auc = float(roc_auc_score(y, pred))
            row = {"auc_percent": 100 * auc, "gain_pp": 100 * (auc - baseline_auc),
                   "duplicate_disjoint_auc_percent": 100 * float(roc_auc_score(y[strict], pred[strict]))}
            if args.bootstrap and key in boot:
                b = np.asarray(boot[key])
                delta = b - np.asarray(boot[baseline])
                row["auc_ci95_percent"] = (100 * np.nanpercentile(b, [2.5, 97.5])).tolist()
                row["paired_gain_ci95_pp"] = (100 * np.nanpercentile(delta, [2.5, 97.5])).tolist()
            result["methods"][key] = row
        if args.bootstrap:
            np.savez_compressed(args.output / (name + "_bootstrap.npz"),
                                **{key: np.asarray(values) for key, values in boot.items()})
        aggregate[name] = result
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if args.dataset_index is None:
            (args.output / "all_datasets.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"dataset": name, "rows": len(y), "methods": len(good), "failed": failed}), flush=True)
        del raw


if __name__ == "__main__":
    main()
