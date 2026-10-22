import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import DATASETS, DATASET_ORDER, FIELDS, MODELS
from .io import read_json, sha256, write_json
from .methods import SOURCE_HASHES, load_methods
from .report import is_complete, write_report


def read_cache(manifest_path, view):
    if view not in ("full", "claim"):
        raise ValueError("Choose the full or claim token view")
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    model, dataset = manifest["model_key"], manifest["dataset_key"]
    pin = DATASETS[dataset]
    model_id, revision = MODELS[model]
    full_rows = manifest["expected_rows"] == pin["rows"]
    full = full_rows and not manifest.get("pilot_only", False)
    valid = (
        manifest["schema"] == "evohallu.logits_cache.v1" and manifest["status"] == "complete"
        and manifest["model_id"] == model_id and manifest["model_revision"] == revision
        and manifest["dataset_sha256"] == pin["sha256"]
        and manifest["archive_member"] == pin["member"] and manifest["top_k"] == 512
        and 0 < manifest["expected_rows"] <= pin["rows"]
        and manifest["cached_rows"] == manifest["expected_rows"]
        and manifest["source_rows"] == pin["rows"]
        and manifest["source_positive"] == pin["is_hallu_1"] and bool(manifest["shards"])
    )
    if not valid:
        raise ValueError(f"Incomplete or incompatible cache: {manifest_path}")

    prefix = "claim_" if view == "claim" else ""
    stats, id_parts, label_parts = {field: [] for field in FIELDS}, [], []
    cursor = 0
    for entry in manifest["shards"]:
        path = manifest_path.parent / entry["path"]
        if entry["start_row"] != cursor or sha256(path) != entry["sha256"]:
            raise ValueError(f"Shard order or checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            ids, labels = shard["row_ids"], shard["labels"]
            n = len(ids)
            expected_ids = [f"{dataset}:{i}" for i in range(cursor, cursor + n)]
            if (ids.tolist() != expected_ids or labels.shape != (n,)
                    or not np.isin(labels, [0, 1]).all() or n != entry["rows"]
                    or entry["stop_row"] != cursor + n):
                raise ValueError(f"Invalid shard IDs, labels or row range: {path}")
            for field in FIELDS:
                name = prefix + field
                values, offsets = shard[name + "_values"], shard[name + "_offsets"]
                valid = (offsets.shape == (n + 1,) and offsets.dtype.kind in "iu"
                         and offsets[0] == 0 and np.all(np.diff(offsets.astype(np.int64)) > 0)
                         and offsets[-1] == len(values) and values.dtype.kind == "f"
                         and np.isfinite(values).all()
                         and (values.ndim == 2 and values.shape[1] == 512
                              if field == "top_k_logits" else values.ndim == 1))
                if not valid:
                    raise ValueError(f"Invalid {name} arrays: {path}")
                stats[field].extend(values[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:]))
            id_parts.append(ids)
            label_parts.append(labels)
            cursor += n

    ids, labels = np.concatenate(id_parts), np.concatenate(label_parts)
    cohort = hashlib.sha256(json.dumps([ids.tolist(), labels.tolist()]).encode()).hexdigest()
    lengths = zip(*(map(len, stats[field]) for field in FIELDS))
    aligned = all(g == s + 1 and e == t == (s if view == "claim" else g) for s, g, e, t in lengths)
    if view == "claim":
        aligned = aligned and all(values[0] == 0 for values in stats["greedy_logprobs"])
    if (cursor != manifest["expected_rows"] or not aligned or len(np.unique(labels)) != 2
            or (full_rows and (int(labels.sum()) != pin["is_hallu_1"]
                              or cohort != pin["cohort_sha256"]))):
        raise ValueError(f"Cache count, class or token alignment mismatch: {manifest_path}")
    return {**manifest, "full_dataset": full, "cohort_sha256": cohort}, ids, labels, stats


def evaluate_cache(manifest_path, output_dir):
    output_dir = Path(output_dir)
    metrics, score_files = [], {}
    cohort = None
    for view in ("full", "claim"):
        manifest, ids, labels, stats = read_cache(manifest_path, view)
        model, dataset = manifest["model_key"], manifest["dataset_key"]
        current = manifest["cohort_sha256"]
        if cohort is not None and current != cohort:
            raise ValueError("Full and claim views do not share IDs and labels")
        cohort = current
        arrays = {"row_ids": ids, "labels": labels}
        for name, estimator in load_methods({"supervised": False, "meta": manifest.get("meta", {})}):
            scores = np.asarray(estimator(stats))
            if scores.shape != labels.shape or not np.isfinite(scores).all():
                raise ValueError(f"Invalid scores: {model}/{dataset}/{view}/{name}")
            fixed = float(roc_auc_score(labels, scores))
            metrics.append({"model": model, "dataset": dataset, "view": view, "method": name,
                            "rows": len(ids), "positive": int(labels.sum()),
                            "full_dataset": manifest["full_dataset"], "auc_fixed": fixed,
                            "auc_folded_compatibility": max(fixed, 1 - fixed), "status": "passed"})
            arrays[name] = scores
        score_path = output_dir / "pairs" / f"{model}__{dataset}" / f"scores_{view}.npz"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(score_path, **arrays)
        score_files[view] = {"path": str(score_path.relative_to(output_dir)), "sha256": sha256(score_path)}
        del stats, arrays
    result = {"model": model, "dataset": dataset, "cohort_sha256": cohort, "metrics": metrics,
              "manifest_sha256": sha256(manifest_path), "score_files": score_files}
    write_json(output_dir / "pairs" / f"{model}__{dataset}" / "evaluation.json", result)
    return result


def evaluate(cache_dir, output_dir, allow_partial=False):
    output_dir = Path(output_dir)
    write_report([], output_dir)
    (output_dir / "evaluation.json").unlink(missing_ok=True)
    manifests = [path for path in Path(cache_dir).glob("*/*/manifest.json")
                 if read_json(path).get("status") == "complete"]
    manifests.sort(key=lambda p: (list(MODELS).index(read_json(p)["model_key"]),
                                 DATASET_ORDER.index(read_json(p)["dataset_key"])))
    pairs = [(read_json(p)["model_key"], read_json(p)["dataset_key"]) for p in manifests]
    if not pairs or len(set(pairs)) != len(pairs) or (not allow_partial and len(pairs) != 27):
        raise ValueError("Expected one complete cache per model/dataset; use --allow-partial for smoke tests")

    metrics, inputs, cohorts = [], [], {}
    for path, (model, dataset) in zip(manifests, pairs):
        # A fresh process releases large token arrays before the next dataset.
        subprocess.run([sys.executable, "-m", "experiments.feature_transfer.evaluate",
                        "--manifest", str(path.resolve()), "--output-dir", str(output_dir.resolve())], check=True)
        pair = read_json(output_dir / "pairs" / f"{model}__{dataset}" / "evaluation.json")
        if dataset in cohorts and cohorts[dataset] != pair["cohort_sha256"]:
            raise ValueError(f"Models do not share the same IDs and labels: {dataset}")
        cohorts[dataset] = pair["cohort_sha256"]
        metrics.extend(pair["metrics"])
        inputs.append({"manifest": str(path.resolve()), **{k: v for k, v in pair.items() if k != "metrics"}})
        print(f"Evaluated {model}/{dataset}", flush=True)
    complete = is_complete(metrics)
    macro = write_report(metrics, output_dir)
    result = {"status": "complete" if complete else "partial", "metric_rows": len(metrics),
              "macro9_rows": len(macro), "method_source_sha256": SOURCE_HASHES,
              "inputs": inputs, "estimator_call_scope": "one whole individual dataset"}
    write_json(output_dir / "evaluation.json", result)
    if not complete and not allow_partial:
        raise ValueError("Incomplete experiment; Average was not published")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen formulas on saved probability features")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache-dir", type=Path)
    source.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        evaluate_cache(args.manifest, args.output_dir)
    else:
        result = evaluate(args.cache_dir, args.output_dir, args.allow_partial)
        print(json.dumps({k: result[k] for k in ("status", "metric_rows", "macro9_rows")}))


if __name__ == "__main__":
    main()
