"""Add newly scored frozen methods while preserving the reference predictions.

The additional evaluation recomputes the common NLL reference on every dataset.
Labels, group IDs, masks, reference scores and paired bootstrap draws must match
exactly. Existing method results are reused without fitting or reselection.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


def read_arrays(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def merge_cohorts(original, additional, output):
    original, additional, output = map(Path, [original, additional, output])
    if output.resolve() in {original.resolve(), additional.resolve()}:
        raise ValueError("Write the merged cohort to its own directory")
    before = json.loads((original / "all_datasets.json").read_text())
    additions = json.loads((additional / "all_datasets.json").read_text())
    if len(before) != 9 or set(before) != set(additions):
        raise ValueError("Both evaluations must contain the same nine datasets")
    output.mkdir(parents=True, exist_ok=True)
    (output / "all_datasets.json").unlink(missing_ok=True)
    (output / "cohort_verified.json").unlink(missing_ok=True)
    aggregate = {}
    for dataset in before:
        old, extra = before[dataset], additions[dataset]
        for key in ["rows", "duplicate_disjoint_rows", "unique_normalized_claims", "bootstrap_unit", "feature_model_metadata"]:
            if old.get(key) != extra.get(key):
                raise ValueError(f"Dataset protocol differs on {dataset}: {key}")
        baseline = "original/SequenceLogProb"
        original_scores = read_arrays(original / (dataset+"_scores.npz"))
        extra_scores = read_arrays(additional / (dataset+"_scores.npz"))
        for key in ["labels", "duplicate_disjoint_mask", "claim_group_ids", baseline]:
            if key not in original_scores or key not in extra_scores or not np.array_equal(original_scores[key], extra_scores[key], equal_nan=True):
                raise ValueError(f"Recomputed reference differs on {dataset}: {key}")
        for suffix, first in [("_scores.npz", original_scores),
                              ("_bootstrap.npz", read_arrays(original / (dataset+"_bootstrap.npz")))]:
            second = extra_scores if suffix == "_scores.npz" else read_arrays(additional / (dataset+suffix))
            if baseline not in second or not np.array_equal(first[baseline], second[baseline], equal_nan=True):
                raise ValueError(f"Reference predictions or bootstrap draws differ on {dataset}")
            for key in set(first) & set(second):
                if not np.array_equal(first[key], second[key], equal_nan=True):
                    raise ValueError(f"Cannot replace an existing prediction: {dataset} {key}")
            merged_arrays = {**second, **first}
            np.savez_compressed(output / (dataset+suffix), **merged_arrays)
        result = copy.deepcopy(old)
        attempted = set(extra["methods"]) | set(extra.get("failed", {}))
        existing = set(old["methods"]) | set(old.get("failed", {}))
        if not attempted-existing:
            raise ValueError("No new method was attempted: " + dataset)
        for key in attempted-existing:
            if key in extra["methods"]:
                result["methods"][key] = extra["methods"][key]
            else:
                result.setdefault("failed", {})[key] = extra["failed"][key]
        aggregate[dataset] = result
        (output / (dataset+".json")).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    manifests = {}
    for key, directory in [("original", original), ("additional", additional)]:
        path = directory / "manifest.json"
        manifests[key] = json.loads(path.read_text()) if path.exists() else None
    manifest = {"reused_frozen_predictions_from": str(original), "new_methods_from": str(additional),
                "recomputed_reference_scores_and_bootstrap_identical": True,
                "protocol_manifests": manifests, "datasets": 9}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    temporary = output / "all_datasets.tmp"
    temporary.write_text(json.dumps(aggregate, indent=2, allow_nan=False)+"\n")
    temporary.replace(output / "all_datasets.json")
    return {"datasets": 9, "output": str(output), "reference_recomputed_and_identical": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_cohorts(args.original, args.additional, args.output)))


if __name__ == "__main__":
    main()
