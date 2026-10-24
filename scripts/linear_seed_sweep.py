"""Validation-only replication of the restricted 60-candidate search control.

Additional seeds are supplementary and never enter primary winner selection.
Only the PopQA training feature file and existing training-control records are read.
"""
import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from search_linear_controls import SCALES, FEATURE_NAMES
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices
from packing.evaluate.uncertainty_quantification.supervised_baselines import compact_features


def search(X, y, train, val, mode, seed, candidates=60):
    rng = np.random.default_rng(seed)
    parent = np.zeros(len(FEATURE_NAMES))
    parent[1] = 1.
    best_train = roc_auc_score(y[train], (X @ parent)[train])
    rows = [{"candidate": -1, "weights": parent.tolist(), "train_auc": float(best_train),
             "validation_auc": float(roc_auc_score(y[val], (X @ parent)[val]))}]
    for index in range(candidates):
        if mode == "independent" or rng.random() < .25:
            weights = np.zeros(len(FEATURE_NAMES))
            chosen = rng.choice(len(weights), size=int(rng.integers(1, 4)), replace=False)
            weights[chosen] = rng.normal(size=len(chosen))
        else:
            weights = parent.copy()
            feature = int(rng.integers(len(weights)))
            weights[feature] += rng.normal(scale=.5)
            nonzero = np.flatnonzero(weights)
            if len(nonzero) > 3:
                weights[min(nonzero, key=lambda i: abs(weights[i]))] = 0
        scores = X @ weights
        train_auc = float(roc_auc_score(y[train], scores[train]))
        rows.append({"candidate": index, "weights": weights.tolist(), "train_auc": train_auc,
                     "validation_auc": float(roc_auc_score(y[val], scores[val]))})
        if train_auc > best_train:
            best_train, parent = train_auc, weights.copy()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=1049)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    started = time.monotonic()
    raw = torch.load("data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt",
                     mmap=True, weights_only=False, map_location="cpu")
    train, val = _prepare_train_val_indices(raw, 0, .2, 441)
    features, labels = [], []
    for start in range(0, len(raw["samples"]), 512):
        stats, y, _, _ = make_all_stats({"samples": raw["samples"][start:start+512], "meta": raw["meta"]}, input_regime="logits")
        features.append(compact_features(stats)); labels.extend(y)
    X, y = np.concatenate(features)/SCALES, np.asarray(labels)
    # Reproduce every coefficient and AUC in all four original controls before
    # using this supplementary implementation for new seeds.
    for mode in ["independent", "evolution"]:
        for seed in [1047, 1048]:
            original = json.loads((Path("out/logs/mod1") / f"uncertainty_quantification_linear_{mode}_{seed}" / "linear_candidates.json").read_text())
            if search(X, y, train, val, mode, seed) != original:
                raise ValueError("Seed sweep does not exactly reproduce the original control")
    output = Path("out/campaign/validation_linear_seed_sweep")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(args.first_seed, args.first_seed+args.seeds):
        for mode in ["independent", "evolution"]:
            rows = search(X, y, train, val, mode, seed)
            winner = max(rows, key=lambda r: r["validation_auc"])
            results.append({"mode": mode, "seed": seed, "selected_validation_auc_percent": 100*winner["validation_auc"],
                            "candidate_records": rows})
        if (seed-args.first_seed+1) % 10 == 0:
            print(json.dumps({"completed_seed_pairs": seed-args.first_seed+1}), flush=True)
    means = {}
    arrays = {}
    for mode in ["independent", "evolution"]:
        values = np.array([r["selected_validation_auc_percent"] for r in results if r["mode"] == mode])
        arrays[mode] = values
        means[mode] = {"n_seeds": len(values), "mean_percent": float(values.mean()),
                       "sample_sd_pp": float(values.std(ddof=1)), "min_max_percent": [float(values.min()), float(values.max())]}
    rng = np.random.default_rng(65041)
    differences = [rng.choice(arrays["evolution"], len(arrays["evolution"]), replace=True).mean()-
                   rng.choice(arrays["independent"], len(arrays["independent"]), replace=True).mean() for _ in range(5000)]
    summary = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "runs": len(results), "candidates_per_run": 60,
               "new_seeds": [args.first_seed, args.first_seed+args.seeds-1], "original_four_controls_reproduced_exactly": True,
               "feature_pool": FEATURE_NAMES, "feature_scales": SCALES.tolist(), "outer_split_seed": 441,
               "mode_summary": means, "evolution_minus_independent_mean_pp": means["evolution"]["mean_percent"]-means["independent"]["mean_percent"],
               "independent_seed_bootstrap_ci95_pp": np.percentile(differences,[2.5,97.5]).tolist(),
               "elapsed_seconds_including_feature_extraction": time.monotonic()-started,
               "held_out_datasets_read": False, "primary_winners_changed": False,
               "scope": "Validation-only seed variability of restricted fixed-feature search; no inference about LLM search or held-out gains"}
    (output / "candidates.json").write_text(json.dumps(results, indent=2)+"\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
