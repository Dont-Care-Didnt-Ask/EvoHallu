"""Cheap, label-guided search of fixed formulas, without an LLM or fitted model.

Uses a fixed human-designed feature pool. It is a restricted-search comparator,
not a claim that this pool and open-ended program generation are equivalent.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices
from packing.evaluate.uncertainty_quantification.supervised_baselines import compact_features, FEATURE_NAMES

SCALES = np.array([30., 100., 5., 3., 15., 10., 2., 5., 5., 5., 3., 3., 1., 5., 6., 5., 5.])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt")
    ap.add_argument("--candidates", type=int, default=60)
    args = ap.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    raw = torch.load(args.raw, mmap=True, weights_only=False, map_location="cpu")
    train, val = _prepare_train_val_indices(raw, 0, .2, 441)
    features, labels = [], []
    for start in range(0, len(raw["samples"]), 512):
        stats, y, _, _ = make_all_stats({"samples": raw["samples"][start:start+512], "meta": raw["meta"]}, input_regime="logits")
        features.append(compact_features(stats))
        labels.extend(y)
    X, y = np.concatenate(features) / SCALES, np.asarray(labels)
    for mode in ["independent", "evolution"]:
        for seed in [1048, 1047]:
            started = time.monotonic()
            rng = np.random.default_rng(seed)
            parent = np.zeros(len(FEATURE_NAMES))
            parent[1] = 1.
            best_train = roc_auc_score(y[train], (X @ parent)[train])
            rows = [{"candidate": -1, "weights": parent.tolist(), "train_auc": float(best_train),
                     "validation_auc": float(roc_auc_score(y[val], (X @ parent)[val]))}]
            for index in range(args.candidates):
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
                        drop = min(nonzero, key=lambda i: abs(weights[i]))
                        weights[drop] = 0
                scores = X @ weights
                train_auc = float(roc_auc_score(y[train], scores[train]))
                val_auc = float(roc_auc_score(y[val], scores[val]))
                rows.append({"candidate": index, "weights": weights.tolist(),
                             "train_auc": train_auc, "validation_auc": val_auc})
                if train_auc > best_train:
                    best_train, parent = train_auc, weights.copy()
            winner = max(rows, key=lambda r: r["validation_auc"])
            selected = np.array(winner["weights"])
            run = Path("out/logs/mod1") / f"uncertainty_quantification_linear_{mode}_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            code = f'''from packing.evaluate.uncertainty_quantification.supervised_baselines import compact_features
class HallucinationDetector(Estimator):
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs", "token_entropies", "top_k_logits"], level="sequence")
    def __call__(self, stats):
        weights = np.array({selected.tolist()!r})
        scales = np.array({SCALES.tolist()!r})
        return (compact_features(stats) / scales) @ weights
'''
            (run / "frozen_estimator.py").write_text(code)
            (run / "linear_candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
            np.save(run / "selected_val_scores.npy", (X @ selected)[val])
            summary = {"status": "finished", "generator": "fixed_feature_linear_" + mode,
                       "seed": seed, "generated_candidates": args.candidates,
                       "selected_validation_auc_percent": 100 * winner["validation_auc"],
                       "selected_train_auc_percent": 100 * winner["train_auc"],
                       "selected_features": {FEATURE_NAMES[i]: float(selected[i]) for i in np.flatnonzero(selected)},
                       "elapsed_seconds": time.monotonic() - started, "total_tokens": 0,
                       "held_out_test_evaluated": False, "feature_pool": FEATURE_NAMES,
                       "feature_scales": SCALES.tolist()}
            (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"mode": mode, "seed": seed, "validation_auc_percent": summary["selected_validation_auc_percent"]}), flush=True)


if __name__ == "__main__":
    main()
