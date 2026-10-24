"""Validation-only permutation diagnostics for the already frozen fitted heads.

No refitting, new detector selection, or held-out quality datasets are involved.
Permutations break correlations; the result is a sensitivity diagnostic, not a
causal attribution or a feature-removal performance guarantee.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.supervised_baselines import (
    compact_features, FEATURE_NAMES, DOCUMENTED_FEATURE_INDICES)
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("out/campaign/fitted_feature_sensitivity.json"))
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    raw = torch.load("data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt",
                     mmap=True, weights_only=False, map_location="cpu")
    _, val = _prepare_train_val_indices(raw, 0, .2, 441, "stratified")
    pieces, labels = [], []
    selected = [raw["samples"][i] for i in val]
    for start in range(0, len(selected), 512):
        stats, y, _, _ = make_all_stats({"samples": selected[start:start+512], "meta": raw["meta"]}, input_regime="logits")
        pieces.append(compact_features(stats))
        labels.extend(y)
    features, labels = np.concatenate(pieces), np.asarray(labels, dtype=int)
    result = {"partition": "standard source-model validation", "rows": len(val),
              "fitted_models_changed": False, "held_out_datasets_read": False,
              "seed": 441, "permutation_repeats": args.repeats,
              "uncertainty": "Standard deviation over permutations, not a confidence interval",
              "limitation": "Marginal permutations break feature correlations and need not represent realizable inputs",
              "models": {}}
    groups = {"GROUP: length": ["length"],
              "GROUP: likelihoods": [x for x in FEATURE_NAMES if x.startswith("nll_")],
              "GROUP: full-vocabulary entropies": [x for x in FEATURE_NAMES if x.startswith("entropy_")],
              "GROUP: top-512 summaries": [x for x in FEATURE_NAMES if x.startswith("top")]}
    for path in [Path("out/campaign/supervised/models.joblib"), Path("out/campaign/supervised_documented/models.joblib")]:
        for name, model in joblib.load(path).items():
            documented = getattr(model, "mod1_feature_set", "full") == "documented"
            indices = DOCUMENTED_FEATURE_INDICES if documented else list(range(len(FEATURE_NAMES)))
            X, names = features[:, indices], [FEATURE_NAMES[i] for i in indices]
            decision = model.decision_function(X)
            probabilities = model.predict_proba(X)[:, 1]
            baseline = roc_auc_score(labels, decision)
            entries = {name: [i] for i, name in enumerate(names)}
            entries.update({key: [names.index(n) for n in members if n in names] for key, members in groups.items()})
            diagnostics = {}
            for key, columns in entries.items():
                if not columns:
                    continue
                rng = np.random.default_rng(441)
                decreases = []
                for _ in range(args.repeats):
                    permuted = X.copy()
                    permutation = rng.permutation(len(labels))
                    permuted[:, columns] = X[permutation][:, columns]
                    decreases.append(100*(baseline-roc_auc_score(labels, model.decision_function(permuted))))
                diagnostics[key] = {"mean_auc_decrease_pp": float(np.mean(decreases)),
                                    "permutation_sd_pp": float(np.std(decreases, ddof=1)),
                                    "features": [names[i] for i in columns]}
            result["models"][name] = {"validation_auc_percent": 100*baseline, "features": names,
                                      "validation_probability_auc_percent": 100*roc_auc_score(labels, probabilities),
                                      "validation_probabilities_at_zero_or_one": int(np.count_nonzero((probabilities==0)|(probabilities==1))),
                                      "scoring": "raw decision function to avoid saturated-probability rank ties",
                                      "sensitivity": diagnostics}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "models": list(result["models"])}))


if __name__ == "__main__":
    main()
