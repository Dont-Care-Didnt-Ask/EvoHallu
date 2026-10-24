"""Fit PopQA-training-only logistic and QDA baselines; five settings each."""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices
from packing.evaluate.uncertainty_quantification.supervised_baselines import compact_features, FEATURE_NAMES, DOCUMENTED_FEATURE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt")
    ap.add_argument("--output", type=Path, default=Path("out/campaign/supervised"))
    ap.add_argument("--split-strategy", choices=["stratified", "claim_group"], default="stratified")
    ap.add_argument("--feature-set", choices=["full", "documented"], default="full")
    ap.add_argument("--trim-header-suffix", type=int, default=0)
    ap.add_argument("--model-name-prefix", default="")
    args = ap.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = torch.load(args.raw, mmap=True, weights_only=False, map_location="cpu")
    if args.trim_header_suffix:
        from evaluate_cached_claims import trim_header_suffix
        raw["samples"] = trim_header_suffix(raw["samples"], args.trim_header_suffix)
    train, val = _prepare_train_val_indices(raw, 0, .2, 441, args.split_strategy)
    pieces, labels = [], []
    for start in range(0, len(raw["samples"]), 512):
        stats, y, _, _ = make_all_stats({"samples": raw["samples"][start:start+512], "meta": raw["meta"]}, input_regime="logits")
        pieces.append(compact_features(stats, args.feature_set))
        labels.extend(y)
    X, y = np.concatenate(pieces), np.asarray(labels, dtype=int)
    groups = np.array([" ".join(s["claim"].casefold().split()) for s in raw["samples"]])
    if args.split_strategy == "claim_group":
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=441).split(X[train], y[train], groups[train]))
    else:
        inner = list(StratifiedKFold(3, shuffle=True, random_state=441).split(X[train], y[train]))
    settings = {"logistic": [(.01, LogisticRegression(C=.01, max_iter=2000)),
                             (.1, LogisticRegression(C=.1, max_iter=2000)),
                             (1., LogisticRegression(C=1., max_iter=2000)),
                             (10., LogisticRegression(C=10., max_iter=2000)),
                             (100., LogisticRegression(C=100., max_iter=2000))],
                "qda": [(r, QuadraticDiscriminantAnalysis(reg_param=r)) for r in [.01, .1, .3, .6, .9]]}
    models, results = {}, {}
    for name, candidates in settings.items():
        records = []
        for parameter, estimator in candidates:
            model = make_pipeline(StandardScaler(), estimator)
            cv = cross_val_score(model, X[train], y[train], scoring="roc_auc", cv=inner, error_score="raise")
            records.append({"parameter": parameter, "inner_auc": float(cv.mean()), "model": model})
        best = max(records, key=lambda row: row["inner_auc"])
        model = best["model"].fit(X[train], y[train])
        model.mod1_feature_set = args.feature_set
        model.mod1_header_suffix_trim = args.trim_header_suffix
        suffix = "_documented" if args.feature_set == "documented" else ""
        models[args.model_name_prefix + name + suffix + "_" + args.split_strategy] = model
        def predict(X):
            return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:,1]
        results[name] = {"chosen_parameter": best["parameter"], "inner_auc": best["inner_auc"],
                         "train_auc": float(roc_auc_score(y[train], predict(X[train]))),
                         "validation_auc": float(roc_auc_score(y[val], predict(X[val]))),
                         "tuning": [{k: v for k, v in row.items() if k != "model"} for row in records]}
    joblib.dump(models, args.output / "models.joblib")
    report = {"features": DOCUMENTED_FEATURE_NAMES if args.feature_set == "documented" else FEATURE_NAMES,
              "feature_set": args.feature_set, "fit_samples": len(train), "validation_samples": len(val),
              "header_suffix_trim_tokens": args.trim_header_suffix, "model_name_prefix": args.model_name_prefix,
              "split_seed": 441, "split_strategy": args.split_strategy, "test_labels_used": False, "models": results}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
