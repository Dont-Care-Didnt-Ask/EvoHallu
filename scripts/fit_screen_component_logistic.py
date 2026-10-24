"""Prepare or fit one CPU-only, source-training screen-component logistic head.

Default only freezes the source pool and writes commands. --execute explicitly
fits PopQA training features; held-out feature caches are never opened here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
import shlex

import numpy as np

from scripts.overnight_component_model import ComponentPool, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/overnight20261021/screen_component_refit")
ABLATION_MANIFEST = Path("results/overnight20261021/screen_component_ablations/manifest.json")
TRAINING_RAW_SHA256 = "8d34da6f189e8bfe67c0bca3e1101607f5b0887d5f1de41b4f373ef08f2f5ced"
CS = [.01, .1, 1., 10., 100.]


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode() if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("Refusing to overwrite a different prepared artifact: " + str(path))
    else:
        path.write_bytes(data)


def json_once(path, payload):
    write_once(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def relative(path, root=ROOT):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def prepare_pool(manifest_path, output, root=ROOT):
    root, output, manifest_path = Path(root), Path(output), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["methods"]
    full = {row["policy"]: row for row in rows if row["role"] == "full"}
    drops = [row for row in rows if row["dropped_component_index"] is not None]
    if set(full) != {"original", "neutral", "diverse", "compositional"} or len(drops) != 11 or len(rows) != 15:
        raise ValueError("Ablation manifest must contain all four full programs and eleven drops")
    programs = []
    for row in rows:
        source = root / row["file"]
        if sha(source) != row["sha256"]:
            raise ValueError("Ablation source hash mismatch: " + row["file"])
        dest = output / "components" / row["method"].removeprefix("new/") / "frozen_estimator.py"
        write_once(dest, source.read_bytes())
        programs.append(dict(method=row["method"], file=relative(dest, root), sha256=row["sha256"],
                             policy=row["policy"], role=row["role"]))
    components = [dict(name=row["policy"]+"/"+row["dropped_component"],
                       full=full[row["policy"]]["method"], drop=row["method"],
                       expression=row["components"][row["dropped_component_index"]]["expression"])
                  for row in drops]
    pool = dict(study="fixed_screen_component_logistic_refit", source_ablation_manifest=relative(manifest_path, root),
                source_ablation_manifest_sha256=sha(manifest_path), selection="All four source-validation screen winners; all eleven drops retained",
                components=components, programs=programs, extra_feature="float64_total_nll",
                source_pool_validation_informed=True, outer_validation_is_independent=False,
                held_out_scores_used_to_choose_features=False)
    path = output / "component_pool.json"
    json_once(path, pool)
    return path, pool


def fit_head(X, y, groups, train, val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if set(groups[train]) & set(groups[val]):
        raise ValueError("Outer fit/validation normalized claims overlap")
    inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=441).split(X[train], y[train], groups[train]))
    folds = []
    for fit, test in inner:
        overlap = len(set(groups[train][fit]) & set(groups[train][test]))
        if overlap:
            raise ValueError("Inner normalized claims overlap")
        folds.append(dict(fit_samples=len(fit), validation_samples=len(test), shared_normalized_claims=overlap))
    records = []
    for parameter in CS:
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=parameter, max_iter=2000))
        auc = cross_val_score(pipeline, X[train], y[train], scoring="roc_auc", cv=inner, error_score="raise")
        records.append(dict(C=parameter, mean_auc=float(auc.mean()), fold_auc=auc.tolist()))
    best = max(records, key=lambda row: row["mean_auc"])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=2000)).fit(X[train], y[train])
    predictions = model.decision_function(X)
    report = dict(chosen_C=best["C"], C_grid=CS, tuning=records, inner_folds=folds,
                  fit_samples=len(train), validation_samples=len(val), outer_shared_normalized_claims=0,
                  scaler_fit_samples=int(model[0].n_samples_seen_),
                  train_auc=float(roc_auc_score(y[train], predictions[train])),
                  validation_auc=float(roc_auc_score(y[val], predictions[val])),
                  outer_validation_is_independent=False, validation_informed_feature_pool=True,
                  standardized_coefficients=model[-1].coef_[0].tolist())
    return model, predictions, report


def wrapper_source(artifact_relative, artifact_sha256, module_sha256):
    return f'''class HallucinationDetector(Estimator):
    def __init__(self, context):
        from scripts.overnight_component_model import FrozenComponentLogistic
        self.frozen_head = FrozenComponentLogistic(context, {artifact_relative!r}, {artifact_sha256!r}, {module_sha256!r})
        super().__init__(context, stats_dependencies=self.frozen_head.stats_dependencies, level="sequence")

    def __call__(self, stats):
        return self.frozen_head(stats)
'''


def prepare_commands(output):
    env = ["env", "PYTHONPATH=src:.", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"]
    prefix = env + [".venv/bin/python"]
    glob = str(output / "frozen/*/frozen_estimator.py")
    common = ["--data-root", "data/claims", "--frozen-glob", glob, "--method-prefix", "new/screen_component_logistic",
              "--bootstrap", "500", "--bootstrap-unit", "claim"]
    source = [prefix + ["scripts/evaluate_parallel.py", "--workers", "2", "--cache-dir", "data/restored/mod1-eval-cache",
                       "--output", str(output / "evaluations/source_nine"), "--model-id", "/workspace/mod1-models/llama31"] + common]
    target = [prefix + ["scripts/evaluate_cached_claims.py", "--mode", "evaluate", "--dataset-index", str(index),
                       "--cpu-threads", "1", "--cache-dir", "data/restored/mod1-transfer-cache", "--output", str(output / "evaluations/target_six"),
                       "--model-id", "unsloth/Llama-3.2-1B-Instruct", "--tokenizer-id", "/workspace/mod1-models/llama31-tokenizer"] + common for index in range(6)]
    result = dict(executed=False, fit=prefix + ["scripts/fit_screen_component_logistic.py", "--output", str(output), "--execute"],
                  source_nine=source, target_six=target)
    json_once(output / "commands.json", result)
    for key in ["source_nine", "target_six"]:
        write_once(output / ("run_"+key+".sh"), "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(shlex.join(command) for command in result[key]) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ABLATION_MANIFEST)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--raw", type=Path, default=Path("data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt"))
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    args.output.resolve().relative_to(ROOT)
    pool_path, pool_manifest = prepare_pool(args.manifest, args.output)
    commands = prepare_commands(args.output)
    if not args.execute:
        print(json.dumps(dict(status="prepared_only_no_training_or_test_scoring", features=12, components=11,
                              pool_sha256=sha(pool_path), commands=commands), indent=2))
        return
    weights_path = args.output / "weights.json"
    if weights_path.exists():
        raise FileExistsError("Refusing to refit an already frozen component head")
    if sha(args.raw) != TRAINING_RAW_SHA256:
        raise ValueError("Training cache differs from the pinned audited PopQA cache")
    import torch
    from packing.evaluate.uncertainty_quantification.features import make_all_stats
    from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    raw = torch.load(args.raw, mmap=True, weights_only=False, map_location="cpu")
    train, val = _prepare_train_val_indices(raw, 0, .2, 441, "claim_group")
    if (len(raw["samples"]), len(train), len(val)) != (7007, 5606, 1401):
        raise ValueError("Unexpected historical grouped partition sizes")
    pool = ComponentPool(ROOT, relative(pool_path), sha(pool_path))
    pieces, labels = [], []
    for start in range(0, len(raw["samples"]), 512):
        stats, y, _, _ = make_all_stats(dict(samples=raw["samples"][start:start+512], meta=raw["meta"]), input_regime="logits")
        pieces.append(pool.transform(stats))
        labels.extend(y)
        print("Extracted training components:", min(start+512,len(raw["samples"])), flush=True)
    X, y = np.concatenate(pieces), np.asarray(labels, dtype=int)
    groups = np.array([" ".join(sample["claim"].casefold().split()) for sample in raw["samples"]])
    model, predictions, report = fit_head(X, y, groups, train, val)
    module_path = ROOT / "scripts/overnight_component_model.py"
    weights = dict(format="screen_component_logistic_v1", feature_names=pool.feature_names,
                   pool_file=relative(pool_path), pool_sha256=sha(pool_path), inference_module_sha256=sha(module_path),
                   scaler_mean=model[0].mean_.tolist(), scaler_scale=model[0].scale_.tolist(),
                   coefficients_standardized=model[-1].coef_[0].tolist(), intercept=float(model[-1].intercept_[0]),
                   classes=model[-1].classes_.tolist(), training_raw_sha256=sha(args.raw), source_fit_only=True)
    if weights["classes"] != [0,1]:
        raise ValueError("Unexpected binary label ordering")
    json_once(weights_path, weights)
    wrapper = args.output / "frozen/screen_component_logistic/frozen_estimator.py"
    write_once(wrapper, wrapper_source(relative(weights_path), sha(weights_path), sha(module_path)))
    # Frozen numerical serialization must reproduce the fitted sklearn pipeline.
    serialized = ((X - np.asarray(weights["scaler_mean"])) / np.asarray(weights["scaler_scale"])) @ np.asarray(weights["coefficients_standardized"]) + weights["intercept"]
    np.testing.assert_allclose(serialized, predictions, atol=1e-10, rtol=1e-10)
    np.savez_compressed(args.output / "training_features_and_predictions.npz", features=X, labels=y, train_indices=train,
                        validation_indices=val, frozen_predictions=serialized)
    report.update(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status="frozen_source_fit_only",
                  feature_names=pool.feature_names, pool_sha256=sha(pool_path), weights_sha256=sha(weights_path),
                  wrapper_sha256=sha(wrapper), inference_module_sha256=sha(module_path),
                  fitter_sha256=sha(__file__), training_raw_sha256=weights["training_raw_sha256"],
                  split_strategy="claim_group", split_seed=441, test_data_read=False, target_refitting=False,
                  joint_supervised_evolution=False, serialized_prediction_parity=True,
                  previous_held_out_results_already_viewed=True, interpretation="Exploratory; source-validation-informed feature pool; outer validation is diagnostic")
    json_once(args.output / "fit_report.json", report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
