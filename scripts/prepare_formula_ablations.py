"""Freeze interpretable, untuned formula ablations before held-out scoring."""
import ast
import copy
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from evaluate_cached_claims import load_estimators
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices


def main():
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    root = Path("out/logs/mod1")
    laguna = root / "uncertainty_quantification_laguna_evo_1048"
    qwen = root / "uncertainty_quantification_qwen38_evo_1047"
    variants = {}
    for name, expression in {
        "laguna_slope_only": "baseline",
        "laguna_slope_range": "baseline + 0.25 * normalized_entropy_range",
        "laguna_slope_variance": "baseline + 0.15 * top1_variance",
    }.items():
        tree = ast.parse((laguna / "frozen_estimator.py").read_text())
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == "final_score" for t in n.targets)]
        assert len(assignments) == 1
        assignments[0].value = ast.parse(expression, mode="eval").body
        variants[name] = laguna, ast.unparse(ast.fix_missing_locations(tree)) + "\n", expression
    variants["qwen_linear_position_nll"] = qwen, '''import numpy as np
class HallucinationDetector(Estimator):
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")
    def __call__(self, stats):
        result = []
        for lp in stats["sequence_logprobs"]:
            nll = -np.asarray(lp, dtype=np.float64)
            weights = np.arange(1, len(nll)+1, dtype=np.float64)
            result.append(len(nll)*np.average(nll, weights=weights) if len(nll) else 0.)
        return np.asarray(result)
''', "n_tokens * position_weighted_mean_nll; float64 removes half-precision cancellation residuals"
    paths = []
    parents = {}
    for name, (parent, code, expression) in variants.items():
        destination = root / ("ablation_" + name)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "frozen_estimator.py"
        if path.exists() and path.read_text() != code:
            raise ValueError("Existing ablation differs: " + str(path))
        path.write_text(code)
        config = json.loads((parent / "trial_config.json").read_text())
        config.update(search_mode="frozen_ablation", num_rounds=0)
        config["trial"]["serving_profile"] = "none"
        (destination / "trial_config.json").write_text(json.dumps(config, indent=2)+"\n")
        metadata = {"parent": str(parent / "frozen_estimator.py"),
                    "parent_sha256": hashlib.sha256((parent / "frozen_estimator.py").read_bytes()).hexdigest(),
                    "expression": expression, "coefficients_retuned": False,
                    "held_out_data_read": False, "preparation_partition": "source-model validation"}
        (destination / "ablation_protocol.json").write_text(json.dumps(metadata, indent=2)+"\n")
        paths.append(path)
        parents[name] = metadata
    raw = torch.load("data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt",
                     mmap=True, weights_only=False, map_location="cpu")
    _, val = _prepare_train_val_indices(raw, 0, .2, 441, "stratified")
    selected = [raw["samples"][i] for i in val]
    estimators, _ = load_estimators(paths + [laguna / "frozen_estimator.py", qwen / "frozen_estimator.py"], raw["meta"])
    estimators = {k:v for k,v in estimators.items() if k.startswith("new/")}
    scores = {k:[] for k in estimators}
    labels = []
    for start in range(0, len(selected), 512):
        stats, y, _, _ = make_all_stats({"samples": selected[start:start+512], "meta": raw["meta"]}, input_regime="logits")
        labels.extend(y)
        for key, estimator in estimators.items():
            prediction = np.asarray(estimator(copy.deepcopy(stats)))
            if prediction.shape != (len(y),) or not np.isfinite(prediction).all():
                raise ValueError("Invalid ablation: " + key)
            scores[key].extend(prediction)
    aucs = {key:100*roc_auc_score(labels, values) for key, values in scores.items()}
    for path in paths:
        key = "new/"+path.parent.name
        summary = {"status": "finished", "generator": "frozen_formula_ablation", "requests": 0,
                   "valid_candidates": 0, "total_tokens": 0, "elapsed_seconds": 0.,
                   "selected_validation_auc_percent": aucs[key], "held_out_test_evaluated": False}
        (path.parent / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    qwen_original = scores["new/"+qwen.name]
    qwen_simple = scores["new/ablation_qwen_linear_position_nll"]
    report = {"untuned_ablations": parents, "validation_auc_percent": aucs,
              "qwen_simplification_prediction_spearman": float(spearmanr(qwen_original, qwen_simple).statistic),
              "qwen_simplification_max_absolute_score_difference": float(np.max(np.abs(np.asarray(qwen_original)-qwen_simple))),
              "all_training_rows_skip_qwen1048_margin_branch": all(
                  len(s["top_k_logits"]) == len(s["sequence_logprobs"])+1 for s in raw["samples"]),
              "training_rows_audited": len(raw["samples"]), "held_out_data_read": False}
    Path("out/campaign/formula_ablation_preparation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
