"""Score the supplied formulas on training/validation only; freeze a shortlist."""
import argparse
import ast
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from packing.evaluate.uncertainty_quantification import found_formulas
from packing.evaluate.uncertainty_quantification.features import make_all_stats
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt")
    ap.add_argument("--output", type=Path, default=Path("out/campaign/original_validation"))
    args = ap.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = torch.load(args.raw, mmap=True, weights_only=False, map_location="cpu")
    classes = {n: c for n, c in inspect.getmembers(found_formulas, inspect.isclass) if c.__module__ == found_formulas.__name__}
    from packing.evaluate.uncertainty_quantification import alignment_baselines
    extras = {n: c for n, c in inspect.getmembers(alignment_baselines, inspect.isclass) if c.__module__ == alignment_baselines.__name__}
    report = {"test_data_used": False, "scoring_chunk_size": 512, "protocols": {}}
    for strategy in ["stratified", "claim_group"]:
        train, val = _prepare_train_val_indices(raw, 0, .2, 441, strategy)
        rows = {n: {"ast_nodes": sum(1 for _ in ast.walk(ast.parse(inspect.getsource(c))))} for n, c in {**classes, **extras}.items()}
        predictions = {}
        for split, indices in [("train", train), ("val", val)]:
            scores = {n: [] for n in rows}
            labels = []
            estimators = {n: c({"supervised": False}) for n, c in classes.items()}
            estimators.update({n: c() for n, c in extras.items()})
            errors = {}
            for start in range(0, len(indices), 512):
                samples = [raw["samples"][int(i)] for i in indices[start:start+512]]
                stats, y, _, _ = make_all_stats({"samples": samples, "meta": raw["meta"]}, input_regime="logits")
                stats["input_tokens"] = [s["input_ids"].tolist() for s in samples]
                labels.extend(y)
                for name, estimator in estimators.items():
                    try:
                        result = np.asarray(estimator(stats))
                        if not np.isfinite(result).all() or result.shape != (len(samples),):
                            raise ValueError("nonfinite scores or wrong output shape")
                        scores[name].extend(result)
                    except Exception as exc:
                        errors[name] = f"{type(exc).__name__}: {exc}"
            for name in rows:
                if name in errors:
                    rows[name][split + "_error"] = errors[name]
                else:
                    rows[name][split + "_auc_percent"] = 100 * float(roc_auc_score(labels, scores[name]))
                    if split == "val":
                        predictions[name] = np.asarray(scores[name])
            if split == "val":
                np.savez_compressed(args.output / f"{strategy}_val_scores.npz", labels=labels, **predictions)
        ranked = sorted([n for n in rows if "val_auc_percent" in rows[n]], key=lambda n: rows[n]["val_auc_percent"], reverse=True)
        report["protocols"][strategy] = {"train_rows": len(train), "val_rows": len(val), "methods": rows,
            "frozen_top_three_by_validation": ranked[:3],
            "complexity_validation_spearman": float(spearmanr([rows[n]["ast_nodes"] for n in ranked], [rows[n]["val_auc_percent"] for n in ranked]).statistic)}
        print(strategy, {n: rows[n]["val_auc_percent"] for n in ranked[:5]}, flush=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
