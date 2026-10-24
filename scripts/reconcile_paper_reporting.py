"""Reproduce the author's clarified per-dataset direction-invariant reporting.

Labels remain is_hallu. Only the reporting transform changes; no model fitting,
candidate/seed reselection, or changes to historical prediction artifacts.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results/next_experiments"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "out/campaign/evaluation_with_4b").glob("*_scores.npz"))
    assert len(paths) == 9
    arrays = [(p, dict(np.load(p))) for p in paths]
    names = sorted(set.intersection(*(set(z) for _,z in arrays)) - {"labels", "claim_group_ids", "duplicate_disjoint_mask"})
    methods = {}
    for name in names:
        per_dataset = []
        for path, z in arrays:
            raw = 100*float(roc_auc_score(z["labels"],z[name]))
            per_dataset.append(dict(dataset=path.name.removesuffix("_scores.npz"), raw_auc_percent=raw,
                                    direction_invariant_auc_percent=max(raw,100-raw), direction_flipped=raw<50))
        methods[name] = dict(raw_macro_auc_percent=float(np.mean([d["raw_auc_percent"] for d in per_dataset])),
            direction_invariant_macro_auc_percent=float(np.mean([d["direction_invariant_auc_percent"] for d in per_dataset])),
            flipped_datasets=[d["dataset"] for d in per_dataset if d["direction_flipped"]], per_dataset=per_dataset)
    published = {"original/SequenceLogProb":68.7, "original/Perplexity":64.2, "original/MeanTokenEntropy":59.9}
    baseline_checks = {name:dict(paper_auc_percent=paper, reproduced_auc_percent=methods[name]["direction_invariant_macro_auc_percent"],
                                matches_paper_rounding=round(methods[name]["direction_invariant_macro_auc_percent"],1)==paper)
                       for name,paper in published.items()}
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), labels="is_hallu",
        source_of_clarification="Paper author's explanation dated 16 October 2026, supplied directly by the user on 21 October 2026",
        reporting_rule="For each method and each dataset: max(AUC_percent, 100-AUC_percent); then unweighted mean over nine datasets",
        baseline_reproduction_resolved=all(r["matches_paper_rounding"] for r in baseline_checks.values()),
        interpretation="Reporting-direction reconciliation; does not imply any UHEAD CSV or is_hallu labeling error",
        selection_rule_unchanged="Training fitness and final program/seed selection retain directed PopQA AUROC",
        no_gpu_or_model_calls=True, historical_predictions_modified=False,
        uncertainty_note="Point estimates only; prior directed-AUROC confidence intervals are not transferred to this transform",
        source_predictions_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        baseline_checks=baseline_checks, methods=methods)
    (output / "paper_reporting.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (output / "paper_reporting.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=["method","raw_macro_auc_percent","direction_invariant_macro_auc_percent","flipped_datasets"])
        writer.writeheader()
        for name,row in methods.items():
            writer.writerow(dict(method=name, **{k:row[k] for k in ["raw_macro_auc_percent","direction_invariant_macro_auc_percent"]},
                                 flipped_datasets=";".join(row["flipped_datasets"])))
    print(json.dumps(dict(baseline_reproduction_resolved=report["baseline_reproduction_resolved"], baseline_checks=baseline_checks),indent=2))


if __name__ == "__main__":
    main()
