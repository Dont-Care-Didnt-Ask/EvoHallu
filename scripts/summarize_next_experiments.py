"""Report prepared/finished studies using training-probe diversity diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]


def normalized(values):
    ranks = rankdata(values).astype(float)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, default=ROOT / "results/next_experiments")
    args = ap.parse_args()
    manifest = json.loads((args.directory / "run_manifest.json").read_text())
    probe = json.loads((args.directory / "training_probe.json").read_text())
    positions = np.asarray(probe["ordered_training_prediction_positions"])
    rows = []
    for job in manifest["jobs"]:
        folder = ROOT / job["expected_logs"]
        row = dict(id=job["id"], phase=job["phase"], policy=job["prompt_policy"], seed=job["seed"],
                   planned_requests=job["requests"], status="not_run")
        if (folder / "summary.json").exists():
            summary = json.loads((folder / "summary.json").read_text())
            cfg = json.loads((folder / "trial_config.json").read_text())
            if cfg["task"].get("split_strategy", "stratified") != probe["split_strategy"] or cfg["task"]["split_seed"] != probe["split_seed"]:
                raise ValueError("Training-probe positions do not match the study split")
            if cfg["task"].get("sample_size", 0) != 0 or cfg["task"]["val_size"] != .2:
                raise ValueError("Training-probe positions require the full historical 80/20 split")
            records = [json.loads(s) for s in (folder / "candidates.jsonl").read_text().splitlines()]
            valid = [r for r in records if "score" in r.get("val", {})]
            vectors, unique, near, constant = defaultdict(list), set(), 0, 0
            previous = []
            baseline = folder / "baseline_train_scores.npy"
            if not baseline.exists():
                raise ValueError("Prepared study must save training predictions")
            previous.append(normalized(np.load(baseline)[positions]))
            for r in valid:
                # Resolve through the current run directory, allowing archived runs to move.
                values = np.load(folder / f"train_scores_{r['code_hash']}.npy")
                if values.shape != (probe["train_rows"],):
                    raise ValueError("Training score count does not match the fixed probe manifest")
                values = values[positions]
                vector = normalized(values)
                if vector is None:
                    constant += 1
                    continue
                fingerprint = hashlib.sha256(np.unique(values, return_inverse=True)[1].astype("<i4").tobytes()).hexdigest()
                unique.add(fingerprint)
                near += any(v is not None and float(vector @ v) >= .995 for v in previous)
                previous.append(vector)
                vectors[r["round"]].append(vector)
            sibling_rhos = [float(vs[i] @ vs[j]) for vs in vectors.values() for i in range(len(vs)) for j in range(i+1,len(vs))]
            row.update(status=summary["status"], requests=summary["requests"],
                full_planned_budget=summary["requests"]==job["requests"], valid=len(valid),
                valid_fraction=len(valid)/len(records) if records else None,
                selected_validation_auc_percent=summary.get("selected_validation_auc_percent"),
                distinct_nonconstant_training_probe_rankings=len(unique), constant_probe_candidates=constant,
                near_previous_or_baseline_probe_candidates=int(near), comparable_sibling_pairs=len(sibling_rhos),
                near_identical_sibling_pairs=sum(r>=.995 for r in sibling_rhos),
                total_tokens=summary["total_tokens"], elapsed_seconds=summary["elapsed_seconds"],
                held_out_test_evaluated=summary["held_out_test_evaluated"])
        rows.append(row)
    report = dict(status="prepared_only" if all(r["status"]=="not_run" for r in rows) else "partial_or_finished",
        diversity_partition="512 fixed training-probe rows; not source validation",
        confirmation_status="No independent held-out confirmation established", runs=rows)
    (args.directory / "study_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines = ["# Prepared study status", "", "Diversity uses the fixed training probe. Pending jobs have no result.", "",
             "| Study | Status | Requests | Valid | Distinct probe rankings | Selected validation AUROC |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append("| "+" | ".join(str(v) for v in [r["id"],r["status"],r.get("requests","—"),
            r.get("valid","—"),r.get("distinct_nonconstant_training_probe_rankings","—"),
            r.get("selected_validation_auc_percent","—")])+" |")
    (args.directory / "STUDY_STATUS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status=report["status"], jobs=len(rows), finished=sum(r["status"]=="finished" for r in rows))))


if __name__ == "__main__":
    main()
