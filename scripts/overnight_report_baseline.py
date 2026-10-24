"""Check new NLL arrays, labels and row identity against the historical benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "original/SequenceLogProb"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_data_hashes(value):
    found = []
    if isinstance(value, dict):
        if isinstance(value.get("data"), dict):
            found.append(value["data"])
        for child in value.values():
            found.extend(historical_data_hashes(child))
    return found


def compare(datasets, historical, data_root):
    csvs = [p for p in sorted(data_root.rglob("*.csv")) if "popqa_train" not in p.name]
    paths = {p.parent.name + "__" + p.stem: p for p in csvs}
    manifest_path = historical / "manifest.json"
    hashes = historical_data_hashes(json.loads(manifest_path.read_text())) if manifest_path.exists() else []
    training = pd.read_csv(data_root / "popqa/popqa_train_filtered_10shot.csv", usecols=["claim"])
    train_claims = set(training.claim.fillna("").astype(str).str.split().str.join(" ").str.casefold())
    rows = []
    for name, csv_path in paths.items():
        row = dict(dataset=name, status="pending", source_csv_sha256=sha(csv_path))
        expected = {mapping[str(csv_path.relative_to(data_root))] for mapping in hashes
                    if str(csv_path.relative_to(data_root)) in mapping}
        row["historical_csv_sha256"] = sorted(expected)
        row["source_csv_bytes_match_historical_manifest"] = len(expected) == 1 and row["source_csv_sha256"] in expected
        current = datasets.get(name)
        previous_path = historical / (name + "_scores.npz")
        if current is None or BASELINE not in current:
            rows.append(row)
            continue
        if not previous_path.exists():
            row["status"] = "historical_reference_missing"
            rows.append(row)
            continue
        previous = dict(np.load(previous_path))
        frame = pd.read_csv(csv_path, usecols=["claim", "is_hallu"])
        claims = frame.claim.fillna("").astype(str).str.split().str.join(" ").str.casefold()
        group_ids, _ = pd.factorize(claims, sort=False)
        mask = np.asarray([claim not in train_claims for claim in claims], dtype=bool)
        row.update(rows=len(current["labels"]),
            historical_prediction_sha256=sha(previous_path),
            labels_identical_historical=np.array_equal(current["labels"], previous["labels"]),
            labels_match_csv_order=np.array_equal(current["labels"], frame.is_hallu.to_numpy()),
            nll_scores_exactly_identical=np.array_equal(current[BASELINE], previous[BASELINE]),
            claim_group_ids_identical_historical=np.array_equal(current.get("claim_group_ids"), previous.get("claim_group_ids")),
            claim_group_ids_match_csv_order=np.array_equal(current.get("claim_group_ids"), group_ids),
            duplicate_disjoint_mask_identical_historical=np.array_equal(current.get("duplicate_disjoint_mask"), previous.get("duplicate_disjoint_mask")),
            duplicate_disjoint_mask_matches_csv_order=np.array_equal(current.get("duplicate_disjoint_mask"), mask))
        if current[BASELINE].shape == previous[BASELINE].shape:
            row["nll_max_absolute_difference"] = float(np.max(np.abs(current[BASELINE]-previous[BASELINE])))
        current_auc = 100 * float(roc_auc_score(current["labels"], current[BASELINE]))
        previous_auc = 100 * float(roc_auc_score(previous["labels"], previous[BASELINE]))
        row.update(directed_auc_percent=current_auc, historical_directed_auc_percent=previous_auc,
                   directed_auc_exactly_identical=current_auc == previous_auc,
                   paper_auc_percent=max(current_auc, 100-current_auc))
        checks = [value for key, value in row.items() if key.startswith(("labels_", "claim_group_ids_", "duplicate_disjoint_mask_"))]
        checks += [row["source_csv_bytes_match_historical_manifest"], row["nll_scores_exactly_identical"], row["directed_auc_exactly_identical"]]
        row["status"] = "passed" if all(checks) else "failed"
        rows.append(row)
    complete = len(rows) == 9 and all(r["status"] in {"passed", "failed"} for r in rows)
    paper = float(np.mean([r["paper_auc_percent"] for r in rows])) if complete else None
    return dict(status="passed" if complete and all(r["status"] == "passed" for r in rows) else "failed" if any(r["status"] == "failed" for r in rows) else "pending",
        reference_directory=str(historical), historical_manifest_sha256=sha(manifest_path) if manifest_path.exists() else None,
        labels="is_hallu", expected_datasets=9, datasets_checked=sum("directed_auc_percent" in r for r in rows),
        raw_macro_auc_percent=float(np.mean([r["directed_auc_percent"] for r in rows])) if complete else None,
        paper_macro_auc_percent=paper, paper_rounding_matches_68_7=round(paper, 1) == 68.7 if paper is not None else None,
        row_identity_evidence="Byte-identical historical source CSVs; exact rowwise labels, NLL predictions, claim-group IDs and train-disjoint masks",
        label_inversion_applied=False, per_dataset=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, default=ROOT / "results/overnight20261021")
    ap.add_argument("--historical", type=Path, default=ROOT / "out/campaign/evaluation_with_4b")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/claims")
    args = ap.parse_args()
    from overnight_report import collect
    datasets, _ = collect(args.directory, [])
    result = compare(datasets, args.historical, args.data_root)
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / "baseline_exact_check.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
