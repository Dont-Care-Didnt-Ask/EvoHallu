"""Audit the supplied splits without exposing individual claim text."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/claims"))
    ap.add_argument("--output", type=Path, default=Path("results/dataset_audit.json"))
    args = ap.parse_args()
    report, claim_sets = [], {}
    for path in sorted(args.root.rglob("*.csv")):
        df = pd.read_csv(path, usecols=["claim", "is_hallu"])
        labels = pd.to_numeric(df.is_hallu, errors="raise")
        if not set(labels.unique()) <= {0, 1}:
            raise ValueError(f"Nonbinary labels: {path}")
        claims = df.claim.fillna("").astype(str).str.split().str.join(" ").str.casefold()
        name = str(path.relative_to(args.root))
        claim_sets[name] = set(claims)
        grouped = pd.DataFrame({"claim": claims, "label": labels}).groupby("claim").label.nunique()
        row = {"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "rows": len(df), "class_counts": {str(int(k)): int(v) for k, v in labels.value_counts().items()},
               "empty_claims": int(claims.eq("").sum()), "duplicate_claim_rows": int(claims.duplicated().sum()),
               "contradictory_claims": int(grouped.gt(1).sum())}
        if "popqa_train" in name:
            train, val = train_test_split(df.index.to_numpy(), test_size=0.2, stratify=labels, random_state=441)
            row.update(train_rows=len(train), validation_rows=len(val),
                       train_validation_claim_overlap=len(set(claims.iloc[train]) & set(claims.iloc[val])))
        report.append(row)
    train_name = next(k for k in claim_sets if "popqa_train" in k)
    for row in report:
        if row["file"] != train_name:
            row["overlap_with_popqa_train_claims"] = len(claim_sets[train_name] & claim_sets[row["file"]])
    result = {"split_seed": 441, "validation_fraction": 0.2, "datasets": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
