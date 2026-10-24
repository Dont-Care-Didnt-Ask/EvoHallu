"""Verify a completed larger cohort preserves the original benchmark outputs."""
import argparse
import json
from pathlib import Path
import numpy as np


def verify_cohort(original, extended, new_prefix="new/uncertainty_quantification_qwen35_4b_"):
    original, extended = Path(original), Path(extended)
    marker = extended / "cohort_verified.json"
    marker.unlink(missing_ok=True)
    before = json.loads((original / "all_datasets.json").read_text())
    after = json.loads((extended / "all_datasets.json").read_text())
    if len(before) != 9 or set(before) != set(after):
        raise ValueError("Both cohorts must contain the same nine datasets")
    compared = 0
    for name in before:
        if not set(before[name]["methods"]) <= set(after[name]["methods"]):
            raise ValueError(f"Previously available method missing on {name}")
        attempted = set(after[name]["methods"]) | set(after[name].get("failed", {}))
        if not any(method.startswith(new_prefix) for method in attempted):
            raise ValueError(f"New writer was not evaluated on {name}")
        with np.load(original / (name+"_scores.npz")) as old, np.load(extended / (name+"_scores.npz")) as new:
            for key in old.files:
                if key not in new.files or not np.array_equal(old[key], new[key], equal_nan=True):
                    raise ValueError(f"Original predictions changed on {name}: {key}")
                compared += 1
    result = {"common_method_predictions_identical": True, "datasets": 9,
              "arrays_compared": compared, "original": str(original), "extended": str(extended)}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    temporary.replace(marker)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path("out/campaign/evaluation"))
    parser.add_argument("--extended", type=Path, default=Path("out/campaign/evaluation_with_4b"))
    parser.add_argument("--new-prefix", default="new/uncertainty_quantification_qwen35_4b_")
    args = parser.parse_args()
    print(json.dumps(verify_cohort(args.original, args.extended, args.new_prefix)))


if __name__ == "__main__":
    main()
