"""Score one additional writer and verify its addition to a completed cohort."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from merge_evaluation_cohorts import merge_cohorts
from verify_augmented_evaluation import verify_cohort


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--run-prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append")
    args = parser.parse_args()
    candidates = args.reference or [Path("out/campaign/evaluation_with_structured_4b"),
                                   Path("out/campaign/evaluation_with_4b"), Path("out/campaign/evaluation")]
    reference = None
    for directory in candidates:
        aggregate = directory / "all_datasets.json"
        marker = directory / "cohort_verified.json"
        verified = marker.exists() and json.loads(marker.read_text()).get("common_method_predictions_identical", False)
        initial = directory in {Path("out/campaign/evaluation"), Path("out/campaign/evaluation_before_9b")}
        if aggregate.exists() and len(json.loads(aggregate.read_text())) == 9 and (verified or initial):
            reference = directory
            break
    if reference is None:
        raise ValueError("A complete reference cohort is required")
    paths = ([args.run/"frozen_estimator.py"] if args.run else
             sorted(Path("out/logs/mod1").glob(args.run_prefix+"*/frozen_estimator.py")))
    if not paths or any(json.loads((p.parent/"summary.json").read_text())["status"] != "finished" for p in paths):
        raise ValueError("The added writers must be frozen before evaluation")
    delta = args.output.with_name(args.output.name+"_delta")
    prefix = "new/"+(args.run.name if args.run else args.run_prefix)
    frozen_glob = str(args.run/"frozen_estimator.py") if args.run else "out/logs/mod1/"+args.run_prefix+"*/frozen_estimator.py"
    subprocess.run([sys.executable, "scripts/evaluate_parallel.py", "--workers", "3", "--bootstrap", "500",
                    "--output", str(delta), "--frozen-glob", frozen_glob,
                    "--method-prefix", prefix], check=True)
    merge_cohorts(reference, delta, args.output)
    print(json.dumps(verify_cohort(reference, args.output, new_prefix=prefix)))


if __name__ == "__main__":
    main()
