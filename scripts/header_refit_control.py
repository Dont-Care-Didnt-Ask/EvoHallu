"""Exploratory source-training-only refit after the fixed header perturbation.

The same CV settings, split and four model families are used. Earlier frozen
heads and predictions stay intact, so input sensitivity and refitting differ.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from merge_evaluation_cohorts import merge_cohorts
from verify_augmented_evaluation import verify_cohort


def main():
    env = os.environ.copy()
    env.update(PYTHONPATH="src:.", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    paths = []
    for feature_set in ["full", "documented"]:
        directory = Path("out/campaign/header_refit_"+feature_set)
        subprocess.run([sys.executable, "scripts/fit_supervised_baselines.py", "--output", str(directory),
                        "--feature-set", feature_set, "--trim-header-suffix", "4",
                        "--model-name-prefix", "header_refit/"], env=env, check=True)
        paths += ["--extra-fitted-models", str(directory / "models.joblib")]
    delta = Path("out/campaign/header_refit_delta")
    command = [sys.executable, "scripts/evaluate_parallel.py", "--workers", "3", "--bootstrap", "500",
               "--trim-header-suffix", "4", "--output", str(delta),
               "--frozen-glob", "out/campaign/no_frozen_programs.py", "--method-prefix", "supervised/header_refit/", *paths]
    subprocess.run(command, env=env, check=True)
    original = Path("out/campaign/header_suffix_ablation")
    output = Path("out/campaign/header_suffix_ablation_refitted")
    merge_cohorts(original, delta, output)
    print(json.dumps(verify_cohort(original, output, new_prefix="supervised/header_refit/")), flush=True)


if __name__ == "__main__":
    main()
