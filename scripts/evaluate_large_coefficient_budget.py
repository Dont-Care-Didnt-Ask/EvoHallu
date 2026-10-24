"""Freeze and test the validation winners of 50 x 60 restricted candidates.

This supplementary higher-budget arm leaves the 60-candidate primary controls
unchanged. Its selection reads only the saved validation-only seed sweep.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from search_linear_controls import SCALES, FEATURE_NAMES


def main():
    root = Path("out/campaign/large_linear_budget_frozen")
    records = json.loads(Path("out/campaign/validation_linear_seed_sweep/candidates.json").read_text())
    sweep = json.loads(Path("out/campaign/validation_linear_seed_sweep/summary.json").read_text())
    if not sweep["original_four_controls_reproduced_exactly"] or sweep["held_out_datasets_read"]:
        raise ValueError("Expected a verified validation-only seed sweep")
    for mode in ["independent", "evolution"]:
        selected_run = max((r for r in records if r["mode"] == mode), key=lambda r:r["selected_validation_auc_percent"])
        selected = max(selected_run["candidate_records"], key=lambda r:r["validation_auc"])
        name = "linear_best50_"+mode
        run = root / name
        run.mkdir(parents=True, exist_ok=True)
        weights = selected["weights"]
        code = f'''from packing.evaluate.uncertainty_quantification.supervised_baselines import compact_features
class HallucinationDetector(Estimator):
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs", "token_entropies", "top_k_logits"], level="sequence")
    def __call__(self, stats):
        weights = np.array({weights!r})
        scales = np.array({SCALES.tolist()!r})
        return (compact_features(stats) / scales) @ weights
'''
        frozen = run / "frozen_estimator.py"
        if frozen.exists() and frozen.read_text() != code:
            raise ValueError("Refusing to replace an already frozen coefficient winner")
        frozen.write_text(code)
        summary = {"status": "finished", "mode": mode, "candidate_budget": 50*60,
                   "seed": selected_run["seed"], "source": "validation-only 50-seed sweep",
                   "selected_validation_auc_percent": selected_run["selected_validation_auc_percent"],
                   "selected_train_auc_percent": 100*selected["train_auc"], "weights": weights,
                   "feature_pool": FEATURE_NAMES, "primary_winners_changed": False,
                   "selection_used_held_out_scores": False}
        (run / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    env = os.environ.copy()
    env.update(PYTHONPATH="src:.", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "scripts/evaluate_parallel.py", "--workers", "3", "--bootstrap", "500",
                    "--output", "out/campaign/linear_best50", "--frozen-glob", str(root / "*" / "frozen_estimator.py"),
                    "--method-prefix", "new/linear_best50_"], env=env, check=True)


if __name__ == "__main__":
    main()
