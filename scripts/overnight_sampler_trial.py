"""Opt-in rank-parent sampling trial without modifying the pinned search code.

Only the score-group weighting call in Island.get_prompt is intercepted.
Quantile eligibility, island choice, sampling without replacement, and the
short-source preference inside a score group use the inherited implementation.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from hydra import main as hydra_main
from omegaconf import OmegaConf
from scipy.stats import rankdata


def rank_uniform_probabilities(scores, uniform_reserve=0.2):
    """Mix ascending linear fitness ranks with uniform exploration.

    For n distinct eligible score groups, the best rank is n and the worst 1:
    p_i = (1-u)*rank_i/[n*(n+1)/2] + u/n. Ties receive average ranks.
    This changes fitness selection pressure; it is not a novelty archive.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Parent fitness scores must be a nonempty finite 1D vector")
    if not np.isfinite(uniform_reserve) or not 0 <= uniform_reserve <= 1:
        raise ValueError("uniform_reserve must be finite and between zero and one")
    ranks = rankdata(values, method="average")
    probabilities = (1 - uniform_reserve) * ranks / ranks.sum() + uniform_reserve / len(values)
    # Match the numerical sum guard in the inherited softmax.
    index = int(np.argmax(probabilities))
    probabilities[index] = 1 - probabilities[:index].sum() - probabilities[index + 1:].sum()
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("Invalid rank-parent probabilities")
    return probabilities


@contextlib.contextmanager
def rank_parent_sampling(database_module, uniform_reserve=0.2, on_call=None):
    """Intercept the eligible-group softmax, preserving within-group softmax.

    The caller code-object check is deliberate: both Island.get_prompt and
    Cluster.sample_program_and_metrics call the same module-level _softmax.
    Replacing _softmax globally with ranks would also change source-length
    selection, confounding this experiment. All other call sites delegate to
    the original function with their original temperature.
    """
    original = database_module._softmax
    parent_code = database_module.Island.get_prompt.__code__
    calls = []

    def dispatch(logits, temperature):
        if sys._getframe(1).f_code is parent_code:
            probabilities = rank_uniform_probabilities(logits, uniform_reserve)
            event = dict(eligible_scores=np.asarray(logits, dtype=float).tolist(),
                         probabilities=probabilities.tolist(),
                         inherited_temperature=float(temperature),
                         uniform_reserve=float(uniform_reserve))
            calls.append(event)
            if on_call is not None:
                on_call(event)
            return probabilities
        return original(logits, temperature)

    database_module._softmax = dispatch
    try:
        yield calls
    finally:
        database_module._softmax = original


@hydra_main(config_path="../configs", config_name="config_uq_trial", version_base="1.3")
def main(cfg):
    from packing.funsearch import programs_database
    from experiments.extension_trial import run_trial

    if not cfg.get("extension_trial") or cfg.search_mode != "evolution":
        raise ValueError("Rank-parent wrapper requires an evolutionary extension trial")
    if cfg.trial.get("parent_sampling") != "rank_uniform":
        raise ValueError("Explicit trial.parent_sampling=rank_uniform is required")
    if not cfg.programdatabaseConfig.temp_sampling_flag:
        raise ValueError("Rank-parent trial requires stochastic score-group sampling")
    reserve = float(cfg.trial.get("parent_sampling_uniform_reserve", 0.2))
    if reserve != 0.2:
        raise ValueError("The overnight arm fixes the uniform reserve at 0.2")
    if cfg.trial.get("resume", False):
        raise ValueError("Rank-parent overnight arms require fresh run names; partial runs are not resumed")
    output = Path(cfg.logs_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        parent_sampling="rank_uniform", uniform_reserve=reserve,
        formula="p_i = 0.8 * ascending_average_rank(fitness_i) / sum(ranks) + 0.2 / n_eligible_groups",
        eligibility="Unchanged Island.get_prompt percentile gate",
        island_selection="Unchanged ProgramsDatabase.get_prompt",
        within_score_group="Unchanged shorter-source softmax at temperature 1.0",
        original_temperature_used_for_parent_weights=False,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inherited_database_sha256=hashlib.sha256(Path(programs_database.__file__).read_bytes()).hexdigest(),
        training_fitness="directed AUROC", validation_selection="unchanged directed AUROC",
        label_field="is_hallu", test_labels_used=False,
    )
    provenance_path = output / "parent_sampling_provenance.json"
    if provenance_path.exists() or (output / "candidates.jsonl").exists():
        raise FileExistsError("Refusing to overwrite rank-parent trial artifacts: " + str(output))
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    event_path = output / "parent_sampling_events.jsonl"

    def record(event):
        with event_path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    OmegaConf.set_struct(cfg, False)
    with rank_parent_sampling(programs_database, reserve, on_call=record) as calls:
        run_trial(cfg)
    summary = json.loads((output / "summary.json").read_text())
    if len(calls) * int(cfg.num_outputs_per_prompt) != summary["requests"]:
        raise ValueError("Parent-weight hook count does not match actual issued round budget")


if __name__ == "__main__":
    main()
