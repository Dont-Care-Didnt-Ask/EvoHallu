"""Training-only convergence diagnostics on synthetic rankings and file pins."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/overnight_convergence_analysis.py"
SPEC = importlib.util.spec_from_file_location("convergence_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_constant_reverse_and_monotonic_transform_rankings():
    values = np.arange(512)
    baseline = analysis.normalized_ranks(values)
    constant, vector = analysis.describe_vector(np.ones(512), baseline, [])
    assert constant["probe_constant"] and not constant["novel"] and vector is None
    same, _ = analysis.describe_vector(values * 2 + 100, baseline, [])
    assert same["near_baseline"] and not same["novel"]
    reverse, _ = analysis.describe_vector(-values, baseline, [])
    assert reverse["baseline_rho"] == pytest.approx(-1)
    assert reverse["novel"] and not reverse["abs_novel"]
    assert reverse["abs_near_baseline"]


def test_previous_archive_crosses_temporal_window_boundaries():
    baseline = analysis.normalized_ranks(np.arange(512))
    values = np.random.default_rng(91).permutation(512)
    previous, vectors, rows = [], {}, []
    for round_index, candidate in [(0, values), (10, values), (20, -values)]:
        description, vector = analysis.describe_vector(candidate, baseline, previous)
        row = dict(round=round_index, index=0, training_valid=True, training_useful=True, **description)
        row["novel_and_training_useful"] = row["novel"]
        row["abs_novel_and_training_useful"] = row["abs_novel"]
        rows.append(row)
        previous.append(vector)
        vectors[(round_index, 0)] = vector
    windows = [analysis.window_statistics(rows, vectors, label, start, end) for start, end, label in analysis.WINDOWS]
    assert windows[0]["novel"] == 1
    assert windows[1]["near_previous"] == 1 and windows[1]["novel"] == 0
    assert windows[2]["novel"] == 1 and windows[2]["abs_novel"] == 0
    assert windows[2]["abs_near_previous"] == 1


def test_source_hash_table_records_verified_snapshot_and_rejects_mutation(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"source":1}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = analysis.Inputs(tmp_path)
    assert inputs.json(path, digest) == dict(source=1)
    assert inputs.hashes == {"source.json": digest}
    with pytest.raises(ValueError, match="Pinned input hash differs"):
        inputs.bytes(path, "0" * 64)
    path.write_text('{"source":2}')
    with pytest.raises(ValueError, match="Input changed"):
        inputs.bytes(path)


def test_execution_source_and_config_pins_are_verified(tmp_path):
    source = tmp_path / "implementation.py"
    source.write_text("# protocol implementation")
    config = tmp_path / "configs/study.yaml"
    config.parent.mkdir()
    config.write_text("# pinned config")
    job = dict(id="run", config="study", config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        prompt_policy="neutral", seed=2107, requests=60, expected_logs="logs/run", mode="evolution")
    manifest = dict(jobs=[dict(job)], study_source_sha256={"implementation.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(manifest))
    job.update(execution_manifest="execution.json", execution_manifest_sha256=hashlib.sha256(execution.read_bytes()).hexdigest())
    inputs = analysis.Inputs(tmp_path)
    analysis.verify_execution_pins(job, {}, inputs)
    assert set(inputs.hashes) == {"execution.json", "implementation.py", "configs/study.yaml"}
    source.write_text("# changed protocol")
    with pytest.raises(ValueError, match="Pinned input hash differs"):
        analysis.verify_execution_pins(job, {}, analysis.Inputs(tmp_path))


def test_probe_and_split_validation():
    probe = dict(validation_used=False, labels="is_hallu", probe_rows=512, train_rows=600,
                 ordered_training_prediction_positions=list(range(512)), split_seed=441, split_strategy="stratified")
    assert len(analysis.validate_probe(probe)) == 512
    bad = dict(probe, ordered_training_prediction_positions=[0] * 512)
    with pytest.raises(ValueError, match="512 distinct"):
        analysis.validate_probe(bad)
    cfg = dict(task=dict(split_seed=441, val_size=.2, sample_size=0),
        trial=dict(prompt_policy="neutral", save_train_predictions=True), search_mode="evolution", seed=2107,
        num_rounds=30, num_outputs_per_prompt=2, model=dict(hf_model_id="writer"))
    job = dict(id="run", seed=2107, prompt_policy="neutral", writer_model="writer")
    analysis.verify_run_config(job, cfg, probe)
    cfg["task"]["split_seed"] = 442
    with pytest.raises(ValueError, match="split differs"):
        analysis.verify_run_config(job, cfg, probe)


def test_partial_or_missing_predictions_cannot_enter_primary_three_seed_averages():
    runs = []
    for number, seed in enumerate(analysis.PRIMARY_SEEDS, 1):
        runs.append(dict(writer="writer", phase="evolution_screen", policy="neutral", parent_sampling="inherited",
            seed=seed, full_budget=number != 3, diagnostic_complete=number != 3,
            windows=[dict(window=label, novel=number) for _, _, label in analysis.WINDOWS]))
    group = analysis.summarize_groups(runs)[0]
    assert not group["primary_three_seed_ready"] and not group["primary_windows"]
    runs[-1]["full_budget"] = True
    assert not analysis.summarize_groups(runs)[0]["primary_three_seed_ready"]
    runs[-1]["diagnostic_complete"] = True
    group = analysis.summarize_groups(runs)[0]
    assert group["primary_three_seed_ready"]
    assert group["primary_windows"][0]["metrics"]["novel"]["mean"] == 2
    assert group["primary_windows"][0]["metrics"]["novel"]["std"] == 1


def test_interface_control_is_separate_from_ordinary_neutral_evolution():
    runs = []
    for phase, novel in [("evolution_screen", 1), ("interface_control", 8)]:
        for seed in analysis.PRIMARY_SEEDS:
            runs.append(dict(writer="writer", phase=phase, policy="neutral", parent_sampling="inherited",
                seed=seed, full_budget=True, diagnostic_complete=True,
                windows=[dict(window=label, novel=novel) for _, _, label in analysis.WINDOWS]))
    groups = {row["phase"]: row for row in analysis.summarize_groups(runs)}
    assert len(groups) == 2 and all(row["primary_three_seed_ready"] for row in groups.values())
    assert groups["evolution_screen"]["primary_windows"][0]["metrics"]["novel"]["mean"] == 1
    assert groups["interface_control"]["primary_windows"][0]["metrics"]["novel"]["mean"] == 8
    job = dict(id="interface", phase="evolution_screen", analysis_phase="interface_control", prompt_policy="neutral", seed=2107,
               expected_logs="missing")
    run = analysis.analyse_run(job, dict(writer="writer"), {}, [], analysis.Inputs(SCRIPT.parents[1]), [])
    assert run["phase"] == "interface_control" and run["status"] == "not_run"
