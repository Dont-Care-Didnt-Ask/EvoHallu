"""Synthetic protocol checks; no model or held-out evaluation is invoked."""
import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/overnight_prepare_confirmation.py"
SPEC = importlib.util.spec_from_file_location("confirmation_preparer", SCRIPT)
preparer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparer)


def discovery(tmp_path, scores=None):
    scores = scores or {"original": [95, 95, 95], "neutral": [80, 80, 80],
                        "diverse": [79, 85, 82], "compositional": [81, 81, 81]}
    jobs = []
    for index, seed in enumerate(preparer.DISCOVERY_SEEDS):
        for policy in preparer.POLICIES:
            identifier = f"discovery_{policy}_{seed}"
            folder = tmp_path / "logs" / identifier
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(dict(status="finished", requests=60,
                selected_validation_auc_percent=scores[policy][index])))
            (folder / "candidates.jsonl").write_text("{}\n" * 60)
            (folder / "frozen_estimator.py").write_text("# selected estimator")
            jobs.append(dict(id=identifier, mode="evolution", phase="compositional_control" if policy == "compositional" else "evolution_screen",
                prompt_policy=policy, seed=seed, requests=60, expected_logs=f"logs/{identifier}",
                config=f"discovery/{identifier}", config_sha256="synthetic-config-hash"))
    return dict(writer=preparer.SOURCE_WRITER, writer_revision=preparer.SOURCE_REVISION,
                jobs=jobs, feature_model="Llama3.1-8b-instruct", labels="is_hallu",
                training_csv_sha256="csv-pin", training_raw_path="raw.pt", training_raw_sha256="raw-pin",
                study_source_sha256={"source.py": "original-pin"})


def test_selection_uses_all_three_validation_seeds_and_excludes_original(tmp_path):
    manifest = discovery(tmp_path)
    selected = preparer.inspect_discovery(manifest, tmp_path)
    assert selected["ready"]
    assert selected["selected_policy"] == "diverse"
    assert selected["mean_selected_validation_auc_percent"]["diverse"] == 82
    assert selected["mean_selected_validation_auc_percent"]["original"] == 95
    assert len(selected["observations"]) == 12
    assert all(len(row["summary_sha256"]) == 64 for row in selected["observations"])
    assert selected["held_out_test_scores_read"] is False


def test_ties_follow_predeclared_policy_order(tmp_path):
    scores = {p: [80, 80, 80] for p in preparer.POLICIES}
    manifest = discovery(tmp_path, scores)
    assert preparer.inspect_discovery(manifest, tmp_path)["selected_policy"] == "neutral"


def test_single_partial_run_prevents_preparation(tmp_path):
    manifest = discovery(tmp_path)
    row = manifest["jobs"][-1]
    path = tmp_path / row["expected_logs"] / "summary.json"
    path.write_text(json.dumps(dict(status="finished", requests=58, selected_validation_auc_percent=99)))
    selection = preparer.inspect_discovery(manifest, tmp_path)
    assert not selection["ready"]
    assert selection["selected_policy"] is None
    assert selection["missing"][0]["job"] == row["id"]
    with pytest.raises(ValueError, match="all 12 full-budget"):
        preparer.prepare(manifest, tmp_path / "run_manifest.json", "hash", selection,
                         tmp_path / "confirmation27b_manifest.json", tmp_path)
    assert not (tmp_path / "confirmation27b_manifest.json.prepare.lock").exists()


def test_ledger_mismatch_refuses_full_budget_claim(tmp_path):
    manifest = discovery(tmp_path)
    (tmp_path / manifest["jobs"][0]["expected_logs"] / "candidates.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="disagrees with ledger"):
        preparer.inspect_discovery(manifest, tmp_path)


def test_prepare_pins_four_jobs_and_does_not_overwrite(tmp_path, monkeypatch):
    manifest = discovery(tmp_path)
    selection = preparer.inspect_discovery(manifest, tmp_path)
    source = tmp_path / "results/next_experiments/run_manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(manifest))
    for name in ["configs/model/qwen38_27b.yaml", "scripts/overnight_prepare_confirmation.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic pinned bytes")
    monkeypatch.setattr(preparer, "verify_pins", lambda *a, **k: None)
    output = source.parent / "confirmation27b_manifest.json"
    result = preparer.prepare(manifest, source, preparer.sha(source), selection, output, tmp_path)
    assert len(result["jobs"]) == 4
    assert {(j["prompt_policy"], j["seed"]) for j in result["jobs"]} == {
        (p, s) for p in ("original", "diverse") for s in [2107, 4319]}
    assert result["writer_revision"] == preparer.REVISION
    assert result["study_source_sha256"]["source.py"] == "original-pin"
    assert "scripts/overnight_prepare_confirmation.py" in result["study_source_sha256"]
    assert "contexts_sha256" not in result
    for row in result["jobs"]:
        path = tmp_path / "configs" / (row["config"] + ".yaml")
        assert preparer.sha(path) == row["config_sha256"]
        cfg = OmegaConf.load(path)
        assert cfg.model.max_tokens == 4096
        assert cfg.num_rounds == 30 and cfg.num_outputs_per_prompt == 2
        assert cfg.trial.save_train_predictions and cfg.trial.shortlist_size == 3
        assert cfg.trial.evaluation_workers == 1 and cfg.trial.max_consecutive_failed_rounds == 31
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preparer.prepare(manifest, source, preparer.sha(source), selection, output, tmp_path)
    assert output.read_bytes() == before


def test_existing_different_config_refuses_before_writing_manifest(tmp_path, monkeypatch):
    manifest = discovery(tmp_path)
    selection = preparer.inspect_discovery(manifest, tmp_path)
    source = tmp_path / "run_manifest.json"
    source.write_text(json.dumps(manifest))
    monkeypatch.setattr(preparer, "verify_pins", lambda *a, **k: None)
    config = tmp_path / "configs/next_experiments/next27b_evo_original_2107.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("do not replace")
    output = tmp_path / "confirmation27b_manifest.json"
    with pytest.raises(FileExistsError, match="config differs"):
        preparer.prepare(manifest, source, preparer.sha(source), selection, output, tmp_path)
    assert not output.exists()
    assert config.read_text() == "do not replace"
