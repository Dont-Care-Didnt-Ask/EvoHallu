"""Optional GLM preparation and startup guards using synthetic CPU-only inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys

from omegaconf import OmegaConf
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


preparer = module("overnight_prepare_glm")
starter = module("overnight_start_glm")


def source_study(root):
    jobs = []
    for seed in (2107, 4319, 8741):
        for policy, score in [("original", 90), ("neutral", 80), ("diverse", 82), ("compositional", 81)]:
            name = f"{policy}_{seed}"
            folder = root / "logs" / name
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(dict(status="finished", requests=60,
                selected_validation_auc_percent=score)))
            (folder / "candidates.jsonl").write_text("{}\n" * 60)
            (folder / "frozen_estimator.py").write_text("# selected")
            jobs.append(dict(id=name, mode="evolution", phase="evolution_screen", prompt_policy=policy,
                seed=seed, requests=60, expected_logs=f"logs/{name}", config=name, config_sha256="synthetic-config-pin"))
    return dict(writer="Qwen/Qwen3.5-9B", writer_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        jobs=jobs, feature_model="Llama3.1-8b-instruct", labels="is_hallu", training_raw_path="raw.pt",
        training_raw_sha256="raw-pin", training_csv_sha256="csv-pin", study_source_sha256={"original.py": "original-pin"})


def test_preparation_requires_twelve_complete_evolution_runs(tmp_path):
    source = source_study(tmp_path)
    (tmp_path / source["jobs"][0]["expected_logs"] / "summary.json").unlink()
    selection = preparer.inspect_discovery(source, tmp_path)
    with pytest.raises(ValueError, match="twelve full-budget"):
        preparer.prepare(source, tmp_path / "run_manifest.json", "hash", selection, tmp_path, tmp_path)
    assert not (tmp_path / "glm47_prepare.lock").exists()


def test_glm_manifests_pin_same_selected_policy_and_both_pairs(tmp_path, monkeypatch):
    source = source_study(tmp_path)
    selection = preparer.inspect_discovery(source, tmp_path)
    assert selection["selected_policy"] == "diverse"
    source_path = tmp_path / "run_manifest.json"
    source_path.write_text(json.dumps(source))
    for name in ["configs/model/glm47_flash.yaml", "scripts/overnight_prepare_glm.py",
                 "scripts/overnight_prepare_confirmation.py", "scripts/overnight_start_glm.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pinned source bytes")
    monkeypatch.setattr(preparer, "verify_discovery", lambda *a, **k: None)
    outputs = preparer.prepare(source, source_path, preparer.sha(source_path), selection, tmp_path, tmp_path)
    assert len(outputs) == 3
    full = json.loads(outputs[0].read_text())
    assert full["writer"] == starter.WRITER and full["writer_revision"] == starter.REVISION
    assert len(full["jobs"]) == 4 and full["requests_by_phase"]["evolution_screen"] == 240
    assert full["study_source_sha256"]["original.py"] == "original-pin"
    assert "configs/model/qwen38_27b.yaml" not in full["study_source_sha256"]
    assert full["selection_provenance"]["held_out_test_scores_read"] is False
    assert full["optional_schedule"]["extension_minimum_minutes_before_writer_cutoff"] == 45
    for pair_path, seed in zip(outputs[1:], (2107, 4319)):
        pair = json.loads(pair_path.read_text())
        assert {j["seed"] for j in pair["jobs"]} == {seed}
        assert {j["prompt_policy"] for j in pair["jobs"]} == {"original", "diverse"}
        assert pair["parent_glm_manifest_sha256"] == preparer.sha(outputs[0])
        for job in pair["jobs"]:
            path = tmp_path / "configs" / (job["config"] + ".yaml")
            cfg = OmegaConf.load(path)
            assert cfg.num_rounds == 30 and cfg.num_outputs_per_prompt == 2
            assert not cfg.trial.enable_thinking and cfg.trial.local_only
            assert cfg.trial.save_train_predictions and cfg.trial.evaluation_workers == 1
            assert preparer.sha(path) == job["config_sha256"]
    before = outputs[0].read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preparer.prepare(source, source_path, preparer.sha(source_path), selection, tmp_path, tmp_path)
    assert outputs[0].read_bytes() == before


def checkpoint(root, monkeypatch):
    monkeypatch.setattr(starter, "WEIGHT_BYTES", 4)
    monkeypatch.setattr(starter, "WEIGHT_SHARDS", 2)
    siblings, receipts = [], []
    for name in ("a.safetensors", "b.safetensors"):
        (root / name).write_bytes(b"ok")
        digest = hashlib.sha256(b"ok").hexdigest()
        siblings.append(dict(rfilename=name, size=2, lfs=dict(sha256=digest)))
        receipts.append(dict(file=name, bytes=2, sha256=digest))
    (root / "download_manifest.json").write_text(json.dumps(dict(model=starter.WRITER, revision=starter.REVISION, files=receipts)))
    (root / "download_metadata.json").write_text(json.dumps(dict(sha=starter.REVISION, private=False, gated=False, siblings=siblings)))
    (root / "model.safetensors.index.json").write_text(json.dumps(dict(metadata=dict(total_size=2), weight_map=dict(a="a.safetensors", b="b.safetensors"))))
    (root / "config.json").write_text(json.dumps(dict(architectures=["Glm4MoeLiteForCausalLM"], dtype="bfloat16")))
    (root / "chat_template.jinja").write_text("enable_thinking")


def test_checkpoint_uses_actual_shard_bytes_instead_of_index_size(tmp_path, monkeypatch):
    checkpoint(tmp_path, monkeypatch)
    result = starter.verify_checkpoint(tmp_path)
    assert result["actual_weight_bytes"] == 4
    assert result["index_total_size_ignored"] == 2
    (tmp_path / "a.safetensors").write_bytes(b"bad size")
    with pytest.raises(ValueError, match="Actual GLM shard size"):
        starter.verify_checkpoint(tmp_path)


def test_cached_receipt_without_hash_is_rehashed(tmp_path, monkeypatch):
    checkpoint(tmp_path, monkeypatch)
    path = tmp_path / "download_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0].pop("sha256")
    path.write_text(json.dumps(manifest))
    result = starter.verify_checkpoint(tmp_path)
    assert result["hash_verification"][0]["verification"] == "actual_file_rehashed"
    (tmp_path / "a.safetensors").write_bytes(b"no")
    with pytest.raises(ValueError, match="hash differs"):
        starter.verify_checkpoint(tmp_path)


def test_server_command_is_eager_bf16_glm_parser_and_thinking_disabled():
    command = starter.server_command(Path("/models/GLM"), 18000)
    assert command[command.index("--reasoning-parser") + 1] == "glm45"
    assert "qwen3" not in command
    assert command[command.index("--attention-backend") + 1] == "TRITON_MLA"
    assert command[command.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert json.loads(command[command.index("--default-chat-template-kwargs") + 1]) == dict(enable_thinking=False)
    assert "--enforce-eager" in command
    assert "--enforce-eager" not in starter.server_command(Path("/models/GLM"), 18000, compiled=True)


@pytest.mark.parametrize("used,utilization,process", [(600, 0, False), (0, 10, False), (0, 0, True)])
def test_gpu_guards_refuse_nonidle_cards(used, utilization, process):
    snapshot = dict(gpus=[dict(index=0, uuid="GPU0", total_memory_MiB=81000, used_memory_MiB=used,
                               utilization_percent=utilization)], compute_processes=[dict(gpu_uuid="GPU0")] if process else [])
    with pytest.raises(RuntimeError, match="not idle"):
        starter.require_idle(snapshot, [0])


def test_gpu_guard_accepts_idle_80gb_card_and_refuses_small_card():
    snapshot = dict(gpus=[dict(index=0, uuid="GPU0", total_memory_MiB=81000,
        used_memory_MiB=20, utilization_percent=0)], compute_processes=[])
    starter.require_idle(snapshot, [0])
    snapshot["gpus"][0]["total_memory_MiB"] = 48000
    with pytest.raises(ValueError, match="80 GB"):
        starter.require_idle(snapshot, [0])


def test_port_guard_refuses_occupied_local_service():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="occupied"):
            starter.require_free_ports([listener.getsockname()[1]])


def test_dry_start_never_checks_hardware_or_spawns(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["overnight_start_glm.py"])
    monkeypatch.setattr(starter, "telemetry", lambda: pytest.fail("Dry run must not query GPU hardware"))
    monkeypatch.setattr(starter.subprocess, "Popen", lambda *a, **k: pytest.fail("Dry run must not spawn servers"))
    starter.main()
    assert json.loads(capsys.readouterr().out)["no_inference"] is True
