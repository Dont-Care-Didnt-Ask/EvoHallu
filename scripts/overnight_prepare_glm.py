"""Prepare optional GLM writer generalization; no downloads or model calls.

Default is a read-only plan. --prepare requires the twelve completed 9B evolution
runs and writes immutable full and per-seed manifests. These manifests do not
automatically schedule or launch inference.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from overnight_prepare_confirmation import inspect_discovery, load_json, sha


ROOT = Path(__file__).resolve().parents[1]
WRITER = "zai-org/GLM-4.7-Flash"
REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
SEEDS = (2107, 4319)
WRITER_CUTOFF = "2026-10-22T01:10:00+00:00"
CONTROLLER_DEADLINE = "2026-10-22T01:40:00+00:00"


def config_bytes(policy, seed):
    config = dict(defaults=["/config_uq_trial", {"override /model": "glm47_flash"}, "_self_"],
        prefix="research20261021", run_identifier_name="nextglm_evo_" + policy, seed=seed,
        search_mode="evolution", num_rounds=30, num_outputs_per_prompt=2,
        trial=dict(prompt_policy=policy, save_train_predictions=True, shortlist_size=3,
            evaluation_workers=1, max_seconds=5400, max_total_tokens=2000000,
            max_consecutive_failed_rounds=31, enable_thinking=False, local_only=True))
    return ("# @package _global_\n" + OmegaConf.to_yaml(OmegaConf.create(config))).encode()


def verify_discovery(source, path, source_sha, selection, root):
    if not selection["ready"]:
        raise ValueError("All twelve full-budget 9B evolution runs must finish before preparing GLM")
    readiness = json.loads((path.parent / "readiness.json").read_text())
    if not readiness.get("cpu_gate_passed") or not readiness.get("all_ten_supplied_csvs_identical"):
        raise ValueError("The discovery readiness audit must pass")
    if source.get("labels") != "is_hallu" or not source.get("training_raw_sha256"):
        raise ValueError("Discovery must retain is_hallu and pin its training cache")
    pins = {**source["study_source_sha256"],
            source["training_raw_path"]: source["training_raw_sha256"],
            "data/claims/popqa/popqa_train_filtered_10shot.csv": source["training_csv_sha256"]}
    for name, expected in pins.items():
        if sha(root / name) != expected:
            raise ValueError("Discovery source/input changed: " + name)
    if sha(path) != source_sha:
        raise ValueError("Discovery manifest changed during preparation")
    for row in selection["observations"]:
        if sha(root / "configs" / (row["config"] + ".yaml")) != row["config_sha256"]:
            raise ValueError("Discovery config changed: " + row["job"])
        for kind in ("summary", "ledger"):
            if sha(root / row[kind + "_path"]) != row[kind + "_sha256"]:
                raise ValueError("Discovery selection artifact changed: " + row["job"])
    cfg = OmegaConf.load(root / "configs/model/glm47_flash.yaml")
    expected = dict(hf_model_id=WRITER, revision=REVISION, temperature=.7, topk=20,
                    topp=.8, max_tokens=4096, repetition_penalty=1.)
    if any(cfg[key] != value for key, value in expected.items()):
        raise ValueError("GLM model config differs from the declared controlled profile")


def exclusive_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(source, source_path, source_sha, selection, output_dir, root=ROOT):
    if not selection["ready"]:
        raise ValueError("All twelve full-budget 9B evolution runs must finish before preparing GLM")
    if output_dir.resolve() != source_path.parent.resolve():
        raise ValueError("GLM manifests must remain beside discovery readiness.json")
    outputs = [output_dir / "glm47_manifest.json", *[output_dir / f"glm47_seed{s}_manifest.json" for s in SEEDS]]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "glm47_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(path.exists() for path in outputs):
            raise FileExistsError("A GLM manifest already exists; refusing to overwrite or reselect")
        verify_discovery(source, source_path, source_sha, selection, root)
        jobs, configs = [], []
        for seed in SEEDS:
            for policy in ("original", selection["selected_policy"]):
                name = f"nextglm_evo_{policy}_{seed}"
                config = "next_experiments/" + name
                path = root / "configs" / (config + ".yaml")
                data = config_bytes(policy, seed)
                if path.exists() and path.read_bytes() != data:
                    raise FileExistsError("Existing GLM config differs: " + str(path))
                configs.append((path, data))
                jobs.append(dict(id=name, phase="evolution_screen", study_phase="writer_generalization",
                    mode="evolution", prompt_policy=policy, seed=seed, requests=60, config=config,
                    command=[".venv/bin/python", "src/experiments/main.py", "--config-name", config],
                    config_sha256=hashlib.sha256(data).hexdigest(),
                    expected_logs=f"out/logs/research20261021/uncertainty_quantification_{name}"))
        pins = dict(source["study_source_sha256"])
        for name in ["configs/model/glm47_flash.yaml", "scripts/overnight_prepare_glm.py",
                     "scripts/overnight_prepare_confirmation.py", "scripts/overnight_start_glm.py"]:
            pins[name] = sha(root / name)
        provenance = dict(source_manifest=str(source_path.relative_to(root)), source_manifest_sha256=source_sha,
                          selection_implementation="scripts/overnight_prepare_confirmation.py::inspect_discovery",
                          **selection)
        manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            status="prepared_only_optional_not_scheduled", study_phase="writer_generalization",
            feature_model=source["feature_model"], labels="is_hallu", writer=WRITER, writer_revision=REVISION,
            seeds=list(SEEDS), jobs=jobs, training_raw_path=source["training_raw_path"],
            training_raw_sha256=source["training_raw_sha256"], training_csv_sha256=source["training_csv_sha256"],
            study_source_sha256=pins, requests_by_phase={"evolution_screen": 240}, selection_provenance=provenance,
            decoding_profile=dict(temperature=.7, top_p=.8, top_k=20, max_tokens=4096,
                repetition_penalty=1., enable_thinking=False, precision="bfloat16",
                official_general_profile=dict(temperature=1., top_p=.95, max_new_tokens=131072),
                official_coding_benchmark_profile=dict(temperature=.7, top_p=1., max_new_tokens=16384),
                interpretation="Common controlled decoding intentionally differs from GLM benchmark recommendations; not a reproduction of those benchmarks."),
            optional_schedule=dict(automatic_inference=False, only_after_27b=False,
                after_completed_source_discovery=True, after_optional_27b_decision=True, pilot_seed=2107,
                extension_seed=4319, extension_requires_measured_pilot_throughput=True,
                extension_minimum_minutes_before_writer_cutoff=45, writer_cutoff_utc=WRITER_CUTOFF,
                controller_deadline_utc=CONTROLLER_DEADLINE, controller_reserve_minutes=30,
                preserve_full_nine_dataset_feature_transfer=True,
                decision_inputs="Source validation, completion status, wall time, and pilot throughput; no test-based policy or seed selection."),
            sources=dict(model_card=f"https://huggingface.co/{WRITER}/blob/{REVISION}/README.md",
                model_config=f"https://huggingface.co/{WRITER}/resolve/{REVISION}/config.json",
                vllm_support="https://docs.vllm.ai/en/v0.31.0/models/supported_models/"))
        for path, data in configs:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(data)
        exclusive_json(outputs[0], manifest)
        for seed, path in zip(SEEDS, outputs[1:]):
            pair = copy.deepcopy(manifest)
            pair.update(seeds=[seed], jobs=[j for j in jobs if j["seed"] == seed],
                        requests_by_phase={"evolution_screen": 120},
                        parent_glm_manifest=str(outputs[0].relative_to(root)), parent_glm_manifest_sha256=sha(outputs[0]),
                        pair_role="pilot" if seed == 2107 else "conditional_throughput_extension")
            exclusive_json(path, pair)
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "results/next_experiments/run_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/next_experiments")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    source_path, output_dir = args.source_manifest.resolve(), args.output_dir.resolve()
    source, digest = load_json(source_path)
    selection = inspect_discovery(source)
    print(json.dumps(dict(event="glm_preparation_plan", ready=selection["ready"], selected_policy=selection["selected_policy"],
        writer=WRITER, revision=REVISION, jobs=4, requests=240, seeds=list(SEEDS),
        missing_discovery_jobs=selection["missing"], automatic_inference=False, writer_cutoff_utc=WRITER_CUTOFF), indent=2))
    if not args.prepare:
        print("DRY RUN ONLY. No downloads, model calls, or generated files.")
        return
    for path in prepare(source, source_path, digest, selection, output_dir):
        print("Prepared immutable optional manifest: " + str(path))
    for seed in SEEDS:
        print(f".venv/bin/python scripts/run_overnight_parallel.py --execute --manifest results/next_experiments/glm47_seed{seed}_manifest.json "
              f"--state results/next_experiments/glm47_seed{seed}_execution_state.json --ports 18000 18001 "
              f"--deadline-utc {CONTROLLER_DEADLINE} --reserve-minutes 30")


if __name__ == "__main__":
    main()
