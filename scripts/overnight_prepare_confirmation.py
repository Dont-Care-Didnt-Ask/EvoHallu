"""Prepare a validation-selected 27B writer confirmation after the 9B suite.

Default is a read-only plan. --prepare writes four configs and an immutable
manifest. No held-out test scores are read and no model requests are made.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("original", "neutral", "diverse", "compositional")
NONORIGINAL = POLICIES[1:]
DISCOVERY_SEEDS = (2107, 4319, 8741)
CONFIRMATION_SEEDS = (2107, 4319)
WRITER = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_WRITER = "Qwen/Qwen3.5-9B"
SOURCE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RULE = ("Among neutral, diverse, and compositional, select the largest arithmetic mean "
        "selected_validation_auc_percent over all three full-budget 9B evolution seeds "
        "2107, 4319, 8741. Exact ties use neutral, then diverse, then compositional. "
        "Original is the comparison arm, never a candidate for the selected arm.")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def inspect_discovery(manifest, root=ROOT):
    if manifest.get("writer") != SOURCE_WRITER or manifest.get("writer_revision") != SOURCE_REVISION:
        raise ValueError("Discovery manifest does not identify the pinned 9B writer")
    relevant = [j for j in manifest["jobs"] if j.get("mode") == "evolution"
                and j.get("prompt_policy") in POLICIES and j.get("seed") in DISCOVERY_SEEDS]
    lookup = {}
    for item in relevant:
        key = item["prompt_policy"], item["seed"]
        if key in lookup:
            raise ValueError("Duplicate discovery policy/seed: " + str(key))
        if item["requests"] != 60:
            raise ValueError("Discovery evolution must use exactly 60 requests: " + item["id"])
        lookup[key] = item
    missing, observations = [], []
    for seed in DISCOVERY_SEEDS:
        for policy in POLICIES:
            item = lookup.get((policy, seed))
            if item is None:
                raise ValueError(f"Discovery manifest is missing {policy}/{seed}")
            folder = root / item["expected_logs"]
            path = folder / "summary.json"
            if not path.exists():
                missing.append(dict(job=item["id"], reason="summary_missing"))
                continue
            summary, digest = load_json(path)
            if summary.get("status") != "finished" or summary.get("requests") != 60:
                missing.append(dict(job=item["id"], reason="not_finished_at_full_60_request_budget",
                                    status=summary.get("status"), requests=summary.get("requests")))
                continue
            ledger = folder / "candidates.jsonl"
            if not ledger.exists():
                missing.append(dict(job=item["id"], reason="candidate_ledger_missing"))
                continue
            rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
            if len(rows) != 60:
                raise ValueError("Full-budget summary disagrees with ledger: " + item["id"])
            if not (folder / "frozen_estimator.py").is_file():
                raise ValueError("Selected estimator is missing: " + item["id"])
            score = summary.get("selected_validation_auc_percent")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError("Invalid selected validation AUROC: " + item["id"])
            observations.append(dict(job=item["id"], policy=policy, seed=seed,
                selected_validation_auc_percent=score, summary_path=str(path.relative_to(root)),
                summary_sha256=digest, ledger_path=str(ledger.relative_to(root)), ledger_sha256=sha(ledger),
                config=item["config"], config_sha256=item["config_sha256"]))
    selected, averages = None, None
    if not missing:
        averages = {policy: mean(o["selected_validation_auc_percent"] for o in observations if o["policy"] == policy)
                    for policy in POLICIES}
        # max retains the first item on exact ties: the predeclared order above.
        selected = max(NONORIGINAL, key=lambda policy: averages[policy])
    return dict(ready=not missing, missing=missing, observations=observations,
                mean_selected_validation_auc_percent=averages, selected_policy=selected,
                selection_rule=RULE, held_out_test_scores_read=False)


def config_bytes(policy, seed):
    config = dict(defaults=["/config_uq_trial", {"override /model": "qwen38_27b"}, "_self_"],
                  prefix="research20261021", run_identifier_name="next27b_evo_" + policy,
                  seed=seed, search_mode="evolution", num_rounds=30, num_outputs_per_prompt=2,
                  model=dict(max_tokens=4096),
                  trial=dict(prompt_policy=policy, save_train_predictions=True, shortlist_size=3,
                             evaluation_workers=1, max_seconds=5400, max_total_tokens=2000000,
                             max_consecutive_failed_rounds=31))
    return ("# @package _global_\n" + OmegaConf.to_yaml(OmegaConf.create(config))).encode()


def verify_pins(source, source_path, selection, root):
    readiness = json.loads((source_path.parent / "readiness.json").read_text())
    if not readiness.get("cpu_gate_passed") or not readiness.get("all_ten_supplied_csvs_identical"):
        raise ValueError("Discovery readiness audit must pass")
    pinned = {**source["study_source_sha256"],
              source["training_raw_path"]: source["training_raw_sha256"],
              "data/claims/popqa/popqa_train_filtered_10shot.csv": source["training_csv_sha256"]}
    for name, expected in pinned.items():
        if sha(root / name) != expected:
            raise ValueError("Audited source/input changed: " + name)
    for row in selection["observations"]:
        if sha(root / "configs" / (row["config"] + ".yaml")) != row["config_sha256"]:
            raise ValueError("Discovery job config changed: " + row["job"])
        for kind in ("summary", "ledger"):
            if sha(root / row[kind + "_path"]) != row[kind + "_sha256"]:
                raise ValueError("Discovery artifact changed during preparation: " + row["job"])
    model = OmegaConf.load(root / "configs/model/qwen38_27b.yaml")
    if model.hf_model_id != WRITER or model.revision != REVISION:
        raise ValueError("27B model config differs from the predeclared checkpoint")
    for key, expected in dict(temperature=.7, topk=20, topp=.8, repetition_penalty=1.).items():
        if model[key] != expected:
            raise ValueError("27B model sampling parameter differs from discovery: " + key)


def prepare(source, source_path, source_sha, selection, output, root=ROOT):
    if not selection["ready"]:
        raise ValueError("Refusing confirmation preparation before all 12 full-budget 9B evolution runs finish")
    if output.parent.resolve() != source_path.parent.resolve():
        raise ValueError("Confirmation manifest must remain beside the discovery readiness.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            raise FileExistsError("Confirmation manifest already exists; refusing to overwrite or reselect: " + str(output))
        verify_pins(source, source_path, selection, root)
        if sha(source_path) != source_sha:
            raise ValueError("Discovery manifest changed during preparation")
        jobs, configs = [], []
        for seed in CONFIRMATION_SEEDS:
            for policy in ("original", selection["selected_policy"]):
                identifier = f"next27b_evo_{policy}_{seed}"
                config_name = "next_experiments/" + identifier
                path = root / "configs" / (config_name + ".yaml")
                data = config_bytes(policy, seed)
                if path.exists() and path.read_bytes() != data:
                    raise FileExistsError("Existing confirmation config differs; refusing to replace: " + str(path))
                configs.append((path, data))
                jobs.append(dict(id=identifier, phase="evolution_screen", study_phase="writer_confirmation",
                    mode="evolution", prompt_policy=policy, seed=seed, requests=60, config=config_name,
                    command=[".venv/bin/python", "src/experiments/main.py", "--config-name", config_name],
                    config_sha256=hashlib.sha256(data).hexdigest(),
                    expected_logs=f"out/logs/research20261021/uncertainty_quantification_{identifier}"))
        source_pins = dict(source["study_source_sha256"])
        for name in ("configs/model/qwen38_27b.yaml", "scripts/overnight_prepare_confirmation.py"):
            source_pins[name] = sha(root / name)
        manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            status="prepared_only_no_model_calls", study_phase="writer_confirmation",
            feature_model=source["feature_model"], labels=source["labels"], writer=WRITER,
            writer_revision=REVISION, seeds=list(CONFIRMATION_SEEDS), jobs=jobs,
            training_csv_sha256=source["training_csv_sha256"], training_raw_path=source["training_raw_path"],
            training_raw_sha256=source["training_raw_sha256"], study_source_sha256=source_pins,
            requests_by_phase={"evolution_screen": 240},
            selection_provenance=dict(source_manifest=str(source_path.relative_to(root)),
                source_manifest_sha256=source_sha, **selection),
            interpretation=("Conditional confirmation of a validation-selected prompt policy on a second writer. "
                "The selected policy can underperform original. No improvement is guaranteed. "
                "These are new writer runs using predeclared seeds, not an independent data holdout."))
        for path, data in configs:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
        data = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        with output.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "results/next_experiments/run_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/next_experiments/confirmation27b_manifest.json")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    source_path, output = args.source_manifest.resolve(), args.output.resolve()
    source, source_sha = load_json(source_path)
    selection = inspect_discovery(source)
    plan = dict(status="ready_to_prepare" if selection["ready"] else "waiting_for_discovery",
                writer=WRITER, writer_revision=REVISION, confirmation_seeds=list(CONFIRMATION_SEEDS),
                planned_jobs=4, planned_requests=240, output=str(output), **selection)
    print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
    if not args.prepare:
        print("DRY RUN ONLY. No files changed. Add --prepare only after all 12 full-budget 9B evolution runs finish.")
        return
    manifest = prepare(source, source_path, source_sha, selection, output)
    print(f"Prepared immutable manifest: {output}; {len(manifest['jobs'])} jobs / 240 requests")
    print("Run only after externally starting the pinned 27B writer server(s).")
    print(".venv/bin/python scripts/run_overnight_parallel.py --execute --phase all "
          "--manifest results/next_experiments/confirmation27b_manifest.json "
          "--state results/next_experiments/confirmation27b_execution_state.json --ports 8000 8001")


if __name__ == "__main__":
    main()
