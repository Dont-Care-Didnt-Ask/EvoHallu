"""Prepare fixed-budget studies and a validation-only inspection shortlist.

No model requests, candidate execution, or benchmark-based candidate selection.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2107, 4319, 8741]
SOURCE_RUN = "uncertainty_quantification_qwen35_9b_evo_1048"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rank_vector(values):
    ranks = rankdata(values).astype(float)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "results/next_experiments")
    args = ap.parse_args()
    output = args.output.resolve()
    source = ROOT / "out/logs/mod1" / SOURCE_RUN
    # Uniformly spaced round positions; selection does not inspect outcomes.
    rounds = np.linspace(0, 29, 15, dtype=int).tolist()
    contexts = []
    for round_index in rounds:
        path = source / f"prompt_{round_index:04d}.json"
        contexts.append(dict(source=str(path.relative_to(ROOT)), source_round_1based=round_index+1,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(), chat=json.loads(path.read_text())))
    context_path = output / "frozen_contexts.json"
    write(context_path, dict(selection_rule="15 uniformly spaced saved round positions, independent of candidate scores",
                            source_run=SOURCE_RUN, adaptive_population_feedback=False, contexts=contexts))
    context_hash = hashlib.sha256(context_path.read_bytes()).hexdigest()

    csv = ROOT / "data/claims/popqa/popqa_train_filtered_10shot.csv"
    labels = pd.read_csv(csv, usecols=["is_hallu"]).is_hallu.to_numpy()
    train, val = train_test_split(np.arange(len(labels)), test_size=.2, stratify=labels, random_state=441)
    probe_positions, _ = train_test_split(np.arange(len(train)), train_size=512,
                                        stratify=labels[train], random_state=1901)
    write(output / "training_probe.json", dict(split_strategy="stratified", split_seed=441,
        csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(), labels="is_hallu",
        train_rows=len(train), validation_rows=len(val), probe_rows=512,
        ordered_training_prediction_positions=probe_positions.tolist(), raw_row_indices=train[probe_positions].tolist(),
        validation_used=False, note="Diagnostic probe for new saved train_scores arrays; not an independent confirmation set"))

    jobs = []
    variants = [("prompt_screen", "prompt_"+p, "frozen_context", p, SEEDS[0], 15) for p in ["original", "neutral", "diverse", "compositional"]]
    variants += [("evolution_screen", "evo_"+p, "evolution", p, seed, 30)
                 for seed in SEEDS for p in ["original", "neutral", "diverse"]]
    variants += [("compositional_control", "evo_compositional", "evolution", "compositional", seed, 30) for seed in SEEDS]
    variants += [("feedback_control", "fixed_neutral", "baseline_seeded", "neutral", seed, 30) for seed in SEEDS]
    for phase, name, mode, policy, seed, n_rounds in variants:
        identifier = f"next_{name}_{seed}"
        config_name = f"next_experiments/{identifier}"
        path = ROOT / "configs" / (config_name + ".yaml")
        config = dict(defaults=["/config_uq_trial", {"override /model": "qwen35_9b"}, "_self_"],
                      prefix="research20261021", run_identifier_name="next_"+name, seed=seed,
                      search_mode=mode, num_rounds=n_rounds, num_outputs_per_prompt=2,
                      trial=dict(prompt_policy=policy, save_train_predictions=True, shortlist_size=3,
                                 evaluation_workers=1, max_seconds=5400, max_total_tokens=2000000,
                                 max_consecutive_failed_rounds=n_rounds+1))
        if mode == "frozen_context":
            config["trial"].update(frozen_contexts_path=str(context_path.relative_to(ROOT)) if context_path.is_relative_to(ROOT) else str(context_path),
                                    frozen_contexts_sha256=context_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# @package _global_\n" + OmegaConf.to_yaml(OmegaConf.create(config)))
        command = [".venv/bin/python", "src/experiments/main.py", "--config-name", config_name]
        jobs.append(dict(id=identifier, phase=phase, mode=mode, prompt_policy=policy, seed=seed,
                         requests=n_rounds*2, config=config_name, command=command,
                         config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         expected_logs=f"out/logs/research20261021/uncertainty_quantification_next_{name}_{seed}"))

    # Export both ordinary top-three and a behaviourally spread top-three for inspection.
    # Validation is allowed for this final inspection; no test-score files are read here.
    inspected = []
    for folder in sorted((ROOT / "out/logs/mod1").glob("*_evo_*")):
        if (folder / "exclude_from_evaluation.json").exists() or not (folder / "candidates.jsonl").exists():
            continue
        cfg = json.loads((folder / "trial_config.json").read_text())
        if cfg.get("search_mode") != "evolution":
            continue
        rows = [json.loads(s) for s in (folder / "candidates.jsonl").read_text().splitlines()]
        ranked = sorted([r for r in rows if "score" in r.get("val", {})], key=lambda r:-r["val"]["score"])
        diverse, vectors = [], []
        for row in ranked:
            vector = rank_vector(np.load(folder / f"val_scores_{row['code_hash']}.npy"))
            if vector is None:
                continue
            if all(float(vector @ old) < .995 for old in vectors):
                diverse.append(row)
                vectors.append(vector)
            if len(diverse) == 3:
                break
        selected = {r["code_hash"] for r in diverse}
        for rank, row in enumerate(ranked, 1):
            if rank > 3 and row["code_hash"] not in selected:
                continue
            path = output / "shortlist" / folder.name / (row["code_hash"][:12] + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            text = row.get("imports", "") + "\n" + row["code"] + "\n"
            path.write_text(text)
            inspected.append(dict(run=folder.name, writer=cfg["model"]["hf_model_id"],
                validation_rank=rank, spread_shortlist=row["code_hash"] in selected,
                round_1based=row["round"]+1, index=row["index"], code_hash=row["code_hash"],
                train_auc=row["train"]["score"], validation_auc=row["val"]["score"],
                source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                frozen_source_sha256=hashlib.sha256(text.encode()).hexdigest()))
    write(output / "candidate_shortlist.json", dict(selection_used="saved PopQA validation only; no test scores",
        ordinary_top_k_per_run=3, spread_rule="greedy validation order; positive rank correlation <0.995 to earlier retained entries; exclude constants",
        primary_winners_unchanged=True, candidates=inspected))
    table = ["# Saved candidates for inspection", "", "These are inspection shortlists selected using historical validation. No new test winner has been selected.", "",
             "| Run | Val rank | Round / index | Train AUROC | Val AUROC | Spread shortlist | Code |",
             "| --- | ---: | --- | ---: | ---: | --- | --- |"]
    for row in inspected:
        path = ROOT / row["source"] if not Path(row["source"]).is_absolute() else Path(row["source"])
        link = path.relative_to(output)
        table.append(f"| {row['run'].removeprefix('uncertainty_quantification_')} | {row['validation_rank']} | {row['round_1based']} / {row['index']} | {row['train_auc']:.4f} | {row['validation_auc']:.4f} | {row['spread_shortlist']} | [view]({link}) |")
    (output / "CANDIDATE_SHORTLIST.md").write_text("\n".join(table)+"\n")
    readiness_path = output / "readiness.json"
    readiness = json.loads(readiness_path.read_text()) if readiness_path.exists() else {}
    manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="prepared_only_no_model_calls", feature_model="Llama3.1-8b-instruct", labels="is_hallu",
        writer="Qwen/Qwen3.5-9B", writer_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        seeds=SEEDS, contexts_sha256=context_hash, jobs=jobs,
        training_csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        training_raw_path="data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats_logits.pt",
        training_raw_sha256=readiness.get("training_cache", {}).get("raw_sha256"),
        study_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            "src/experiments/extension_trial.py", "src/experiments/prompt_policies.py",
            "src/packing/funsearch/programs_database.py", "src/packing/model/prompt.py",
            "src/packing/evaluate/uncertainty_quantification/task_uncertainty_quantification.py",
            "src/packing/evaluate/uncertainty_quantification/features.py",
            "configs/config_uq_trial.yaml", "configs/config_uq.yaml", "configs/model/qwen35_9b.yaml",
            "configs/task/uncertainty_quantification.yaml"]},
        requests_by_phase={phase:sum(j["requests"] for j in jobs if j["phase"]==phase)
                           for phase in ["prompt_screen", "evolution_screen", "feedback_control", "compositional_control"]},
        later_design_only=["immutable wrapper + smoke/repair", "behaviour-aware archive + rank parent sampling",
                           "two-stage sketch/implementation", "90-round vs three-restart portfolios",
                           "supervised evolved features", "new feature-model extraction"])
    write(output / "run_manifest.json", manifest)
    print(json.dumps(dict(status=manifest["status"], jobs=len(jobs),
                          requests_by_phase=manifest["requests_by_phase"], shortlist_programs=len(inspected),
                          manifest=str(output / "run_manifest.json")), indent=2))


if __name__ == "__main__":
    main()
