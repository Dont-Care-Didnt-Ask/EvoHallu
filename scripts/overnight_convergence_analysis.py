"""Describe temporal diversity using the fixed 512-row training probe only.

Reads saved scores and metadata; executes no candidate code or model. Validation
and test outcomes never enter novelty or training-usefulness diagnostics. Partial
runs remain visible and cannot enter the primary three-seed averages.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
PROBE_SHA256 = "09fe41844ce427962f3ad5ae1d52771684c5f9b8ce9374d9e4dbee591b42e7dc"
THRESHOLD = .995
PRIMARY_SEEDS = (2107, 4319, 8741)
WINDOWS = ((0, 9, "1-10"), (10, 19, "11-20"), (20, 29, "21-30"))


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.hashes = {}

    def bytes(self, path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        key = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        if expected is not None and digest != expected:
            raise ValueError("Pinned input hash differs: " + key)
        if key in self.hashes and self.hashes[key] != digest:
            raise ValueError("Input changed while this diagnostic snapshot was being read: " + key)
        self.hashes[key] = digest
        return data

    def json(self, path, expected=None):
        return json.loads(self.bytes(path, expected))

    def scores(self, path, rows):
        result = np.load(io.BytesIO(self.bytes(path)), allow_pickle=False)
        if result.shape != (rows,) or not np.isfinite(result).all():
            raise ValueError("Saved training scores must have the expected finite vector shape: " + str(path))
        return result


def normalized_ranks(values):
    ranks = rankdata(values).astype(np.float64)
    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    return ranks / norm if norm else None


def describe_vector(values, baseline, previous):
    vector = normalized_ranks(values)
    if vector is None:
        return dict(probe_constant=True, baseline_rho=None, max_previous_rho=None,
                    max_previous_abs_rho=None, near_baseline=False, near_previous=False,
                    abs_near_baseline=False, abs_near_previous=False, novel=False, abs_novel=False), None
    rho = float(np.clip(vector @ baseline, -1, 1))
    correlations = [float(np.clip(vector @ old, -1, 1)) for old in previous]
    nearest = max(correlations) if correlations else None
    absolute = max(map(abs, correlations)) if correlations else None
    near_baseline = rho >= THRESHOLD
    near_previous = nearest is not None and nearest >= THRESHOLD
    abs_baseline = abs(rho) >= THRESHOLD
    abs_previous = absolute is not None and absolute >= THRESHOLD
    return dict(probe_constant=False, baseline_rho=rho, max_previous_rho=nearest,
                max_previous_abs_rho=absolute, near_baseline=near_baseline, near_previous=near_previous,
                abs_near_baseline=abs_baseline, abs_near_previous=abs_previous,
                novel=not (near_baseline or near_previous), abs_novel=not (abs_baseline or abs_previous)), vector


def validate_probe(probe):
    positions = np.asarray(probe["ordered_training_prediction_positions"])
    if (probe.get("validation_used") is not False or probe.get("labels") != "is_hallu"
            or probe.get("probe_rows") != 512 or positions.shape != (512,)
            or not np.issubdtype(positions.dtype, np.integer) or len(np.unique(positions)) != 512
            or np.any(positions < 0) or np.any(positions >= probe["train_rows"])):
        raise ValueError("Expected the pinned 512 distinct training-only is_hallu probe positions")
    return positions


def verify_run_config(job, config, probe):
    task, trial = config["task"], config["trial"]
    if (task.get("split_strategy", "stratified") != probe["split_strategy"]
            or task["split_seed"] != probe["split_seed"] or task.get("sample_size", 0) != 0
            or task["val_size"] != .2):
        raise ValueError("Run split differs from the fixed training probe: " + job["id"])
    if (config["search_mode"] != "evolution" or config["seed"] != job["seed"]
            or config["num_rounds"] != 30 or config["num_outputs_per_prompt"] != 2
            or trial.get("prompt_policy", "original") != job["prompt_policy"]
            or not trial.get("save_train_predictions") or task.get("supervised", False)):
        raise ValueError("Run does not match the declared adaptive 60-request protocol: " + job["id"])
    if job.get("writer_model", config["model"]["hf_model_id"]) != config["model"]["hf_model_id"]:
        raise ValueError("Run writer differs from the analysis job identity: " + job["id"])


def verify_execution_pins(job, manifest, inputs):
    if job.get("execution_manifest"):
        original = inputs.json(inputs.root / job["execution_manifest"], job["execution_manifest_sha256"])
        source = next(row for row in original["jobs"] if row["id"] == job["id"])
        for key in ("config", "config_sha256", "prompt_policy", "seed", "requests", "expected_logs", "mode"):
            if source[key] != job[key]:
                raise ValueError("Analysis identity differs from execution manifest: " + job["id"] + " / " + key)
    else:
        original = manifest
    for name, expected in original.get("study_source_sha256", {}).items():
        inputs.bytes(inputs.root / name, expected)
    inputs.bytes(inputs.root / "configs" / (job["config"] + ".yaml"), job["config_sha256"])
    return original


def window_statistics(rows, vectors, label, start, end):
    selected = [row for row in rows if start <= row["round"] <= end]
    valid = [row for row in selected if row["training_valid"]]
    nonconstant = [row for row in valid if not row["probe_constant"]]
    counts = dict(requests_observed=len(selected), training_valid=len(valid), nonconstant=len(nonconstant),
        probe_constant=sum(row["probe_constant"] for row in valid),
        missing_train_predictions=sum(row.get("diagnostic_error") == "missing_train_predictions" for row in selected))
    keys = ("near_baseline", "near_previous", "abs_near_baseline", "abs_near_previous", "novel", "abs_novel",
            "training_useful", "novel_and_training_useful", "abs_novel_and_training_useful")
    for key in keys:
        source = valid if key == "training_useful" else nonconstant
        counts[key] = sum(bool(row.get(key)) for row in source)
    siblings = []
    for round_index in range(start, end + 1):
        pair = [vector for (r, _), vector in vectors.items() if r == round_index]
        if len(pair) == 2:
            siblings.append(float(np.clip(pair[0] @ pair[1], -1, 1)))
    counts.update(comparable_sibling_pairs=len(siblings), near_sibling_pairs=sum(r >= THRESHOLD for r in siblings),
                  abs_near_sibling_pairs=sum(abs(r) >= THRESHOLD for r in siblings))
    for key in ("near_baseline", "near_previous", "abs_near_baseline", "abs_near_previous", "novel", "abs_novel",
                "novel_and_training_useful", "abs_novel_and_training_useful"):
        counts[key + "_fraction_nonconstant"] = counts[key] / len(nonconstant) if nonconstant else None
    counts["training_valid_fraction_requests"] = len(valid) / len(selected) if selected else None
    return dict(window=label, **counts)


def analyse_run(job, manifest, probe, positions, inputs, warnings):
    folder = inputs.root / job["expected_logs"]
    identity = dict(job=job["id"], phase=job.get("analysis_phase", job["phase"]), policy=job["prompt_policy"], seed=job["seed"],
                    writer=job.get("writer_model", manifest["writer"]), parent_sampling=job.get("parent_sampling", "inherited"))
    if not (folder / "trial_config.json").exists():
        return dict(**identity, status="not_run", full_budget=False, observed_requests=0, windows=[], candidates=[])
    execution = verify_execution_pins(job, manifest, inputs)
    config = inputs.json(folder / "trial_config.json")
    verify_run_config(job, config, probe)
    if (config["model"]["hf_model_id"] != execution["writer"]
            or config["model"].get("revision") != execution["writer_revision"]):
        raise ValueError("Run writer checkpoint differs from its pinned execution manifest: " + job["id"])
    if config["task"]["train_raw_path"] != manifest["training_raw_path"]:
        raise ValueError("Run training-cache path differs from the audited study: " + job["id"])
    summary = inputs.json(folder / "summary.json") if (folder / "summary.json").exists() else {}
    if not (folder / "baseline_train_scores.npy").exists():
        return dict(**identity, status="initializing", full_budget=False, observed_requests=0, windows=[], candidates=[])
    baseline = normalized_ranks(inputs.scores(folder / "baseline_train_scores.npy", probe["train_rows"])[positions])
    if baseline is None:
        raise ValueError("The pinned NLL baseline is constant on the training probe")
    baseline_fitness = summary.get("baseline_train", {}).get("score")
    if (isinstance(baseline_fitness, bool) or not isinstance(baseline_fitness, (int, float))
            or not math.isfinite(baseline_fitness) or not 0 <= baseline_fitness <= 100):
        raise ValueError("Missing finite training baseline fitness: " + job["id"])
    records = []
    ledger = folder / "candidates.jsonl"
    if ledger.exists():
        lines = inputs.bytes(ledger).splitlines()
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1 or summary.get("status") == "finished":
                    raise
                warnings.append("Trailing in-progress ledger line excluded: " + job["id"])
    keys = [(row["round"], row["index"]) for row in records]
    if len(set(keys)) != len(keys) or any(not 0 <= r < 30 or index not in (0, 1) for r, index in keys):
        raise ValueError("Duplicate or out-of-budget round/index identity: " + job["id"])
    full = (summary.get("status") == "finished" and summary.get("requests") == 60 and len(records) == 60
            and set(keys) == {(r, i) for r in range(30) for i in (0, 1)})
    rows, previous, vectors = [], [], {}
    for record in sorted(records, key=lambda row: (row["round"], row["index"])):
        row = dict(round=record["round"], round_1based=record["round"]+1, index=record["index"],
                   code_hash=record.get("code_hash"), training_valid=False, training_useful=False)
        fitness = record.get("train", {}).get("score")
        if (isinstance(fitness, (int, float)) and not isinstance(fitness, bool)
                and math.isfinite(fitness) and 0 <= fitness <= 100):
            path = folder / f"train_scores_{record['code_hash']}.npy"
            if not path.exists():
                row["diagnostic_error"] = "missing_train_predictions"
            else:
                values = inputs.scores(path, probe["train_rows"])[positions]
                description, vector = describe_vector(values, baseline, previous)
                row.update(description, training_valid=True, training_fitness=fitness,
                           training_useful=fitness >= baseline_fitness)
                row["novel_and_training_useful"] = row["novel"] and row["training_useful"]
                row["abs_novel_and_training_useful"] = row["abs_novel"] and row["training_useful"]
                if vector is not None:
                    previous.append(vector)
                    vectors[(row["round"], row["index"])] = vector
        rows.append(row)
    windows = [window_statistics(rows, vectors, label, start, end) for start, end, label in WINDOWS]
    identity["parent_sampling"] = config["trial"].get("parent_sampling", identity["parent_sampling"])
    diagnostic_complete = full and all(row.get("diagnostic_error") != "missing_train_predictions" for row in rows)
    return dict(**identity, writer_revision=config["model"].get("revision"),
        status="full_budget" if diagnostic_complete else "full_budget_missing_predictions" if full else "partial",
        full_budget=full, diagnostic_complete=diagnostic_complete, observed_requests=len(records),
        summary_status=summary.get("status"), baseline_training_fitness=baseline_fitness,
        windows=windows, candidates=rows)


def summarize_groups(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["writer"], run["phase"], run["policy"], run["parent_sampling"])].append(run)
    results = []
    for (writer, phase, policy, parent), members in sorted(groups.items()):
        finished = [run for run in members if run["full_budget"] and run.get("diagnostic_complete", False)]
        primary_ready = ({run["seed"] for run in members} == set(PRIMARY_SEEDS)
                         and {run["seed"] for run in finished} == set(PRIMARY_SEEDS) and len(members) == 3)
        row = dict(writer=writer, phase=phase, policy=policy, parent_sampling=parent,
                   planned_seeds=sorted(run["seed"] for run in members), complete_seeds=sorted(run["seed"] for run in finished),
                   primary_three_seed_ready=primary_ready, primary_windows=[])
        if primary_ready:
            for index, (_, _, label) in enumerate(WINDOWS):
                values = [run["windows"][index] for run in finished]
                metrics = {}
                for key in values[0]:
                    if key == "window":
                        continue
                    samples = [value[key] for value in values if value[key] is not None]
                    metrics[key] = dict(mean=float(np.mean(samples)) if samples else None,
                        std=float(np.std(samples, ddof=1)) if len(samples) > 1 else None, seeds_with_defined_metric=len(samples))
                row["primary_windows"].append(dict(window=label, metrics=metrics))
        results.append(row)
    return results


def csv_file(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/overnight20261021/analysis_manifest.json")
    parser.add_argument("--probe", type=Path, default=ROOT / "results/next_experiments/training_probe.json")
    parser.add_argument("--expected-probe-sha256", default=PROBE_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "results/overnight20261021/convergence_analysis")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    inputs = Inputs(ROOT)
    inputs.bytes(Path(__file__))
    manifest = inputs.json(args.manifest)
    probe = inputs.json(args.probe, args.expected_probe_sha256)
    positions = validate_probe(probe)
    inputs.bytes(ROOT / "data/claims/popqa/popqa_train_filtered_10shot.csv", probe["csv_sha256"])
    if manifest["training_csv_sha256"] != probe["csv_sha256"]:
        raise ValueError("Analysis manifest and training probe identify different CSVs")
    jobs = [job for job in manifest["jobs"] if job.get("mode") == "evolution" and job.get("requests") == 60]
    warnings = []
    runs = [analyse_run(job, manifest, probe, positions, inputs, warnings) for job in jobs]
    groups = summarize_groups(runs)
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), threshold=THRESHOLD,
        partition="Fixed 512-row training probe; no validation/test predictions or labels read",
        novelty_reference="NLL baseline and all earlier nonconstant training-valid candidates in the same run; earlier windows remain in the archive",
        candidate_order="Round then sibling index; the earlier-index sibling enters the previous-candidate archive, matching saved ledger order",
        training_useful="Exploratory: saved directed training fitness >= saved directed NLL training fitness; not held-out quality",
        validity="Training score is finite and its full-length finite prediction vector is available; validation success is not required",
        abs_correlation_supplement="Also treats reversed rankings as near-duplicates; signed rho is the prespecified primary diagnostic",
        primary_aggregation="Only arms with all three seeds 2107,4319,8741 at the full 60-request budget; partial and two-seed arms remain individual diagnostics",
        arm_separation="Writer, analysis phase, prompt policy, and parent sampling define separate groups; interface_control is never pooled with ordinary neutral evolution",
        interpretation_limit="The previous-candidate archive grows over time. A decline in novelty alone does not establish collapse; compare baseline and sibling similarity too.",
        warnings=warnings, input_sha256=inputs.hashes, runs=runs, groups=groups)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "convergence.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    window_rows, candidate_rows = [], []
    for run in runs:
        identity = {k: run[k] for k in ("job", "writer", "phase", "policy", "parent_sampling", "seed", "status", "full_budget")}
        window_rows.extend(dict(**identity, **window) for window in run["windows"])
        candidate_rows.extend(dict(**identity, **candidate) for candidate in run["candidates"])
    csv_file(args.output / "convergence_windows.csv", window_rows)
    csv_file(args.output / "convergence_candidates.csv", candidate_rows)
    lines = ["# Temporal evolution diversity", "", report["partition"] + ".", "",
        "Training-useful means training fitness at least the NLL baseline; this is exploratory and reuses the training objective.", "",
        "Primary averages require all three complete seeds. Partial runs are shown below and never enter those averages.", "",
        "| Run | Status | Rounds | Requests | Training-valid | Nonconstant | Near baseline | Near previous | Novel | Novel + training-useful | Abs-novel |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in window_rows:
        if row["requests_observed"]:
            lines.append("| " + " | ".join(str(row[key]) for key in ["job", "status", "window", "requests_observed", "training_valid", "nonconstant",
                "near_baseline", "near_previous", "novel", "novel_and_training_useful", "abs_novel"]) + " |")
    lines += ["", "## Complete three-seed averages", ""]
    ready = [group for group in groups if group["primary_three_seed_ready"]]
    if not ready:
        lines.append("No complete three-seed arm yet; no primary convergence conclusion is available.")
    for group in ready:
        lines.append(f"- {group['writer']} / {group['phase']} / {group['policy']} / {group['parent_sampling']}:")
        for window in group["primary_windows"]:
            metrics = window["metrics"]
            lines.append(f"  - Rounds {window['window']}: mean novel count {metrics['novel']['mean']:.2f}, mean novel + training-useful {metrics['novel_and_training_useful']['mean']:.2f}; all three seeds complete.")
    lines += ["", report["interpretation_limit"], "", "Hashes, absolute-correlation supplements, sibling similarity, and per-candidate diagnostics are preserved in JSON/CSV."]
    (args.output / "CONVERGENCE.md").write_text("\n".join(lines)+"\n")
    if args.plot and ready:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(9, 5))
        for group in ready:
            means = [w["metrics"]["novel_fraction_nonconstant"]["mean"] for w in group["primary_windows"]]
            axis.plot(range(3), means, marker="o", label=f"{group['phase']} / {group['policy']} / {group['parent_sampling']}")
        axis.set(xticks=range(3), xticklabels=[w[2] for w in WINDOWS], ylim=(0, 1),
                 xlabel="Evolution rounds", ylabel="Mean fraction novel among nonconstant training-valid candidates")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output / "convergence.png", dpi=180)
        plt.close(figure)
    print(json.dumps(dict(runs=len(runs), started=sum(run["status"] != "not_run" for run in runs),
        complete_runs=sum(run["full_budget"] for run in runs), complete_three_seed_groups=len(ready),
        output=str(args.output), model_calls=False, candidate_code_executed=False)))


if __name__ == "__main__":
    main()
