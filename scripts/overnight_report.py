"""Aggregate immutable overnight cohorts without selecting methods on test AUROC."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "original/SequenceLogProb"
NON_METHOD = {"labels", "duplicate_disjoint_mask", "claim_group_ids"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(values):
    ranks = rankdata(values).astype(float)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm else None


def distribution(values):
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                minimum=float(min(values)) if values else None,
                maximum=float(max(values)) if values else None)


def diagnostics(manifest, probe, warnings):
    """Use the same training-only probe and 0.995 rank threshold as the study summary."""
    positions = np.asarray(probe["ordered_training_prediction_positions"])
    rows = []
    for job in manifest["jobs"]:
        folder = ROOT / job["expected_logs"]
        row = {k: job[k] for k in ("id", "phase", "prompt_policy", "seed", "requests")}
        row["planned_requests"] = row.pop("requests")
        row["status"] = "not_run"
        path = folder / "summary.json"
        if not path.exists():
            rows.append(row)
            continue
        try:
            summary = json.loads(path.read_text())
            cfg = json.loads((folder / "trial_config.json").read_text())
            task = cfg["task"]
            if (task.get("split_strategy", "stratified") != probe["split_strategy"] or
                task["split_seed"] != probe["split_seed"] or task.get("sample_size", 0) != 0 or task["val_size"] != .2):
                raise ValueError("Run split does not match the fixed training probe")
            records = []
            for line in (folder / "candidates.jsonl").read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    warnings.append(f"Incomplete candidate record skipped: {job['id']}")
            valid = [r for r in records if "score" in r.get("val", {})]
            vectors, previous, unique, near, constant = defaultdict(list), [], set(), 0, 0
            baseline = folder / "baseline_train_scores.npy"
            if baseline.exists():
                previous.append(normalized(np.load(baseline)[positions]))
            missing = 0
            for r in valid:
                path = folder / f"train_scores_{r['code_hash']}.npy"
                if not path.exists():
                    missing += 1
                    continue
                values = np.load(path)
                if values.shape != (probe["train_rows"],):
                    raise ValueError("Saved train score length does not match probe")
                values = values[positions]
                vector = normalized(values)
                if vector is None:
                    constant += 1
                    continue
                fingerprint = hashlib.sha256(np.unique(values, return_inverse=True)[1].astype("<i4").tobytes()).hexdigest()
                unique.add(fingerprint)
                near += any(v is not None and float(vector @ v) >= .995 for v in previous)
                previous.append(vector)
                vectors[r["round"]].append(vector)
            correlations = [float(a @ b) for vs in vectors.values() for i, a in enumerate(vs) for b in vs[i+1:]]
            row.update(status=summary["status"], requests=summary["requests"],
                full_planned_budget=summary["requests"] == job["requests"], valid_candidates=len(valid),
                valid_fraction=len(valid)/len(records) if records else None,
                selected_validation_auc_percent=summary.get("selected_validation_auc_percent"),
                distinct_nonconstant_training_probe_rankings=len(unique), constant_probe_candidates=constant,
                near_previous_or_baseline_probe_candidates=int(near), comparable_sibling_pairs=len(correlations),
                near_identical_sibling_pairs=sum(r >= .995 for r in correlations),
                missing_train_predictions=missing, total_tokens=summary["total_tokens"],
                elapsed_seconds=summary["elapsed_seconds"])
            if missing:
                warnings.append(f"Missing {missing} training prediction arrays: {job['id']}")
        except (OSError, ValueError, KeyError) as exc:
            row.update(status="diagnostics_incomplete", diagnostic_error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def collect(directory, warnings):
    """Merge additive cohorts only when shared saved predictions agree exactly."""
    datasets, sources = {}, {}
    for path in sorted((directory / "evaluations").glob("*/*_scores.npz")):
        name = path.name.removesuffix("_scores.npz")
        z = dict(np.load(path))
        if name not in datasets:
            datasets[name] = {k: z[k] for k in NON_METHOD if k in z}
        target = datasets[name]
        for key, value in z.items():
            if key in target and not np.array_equal(target[key], value):
                raise ValueError(f"Conflicting saved predictions/row identity: {name}, {key}, {path}")
            target[key] = value
        sources[str(path.relative_to(ROOT))] = sha(path)
    if len(datasets) != 9:
        warnings.append(f"Only {len(datasets)}/9 test datasets have saved predictions")
    return datasets, sources


def load_bootstrap(directory):
    result = defaultdict(dict)
    for path in sorted((directory / "evaluations").glob("*/*_bootstrap.npz")):
        name = path.name.removesuffix("_bootstrap.npz")
        for method, values in dict(np.load(path)).items():
            old = result[name].get(method)
            if old is None or len(values) > len(old):
                result[name][method] = values
    return result


def interval(values):
    return np.nanpercentile(values, [2.5, 97.5]).tolist()


def matched_comparisons(primary):
    """Pair full-budget nine-dataset runs with matching writer/seed/request counts.

    Original evolution is a broad reference. Neutral evolution is the causal
    comparator for feedback, rank-parent, and interface/metric controls; those contrasts are
    separate, explicitly named groups. Fixed-context and second-writer prompt
    comparisons stay within their own phases.
    """
    eligible = [row for row in primary if row.get("identity") and
                row["identity"].get("role") == "primary" and row.get("status") == "complete" and
                row["identity"].get("full_budget") is True]
    lookup = {}
    for row in eligible:
        identity = row["identity"]
        key = (identity["phase"], identity["policy"], identity["writer_model"],
               identity["seed"], identity["requests"])
        if key in lookup:
            raise ValueError("Ambiguous matched reference/run identity: " + str(key))
        lookup[key] = row
    specifications = [
        ("original_evolution_reference", {"evolution_screen", "feedback_control", "sampler_control", "interface_control"},
         "evolution_screen", "original"),
        ("fixed_examples_vs_neutral_evolution", {"feedback_control"}, "evolution_screen", "neutral"),
        ("rank_sampling_vs_neutral_evolution", {"sampler_control"}, "evolution_screen", "neutral"),
        ("interface_metric_contract_vs_neutral_evolution", {"interface_control"}, "evolution_screen", "neutral"),
        ("fixed_context_prompt_reference", {"prompt_screen"}, "prompt_screen", "original"),
        ("writer_confirmation_prompt_reference", {"writer_confirmation"}, "writer_confirmation", "original"),
        ("writer_generalization_prompt_reference", {"writer_generalization"}, "writer_generalization", "original"),
    ]
    grouped = defaultdict(list)
    for comparison, phases, reference_phase, reference_policy in specifications:
        for row in eligible:
            identity = row["identity"]
            if identity["phase"] not in phases:
                continue
            reference = lookup.get((reference_phase, reference_policy, identity["writer_model"],
                                    identity["seed"], identity["requests"]))
            if reference is None or reference is row:
                continue
            delta = dict(seed=identity["seed"], method=row["method"], reference=reference["method"],
                requests=identity["requests"], reference_requests=reference["identity"]["requests"],
                equal_requests=True, both_full_planned_budgets=True,
                directed_gain_pp=row["directed_macro_auc_percent"]-reference["directed_macro_auc_percent"],
                paper_gain_pp=row["paper_macro_auc_percent"]-reference["paper_macro_auc_percent"])
            grouped[(comparison, identity["phase"], identity["policy"], identity["writer_model"],
                     reference_phase, reference_policy)].append(delta)
    results = []
    for (comparison, phase, policy, writer, reference_phase, reference_policy), values in sorted(grouped.items()):
        values.sort(key=lambda row: (row["seed"], row["method"]))
        results.append(dict(comparison=comparison, phase=phase, policy=policy, writer_model=writer,
            reference_phase=reference_phase, reference_policy=reference_policy, per_seed=values,
            directed_gain_pp=distribution([row["directed_gain_pp"] for row in values]),
            paper_gain_pp=distribution([row["paper_gain_pp"] for row in values])))
    return results


def all_seed_primary_groups(primary):
    """Keep partial/unverified request budgets out of primary seed summaries."""
    grouped = defaultdict(list)
    for row in primary:
        identity = row.get("identity")
        if not identity or identity.get("role") != "primary" or identity.get("full_budget") is not True:
            continue
        grouped[(identity["phase"], identity["policy"], identity["writer_model"])].append(row)
    groups = []
    for (phase, policy, writer), values in sorted(grouped.items()):
        complete = [row for row in values if row["status"] == "complete"]
        groups.append(dict(phase=phase, policy=policy, writer_model=writer,
            staged_seeds=[row["identity"]["seed"] for row in values],
            evaluated_seeds=[row["identity"]["seed"] for row in complete],
            directed_macro=distribution([row["directed_macro_auc_percent"] for row in complete]),
            paper_macro=distribution([row["paper_macro_auc_percent"] for row in complete]),
            selected_validation=distribution([row["identity"]["validation_auc_percent"] for row in values])))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, default=ROOT / "results/overnight20261021")
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/next_experiments/run_manifest.json")
    ap.add_argument("--probe", type=Path, default=ROOT / "results/next_experiments/training_probe.json")
    ap.add_argument("--historical", type=Path, default=ROOT / "out/campaign/evaluation_with_4b")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/claims")
    args = ap.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    warnings = []
    frozen_path = args.directory / "frozen_manifest.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {"methods": []}
    identities = {row["method"]: row for row in frozen["methods"]}
    manifest = json.loads(args.manifest.read_text())
    probe = json.loads(args.probe.read_text())
    runs = diagnostics(manifest, probe, warnings)
    datasets, source_hashes = collect(args.directory, warnings)
    boot = load_bootstrap(args.directory)
    methods = sorted(set(identities) | {m for z in datasets.values() for m in z if m not in NON_METHOD})
    results = []
    for method in methods:
        row = dict(method=method, identity=identities.get(method), per_dataset=[])
        identity = row["identity"]
        if identity and identity.get("role") == "primary":
            row["interpretation_role"] = ("full_budget_primary" if identity.get("full_budget") is True
                                          else "partial_or_unverified_budget_exploratory")
        else:
            row["interpretation_role"] = "secondary_selection_diagnostic" if identity else "fixed_fitted_or_reference_method"
        for name, z in sorted(datasets.items()):
            if method not in z:
                continue
            raw = 100 * float(roc_auc_score(z["labels"], z[method]))
            item = dict(dataset=name, rows=len(z["labels"]), directed_auc_percent=raw,
                        paper_auc_percent=max(raw, 100-raw), direction_flipped=raw < 50)
            mask = z.get("duplicate_disjoint_mask")
            if mask is not None and len(np.unique(z["labels"][mask])) == 2:
                strict = 100 * float(roc_auc_score(z["labels"][mask], z[method][mask]))
                item.update(duplicate_disjoint_rows=int(mask.sum()), duplicate_disjoint_directed_auc_percent=strict,
                            duplicate_disjoint_paper_auc_percent=max(strict, 100-strict))
            row["per_dataset"].append(item)
        row["datasets_completed"] = len(row["per_dataset"])
        row["status"] = "complete" if row["datasets_completed"] == 9 else "pending_or_failed"
        if row["status"] == "complete":
            row["directed_macro_auc_percent"] = float(np.mean([d["directed_auc_percent"] for d in row["per_dataset"]]))
            row["paper_macro_auc_percent"] = float(np.mean([d["paper_auc_percent"] for d in row["per_dataset"]]))
            if all("duplicate_disjoint_directed_auc_percent" in d for d in row["per_dataset"]):
                row["duplicate_disjoint_directed_macro_auc_percent"] = float(np.mean([d["duplicate_disjoint_directed_auc_percent"] for d in row["per_dataset"]]))
                row["duplicate_disjoint_paper_macro_auc_percent"] = float(np.mean([d["duplicate_disjoint_paper_auc_percent"] for d in row["per_dataset"]]))
            names = [d["dataset"] for d in row["per_dataset"]]
            if all(method in boot[n] and BASELINE in boot[n] and len(boot[n][method]) for n in names):
                count = min(min(len(boot[n][method]), len(boot[n][BASELINE])) for n in names)
                b = np.asarray([boot[n][method][:count] for n in names]) * 100
                reference = np.asarray([boot[n][BASELINE][:count] for n in names]) * 100
                raw_macro, reference_raw = b.mean(axis=0), reference.mean(axis=0)
                paper_macro = np.maximum(b, 100-b).mean(axis=0)
                reference_paper = np.maximum(reference, 100-reference).mean(axis=0)
                row["paired_claim_bootstrap"] = dict(samples=count,
                    directed_macro_ci95_percent=interval(raw_macro),
                    directed_gain_vs_nll_ci95_pp=interval(raw_macro-reference_raw),
                    paper_macro_ci95_percent=interval(paper_macro),
                    paper_gain_vs_nll_ci95_pp=interval(paper_macro-reference_paper))
        results.append(row)
    primary = [r for r in results if r["identity"] and r["identity"]["role"] == "primary"]
    groups = all_seed_primary_groups(primary)
    partial = [row for row in primary if row["identity"].get("full_budget") is not True]
    matched = matched_comparisons(primary)
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        labels="is_hallu", reporting="max(AUC,100-AUC) within each dataset, then unweighted mean of nine",
        selection="Frozen directed-PopQA-validation winner from every run; optional prespecified validation ranks 2–3",
        test_labels_used_for_selection=False, best_test_seed_selected=False,
        bootstrap="Paired normalized-claim cluster resamples; folds reapplied per dataset in each bootstrap replicate",
        confidence_limit="Intervals condition on each frozen program; they do not include variability of writer training/search",
        paper_comparison="Paper references are rounded context, not matched significance tests: NLL 68.7, GPT 70.8, Claude 73.4",
        source_prediction_sha256=source_hashes,
        evaluation_manifest_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((args.directory / "evaluations").glob("*/manifest.json"))},
        warnings=warnings,
        requests_completed=sum(r.get("requests", 0) for r in runs), planned_requests=sum(j["requests"] for j in manifest["jobs"]),
        jobs_finished=sum(r["status"] == "finished" for r in runs), jobs_planned=len(runs),
        runs=runs, methods=results, all_seed_primary_groups=groups, matched_seed_comparisons=matched,
        primary_group_budget_rule="Only identity.full_budget=true; partial/unverified identities remain individual exploratory records",
        excluded_partial_primary_identities=[dict(method=row["method"], identity=row["identity"]) for row in partial])
    baseline = next((r for r in results if r["method"] == BASELINE and r["status"] == "complete"), None)
    report["baseline_check"] = None if baseline is None else dict(
        historical_paper_nll_auc_percent=68.7,
        current_directed_macro_auc_percent=baseline["directed_macro_auc_percent"],
        current_paper_macro_auc_percent=baseline["paper_macro_auc_percent"],
        matches_historical_paper_rounding=round(baseline["paper_macro_auc_percent"], 1) == 68.7,
        note="Expected only for the unchanged historical Llama3.1-8b-instruct feature caches and dataset assembly")
    from overnight_report_baseline import compare
    report["baseline_exact_check"] = compare(datasets, args.historical, args.data_root)
    (args.directory / "baseline_exact_check.json").write_text(json.dumps(report["baseline_exact_check"], indent=2, allow_nan=False)+"\n")
    (args.directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (args.directory / "methods.csv").open("w") as stream:
        columns = ["method", "status", "interpretation_role", "datasets_completed", "directed_macro_auc_percent", "paper_macro_auc_percent"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in columns} for r in results)
    with (args.directory / "per_dataset.csv").open("w") as stream:
        columns = ["method", "dataset", "rows", "directed_auc_percent", "paper_auc_percent", "direction_flipped"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"method": r["method"], **{k: d.get(k) for k in columns if k != "method"}}
                         for r in results for d in r["per_dataset"])
    lines = ["# EvoHallu overnight experiments", "", f"Completed {report['jobs_finished']}/{report['jobs_planned']} jobs; {report['requests_completed']}/{report['planned_requests']} requests.", "",
        "Labels are `is_hallu`. Paper-compatible reporting folds AUROC within each dataset before averaging. Selection uses directed PopQA validation only; every seed is retained.", "",
        "## Frozen methods", "", "| Method | Interpretation | Datasets | Directed AUROC | Paper-compatible AUROC |", "| --- | --- | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['method']} | {r['interpretation_role']} | {r['datasets_completed']}/9 | {r.get('directed_macro_auc_percent', 'pending')} | {r.get('paper_macro_auc_percent', 'pending')} |")
    exact = report["baseline_exact_check"]
    lines += ["", "## Baseline reproducibility", "", f"Exact historical comparison: **{exact['status']}**, {exact['datasets_checked']}/9 datasets checked. Paper-compatible NLL AUROC: {exact['paper_macro_auc_percent']}. Labels remain `is_hallu`.", "",
              exact["row_identity_evidence"], "", "## All-seed primary results", "", "Only full-budget primary identities enter these groups. Standard deviations below describe variation across the completed seeds; they are not confidence intervals.", "",
              "| Phase / policy | Writer | Evaluated seeds | Paper AUROC mean | Seed SD |", "| --- | --- | --- | ---: | ---: |"]
    for g in groups:
        lines.append(f"| {g['phase']} / {g['policy']} | {g['writer_model']} | {g['evaluated_seeds']} | {g['paper_macro']['mean']} | {g['paper_macro']['std']} |")
    lines += ["", "## Matched comparisons", "",
              "Only completed nine-dataset primary methods with full planned budgets are paired. Writers, seeds and request counts must match. Positive deltas favor the named arm over its reference.", "",
              "| Comparison | Arm | Reference | Matched seeds | Paper AUROC delta, pp |", "| --- | --- | --- | --- | ---: |"]
    for comparison in matched:
        seeds = [row["seed"] for row in comparison["per_seed"]]
        lines.append(f"| {comparison['comparison']} | {comparison['phase']} / {comparison['policy']} | {comparison['reference_phase']} / {comparison['reference_policy']} | {seeds} | {comparison['paper_gain_pp']['mean']} |")
    lines += ["", "## Training-probe diagnostics", "", "The fixed 512-row training probe is separate from validation. Rank similarity ≥0.995 is the prespecified near-duplicate diagnostic.", "",
              "| Run | Status | Requests | Valid | Distinct rankings | Near-identical sibling pairs / comparable | Validation AUROC |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in runs:
        sibling = f"{r.get('near_identical_sibling_pairs', '—')}/{r.get('comparable_sibling_pairs', '—')}"
        lines.append(f"| {r['id']} | {r['status']} | {r.get('requests', '—')} | {r.get('valid_candidates', '—')} | {r.get('distinct_nonconstant_training_probe_rankings', '—')} | {sibling} | {r.get('selected_validation_auc_percent', '—')} |")
    lines += ["", "## Interpretation limits", "", report["paper_comparison"], "", report["confidence_limit"], "",
              "Optional validation shortlist entries are secondary diagnostics; no test-best program or test-best seed is promoted to the primary result."]
    if warnings:
        lines += ["", "## Pending data and warnings", ""] + ["- " + w for w in warnings]
    (args.directory / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(jobs_finished=report["jobs_finished"], requests_completed=report["requests_completed"],
                          methods_complete=sum(r["status"] == "complete" for r in results), warnings=len(warnings))))


if __name__ == "__main__":
    main()
