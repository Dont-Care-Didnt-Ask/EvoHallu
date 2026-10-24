"""Build portable tables and research figures from saved campaign artifacts."""
from __future__ import annotations
import ast
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, rankdata


def read_records(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A live backup may contain an incomplete final append.
                continue
    return rows


def macro_results(directory):
    aggregate_file = directory / "all_datasets.json"
    aggregate = json.loads(aggregate_file.read_text()) if aggregate_file.exists() else {}
    macro = {}
    if len(aggregate) != 9:
        return aggregate, macro
    methods = set.intersection(*(set(data["methods"]) for data in aggregate.values()))
    bootstrap = {name: np.load(directory / (name + "_bootstrap.npz")) for name in aggregate
                 if (directory / (name + "_bootstrap.npz")).exists()}
    baseline_name = "original/SequenceLogProb"
    baseline = np.mean([d["methods"][baseline_name]["auc_percent"] for d in aggregate.values()])
    for method in sorted(methods):
        values = [d["methods"][method]["auc_percent"] for d in aggregate.values()]
        record = {"macro_auc_percent": float(np.mean(values)), "gain_pp": float(np.mean(values)-baseline),
                  "duplicate_disjoint_macro_auc_percent": float(np.mean([
                      d["methods"][method]["duplicate_disjoint_auc_percent"] for d in aggregate.values()]))}
        if len(bootstrap) == 9 and all(method in b.files for b in bootstrap.values()):
            b = np.stack([bootstrap[name][method] for name in aggregate]).mean(axis=0)
            ref = np.stack([bootstrap[name][baseline_name] for name in aggregate]).mean(axis=0)
            record["macro_auc_ci95_percent"] = (100*np.nanpercentile(b, [2.5, 97.5])).tolist()
            record["paired_macro_gain_ci95_pp"] = (100*np.nanpercentile(b-ref, [2.5, 97.5])).tolist()
        macro[method] = record
    for saved in bootstrap.values():
        saved.close()
    return aggregate, macro


def paired_macro_difference(left_dir, right_dir, left_data, right_data, left, right):
    """Pair saved draws only after confirming identical ordered resampling units."""
    if set(left_data) != set(right_data) or len(left_data) != 9:
        return None
    draws = []
    for name in left_data:
        if left_data[name].get("bootstrap_unit") != right_data[name].get("bootstrap_unit"):
            return None
        try:
            with np.load(left_dir / (name+"_scores.npz")) as a, np.load(right_dir / (name+"_scores.npz")) as b:
                if not all(np.array_equal(a[key], b[key]) for key in
                           ["labels", "claim_group_ids", "duplicate_disjoint_mask"]):
                    return None
            with np.load(left_dir / (name+"_bootstrap.npz")) as a, np.load(right_dir / (name+"_bootstrap.npz")) as b:
                if left not in a.files or right not in b.files or a[left].shape != b[right].shape:
                    return None
                draws.append(a[left]-b[right])
        except (FileNotFoundError, KeyError):
            return None
    return (100*np.nanpercentile(np.stack(draws).mean(axis=0), [2.5, 97.5])).tolist()


def method_label(name):
    if name == "original/SequenceLogProb":
        return "Full-template NLL baseline"
    if name == "alignment/ClaimOnlySequenceLogProb":
        return "Claim-only NLL baseline"
    if name.startswith("original/"):
        return "Best supplied formula (validation)"
    if name.startswith("supervised/"):
        label = "Logistic" if "logistic" in name else "QDA"
        count = 14 if "documented" in name else 17
        return f"{label} ({count} fixed features)"
    if "linear" in name:
        return "Coefficient search (17-feature pool)"
    if "qwen38" in name:
        return "Qwen3.8-27B writer"
    if "laguna" in name:
        return "Laguna XS 2.1 writer"
    if "qwen35_4b" in name:
        return "Qwen3.5-4B writer"
    if "qwen35" in name:
        return "Qwen3.5-9B writer"
    if "devstral" in name:
        return "Devstral Small 2 writer (published FP8)"
    return name


def select_primary_runs(runs):
    """Choose source-validation winners before considering test availability."""
    chosen = []
    for model in sorted({r["model"] for r in runs}):
        eligible = [r for r in runs if r["model"] == model and r["search_mode"] == "evolution"
                    and r["split"] == "stratified" and r["status"] == "finished" and not r.get("protocol_excluded", False) and "pilot" not in r["run"]]
        if eligible:
            chosen.append(max(eligible, key=lambda r:r["validation_auc_percent"])["run"])
    linear = [r for r in runs if r["search_mode"] == "fixed_feature_linear_evolution" and r["status"] == "finished"]
    if linear:
        chosen.append(max(linear, key=lambda r:r["validation_auc_percent"])["run"])
    return chosen


def dataset_label(name):
    group, stem = name.split("__", 1)
    if group == "geometry":
        return {"cities":"Cities", "common_claim_true_false":"CommonClaim",
                "companies_true_false":"Companies", "counterfact_true_false":"CounterFact"}.get(stem,stem)
    if group.startswith("UHEAD"):
        return "UHEAD"
    return {"popqa":"PopQA", "x-fact":"X-Fact", "WildHallucinations":"WildHall"}.get(group,group)


def main():
    output = Path("results")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs, trajectories = [], {}
    for summary_path in sorted(Path("out/logs/mod1").glob("*/summary.json")):
        directory = summary_path.parent
        summary = json.loads(summary_path.read_text())
        config_path = directory / "trial_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        records = read_records(directory / "candidates.jsonl")
        row = {"run": directory.name, "status": summary["status"],
               "model": config.get("model", {}).get("hf_model_id", "fixed feature pool"),
               "search_mode": config.get("search_mode", summary.get("generator", "unknown")),
               "seed": config.get("seed", summary.get("seed")),
               "split": config.get("task", {}).get("split_strategy", "stratified"),
               "serving_profile": config.get("trial", {}).get("serving_profile", "vllm_eager") if config else "none",
               "writer_weight_precision": config.get("model", {}).get("weight_precision", "bfloat16") if config else "none",
               "requested_rounds": config.get("num_rounds"),
               "requests": summary.get("requests", 0),
               "valid_candidates": summary.get("valid_candidates", summary.get("generated_candidates", 0)),
               "validation_auc_percent": summary.get("selected_validation_auc_percent", summary.get("best_validation_auc_percent")),
               "elapsed_seconds": summary.get("elapsed_seconds"), "reported_tokens": summary.get("total_tokens", 0)}
        exclusion = directory / "exclude_from_evaluation.json"
        row["protocol_excluded"] = exclusion.exists()
        if exclusion.exists():
            row["protocol_exclusion"] = json.loads(exclusion.read_text())
        row["requests_without_usage"] = sum(not r.get("usage") for r in records)
        valid = [r for r in records if "score" in r.get("val", {})]
        row["invalid_candidate_errors"] = dict(Counter(
            r.get("error") or r.get("train", {}).get("error") or
            r.get("val", {}).get("error") or "Unspecified evaluation failure"
            for r in records if "score" not in r.get("val", {})))
        row["valid_request_fraction"] = len(valid) / len(records) if records else None
        if len(valid) >= 5:
            complexity = [sum(1 for _ in ast.walk(ast.parse(r["code"]))) for r in valid]
            auc = [r["val"]["score"] for r in valid]
            if len(set(complexity)) > 1 and len(set(auc)) > 1:
                row["complexity_validation_spearman"] = float(spearmanr(complexity, auc).statistic)
            row["mean_train_validation_gap_pp"] = float(np.mean([r["train"]["score"]-r["val"]["score"] for r in valid]))
        prediction_vectors = []
        for candidate in valid:
            saved = candidate.get("val_scores_file")
            if saved and Path(saved).exists():
                vector = np.load(saved)
                if np.std(vector) > 0:
                    prediction_vectors.append(vector)
        if len(prediction_vectors) >= 2:
            ranks = rankdata(np.stack(prediction_vectors), axis=1)
            corr = np.corrcoef(ranks)
            triangle = corr[np.triu_indices(len(corr), 1)]
            row["mean_prediction_spearman"] = float(np.nanmean(triangle))
            row["prediction_pairs_above_0995_fraction"] = float(np.mean(triangle >= .995))
            row["diversity_candidates_with_saved_predictions"] = len(prediction_vectors)
        reference_path = Path("out/campaign/original_validation") / (row["split"] + "_val_scores.npz")
        if valid and reference_path.exists():
            winner = max(valid, key=lambda r:r["val"]["score"])
            winner_path = winner.get("val_scores_file")
            if winner_path and Path(winner_path).exists():
                winner_ranks = rankdata(np.load(winner_path))
                nearest = []
                with np.load(reference_path) as reference:
                    for name in reference.files:
                        if name == "labels" or len(reference[name]) != len(winner_ranks):
                            continue
                        ranks = rankdata(reference[name])
                        if np.std(ranks) == 0 or np.std(winner_ranks) == 0:
                            continue
                        nearest.append((float(np.corrcoef(ranks, winner_ranks)[0,1]), name,
                                        bool(np.array_equal(ranks, winner_ranks))))
                if nearest:
                    rho, name, equal = max(nearest)
                    row["nearest_supplied_formula_on_validation"] = {"formula": name,
                        "prediction_spearman": rho, "identical_ranks": equal,
                        "candidate_hash": winner["code_hash"]}
        by_round = {}
        for candidate in records:
            group = by_round.setdefault(candidate["round"], [])
            group.append(candidate)
        baseline = summary.get("baseline_val", {}).get("score", 0)
        minutes, best = [0.], [baseline]
        generation_seconds = 0.
        current = baseline
        for _, candidates in sorted(by_round.items()):
            generation_seconds += max(r.get("generation_seconds", 0) for r in candidates)
            current = max([current] + [r["val"]["score"] for r in candidates if "score" in r.get("val", {})])
            minutes.append(generation_seconds/60)
            best.append(current)
        row["generation_seconds_including_failed_requests"] = generation_seconds
        if records and not exclusion.exists():
            trajectories[directory.name] = (minutes, best)
        runs.append(row)
    evaluation_dir = Path("out/campaign/evaluation")
    augmented_dir = Path("out/campaign/evaluation_with_4b")
    verified_file = augmented_dir / "cohort_verified.json"
    augmented_verified = verified_file.exists() and json.loads(verified_file.read_text()).get("common_method_predictions_identical", False)
    if augmented_verified:
        evaluation_dir = augmented_dir
    structured_dir = Path("out/campaign/evaluation_with_structured_4b")
    structured_marker = structured_dir / "cohort_verified.json"
    structured_verified = structured_marker.exists() and json.loads(structured_marker.read_text()).get("common_method_predictions_identical", False)
    if structured_verified:
        evaluation_dir = structured_dir
    devstral_dir = Path("out/campaign/evaluation_with_devstral")
    devstral_marker = devstral_dir / "cohort_verified.json"
    devstral_verified = devstral_marker.exists() and json.loads(devstral_marker.read_text()).get("common_method_predictions_identical", False)
    if devstral_verified:
        evaluation_dir = devstral_dir
    # Pick the most complete verified extension, independent of its scores or
    # the chronological order in which optional trials could fit the budget.
    candidates = [Path("out/campaign/evaluation_before_9b"), Path("out/campaign/evaluation")]
    candidates += [d for d, verified in [(augmented_dir, augmented_verified),
                  (structured_dir, structured_verified), (devstral_dir, devstral_verified)] if verified]
    def attempted_methods(directory):
        path = directory / "all_datasets.json"
        if not path.exists():
            return -1
        data = json.loads(path.read_text())
        if len(data) != 9:
            return -1
        return len(set.union(*(set(d["methods"]) | set(d.get("failed", {})) for d in data.values())))
    evaluation_dir = max(candidates, key=attempted_methods)
    aggregate, macro = macro_results(evaluation_dir)
    complete = len(aggregate) == 9
    if complete:
        for row in runs:
            row.update(macro.get("new/" + row["run"], {}))
    supplementary = {}
    supplementary_data = {}
    for name in ["header_suffix_ablation", "transfer_llama32_1b", "linear_best50"]:
        extra_directory = Path("out/campaign") / name
        variants = {"transfer_llama32_1b": "transfer_llama32_1b_augmented",
                    "header_suffix_ablation": "header_suffix_ablation_refitted"}
        if name in variants:
            augmented = Path("out/campaign") / variants[name]
            marker = augmented / "cohort_verified.json"
            if marker.exists() and json.loads(marker.read_text()).get("common_method_predictions_identical", False):
                extra_directory = augmented
        extra_data, extra_macro = macro_results(extra_directory)
        supplementary_data[name] = extra_data
        supplementary[name] = {"datasets_evaluated": len(extra_data), "macro_results": extra_macro,
                               "evaluation_directory": str(extra_directory),
                               "method_failures": {key: value.get("failed", {}) for key, value in extra_data.items()}}
        if complete and extra_macro:
            supplementary[name]["change_from_primary_macro_auc_pp"] = {
                method: extra_macro[method]["macro_auc_percent"] - macro[method]["macro_auc_percent"]
                for method in extra_macro if method in macro}
            supplementary[name]["paired_change_ci95_pp"] = {
                method: paired_macro_difference(extra_directory, evaluation_dir,
                                                extra_data, aggregate, method, method)
                for method in extra_macro if method in macro}
    target = supplementary["transfer_llama32_1b"]
    restricted = supplementary["linear_best50"]
    if target["datasets_evaluated"] == 9 and restricted["datasets_evaluated"] == 9:
        shared = set(target["macro_results"]) & set(restricted["macro_results"])
        target["supplementary_fixed_search_change_from_source_macro_auc_pp"] = {
            m: target["macro_results"][m]["macro_auc_percent"]-restricted["macro_results"][m]["macro_auc_percent"]
            for m in shared if m.startswith("new/")}
    replicates = {}
    for model in sorted({r["model"] for r in runs}):
        group = [r for r in runs if r["model"] == model and r["search_mode"] == "evolution"
                 and r["split"] == "stratified" and r["status"] == "finished" and not r.get("protocol_excluded", False) and "pilot" not in r["run"]]
        if not group:
            continue
        entry = {"n": len(group), "runs": [r["run"] for r in group],
                 "validation_auc_mean_percent": float(np.mean([r["validation_auc_percent"] for r in group])),
                 "validation_auc_range_percent": [min(r["validation_auc_percent"] for r in group),
                                                   max(r["validation_auc_percent"] for r in group)]}
        if all("macro_auc_percent" in r for r in group):
            values = [r["macro_auc_percent"] for r in group]
            entry.update(held_out_macro_mean_percent=float(np.mean(values)),
                         held_out_macro_range_percent=[min(values), max(values)])
            if len(values) > 1:
                entry["held_out_macro_sample_sd_pp"] = float(np.std(values, ddof=1))
        replicates[model] = entry
    primary_runs = select_primary_runs(runs)
    header_refits = []
    header = supplementary["header_suffix_ablation"]
    for classifier in ["logistic_documented_stratified", "qda_documented_stratified", "logistic_stratified", "qda_stratified"]:
        old, new = "supervised/"+classifier, "supervised/header_refit/"+classifier
        if old in macro and old in header["macro_results"] and new in header["macro_results"]:
            hdir = Path(header["evaluation_directory"])
            hd = supplementary_data["header_suffix_ablation"]
            header_refits.append({"original_method": old, "refitted_method": new,
                "original_macro_auc_percent": macro[old]["macro_auc_percent"],
                "frozen_trimmed_macro_auc_percent": header["macro_results"][old]["macro_auc_percent"],
                "refitted_trimmed_macro_auc_percent": header["macro_results"][new]["macro_auc_percent"],
                "refit_minus_frozen_macro_auc_pp": header["macro_results"][new]["macro_auc_percent"]-header["macro_results"][old]["macro_auc_percent"],
                "paired_refit_gain_ci95_pp": paired_macro_difference(hdir,hdir,hd,hd,new,old),
                "refit_minus_original_full_macro_auc_pp": header["macro_results"][new]["macro_auc_percent"]-macro[old]["macro_auc_percent"],
                "paired_refit_minus_original_full_ci95_pp": paired_macro_difference(hdir,evaluation_dir,hd,aggregate,new,old)})
    for feature_set in ["full", "documented"]:
        path = Path("out/campaign/header_refit_"+feature_set+"/results.json")
        if path.exists():
            (output / "validation_controls" / ("header_refit_"+feature_set+".json")).write_text(path.read_text())
    seed_sweep = None
    seed_sweep_values = {}
    sweep_dir = Path("out/campaign/validation_linear_seed_sweep")
    if (sweep_dir / "summary.json").exists() and (sweep_dir / "candidates.json").exists():
        seed_sweep = json.loads((sweep_dir / "summary.json").read_text())
        seed_records = json.loads((sweep_dir / "candidates.json").read_text())
        seed_sweep_values = {mode: {r["seed"]: r["selected_validation_auc_percent"] for r in seed_records if r["mode"] == mode}
                             for mode in ["independent", "evolution"]}
        if set(seed_sweep_values["independent"]) != set(seed_sweep_values["evolution"]):
            raise ValueError("Supplementary search modes must have the same seed blocks")
        delta = np.array([seed_sweep_values["evolution"][seed]-seed_sweep_values["independent"][seed]
                          for seed in sorted(seed_sweep_values["independent"])])
        rng = np.random.default_rng(65043)
        bootstrap = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)])
        seed_sweep["paired_seed_bootstrap_ci95_pp"] = np.percentile(bootstrap, [2.5,97.5]).tolist()
        seed_sweep["paired_seed_difference_sd_pp"] = float(delta.std(ddof=1))
        (output / "validation_controls" / "linear_seed_sweep.json").write_text(json.dumps(seed_sweep, indent=2)+"\n")
    comparator_differences = []
    comparators = ["original/LogProbSlope_BottomMassAvg", "supervised/logistic_documented_stratified",
                   "supervised/logistic_stratified"]
    # Prespecified validation-selected writers versus fixed comparison methods;
    # these are exploratory paired contrasts, without multiplicity adjustment.
    for run in primary_runs:
        left = "new/"+run
        for right in comparators:
            if left in macro and right in macro:
                comparator_differences.append({
                    "method": left, "comparator": right,
                    "macro_difference_pp": macro[left]["macro_auc_percent"]-macro[right]["macro_auc_percent"],
                    "paired_ci95_pp": paired_macro_difference(evaluation_dir, evaluation_dir, aggregate, aggregate, left, right)})
    formula_ablations = []
    for child, parent in [
        ("ablation_laguna_slope_only", "uncertainty_quantification_laguna_evo_1048"),
        ("ablation_laguna_slope_range", "uncertainty_quantification_laguna_evo_1048"),
        ("ablation_laguna_slope_variance", "uncertainty_quantification_laguna_evo_1048"),
        ("ablation_qwen_linear_position_nll", "uncertainty_quantification_qwen38_evo_1047"),
    ]:
        left, right = "new/"+child, "new/"+parent
        if left not in macro or right not in macro:
            continue
        comparison = {"ablation": child, "parent": parent,
                      "ablation_minus_parent_macro_auc_pp": macro[left]["macro_auc_percent"]-macro[right]["macro_auc_percent"]}
        draws = []
        for dataset in aggregate:
            path = evaluation_dir / (dataset+"_bootstrap.npz")
            if not path.exists():
                continue
            with np.load(path) as saved:
                if left in saved.files and right in saved.files:
                    draws.append(saved[left]-saved[right])
        if len(draws) == 9:
            comparison["paired_ci95_pp"] = (100*np.nanpercentile(np.stack(draws).mean(axis=0), [2.5,97.5])).tolist()
        formula_ablations.append(comparison)
    feedback_comparisons = []
    for evolution in runs:
        if evolution["search_mode"] != "evolution" or evolution["seed"] != 1048 or evolution["split"] != "stratified" or "pilot" in evolution["run"]:
            continue
        for control in runs:
            if control["model"] != evolution["model"] or control["seed"] != 1048 or control["split"] != "stratified" or control["protocol_excluded"] or control["search_mode"] not in {"independent", "baseline_seeded", "structured"}:
                continue
            left, right = "new/"+evolution["run"], "new/"+control["run"]
            if left not in macro or right not in macro:
                continue
            comparison = {"model": evolution["model"], "seed": 1048, "control_mode": control["search_mode"],
                          "evolution_run": evolution["run"], "control_run": control["run"],
                          "evolution_minus_control_macro_auc_pp": macro[left]["macro_auc_percent"]-macro[right]["macro_auc_percent"]}
            draws = []
            for dataset in aggregate:
                path = evaluation_dir / (dataset+"_bootstrap.npz")
                if path.exists():
                    with np.load(path) as saved:
                        if left in saved.files and right in saved.files:
                            draws.append(saved[left]-saved[right])
            if len(draws) == 9:
                comparison["paired_ci95_pp"] = (100*np.nanpercentile(np.stack(draws).mean(axis=0), [2.5,97.5])).tolist()
            feedback_comparisons.append(comparison)
    report = {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "held_out_evaluation_complete": complete, "datasets_evaluated": len(aggregate),
              "primary_evaluation_directory": str(evaluation_dir),
              "bootstrap_units": sorted({d.get("bootstrap_unit", "row") for d in aggregate.values()}),
              "optional_4b_evaluation_verified": augmented_verified,
              "optional_4b_structured_evaluation_verified": structured_verified,
              "optional_devstral_evaluation_verified": devstral_verified,
              "runs": runs, "held_out_macro_results": macro, "supplementary": supplementary,
              "writer_evolution_replicates": replicates,
              "primary_run_selection": [{"run": name, "macro_result_available": "new/"+name in macro} for name in primary_runs],
              "held_out_method_failures": {name: data.get("failed", {}) for name, data in aggregate.items()},
              "feedback_comparisons_seed_1048": feedback_comparisons,
              "validation_selected_comparator_differences": comparator_differences,
              "restricted_search_validation_seed_sweep": seed_sweep,
              "exploratory_header_refit_comparisons": header_refits,
              "untuned_formula_ablations": formula_ablations,
              "notes": ["Writer winners and original comparators selected using validation, before held-out evaluation.",
                        "Macro averages weight the nine datasets equally.",
                        "Bootstrap intervals and contrasts are exploratory, conditional on the fixed datasets, without multiplicity adjustment.",
                        "Cross-condition paired intervals verify identical ordered labels and claim groups; all cohorts use the same deterministic per-dataset bootstrap seed.",
                        "Replicate means include every completed primary evolution seed; n=2 is exploratory.",
                        "Reported tokens exclude usage unavailable after a failed or timed-out request.",
                        "Timing includes pilot and infrastructure overhead; generation curves isolate request time."]}
    (output / "campaign_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = ["run", "model", "search_mode", "seed", "split", "serving_profile", "writer_weight_precision", "status", "protocol_excluded", "requests", "valid_candidates",
               "validation_auc_percent", "macro_auc_percent", "gain_pp", "elapsed_seconds", "reported_tokens"]
    with (output / "run_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)
    with (output / "supplementary_method_table.csv").open("w", newline="") as stream:
        columns = ["condition", "method", "macro_auc_percent", "gain_pp", "duplicate_disjoint_macro_auc_percent",
                   "macro_auc_ci95_percent", "paired_macro_gain_ci95_pp"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for condition, data in supplementary.items():
            writer.writerows({"condition": condition, "method": name, **values} for name, values in data["macro_results"].items())
    if complete:
        with (output / "method_table.csv").open("w", newline="") as stream:
            columns = ["method", "macro_auc_percent", "gain_pp", "duplicate_disjoint_macro_auc_percent",
                       "macro_auc_ci95_percent", "paired_macro_gain_ci95_pp"]
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows({"method": name, **values} for name, values in macro.items())
        with (output / "dataset_table.csv").open("w", newline="") as stream:
            columns = ["dataset", "method", "rows", "duplicate_disjoint_rows", "unique_normalized_claims",
                       "bootstrap_unit", "auc_percent", "gain_pp", "duplicate_disjoint_auc_percent",
                       "auc_ci95_percent", "paired_gain_ci95_pp"]
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for name, dataset in aggregate.items():
                for method, values in dataset["methods"].items():
                    writer.writerow({"dataset": dataset_label(name), "method": method,
                        **{key: dataset[key] for key in ["rows", "duplicate_disjoint_rows",
                                                        "unique_normalized_claims", "bootstrap_unit"]},
                        **values})
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    with PdfPages(output / "experiment_figures.pdf") as pdf:
        if header_refits:
            fig, axes = plt.subplots(2,2,figsize=(11,7),sharey=True)
            for ax, row in zip(axes.flat, header_refits):
                values = [row["original_macro_auc_percent"],row["frozen_trimmed_macro_auc_percent"],row["refitted_trimmed_macro_auc_percent"]]
                intervals = [macro[row["original_method"]]["macro_auc_ci95_percent"],
                             header["macro_results"][row["original_method"]]["macro_auc_ci95_percent"],
                             header["macro_results"][row["refitted_method"]]["macro_auc_ci95_percent"]]
                for i,(value,(lower,upper)) in enumerate(zip(values,intervals)):
                    ax.errorbar(i,value,yerr=[[max(0,value-lower)],[max(0,upper-value)]],fmt="o",capsize=3,color=["#666666","#b85c2b","#3569a7"][i])
                    ax.annotate(f"{value:.2f}",(i,value),xytext=(7,0),textcoords="offset points",fontsize=8)
                ax.set_xticks([0,1,2],["Original inputs","Trim, frozen head","Trim, refit head"],rotation=12)
                ax.set(xlim=(-.4,2.5),title=method_label(row["original_method"]))
                ax.grid(axis="y",alpha=.2)
            for ax in list(axes.flat)[len(header_refits):]:
                ax.set_visible(False)
            axes[0,0].set_ylabel("Held-out macro ROC-AUC (%)")
            axes[1,0].set_ylabel("Held-out macro ROC-AUC (%)")
            fig.suptitle("Exploratory header control: frozen perturbation versus training-only refit")
            fig.text(.5,.01,"Same split and five-setting CV. Intervals resample held-out claims with fitted weights fixed.",ha="center",fontsize=8)
            fig.tight_layout(rect=(0,.04,1,.96))
            fig.savefig(figures / "header_refit_controls.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
        if seed_sweep is not None:
            fig, ax = plt.subplots(figsize=(8,5))
            values = [list(seed_sweep_values[mode].values()) for mode in ["independent", "evolution"]]
            ax.boxplot(values, positions=[0,1], showfliers=False, widths=.45)
            jitter = np.random.default_rng(65044)
            for i, group in enumerate(values):
                ax.scatter(i+jitter.uniform(-.13,.13,len(group)), group, s=16, alpha=.65, color=["#b85c2b", "#3569a7"][i])
                ax.text(i+.30, np.mean(group), f"mean {np.mean(group):.2f}", fontsize=8)
            ax.set_xticks([0,1], ["Independent", "Evolution"])
            ax.set(xlim=(-.5,1.9), ylabel="Best validation ROC-AUC (%)",
                   title=f"Restricted 60-candidate search: {len(values[0])} fresh seeds per mode")
            ax.grid(axis="y",alpha=.2)
            fig.text(.5,.015,"Fixed 17-feature pool; validation only. Additional seeds do not change primary benchmark selection.",ha="center",fontsize=8)
            fig.tight_layout(rect=(0,.045,1,1))
            fig.savefig(figures / "restricted_search_seed_variation.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
        models = sorted({row["model"] for row in runs if row["run"] in trajectories})
        if models:
            columns = min(2, len(models))
            rows = (len(models) + columns - 1) // columns
            fig, axes = plt.subplots(rows, columns, figsize=(5.5*columns, 4*rows), squeeze=False)
            styles = {"evolution": "-", "independent": "--", "baseline_seeded": ":", "structured": "-."}
            for ax, model in zip(axes.flat, models):
                for row in runs:
                    if row["model"] == model and row["run"] in trajectories:
                        x, y = trajectories[row["run"]]
                        label = row["search_mode"] + f" · {row['seed']}"
                        if row["split"] != "stratified":
                            label += " · grouped"
                        if "pilot" in row["run"]:
                            label = "pilot · " + str(row["seed"])
                        ax.step(x, y, where="post", label=label,
                                linestyle=styles.get(row["search_mode"], "-"), linewidth=1.5)
                profiles = ", ".join(sorted({r["serving_profile"].replace("_", " ") for r in runs
                                             if r["model"] == model and r["run"] in trajectories}))
                ax.set(title=model.split("/")[-1]+"\n"+profiles, xlabel="Cumulative generation time (minutes)", ylabel="Best validation ROC-AUC (%)")
                ax.grid(alpha=.2)
                ax.legend(fontsize=7, loc="lower right")
            for ax in list(axes.flat)[len(models):]:
                ax.set_visible(False)
            fig.suptitle("Validation selection versus local model generation cost")
            fig.tight_layout()
            fig.savefig(figures / "validation_vs_generation_cost.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
        if complete:
            selected = ["original/SequenceLogProb"]
            if "alignment/ClaimOnlySequenceLogProb" in macro:
                selected.append("alignment/ClaimOnlySequenceLogProb")
            original_file = Path("results/validation_controls/original_formulas.json")
            if original_file.exists():
                original = json.loads(original_file.read_text())["protocols"]["stratified"]["frozen_top_three_by_validation"][0]
                selected.append("original/" + original)
            selected += ["new/" + name for name in primary_runs if "linear" not in name]
            for method in ["logistic_documented_stratified", "qda_documented_stratified", "logistic_stratified", "qda_stratified"]:
                if "supervised/"+method in macro:
                    selected.append("supervised/"+method)
            selected += ["new/" + name for name in primary_runs if "linear" in name]
            labels = [method_label(name) for name in selected]
            values = np.array([macro.get(name, {}).get("macro_auc_percent", np.nan) for name in selected])
            fig, ax = plt.subplots(figsize=(10, max(4, len(selected)*.5)))
            ax.plot(values, np.arange(len(selected)), "o", color="#3569a7", markersize=6)
            ax.set_yticks(np.arange(len(selected)), labels)
            ax.set_xlim(max(0,float(np.nanmin(values))-8), min(100,float(np.nanmax(values))+5))
            for i, name in enumerate(selected):
                if name not in macro:
                    ax.text(ax.get_xlim()[0]+.7, i, "N/A across all nine datasets", va="center", fontsize=8, color="#777777")
                    continue
                if "macro_auc_ci95_percent" in macro[name]:
                    lower, upper = macro[name]["macro_auc_ci95_percent"]
                    ax.errorbar(values[i], i, xerr=[[max(0,values[i]-lower)], [max(0,upper-values[i])]], fmt="none", color="black", capsize=3)
                else:
                    upper = values[i]
                ax.text(max(values[i], upper)+.25, i, f"{values[i]:.2f}", va="center", fontsize=8)
            ax.set(xlabel="Macro ROC-AUC across nine held-out datasets (%)", title="Frozen methods; writer seeds selected by validation")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(figures / "held_out_macro_auc.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
            dataset_names = list(aggregate)
            gain = np.array([[aggregate[d]["methods"].get(m, {}).get("gain_pp", np.nan) for d in dataset_names] for m in selected[1:]])
            limit = max(1., float(np.nanmax(np.abs(gain))))
            fig, ax = plt.subplots(figsize=(12, max(4, len(selected)*.55)))
            color_map = plt.get_cmap("RdBu").with_extremes(bad="#dddddd")
            heat = ax.imshow(gain, cmap=color_map, vmin=-limit, vmax=limit, aspect="auto")
            ax.set_yticks(np.arange(len(labels)-1), labels[1:])
            ax.set_xticks(np.arange(9), [dataset_label(n) for n in dataset_names], rotation=35, ha="right")
            for i in range(len(gain)):
                for j in range(9):
                    text = f"{gain[i,j]:+.1f}" if np.isfinite(gain[i,j]) else "n/a"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8,
                            color="white" if abs(gain[i,j]) > .65*limit else "black")
            ax.set_title("Change from sequence NLL on each held-out dataset (ROC-AUC points)")
            fig.colorbar(heat, ax=ax, label="ROC-AUC gain (percentage points)")
            fig.tight_layout()
            fig.savefig(figures / "held_out_dataset_gains.png", dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
            control_rows = [r for r in runs if r["seed"] == 1048 and r["split"] == "stratified"
                            and not r["protocol_excluded"] and "pilot" not in r["run"]
                            and r["search_mode"] in {"evolution", "independent", "baseline_seeded", "structured"}
                            and "new/"+r["run"] in macro]
            control_models = sorted(model for model in {r["model"] for r in control_rows}
                                    if len({r["search_mode"] for r in control_rows if r["model"] == model}) >= 2)
            if control_models:
                columns = min(2, len(control_models))
                rows_count = (len(control_models)+columns-1)//columns
                fig, axes = plt.subplots(rows_count, columns, figsize=(6*columns, 4.2*rows_count), squeeze=False, sharey=True)
                mode_order = ["evolution", "independent", "baseline_seeded", "structured"]
                colors = ["#3569a7", "#b85c2b", "#7864a1", "#00877f"]
                for ax, model in zip(axes.flat, control_models):
                    rows = sorted([r for r in control_rows if r["model"]==model], key=lambda r:mode_order.index(r["search_mode"]))
                    for i, row in enumerate(rows):
                        value = macro["new/"+row["run"]]
                        interval = value.get("paired_macro_gain_ci95_pp", [value["gain_pp"]]*2)
                        ax.errorbar(i, value["gain_pp"], yerr=[[max(0,value["gain_pp"]-interval[0])], [max(0,interval[1]-value["gain_pp"])]],
                                    fmt="o", color=colors[mode_order.index(row["search_mode"])], capsize=3)
                    ax.axhline(0, color="#777777", linewidth=.8)
                    ax.set_xticks(range(len(rows)), [r["search_mode"].replace("_", " ")+f"\n{r['requests']} requests" for r in rows], rotation=20, ha="right")
                    ax.set_title(model.split("/")[-1])
                    ax.grid(axis="y", alpha=.2)
                for ax in list(axes.flat)[len(control_models):]:
                    ax.set_visible(False)
                for ax in axes[:,0]:
                    ax.set_ylabel("Macro ROC-AUC gain over sequence NLL (points)")
                fig.suptitle("Search controls: same writer and seed 1048; paired bootstrap intervals")
                fig.tight_layout()
                fig.savefig(figures / "search_control_comparison.png", dpi=200)
                pdf.savefig(fig)
                plt.close(fig)
            extra_present = [(n, supplementary[n]["macro_results"]) for n in supplementary
                             if n in {"header_suffix_ablation", "transfer_llama32_1b"} and supplementary[n]["datasets_evaluated"] == 9]
            if extra_present:
                fig, ax = plt.subplots(figsize=(10, max(4, len(selected)*.5)))
                for i, (name, values_by_method) in enumerate([("Primary Llama-3.1-8B features", macro)] + extra_present):
                    positions = [j for j, m in enumerate(selected) if m in values_by_method]
                    ax.plot([values_by_method[selected[j]]["macro_auc_percent"] for j in positions],
                            np.array(positions) + (i-1)*.12, "o", label=name.replace("_", " "), markersize=5)
                ax.set_yticks(np.arange(len(labels)), labels)
                ax.invert_yaxis()
                ax.set(xlabel="Macro ROC-AUC (%)", title="Frozen-method sensitivity: header tokens and feature model")
                fig.text(.5, .01, "Missing markers indicate an unavailable result or a method outside that frozen cohort.",
                         ha="center", fontsize=7, color="#555555")
                ax.grid(axis="x", alpha=.2)
                ax.legend(fontsize=8)
                fig.tight_layout(rect=(0,.035,1,1))
                fig.savefig(figures / "frozen_method_sensitivity.png", dpi=200)
                pdf.savefig(fig)
                plt.close(fig)
            if formula_ablations:
                names = {"ablation_laguna_slope_only": "Laguna: slope only",
                         "ablation_laguna_slope_range": "Laguna: slope + entropy range",
                         "ablation_laguna_slope_variance": "Laguna: slope + top-1 variance",
                         "ablation_qwen_linear_position_nll": "Qwen: simplified position-weighted NLL"}
                fig, ax = plt.subplots(figsize=(10, 4))
                for i, comparison in enumerate(formula_ablations):
                    value = comparison["ablation_minus_parent_macro_auc_pp"]
                    lower, upper = comparison.get("paired_ci95_pp", [value, value])
                    ax.errorbar(value, i, xerr=[[max(0, value-lower)], [max(0,upper-value)]],
                                fmt="o", color="#3569a7", capsize=3)
                ax.set_yticks(range(len(formula_ablations)), [names[c["ablation"]] for c in formula_ablations])
                ax.axvline(0, color="#777777", linewidth=.8)
                ax.invert_yaxis()
                ax.set(xlabel="Ablation minus full formula: macro ROC-AUC change (points)",
                       title="Untuned simplifications across nine datasets; paired 95% intervals")
                ax.grid(axis="x", alpha=.2)
                fig.tight_layout()
                fig.savefig(figures / "formula_simplification.png", dpi=200)
                pdf.savefig(fig)
                plt.close(fig)
    print(json.dumps({"runs": len(runs), "held_out_complete": complete, "report": str(output / "campaign_summary.json")}))


if __name__ == "__main__":
    main()
