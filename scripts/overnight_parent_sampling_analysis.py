"""Compare parent weights on the same observed eligible pools; no model calls.

This conditions on pools produced by rank-policy runs. It does not estimate a
causal effect on evolution outcomes or guarantee behavioural diversity.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2107, 4319, 8741)
METRICS = ("n_eligible", "rank_normalized_entropy", "inherited_normalized_entropy", "entropy_delta",
           "rank_top_group_probability", "inherited_top_group_probability", "top_probability_delta",
           "rank_worst_group_probability", "inherited_worst_group_probability", "worst_probability_delta",
           "rank_expected_training_fitness", "inherited_expected_training_fitness", "expected_fitness_delta",
           "total_variation_distance", "rank_two_parent_expected_mean_fitness", "inherited_two_parent_expected_mean_fitness",
           "two_parent_expected_mean_fitness_delta", "rank_two_parent_top_group_inclusion_probability",
           "inherited_two_parent_top_group_inclusion_probability", "rank_two_parent_worst_group_inclusion_probability",
           "inherited_two_parent_worst_group_inclusion_probability", "rank_two_parent_set_normalized_entropy",
           "inherited_two_parent_set_normalized_entropy", "two_parent_set_entropy_delta", "two_parent_set_total_variation_distance")


def two_parent_sets(probabilities):
    """Enumerate exact unordered two-parent set probabilities, without sampling."""
    pairs = [(i,j) for i in range(len(probabilities)) for j in range(i+1, len(probabilities))]
    values = np.asarray([probabilities[i]*probabilities[j]/(1-probabilities[i])
                         +probabilities[j]*probabilities[i]/(1-probabilities[j]) for i,j in pairs])
    if pairs and not np.isclose(values.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Two-parent set probabilities do not sum to one")
    return pairs, values


def inclusion_two(probabilities):
    """Exact inclusion probabilities for <=2 weighted draws without replacement."""
    if len(probabilities) == 1:
        return np.ones(1)
    pairs, values = two_parent_sets(probabilities)
    result = np.zeros(len(probabilities))
    for (i,j), probability in zip(pairs, values):
        result[i] += probability
        result[j] += probability
    return result


def analyse_event(event):
    scores = np.asarray(event["eligible_scores"], dtype=float)
    actual = np.asarray(event["probabilities"], dtype=float)
    temperature, reserve = float(event["inherited_temperature"]), float(event["uniform_reserve"])
    if (scores.ndim != 1 or not len(scores) or actual.shape != scores.shape
            or not np.isfinite(scores).all() or not np.isfinite(actual).all()
            or not np.isfinite(temperature) or temperature <= 0 or reserve != .2
            or (actual <= 0).any() or not np.isclose(actual.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Invalid recorded score pool or probability vector")
    n = len(scores)
    ranks = rankdata(scores, method="average")
    expected = (1-reserve)*ranks/ranks.sum()+reserve/n
    if not np.allclose(actual, expected, atol=1e-12, rtol=0):
        raise ValueError("Recorded probabilities differ from the pinned rank+.2 uniform rule")
    inherited = softmax(scores/temperature)
    index = int(np.argmax(inherited))
    inherited[index] = 1-inherited[:index].sum()-inherited[index+1:].sum()
    # Island.get_prompt adds this guard to either weighting result.
    actual = (actual+1e-20)/(actual+1e-20).sum()
    inherited = (inherited+1e-20)/(inherited+1e-20).sum()
    degenerate = len(np.unique(scores)) == 1
    row = dict(n_eligible=n, one_group_degenerate=degenerate, group_set_selection_can_change=n>2,
               inherited_temperature=temperature, uniform_reserve=reserve,
               total_variation_distance=float(np.abs(actual-inherited).sum()/2))
    set_distributions = {}
    for prefix, probabilities in (("rank", actual), ("inherited", inherited)):
        inclusion = inclusion_two(probabilities)
        pairs, set_probabilities = two_parent_sets(probabilities)
        set_distributions[prefix] = set_probabilities
        expected_mean = float(sum(probability*(scores[i]+scores[j])/2
                                  for (i,j), probability in zip(pairs, set_probabilities))) if pairs else float(scores[0])
        row.update({prefix+"_normalized_entropy": float(-np.dot(probabilities, np.log(probabilities))/np.log(n)) if n>1 else None,
            prefix+"_top_group_probability": float(probabilities[scores==scores.max()].sum()),
            prefix+"_worst_group_probability": float(probabilities[scores==scores.min()].sum()),
            prefix+"_expected_training_fitness": float(np.dot(probabilities, scores)),
            prefix+"_two_parent_expected_mean_fitness": expected_mean,
            prefix+"_two_parent_top_group_inclusion_probability": float(inclusion[scores==scores.max()].sum()),
            prefix+"_two_parent_worst_group_inclusion_probability": float(inclusion[scores==scores.min()].sum()),
            prefix+"_two_parent_set_normalized_entropy": float(-np.dot(set_probabilities, np.log(set_probabilities))/np.log(len(pairs))) if len(pairs)>1 else None})
    for name, suffix in (("entropy", "normalized_entropy"), ("top_probability", "top_group_probability"),
                         ("worst_probability", "worst_group_probability"), ("expected_fitness", "expected_training_fitness"),
                         ("two_parent_expected_mean_fitness", "two_parent_expected_mean_fitness")):
        row[name+"_delta"] = row["rank_"+suffix]-row["inherited_"+suffix] if not degenerate else None
    row["two_parent_set_entropy_delta"] = (row["rank_two_parent_set_normalized_entropy"]-row["inherited_two_parent_set_normalized_entropy"]) if n>2 else None
    row["two_parent_set_total_variation_distance"] = float(np.abs(set_distributions["rank"]-set_distributions["inherited"]).sum()/2)
    return row


def summarize(rows):
    selected = [row for row in rows if not row["one_group_degenerate"]]
    result = dict(all_events=len(rows), excluded_one_group_events=len(rows)-len(selected), nondegenerate_events=len(selected),
                  group_set_selection_changeable_events=sum(row["group_set_selection_can_change"] for row in selected))
    def means(members):
        values = {key: [row[key] for row in members if row[key] is not None] for key in METRICS}
        return {key: float(np.mean(items)) if items else None for key, items in values.items()}
    result["nondegenerate_event_means"] = means(selected)
    result["defined_nondegenerate_events_per_metric"] = {key: sum(row[key] is not None for row in selected) for key in METRICS}
    changeable = [row for row in selected if row["group_set_selection_can_change"]]
    result["changeable_group_set_event_means"] = means(changeable)
    result["rank_increases_expected_first_draw_fitness_events"] = sum(row["expected_fitness_delta"]>1e-10 for row in selected)
    result["rank_decreases_expected_first_draw_fitness_events"] = sum(row["expected_fitness_delta"]< -1e-10 for row in selected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/next_experiments/sampler_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/overnight20261021/parent_sampling_analysis")
    args = parser.parse_args()
    hashes, warnings = {}, []

    def read(path, expected=None):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if expected is not None and expected != digest:
            raise ValueError("Pinned source/config differs: " + name)
        hashes[name] = digest
        return data

    def lines(path):
        data = read(path)
        result = []
        for index, line in enumerate(data.splitlines()):
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(data.splitlines())-1 or data.endswith(b"\n"):
                    raise
                warnings.append("Excluded incomplete final JSONL record: " + str(path))
        return result

    read(Path(__file__))
    manifest = json.loads(read(args.manifest))
    for name, digest in manifest["study_source_sha256"].items():
        read(ROOT / name, digest)
    rows, runs = [], []
    for job in manifest["jobs"]:
        if job.get("parent_sampling") != "rank_uniform" or job["requests"] != 60:
            continue
        read(ROOT / "configs" / (job["config"]+".yaml"), job["config_sha256"])
        folder = ROOT / job["expected_logs"]
        events = lines(folder / "parent_sampling_events.jsonl") if (folder / "parent_sampling_events.jsonl").exists() else []
        summary = json.loads(read(folder / "summary.json")) if (folder / "summary.json").exists() else {}
        ledger = lines(folder / "candidates.jsonl") if (folder / "candidates.jsonl").exists() else []
        if events:
            config = json.loads(read(folder / "trial_config.json"))
            provenance = json.loads(read(folder / "parent_sampling_provenance.json"))
            if (config["programdatabaseConfig"]["functions_per_prompt"] != 2
                    or config["trial"].get("parent_sampling") != "rank_uniform" or config["seed"] != job["seed"]
                    or provenance["wrapper_sha256"] != manifest["study_source_sha256"]["scripts/overnight_sampler_trial.py"]
                    or provenance["inherited_database_sha256"] != manifest["study_source_sha256"]["src/packing/funsearch/programs_database.py"]):
                raise ValueError("Unexpected parent mechanism/config provenance: " + job["id"])
            if any(event["inherited_temperature"] != config["programdatabaseConfig"]["temp"] for event in events):
                raise ValueError("Recorded inherited temperature differs from saved config")
        keys = [(record["round"], record["index"]) for record in ledger]
        full = (summary.get("status") == "finished" and summary.get("requests") == 60 and len(ledger) == 60
                and set(keys) == {(r,i) for r in range(30) for i in (0,1)} and len(events) == 30)
        status = "full_budget" if full else "partial" if events else "not_run"
        run_rows = [dict(job=job["id"], seed=job["seed"], status=status, event_1based=index+1, **analyse_event(event))
                    for index, event in enumerate(events)]
        runs.append(dict(job=job["id"], seed=job["seed"], status=status, full_budget=full, **summarize(run_rows)))
        rows.extend(run_rows)
    complete = len(runs)==3 and {run["seed"] for run in runs}==set(SEEDS) and all(run["full_budget"] for run in runs)
    all_summary = summarize(rows)
    seed_metrics = [run["nondegenerate_event_means"] for run in runs]
    primary = {key: float(np.mean([metrics[key] for metrics in seed_metrics])) if all(metrics[key] is not None for metrics in seed_metrics) else None
               for key in METRICS} if complete else None
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status="complete_three_seeds" if complete else "partial",
        conditioning="Both rules evaluated on each same eligible fitness-group pool observed in rank-policy runs; not a causal evolution-outcome comparison or a diversity guarantee.",
        mechanism="Inherited softmax(score / recorded_temperature) versus recorded 0.8 * ascending_average_rank / sum(ranks) + 0.2 / n. Eligibility, island choice, and within-group shorter-source weighting remain inherited.",
        probabilities="First-draw entropy, top/worst probability, and expected training fitness describe the first weighted draw. Two parents are sampled without replacement, then sorted; n<=2 always selects every eligible group. Exact unordered-set probabilities, set entropy/TV, group inclusion, and expected mean fitness describe this actual two-parent group selection.",
        unordered_set_formula="P({i,j}) = p_i*p_j/(1-p_i) + p_j*p_i/(1-p_j), i<j; inclusion_i = sum of sets containing i; expected parent mean = sum P({i,j})*(score_i+score_j)/2. Set normalized entropy undefined for n<=2 (only one possible set).",
        fitness_units="Directed training AUROC percentage points. No validation or test outcomes used.",
        averaging="All nondegenerate recorded events contribute to descriptive means, including partial runs. One-fitness-group events excluded. Primary means weight seeds equally and require all three full 60-request runs plus 30 events each.",
        all_recorded_events_summary=all_summary, primary_equal_seed_metric_means=primary, runs=runs, events=rows,
        warnings=warnings, input_sha256=hashes, model_calls=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parent_sampling.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (args.output / "parent_sampling_events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    means = all_summary["nondegenerate_event_means"]
    text = ["# Parent-selection mechanism", "", "Status: **"+report["status"]+"**. "+report["conditioning"], "",
        report["mechanism"], "", report["probabilities"], "",
        f"Observed calls: {len(rows)}; excluded one-group calls: {all_summary['excluded_one_group_events']}; comparable calls: {all_summary['nondegenerate_events']}; pools with >2 eligible groups: {all_summary['group_set_selection_changeable_events']}. Recorded inherited temperatures: {sorted({row['inherited_temperature'] for row in rows})}.", "",
        "| Metric, mean across nondegenerate observed calls | Rank + 20% uniform | Inherited softmax |",
        "| --- | ---: | ---: |"]
    for name, suffix in (("First-draw normalized entropy", "normalized_entropy"), ("First-draw best-group probability", "top_group_probability"),
                         ("First-draw worst-group probability", "worst_group_probability"), ("Expected first-draw fitness", "expected_training_fitness"),
                         ("Expected mean fitness of two selected groups", "two_parent_expected_mean_fitness"),
                         ("Best-group inclusion probability in two-parent set", "two_parent_top_group_inclusion_probability"),
                         ("Worst-group inclusion probability in two-parent set", "two_parent_worst_group_inclusion_probability"),
                         ("Two-parent set normalized entropy (>2 eligible groups)", "two_parent_set_normalized_entropy")):
        values = [means[prefix+suffix] for prefix in ("rank_", "inherited_")]
        text.append("| "+name+" | "+" | ".join(f"{value:.5f}" if value is not None else "undefined" for value in values)+" |")
    tv = means["total_variation_distance"]
    text += ["", f"Mean total-variation distance: {tv:.5f}." if tv is not None else "No nondegenerate calls yet.", "",
             "Rank weighting can increase or decrease exploitation relative to temperature 40 depending on score gaps; use the signed differences above rather than assuming the direction.", "",
             report["fitness_units"], "", report["averaging"], "", "Per-seed summaries, exact two-parent metrics, and input hashes are preserved in JSON; per-event metrics are in CSV."]
    if means["two_parent_set_total_variation_distance"] is not None:
        text += ["", f"Mean two-parent set total-variation distance: {means['two_parent_set_total_variation_distance']:.5f}.", "", report["unordered_set_formula"]]
    if primary:
        text += ["", f"All three seeds complete. Equal-seed mean changes: expected first-draw fitness {primary['expected_fitness_delta']:+.5f}, expected two-parent mean fitness {primary['two_parent_expected_mean_fitness_delta']:+.5f}, first-draw normalized entropy {primary['entropy_delta']:+.5f}."]
    (args.output / "PARENT_SAMPLING.md").write_text("\n".join(text)+"\n")
    print(json.dumps(dict(status=report["status"], events=len(rows), nondegenerate_events=all_summary["nondegenerate_events"],
                         complete_seeds=sum(run["full_budget"] for run in runs), output=str(args.output), model_calls=False)))


if __name__ == "__main__":
    main()
