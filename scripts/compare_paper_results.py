"""Compare frozen campaign results with published values and audit UHEAD polarity.

This is CPU-only postprocessing. It never changes labels, saved predictions,
fitted models, selected seeds or any primary benchmark artifact. The secondary
condition uses the raw UHEAD `verified` field, supported by pinned upstream
annotation code. Exact provenance of the supplied CSV conversion is unavailable.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


PAPER = "https://aclanthology.org/2026.acl-srw.115.pdf"
BASELINE = "original/SequenceLogProb"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory):
    data = json.loads((directory / "all_datasets.json").read_text())
    if len(data) != 9:
        raise ValueError("Comparison requires a completed nine-dataset cohort")
    uhead = next(name for name in data if name.startswith("UHEAD_"))
    draws = {}
    for dataset in data:
        with np.load(directory / (dataset + "_bootstrap.npz")) as saved:
            draws[dataset] = {name: saved[name] for name in saved.files}
    methods = set.intersection(*(set(item["methods"]) for item in data.values()))
    results = {}
    for name in sorted(methods):
        original = [d["methods"][name]["auc_percent"] for d in data.values()]
        changed = [100 - d["methods"][name]["auc_percent"] if k == uhead
                   else d["methods"][name]["auc_percent"] for k, d in data.items()]
        result = {"supplied_label_macro_auc_percent": float(np.mean(original)),
                  "uhead_verified_macro_auc_percent": float(np.mean(changed)),
                  "uhead_supplied_auc_percent": data[uhead]["methods"][name]["auc_percent"],
                  "uhead_verified_auc_percent": 100 - data[uhead]["methods"][name]["auc_percent"]}
        if all(name in draws[k] for k in data):
            x = np.mean([1 - draws[k][name] if k == uhead else draws[k][name]
                         for k in data], axis=0)
            result["uhead_verified_macro_auc_ci95_percent"] = (100 * np.percentile(x, [2.5, 97.5])).tolist()
            ref = np.mean([1 - draws[k][BASELINE] if k == uhead else draws[k][BASELINE]
                           for k in data], axis=0)
            result["uhead_verified_gain_ci95_pp"] = (100 * np.percentile(x-ref, [2.5, 97.5])).tolist()
        results[name] = result
    for result in results.values():
        result["uhead_verified_gain_over_nll_pp"] = result["uhead_verified_macro_auc_percent"] - results[BASELINE]["uhead_verified_macro_auc_percent"]
    return data, results, draws, uhead


def main():
    summary = json.loads(Path("results/campaign_summary.json").read_text())
    directory = Path(summary["primary_evaluation_directory"])
    data, methods, draws, uhead = summarize(directory)
    protected = [directory / "all_datasets.json", directory / (uhead+"_scores.npz"),
                 directory / (uhead+"_bootstrap.npz"), Path("results/campaign_summary.json")]
    before = {str(p): digest(p) for p in protected}
    csv_path = next(p for p in Path("data/claims").rglob("*.csv") if "UHEAD" in str(p))
    original_audit = json.loads(Path("results/dataset_audit.json").read_text())
    expected_csv_hash = next(row["sha256"] for row in original_audit["datasets"] if "UHEAD" in row["file"])
    assert digest(csv_path) == expected_csv_hash
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    supplied = np.array([float(row["is_hallu"]) for row in rows])
    verified = np.array([float(row["verified"]) for row in rows])
    is_correct = np.array([float(row["is_correct"]) for row in rows])
    assert len(rows) == 2057
    assert np.array_equal(verified, is_correct)
    assert np.array_equal(verified, 1-supplied)
    with np.load(directory / (uhead+"_scores.npz")) as scores:
        assert np.array_equal(scores["labels"], supplied)
        for name in methods:
            original = 100 * roc_auc_score(supplied, scores[name])
            alternate = 100 * roc_auc_score(verified, scores[name])
            assert np.isclose(original, methods[name]["uhead_supplied_auc_percent"], atol=1e-9)
            assert np.isclose(alternate, 100-original, atol=1e-9)
        # Check the complementary-AUC identity with nonuniform bootstrap weights.
        rng = np.random.default_rng(20261015)
        counts = rng.integers(1, 5, len(supplied))
        a = roc_auc_score(supplied, scores[BASELINE], sample_weight=counts)
        b = roc_auc_score(verified, scores[BASELINE], sample_weight=counts)
        assert np.isclose(a+b, 1, atol=1e-12)
    feedback = []
    for comparison in summary["feedback_comparisons_seed_1048"]:
        left, right = ["new/"+comparison[k] for k in ["evolution_run", "control_run"]]
        paired = np.mean([-(draws[k][left]-draws[k][right]) if k == uhead
                          else draws[k][left]-draws[k][right] for k in data], axis=0)
        feedback.append(dict(comparison,
            uhead_verified_evolution_minus_control_macro_auc_pp=methods[left]["uhead_verified_macro_auc_percent"]-methods[right]["uhead_verified_macro_auc_percent"],
            uhead_verified_paired_ci95_pp=(100*np.percentile(paired, [2.5,97.5])).tolist()))
    comparators = []
    for item in summary["validation_selected_comparator_differences"]:
        left, right = item["method"], item["comparator"]
        paired = np.mean([-(draws[k][left]-draws[k][right]) if k == uhead
                          else draws[k][left]-draws[k][right] for k in data], axis=0)
        comparators.append(dict(item,
            uhead_verified_macro_difference_pp=methods[left]["uhead_verified_macro_auc_percent"]-methods[right]["uhead_verified_macro_auc_percent"],
            uhead_verified_paired_ci95_pp=(100*np.percentile(paired, [2.5,97.5])).tolist()))
    ablations = []
    for item in summary["untuned_formula_ablations"]:
        left, right = "new/"+item["ablation"], "new/"+item["parent"]
        paired = np.mean([-(draws[k][left]-draws[k][right]) if k == uhead
                          else draws[k][left]-draws[k][right] for k in data], axis=0)
        ablations.append(dict(item,
            uhead_verified_ablation_minus_parent_macro_auc_pp=methods[left]["uhead_verified_macro_auc_percent"]-methods[right]["uhead_verified_macro_auc_percent"],
            uhead_verified_paired_ci95_pp=(100*np.percentile(paired, [2.5,97.5])).tolist()))
    supplementary = {}
    for name, item in summary["supplementary"].items():
        if item["datasets_evaluated"] == 9:
            _, results, _, _ = summarize(Path(item["evaluation_directory"]))
            supplementary[name] = results
    # Published values transcribed from Figure 2; missing identities stay missing.
    published = [
        ("Sequence Probability", 68.7, BASELINE, "ranking-equivalent baseline"),
        ("Perplexity", 64.2, "original/Perplexity", "ranking-equivalent baseline"),
        ("Mean Token Entropy", 59.9, "original/MeanTokenEntropy", "same named baseline; remaining extraction differences possible"),
        ("RAUQ", 61.8, None, "not evaluated: attention statistics not cached"),
        ("AttentionScore", 56.0, None, "not evaluated: attention statistics not cached"),
        ("Evo-Gpt-1", 70.8, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Gpt-3", 70.6, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Gpt-5", 70.6, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Gpt-7", 68.7, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Gpt-9", 68.7, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Gpt-11", 64.5, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Gpt-13", 63.8, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Gpt-15", 62.0, None, "no certified paper-ID to repository-class mapping"),
        ("Evo-Anth-1", 73.4, None, "historical Claude implementation/population unavailable"),
        ("Evo-Anth-8", 72.7, None, "historical Claude implementation/population unavailable"),
        ("Evo-Anth-15", 71.9, None, "historical Claude implementation/population unavailable"),
        ("Evo-Anth-22", 71.6, None, "historical Claude implementation/population unavailable"),
        ("Evo-Anth-30", 71.0, None, "historical Claude implementation/population unavailable"),
    ]
    comparisons = []
    for name, value, counterpart, identity in published:
        local = methods.get(counterpart, {})
        aligned = local.get("uhead_verified_macro_auc_percent")
        comparisons.append({"paper_method": name, "paper_figure_2_auc_percent": value,
            "local_method": counterpart or "", "identity_status": identity,
            "supplied_label_auc_percent": local.get("supplied_label_macro_auc_percent", ""),
            "uhead_verified_auc_percent": aligned if aligned is not None else "",
            "uhead_verified_minus_published_pp": aligned-value if aligned is not None else "",
            "paper_source": PAPER+"#page=4"})
    write_csv(Path("results/paper_figure2_comparison.csv"), comparisons)
    write_csv(Path("results/uhead_label_convention_methods.csv"),
              [dict(method=name, **value) for name, value in methods.items()])
    # Table 3 preserves per-dataset values: its rounded cells are not Figure 2 macros.
    table3 = {"PopQA":(83,84), "AVeriTeC":(68,65), "X-Fact":(53,57),
              "UHEAD":(68,69), "Cities":(99,98), "Companies":(89,87),
              "CommonClaim":(69,67), "CounterFact":(78,77), "WildHall":(65,65)}
    with Path("results/dataset_table.csv").open() as stream:
        dataset_rows = list(csv.DictReader(stream))
    per_dataset = []
    for dataset, (evo, logreg) in table3.items():
        row = {"dataset": dataset, "paper_table3_evo_anth1_auc_percent_rounded": evo,
               "paper_table3_logreg1_auc_percent_rounded": logreg,
               "paper_table3_feature_basis": "Evo-Anth-1 custom features; not our 14-feature pool"}
        for tag, method in [("qwen38_evo1047", "new/uncertainty_quantification_qwen38_evo_1047"),
                            ("logistic14", "supervised/logistic_documented_stratified")]:
            value = float(next(x["auc_percent"] for x in dataset_rows if x["dataset"] == dataset and x["method"] == method))
            row[tag+"_supplied_label_auc_percent"] = value
            row[tag+"_uhead_verified_auc_percent"] = 100-value if dataset == "UHEAD" else value
        per_dataset.append(row)
    write_csv(Path("results/paper_table3_context.csv"), per_dataset)
    after = {str(p): digest(p) for p in protected}
    assert before == after
    report = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_status": "posthoc annotation-polarity audit; primary artifacts preserved",
        "label_provenance": json.loads(Path("results/paper_label_provenance.json").read_text()),
        "uhead_csv_audit": {"file":str(csv_path), "sha256":digest(csv_path), "rows":len(rows),
            "verified_equals_is_correct_all_rows":True, "is_hallu_equals_one_minus_verified_all_rows":True,
            "csv_sha256_matches_original_campaign_audit":True,
            "verified_positive_rows":int(verified.sum()), "supplied_is_hallu_positive_rows":int(supplied.sum()),
            "prediction_labels_match_supplied_csv":True,
            "inference":"The supplied UHEAD CSV appears to invert the upstream false=1 convention during conversion. The exact CSV conversion script is unavailable; this is not a certified paper reproduction."},
        "primary_artifact_sha256_before_and_after":before, "primary_artifacts_unchanged":True,
        "protocol":{"secondary_label_field":"verified for UHEAD only; original is_hallu everywhere else",
            "predictions_models_seeds_and_splits_unchanged":True,
            "bootstrap":"Complement each UHEAD per-method AUC draw; preserve original claim resampling and every other dataset draw",
            "new_gpu_hours":0, "new_model_generations":0,
            "do_not_use":"Do not choose dataset polarity by maximizing AUC; compare only the independently source-supported convention."},
        "methods":methods, "feedback_comparisons":feedback, "formula_ablations":ablations,
        "validation_selected_comparator_differences":comparators,
        "supplementary":supplementary, "paper_figure2_comparisons":comparisons,
        "paper_table3_note":"Rounded per-dataset context only; different feature pools. Do not infer a controlled superiority comparison or reconstruct Figure 2 by averaging this table.",
        "paper_internal_consistency_check":{
            "table3_evo_anth1_mean_of_nine_displayed_test_auc_percent":float(np.mean([x[0] for x in table3.values()])),
            "figure2_evo_anth1_reported_macro_auc_percent":73.4,
            "note":"The displayed Table 3 mean and Figure 2 macro differ by more than rounding alone explains; the PDF does not give an explicit reconciliation. Keep the two published results separate."}}
    Path("results/paper_alignment.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"methods":len(methods), "uhead_rows_audited":len(rows),
        "primary_artifacts_unchanged":True, "baseline_secondary":methods[BASELINE],
        "qwen27_secondary":methods["new/uncertainty_quantification_qwen38_evo_1047"]}))


if __name__ == "__main__":
    main()
