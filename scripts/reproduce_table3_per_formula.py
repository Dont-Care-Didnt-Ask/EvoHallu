"""CPU-only per-formula coefficient refits from frozen component predictions.

This is an exploratory analogue of Table 3, using all four Qwen3.5-9B screen
winners. It neither refits the historical pooled head nor opens raw LLM caches.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/overnight20261021"
POLICIES = ("original", "neutral", "diverse", "compositional")
WEIGHTS = {"original": [.7, .3], "neutral": [1., 4., 2.],
           "diverse": [1., 2., 2.5], "compositional": [.4, .3, 3.]}
DATASETS = [
    ("popqa__popqa_test_filtered_10shot", "PopQA test"),
    ("AVeriTeC__dev_filtered_10shot", "AVeriTeC"),
    ("x-fact__test_filtered_10shot", "X-Fact"),
    ("UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot", "UHEAD"),
    ("geometry__cities", "Cities"),
    ("geometry__companies_true_false", "Companies"),
    ("geometry__common_claim_true_false", "Common Claims"),
    ("geometry__counterfact_true_false", "CounterFact"),
    ("WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot", "WildHallucinations"),
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def auc(y, prediction):
    assert prediction.shape == y.shape and np.isfinite(prediction).all()
    return float(100 * roc_auc_score(y, prediction))


def fold(value):
    return max(value, 100 - value)


def raw_parameters(mean, scale, beta, intercept):
    coefficient = beta / scale
    return coefficient, float(intercept - mean @ coefficient)


def evaluate_rows(policy, scope, name, labels, evo, lr):
    a, b = auc(labels, evo), auc(labels, lr)
    return dict(policy=policy, scope=scope, dataset=name, samples=len(labels),
                evo_directed=a, logreg_directed=b, directed_delta=b-a,
                evo_folded=fold(a), logreg_folded=fold(b), folded_delta=fold(b)-fold(a))


def reconstruction_check(policy, dataset, labels, original, features):
    reconstructed = features @ np.asarray(WEIGHTS[policy])
    difference = float(np.max(np.abs(original-reconstructed)))
    auc_difference = abs(auc(labels, original)-auc(labels, reconstructed))
    if difference > 1e-3 or auc_difference > .001:
        raise ValueError(f"Component reconstruction guard failed: {policy}/{dataset}")
    return dict(policy=policy, dataset=dataset, max_score_error=difference,
                absolute_auc_error_pp=auc_difference, passed=True)


def train_only_fit(features, labels, groups, train, validation, C_grid):
    assert not set(groups[train]) & set(groups[validation])
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=441).split(
        features[train], labels[train], groups[train]))
    fold_records = []
    for fit, holdout in folds:
        assert not set(groups[train][fit]) & set(groups[train][holdout])
        fold_records.append(dict(fit_samples=len(fit), holdout_samples=len(holdout),
                                 shared_normalized_claims=0))
    tuning = []
    for C in C_grid:
        scores = []
        for fit, holdout in folds:
            pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            pipeline.fit(features[train][fit], labels[train][fit])
            assert int(pipeline[0].n_samples_seen_) == len(fit)
            scores.append(auc(labels[train][holdout], pipeline.decision_function(features[train][holdout])))
        tuning.append(dict(C=C, mean_auc=float(np.mean(scores)), fold_auc=scores))
    selected = max(tuning, key=lambda row: row["mean_auc"])
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=selected["C"], max_iter=2000))
    pipeline.fit(features[train], labels[train])
    assert pipeline[-1].classes_.tolist() == [0, 1]
    assert int(pipeline[0].n_samples_seen_) == len(train)
    return pipeline, dict(chosen_C=selected["C"], tuning=tuning, inner_folds=fold_records,
                          fit_samples=len(train), diagnostic_validation_samples=len(validation),
                          outer_shared_normalized_claims=0,
                          validation_used_for_hyperparameter_selection=False,
                          held_out_inputs_read_before_model_frozen=False)


def render_report(result, output):
    source = [r for r in result["metrics"] if r["scope"] == "source_nine"]
    macro = {r["policy"]: r for r in result["metrics"] if r["scope"] == "macro_nine"}
    train = {r["policy"]: r for r in result["metrics"] if r["scope"] == "fit_train"}
    validation = {r["policy"]: r for r in result["metrics"] if r["scope"] == "diagnostic_validation"}
    lines = [
        "# evohallu — Table 3: evolution versus logistic refitting", "",
        "New exploratory CPU follow-up completed 22 October 2026. This reproduces the **comparison design** of [Table 3 in EvoHallu](https://aclanthology.org/2026.acl-srw.115.pdf#page=7): keep one generated formula's feature set fixed and compare its original weights with weights fitted by logistic regression. It uses our four frozen **Qwen3.5-9B fixed-context screen winners**, not the authors' Claude programs or their best/median candidate ranks.", "",
        "Each logistic model uses only its own formula's two or three components. There is **no pooling across formulas and no extra NLL column**. The historical pooled twelve-column head scoring 73.633 is a different experiment and is unchanged.", "",
        "Source probability/logit features come from **Llama3.1-8B-Instruct**. No LLM calls, GPU work, new candidate generation or target-model fitting were performed.", "",
        "## Evaluation corresponding to Table 3", "",
        "AUROC × 100. Each test-dataset cell uses `max(AUC, 100-AUC)`; Mean is the equal mean of the nine test cells, excluding PopQA fit and diagnostic validation. Evo uses the actual saved frozen-program predictions. LogReg is fitted only on 5,606 PopQA rows; the 1,401 grouped outer-validation rows are diagnostic. Scaling and C selection use three claim-grouped folds inside fitting rows only, with C ∈ {0.01, 0.1, 1, 10, 100}.", "",
        "| Method | PopQA fit | PopQA test | AVeriTeC | X-Fact | UHEAD | Cities | Companies | Common Claims | CounterFact | WildHallucinations | Mean nine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        for model in ("evo", "logreg"):
            values = [train[policy][model+"_directed"]]
            values += [next(r[model+"_folded"] for r in source if r["policy"] == policy and r["dataset"] == name) for _, name in DATASETS]
            values += [macro[policy][model+"_folded"]]
            lines.append("| "+("Evo" if model == "evo" else "LogReg")+" / "+policy+" | "+" | ".join(f"{v:.2f}" for v in values)+" |")
    lines += ["", "| Formula | Evo mean | LogReg mean | Difference, pp | Chosen C | Evo diagnostic validation | LogReg diagnostic validation |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy in POLICIES:
        m, v = macro[policy], validation[policy]
        lines.append(f"| {policy} | {m['evo_folded']:.3f} | {m['logreg_folded']:.3f} | {m['folded_delta']:+.3f} | {result['models'][policy]['chosen_C']:g} | {v['evo_directed']:.3f} | {v['logreg_directed']:.3f} |")
    lines += ["", "## Coefficients on the same feature definitions", "",
              "Here `φ` denotes the unweighted exported component. The original formula is approximately `Σ a_j φ_j`, with its original finite-precision output retained for evaluation. Logistic score is `b + Σ w_j φ_j`; sigmoid is unnecessary for AUROC. **The raw-input w values below are directly comparable in units with a.** They are not the standardized coefficients. Multiplying an entire score by a positive constant does not change AUROC, so absolute coefficient size alone is not importance.", ""]
    for policy in POLICIES:
        model = result["models"][policy]
        lines += [f"### {policy}", "", "| Feature φ | Evolution a | Logistic raw-input w | Evolution, standardized basis | Logistic, standardized basis |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in result["coefficients"]:
            if row["policy"] == policy:
                lines.append(f"| {row['feature_name']} | {row['evolution_raw']:.6g} | {row['logreg_raw']:.6g} | {row['evolution_standardized']:.6g} | {row['logreg_standardized']:.6g} |")
        lines += ["", f"Raw-input intercept **{model['intercept_raw']:.8g}**; standardized intercept **{model['intercept_standardized']:.8g}**."]
        rho = model["raw_coefficient_pearson"]
        lines += ["Pearson correlation between the raw coefficient vectors: **"+(f"{rho:.6f}" if rho is not None else "not interpreted for a two-entry vector")+"**.", ""]
    lines += ["## Feature definitions", ""]
    for policy in POLICIES:
        for row in result["coefficients"]:
            if row["policy"] == policy:
                lines.append(f"- **{row['feature_name']}**: {row['definition']}")
    lines += ["", "## Directed AUROC companion", "",
              "These retain the intended hallucination-score direction, without dataset-specific folding. Fit and validation are the same in both presentations.", "",
              "| Method | PopQA test | AVeriTeC | X-Fact | UHEAD | Cities | Companies | Common Claims | CounterFact | WildHallucinations | Mean nine |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy in POLICIES:
        for model in ("evo", "logreg"):
            values = [next(r[model+"_directed"] for r in source if r["policy"] == policy and r["dataset"] == name) for _, name in DATASETS]
            values += [macro[policy][model+"_directed"]]
            lines.append("| "+("Evo" if model == "evo" else "LogReg")+" / "+policy+" | "+" | ".join(f"{v:.2f}" for v in values)+" |")
    checks = result["reconstruction_checks"]
    lines += ["", "## Numerical and interpretation limits", "",
              "The saved feature pool consists of **full-minus-drop** vectors, preserving each frozen program's output rounding and guards. For this follow-up each vector is divided by its original effective coefficient to define φ. This is a controlled numerical approximation to extracting each native arithmetic term, not a bitwise decomposition. It does not alter frozen sources.", "",
              f"Across all training/validation and nine-dataset comparisons, maximum reconstruction score error is **{max(r['max_score_error'] for r in checks):.9g}**, and maximum AUROC discrepancy is **{max(r['absolute_auc_error_pp'] for r in checks):.9g} percentage points**. Both passed the predeclared limits of 0.001 score and 0.001 pp. Original Evo table cells use saved outputs, not this reconstruction.", "",
              "All four formula families were retained. No held-out score selected features, C, a model, a coefficient sign or a replacement formula. Nevertheless the candidates were discovered using validation information and the benchmark had already been viewed, so this is a **post-benchmark exploratory follow-up**, not independent confirmation. Outer validation is diagnostic; the grouped refit split differs from the original evolutionary split. Claim-grouping is not entity/relation separation. Test overlap masks are retained in the exported predictions.", "",
              "Coefficient correlations with only two features are uninformative; three-feature correlations are descriptive, not evidence of a general search/gradient distinction. Correlated predictors and regularization affect coefficients. AUROC does not measure calibration, and bootstrap/search-seed significance is not claimed here.", "",
              f"Runtime: **{result['elapsed_seconds']:.2f} CPU wall-clock seconds**, excluding subsequent independent validation and report packaging. Python {result['environment']['python']}, NumPy {result['environment']['numpy']}, scikit-learn {result['environment']['sklearn']}. This newly fitted local runtime is recorded separately from the historical GPU environment.", ""]
    (output/"TABLE3_REPORT.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT/"results/table3_20261022")
    args = ap.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT)
    plan_path = output/"PLAN.json"
    plan = json.loads(plan_path.read_text())
    if (output/"RESULTS.json").exists():
        raise FileExistsError("Refusing to refit a completed study")
    started = time.monotonic()
    for row in plan["inputs"]:
        assert sha(ROOT/row["file"]) == row["sha256"], row["file"]
    assert plan["original_effective_coefficients"] == WEIGHTS
    pool = json.loads((OLD/"screen_component_refit/component_pool.json").read_text())
    manifest = json.loads((OLD/"screen_component_ablations/manifest.json").read_text())
    with np.load(OLD/"screen_component_refit/training_features_and_predictions.npz", allow_pickle=False) as z:
        X = z["features"]
        y = z["labels"]
        train, validation = z["train_indices"], z["validation_indices"]
    with (ROOT/"data/claims/popqa/popqa_train_filtered_10shot.csv").open() as stream:
        claims = list(csv.DictReader(stream))
    groups = np.asarray([" ".join(row["claim"].casefold().split()) for row in claims])
    label_map = {"false": 0, "true": 1}
    csv_labels = np.array([label_map.get(row["is_hallu"].casefold(), -1) if row["is_hallu"].casefold() in label_map else int(float(row["is_hallu"])) for row in claims])
    np.testing.assert_array_equal(csv_labels, y)
    assert X.shape == (7007, 12) and (len(train), len(validation)) == (5606, 1401)
    assert len(set(train) & set(validation)) == 0
    assert np.array_equal(np.sort(np.r_[train,validation]), np.arange(len(y)))
    old_train, old_validation = train_test_split(np.arange(len(y)), test_size=.2, random_state=441, stratify=y)
    models, pipelines, columns, training, checks, coefficients, metrics = {}, {}, {}, {}, [], [], []
    training_payload = dict(labels=y, train_indices=train, validation_indices=validation, normalized_claim_groups=groups)
    for policy in POLICIES:
        columns[policy] = [i for i,c in enumerate(pool["components"]) if c["name"].startswith(policy+"/")]
        evo_weights = np.asarray(WEIGHTS[policy])
        features = X[:,columns[policy]] / evo_weights
        assert np.isfinite(features).all()
        entry = next(r for r in manifest["methods"] if r["policy"] == policy and r["role"] == "full")
        source = ROOT/entry["file"]
        assert sha(source) == entry["sha256"]
        ledger = ROOT/("out/logs/research20261021/uncertainty_quantification_next_prompt_"+policy+"_2107")
        evo = np.empty(len(y))
        evo[old_train] = np.load(ledger/("train_scores_"+entry["source_candidate_hash"]+".npy"),allow_pickle=False)
        evo[old_validation] = np.load(ledger/("val_scores_"+entry["source_candidate_hash"]+".npy"),allow_pickle=False)
        checks += [reconstruction_check(policy, "fit_train", y[train], evo[train], features[train]),
                   reconstruction_check(policy, "diagnostic_validation", y[validation], evo[validation], features[validation])]
        pipeline, fit_report = train_only_fit(features, y, groups, train, validation, plan["C_grid"])
        mean, scale = pipeline[0].mean_, pipeline[0].scale_
        beta, intercept = pipeline[-1].coef_[0], float(pipeline[-1].intercept_[0])
        raw_coef, raw_intercept = raw_parameters(mean, scale, beta, intercept)
        pred = ((features-mean)/scale) @ beta + intercept
        np.testing.assert_allclose(pred, pipeline.decision_function(features),rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(pred, features @ raw_coef + raw_intercept,rtol=1e-10,atol=1e-10)
        model = dict(**fit_report, feature_names=[pool["components"][i]["name"] for i in columns[policy]],
                     original_effective_coefficients=evo_weights.tolist(), scaler_mean=mean.tolist(),scaler_scale=scale.tolist(),
                     coefficients_standardized=beta.tolist(), coefficients_raw=raw_coef.tolist(),
                     intercept_standardized=intercept,intercept_raw=raw_intercept,
                     raw_coefficient_pearson=None if len(beta)<3 else float(np.corrcoef(evo_weights,raw_coef)[0,1]),
                     standardized_coefficient_cosine=float((evo_weights*scale)@beta/np.linalg.norm(evo_weights*scale)/np.linalg.norm(beta)),
                     frozen_evo_source_sha256=entry["sha256"])
        dump(output/(policy+"_weights.json"),model)
        models[policy], pipelines[policy] = model,pipeline
        for j,i in enumerate(columns[policy]):
            component = pool["components"][i]
            definition = next(r["components"][j]["definition"] for r in manifest["methods"] if r["policy"]==policy and r["role"]=="full")
            coefficients.append(dict(policy=policy,feature_name=component["name"],definition="Unweighted exported full-minus-drop component: "+definition+f"; divided by {evo_weights[j]:g}.",
                                     evolution_raw=float(evo_weights[j]),logreg_raw=float(raw_coef[j]),
                                     evolution_standardized=float(evo_weights[j]*scale[j]),logreg_standardized=float(beta[j]),
                                     evolution_positive_L1=float(evo_weights[j]/np.sum(np.abs(evo_weights))),
                                     logreg_positive_L1=float(raw_coef[j]/np.sum(np.abs(raw_coef))),
                                     scaler_mean=float(mean[j]),scaler_scale=float(scale[j])))
        metrics += [evaluate_rows(policy,"fit_train","PopQA fit",y[train],evo[train],pred[train]),
                    evaluate_rows(policy,"diagnostic_validation","PopQA diagnostic validation",y[validation],evo[validation],pred[validation])]
        training_payload.update({policy+"/features":features,policy+"/evo":evo,policy+"/logreg":pred})
        print(json.dumps(dict(stage="frozen_before_test_evaluation",policy=policy,C=model["chosen_C"],raw_coefficients=model["coefficients_raw"])),flush=True)
    np.savez_compressed(output/"training_predictions_and_features.npz",**training_payload)
    dump(output/"MODELS_FROZEN_BEFORE_TEST.json",dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         weights={p:sha(output/(p+"_weights.json")) for p in POLICIES},test_scoring_started=False))
    pred_dir=output/"source_nine";pred_dir.mkdir(exist_ok=False)
    counts={}
    for dataset, name in DATASETS:
        path=OLD/"screen_component_ablations/evaluations/source_nine"/(dataset+"_scores.npz")
        with np.load(path,allow_pickle=False) as saved:
            labels=saved["labels"];counts[name]=len(labels)
            data={k:saved[k] for k in ("labels","duplicate_disjoint_mask","claim_group_ids")}
            for policy in POLICIES:
                cs=[pool["components"][i] for i in columns[policy]]
                features=np.column_stack([saved[c["full"]]-saved[c["drop"]] for c in cs])/np.asarray(WEIGHTS[policy])
                evo=saved["new/screen_"+policy+"__full"]
                checks.append(reconstruction_check(policy,name,labels,evo,features))
                model=models[policy]
                pred=((features-np.asarray(model["scaler_mean"]))/np.asarray(model["scaler_scale"]))@np.asarray(model["coefficients_standardized"])+model["intercept_standardized"]
                np.testing.assert_allclose(pred,pipelines[policy].decision_function(features),rtol=1e-11,atol=1e-11)
                np.testing.assert_allclose(pred,features@np.asarray(model["coefficients_raw"])+model["intercept_raw"],rtol=1e-10,atol=1e-10)
                metrics.append(evaluate_rows(policy,"source_nine",name,labels,evo,pred))
                mask=data["duplicate_disjoint_mask"]
                metrics.append(evaluate_rows(policy,"strict_source_nine",name,labels[mask],evo[mask],pred[mask]))
                data.update({policy+"/features":features,policy+"/evo":evo,policy+"/logreg":pred})
        np.savez_compressed(pred_dir/(dataset+"_scores.npz"),**data)
        print(json.dumps(dict(stage="scored",dataset=name,rows=counts[name])),flush=True)
    assert sum(counts.values())==97488 and len(counts)==9
    for policy in POLICIES:
        for scope,name in [("source_nine","macro_nine"),("strict_source_nine","strict_macro_nine")]:
            rows=[r for r in metrics if r["policy"]==policy and r["scope"]==scope]
            assert len(rows)==9
            row=dict(policy=policy,scope=name,dataset="MACRO",samples=sum(r["samples"] for r in rows))
            row.update({k:float(np.mean([r[k] for r in rows])) for k in ["evo_directed","logreg_directed","directed_delta","evo_folded","logreg_folded","folded_delta"]})
            metrics.append(row)
    for row in plan["inputs"]:
        assert sha(ROOT/row["file"])==row["sha256"],row["file"]
    result=dict(status="complete_exploratory_per_formula_refit",created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                plan_sha256=sha(plan_path),script_sha256=sha(__file__),models=models,coefficients=coefficients,
                metrics=metrics,reconstruction_checks=checks,source_rows=counts,original_inputs_unchanged=True,
                pooled_head_unchanged=True,new_model_calls=False,gpu_used=False,
                environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
                elapsed_seconds=time.monotonic()-started)
    dump(output/"RESULTS.json",result)
    for name,rows in [("TABLE3.csv",metrics),("COEFFICIENTS.csv",coefficients)]:
        with (output/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    render_report(result,output)
    manifest={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.rglob("*") if p.is_file()}
    dump(output/"ARTIFACT_SHA256.json",manifest)
    print(json.dumps(dict(status=result["status"],elapsed_seconds=result["elapsed_seconds"],
                         summaries=[r for r in metrics if r["scope"]=="macro_nine"]),indent=2),flush=True)


if __name__=="__main__":
    main()
