"""Summarize existing frozen-screen ablation NPZs; never score or fit a model."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/overnight20261021/screen_component_ablations"
BASELINE = "original/SequenceLogProb"
ROW_KEYS = ("labels", "duplicate_disjoint_mask", "claim_group_ids")
POLICY_ORDER = ("original", "neutral", "diverse", "compositional")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def folded(values):
    return np.maximum(values, 100-values)


def ci(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(ci95_pp=np.percentile(finite, [2.5,97.5]).tolist() if len(finite) else None,
                finite_replicates=len(finite), requested_replicates=len(values))


def load_scope(folder, names, expected, errors):
    """Do not pool scopes or silently replace missing datasets with a subset."""
    scores, boot, files, failures = {}, {}, {}, []
    for name in names:
        path = folder / (name+"_scores.npz")
        if not path.exists():
            continue
        try:
            with np.load(path) as archive:
                arrays = {k:archive[k] for k in archive.files}
        except (OSError, ValueError, EOFError) as exc:
            failures.append(dict(dataset=name,status="possibly_copying",error=str(exc)))
            continue
        if any(k not in arrays for k in ROW_KEYS):
            errors.append(dict(scope=folder.name,dataset=name,error="Missing row identity fields"))
            continue
        n = len(arrays["labels"])
        if any(len(arrays[k]) != n for k in arrays):
            errors.append(dict(scope=folder.name,dataset=name,error="Prediction/identity row counts differ"))
            continue
        if not set(np.unique(arrays["labels"])) == {0,1}:
            errors.append(dict(scope=folder.name,dataset=name,error="Labels must contain both binary classes"))
            continue
        if any(not np.isfinite(arrays[k]).all() for k in expected if k in arrays):
            errors.append(dict(scope=folder.name,dataset=name,error="Nonfinite saved predictions"))
            continue
        scores[name] = arrays
        files[relative(path)] = sha(path)
        bpath = folder / (name+"_bootstrap.npz")
        if bpath.exists():
            try:
                with np.load(bpath) as archive:
                    boot[name] = {k:archive[k] for k in archive.files}
                files[relative(bpath)] = sha(bpath)
            except (OSError, ValueError, EOFError) as exc:
                failures.append(dict(dataset=name,status="bootstrap_possibly_copying",error=str(exc)))
        result_path = folder / (name+".json")
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                failures.extend(dict(dataset=name,method=key,error=message) for key,message in result.get("failed",{}).items() if key in expected)
            except (OSError,json.JSONDecodeError):
                failures.append(dict(dataset=name,status="result_json_possibly_copying"))
    return scores,boot,files,failures


def metric_pair(labels, full, other):
    a,b = 100*roc_auc_score(labels,full),100*roc_auc_score(labels,other)
    return dict(full_directed_auc_percent=float(a), other_directed_auc_percent=float(b),
                full_paper_auc_percent=float(max(a,100-a)),other_paper_auc_percent=float(max(b,100-b)),
                directed_delta_pp=float(a-b),paper_delta_pp=float(max(a,100-a)-max(b,100-b)))


def bootstrap_difference(boot, names, full, other, count=500):
    if not all(name in boot and all(key in boot[name] and len(boot[name][key]) == count for key in [full,other]) for name in names):
        return dict(status="pending_500_paired_claim_bootstraps")
    a = np.stack([boot[name][full] for name in names])*100
    b = np.stack([boot[name][other] for name in names])*100
    # An incomplete bootstrap draw is dropped as a whole, never averaged over
    # fewer datasets; the number of usable draws is explicitly recorded.
    valid = np.all(np.isfinite(a)&np.isfinite(b),axis=0)
    directed = np.full(count,np.nan)
    paper = np.full(count,np.nan)
    directed[valid] = np.mean(a[:,valid]-b[:,valid],axis=0)
    paper[valid] = np.mean(folded(a[:,valid])-folded(b[:,valid]),axis=0)
    return dict(status="complete",unit="claim",datasets=len(names),paired_by_replicate_index=True,
                directed=ci(directed),paper=ci(paper))


def compare_scope(scores, boot, names, full, other):
    rows = []
    for name in names:
        arrays = scores.get(name,{})
        if full in arrays and other in arrays:
            row = dict(dataset=name,rows=len(arrays["labels"]),**metric_pair(arrays["labels"],arrays[full],arrays[other]))
            row["paired_claim_bootstrap"] = bootstrap_difference(boot,[name],full,other)
            rows.append(row)
    result = dict(status="complete" if len(rows)==len(names) else "partial_pending_or_failed",
                  expected_datasets=len(names),datasets_scored=len(rows),missing_datasets=[n for n in names if n not in {r["dataset"] for r in rows}],per_dataset=rows)
    if len(rows)==len(names):
        result["macro"] = {key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key.endswith("_percent") or key.endswith("_pp")}
        result["paired_claim_bootstrap"] = bootstrap_difference(boot,names,full,other)
    return result


def primary_reference_map(manifest, registry, errors):
    result = {}
    for row in manifest["methods"]:
        if row["role"] != "full":
            continue
        matches = [r for r in registry.get("methods",[]) if r.get("role")=="primary" and r.get("phase")=="prompt_screen"
                   and r.get("policy")==row["policy"] and r.get("source_sha256")==row["original_source_sha256"]]
        if len(matches) != 1:
            errors.append(dict(policy=row["policy"],error="Expected exactly one source-primary identity with matching original SHA"))
        else:
            result[row["policy"]] = matches[0]["method"]
    return result


def reference_parity(folder, arrays, name, full, reference, source_sha, errors, scope):
    """Compare against every available prediction of the identical primary."""
    checks = []
    for path in sorted(folder.rglob(name+"_scores.npz")):
        try:
            with np.load(path) as archive:
                if reference not in archive:
                    continue
                meta_path = path.parent/"manifest.json"
                if not meta_path.exists():
                    checks.append(dict(file=relative(path),status="pending_reference_manifest"))
                    continue
                meta = json.loads(meta_path.read_text())
                declared_sha = meta.get("frozen_estimators",{}).get(reference,{}).get("sha256")
                if declared_sha != source_sha:
                    errors.append(dict(scope=scope,dataset=name,method=full,reference_file=relative(path),error="Reference source hash mismatch"))
                    continue
                equal = {key: key in archive and np.array_equal(arrays[key],archive[key]) for key in ROW_KEYS}
                prediction_equal = np.array_equal(arrays[full],archive[reference])
                baseline_equal = BASELINE in arrays and BASELINE in archive and np.array_equal(arrays[BASELINE],archive[BASELINE])
                passed = all(equal.values()) and prediction_equal and baseline_equal
                checks.append(dict(file=relative(path),sha256=sha(path),status="exact_match" if passed else "MISMATCH",
                                   row_identity_exact=equal,full_control_predictions_exact=prediction_equal,nll_predictions_exact=baseline_equal))
                if not passed:
                    errors.append(dict(scope=scope,dataset=name,method=full,error="Full control/reference prediction or row identity mismatch",file=relative(path)))
        except (OSError,ValueError,EOFError,json.JSONDecodeError) as exc:
            checks.append(dict(file=relative(path),status="pending_reference_copy",error=str(exc)))
    status = "exact_match" if checks and all(r["status"]=="exact_match" for r in checks) else "pending_reference" if not checks or not any(r["status"]=="MISMATCH" for r in checks) else "MISMATCH"
    return dict(status=status,references=checks)


def control_parity(manifest, primary, source, target, names9, names6, args, errors):
    rows = []
    for row in manifest["methods"]:
        if row["role"]!="full":
            continue
        for scope,scores,names,folder in [("source_nine",source,names9,args.source_reference),("target_six",target,names6,args.target_reference)]:
            checks = []
            for name in names:
                if name not in scores or row["method"] not in scores[name] or row["policy"] not in primary:
                    checks.append(dict(dataset=name,status="pending_ablation_or_primary"))
                else:
                    checks.append(dict(dataset=name,**reference_parity(folder,scores[name],name,row["method"],primary[row["policy"]],row["original_source_sha256"],errors,scope)))
            rows.append(dict(policy=row["policy"],scope=scope,expected_datasets=len(names),
                exact_matching_datasets=sum(r["status"]=="exact_match" for r in checks),
                status="exact_match" if all(r["status"]=="exact_match" for r in checks) else "pending_or_mismatch",per_dataset=checks))
    return rows


def transfer_method(source,target,source_boot,target_boot,names,method,errors):
    rows, a, b = [],[],[]
    for name in names:
        before,after=source.get(name,{}),target.get(name,{})
        if method not in before or method not in after:
            continue
        if any(not np.array_equal(before[key],after[key]) for key in ROW_KEYS):
            errors.append(dict(dataset=name,method=method,error="Source/target row identities differ"))
            continue
        pair=metric_pair(before["labels"],after[method],before[method])
        rows.append(dict(dataset=name,rows=len(before["labels"]),row_identity_verified=True,
            source_directed_auc_percent=pair["other_directed_auc_percent"],target_directed_auc_percent=pair["full_directed_auc_percent"],
            source_paper_auc_percent=pair["other_paper_auc_percent"],target_paper_auc_percent=pair["full_paper_auc_percent"],
            target_minus_source_directed_pp=pair["directed_delta_pp"],target_minus_source_paper_pp=pair["paper_delta_pp"]))
        if all(method in bag.get(name,{}) and len(bag[name][method])==500 for bag in [source_boot,target_boot]):
            a.append(source_boot[name][method]*100)
            b.append(target_boot[name][method]*100)
    result=dict(method=method,expected_matched_datasets=6,matched_datasets=len(rows),status="complete" if len(rows)==6 else "partial_pending_or_failed",per_dataset=rows)
    if len(rows)==6:
        result["macro"]={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key.endswith("_percent") or key.endswith("_pp")}
        if len(a)==6:
            a,b=np.asarray(a),np.asarray(b)
            valid=np.all(np.isfinite(a)&np.isfinite(b),axis=0)
            d,p=np.full(500,np.nan),np.full(500,np.nan)
            d[valid]=np.mean(b[:,valid]-a[:,valid],axis=0)
            p[valid]=np.mean(folded(b[:,valid])-folded(a[:,valid]),axis=0)
            result["paired_claim_bootstrap"]=dict(status="complete",unit="claim",datasets=6,directed=ci(d),paper=ci(p))
        else:
            result["paired_claim_bootstrap"]=dict(status="pending_500_paired_claim_bootstraps")
    return result


def validate_evaluation_manifest(folder, methods, errors):
    path=folder/"manifest.json"
    if not path.exists():
        return dict(status="pending")
    try:
        meta=json.loads(path.read_text())
    except json.JSONDecodeError:
        return dict(status="possibly_copying")
    expected={row["method"]:row["sha256"] for row in methods}
    actual={key:value["sha256"] for key,value in meta.get("frozen_estimators",{}).items() if key.startswith("new/")}
    flags=dict(exact_15_method_source_hashes=actual==expected,claim_bootstrap=meta.get("bootstrap_unit")=="claim",
               samples_500=meta.get("bootstrap_samples")==500,scoring_chunks_512=meta.get("scoring_chunk_size")==512,
               no_header_trim=meta.get("trimmed_header_suffix_tokens")==0,no_primary_seed_filter=meta.get("primary_only") is False)
    if not all(flags.values()):
        errors.append(dict(scope=folder.name,error="Ablation evaluator protocol mismatch",checks=flags))
    return dict(status="passed" if all(flags.values()) else "MISMATCH",file=relative(path),sha256=sha(path),checks=flags)


def write_csv(path, report):
    columns=["record_type","scope","policy","method","dropped_component","dataset","status","datasets_scored","expected_datasets",
             "directed_auc_percent","paper_auc_percent","directed_delta_pp","paper_delta_pp","directed_ci95_low_pp","directed_ci95_high_pp",
             "paper_ci95_low_pp","paper_ci95_high_pp","full_directed_auc_percent","other_directed_auc_percent","full_paper_auc_percent","other_paper_auc_percent",
             "source_directed_auc_percent","target_directed_auc_percent","source_paper_auc_percent","target_paper_auc_percent","target_minus_source_directed_pp","target_minus_source_paper_pp"]
    rows=[]
    for kind,key in [("method","methods"),("full_minus_drop","component_comparisons")]:
        for item in report[key]:
            base={k:item[k] for k in ["scope","policy","method","dropped_component","status","datasets_scored","expected_datasets"] if k in item}
            for values in ([dict(dataset="MACRO",**item["macro"],paired_claim_bootstrap=item.get("paired_claim_bootstrap",{}))] if "macro" in item else [])+item["per_dataset"]:
                row=dict(record_type=kind,**base,**{k:v for k,v in values.items() if k in columns})
                for measure in ["directed","paper"]:
                    limits=values.get("paired_claim_bootstrap",{}).get(measure,{}).get("ci95_pp")
                    if limits is not None:
                        row[measure+"_ci95_low_pp"],row[measure+"_ci95_high_pp"]=limits
                rows.append(row)
    for item in report["matched_six_model_transfer"]:
        for values in ([dict(dataset="MACRO",**item["macro"],paired_claim_bootstrap=item.get("paired_claim_bootstrap",{}))] if "macro" in item else [])+item["per_dataset"]:
            row=dict(record_type="target_minus_source",scope="matched_six",method=item["method"],status=item["status"],**{k:v for k,v in values.items() if k in columns})
            for measure in ["directed","paper"]:
                limits=values.get("paired_claim_bootstrap",{}).get(measure,{}).get("ci95_pp")
                if limits is not None:
                    row[measure+"_ci95_low_pp"],row[measure+"_ci95_high_pp"]=limits
            rows.append(row)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "pending" if value is None else f"{value:.3f}"


def interval(row,measure):
    limits=row.get("paired_claim_bootstrap",{}).get(measure,{}).get("ci95_pp")
    return "pending" if limits is None else f"[{limits[0]:.3f}, {limits[1]:.3f}]"


def write_markdown(path,report):
    lines=["# EvoHallu frozen-screen component ablations","",f"Status: **{report['status']}**; snapshot {report['created_utc']}.","",
        "All four source-validation screen winners and all eleven predetermined component drops are retained. Coefficients are fixed; no test-based variant selection or target fitting was performed. This is one writer seed per screen policy.","",
        "Source uses Llama3.1-8b-instruct features on nine datasets. Target uses Llama3.2-1B-Instruct features and the original shared tokenizer on exactly six datasets. Cross-model comparisons use source and target on those same six datasets. Labels are `is_hallu`.","",
        "Positive full-minus-drop means the retained component helps the frozen formula. Directed AUROC and paper AUROC are both shown; paper AUROC folds each dataset with `max(AUC,100-AUC)` before averaging. Confidence intervals use 500 shared claim-level bootstrap replicates and condition on the frozen program. No incomplete dataset subset is silently averaged.","",
        "| Cohort | Complete datasets with all 15 programs + NLL | Protocol |","| --- | ---: | --- |"]
    for scope,item in report["coverage"].items():
        lines.append(f"| {scope} | {item['fully_scored_datasets']}/{item['expected_datasets']} | {report['evaluator_protocols'][scope]['status']} |")
    lines += ["","| Policy | Exact source controls | Exact target controls |","| --- | ---: | ---: |"]
    for policy in POLICY_ORDER:
        checks={r["scope"]:r for r in report["full_control_parity"] if r["policy"]==policy}
        lines.append(f"| {policy} | {checks['source_nine']['exact_matching_datasets']}/9 | {checks['target_six']['exact_matching_datasets']}/6 |")
    lines += ["","Every full control is checked against existing primary predictions, source hashes, NLL predictions, labels, duplicate-disjoint masks, and claim-group IDs. A missing reference remains pending; any mismatch invalidates the report.","",
        "| Policy / dropped term | Source9 directed Δ [95% CI] | Source9 paper Δ [95% CI] | Target6 directed Δ [95% CI] | Target6 paper Δ [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for item in report["component_comparisons"]:
        if item["scope"]!="source_nine":
            continue
        twin=next(r for r in report["component_comparisons"] if r["scope"]=="target_six" and r["method"]==item["method"])
        values=[]
        for row in [item,twin]:
            for measure in ["directed","paper"]:
                values.append(fmt(row.get("macro",{}).get(measure+"_delta_pp"))+" "+interval(row,measure))
        lines.append("| "+item["policy"]+" / "+item["dropped_component"]+" | "+" | ".join(values)+" |")
    lines += ["","| Program | Source same-six paper AUROC | Target six paper AUROC | Target − source [95% CI] |","| --- | ---: | ---: | ---: |"]
    for row in report["matched_six_model_transfer"]:
        macro=row.get("macro",{})
        lines.append(f"| {row['method']} | {fmt(macro.get('source_paper_auc_percent'))} | {fmt(macro.get('target_paper_auc_percent'))} | {fmt(macro.get('target_minus_source_paper_pp'))} {interval(row,'paper')} |")
    lines += ["","Complete method/per-dataset scores, directed transfer differences, bootstrap usable counts, prediction hashes, and individual control checks are in JSON/CSV. All terms remain in fixed original/neutral/diverse/compositional source order, not sorted by test performance.","",
        "Limitations: final terms share token-probability inputs. Diverse's original short-sequence NLL fallback remains; preflight requires n≥5 for all caches. Original Pearson NaN behavior and all numerical guards remain unchanged. These component effects are exploratory and are not independent causal effects or a multi-seed policy comparison."]
    if report["errors"]:
        lines += ["","Integrity errors:",""]+["- `"+json.dumps(e,sort_keys=True)+"`" for e in report["errors"]]
    if report["estimator_or_copy_failures"]:
        lines += ["","Pending copies / estimator failures:",""]+["- `"+json.dumps(e,sort_keys=True)+"`" for e in report["estimator_or_copy_failures"]]
    path.write_text("\n".join(lines)+"\n")


def plot_results(directory,report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    rows=[r for r in report["component_comparisons"] if r["scope"]=="source_nine"]
    if not any("macro" in r for r in report["component_comparisons"]):
        return dict(status="pending_complete_component_comparison")
    fig,axes=plt.subplots(1,2,figsize=(12,6.8),sharey=True)
    colors={"source_nine":"#326eab","target_six":"#ba6735"}
    for ax,measure in zip(axes,["directed","paper"]):
        for i,row in enumerate(rows):
            for scope,offset in [("source_nine",-.13),("target_six",.13)]:
                item=next(r for r in report["component_comparisons"] if r["scope"]==scope and r["method"]==row["method"])
                point=item.get("macro",{}).get(measure+"_delta_pp")
                if point is None:
                    continue
                limits=item.get("paired_claim_bootstrap",{}).get(measure,{}).get("ci95_pp")
                y=i+offset
                if limits is not None:
                    ax.plot(limits,[y,y],color=colors[scope],lw=1.6)
                ax.plot(point,y,"o",color=colors[scope],ms=4,label=("Source 9 datasets" if scope=="source_nine" else "Target 6 datasets") if i==0 else None)
        ax.axvline(0,color="0.5",lw=.8)
        ax.grid(axis="x",alpha=.18)
        ax.set_title("Directed AUROC" if measure=="directed" else "Paper-folded AUROC")
        ax.set_xlabel("Full − component-drop macro AUROC (percentage points)")
    axes[0].set_yticks(range(len(rows)),[r["policy"]+" / "+r["dropped_component"].replace("_"," ") for r in rows])
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="upper center",ncol=2,bbox_to_anchor=(.65,.96),frameon=False)
    fig.suptitle("Frozen-screen component ablations · "+report["status"],y=.99)
    fig.text(.52,.015,"Fixed term order; 95% paired claim-bootstrap CIs; one writer seed/policy. Source9 and target6 are different scopes.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.91))
    paths=[]
    for suffix in ["png","pdf"]:
        path=directory/("component_ablation_effects."+suffix)
        fig.savefig(path,dpi=190)
        paths.append(relative(path))
    plt.close(fig)
    return dict(status="written",files=paths)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",type=Path,default=DEFAULT)
    ap.add_argument("--registry",type=Path,default=ROOT/"results/overnight20261021/frozen_manifest.json")
    ap.add_argument("--source-reference",type=Path,default=ROOT/"results/overnight20261021/evaluations")
    ap.add_argument("--target-reference",type=Path,default=ROOT/"results/overnight20261021/transfer_llama32_1b")
    ap.add_argument("--plot",action="store_true")
    args=ap.parse_args()
    os.chdir(ROOT)
    directory=args.directory.resolve()
    manifest=json.loads((directory/"manifest.json").read_text())
    registry=json.loads(args.registry.read_text())
    names9=[r["name"] for r in manifest["datasets"]]
    names6=[names9[i] for i in manifest["target_matched_indices"]]
    if len(names9)!=9 or len(names6)!=6 or len(manifest["methods"])!=15:
        raise ValueError("Unexpected prespecified design")
    errors=[]
    for row in manifest["methods"]:
        if sha(ROOT/row["file"])!=row["sha256"]:
            errors.append(dict(method=row["method"],error="Exported program SHA changed"))
    expected=[r["method"] for r in manifest["methods"]]+[BASELINE]
    source,sb,source_files,sf=load_scope(directory/"evaluations/source_nine",names9,expected,errors)
    target,tb,target_files,tf=load_scope(directory/"evaluations/target_six",names6,expected,errors)
    protocols={scope:validate_evaluation_manifest(directory/"evaluations"/scope,manifest["methods"],errors) for scope in ["source_nine","target_six"]}
    coverage={}
    for scope,scores,names,boot in [("source_nine",source,names9,sb),("target_six",target,names6,tb)]:
        complete=[n for n in names if n in scores and all(key in scores[n] for key in expected)]
        missing={n:[key for key in expected if key not in scores.get(n,{})] for n in names if n not in complete}
        boot_complete=[n for n in names if n in boot and all(key in boot[n] and len(boot[n][key])==500 for key in expected)]
        coverage[scope]=dict(expected_datasets=len(names),fully_scored_datasets=len(complete),complete_datasets=complete,
            missing_methods_by_dataset=missing,complete_500_bootstrap_datasets=len(boot_complete),all_predictions_complete=len(complete)==len(names),
            all_bootstraps_complete=len(boot_complete)==len(names))
    primary=primary_reference_map(manifest,registry,errors)
    parity=control_parity(manifest,primary,source,target,names9,names6,args,errors)
    comparisons=[]
    methods=[]
    for scope,scores,boot,names in [("source_nine",source,sb,names9),("target_six",target,tb,names6)]:
        for row in manifest["methods"]:
            comparison=compare_scope(scores,boot,names,row["method"],BASELINE)
            comparison.update(scope=scope,policy=row["policy"],method=row["method"],role=row["role"],comparison="method_minus_nll")
            if "macro" in comparison:
                comparison["macro"]["directed_auc_percent"]=comparison["macro"]["full_directed_auc_percent"]
                comparison["macro"]["paper_auc_percent"]=comparison["macro"]["full_paper_auc_percent"]
            methods.append(comparison)
            if row["role"]!="full":
                full=next(r["method"] for r in manifest["methods"] if r["policy"]==row["policy"] and r["role"]=="full")
                result=compare_scope(scores,boot,names,full,row["method"])
                result.update(scope=scope,policy=row["policy"],method=row["method"],full_method=full,dropped_component=row["dropped_component"],comparison="full_minus_drop")
                comparisons.append(result)
    transfer=[transfer_method(source,target,sb,tb,names6,method,errors) for method in expected]
    complete=(all(r["all_predictions_complete"] and r["all_bootstraps_complete"] for r in coverage.values())
              and all(r["status"]=="exact_match" for r in parity) and all(r["status"]=="passed" for r in protocols.values()))
    report=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status="INVALID_integrity_mismatch" if errors else "complete" if complete else "partial_pending_or_failed",
        manifest_sha256=sha(directory/"manifest.json"),report_script_sha256=sha(__file__),source_feature_model="Llama3.1-8b-instruct",target_feature_model="Llama3.2-1B-Instruct",
        labels="is_hallu",selection="All four source-validation-selected screen winners; all eleven predetermined fixed component drops",target_fitting=False,test_based_variant_selection=False,
        macro_rule="Only all prescribed datasets; per-dataset fold max(AUC,100-AUC) before equal-weight macro",bootstrap_rule="500 deterministic shared claim bootstrap replicates; only fully finite matched draws enter intervals",
        source_prespecified_nine=names9,target_and_matched_source_prespecified_six=names6,coverage=coverage,evaluator_protocols=protocols,
        full_control_parity=parity,methods=methods,component_comparisons=comparisons,matched_six_model_transfer=transfer,
        prediction_and_bootstrap_sha256={**source_files,**target_files},errors=errors,estimator_or_copy_failures=sf+tf)
    if args.plot and not errors:
        report["plot"]=plot_results(directory,report)
    dump(directory/"ABLATION_REPORT.json",report)
    write_csv(directory/"ABLATION_REPORT.csv",report)
    write_markdown(directory/"ABLATION_REPORT.md",report)
    print(json.dumps(dict(status=report["status"],coverage={key:dict(fully_scored_datasets=value["fully_scored_datasets"],expected_datasets=value["expected_datasets"],complete_500_bootstrap_datasets=value["complete_500_bootstrap_datasets"]) for key,value in coverage.items()},integrity_errors=len(errors),output=relative(directory/"ABLATION_REPORT.md"))))


if __name__=="__main__":
    main()
