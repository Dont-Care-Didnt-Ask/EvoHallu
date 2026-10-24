"""Read-only report for the single exploratory frozen screen-component head.

Loads saved predictions/bootstrap arrays only; never imports or calls a detector,
fits a head, opens raw feature caches, or changes primary experiment groups.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/overnight20261021/screen_component_refit"
METHOD = "new/screen_component_logistic"
BASELINE = "original/SequenceLogProb"
ROW_KEYS = ("labels", "claim_group_ids", "duplicate_disjoint_mask")
COUNT = 500


class PendingCopy(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, payload):
    temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def folded(values):
    return np.maximum(values,100-values)


def interval(values):
    a = np.asarray(values)
    finite = a[np.isfinite(a)]
    return dict(ci95_pp=np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None,
                finite_replicates=len(finite),requested_replicates=len(a))


def load_npz(path):
    try:
        with np.load(path) as archive:
            return {key:archive[key] for key in archive.files}
    except (OSError,ValueError,EOFError,zipfile.BadZipFile) as exc:
        raise PendingCopy(str(exc)) from exc


def row_identities(csv_path, training_claims):
    frame = pd.read_csv(csv_path,usecols=["claim","is_hallu"])
    claims = [" ".join(str(claim).casefold().split()) for claim in frame.claim]
    groups,_ = pd.factorize(np.asarray(claims,dtype=object),sort=False)
    return dict(labels=frame.is_hallu.to_numpy(dtype=int),claim_group_ids=groups,
                duplicate_disjoint_mask=np.asarray([claim not in training_claims for claim in claims])), dict(
                    dataset_csv_sha256=sha(csv_path),ordered_normalized_claims_labels_sha256=hashlib.sha256(
                        json.dumps(list(zip(claims,frame.is_hallu.astype(int).tolist())),ensure_ascii=False,separators=(",",":")).encode()).hexdigest())


def validate_rows(arrays, expected):
    return all(key in arrays and np.array_equal(arrays[key],expected[key]) for key in ROW_KEYS)


def baseline_parity(reference_folder, name, arrays):
    checks = []
    for path in sorted(reference_folder.rglob(name+"_scores.npz")):
        try:
            z = load_npz(path)
            if BASELINE not in z:
                continue
            same_rows = all(key in z and np.array_equal(z[key],arrays[key]) for key in ROW_KEYS)
            same_nll = np.array_equal(z[BASELINE],arrays[BASELINE])
            checks.append(dict(file=str(path),sha256=sha(path),row_identity_exact=same_rows,
                               canonical_nll_predictions_exact=same_nll,status="exact_match" if same_rows and same_nll else "MISMATCH"))
        except (OSError,ValueError,EOFError,PendingCopy) as exc:
            checks.append(dict(file=str(path),status="possibly_copying",error=str(exc)))
    status = "exact_match" if checks and all(c["status"]=="exact_match" for c in checks) else "MISMATCH" if any(c["status"]=="MISMATCH" for c in checks) else "pending_reference"
    return dict(status=status,references=checks)


def validate_protocol(folder, wrapper_sha, dataset_sha, target=False):
    path = folder/"manifest.json"
    if not path.exists():
        return dict(status="pending_manifest")
    try:
        meta = json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return dict(status="possibly_copying")
    frozen = {name:row["sha256"] for name,row in meta.get("frozen_estimators",{}).items() if name.startswith("new/")}
    checks = dict(single_frozen_wrapper=frozen=={METHOD:wrapper_sha},
                  bootstrap_500_claims=meta.get("bootstrap_samples")==COUNT and meta.get("bootstrap_unit")=="claim",
                  chunk_512=meta.get("scoring_chunk_size")==512,no_trim=meta.get("trimmed_header_suffix_tokens")==0,
                  no_primary_filter=meta.get("primary_only") is False,method_prefix=meta.get("method_prefix")==METHOD,
                  all_csv_hashes_match=meta.get("data")==dataset_sha,forward_float32=meta.get("dtype")=="float32",
                  shared_target_tokenizer=not target or str(meta.get("tokenizer","")).endswith("llama31-tokenizer"),
                  feature_model=(meta.get("feature_model")=="unsloth/Llama-3.2-1B-Instruct" if target else str(meta.get("feature_model","")).endswith("llama31")))
    return dict(status="passed" if all(checks.values()) else "MISMATCH",file=str(path),sha256=sha(path),checks=checks)


def load_scope(folder,names,identities,reference,protocol,errors):
    scores,boot,rows,files = {},{},{},{}
    for name in names:
        path = folder/(name+"_scores.npz")
        if not path.exists():
            rows[name] = dict(dataset=name,status="pending_predictions")
            continue
        try:
            z = load_npz(path)
            if not validate_rows(z,identities[name]):
                raise ValueError("CSV labels, claim-group IDs or duplicate-disjoint mask differ")
            n = len(z["labels"])
            if any(key not in z or z[key].shape!=(n,) or not np.isfinite(z[key]).all() for key in [METHOD,BASELINE]):
                raise ValueError("Required finite head/NLL predictions missing or wrong shape")
            parity = baseline_parity(reference,name,z)
            if parity["status"]=="MISMATCH":
                raise ValueError("Canonical NLL reference prediction/row identity mismatch")
            rows[name] = dict(dataset=name,rows=n,csv_row_fields_exact=True,canonical_nll_reference=parity,
                              status="verified" if parity["status"]=="exact_match" and protocol["status"]=="passed" else "pending_reference_or_protocol")
            files[str(path)] = sha(path)
            # No unverifiable row/protocol is allowed to enter any macro.
            if rows[name]["status"]=="verified":
                scores[name] = z
            bpath = folder/(name+"_bootstrap.npz")
            if bpath.exists():
                b = load_npz(bpath)
                if all(key in b and b[key].shape==(COUNT,) for key in [METHOD,BASELINE]):
                    if any(np.any((b[key][np.isfinite(b[key])]<0)|(b[key][np.isfinite(b[key])]>1)) for key in [METHOD,BASELINE]):
                        raise ValueError("Saved bootstrap AUC outside [0,1]")
                    boot[name] = b
                    files[str(bpath)] = sha(bpath)
                else:
                    rows[name]["bootstrap_status"] = "pending_or_invalid_500_replicates"
            result_path = folder/(name+".json")
            if result_path.exists():
                result = json.loads(result_path.read_text())
                failures = {key:value for key,value in result.get("failed",{}).items() if key in [METHOD,BASELINE]}
                if failures:
                    raise ValueError("Saved estimator failure: "+json.dumps(failures))
        except (OSError,EOFError,json.JSONDecodeError,PendingCopy) as exc:
            rows[name] = dict(dataset=name,status="possibly_copying",error=str(exc))
            scores.pop(name,None)
            boot.pop(name,None)
        except ValueError as exc:
            rows[name] = dict(dataset=name,status="MISMATCH_or_invalid",error=str(exc))
            errors.append(dict(scope=folder.name,dataset=name,error=str(exc)))
            scores.pop(name,None)
            boot.pop(name,None)
    return scores,boot,list(rows.values()),files


def scope_summary(scores,boots,names):
    rows = []
    for name in names:
        if name not in scores:
            continue
        z = scores[name]
        a,b = 100*roc_auc_score(z["labels"],z[METHOD]),100*roc_auc_score(z["labels"],z[BASELINE])
        strict = z["duplicate_disjoint_mask"].astype(bool)
        row = dict(dataset=name,rows=len(z["labels"]),head_directed_auc_percent=float(a),nll_directed_auc_percent=float(b),
                   head_paper_auc_percent=float(max(a,100-a)),nll_paper_auc_percent=float(max(b,100-b)),
                   directed_gain_pp=float(a-b),paper_gain_pp=float(max(a,100-a)-max(b,100-b)),strict_rows=int(strict.sum()))
        if len(np.unique(z["labels"][strict]))==2:
            sa,sb = 100*roc_auc_score(z["labels"][strict],z[METHOD][strict]),100*roc_auc_score(z["labels"][strict],z[BASELINE][strict])
            row.update(strict_head_directed_auc_percent=float(sa),strict_nll_directed_auc_percent=float(sb),
                       strict_head_paper_auc_percent=float(max(sa,100-sa)),strict_nll_paper_auc_percent=float(max(sb,100-sb)))
        row["paired_claim_bootstrap"] = bootstrap_scope(boots,[name])
        rows.append(row)
    result = dict(status="complete_predictions" if len(rows)==len(names) else "partial_pending",expected_datasets=len(names),
                  verified_datasets=len(rows),missing_datasets=[n for n in names if n not in scores],per_dataset=rows)
    if len(rows)==len(names):
        fields = [key for key in rows[0] if key.endswith("_percent") or key.endswith("_pp")]
        result["macro"] = {key:float(np.mean([row[key] for row in rows])) for key in fields if all(key in row for row in rows)}
        result["paired_claim_bootstrap"] = bootstrap_scope(boots,names)
    return result


def bootstrap_scope(boots,names):
    if not all(name in boots and all(key in boots[name] and boots[name][key].shape==(COUNT,) for key in [METHOD,BASELINE]) for name in names):
        return dict(status="pending_500_paired_claim_bootstraps")
    a = 100*np.stack([boots[name][METHOD] for name in names])
    b = 100*np.stack([boots[name][BASELINE] for name in names])
    valid = np.all(np.isfinite(a)&np.isfinite(b),axis=0)
    out = dict(status="complete",unit="claim",datasets=len(names),paired_by_replicate_index=True)
    for measure,head,baseline in [("directed",a,b),("paper",folded(a),folded(b))]:
        for label,values in [("head",head),("nll",baseline),("gain",head-baseline)]:
            draws = np.full(COUNT,np.nan)
            draws[valid] = np.mean(values[:,valid],axis=0)
            out[label+"_"+measure] = interval(draws)
    return out


def transfer_summary(source,target,sb,tb,names):
    rows = []
    for name in names:
        if name not in source or name not in target:
            continue
        if any(not np.array_equal(source[name][key],target[name][key]) for key in ROW_KEYS):
            raise ValueError("Matched source/target row identity mismatch: "+name)
        a = scope_summary({name:source[name]}, {}, [name])["per_dataset"][0]
        b = scope_summary({name:target[name]}, {}, [name])["per_dataset"][0]
        rows.append(dict(dataset=name,rows=a["rows"],row_identity_verified=True,
                         **{f"target_minus_source_{key}":b[key]-a[key] for key in ["head_directed_auc_percent","head_paper_auc_percent","nll_directed_auc_percent","nll_paper_auc_percent","directed_gain_pp","paper_gain_pp"]}))
        rows[-1]["paired_claim_bootstrap"] = bootstrap_transfer(sb,tb,[name])
    result = dict(status="complete_predictions" if len(rows)==len(names) else "partial_pending",expected_datasets=len(names),matched_datasets=len(rows),per_dataset=rows)
    if len(rows)==len(names):
        result["macro"] = {key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key.startswith("target_minus_source_")}
        result["paired_claim_bootstrap"] = bootstrap_transfer(sb,tb,names)
    return result


def bootstrap_transfer(sb,tb,names):
    if not all(name in bag and all(key in bag[name] and bag[name][key].shape==(COUNT,) for key in [METHOD,BASELINE]) for bag in [sb,tb] for name in names):
        return dict(status="pending_500_paired_claim_bootstraps")
    a,b,c,d = [100*np.stack([bag[name][method] for name in names]) for bag,method in [(sb,METHOD),(tb,METHOD),(sb,BASELINE),(tb,BASELINE)]]
    valid = np.all(np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&np.isfinite(d),axis=0)
    boot = dict(status="complete",unit="claim",paired_by_replicate_index=True)
    for measure,f in [("directed",lambda x:x),("paper",folded)]:
        for label,values in [("head",f(b)-f(a)),("nll",f(d)-f(c)),("gain_change",(f(b)-f(d))-(f(a)-f(c)))]:
            draws=np.full(COUNT,np.nan)
            draws[valid]=np.mean(values[:,valid],axis=0)
            boot[label+"_"+measure]=interval(draws)
    return boot


def audit_fit(directory,errors):
    paths = {key:directory/name for key,name in [
        ("fit","fit_report.json"),("weights","weights.json"),("pool","component_pool.json"),
        ("wrapper","frozen/screen_component_logistic/frozen_estimator.py")]}
    if not all(path.exists() for path in paths.values()):
        return dict(status="pending_fit_artifacts",missing=[str(path) for path in paths.values() if not path.exists()]),None
    try:
        fit,weights,pool = [json.loads(paths[key].read_text()) for key in ["fit","weights","pool"]]
        checks = dict(pool_hash=sha(paths["pool"])==weights["pool_sha256"]==fit["pool_sha256"],
                      weights_hash=sha(paths["weights"])==fit["weights_sha256"],wrapper_hash=sha(paths["wrapper"])==fit["wrapper_sha256"],
                      module_hash=sha(ROOT/"scripts/overnight_component_model.py")==weights["inference_module_sha256"]==fit["inference_module_sha256"],
                      fitter_hash=sha(ROOT/"scripts/fit_screen_component_logistic.py")==fit["fitter_sha256"],
                      all_fifteen_sources=len(pool["programs"])==15 and all(sha(ROOT/item["file"])==item["sha256"] for item in pool["programs"]),
                      all_twelve_columns=len(weights["feature_names"])==12 and weights["feature_names"]==fit["feature_names"]==[item["name"] for item in pool["components"]]+["float64_total_nll"],
                      exact_grouped_split=fit["fit_samples"]==5606 and fit["validation_samples"]==1401 and fit["split_strategy"]=="claim_group" and fit["split_seed"]==441,
                      fit_only_scaler=fit["scaler_fit_samples"]==5606,grouped_inner_cv=len(fit["inner_folds"])==3 and all(fold["shared_normalized_claims"]==0 for fold in fit["inner_folds"]),
                      fixed_grid=fit["C_grid"]==[.01,.1,1.,10.,100.],source_fit_only=weights["source_fit_only"] is True and fit["target_refitting"] is False and fit["test_data_read"] is False,
                      diagnostic_validation=fit["outer_validation_is_independent"] is False and fit["validation_informed_feature_pool"] is True,
                      label_order=weights["classes"]==[0,1],serialization=fit["serialized_prediction_parity"] is True)
        checks["pinned_source_training_cache"] = weights["training_raw_sha256"]==fit["training_raw_sha256"]=="8d34da6f189e8bfe67c0bca3e1101607f5b0887d5f1de41b4f373ef08f2f5ced"
        if not all(checks.values()):
            errors.append(dict(error="Frozen fit provenance mismatch",checks=checks))
        return dict(status="passed" if all(checks.values()) else "MISMATCH",checks=checks,
                    source_fitting_report=fit,feature_pool_selection_caveat="All four source programs were validation-selected; outer validation is diagnostic, not independent.",
                    artifact_sha256={str(path):sha(path) for path in paths.values()}),fit["wrapper_sha256"]
    except (OSError,ValueError,KeyError) as exc:
        errors.append(dict(error="Fit artifact audit failed: "+str(exc)))
        return dict(status="pending_or_invalid",error=str(exc)),None


def markdown(path,report):
    fmt = lambda x:"pending" if x is None else f"{x:.3f}"
    lines=["# Frozen screen-component logistic refit", "",f"Status: **{report['status']}**. One exploratory fitted head; no primary search group is changed.","",
           "All eleven full-minus-drop components from all four screen winners plus `float64_total_nll` were retained. Coefficients use the source-only grouped 5,606/1,401 split, three-fold fit-only CV and five C values; the scaler stays inside CV. Source programs were already validation-selected, so outer validation is diagnostic. Benchmark results had already been viewed: this follow-up is exploratory.","",
           "| Scope | Verified datasets | Head directed | NLL directed | Gain | Head paper | NLL paper | Gain |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,item in report["scopes"].items():
        m=item.get("macro",{})
        lines.append("| "+name+f" | {item['verified_datasets']}/{item['expected_datasets']} | "+" | ".join(fmt(m.get(key)) for key in ["head_directed_auc_percent","nll_directed_auc_percent","directed_gain_pp","head_paper_auc_percent","nll_paper_auc_percent","paper_gain_pp"])+" |")
    lines += ["","Paper AUROC folds each dataset using `max(AUC,100−AUC)` before the unweighted mean. The canonical NLL reference retains its original float16 accumulation; the extra fitted feature uses float64 accumulation. Six-dataset and nine-dataset means describe different scopes.","",
              "| Scope | Directed gain 95% CI | Paper gain 95% CI |","| --- | --- | --- |"]
    for name,item in report["scopes"].items():
        boot=item.get("paired_claim_bootstrap",{})
        cells=[]
        for measure in ["directed","paper"]:
            c=boot.get("gain_"+measure,{}).get("ci95_pp")
            cells.append("pending" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]")
        lines.append("| "+name+" | "+" | ".join(cells)+" |")
    transfer=report["matched_transfer"]
    lines += ["","Matched target minus source compares the same six datasets and rows. It never subtracts the source-nine average from target-six.","",
              "| Transfer quantity | Directed delta | Paper delta | Directed 95% CI | Paper 95% CI |","| --- | ---: | ---: | --- | --- |"]
    for label,key in [("Frozen head","head"),("Canonical NLL","nll"),("Head gain over NLL","gain_change")]:
        suffixes=[f"{key}_{measure}_auc_percent" if key!="gain_change" else f"{measure}_gain_pp" for measure in ["directed","paper"]]
        values=[fmt(transfer.get("macro",{}).get("target_minus_source_"+suffix)) for suffix in suffixes]
        cis=[]
        for measure in ["directed","paper"]:
            c=transfer.get("paired_claim_bootstrap",{}).get(key+"_"+measure,{}).get("ci95_pp")
            cis.append("pending" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]")
        lines.append("| "+label+" | "+" | ".join(values+cis)+" |")
    fit=report["fit_audit"].get("source_fitting_report",{})
    lines += ["",f"Source fit: C={fit.get('chosen_C','pending')}; training directed AUC={fmt(100*fit['train_auc'] if 'train_auc' in fit else None)}; diagnostic outer-validation AUC={fmt(100*fit['validation_auc'] if 'validation_auc' in fit else None)}.","",
              "Confidence intervals use the saved 500 shared claim-bootstrap replicates with fixed features and weights. They measure held-out claim uncertainty conditional on this selected head, not search-seed uncertainty. Only draws finite on every prescribed dataset enter macro intervals. No incomplete dataset subset is averaged. JSON/CSV retain per-dataset scores, strict duplicate-disjoint diagnostics, coverage, reference parity and artifact hashes."]
    if report["errors"]:
        lines += ["","Integrity errors:",""]+["- "+json.dumps(error,sort_keys=True) for error in report["errors"]]
    path.write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",type=Path,default=DEFAULT)
    ap.add_argument("--design-manifest",type=Path,default=ROOT/"results/overnight20261021/screen_component_ablations/manifest.json")
    ap.add_argument("--source-reference",type=Path,default=ROOT/"results/overnight20261021/evaluations")
    ap.add_argument("--target-reference",type=Path,default=ROOT/"results/overnight20261021/transfer_llama32_1b")
    args=ap.parse_args()
    manifest=json.loads(args.design_manifest.read_text())
    datasets=manifest["datasets"]
    names9=[row["name"] for row in datasets]
    names6=[names9[i] for i in manifest["target_matched_indices"]]
    if len(names9)!=9 or manifest["target_matched_indices"]!=list(range(6)):
        raise ValueError("Unexpected frozen source-nine/target-six design")
    training=ROOT/"data/claims/popqa/popqa_train_filtered_10shot.csv"
    if sha(training)!=manifest["data_sha256"][str(training.relative_to(ROOT))]:
        raise ValueError("Training claim-identity CSV changed")
    training_claims={" ".join(str(x).casefold().split()) for x in pd.read_csv(training,usecols=["claim"]).claim}
    identities,identity_receipts={},{}
    for row in datasets:
        path=ROOT/row["csv"]
        if sha(path)!=manifest["data_sha256"][row["csv"]]:
            raise ValueError("Frozen benchmark CSV changed: "+row["csv"])
        identities[row["name"]],identity_receipts[row["name"]]=row_identities(path,training_claims)
    errors=[]
    fit,wrapper_sha=audit_fit(args.directory,errors)
    dataset_sha={str(Path(row["csv"]).relative_to("data/claims")):manifest["data_sha256"][row["csv"]] for row in datasets}
    protocols={key:validate_protocol(args.directory/"evaluations"/key,wrapper_sha,dataset_sha,target=key=="target_six") for key in ["source_nine","target_six"]}
    for key,item in protocols.items():
        if item["status"]=="MISMATCH":errors.append(dict(scope=key,error="Evaluator protocol mismatch",checks=item["checks"]))
    source,sb,source_rows,source_files=load_scope(args.directory/"evaluations/source_nine",names9,identities,args.source_reference,protocols["source_nine"],errors)
    target,tb,target_rows,target_files=load_scope(args.directory/"evaluations/target_six",names6,identities,args.target_reference,protocols["target_six"],errors)
    scopes={"source_nine":scope_summary(source,sb,names9),"matched_source_six":scope_summary(source,sb,names6),"target_six":scope_summary(target,tb,names6)}
    try:
        transfer=transfer_summary(source,target,sb,tb,names6)
    except ValueError as exc:
        errors.append(dict(error=str(exc)))
        transfer=dict(status="MISMATCH")
    complete=fit["status"]=="passed" and all(item["status"]=="complete_predictions" and item.get("paired_claim_bootstrap",{}).get("status")=="complete" for item in scopes.values()) and transfer.get("paired_claim_bootstrap",{}).get("status")=="complete"
    report=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status="INVALID_integrity_error" if errors else "complete_exploratory_report" if complete else "partial_pending",
                method=METHOD,baseline=BASELINE,fitted_head_count=1,primary_registry_modified=False,real_model_or_gpu_calls=0,
                fit_audit=fit,prespecified_source_nine=names9,prespecified_target_and_matched_source_six=names6,
                dataset_row_identity=identity_receipts,evaluator_protocols=protocols,row_and_baseline_audits={"source_nine":source_rows,"target_six":target_rows},
                scopes=scopes,matched_transfer=transfer,prediction_and_bootstrap_sha256={**source_files,**target_files},
                bootstrap_rule="500 saved deterministic claim-level replicates; shared seed 441+int(SHA256(dataset_name)[:8],16), matched rows/group IDs, same replicate indices across methods/models; fold per dataset per draw, then equal-weight mean; no incomplete-dataset means",
                limitations=["Validation-informed source program pool: outer validation is diagnostic","Previously viewed benchmark outcomes: exploratory study","Fixed weights/source programs; claim bootstrap does not include search/selection uncertainty","Canonical SequenceLogProb and float64_total_nll are numerically distinct"],errors=errors)
    dump(args.directory/"REFIT_REPORT.json",report)
    markdown(args.directory/"REFIT_REPORT.md",report)
    with (args.directory/"REFIT_REPORT.csv").open("w",newline="") as stream:
        fields=["scope","dataset","head_directed_auc_percent","nll_directed_auc_percent","directed_gain_pp","head_paper_auc_percent","nll_paper_auc_percent","paper_gain_pp"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for scope,item in scopes.items():
            for row in item["per_dataset"]+([dict(dataset="MACRO",**item["macro"])] if "macro" in item else []):writer.writerow(dict(scope=scope,**row))
    print(json.dumps(dict(status=report["status"],source9_verified=len(source),target6_verified=len(target),errors=len(errors),output=str(args.directory/"REFIT_REPORT.md"))))


if __name__=="__main__":
    main()
