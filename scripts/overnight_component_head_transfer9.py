"""Add the three missing target datasets to the ALREADY frozen component head.

Default prepares an immutable plan only. --execute waits for pinned completion
and the passed 427-row numerical guard, audits caches on CPU, copies the six
old results unchanged, and scores only indices 6/7/8. All outputs are additive;
the original six-dataset report, weights, helper and primary registry are never
modified. There is no fitting, GPU inference, extraction, or pod shutdown.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import overnight_component_refit_report as report
from scripts import overnight_transfer_eval as provenance


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/overnight20261021"
HEAD=OUT/"screen_component_refit"
DEFAULT=HEAD/"transfer_nine"
DEADLINE=dt.datetime(2026,10,22,2,6,tzinfo=dt.timezone.utc)
METHOD=report.METHOD
BASELINE=report.BASELINE
NEW_NAMES=["geometry__counterfact_true_false","popqa__popqa_test_filtered_10shot","x-fact__test_filtered_10shot"]


def now():
    return dt.datetime.now(dt.timezone.utc)


def read(path):
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_size>4*1024**2:
        raise ValueError("Unsafe receipt path or excessive receipt size")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    report.dump(path,value)


def rel(path,root=None):
    root=ROOT if root is None else root
    return str(Path(path).resolve().relative_to(root.resolve()))


def copy_once(source,destination,expected):
    if source.is_symlink() or report.sha(source)!=expected:
        raise ValueError("Frozen source file changed: "+str(source))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Refusing a symlinked destination")
    if destination.exists():
        if report.sha(destination)!=expected:
            raise ValueError("Refusing to overwrite a distinct target-nine artifact")
        return
    temporary=destination.with_suffix(destination.suffix+".copy.part")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("Unreviewed partial copy exists")
    shutil.copyfile(source,temporary)
    if report.sha(temporary)!=expected:
        raise ValueError("Copy SHA mismatch")
    os.link(temporary,destination)
    temporary.unlink()


def assert_unchanged(plan,root=None):
    root=ROOT if root is None else root
    for name,digest in plan["protected_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Protected path escapes the repository")
        path=root/name
        if path.is_symlink() or not path.is_file() or report.sha(path)!=digest:
            raise ValueError("Protected source/six-result file changed: "+name)


def completion_ready(receipt):
    if not receipt or receipt.get("status")!="complete_three_missing_caches":
        return False
    if not provenance.completion_numerical_reference_passed(receipt):
        raise ValueError("Complete target receipt lacks the passed full 427-row current-Torch numerical preflight")
    return True


class Bounds:
    def __init__(self,deadline=DEADLINE,cancel_paths=(),stop=None,clock=now):
        self.deadline,self.cancel_paths,self.stop,self.clock=deadline,list(cancel_paths),stop or threading.Event(),clock

    def check(self):
        if self.stop.is_set() or any(p.exists() for p in self.cancel_paths):
            raise InterruptedError("Transfer-nine cancelled manually")
        if self.clock()>=self.deadline:
            raise TimeoutError("Transfer-nine CPU deadline reached")


def terminate(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=2)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        child.wait()


def run_bounded(command,log,bounds):
    bounds.check()
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,"PYTHONPATH":"src:.","CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
        try:
            while child.poll() is None:
                bounds.check()
                time.sleep(.2)
            bounds.check()
            if child.returncode:
                raise RuntimeError(f"CPU child exited {child.returncode}; see {log}")
        finally:
            terminate(child)


def evaluator_command(plan,index,directory,python=None):
    if index not in [6,7,8]:
        raise ValueError("Only the three prespecified additions may be scored")
    return [python or sys.executable,"scripts/evaluate_cached_claims.py","--mode","evaluate",
        "--dataset-index",str(index),"--data-root","data/claims","--cache-dir","data/restored/mod1-transfer-cache",
        "--output",rel(directory),"--cpu-threads","1","--bootstrap","500","--bootstrap-unit","claim",
        "--model-id","unsloth/Llama-3.2-1B-Instruct","--tokenizer-id","/workspace/mod1-models/llama31-tokenizer",
        "--frozen-glob",plan["frozen_wrapper"],"--method-prefix",METHOD]


def prepare(directory):
    errors=[]
    fit,wrapper=report.audit_fit(HEAD,errors)
    if fit["status"]!="passed" or errors:
        raise ValueError("Frozen head fit/source provenance must pass before planning")
    previous=read(HEAD/"REFIT_REPORT.json")
    if not previous or previous.get("status")!="complete_exploratory_report":
        raise ValueError("Complete and preserve the original source-nine/target-six head report first")
    manifest_path=OUT/"screen_component_ablations/manifest.json"
    design=json.loads(manifest_path.read_text())
    datasets=design["datasets"]
    if len(datasets)!=9 or [r["index"] for r in datasets]!=list(range(9)) or [r["name"] for r in datasets[6:]]!=NEW_NAMES:
        raise ValueError("Unexpected frozen nine-dataset order")
    pool=json.loads((HEAD/"component_pool.json").read_text())
    protected=[HEAD/name for name in ["fit_report.json","weights.json","component_pool.json","frozen/screen_component_logistic/frozen_estimator.py"]]
    protected += [ROOT/r["file"] for r in pool["programs"]]
    protected += [ROOT/"scripts"/name for name in ["overnight_component_model.py","fit_screen_component_logistic.py","evaluate_cached_claims.py"]]
    protected += [manifest_path]
    for scope,rows in [("source_nine",datasets),("target_six",datasets[:6])]:
        protected.append(HEAD/"evaluations"/scope/"manifest.json")
        for row in rows:
            protected += [HEAD/"evaluations"/scope/(row["name"]+suffix) for suffix in ["_scores.npz","_bootstrap.npz",".json"]]
    protected += [HEAD/name for name in ["REFIT_REPORT.md","REFIT_REPORT.json","REFIT_REPORT.csv"]]
    fingerprints={rel(path):report.sha(path) for path in protected}
    dataset_sha={str(Path(row["csv"]).relative_to("data/claims")):design["data_sha256"][row["csv"]] for row in datasets}
    for scope,target in [("source_nine",False),("target_six",True)]:
        if report.validate_protocol(HEAD/"evaluations"/scope,wrapper,dataset_sha,target)["status"]!="passed":
            raise ValueError("Original head evaluation protocol does not match the pinned head")
    payload=dict(study="prospective_frozen_component_head_three_dataset_extension",status="prepared_not_executed",
        head_method=METHOD,baseline=BASELINE,frozen_wrapper=rel(HEAD/"frozen/screen_component_logistic/frozen_estimator.py"),
        frozen_wrapper_sha256=wrapper,head_directory=rel(HEAD),new_output_directory=rel(directory),datasets=datasets,
        reused_target_dataset_indices=list(range(6)),new_target_dataset_indices=[6,7,8],source_comparison_dataset_indices=list(range(9)),
        protected_sha256=fingerprints,data_sha256=design["data_sha256"],data_protocol_sha256=dataset_sha,
        fitting=False,source_or_target_reselection=False,original_six_report_preserved=True,primary_registry_modified=False,
        required_numerical_reference_rows=427,completion_receipt=rel(OUT/"remaining_transfer_completion.json"),
        target_cache_directory="data/restored/mod1-transfer-cache",source_cache_directory="data/restored/mod1-eval-cache",
        deadline_utc=DEADLINE.isoformat(),bootstrap_samples=500,bootstrap_unit="claim",scoring_chunk_size=512)
    existing=read(directory/"PLAN.json")
    if existing is not None and existing!=payload:
        raise ValueError("Existing transfer-nine plan has different frozen artifacts; refusing replacement")
    if existing is None:
        dump(directory/"PLAN.json",payload)
    commands=[dict(dataset_index=i,dataset=datasets[i]["name"],command=evaluator_command(payload,i,directory/"computed"/f"index_{i:02d}",".venv/bin/python")) for i in [6,7,8]]
    dump(directory/"commands.json",dict(executed=False,execute_requires_passed_completion_and_cache_audit=True,commands=commands))
    (directory/"DESIGN.md").write_text("""# Optional full-nine transfer of the frozen component head

This adds exactly CounterFact, PopQA and X-Fact (sorted CSV indices 6/7/8) to the already completed target-six study. All six original dataset JSONs, score NPZs, bootstrap NPZs and evaluator manifest are copied byte-identically into a new `target_nine/` folder; their originals and `REFIT_REPORT.*` remain untouched. Existing source-nine head predictions are reused. No coefficient, scaler, feature source, wrapper, helper or primary registry is changed or refitted. The four previously selected programs, eleven additive components and extra float64 NLL retain their earlier selection caveat. This is a prospective coverage addition, not a new selected control or an independent confirmatory head.

Default execution only prepares this plan. Explicit `--execute` runs on the pod, waits in bounded polls for `complete_three_missing_caches` and the main transfer protocol's passed 427-row current-Torch gross-drift guard, then validates all nine raw target-cache SHA256s, architecture/float32 metadata, pinned model/tokenizer provenance, CSV claims/labels and source/target token IDs using the existing full-nine auditor. A bounded CPU-only child performs this audit; no extraction/GPU inference can occur. Only three cached-evaluation commands run, each with `--mode evaluate`, fixed 512-row chunks and 500 claim bootstraps. The original six files are checked again after work.

`TRANSFER9_REPORT.*` contains source9 and target9 directed/folded macro AUROC, duplicate-disjoint point macros, paired fixed-head claim-bootstrap CIs and matched-nine transfer. Missing datasets never enter a shorter macro. Canonical NLL predictions, row identities and bootstrap draws are compared with the main full-nine transfer cohort when available; otherwise this separate parity check stays explicitly pending. Any mismatch invalidates the result. The original six-dataset report is never rewritten. Re-run `--report-only` after the main transfer finishes to refresh pending external-reference checks without rescoring.

The runner is bounded by the earlier active-work timeout and **02:06 UTC on 22 October 2026**, checks manual cancellation, and never stops the pod. `--execute --once` returns immediately if completion is not ready; ordinary `--execute` waits. `--audit-only` is an internal CPU validation child mode. Inspect PLAN/commands before execution. Original frozen-file checksums must still match. The 427-row guard permits the predeclared small numerical drift; it is not bitwise equivalence of old/new feature-extraction runtimes.
""")
    return payload


def reuse_six(plan,directory):
    assert_unchanged(plan)
    target=directory/"target_nine"
    entries=[]
    for row in plan["datasets"][:6]:
        for suffix in ["_scores.npz","_bootstrap.npz",".json"]:
            source=HEAD/"evaluations/target_six"/(row["name"]+suffix)
            fingerprint=plan["protected_sha256"][rel(source)]
            copy_once(source,target/source.name,fingerprint)
            entries.append(dict(source=rel(source),copy=rel(target/source.name),sha256=fingerprint,byte_identical=True))
    source=HEAD/"evaluations/target_six/manifest.json"
    copy_once(source,target/"manifest.json",plan["protected_sha256"][rel(source)])
    assert_unchanged(plan)
    dump(directory/"six_reuse_receipt.json",dict(status="six_original_results_copied_byte_identically",files=entries,original_six_unchanged=True))


def pin_cache_identity(directory,completion_sha,rows):
    identity=dict(completion_receipt_sha256=completion_sha,
                  target_raw_sha256={row["dataset"]:row["raw_sha256"] for row in rows})
    path=directory/"TARGET_CACHE_COHORT.json"
    old=read(path)
    if old is not None and old!=identity:
        raise ValueError("Target cache cohort changed; never reuse old head outputs for new raw caches")
    if old is None:
        dump(path,identity)
    return identity


def cache_audit(plan,directory):
    assert_unchanged(plan)
    completion_path=ROOT/plan["completion_receipt"]
    completion=read(completion_path)
    if not completion_ready(completion):
        raise ValueError("No complete, numerically verified target cache receipt")
    completion_sha=report.sha(completion_path)
    names=[r["name"] for r in plan["datasets"]]
    args=SimpleNamespace(completion_receipt=completion_path,directory=OUT,coverage="nine",
        cache_dir=ROOT/plan["target_cache_directory"],source_cache_dir=ROOT/plan["source_cache_directory"])
    files=[ROOT/r["csv"] for r in plan["datasets"]]
    if any(not (args.cache_dir/(name+".pt")).is_file() for name in names):
        raise ValueError("All nine completed target caches must exist")
    if any(not (args.source_cache_dir/(name+".pt")).is_file() for name in names):
        raise ValueError("All nine source caches must remain available for exact source/target input-ID verification")
    for path,row in zip(files,plan["datasets"]):
        if report.sha(path)!=plan["data_sha256"][row["csv"]]:
            raise ValueError("Frozen CSV changed")
    audit=provenance.audit_target_caches(args,names,files,{name:i for i,name in enumerate(names)})
    if audit.get("status")!="passed" or audit.get("numerical_reference_preflight_verified") is not True:
        raise ValueError("Main full-nine cache and numerical provenance audit failed")
    if report.sha(completion_path)!=completion_sha:
        raise ValueError("Completion receipt changed during the cache audit")
    pin_cache_identity(directory,completion_sha,audit["per_dataset"])
    assert_unchanged(plan)
    dump(directory/"cache_audit.json",dict(status="passed",completion_receipt_sha256=completion_sha,audit=audit))


def load_checked(folder,names,identities,protocol,errors):
    scores,boots,receipts={}, {}, []
    for name in names:
        point=folder/(name+"_scores.npz")
        boot=folder/(name+"_bootstrap.npz")
        result=folder/(name+".json")
        if not all(p.exists() for p in [point,boot,result]):
            receipts.append(dict(dataset=name,status="pending_predictions_or_bootstrap"));continue
        try:
            z,b=report.load_npz(point),report.load_npz(boot)
            if protocol["status"]!="passed" or not report.validate_rows(z,identities[name]):
                raise ValueError("Evaluation protocol or CSV labels/mask/groups mismatch")
            n=len(z["labels"])
            if any(key not in z or z[key].shape!=(n,) or not np.isfinite(z[key]).all() or key not in b or b[key].shape!=(500,) for key in [METHOD,BASELINE]):
                raise ValueError("Finite head/NLL predictions or 500 bootstraps missing")
            if any(np.any((b[key][np.isfinite(b[key])]<0)|(b[key][np.isfinite(b[key])]>1)) for key in [METHOD,BASELINE]):
                raise ValueError("Saved bootstrap AUROC is outside [0,1]")
            saved=json.loads(result.read_text())
            if any(key in saved.get("failed",{}) for key in [METHOD,BASELINE]):
                raise ValueError("Saved estimator failure")
            scores[name],boots[name]=z,b
            receipts.append(dict(dataset=name,status="verified",prediction_sha256=report.sha(point),bootstrap_sha256=report.sha(boot)))
        except (ValueError,OSError,report.PendingCopy) as exc:
            errors.append(dict(dataset=name,error=str(exc)))
            receipts.append(dict(dataset=name,status="MISMATCH_or_invalid",error=str(exc)))
    return scores,boots,receipts


def main_nine_parity(directory,names,scores,boots):
    audits=[]
    for name in names:
        matches=[]
        for protocol_path in sorted((directory/"transfer_llama32_1b").glob("*/transfer_protocol.json")):
            protocol=read(protocol_path)
            cache=(protocol or {}).get("target_cache_audit") or {}
            if not protocol or protocol.get("coverage")!="nine" or set(protocol.get("prespecified_transfer_subset",[]))!=set(names) or cache.get("status")!="passed" or cache.get("numerical_reference_preflight_verified") is not True:
                continue
            for sub in ["new","supervised"]:
                point=protocol_path.parent/sub/(name+"_scores.npz")
                boot=protocol_path.parent/sub/(name+"_bootstrap.npz")
                if not point.exists() or name not in scores:
                    continue
                try:
                    z=report.load_npz(point)
                except report.PendingCopy:
                    matches.append(dict(file=str(point),status="pending_reference_copy"));continue
                if BASELINE not in z:
                    continue
                same=all(key in z and np.array_equal(z[key],scores[name][key]) for key in report.ROW_KEYS)
                same=same and np.array_equal(z[BASELINE],scores[name][BASELINE])
                boot_match=None
                if boot.exists() and name in boots:
                    try:b=report.load_npz(boot)
                    except report.PendingCopy:
                        matches.append(dict(file=str(point),status="pending_reference_copy"));continue
                    if BASELINE in b:
                        boot_match=np.array_equal(b[BASELINE],boots[name][BASELINE],equal_nan=True)
                matches.append(dict(file=str(point),status="MISMATCH" if not same or boot_match is False else "exact_match" if boot_match is True else "pending_reference_bootstrap",row_and_nll_predictions_exact=same,nll_bootstrap_exact=boot_match))
        audits.append(dict(dataset=name,status="MISMATCH" if any(r["status"]=="MISMATCH" for r in matches) else "exact_match" if matches and all(r["status"]=="exact_match" for r in matches) else "pending_main_full_nine_reference",references=matches))
    return dict(status="MISMATCH" if any(r["status"]=="MISMATCH" for r in audits) else "exact_match" if all(r["status"]=="exact_match" for r in audits) else "pending_main_full_nine_reference",per_dataset=audits)


def summarize(plan,directory):
    assert_unchanged(plan)
    names=[row["name"] for row in plan["datasets"]]
    training=ROOT/"data/claims/popqa/popqa_train_filtered_10shot.csv"
    if report.sha(training)!=plan["data_sha256"][rel(training)]:
        raise ValueError("Training claim-identity CSV changed")
    claims={" ".join(str(x).casefold().split()) for x in pd.read_csv(training,usecols=["claim"]).claim}
    identities={}
    for row in plan["datasets"]:
        path=ROOT/row["csv"]
        if report.sha(path)!=plan["data_sha256"][row["csv"]]:
            raise ValueError("Frozen CSV changed")
        identities[row["name"]],_=report.row_identities(path,claims)
    errors=[]
    protocols={key:report.validate_protocol(folder,plan["frozen_wrapper_sha256"],plan["data_protocol_sha256"],target) for key,folder,target in [
        ("source_nine",HEAD/"evaluations/source_nine",False),("target_nine",directory/"target_nine",True)]}
    source,sb,srows=load_checked(HEAD/"evaluations/source_nine",names,identities,protocols["source_nine"],errors)
    target,tb,trows=load_checked(directory/"target_nine",names,identities,protocols["target_nine"],errors)
    audit=read(directory/"cache_audit.json")
    completion_path=ROOT/plan["completion_receipt"]
    pinned=read(directory/"TARGET_CACHE_COHORT.json")
    cache_ready=bool(audit and pinned and completion_path.exists() and completion_ready(read(completion_path))
        and audit.get("status")=="passed" and audit.get("audit",{}).get("numerical_reference_preflight_verified") is True
        and audit.get("completion_receipt_sha256")==pinned.get("completion_receipt_sha256")==report.sha(completion_path)
        and pinned.get("target_raw_sha256")=={r["dataset"]:r["raw_sha256"] for r in audit["audit"]["per_dataset"]})
    parity=main_nine_parity(OUT,names,target,tb)
    if parity["status"]=="MISMATCH":
        errors.append(dict(error="NLL/reference rows or bootstrap mismatch against main full-nine transfer"))
    trusted_target=target if cache_ready and parity["status"]!="MISMATCH" else {}
    scopes={"source_nine":report.scope_summary(source,sb,names),"target_nine":report.scope_summary(trusted_target,tb,names)}
    try:matched=report.transfer_summary(source,trusted_target,sb,tb,names)
    except ValueError as exc:
        errors.append(dict(error=str(exc)));matched=dict(status="MISMATCH")
    complete=cache_ready and all(r["verified_datasets"]==9 and r.get("paired_claim_bootstrap",{}).get("status")=="complete" for r in scopes.values())
    status="INVALID_integrity_error" if errors else "complete_exploratory_nine" if complete and parity["status"]=="exact_match" else "complete_predictions_main_reference_pending" if complete else "partial_pending"
    result=dict(created_utc=now().isoformat(),status=status,method=METHOD,baseline=BASELINE,exploratory=True,fitting_or_reselection=False,
        original_six_report_preserved=True,plan_sha256=report.sha(directory/"PLAN.json"),prespecified_datasets=names,source_feature_model="Llama3.1-8b-instruct",
        target_feature_model="Llama3.2-1B-Instruct",shared_source_tokenizer=True,cache_audit=audit,main_full_nine_nll_parity=parity,
        evaluator_protocols=protocols,row_audits={"source_nine":srows,"target_nine":trows},scopes=scopes,matched_transfer_nine=matched,
        bootstrap_rule="500 shared deterministic claim draws; per-dataset folding before all-nine mean; no partial-dataset averaging",
        limits=["The feature pool was previously selected using source train/validation labels; coefficient CV does not nest feature discovery","Benchmarks were viewed before the original head follow-up; exploratory fixed-head transfer","427-row gross-drift guard is not bitwise backend equivalence","Duplicate-disjoint diagnostics remove normalized-claim overlap only"],errors=errors)
    dump(directory/"TRANSFER9_REPORT.json",result)
    lines=["# Frozen component head: optional full-nine transfer","",f"Status: **{status}**. The original six-dataset report is preserved.","",
        "Same previously frozen 12-column head, coefficients, scaler, wrapper and helper; no target refit or selection. Six target results are byte-identical copies; only the three prespecified additions were scored. Source and target comparisons both use all nine datasets. Labels remain `is_hallu`.","",
        "| Scope | Verified datasets | Directed head / NLL | Paper head / NLL | Paper gain [95% CI] | Strict-disjoint paper head / NLL |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt=lambda x:"pending" if x is None else f"{x:.3f}"
    for scope,item in scopes.items():
        m=item.get("macro",{});c=item.get("paired_claim_bootstrap",{}).get("gain_paper",{}).get("ci95_pp")
        lines.append(f"| {scope} | {item['verified_datasets']}/9 | {fmt(m.get('head_directed_auc_percent'))} / {fmt(m.get('nll_directed_auc_percent'))} | {fmt(m.get('head_paper_auc_percent'))} / {fmt(m.get('nll_paper_auc_percent'))} | {fmt(m.get('paper_gain_pp'))} "+("pending" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]")+f" | {fmt(m.get('strict_head_paper_auc_percent'))} / {fmt(m.get('strict_nll_paper_auc_percent'))} |")
    lines += ["",f"Main full-nine NLL prediction/bootstrap parity: **{parity['status']}**. Missing external references remain pending, never inferred.","",
        "| Matched target − source | Directed delta [95% CI] | Paper delta [95% CI] |","| --- | ---: | ---: |"]
    for kind in ["head","nll"]:
        cells=[]
        for metric in ["directed","paper"]:
            point=matched.get("macro",{}).get(f"target_minus_source_{kind}_{metric}_auc_percent")
            c=matched.get("paired_claim_bootstrap",{}).get(kind+"_"+metric,{}).get("ci95_pp")
            cells.append(fmt(point)+(" pending" if c is None else f" [{c[0]:.3f}, {c[1]:.3f}]"))
        lines.append("| "+kind+" | "+" | ".join(cells)+" |")
    lines += ["","Paper AUROC folds every method/dataset as max(AUC,100−AUC) before equal weighting. Intervals condition on fixed selected programs, weights and these datasets. They exclude feature-discovery/search uncertainty; direction folding uses dataset labels. The feature pool and prior outer validation were selection-informed. This result is exploratory and is not matched evidence of beating the paper/Claude."]
    if errors:
        lines += ["","Integrity errors:"]+["- "+str(error) for error in errors]
    (directory/"TRANSFER9_REPORT.md").write_text("\n".join(lines)+"\n")
    with (directory/"TRANSFER9_REPORT.csv").open("w",newline="") as stream:
        fields=["scope","dataset","head_directed_auc_percent","nll_directed_auc_percent","directed_gain_pp","head_paper_auc_percent","nll_paper_auc_percent","paper_gain_pp","strict_head_paper_auc_percent","strict_nll_paper_auc_percent"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for scope,item in scopes.items():
            for row in item["per_dataset"]+([dict(dataset="MACRO",**item["macro"])] if "macro" in item else []):
                writer.writerow(dict(scope=scope,**row))
    assert_unchanged(plan)
    return result


def execute(plan,directory,args,bounds):
    while not completion_ready(read(ROOT/plan["completion_receipt"])):
        bounds.check()
        dump(directory/"execution_receipt.json",dict(status="waiting_for_completed_numeric_guard",updated_utc=now().isoformat(),deadline_utc=bounds.deadline.isoformat()))
        if args.once:
            return
        until=time.monotonic()+min(args.poll_seconds,max(0,(bounds.deadline-now()).total_seconds()))
        while time.monotonic()<until:
            bounds.check();time.sleep(min(.25,max(0,until-time.monotonic())))
    bounds.check();assert_unchanged(plan)
    active=Bounds(min(bounds.deadline,now()+dt.timedelta(seconds=args.max_seconds)),bounds.cancel_paths,bounds.stop)
    run_bounded([sys.executable,"scripts/overnight_component_head_transfer9.py","--execute","--audit-only","--directory",rel(directory),"--deadline-utc",active.deadline.isoformat()],directory/"cache_audit.log",active)
    reuse_six(plan,directory)
    def one(index):
        active.check();assert_unchanged(plan)
        name=plan["datasets"][index]["name"]
        final=directory/"target_nine"
        if all((final/(name+suffix)).exists() for suffix in ["_scores.npz","_bootstrap.npz",".json"]):
            return dict(dataset=name,status="reused_existing_target_nine_addition")
        work=directory/"computed"/f"index_{index:02d}"
        run_bounded(evaluator_command(plan,index,work),directory/(f"dataset_{index:02d}.log"),active)
        if report.sha(work/"manifest.json")!=plan["protected_sha256"][rel(HEAD/"evaluations/target_six/manifest.json")]:
            raise ValueError("New evaluator manifest differs from the original six-result protocol")
        for suffix in ["_scores.npz","_bootstrap.npz",".json"]:
            source=work/(name+suffix)
            copy_once(source,final/source.name,report.sha(source))
        assert_unchanged(plan)
        return dict(dataset=name,status="scored_and_published")
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(one,index):index for index in [6,7,8]}
        for future in concurrent.futures.as_completed(pending):
            try:jobs.append(future.result())
            except Exception as exc:jobs.append(dict(dataset_index=pending[future],status="failed_or_bounded",error=type(exc).__name__+": "+str(exc)))
            dump(directory/"execution_receipt.json",dict(status="three_additions_running_or_finished",jobs=jobs,deadline_utc=active.deadline.isoformat(),original_six_unchanged=True))
    active.check()
    reuse_six(plan,directory)
    result=summarize(plan,directory)
    dump(directory/"execution_receipt.json",dict(status=result["status"],jobs=jobs,finished_utc=now().isoformat(),original_six_unchanged=True))


def parse_deadline(value):
    parsed=dt.datetime.fromisoformat(value.replace("Z","+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset()!=dt.timedelta(0) or parsed>DEADLINE:
        raise argparse.ArgumentTypeError("Use UTC no later than 2026-10-22T02:06:00Z")
    return parsed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",type=Path,default=DEFAULT)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--report-only",action="store_true")
    ap.add_argument("--audit-only",action="store_true",help=argparse.SUPPRESS)
    ap.add_argument("--once",action="store_true")
    ap.add_argument("--workers",type=int,choices=[1,2],default=2)
    ap.add_argument("--poll-seconds",type=float,default=20)
    ap.add_argument("--max-seconds",type=float,default=1200)
    ap.add_argument("--deadline-utc",type=parse_deadline,default=DEADLINE)
    args=ap.parse_args()
    if not 1<=args.poll_seconds<=30 or not 1<=args.max_seconds<=1800:
        ap.error("Poll must be1..30s; active work must be1..1800s")
    directory=args.directory.resolve();directory.relative_to(ROOT)
    if directory==HEAD or directory==OUT or directory==ROOT or any(directory.is_relative_to(HEAD/name) for name in ["evaluations","frozen","components"]):
        ap.error("Choose a distinct additive output subdirectory")
    if args.audit_only and not args.execute:
        ap.error("Internal cache audit requires explicit --execute")
    if args.execute and ROOT.resolve()!=Path("/workspace/mod1").resolve():
        ap.error("Execution is restricted to the authorized pod checkout /workspace/mod1")
    plan=read(directory/"PLAN.json") or prepare(directory)
    assert_unchanged(plan)
    if args.report_only:
        result=summarize(plan,directory);print(json.dumps(dict(status=result["status"],new_scoring_calls=0)));return
    if not args.execute:
        print(json.dumps(dict(status="prepared_not_executed",new_dataset_indices=[6,7,8],reuse_original_six=True,completion_ready=completion_ready(read(ROOT/plan["completion_receipt"])),output=str(directory),gpu_calls=0)));return
    bounds=Bounds(args.deadline_utc,[OUT/name for name in ["watchdog.cancel","component_head_transfer9.cancel","manual.stop","manual_stop"]])
    bounds.check()
    if args.audit_only:
        cache_audit(plan,directory);return
    with (directory/"runner.lock").open("a") as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status="another_transfer9_runner_is_active")));return
        handlers={sig:signal.getsignal(sig) for sig in [signal.SIGTERM,signal.SIGINT]}
        for sig in handlers:signal.signal(sig,lambda signum,frame:bounds.stop.set())
        try:execute(plan,directory,args,bounds)
        except (TimeoutError,InterruptedError) as exc:
            dump(directory/"execution_receipt.json",dict(status="bounded_or_cancelled",error=str(exc),finished_utc=now().isoformat(),no_gpu_or_pod_stop=True))
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=="__main__":
    main()
