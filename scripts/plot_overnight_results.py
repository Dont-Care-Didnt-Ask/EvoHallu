"""Plot complete frozen-program results without choosing a best test seed.

Writes standalone PNG/PDF figures and a plotting receipt only. Source report,
transfer reports, manifests and prediction files are never modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "original/SequenceLogProb"
POLICIES = ("original", "neutral", "diverse", "compositional")
PANELS = ("screen", "adaptive", "larger_writer")
PHASE_ORDER = {"prompt_screen":0, "evolution_screen":1, "compositional_control":1,
               "feedback_control":2, "sampler_control":3, "interface_control":4}
COLORS = {"original":"#596579", "neutral":"#347bb4", "diverse":"#33956c", "compositional":"#d18a32"}


def read_json(path, inputs):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def panel(identity, reference_writer):
    if identity["phase"] == "prompt_screen":
        return "screen"
    if identity["writer_model"] != reference_writer or "confirmation" in identity["phase"]:
        return "larger_writer"
    return "adaptive"


def group_key(identity):
    return (identity["phase"], identity["policy"], identity["writer_model"], identity["planned_requests"])


def sort_key(key):
    phase,policy,writer,budget = key
    return (PHASE_ORDER.get(phase,4),POLICIES.index(policy) if policy in POLICIES else 99,phase,policy,writer,budget)


def display_label(key, actual, expected):
    phase,policy,writer,budget = key
    name = policy
    if phase in {"writer_confirmation", "writer_generalization"}:
        name = writer.split("/")[-1] + " · " + policy
    elif phase == "feedback_control":
        name += " · fixed examples"
    elif phase == "sampler_control":
        name += " · rank sampler"
    elif phase == "interface_control":
        name += " · interface contract"
    elif phase not in {"prompt_screen","evolution_screen","compositional_control"} and "confirmation" not in phase:
        name += " · " + phase
    return f"{name}   n={actual}/{expected}"


def expected_groups(manifest):
    result = defaultdict(set)
    for job in manifest.get("jobs",[]):
        writer = job.get("writer_model",manifest.get("writer","unknown"))
        result[(job.get("analysis_phase",job["phase"]),job["prompt_policy"],writer,job["requests"])].add(job["seed"])
    return result


def panel_title(name, keys):
    writers = sorted({key[2].split("/")[-1] for key in keys})
    budgets = sorted({key[3] for key in keys})
    budget = "/".join(str(x) for x in budgets)
    title = {"screen":"Frozen-context screen", "adaptive":"Search and separate controls",
             "larger_writer":"Larger writer confirmation"}[name]
    return title + " · " + ", ".join(writers) + f" · {budget} requests per run"


def axes_for(groups, reference_writer):
    panels = defaultdict(list)
    for key in groups:
        identity=dict(phase=key[0],policy=key[1],writer_model=key[2])
        panels[panel(identity,reference_writer)].append(key)
    names = [name for name in PANELS if panels[name]]
    heights = [max(1.9,.47*len(panels[name])+1) for name in names]
    fig, axes = plt.subplots(len(names),1,figsize=(10.1,sum(heights)+1.45),squeeze=False,
        gridspec_kw={"height_ratios":heights},layout="constrained")
    return fig,[(name,axes[i,0],sorted(panels[name],key=sort_key)) for i,name in enumerate(names)]


def limits(values):
    low,high=min(values),max(values)
    margin=max(1.1,(high-low)*.13)
    return max(0,low-margin),min(100,high+margin)


def save(fig,output,stem):
    paths=[]
    for suffix in ("png","pdf"):
        path=output/(stem+"."+suffix)
        fig.savefig(path,dpi=240)
        paths.append(str(path))
    plt.close(fig)
    return paths


def primary_budget_filter(rows):
    """Match the primary-report rule: only explicitly verified full budgets."""
    primary=[row for row in rows if row.get("identity") and row["identity"].get("role")=="primary"]
    eligible=[row for row in primary if row["identity"].get("full_budget") is True]
    ignored=[dict(method=row["method"],job_id=row["identity"].get("job_id"),
        phase=row["identity"].get("phase"),policy=row["identity"].get("policy"),
        seed=row["identity"].get("seed"),full_budget=row["identity"].get("full_budget"),
        requests=row["identity"].get("requests"),planned_requests=row["identity"].get("planned_requests"),
        reason="partial_or_unverified_planned_request_budget") for row in primary if row["identity"].get("full_budget") is not True]
    return eligible,ignored


def source_plot(report,manifest,args,receipt):
    metric=args.metric+"_macro_auc_percent"
    selected,ignored=primary_budget_filter(report.get("methods",[]))
    available=[row for row in selected if row.get("status")=="complete" and metric in row and
               row.get("datasets_completed")==9 and np.isfinite(row[metric])]
    if not available:
        receipt["source"]={"status":"skipped_no_complete_nine_dataset_primary_results",
                           "ignored_partial_or_unverified_primary_identities":ignored}
        return
    groups=defaultdict(list)
    for row in available:
        groups[group_key(row["identity"])].append(row)
    expected=expected_groups(manifest)
    reference=manifest.get("writer","Qwen/Qwen3.5-9B")
    baseline=next((row[metric] for row in report.get("methods",[]) if row["method"]==BASELINE and row.get("status")=="complete" and metric in row),None)
    values=[row[metric] for row in available]
    for rows in groups.values():
        data=np.asarray([r[metric] for r in rows]);deviation=float(np.std(data,ddof=1)) if len(data)>1 else 0.
        values += [float(data.mean()-deviation),float(data.mean()+deviation)]
    if baseline is not None:values.append(baseline)
    fig,panels=axes_for(groups,reference)
    details=[]
    for name,ax,keys in panels:
        labels=[]
        for y,key in enumerate(keys):
            rows=sorted(groups[key],key=lambda row:row["identity"]["seed"])
            planned=expected.get(key) or {row["identity"]["seed"] for row in rows}
            labels.append(display_label(key,len(rows),len(planned)))
            seed_order=sorted(planned | {row["identity"]["seed"] for row in rows})
            offsets=dict(zip(seed_order,np.linspace(-.15,.15,len(seed_order)))) if len(seed_order)>1 else {seed_order[0]:-.10}
            color=COLORS.get(key[1],"#777777")
            data=np.asarray([row[metric] for row in rows])
            for row in rows:
                identity=row["identity"]
                ax.scatter(row[metric],y+offsets[identity["seed"]],s=35,color=color,
                    marker="o",zorder=3)
            deviation=float(np.std(data,ddof=1)) if len(data)>1 else 0.
            ax.errorbar(float(data.mean()),y,xerr=deviation,fmt="D",markersize=4,color="#242424",
                        ecolor="#242424",capsize=3,elinewidth=1.2,zorder=4)
            details.append(dict(phase=key[0],policy=key[1],writer=key[2],planned_requests=key[3],
                evaluated_seeds=[r["identity"]["seed"] for r in rows],planned_seeds=sorted(planned),
                methods=[r["method"] for r in rows],mean=float(data.mean()),seed_sample_sd=deviation if len(data)>1 else None,
                full_planned_request_budgets_verified=True))
        if baseline is not None:ax.axvline(baseline,color="#555555",linestyle="--",linewidth=1.2,zorder=0)
        ax.set(yticks=range(len(keys)),yticklabels=labels,xlim=limits(values),title=panel_title(name,keys))
        ax.invert_yaxis();ax.grid(axis="x",alpha=.18);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("EvoHallu: source-validation-selected programs on Llama3.1-8b-instruct features",fontsize=12)
    legend=[Line2D([0],[0],marker="o",color="none",markerfacecolor="#347bb4",markeredgecolor="#347bb4",label="Each frozen search seed"),
            Line2D([0],[0],marker="D",color="#242424",markersize=4,label="Mean ± sample SD across seeds")]
    if baseline is not None:legend.append(Line2D([0],[0],linestyle="--",color="#555555",label=f"NLL {baseline:.2f}"))
    fig.legend(handles=legend,loc="outside lower center",ncol=3,fontsize=8)
    missing=[row["method"] for row in selected if row not in available]
    omitted_groups=[key for key in expected if key not in groups]
    note=f"n = full-budget completed / planned seeds; {len(omitted_groups)} groups without complete scores. SD is seed spread, not a CI."
    panels[-1][1].set_xlabel("Nine-dataset AUROC × 100"+(" · per-dataset direction folded" if args.metric=="paper" else " · directed")+"\n"+note,fontsize=9)
    receipt["source"]=dict(status="plotted",metric=metric,nll=baseline,groups=details,
        ignored_partial_or_unverified_primary_identities=ignored,
        incomplete_staged_methods=missing,planned_groups_without_results=[list(key) for key in sorted(omitted_groups,key=sort_key)],
        files=save(fig,args.output,"source_auc_all_seeds"))


def transfer_value(row,side,metric,size):
    key=f"{side}_matched_subset_{metric}_macro_auc_percent"
    if key in row:return row[key]
    word="nine" if size==9 else "six"
    return row.get(f"target_{word}_dataset_{metric}_macro_auc_percent" if side=="target" else f"source_same_{word}_{metric}_macro_auc_percent")


def latest_complete_transfer(args,inputs,scanned):
    candidates=[args.transfer_report] if args.transfer_report else sorted(
        (args.directory/"transfer_llama32_1b").glob("*/transfer_report.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in candidates:
        if path is None or not path.exists():continue
        report=read_json(path,inputs)
        primary,ignored=primary_budget_filter(report.get("methods",[]))
        size=report.get("subset_size",len(report.get("prespecified_transfer_subset",[])))
        complete=bool(size in (6,9) and primary and all(row.get("status") in {"six_dataset_transfer_complete","nine_dataset_transfer_complete"} and
               transfer_value(row,"source",args.metric,size) is not None and transfer_value(row,"target",args.metric,size) is not None for row in primary))
        scanned.append(dict(report=str(path),eligible_full_budget_primary_methods=len(primary),
            complete_for_primary_plot=complete,ignored_partial_or_unverified_primary_identities=ignored))
        if complete:
            return path,report
    return None,None


def transfer_plot(report,manifest,args,receipt,path):
    size=report["subset_size"]
    primary,ignored=primary_budget_filter(report["methods"])
    groups=defaultdict(list)
    for row in primary:groups[group_key(row["identity"])].append(row)
    if not groups:return
    expected=expected_groups(manifest);reference=manifest.get("writer","Qwen/Qwen3.5-9B")
    baseline=next((r for r in report["methods"] if r["method"]==BASELINE),{})
    baseline_source=transfer_value(baseline,"source",args.metric,size)
    baseline_target=transfer_value(baseline,"target",args.metric,size)
    values=[transfer_value(r,side,args.metric,size) for r in primary for side in ("source","target")]
    values += [value for value in (baseline_source,baseline_target) if value is not None]
    fig,panels=axes_for(groups,reference);details=[]
    for name,ax,keys in panels:
        labels=[]
        for y,key in enumerate(keys):
            rows=sorted(groups[key],key=lambda r:r["identity"]["seed"])
            planned=expected.get(key) or {r["identity"]["seed"] for r in rows}
            labels.append(display_label(key,len(rows),len(planned)))
            source=np.asarray([transfer_value(r,"source",args.metric,size) for r in rows])
            target=np.asarray([transfer_value(r,"target",args.metric,size) for r in rows])
            offsets=np.linspace(-.15,.15,len(rows)) if len(rows)>1 else [-.1]
            for i,offset in enumerate(offsets):
                ax.plot([source[i],target[i]],[y+offset,y+offset],color="#aaaaaa",linewidth=.8,zorder=1)
                ax.scatter(source[i],y+offset,color="#347bb4",s=21,zorder=2)
                ax.scatter(target[i],y+offset,color="#d48632",s=21,zorder=2)
            for data,color,marker in ((source,"#347bb4","o"),(target,"#d48632","s")):
                spread=float(np.std(data,ddof=1)) if len(data)>1 else 0.
                ax.errorbar(float(data.mean()),y,xerr=spread,fmt=marker,markersize=5,color=color,capsize=2,zorder=4)
                values += [float(data.mean()-spread),float(data.mean()+spread)]
            details.append(dict(phase=key[0],policy=key[1],writer=key[2],methods=[r["method"] for r in rows],
                paired_seeds=[r["identity"]["seed"] for r in rows],planned_seeds=sorted(planned),
                source_mean=float(source.mean()),target_mean=float(target.mean()),target_minus_source_mean=float((target-source).mean())))
        if baseline_source is not None:ax.axvline(baseline_source,color="#347bb4",linestyle="--",linewidth=1)
        if baseline_target is not None:ax.axvline(baseline_target,color="#d48632",linestyle="--",linewidth=1)
        ax.set(yticks=range(len(keys)),yticklabels=labels,title=panel_title(name,keys))
        ax.invert_yaxis();ax.grid(axis="x",alpha=.18);ax.spines[["top","right"]].set_visible(False)
    for _,ax,_ in panels:ax.set_xlim(limits(values))
    panels[-1][1].set_xlabel(f"Matched {size}-dataset AUROC × 100"+(" · per-dataset direction folded" if args.metric=="paper" else " · directed")+
        "\nThin links pair a search seed; larger markers show means ± seed SD. Both models use the same datasets.\n"
        "Source validation fixes programs; no target refitting. n = full-budget completed / planned seeds.",fontsize=9)
    fig.suptitle(f"Same frozen source-selected programs: Llama3.1-8b → Llama3.2-1B · {size}/9 datasets",fontsize=12)
    legend=[Line2D([0],[0],color="#347bb4",marker="o",label="Source 8B features"),
            Line2D([0],[0],color="#d48632",marker="s",label="Target 1B features"),
            Line2D([0],[0],linestyle="--",color="#777777",label="NLL for each feature model")]
    fig.legend(handles=legend,loc="outside lower center",ncol=3,fontsize=8)
    receipt["transfer"]=dict(status="plotted",report=str(path),coverage=size,
        ignored_partial_or_unverified_primary_identities=ignored,
        prespecified_datasets=report["prespecified_transfer_subset"],groups=details,
        baseline_source=baseline_source,baseline_target=baseline_target,
        files=save(fig,args.output,f"feature_model_transfer_matched{size}"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",type=Path,default=ROOT/"results/overnight20261021")
    ap.add_argument("--report",type=Path)
    ap.add_argument("--manifest",type=Path)
    ap.add_argument("--transfer-report",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--metric",choices=["paper","directed"],default="paper")
    args=ap.parse_args();args.output=args.output or args.directory/"figures";args.output.mkdir(parents=True,exist_ok=True)
    args.report=args.report or args.directory/"report.json";args.manifest=args.manifest or args.directory/"analysis_manifest.json"
    inputs={};manifest=read_json(args.manifest,inputs) if args.manifest.exists() else {}
    receipt=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),metric=args.metric,
        selection="Every complete validation-frozen primary seed with identity.full_budget=true; no test-best seed or test-score ordering",
        primary_group_budget_rule="Only explicitly verified full-budget identities enter primary figures; partial/unverified identities are listed as ignored",
        inputs_sha256=inputs)
    if args.report.exists():source_plot(read_json(args.report,inputs),manifest,args,receipt)
    else:receipt["source"]={"status":"skipped_source_report_unavailable"}
    scanned=[]
    path,transfer=latest_complete_transfer(args,inputs,scanned)
    receipt["transfer_reports_examined"]=scanned
    if transfer is not None:transfer_plot(transfer,manifest,args,receipt,path)
    else:receipt["transfer"]={"status":"skipped_no_complete_transfer_cohort"}
    (args.output/"plot_manifest.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:receipt[key] for key in ("source","transfer")},indent=2))


if __name__=="__main__":
    main()
