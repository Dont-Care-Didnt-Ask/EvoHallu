"""Optional isolated validation-top-three audit; cached CPU evaluation only.

Default invocation writes a plan without staging or scoring. --execute waits for
the exact twelve full-budget source runs and their primary predictions, then
reuses overnight_eval in a separate directory. No primary registry is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from scripts import overnight_eval as evaluator, overnight_report as reporter
    from scripts.overnight_report_baseline import compare as compare_baseline
except ModuleNotFoundError:
    import overnight_eval as evaluator
    import overnight_report as reporter
    from overnight_report_baseline import compare as compare_baseline

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("original", "neutral", "diverse", "compositional")
SEEDS = (2107, 4319, 8741)
BASELINE = reporter.BASELINE
ROW_KEYS = ("labels", "claim_group_ids", "duplicate_disjoint_mask")
HARD_DEADLINE = dt.datetime(2026, 10, 22, 2, 6, tzinfo=dt.timezone.utc)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.resolve().relative_to(ROOT))


def subset(source):
    """Identify the requested cohort by exact IDs, never by test outcomes."""
    expected = {f"next_evo_{policy}_{seed}" for policy in POLICIES for seed in SEEDS}
    jobs = [row for row in source["jobs"] if row["id"] in expected]
    if len(jobs) != 12 or {row["id"] for row in jobs} != expected:
        raise ValueError("Expected the exact twelve predefined source adaptive jobs")
    for job in jobs:
        policy = job["prompt_policy"]
        phase = "compositional_control" if policy == "compositional" else "evolution_screen"
        if (policy not in POLICIES or job["seed"] not in SEEDS or
                job["id"] != f"next_evo_{policy}_{job['seed']}" or
                job["phase"] != phase or job["requests"] != 60 or job["mode"] != "evolution"):
            raise ValueError("Unexpected adaptive identity, phase or request budget")
    if source["writer"] != "Qwen/Qwen3.5-9B":
        raise ValueError("This secondary cohort is restricted to the pinned 9B writer")
    return jobs


def validation_rows(run, summary):
    rows = summary.get("validation_shortlist", [])
    config = json.loads((run / "trial_config.json").read_text())
    if int(config["trial"].get("shortlist_size", 1)) != 3:
        raise ValueError("Source run did not export the predefined top-three shortlist")
    if len(rows) != min(3, summary["valid_candidates"]+1):
        raise ValueError("Missing exported valid shortlist candidates")
    if [r["rank"] for r in rows] != list(range(1, len(rows) + 1)) or len(rows) > 3:
        raise ValueError("Malformed validation shortlist ranks")
    if not rows or any(rows[i]["validation_auc_percent"] < rows[i+1]["validation_auc_percent"]
                       for i in range(len(rows)-1)):
        raise ValueError("Missing or unordered validation shortlist")
    for row in rows:
        path = run / "validation_shortlist" / Path(row["file"]).name
        if sha(path) != row["sha256"]:
            raise ValueError("Changed validation shortlist source")
    first = rows[0]
    if (first["candidate"] != summary["selected_candidate"] or
            first["validation_auc_percent"] != summary["selected_validation_auc_percent"] or
            first["sha256"] != sha(run / "frozen_estimator.py")):
        raise ValueError("Rank one does not match the already selected primary")
    # The generator inserts baseline first before stable descending sorting:
    # equal validation scores therefore retain baseline ahead of proposals.
    baseline = next((r for r in rows if r["candidate"] == "baseline"), None)
    if baseline is not None and any(r["validation_auc_percent"] == baseline["validation_auc_percent"]
                                    for r in rows[:baseline["rank"]-1]):
        raise ValueError("Baseline tie priority differs from the source selection rule")
    return rows


def readiness(jobs, primary_directory):
    registry_path = primary_directory / "frozen_manifest.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"methods": []}
    identities = {r["job_id"]: r for r in registry["methods"] if r["role"] == "primary"}
    coverage = defaultdict(set)
    for path in sorted((primary_directory / "evaluations").glob("*/*_scores.npz")):
        try:
            with np.load(path) as z:
                for method in z.files:
                    coverage[method].add(path.name.removesuffix("_scores.npz"))
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            # The existing evaluator writes NPZ files before their closing ZIP
            # directory is available. An in-progress file is not readiness.
            continue
    pending, completed = [], []
    for job in jobs:
        run = ROOT / job["expected_logs"]
        path = run / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
        if summary.get("status") != "finished" or summary.get("requests") != 60:
            pending.append(dict(job=job["id"], reason="source run not finished at full 60-request budget"))
            continue
        validation_rows(run, summary)
        identity = identities.get(job["id"])
        if identity is None:
            pending.append(dict(job=job["id"], reason="primary not yet frozen by original CPU driver"))
            continue
        if (identity["source_sha256"] != sha(run / "frozen_estimator.py") or
                identity["source_sha256"] != sha(ROOT / identity["staged_file"]) or
                identity["candidate"] != summary["selected_candidate"] or
                identity["validation_auc_percent"] != summary["selected_validation_auc_percent"] or
                identity.get("full_budget") is not True or identity["requests"] != 60 or
                identity["writer_model"] != "Qwen/Qwen3.5-9B" or identity["seed"] != job["seed"]):
            raise ValueError("Original primary identity differs from source run")
        if len(coverage[identity["method"]]) != 9:
            pending.append(dict(job=job["id"], reason="original primary predictions not complete on nine datasets"))
            continue
        completed.append(identity)
    return dict(ready=len(completed) == 12, pending=pending, original_primaries=completed)


def freeze(args, jobs):
    status = readiness(jobs, args.primary_directory)
    if not status["ready"]:
        raise RuntimeError("Refusing to freeze an incomplete twelve-run cohort")
    original_hash = sha(args.primary_directory / "frozen_manifest.json")
    frozen = evaluator.stage(SimpleNamespace(manifest=args.directory / "source_manifest.json",
        directory=args.directory, include_prompt_screen=False, include_train_selected=False, shortlist=3))
    by_job = defaultdict(list)
    for row in frozen["methods"]:
        by_job[row["job_id"]].append(row)
    slots = []
    for job in jobs:
        run = ROOT / job["expected_logs"]
        summary = json.loads((run / "summary.json").read_text())
        rows = {r["rank"]: r for r in validation_rows(run, summary)}
        for rank in (1, 2, 3):
            slot = dict(job_id=job["id"], policy=job["prompt_policy"], seed=job["seed"], validation_rank=rank)
            if rank not in rows:
                slot.update(status="missing_valid_candidate", reason="fewer than three exported valid proposals including baseline")
            else:
                row = rows[rank]
                entry = next(r for r in by_job[job["id"]] if r["source_sha256"] == row["sha256"])
                slot.update(status="frozen", candidate=row["candidate"], validation_auc_percent=row["validation_auc_percent"],
                    source_sha256=row["sha256"], staged_file=entry["staged_file"], staged_method=entry["method"],
                    staged_role=entry["role"], aliases_earlier_source=entry["validation_rank"] != rank,
                    original_primary_method=f"new/{job['id']}__primary")
            slots.append(slot)
    if sha(args.primary_directory / "frozen_manifest.json") != original_hash:
        raise RuntimeError("Original registry changed while freezing; retry after the driver completes")
    evaluator.atomic_json(args.directory / "rank_slots.json", dict(slots=slots,
        original_primary_registry_sha256=original_hash, own_registry_sha256=sha(args.directory / "frozen_manifest.json"),
        rank_ties="stable source ordering; baseline inserted first; then proposal order", all_seeds_preserved=True,
        source_deduplication="identical source hashes share predictions; all original rank slots retained"))
    return slots


def metrics(method, datasets, names):
    missing = [name for name in names if name not in datasets or method not in datasets[name]]
    if missing:
        return dict(status="incomplete", missing_datasets=missing)
    per_dataset = []
    for name in names:
        z = datasets[name]
        score = z[method]
        if score.shape != z["labels"].shape or not np.isfinite(score).all():
            raise ValueError("Invalid saved score vector")
        raw = 100 * float(roc_auc_score(z["labels"], score))
        mask = z["duplicate_disjoint_mask"]
        strict = 100 * float(roc_auc_score(z["labels"][mask], score[mask]))
        per_dataset.append(dict(dataset=name, directed=raw, paper=max(raw, 100-raw),
            strict_directed=strict, strict_paper=max(strict, 100-strict)))
    return dict(status="complete", per_dataset=per_dataset,
        **{key:float(np.mean([row[key] for row in per_dataset]))
           for key in ("directed", "paper", "strict_directed", "strict_paper")})


def summarize(args):
    slots = json.loads((args.directory / "rank_slots.json").read_text())["slots"]
    primary, primary_files = reporter.collect(args.primary_directory, [])
    extra, extra_files = reporter.collect(args.directory, [])
    csvs = [p for p in sorted(args.data_root.rglob("*.csv")) if "popqa_train" not in p.name]
    names = [p.parent.name + "__" + p.stem for p in csvs]
    if len(names) != 9:
        raise ValueError("Exactly nine source datasets required")
    if set(primary) != set(names):
        raise ValueError("Primary reference lacks exact source-nine coverage")
    for name in extra:
        if name not in primary:
            raise ValueError("Unexpected secondary dataset")
        for key in (*ROW_KEYS, BASELINE):
            if key not in extra[name] or not np.array_equal(primary[name][key], extra[name][key]):
                raise ValueError("Secondary baseline or row identity differs from primary")
    for slot in slots:
        if slot["status"] == "missing_valid_candidate":
            continue
        if sha(ROOT / slot["staged_file"]) != slot["source_sha256"]:
            raise ValueError("Changed frozen shortlist source")
        reference = slot["staged_role"] == "primary"
        dataset = primary if reference else extra
        method = slot["original_primary_method"] if reference else slot["staged_method"]
        if reference and slot["candidate"] == "baseline":
            for name in names:
                if not np.array_equal(primary[name][method], primary[name][BASELINE]):
                    raise ValueError("Baseline-selected primary predictions differ from canonical NLL")
        slot.update(predictions_from="original primary reference" if reference else "isolated secondary evaluation",
                    metrics=metrics(method, dataset, names))
    groups, collections = [], []
    for policy in POLICIES:
        for rank in (1, 2, 3):
            rows = [r for r in slots if r["policy"] == policy and r["validation_rank"] == rank]
            complete = len(rows) == 3 and all(r.get("metrics",{}).get("status") == "complete" for r in rows)
            groups.append(dict(policy=policy, validation_rank=rank, status="complete" if complete else "incomplete",
                seeds=[r["seed"] for r in rows], **({key:reporter.distribution([r["metrics"][key] for r in rows])
                for key in ("directed", "paper", "strict_directed", "strict_paper")} if complete else {})))
    for job in sorted({r["job_id"] for r in slots}):
        rows = [r for r in slots if r["job_id"] == job]
        complete = all(r.get("metrics",{}).get("status") == "complete" for r in rows)
        collections.append(dict(job_id=job, policy=rows[0]["policy"], seed=rows[0]["seed"],
            status="complete" if complete else "incomplete", unique_sources=len({r.get("source_sha256") for r in rows if "source_sha256" in r}),
            **({key:float(np.mean([r["metrics"][key] for r in rows]))
            for key in ("directed", "paper", "strict_directed", "strict_paper")} if complete else {})))
    collection_groups=[]
    for policy in POLICIES:
        rows=[r for r in collections if r["policy"]==policy]
        complete=len(rows)==3 and all(r["status"]=="complete" for r in rows)
        collection_groups.append(dict(policy=policy,status="complete" if complete else "incomplete",
            **({key:reporter.distribution([r[key] for r in rows])
                for key in ("directed","paper","strict_directed","strict_paper")} if complete else {})))
    baseline_audit = compare_baseline(primary, args.historical, args.data_root)
    if baseline_audit["status"] != "passed":
        raise ValueError("Primary canonical baseline/CSV identity audit failed")
    complete = all(row.get("metrics",{}).get("status") == "complete" for row in slots)
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="complete_secondary_shortlist" if complete else "incomplete_secondary_shortlist", slots=slots,
        policy_rank_all_seed_groups=groups, per_run_top_three_mean_individual_auc=collections,
        policy_top_three_collection_all_seed_groups=collection_groups,
        interpretation="Top-three means average individual method AUROCs, not predictions or ensembles. No test-best or oracle selection.",
        selection="Source directed PopQA validation ranks, preserving baseline and stable ties; all twelve runs and seeds retained",
        primary_registry_modified=False, feature_extraction_or_llm_calls=0, source_prediction_sha256=primary_files,
        extra_prediction_sha256=extra_files, baseline_exact_audit=baseline_audit,
        planned_additional_rank_slots=24, unique_additional_sources=len({r["staged_method"] for r in slots if r.get("staged_role", "primary") != "primary"}),
        limitations="Exploratory retained collection; prior benchmark exposure, inherited split duplicates, no new blind test. Intervals condition on frozen programs.")
    evaluator.atomic_json(args.directory / "SHORTLIST_REPORT.json", report)
    lines = ["# Validation-shortlist secondary analysis", "", "Status: **"+report["status"]+"**.", "", report["interpretation"], "",
        "| Policy | Validation rank | Seeds | Directed mean ± SD | Folded mean ± SD | Duplicate-disjoint folded |",
        "| --- | ---: | --- | ---: | ---: | ---: |"]
    def display(row, key):
        value=row.get(key)
        return f"{value['mean']:.3f} ± {value['std']:.3f}" if value else "PENDING"
    for row in groups:
        lines.append(f"| {row['policy']} | {row['validation_rank']} | {row['seeds']} | {display(row,'directed')} | {display(row,'paper')} | {display(row,'strict_paper')} |")
    lines += ["", "Per-run frozen ranks (no test-based ordering):", "",
        "| Run | Validation rank | Validation AUROC | Directed nine | Folded nine | Strict folded nine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in slots:
        values=row.get("metrics",{})
        shown=[f"{values[key]:.3f}" if key in values else "PENDING" for key in ("directed","paper","strict_paper")]
        val=f"{row['validation_auc_percent']:.4f}" if "validation_auc_percent" in row else "missing valid candidate"
        lines.append(f"| {row['job_id']} | {row['validation_rank']} | {val} | {' | '.join(shown)} |")
    lines += ["", "Top-three collection means average individual method AUROCs; they are not ensembles:", "",
        "| Policy | Directed collection mean ± SD across seeds | Folded collection mean ± SD across seeds |",
        "| --- | ---: | ---: |"]
    for row in collection_groups:
        lines.append(f"| {row['policy']} | {display(row,'directed')} | {display(row,'paper')} |")
    lines += ["", "All per-run validation scores, nine-dataset scores, strict scores, identities and hashes are in [JSON](SHORTLIST_REPORT.json).", "",
        "The [source gallery](SOURCE_GALLERY.md) includes every rank slot, including baseline/duplicate aliases and missing valid candidates.", "", report["limitations"]]
    (args.directory / "SHORTLIST_REPORT.md").write_text("\n".join(lines)+"\n")
    gallery=["# Frozen validation-top-three source gallery", "", "Ranks are chosen only by source PopQA validation; no benchmark sorting.", "",
        "| Run | Rank | Validation AUROC | Candidate | Frozen code | Source SHA-256 |", "| --- | ---: | ---: | --- | --- | --- |"]
    for row in slots:
        if "staged_file" not in row:
            gallery.append(f"| {row['job_id']} | {row['validation_rank']} | missing | fewer valid candidates | — | — |")
        else:
            target=os.path.relpath(ROOT / row["staged_file"],args.directory)
            gallery.append(f"| {row['job_id']} | {row['validation_rank']} | {row['validation_auc_percent']:.4f} | {row['candidate']} | [source]({target}) | `{row['source_sha256']}` |")
    (args.directory / "SOURCE_GALLERY.md").write_text("\n".join(gallery)+"\n")
    return report


def validated_deadline(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("Deadline must be timezone-aware UTC")
    if parsed > HARD_DEADLINE:
        raise ValueError("Deadline cannot exceed 2026-10-22 02:06 UTC")
    return parsed.timestamp()


def check_stop(cancel_paths, deadline):
    marker = next((path for path in cancel_paths if path.exists()), None)
    if marker is not None:
        raise KeyboardInterrupt(f"Shortlist cancelled by marker: {marker}")
    if time.time() >= deadline:
        raise TimeoutError("Shortlist absolute/active deadline reached")


def await_ready(jobs, primary_directory, absolute_deadline, active_seconds, cancel_paths):
    """Readiness waiting uses only the absolute gate; active time starts afterward."""
    while True:
        check_stop(cancel_paths, absolute_deadline-30)
        status = readiness(jobs, primary_directory)
        if status["ready"]:
            started = time.time()
            return status, started, min(absolute_deadline, started+active_seconds)
        time.sleep(min(20, max(.1, absolute_deadline-time.time()-30)))


@contextmanager
def execution_guard(deadline):
    """SIGTERM/SIGINT unwind through bounded's owned-process cleanup.

    The real-time alarm also bounds CPU reporting, which has no subprocess wait.
    """
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Shortlist received signal {signum}")
    def timed_out(signum, frame):
        raise TimeoutError("Shortlist absolute/active deadline reached")
    old_term = signal.signal(signal.SIGTERM, terminate)
    old_alarm = signal.signal(signal.SIGALRM, timed_out)
    old_timer = signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.time()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGALRM, old_alarm)
        if old_timer[0] or old_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def bounded(command, deadline, log, cancel_paths=()):
    remaining = deadline-time.time()-30
    if remaining <= 0:
        raise TimeoutError("No CPU evaluation time remains before the fixed deadline")
    check_stop(cancel_paths, deadline-30)
    with log.open("a") as stream:
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ,"PYTHONPATH":"src:.","CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
        try:
            while True:
                check_stop(cancel_paths, deadline-30)
                try:
                    result=child.wait(timeout=min(1, max(.01, deadline-time.time()-30)))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if result:
                raise subprocess.CalledProcessError(result, command)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",type=Path,default=ROOT/"results/next_experiments/run_manifest.json")
    ap.add_argument("--primary-directory",type=Path,default=ROOT/"results/overnight20261021")
    ap.add_argument("--directory",type=Path,default=ROOT/"results/overnight20261021/shortlist_secondary")
    ap.add_argument("--cache-dir",type=Path,default=Path("/dev/shm/mod1-eval-cache"))
    ap.add_argument("--data-root",type=Path,default=ROOT/"data/claims")
    ap.add_argument("--historical",type=Path,default=ROOT/"out/campaign/evaluation_with_4b")
    ap.add_argument("--deadline-utc",default="2026-10-22T02:06:00Z")
    ap.add_argument("--max-active-seconds",type=int,choices=[1800,3600],default=1800)
    ap.add_argument("--cancel-file",type=Path,action="append",default=[],help="Additional stop marker; default campaign and shortlist markers always apply")
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--report-only",action="store_true")
    args=ap.parse_args()
    absolute_deadline=validated_deadline(args.deadline_utc)
    os.chdir(ROOT)
    args.directory=args.directory.resolve();args.primary_directory=args.primary_directory.resolve()
    if args.directory == args.primary_directory or args.primary_directory.is_relative_to(args.directory):
        raise ValueError("Secondary output must be isolated from the primary registry/evaluations")
    if any(args.directory.is_relative_to(args.primary_directory / name) for name in ("evaluations", "frozen")):
        raise ValueError("Secondary output cannot live in primary staging or evaluation folders")
    source=json.loads(args.manifest.read_text());jobs=subset(source)
    args.directory.mkdir(parents=True,exist_ok=True)
    path=args.directory/"source_manifest.json"
    selected=dict(writer=source["writer"],writer_revision=source["writer_revision"],jobs=jobs,
        source_manifest=rel(args.manifest),source_manifest_sha256=sha(args.manifest),
        selection="Exact twelve predefined full-budget adaptive source jobs only; no test selection")
    if path.exists() and json.loads(path.read_text()) != selected:
        raise ValueError("Refusing to change an existing secondary cohort")
    evaluator.atomic_json(path,selected)
    command=[sys.executable,"scripts/overnight_eval.py","--manifest",rel(path),"--directory",rel(args.directory),
        "--shortlist","3","--roles","valrank02,valrank03","--kind","new","--workers","4","--bootstrap","500",
        "--jobs",",".join(j["id"] for j in jobs),"--cache-dir",str(args.cache_dir),"--data-root",str(args.data_root),"--execute"]
    status=readiness(jobs,args.primary_directory)
    cancel_paths=[args.primary_directory/"watchdog.cancel",args.primary_directory/"shortlist_secondary.cancel",
                  args.directory/"shortlist_secondary.cancel",*args.cancel_file]
    evaluator.atomic_json(args.directory/"PLAN.json",dict(execute_requested=args.execute,ready=status["ready"],pending=status["pending"],
        command=command,helper_sha256=sha(Path(__file__)),deadline_utc=args.deadline_utc,max_active_seconds=args.max_active_seconds,
        active_budget_starts="after all twelve source runs and primary-nine predictions are ready",
        waiting_gate="absolute UTC deadline only; 20-second polling",hard_deadline_utc=HARD_DEADLINE.isoformat(),
        cancel_files=[str(path) for path in cancel_paths],
        planned_rank_slots=36,planned_extra_rank_slots=24,baseline_source_aliases_reuse_saved_predictions=True,
        primary_registry_modified=False,cpu_only=True,source_manifest_sha256=sha(args.manifest)))
    if args.report_only:
        summarize(args);return
    if not args.execute:
        print(json.dumps(dict(plan=rel(args.directory/"PLAN.json"),ready=status["ready"],pending=status["pending"]),indent=2));return
    with execution_guard(absolute_deadline):
        status,active_started,deadline=await_ready(jobs,args.primary_directory,absolute_deadline,args.max_active_seconds,cancel_paths)
        signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
        check_stop(cancel_paths,deadline)
        freeze(args,jobs)
        bounded(command,deadline,args.directory/"evaluation.log",cancel_paths)
        check_stop(cancel_paths,deadline)
        report=summarize(args)
        check_stop(cancel_paths,deadline)
        evaluator.atomic_json(args.directory/"execution_receipt.json",dict(status=report["status"],finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            deadline_utc=args.deadline_utc,active_started_utc=dt.datetime.fromtimestamp(active_started,dt.timezone.utc).isoformat(),
            effective_active_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),
            max_active_seconds=args.max_active_seconds,helper_sha256=sha(Path(__file__)),cpu_only=True,primary_registry_modified=False,
            own_registry_sha256=sha(args.directory/"frozen_manifest.json")))


if __name__=="__main__":
    main()
