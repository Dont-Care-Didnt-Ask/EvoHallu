"""Freeze validation-selected overnight programs and score existing feature caches.

This adds an immutable, uniquely named staging layer around the existing CPU
evaluator. It neither generates model features nor selects seeds on test data.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def stage(args):
    source = json.loads(args.manifest.read_text())
    args.directory.mkdir(parents=True, exist_ok=True)
    registry_path = args.directory / "frozen_manifest.json"
    old = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    entries = {row["method"]: row for row in old.get("methods", [])}
    jobs = []
    for job in source["jobs"]:
        run = ROOT / job["expected_logs"]
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        job_status = dict(id=job["id"], phase=job["phase"], seed=job["seed"],
                          policy=job["prompt_policy"], status=summary.get("status", "not_run"))
        jobs.append(job_status)
        if job["phase"] == "prompt_screen" and not args.include_prompt_screen:
            job_status["staging_status"] = "diagnostic_only"
            continue
        if summary.get("status") != "finished":
            job_status["staging_status"] = "pending"
            continue
        config = json.loads((run / "trial_config.json").read_text())
        primary = run / "frozen_estimator.py"
        if not primary.exists() or "selected_validation_auc_percent" not in summary:
            raise ValueError(f"Finished run has no frozen validation winner: {run}")
        selections = [("primary", primary, summary["selected_candidate"],
                       summary["selected_validation_auc_percent"], 1, primary.read_text(), None)]
        seen = {digest(primary)}
        shortlist = summary.get("validation_shortlist", [])
        if any(shortlist[i]["validation_auc_percent"] < shortlist[i+1]["validation_auc_percent"]
               for i in range(len(shortlist)-1)):
            raise ValueError(f"Validation shortlist is not ordered: {run}")
        for row in shortlist:
            if row["rank"] > args.shortlist:
                continue
            path = run / "validation_shortlist" / Path(row["file"]).name
            if digest(path) != row["sha256"]:
                raise ValueError(f"Changed shortlist source: {path}")
            if row["sha256"] in seen:
                continue
            seen.add(row["sha256"])
            selections.append((f"valrank{row['rank']:02d}", path, row["candidate"],
                               row["validation_auc_percent"], row["rank"], path.read_text(), None))
        if args.include_train_selected:
            records_path = run / "candidates.jsonl"
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            eligible = [r for r in records if "score" in r.get("train", {})]
            # Stable max keeps the earliest candidate for equal rounded fitness.
            winner = max(eligible, key=lambda r: r["train"]["score"], default=None)
            baseline_train = summary["baseline_train"]["score"]
            if winner is None or baseline_train >= winner["train"]["score"]:
                source = ROOT / "src/packing/evaluate/uncertainty_quantification/task_uncertainty_quantification.py"
                contents = source.read_text()
                function = next(n for n in ast.parse(contents).body if isinstance(n, ast.FunctionDef) and n.name == "get_initial_func")
                initial = next(n for n in function.body if isinstance(n, ast.ClassDef))
                code = textwrap.dedent(ast.get_source_segment(contents, initial)) + "\n"
                selections.append(("trainbest", source, "baseline", summary["baseline_val"]["score"], None, code, baseline_train))
            else:
                code = winner.get("imports", "") + "\n" + winner["code"] + "\n"
                selections.append(("trainbest", records_path, winner["code_hash"], winner.get("val", {}).get("score"),
                                   None, code, winner["train"]["score"]))
        for role, path, candidate, validation, rank, text, train_score in selections:
            folder = args.directory / "frozen" / f"{job['id']}__{role}"
            dest = folder / "frozen_estimator.py"
            method = "new/" + folder.name
            entry = dict(method=method, job_id=job["id"], phase=job["phase"],
                         policy=job["prompt_policy"], seed=job["seed"],
                         writer_model=config["model"]["hf_model_id"],
                         role=role, validation_rank=rank, candidate=candidate,
                         validation_auc_percent=validation, requests=summary["requests"],
                         planned_requests=job["requests"], full_budget=summary["requests"] == job["requests"],
                         source_file=relative(path), source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         staged_file=relative(dest), selection="directed PopQA validation AUROC within this run",
                         held_out_labels_used_for_selection=False)
            if role == "trainbest":
                entry.update(selection="maximum directed training fitness including baseline; equal fitness keeps earlier candidate",
                             training_auc_percent=train_score, source_container_sha256=digest(path))
            if method in entries and entries[method] != entry:
                raise ValueError(f"Refusing to change an already frozen identity: {method}")
            folder.mkdir(parents=True, exist_ok=True)
            if dest.exists() and digest(dest) != entry["source_sha256"]:
                raise ValueError(f"Existing staged source differs: {dest}")
            if not dest.exists():
                dest.write_text(text)
            entries[method] = entry
        job_status["staging_status"] = "staged"
    ordered = sorted(entries.values(), key=lambda row: row["method"])
    identity = [{k: row[k] for k in ("method", "source_sha256")} for row in ordered]
    cohort = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    frozen = dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_manifest=relative(args.manifest), source_manifest_sha256=digest(args.manifest),
                  cohort_id=cohort, shortlist_limit=args.shortlist,
                  primary_selection_uses_test_labels=False, all_seeds_preserved=True,
                  methods=ordered, jobs=jobs)
    atomic_json(registry_path, frozen)
    return frozen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/next_experiments/run_manifest.json")
    ap.add_argument("--directory", type=Path, default=ROOT / "results/overnight20261021")
    ap.add_argument("--shortlist", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--include-prompt-screen", action="store_true")
    ap.add_argument("--include-train-selected", action="store_true", help="Freeze an additional training-fitness winner per run")
    ap.add_argument("--roles", default="primary,valrank02,valrank03,trainbest", help="Comma-separated staged roles to score")
    ap.add_argument("--jobs", help="Comma-separated job IDs to score; default preserves all staged jobs")
    ap.add_argument("--execute", action="store_true", help="Default is staging and an exact command plan only")
    ap.add_argument("--kind", choices=["new", "supervised", "all"], default="all")
    ap.add_argument("--cache-dir", type=Path, default=Path("/dev/shm/mod1-eval-cache"))
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/claims")
    ap.add_argument("--fitted-models", type=Path, default=ROOT / "out/research20261021/supervised_grouped/models.joblib")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--feature-model", default="/workspace/mod1-models/llama31")
    args = ap.parse_args()
    os.chdir(ROOT)
    args.directory = args.directory.resolve()
    frozen = stage(args)
    roles = args.roles.split(",")
    if not roles or any(role not in {"primary", "valrank02", "valrank03", "trainbest"} for role in roles):
        raise ValueError("Unknown selection role")
    jobs = {job.strip() for job in args.jobs.split(",")} if args.jobs else None
    known_jobs = {job["id"] for job in frozen["jobs"]} | {row["job_id"] for row in frozen["methods"]}
    if jobs is not None and not jobs <= known_jobs:
        raise ValueError(f"Unknown requested job IDs: {sorted(jobs-known_jobs)}")
    selected = [row for row in frozen["methods"] if row["role"] in roles and (jobs is None or row["job_id"] in jobs)]
    selected_hash = hashlib.sha256(json.dumps([{k:r[k] for k in ("method", "source_sha256")} for r in selected], sort_keys=True).encode()).hexdigest()[:16]
    commands = []
    for kind in (["new", "supervised"] if args.kind == "all" else [args.kind]):
        if kind == "new" and not selected:
            continue
        if kind == "supervised" and not args.fitted_models.exists():
            continue
        artifact_hash = digest(args.fitted_models)[:16] if kind == "supervised" else selected_hash
        output = args.directory / "evaluations" / f"{kind}_{artifact_hash}"
        command = [sys.executable, "scripts/evaluate_parallel.py", "--workers", str(args.workers),
                   "--bootstrap", str(args.bootstrap), "--bootstrap-unit", "claim", "--resume",
                   "--cache-dir", str(args.cache_dir), "--data-root", str(args.data_root),
                   "--model-id", args.feature_model, "--output", str(output),
                   "--frozen-glob", selected[0]["staged_file"] if kind == "new" else str((args.directory / "no_frozen_for_supervised/*.py").relative_to(ROOT)),
                   "--method-prefix", kind + "/"]
        if kind == "new":
            for row in selected[1:]:
                command += ["--extra-frozen-glob", row["staged_file"]]
        if kind == "supervised":
            command += ["--extra-fitted-models", str(args.fitted_models)]
        plan = dict(kind=kind, output=relative(output), command=command,
                    cohort_id=artifact_hash, roles=roles if kind == "new" else None,
                    jobs=sorted(jobs) if jobs is not None and kind == "new" else None,
                    bootstrap_unit="normalized claim", bootstrap_samples=args.bootstrap)
        commands.append(plan)
        if args.execute:
            if kind == "new":
                for row in frozen["methods"]:
                    if digest(ROOT / row["staged_file"]) != row["source_sha256"]:
                        raise ValueError("Frozen source changed before evaluation")
            print(json.dumps({"starting": kind, "output": relative(output)}), flush=True)
            subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": "src:.",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    atomic_json(args.directory / "evaluation_commands.json", dict(executed=args.execute, commands=commands))
    print(json.dumps(dict(staged_methods=len(frozen["methods"]), cohort_id=frozen["cohort_id"],
                          pending_jobs=sum(j["staging_status"] == "pending" for j in frozen["jobs"]),
                          requested_jobs_without_selected_method=sorted(jobs-{r["job_id"] for r in selected}) if jobs is not None else [],
                          commands=commands), indent=2))


if __name__ == "__main__":
    main()
