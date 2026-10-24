"""Read-only failure taxonomy for saved overnight candidate ledgers.

No candidate repair, model calls, test-data access or quality-based selection.
One primary category is assigned to each persisted ledger row. Pipeline-valid
constant scores are reported separately from pipeline failures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "generation_http": "Generation request HTTP/server/client failure",
    "generation_timeout": "Generation request timeout or connection timeout",
    "generation_context_limit": "Generation server explicitly reports a context-length limit",
    "truncation_parse_extraction": "Syntax/extraction failure with an explicit length/max-token finish reason",
    "syntax_extraction": "Syntax, indentation or class/code extraction failure without explicit truncation evidence",
    "wrong_estimator_constructor": "Estimator inheritance/constructor interface mismatch",
    "imports_api": "Missing import/name, wrong import location or invalid library/object API",
    "missing_statistic_key": "Missing key during candidate execution; original key is retained",
    "ragged_shape_alignment_nonfinite": "Array/ragged/axis/index/alignment handling or nonfinite numeric output",
    "evaluation_memory": "Explicit candidate memory-limit/allocation failure",
    "evaluation_timeout": "Explicit candidate evaluation timeout",
    "exact_duplicate_code": "Pipeline rejected an exact duplicate generated program",
    "valid_constant_scores": "Pipeline-valid but exactly constant on all saved training rows",
    "valid_nonconstant_scores": "Pipeline-valid and nonconstant on saved training rows",
    "valid_constant_status_unknown": "Pipeline-valid but training score artifact absent or unreadable",
    "other_failure": "Failure not safely assigned to the preceding categories",
}
VALID_CATEGORIES = {"valid_constant_scores", "valid_nonconstant_scores", "valid_constant_status_unknown"}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def is_truncated(row):
    return row.get("finish_reason") in {"length", "max_tokens", "max_output_tokens"}


def classify(row, constant_status="unknown"):
    """Use explicit errors/finish reasons, not token count or an AUROC value."""
    if "score" in row.get("val", {}):
        return {"constant": "valid_constant_scores", "nonconstant": "valid_nonconstant_scores"}.get(constant_status, "valid_constant_status_unknown")
    errors = {name: row[name]["error"] for name in ("train", "val") if row.get(name, {}).get("error")}
    if row.get("error"):
        errors["response_or_parse"] = row["error"]
    text = "\n".join(str(value) for value in errors.values()).lower()
    if row.get("error") == "exact duplicate code":
        return "exact_duplicate_code"
    generation_failure = bool(row.get("error") and not row.get("content") and "finish_reason" not in row)
    if generation_failure:
        if any(token in text for token in ("maximum context", "context length", "context_length", "context window", "max_model_len", "too many tokens", "context limit")):
            return "generation_context_limit"
        if any(token in text for token in ("timeout", "timed out", "time out")):
            return "generation_timeout"
        return "generation_http"
    syntax = any(token in text for token in ("syntaxerror", "indentationerror", "taberror", "no hallucinationdetector", "could not extract", "failed to extract", "unterminated", "unexpected indent"))
    if syntax:
        return "truncation_parse_extraction" if is_truncated(row) else "syntax_extraction"
    if any(token in text for token in ("exceeded", "memoryerror", "out of memory", "cannot allocate", "unable to allocate", "memory limit")) and any(token in text for token in ("memory", "rss", "gib", "mib", "allocate")):
        return "evaluation_memory"
    if any(token in text for token in ("evaluation timeout", "estimator exceeded", "candidate timeout", "timed out", "timeout after")):
        return "evaluation_timeout"
    if any(token in text for token in ("object.__init__() takes exactly one argument", "unexpected keyword argument 'stats_dependencies'", 'unexpected keyword argument "stats_dependencies"', "estimator.__init__", "baseestimator.__init__", "must be an estimator subclass")):
        return "wrong_estimator_constructor"
    if "keyerror" in text or "missing statistic" in text or "missing stats" in text:
        return "missing_statistic_key"
    if any(token in text for token in ("has no attribute 'shape'", 'has no attribute "shape"', "has no attribute 'ndim'", "has no attribute 'reshape'")):
        return "ragged_shape_alignment_nonfinite"
    if any(token in text for token in ("importerror", "modulenotfounderror", "cannot import", "no module named", "nameerror", "attributeerror", "has no attribute", "notimplementederror", "unexpected keyword argument")):
        return "imports_api"
    if any(token in text for token in ("shape", "broadcast", "inhomogeneous", "ragged", "axiserror", "indexerror", "axis ", "array element", "length-1 arrays", "scalar arrays", "python scalars", "nonfinite", "nan", "infinity", "not finite", "array dimensions", "dimension mismatch", "size mismatch", "squeeze", "out of bounds")):
        return "ragged_shape_alignment_nonfinite"
    return "other_failure"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/overnight20261021/analysis_manifest.json")
    ap.add_argument("--probe", type=Path, default=ROOT / "results/next_experiments/training_probe.json")
    ap.add_argument("--output", type=Path, default=ROOT / "results/overnight20261021/failure_audit")
    args = ap.parse_args()
    hashes, warnings = {}, []

    def read(path):
        value = path.read_bytes()
        hashes[str(path.resolve().relative_to(ROOT))] = sha(value)
        return value

    def read_json(path):
        return json.loads(read(path))

    manifest = read_json(args.manifest)
    probe = read_json(args.probe) if args.probe.exists() else None
    identities = [job["id"] for job in manifest["jobs"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Analysis manifest contains duplicate job IDs")
    run_rows, candidate_rows, pending_rows = [], [], []
    for job in manifest["jobs"]:
        folder = ROOT / job["expected_logs"]
        summary = read_json(folder / "summary.json") if (folder / "summary.json").exists() else {}
        config = read_json(folder / "trial_config.json") if (folder / "trial_config.json").exists() else {}
        writer = config.get("model", {}).get("hf_model_id", job.get("writer_model", manifest.get("writer", "unknown")))
        records, line_numbers, raw_line_hashes = {}, {}, {}
        ledger = folder / "candidates.jsonl"
        if ledger.exists():
            for line_number, line in enumerate(read(ledger).decode().splitlines(), 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f"Incomplete ledger JSON skipped: {job['id']}, line {line_number}")
                    continue
                key = (row["round"], row["index"])
                if key in records:
                    warnings.append(f"Repeated ledger candidate identity: {job['id']}, {key}; latest persisted row retained")
                records[key], line_numbers[key], raw_line_hashes[key] = row, line_number, sha(line.encode())
        responses = {}
        for path in sorted(folder.glob("responses_*.json")):
            match = re.fullmatch(r"responses_(\d+)\.json", path.name)
            if not match:
                continue
            try:
                values = read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"Unreadable response snapshot: {path}: {exc}")
                continue
            for index, response in enumerate(values):
                responses[(int(match.group(1)), index)] = response
        outcomes = dict(responses)
        outcomes.update({key: row for key, row in records.items() if key not in outcomes})
        categories = Counter()
        probe_constants, unknown_artifacts = 0, 0
        for key, row in sorted(records.items()):
            valid = "score" in row.get("val", {})
            errors = {name:row[name]["error"] for name in ("train", "val") if row.get(name,{}).get("error")}
            if row.get("error"):
                errors["response_or_parse"] = row["error"]
            constant_status = "unknown"
            artifact = None
            if row.get("code_hash") and ("score" in row.get("train", {}) or valid):
                score_path = folder / f"train_scores_{row['code_hash']}.npy"
                if score_path.exists():
                    try:
                        data = score_path.read_bytes()
                        hashes[str(score_path.relative_to(ROOT))] = sha(data)
                        values = np.load(score_path, mmap_mode="r")
                        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                            raise ValueError("Saved training scores are empty, nonfinite or not a one-dimensional vector")
                        constant_status = "constant" if np.all(values == values[0]) else "nonconstant"
                        artifact = dict(file=str(score_path.relative_to(ROOT)), sha256=sha(data), rows=len(values),
                                        exactly_constant=constant_status == "constant")
                        task = config.get("task", {})
                        if (probe is not None and len(values) == probe["train_rows"] and
                            task.get("split_strategy", "stratified") == probe["split_strategy"] and
                            task.get("split_seed") == probe["split_seed"] and task.get("val_size") == .2):
                            positions = probe["ordered_training_prediction_positions"]
                            constant_probe = bool(np.all(values[positions] == values[positions][0]))
                            artifact["fixed_training_probe_constant"] = constant_probe
                            probe_constants += int(valid and constant_probe)
                    except (OSError, ValueError, TypeError) as exc:
                        artifact = dict(file=str(score_path.relative_to(ROOT)),error=f"{type(exc).__name__}: {exc}")
                        warnings.append(f"Training artifact unreadable: {job['id']}, {key}: {exc}")
            unknown_artifacts += int(valid and constant_status == "unknown")
            category = classify(row, constant_status)
            categories[category] += 1
            recomputed = sha((row.get("imports", "") + "\n" + row["code"]).encode()) if "code" in row else None
            candidate_rows.append(dict(job_id=job["id"], phase=job["phase"], policy=job["prompt_policy"],
                writer_model=writer, seed=job["seed"], round=key[0], index=key[1],
                pipeline_valid=valid, primary_category=category, actual_errors=errors,
                finish_reason=row.get("finish_reason"), explicitly_truncated=is_truncated(row),
                completion_tokens=row.get("usage", {}).get("completion_tokens"),
                configured_max_completion_tokens=config.get("model", {}).get("max_tokens"),
                prompt_sha256=row.get("prompt_sha256"), code_hash=row.get("code_hash"),
                recomputed_code_hash=recomputed, record_code_hash_matches_computed=(row.get("code_hash") == recomputed) if recomputed else None,
                response_content_sha256=sha(row.get("content", "").encode()),
                ledger_file=str(ledger.relative_to(ROOT)), ledger_line=line_numbers[key],
                ledger_line_sha256=raw_line_hashes[key], training_score_artifact=artifact))
        for key,response in sorted(outcomes.items()):
            if key not in records:
                pending_rows.append(dict(job_id=job["id"],round=key[0],index=key[1],status="generation_outcome_saved_awaiting_ledger",
                    actual_error=response.get("error"),finish_reason=response.get("finish_reason"),
                    completion_tokens=response.get("usage",{}).get("completion_tokens")))
        valid_count = sum(categories[name] for name in VALID_CATEGORIES)
        full_budget = summary.get("status") == "finished" and summary.get("requests") == job["requests"] and len(records) == job["requests"]
        run_rows.append(dict(job_id=job["id"],phase=job["phase"],policy=job["prompt_policy"],writer_model=writer,
            seed=job["seed"],run_status=summary.get("status","not_run"),full_planned_budget_completed=full_budget,
            planned_requests=job["requests"],requests_with_generation_outcome=len(outcomes),ledger_rows=len(records),
            generation_outcomes_waiting_for_ledger=len(set(outcomes)-set(records)),
            request_budget_not_observed=max(0,job["requests"]-len(outcomes)),pipeline_valid=valid_count,
            pipeline_invalid=len(records)-valid_count,
            valid_fraction_of_ledger=valid_count/len(records) if records else None,
            valid_constant_training_scores=categories["valid_constant_scores"],valid_constant_probe_scores=probe_constants,
            valid_training_artifact_unknown=unknown_artifacts,categories=dict(categories),
            explicitly_truncated_generation_outcomes=sum(is_truncated(row) for row in outcomes.values()),
            known_prompt_tokens=sum(int(row.get("usage",{}).get("prompt_tokens") or 0) for row in outcomes.values()),
            known_completion_tokens=sum(int(row.get("usage",{}).get("completion_tokens") or 0) for row in outcomes.values()),
            known_total_tokens=sum(int(row.get("usage",{}).get("total_tokens") or 0) for row in outcomes.values()),
            generation_outcomes_without_token_usage=sum(not row.get("usage") for row in outcomes.values())))
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["phase"],row["policy"],row["writer_model"])].append(row)
    groups = []
    for (phase,policy,writer), rows in sorted(grouped.items()):
        counts = Counter()
        for row in rows:
            counts.update(row["categories"])
        total = sum(row["ledger_rows"] for row in rows)
        valid = sum(row["pipeline_valid"] for row in rows)
        groups.append(dict(phase=phase,policy=policy,writer_model=writer,jobs=len(rows),
            finished_full_budget_jobs=sum(row["full_planned_budget_completed"] for row in rows),
            planned_requests=sum(row["planned_requests"] for row in rows),
            requests_with_generation_outcome=sum(row["requests_with_generation_outcome"] for row in rows),
            ledger_rows=total,pipeline_valid=valid,pipeline_invalid=total-valid,
            valid_fraction_of_ledger=valid/total if total else None,categories=dict(counts)))
    total_counts = Counter(row["primary_category"] for row in candidate_rows)
    failures = Counter({key:value for key,value in total_counts.items() if key not in VALID_CATEGORIES})
    errors = Counter(error for row in candidate_rows if not row["pipeline_valid"] for error in row["actual_errors"].values())
    full = all(row["full_planned_budget_completed"] for row in run_rows)
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="all_planned_jobs_complete" if full else "partial_campaign_snapshot",labels="is_hallu",
        model_calls=False,candidate_code_modified=False,test_data_or_predictions_read=False,
        classification_unit="One primary category per persisted candidate ledger row; saved response-only requests remain pending",
        validity_denominator="Persisted ledger rows, not unissued budget or in-flight/pending evaluation requests",
        constant_definition="Exact equality on every saved training score; AUROC 50 alone is insufficient evidence",
        truncation_definition="Only explicit finish_reason length/max_tokens/max_output_tokens; syntax errors or short output do not imply token exhaustion",
        missing_statistic_caveat="KeyError records identify a missing runtime key, not necessarily a missing dataset feature",
        categories=CATEGORIES,source_files_sha256=hashes,warnings=warnings,
        jobs_planned=len(run_rows),jobs_finished_full_budget=sum(r["full_planned_budget_completed"] for r in run_rows),
        planned_requests=sum(r["planned_requests"] for r in run_rows),
        requests_with_generation_outcome=sum(r["requests_with_generation_outcome"] for r in run_rows),
        ledger_rows=len(candidate_rows),pipeline_valid=sum(r["pipeline_valid"] for r in candidate_rows),
        category_counts=dict(total_counts),pipeline_failure_counts=dict(failures),
        most_common_pipeline_bottleneck=failures.most_common(1)[0] if failures else None,
        actual_error_frequencies=[dict(error=error,count=count) for error,count in errors.most_common()],
        groups=groups,runs=run_rows,candidates=candidate_rows,pending_response_only_requests=pending_rows)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"failure_audit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines=["# Overnight candidate failure audit","",f"**{report['status']}**: {report['ledger_rows']} persisted candidate rows, {report['pipeline_valid']} pipeline-valid; {report['requests_with_generation_outcome']}/{report['planned_requests']} requests have saved generation outcomes.","",
        "Counts describe the current backup snapshot. Unissued requests and responses awaiting a persisted evaluation result are not counted as failures. Valid constant detectors are separate from invalid programs.","",
        "## Observed validity bottleneck",""]
    if failures:
        category,count=failures.most_common(1)[0]
        lines += [f"The largest observed failure category is **{CATEGORIES[category]}**: {count}/{sum(failures.values())} invalid ledger rows. This identifies the present bottleneck; it is not a causal comparison between unfinished arms."]
    else:
        lines += ["No classified pipeline failures in the saved ledger rows."]
    lines += ["","| Category | Candidate rows |","| --- | ---: |"]
    lines += [f"| {CATEGORIES[category]} | {count} |" for category,count in total_counts.most_common()]
    lines += ["","## Phase, policy and writer","","| Phase / policy | Writer | Saved outcomes / budget | Ledger rows | Valid | Invalid | Valid constant |","| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        lines.append(f"| {g['phase']} / {g['policy']} | {g['writer_model']} | {g['requests_with_generation_outcome']}/{g['planned_requests']} | {g['ledger_rows']} | {g['pipeline_valid']} | {g['pipeline_invalid']} | {g['categories'].get('valid_constant_scores',0)} |")
    lines += ["","## Actual error examples","","All original error strings, source ledger/line hashes and generated-code hashes are retained in JSON. Most common exact messages:",""]
    for error,count in errors.most_common(10):
        lines += [f"Count {count}:","","```text",error,"```",""]
    syntax_without_truncation=[r for r in candidate_rows if r["primary_category"]=="syntax_extraction"]
    lines += ["## Truncation and constant-output interpretation","",
        f"Observed syntax/extraction failures without an explicit length finish reason: **{len(syntax_without_truncation)}**. They are not attributed to the token cap.","",
        "A constant training prediction vector is a valid but degenerate detector, not an import/parser/runtime failure. A 50-AUROC result does not establish constant predictions; the actual saved vector is checked. Missing score artifacts remain unknown.","",
        "No candidate source was repaired, no model was called, and no held-out dataset score was inspected. This audit measures implementation validity and numeric degeneracy, not detection quality."]
    if warnings:
        lines += ["","## Partial-input warnings",""]+["- "+warning for warning in warnings]
    (args.output/"FAILURE_AUDIT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status=report["status"],ledger_rows=report["ledger_rows"],pipeline_valid=report["pipeline_valid"],
        request_outcomes=report["requests_with_generation_outcome"],planned_requests=report["planned_requests"],
        failure_counts=dict(failures),valid_constant_scores=total_counts["valid_constant_scores"],output=str(args.output)),indent=2))


if __name__=="__main__":
    main()
