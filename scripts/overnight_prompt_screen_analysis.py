"""Analyse the matched 15-context/four-policy screen using saved training scores.

Read-only inputs; no model calls and no validation/test-score selection. Full
paired comparisons require all four prespecified 30-request jobs to finish.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("original", "neutral", "diverse", "compositional")
THRESHOLD = .995


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True).encode())


def rank_vector(values):
    ranks = rankdata(values).astype(float)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm else None


def failure_category(row):
    if "score" in row.get("val", {}):
        return None
    if row.get("error") == "exact duplicate code":
        return "exact_duplicate_code"
    if row.get("error"):
        return "request_failure" if not row.get("content") else "parse_or_response_failure"
    if row.get("train", {}).get("error"):
        return "training_evaluation_failure"
    if row.get("val", {}).get("error"):
        return "validation_evaluation_failure"
    return "evaluation_incomplete"


def numeric_distribution(values):
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                minimum=float(min(values)) if values else None,
                maximum=float(max(values)) if values else None)


def analyse(args):
    hashes, warnings, protocol_errors = {}, [], []

    def read_bytes(path):
        value = path.read_bytes()
        hashes[str(path.resolve().relative_to(ROOT))] = digest(value)
        return value

    def read_json(path):
        return json.loads(read_bytes(path))

    manifest = read_json(args.manifest)
    frozen = read_json(args.contexts)
    probe = read_json(args.probe)
    contexts = frozen["contexts"]
    if len(contexts) != 15 or probe["probe_rows"] != 512 or len(probe["ordered_training_prediction_positions"]) != 512:
        raise ValueError("The prespecified study requires exactly 15 contexts and 512 training-probe rows")
    if manifest["contexts_sha256"] != digest(args.contexts.read_bytes()):
        raise ValueError("Frozen-context manifest differs from the study plan")
    if probe.get("validation_used") is not False or probe["labels"] != "is_hallu":
        raise ValueError("Probe must be the prespecified training-only is_hallu probe")
    positions = np.asarray(probe["ordered_training_prediction_positions"])
    if len(np.unique(positions)) != 512 or np.any(positions < 0) or np.any(positions >= probe["train_rows"]):
        raise ValueError("Training probe must contain 512 distinct in-range prediction positions")
    if frozen.get("adaptive_population_feedback") is not False:
        raise ValueError("Matched prompt screening must use frozen contexts without adaptive population feedback")
    jobs = {job["prompt_policy"]:job for job in manifest["jobs"] if job["phase"] == "prompt_screen"}
    if set(jobs) != set(POLICIES) or any(job["requests"] != 30 for job in jobs.values()):
        raise ValueError("Expected the four prespecified 30-request prompt-screen jobs")
    from experiments.prompt_policies import candidate_prompt
    policies, configs = {}, {}
    source_contexts = []
    for index, context in enumerate(contexts):
        source = ROOT / context["source"]
        row = dict(context_index=index, context_id=f"context_{index:02d}",
                   original_source=context["source"], source_round_1based=context["source_round_1based"],
                   expected_source_file_sha256=context["sha256"], base_chat_sha256=canonical(context["chat"]))
        if source.exists():
            row["source_file_sha256"] = digest(read_bytes(source))
            row["source_file_hash_matches"] = row["source_file_sha256"] == context["sha256"]
            if not row["source_file_hash_matches"]:
                protocol_errors.append(f"Frozen historical source prompt changed: {source}")
        else:
            row["source_file_hash_matches"] = None
            warnings.append(f"Historical source file unavailable; frozen chat remains hashed: {context['source']}")
        source_contexts.append(row)
    for policy in POLICIES:
        job = jobs[policy]
        folder = ROOT / job["expected_logs"]
        records, responses, problems = {}, {}, []
        summary = read_json(folder / "summary.json") if (folder / "summary.json").exists() else {}
        cfg = read_json(folder / "trial_config.json") if (folder / "trial_config.json").exists() else None
        if cfg:
            task = cfg["task"]
            if (cfg.get("search_mode") != "frozen_context" or cfg["trial"].get("prompt_policy") != policy or
                cfg["trial"].get("frozen_contexts_sha256") != manifest["contexts_sha256"] or
                task.get("split_strategy", "stratified") != probe["split_strategy"] or
                task["split_seed"] != probe["split_seed"] or task.get("sample_size", 0) != 0 or task["val_size"] != .2):
                problems.append("Run configuration does not match the prespecified fixed-context/training-probe design")
            configs[policy] = dict(model=cfg["model"], seed=cfg["seed"],
                enable_thinking=cfg["trial"].get("enable_thinking", False),
                num_rounds=cfg["num_rounds"], outputs_per_context=cfg["num_outputs_per_prompt"])
        record_path = folder / "candidates.jsonl"
        if record_path.exists():
            for line in read_bytes(record_path).decode().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f"Incomplete candidate JSON line skipped: {job['id']}")
                    continue
                key = (row["round"], row["index"])
                if key in records:
                    problems.append(f"Repeated candidate identity: {key}")
                if key[0] not in range(15) or key[1] not in (0, 1):
                    problems.append(f"Unexpected candidate identity: {key}")
                records[key] = row
        prepared_contexts = 0
        context_rows = []
        for index, context in enumerate(contexts):
            base_path = folder / f"prompt_{index:04d}.json"
            request_path = folder / f"candidate_prompts_{index:04d}.json"
            response_path = folder / f"responses_{index:04d}.json"
            actual_chats = None
            if base_path.exists():
                prepared_contexts += 1
                if read_json(base_path) != context["chat"]:
                    problems.append(f"Saved base chat differs from frozen context {index}")
            if request_path.exists():
                actual_chats = read_json(request_path)
                expected = [candidate_prompt(context["chat"], policy, index, sibling) for sibling in range(2)]
                if actual_chats != expected:
                    problems.append(f"Actual policy prompts differ from the prespecified transform: context {index}")
            if response_path.exists():
                saved = read_json(response_path)
                if len(saved) != 2:
                    problems.append(f"Expected two saved responses in context {index}")
                for sibling, response in enumerate(saved):
                    responses[(index,sibling)] = response
            children, vectors = [], []
            for sibling in range(2):
                key = (index, sibling)
                record = records.get(key)
                response = responses.get(key)
                item = dict(index=sibling, candidate_record_available=record is not None,
                            response_available=response is not None)
                if record is None:
                    children.append(item)
                    vectors.append(None)
                    continue
                valid = "score" in record.get("val", {})
                item.update(pipeline_valid=valid, failure_category=failure_category(record),
                    errors={name:record[name]["error"] for name in ("train", "val") if record.get(name,{}).get("error")},
                    response_error=record.get("error"), code_hash=record.get("code_hash"),
                    content_sha256=digest(record.get("content", "").encode()),
                    finish_reason=record.get("finish_reason"), request_seed=record.get("request_seed"),
                    prompt_sha256=record.get("prompt_sha256"), duplicate_code=record.get("duplicate", False))
                expected_seed = int(job["seed"]) + index * 100 + sibling
                if record.get("request_seed") != expected_seed:
                    problems.append(f"Request seed differs: context {index}, sibling {sibling}")
                if actual_chats is None or len(actual_chats) != 2 or record.get("prompt_sha256") != canonical(actual_chats[sibling]):
                    problems.append(f"Candidate prompt hash unavailable or inconsistent: context {index}, sibling {sibling}")
                if response is not None:
                    for field in ("content", "usage", "finish_reason", "generation_seconds"):
                        if record.get(field) != response.get(field):
                            problems.append(f"Saved response/record differs in {field}: context {index}, sibling {sibling}")
                vector = None
                if valid:
                    score_path = folder / f"train_scores_{record['code_hash']}.npy"
                    if not score_path.exists():
                        item["probe_status"] = "training_predictions_missing"
                    else:
                        try:
                            values = np.load(score_path)
                            hashes[str(score_path.relative_to(ROOT))] = digest(score_path.read_bytes())
                            if values.shape != (probe["train_rows"],) or not np.isfinite(values).all():
                                raise ValueError("Saved training predictions have the wrong shape or nonfinite values")
                            selected = np.asarray(values[positions])
                            item["probe_score_sha256"] = digest(selected.astype("<f8").tobytes())
                            vector = rank_vector(selected)
                            item["probe_status"] = "constant" if vector is None else "nonconstant"
                        except (OSError, ValueError) as exc:
                            item["probe_status"] = "training_predictions_unreadable"
                            warnings.append(f"{job['id']}: {exc}")
                else:
                    item["probe_status"] = "pipeline_invalid"
                children.append(item)
                vectors.append(vector)
            complete = all(child["candidate_record_available"] for child in children)
            valid_count = sum(child.get("pipeline_valid", False) for child in children)
            diagnostic_complete = complete and all(not child.get("pipeline_valid") or child.get("probe_status") in {"constant", "nonconstant"} for child in children)
            eligible = complete and valid_count == 2 and all(v is not None for v in vectors)
            rho = float(np.clip(vectors[0] @ vectors[1], -1., 1.)) if eligible else None
            context_rows.append(dict(context_id=f"context_{index:02d}", context_index=index,
                base_chat_sha256=canonical(context["chat"]), complete=complete, diagnostic_complete=diagnostic_complete,
                valid_candidates=valid_count if complete else None,
                two_valid_nonconstant=int(eligible) if diagnostic_complete else None,
                sibling_spearman_rho=rho,
                joint_diverse_pair=int(eligible and rho < THRESHOLD) if diagnostic_complete else None,
                joint_direction_invariant_diverse_pair=int(eligible and abs(rho) < THRESHOLD) if diagnostic_complete else None,
                candidates=children))
        # Prefer the immutable saved response for token/latency accounting; the
        # candidate record fills the gap when a backup has not copied that file.
        attempts = dict(responses)
        attempts.update({key:row for key,row in records.items() if key not in attempts})
        complete_rows = [row for row in context_rows if row["complete"]]
        correlations = [row["sibling_spearman_rho"] for row in context_rows if row["sibling_spearman_rho"] is not None]
        failures = Counter(failure_category(row) for row in records.values() if failure_category(row))
        missing_probe = sum(child.get("probe_status") in {"training_predictions_missing", "training_predictions_unreadable"} for row in context_rows for child in row["candidates"])
        full = (summary.get("status") == "finished" and summary.get("requests") == 30 and
                len(records) == 30 and len(complete_rows) == 15 and missing_probe == 0 and not problems)
        total_tokens = sum(int(row.get("usage",{}).get("total_tokens") or 0) for row in attempts.values())
        if summary.get("status") == "finished" and summary.get("total_tokens") != total_tokens:
            problems.append("Saved token usage differs from final summary total")
            full = False
        generation = [float(row["generation_seconds"]) for row in attempts.values() if row.get("generation_seconds") is not None]
        maximum_round_generation = [max(float(attempts[(i,j)]["generation_seconds"]) for j in range(2))
            for i in range(15) if all((i,j) in attempts and attempts[(i,j)].get("generation_seconds") is not None for j in range(2))]
        policy_row = dict(policy=policy, job_id=job["id"], status=summary.get("status","not_run"), full_30_request_screen=full,
            planned_contexts=15, planned_requests=30, prepared_or_inflight_contexts=prepared_contexts,
            issued_contexts_with_saved_outcomes=len({key[0] for key in attempts}),
            requests_with_saved_outcomes=len(attempts), evaluated_candidate_records=len(records),
            fully_evaluated_contexts=len(complete_rows), pipeline_valid_requests=sum("score" in row.get("val",{}) for row in records.values()),
            valid_candidates_per_context_histogram={str(k):sum(row["valid_candidates"] == k for row in complete_rows) for k in range(3)},
            validity_denominator="Two requests per context; the 0/1/2 histogram uses fully evaluated contexts only",
            missing_training_probe_arrays=missing_probe,
            constant_probe_candidates=sum(child.get("probe_status") == "constant" for row in context_rows for child in row["candidates"]),
            joint_diverse_pairs_observed=sum(row["joint_diverse_pair"] == 1 for row in context_rows),
            joint_diverse_pair_planned_denominator=15,
            joint_diverse_pair_fraction_all_15=float(np.mean([row["joint_diverse_pair"] for row in context_rows])) if full else None,
            conditional_sibling_spearman=numeric_distribution(correlations),
            conditional_pair_denominator=len(correlations), failure_counts=dict(failures), protocol_errors=problems,
            token_usage=dict(prompt_tokens=sum(int(row.get("usage",{}).get("prompt_tokens") or 0) for row in attempts.values()),
                completion_tokens=sum(int(row.get("usage",{}).get("completion_tokens") or 0) for row in attempts.values()),
                total_tokens=total_tokens, requests_without_token_usage=sum(not row.get("usage") for row in attempts.values())),
            finish_reason_counts=dict(Counter(row.get("finish_reason", "unavailable") for row in attempts.values())),
            truncated_responses=sum(row.get("finish_reason") == "length" for row in attempts.values()),
            generation_latency_seconds=numeric_distribution(generation),
            generation_request_seconds_sum=float(sum(generation)),
            generation_round_max_seconds_sum=float(sum(maximum_round_generation)),
            evaluation_request_seconds_sum=float(sum(row.get("evaluation_seconds",0) for row in records.values())),
            run_elapsed_seconds=summary.get("elapsed_seconds"), contexts=context_rows)
        policies[policy] = policy_row
        protocol_errors.extend(policy+": "+problem for problem in problems)
    if configs and any(config != next(iter(configs.values())) for config in configs.values()):
        protocol_errors.append("Writer/generation settings or request seed/round budgets differ across policy arms")
    full = len(configs) == 4 and all(policies[policy]["full_30_request_screen"] for policy in POLICIES) and not protocol_errors
    comparisons = []
    if full:
        rng = np.random.default_rng(args.bootstrap_seed)
        draws = rng.integers(0,15,size=(args.bootstrap,15))
        metrics = dict(request_valid_fraction=lambda row:row["valid_candidates"]/2,
                       two_pipeline_valid_fraction=lambda row:float(row["valid_candidates"] == 2),
                       two_valid_nonconstant_fraction=lambda row:row["two_valid_nonconstant"],
                       joint_diverse_pair_fraction=lambda row:row["joint_diverse_pair"],
                       joint_direction_invariant_diverse_pair_fraction=lambda row:row["joint_direction_invariant_diverse_pair"])
        for policy in POLICIES[1:]:
            record = dict(policy=policy, reference="original", paired_contexts=15, bootstrap_samples=args.bootstrap,
                          bootstrap_seed=args.bootstrap_seed, metrics={})
            for name,function in metrics.items():
                values = np.asarray([function(row) for row in policies[policy]["contexts"]])
                reference = np.asarray([function(row) for row in policies["original"]["contexts"]])
                delta = values-reference
                resamples = 100 * delta[draws].mean(axis=1)
                record["metrics"][name] = dict(policy_fraction=float(values.mean()),original_fraction=float(reference.mean()),
                    paired_difference_pp=float(100*delta.mean()),paired_context_bootstrap_ci95_pp=np.percentile(resamples,[2.5,97.5]).tolist(),
                    denominator="All 15 matched frozen contexts; failed/constant candidates are not dropped from joint outcomes")
            comparisons.append(record)
    return dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="complete_matched_screen" if full else "partial_or_protocol_pending", full_paired_comparisons_available=full,
        design="15 fixed historical parent contexts × four prompt policies × two seeded siblings = 120 requests",
        frozen_context_selection_rule=frozen.get("selection_rule"), adaptive_population_feedback=False,
        writer=manifest["writer"], writer_revision=manifest["writer_revision"],labels="is_hallu",
        probe_partition="512 prespecified training rows; no validation/test prediction arrays read",
        validity_definition="A candidate completed the pipeline and has a validation-evaluation score; score magnitude is not used",
        joint_diversity_definition="Both siblings pipeline-valid and nonconstant on the training probe, with directed Spearman rho < 0.995; denominator all 15 contexts",
        conditional_rho_definition="Descriptive Spearman correlation only among two-valid, nonconstant sibling pairs; denominator differs by policy",
        direction_invariant_supplement="abs(rho)<0.995 additionally excludes reversed rankings, which can be equivalent under paper direction-folded reporting",
        model_calls=False, validation_or_test_used_for_selection=False, source_files_sha256=hashes,
        prompt_policy_source_sha256=digest((ROOT/'src/experiments/prompt_policies.py').read_bytes()),
        frozen_contexts=source_contexts,generation_settings_by_policy=configs,policies=policies,
        paired_context_comparisons=comparisons,protocol_errors=protocol_errors,warnings=warnings,
        interpretation_limits=["Only 15 fixed contexts, one writer and one generation seed; bootstrap intervals describe context variability within this screen.",
            "Historical parent contexts come from one trajectory and need not be independent; percentile intervals are exploratory, without multiplicity adjustment.",
            "Diversity and interface validity are training-probe diagnostics; neither demonstrates better held-out AUROC or OOD generalization.",
            "Conditional sibling correlation excludes invalid/constant pairs and must be read alongside unconditional joint diversity.",
            "Latency includes server load/concurrency and does not measure isolated GPU throughput."])


def write_outputs(args, report):
    args.output.mkdir(parents=True,exist_ok=True)
    if report["full_paired_comparisons_available"] and not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x = np.arange(4)
            valid = [report["policies"][p]["pipeline_valid_requests"]/30*100 for p in POLICIES]
            joint = [report["policies"][p]["joint_diverse_pair_fraction_all_15"]*100 for p in POLICIES]
            fig,ax = plt.subplots(figsize=(8.4,4.3),layout="constrained")
            ax.bar(x-.19,valid,.38,label="Valid requests / 30",color="#4675a9")
            ax.bar(x+.19,joint,.38,label="Two valid, nonconstant, different siblings / 15",color="#de9347")
            ax.set(xticks=x,xticklabels=POLICIES,ylabel="Percent",ylim=(0,105),title="Matched prompt screen: validity and joint diversity")
            ax.legend(loc="upper center",fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            fig.savefig(args.output/'validity_diversity.png',dpi=220)
            fig.savefig(args.output/'validity_diversity.pdf')
            plt.close(fig)
        except ImportError as exc:
            report["warnings"].append("Plot unavailable: "+str(exc))
    (args.output/'prompt_screen_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (args.output/'contexts.csv').open('w') as stream:
        columns=['policy','context_id','complete','valid_candidates','two_valid_nonconstant','sibling_spearman_rho','joint_diverse_pair','joint_direction_invariant_diverse_pair']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows(dict(policy=p,**{key:row.get(key) for key in columns if key!='policy'}) for p in POLICIES for row in report['policies'][p]['contexts'])
    lines=['# Matched prompt-screen analysis','',report['design']+'.','',
        '**Status: '+report['status']+'**. Full paired comparisons require all four 30-request screens and complete provenance/probe diagnostics.','',
        'The primary outcome counts contexts with two valid, nonconstant, behaviorally different siblings (Spearman rho <0.995), divided by **all 15 contexts**. Conditional correlation is a separate descriptive measure among eligible pairs. No validation/test score magnitude is used.','',
        '| Policy | Saved requests / 30 | Completed contexts / 15 | Valid requests | Contexts with 0 / 1 / 2 valid | Joint diverse pairs / 15 | Conditional rho mean (pairs) | Truncated |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for p in POLICIES:
        r=report['policies'][p];hist=r['valid_candidates_per_context_histogram'];rho=r['conditional_sibling_spearman'];rho_text='pending' if rho['mean'] is None else f"{rho['mean']:.4f} ({rho['n']})"
        joint=f"{r['joint_diverse_pairs_observed']}/15" if r['full_30_request_screen'] else f"{r['joint_diverse_pairs_observed']} observed; pending"
        lines.append(f"| {p} | {r['requests_with_saved_outcomes']}/30 | {r['fully_evaluated_contexts']}/15 | {r['pipeline_valid_requests']} | {hist['0']} / {hist['1']} / {hist['2']} | {joint} | {rho_text} | {r['truncated_responses']} |")
    lines+=['','## Paired context comparisons','']
    if report['full_paired_comparisons_available']:
        lines+=[f"Percentile 95% intervals from {args.bootstrap:,} paired resamples of the same 15 contexts; exploratory and unadjusted for multiple comparisons.",'',
            '| Policy vs original | Valid-request change, pp [95% interval] | Joint-diverse-context change, pp [95% interval] |','| --- | ---: | ---: |']
        for r in report['paired_context_comparisons']:
            cells=[]
            for name in ['request_valid_fraction','joint_diverse_pair_fraction']:
                m=r['metrics'][name];lo,hi=m['paired_context_bootstrap_ci95_pp'];cells.append(f"{m['paired_difference_pp']:.2f} [{lo:.2f}, {hi:.2f}]")
            lines.append('| '+r['policy']+' | '+' | '.join(cells)+' |')
    else:
        lines+=['Pending: no cross-policy effect estimate is produced from an unfinished or inconsistent screen.']
    lines+=['','## Resource accounting','', '| Policy | Prompt tokens | Completion tokens | Total tokens | Sum request generation seconds | Sum round maximum generation seconds | Run elapsed seconds |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in POLICIES:
        r=report['policies'][p];usage=r['token_usage'];lines.append(f"| {p} | {usage['prompt_tokens']} | {usage['completion_tokens']} | {usage['total_tokens']} | {r['generation_request_seconds_sum']:.1f} | {r['generation_round_max_seconds_sum']:.1f} | {r['run_elapsed_seconds']} |")
    lines+=['','Generation requests overlap; their summed latency is not elapsed GPU time. Failure categories, candidate errors, per-context outcomes, prompt hashes and training-array hashes are retained in JSON.','','## Interpretation limits','']
    lines+=['- '+limit for limit in report['interpretation_limits']]
    if report['protocol_errors'] or report['warnings']:
        lines+=['','## Pending data and protocol issues','']+['- '+item for item in report['protocol_errors']+report['warnings']]
    (args.output/'PROMPT_SCREEN_ANALYSIS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=ROOT/'results/next_experiments/run_manifest.json')
    ap.add_argument('--contexts',type=Path,default=ROOT/'results/next_experiments/frozen_contexts.json')
    ap.add_argument('--probe',type=Path,default=ROOT/'results/next_experiments/training_probe.json')
    ap.add_argument('--output',type=Path,default=ROOT/'results/overnight20261021/prompt_screen_analysis')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--bootstrap-seed',type=int,default=9041)
    ap.add_argument('--no-plot',action='store_true')
    args=ap.parse_args()
    if args.bootstrap <= 0:
        raise ValueError('Bootstrap samples must be positive')
    report=analyse(args);write_outputs(args,report)
    print(json.dumps(dict(status=report['status'],full_paired_comparisons_available=report['full_paired_comparisons_available'],
        saved_requests=sum(r['requests_with_saved_outcomes'] for r in report['policies'].values()),
        valid_requests=sum(r['pipeline_valid_requests'] for r in report['policies'].values()),
        protocol_errors=len(report['protocol_errors']),output=str(args.output)),indent=2))


if __name__=='__main__':
    main()
