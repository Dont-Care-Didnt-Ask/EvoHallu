"""Audit saved evolution histories; no generation, candidate execution, or GPU.

Validation-vector diversity is diagnostic only. Future search-time novelty must
use training/probe predictions, not these validation vectors.
"""
from __future__ import annotations
import contextlib
import datetime as dt
import hashlib
import inspect
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr

from experiments.extension_trial import build_prompt
from packing.evaluate.uncertainty_quantification import found_formulas
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import get_initial_func
from packing.funsearch.programs_database import ProgramsDatabase
from packing.utils.functions import function_to_string


def read(path):
    return json.loads(path.read_text())


def fingerprint(values):
    _, ranks = np.unique(np.asarray(values, dtype=np.float32), return_inverse=True)
    return hashlib.sha256(ranks.astype('<i4').tobytes()).hexdigest()


def standardized_ranks(values):
    ranks = rankdata(values).astype(np.float64)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm else np.zeros_like(ranks)


def error_family(row):
    error = row.get('error') or row.get('train', {}).get('error') or row.get('val', {}).get('error') or 'unknown'
    if error.startswith('ImportError:'):
        return 'import_or_name'
    if 'NaN' in error or 'infinity' in error or 'float32' in error or 'float16' in error:
        return 'numerical_or_dtype'
    if any(s in error for s in ['shape', 'broadcast', 'indices', 'index', 'out of bounds', 'ragged', 'inhomogeneous', 'sequence.']):
        return 'shape_or_index'
    if any(s in error for s in ['Estimator', '__call__', '__init__', 'abstract class']):
        return 'class_interface'
    if any(s in error for s in ['ImportError', 'NameError', 'AttributeError', 'UnboundLocalError']):
        return 'import_or_name'
    if any(s in error for s in ['SyntaxError', 'No HallucinationDetector']):
        return 'parse_or_syntax'
    return 'other'


def main():
    root = Path.cwd()
    reference_names = {n for n, c in inspect.getmembers(found_formulas, inspect.isclass)
                       if c.__module__ == found_formulas.__name__}
    ref_npz = np.load(root / 'out/campaign/original_validation/stratified_val_scores.npz')
    reference = {n: standardized_ranks(np.asarray(ref_npz[n], dtype=np.float32)) for n in reference_names}
    reference_fingerprints = {fingerprint(ref_npz[n]): n for n in reference_names}
    runs, sampler_rows = [], []
    for config_path in sorted((root / 'out/logs/mod1').glob('*/trial_config.json')):
        folder = config_path.parent
        if not (folder / 'candidates.jsonl').exists():
            continue
        cfg = OmegaConf.create(read(config_path))
        excluded = (folder / 'exclude_from_evaluation.json').exists()
        if excluded or 'pilot' in folder.name:
            continue
        summary = read(folder / 'summary.json')
        rows = [json.loads(line) for line in (folder / 'candidates.jsonl').read_text().splitlines()]
        valid = [r for r in rows if 'score' in r.get('val', {})]
        baseline = np.load(folder / 'baseline_val_scores.npy')
        previous_ranks = [standardized_ranks(baseline)]
        fingerprints = {fingerprint(baseline)}
        exact_rank_repeats = near_repeats = reference_exact = 0
        constant = 0
        for row in valid:
            values = np.load(folder / f"val_scores_{row['code_hash']}.npy")
            fp = fingerprint(values)
            exact_rank_repeats += fp in fingerprints
            reference_exact += fp in reference_fingerprints
            fingerprints.add(fp)
            vector = standardized_ranks(values)
            if np.linalg.norm(vector) == 0:
                constant += 1
            rho = max(float(vector @ old) for old in previous_ranks)
            near_repeats += rho >= .995
            previous_ranks.append(vector)
        selected = next((r for r in valid if r['code_hash'] == summary['selected_candidate']), None)
        chosen_values = baseline if selected is None else np.load(folder / f"val_scores_{selected['code_hash']}.npy")
        chosen_vector = standardized_ranks(chosen_values)
        nearest = max(reference, key=lambda n: float(chosen_vector @ reference[n]))
        curves = {}
        for budget in [10, 20, 40, 60]:
            if len(rows) >= budget:
                prefix = [r['val']['score'] for r in rows[:budget] if 'score' in r.get('val', {})]
                curves[str(budget)] = max([summary['baseline_val']['score'], *prefix])
        errors = Counter(error_family(r) for r in rows if 'score' not in r.get('val', {}))
        valid_train = [r for r in rows if 'score' in r.get('train', {})]
        train_best = max(valid_train, key=lambda r:r['train']['score'], default=None)
        selection_disagrees = selected is not None and train_best is not None and selected['train']['score'] < train_best['train']['score']
        entry = dict(run=folder.name, writer=cfg.model.hf_model_id, mode=cfg.search_mode, seed=int(cfg.seed),
                     requests=len(rows), valid=len(valid), valid_fraction=len(valid)/len(rows),
                     invalid_error_families=dict(errors), truncated_responses=sum(r.get('finish_reason')=='length' for r in rows),
                     baseline_included_unique_rank_patterns=len(fingerprints), exact_rank_repeats=exact_rank_repeats,
                     near_rank_repeats_0995=near_repeats, constant_predictions=constant,
                     candidates_with_ranks_equal_to_a_supplied_formula=reference_exact,
                     selected_round_1based=None if selected is None else selected['round']+1,
                     selected_validation_auc=summary['selected_validation_auc_percent'],
                     best_validation_by_request_budget=curves,
                     train_validation_candidate_auc_spearman=float(spearmanr([r['train']['score'] for r in valid], [r['val']['score'] for r in valid]).statistic) if len(valid)>1 else None,
                     selected_train_auc_below_best_train=selection_disagrees,
                     selected_nearest_supplied_formula=nearest,
                     selected_nearest_supplied_formula_spearman=float(chosen_vector @ reference[nearest]),
                     selected_exact_saved_rank_match_to_supplied=fingerprint(chosen_values) in reference_fingerprints,
                     recorded_elapsed_seconds=summary['elapsed_seconds'])
        if cfg.search_mode == 'evolution':
            first, _, _ = get_initial_func(cfg.task)
            initial_code = function_to_string(first)
            captured = {}
            with contextlib.redirect_stdout(io.StringIO()):
                db = ProgramsDatabase(cfg.programdatabaseConfig)
                db.register_program(initial_code, summary['baseline_train']['score'], None, {})
                original_get_prompt = db.get_prompt
                def capture_prompt(*args, **kwargs):
                    result = original_get_prompt(*args, **kwargs)
                    captured['programs'], captured['scores'] = result[0], result[1]
                    return result
                db.get_prompt = capture_prompt
                improved = compared = 0
                for round_index in sorted({r['round'] for r in rows}):
                    chat, island = build_prompt(cfg, db, initial_code, summary['baseline_train']['score'], round_index)
                    assert chat == read(folder / f'prompt_{round_index:04d}.json'), (folder.name, round_index)
                    scores = np.array(list(db._islands[island]._clusters.keys()), dtype=float)
                    n_parents = min(len(scores), int(cfg.programdatabaseConfig.functions_per_prompt))
                    percentile = cfg.initial_percentile + round_index/int(cfg.num_rounds)*(cfg.final_percentile-cfg.initial_percentile)
                    threshold = np.percentile(scores, 100-percentile*100, method='nearest')
                    filtered = np.sum(scores > threshold) > n_parents
                    eligible = scores[scores > threshold] if filtered else scores
                    probabilities = softmax(eligible / float(cfg.programdatabaseConfig.temp))
                    sampler_rows.append(dict(run=folder.name, round_1based=round_index+1, island=island,
                        score_groups=len(scores), eligible_groups=len(eligible), quantile_filter_applied=bool(filtered),
                        relative_effective_sample_size=float(1/(len(probabilities)*np.sum(probabilities**2))),
                        best_group_probability=float(probabilities.max()),
                        largest_to_smallest_sampling_probability=float(probabilities.max()/probabilities.min()),
                        parent_train_scores=[float(s) for s in captured['scores']]))
                    batch = [r for r in rows if r['round']==round_index]
                    for row in batch:
                        assert row['island']==island
                        if 'score' in row.get('train', {}):
                            compared += 1
                            improved += row['train']['score'] > max(captured['scores'])
                            db.register_program(row['imports']+'\n'+row['code'], row['train']['score'], island, {})
                entry['valid_offspring_above_best_parent_train'] = improved
                entry['valid_offspring_compared_to_parents'] = compared
        runs.append(entry)
    primary = [r for r in runs if r['mode']=='evolution']
    assert len(primary)==7 and sum(r['requests'] for r in primary)==402 and sum(r['valid'] for r in primary)==283
    assert len(sampler_rows)==201
    multiple = [r for r in sampler_rows if r['eligible_groups']>1]
    rank_sensitivity = {}
    for alpha in [.5, 1., 2.]:
        effective_sizes, top_probabilities = [], []
        for row in multiple:
            n = row['eligible_groups']
            weights = np.arange(1, n+1, dtype=float)**(-alpha)
            probability = .8*weights/weights.sum() + .2/n
            effective_sizes.append(float(1/(n*np.sum(probability**2))))
            top_probabilities.append(float(probability.max()))
        rank_sensitivity[str(alpha)] = dict(uniform_mixture_fraction=.2,
            median_relative_effective_sample_size=float(np.median(effective_sizes)),
            median_best_group_probability=float(np.median(top_probabilities)),
            scope='Counterfactual probability distributions on recorded eligible-group counts only; no counterfactual offspring or quality measured')
    by_writer = {}
    for writer in sorted({r['writer'] for r in primary}):
        selected = [r for r in primary if r['writer']==writer]
        n, v = sum(r['requests'] for r in selected), sum(r['valid'] for r in selected)
        by_writer[writer] = dict(requests=n, valid=v, valid_fraction=v/n,
            rank_near_repeat_fraction=sum(r['near_rank_repeats_0995'] for r in selected)/v,
            invalid_error_families=dict(sum((Counter(r['invalid_error_families']) for r in selected), Counter())))
    summary = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), no_gpu_or_model_calls=True,
        generated_candidate_code_executed=False, source='Saved candidate ledgers, source-validation prediction arrays, and deterministic parent-sampling replay',
        novelty_scope='Posthoc saved-validation-rank diagnostics only. Search-time novelty must instead use training/probe predictions. Float32 saved ranks do not prove functional equivalence.',
        primary_evolution_runs=len(primary), primary_requests=402, primary_valid_candidates=283,
        by_writer=by_writer,
        sampler=dict(rounds_replayed=len(sampler_rows), prompt_mismatches=0,
            rounds_quantile_filter_applied=sum(r['quantile_filter_applied'] for r in sampler_rows),
            rounds_with_multiple_eligible_groups=len(multiple),
            median_relative_effective_sample_size=float(np.median([r['relative_effective_sample_size'] for r in multiple])),
            minimum_relative_effective_sample_size=float(min(r['relative_effective_sample_size'] for r in multiple)),
            median_largest_to_smallest_probability=float(np.median([r['largest_to_smallest_sampling_probability'] for r in multiple])),
            median_best_group_probability=float(np.median([r['best_group_probability'] for r in multiple])),
            p90_largest_to_smallest_probability=float(np.percentile([r['largest_to_smallest_sampling_probability'] for r in multiple],90)),
            rounds_with_at_least_five_groups_but_no_quantile_filter=sum(r['score_groups']>=5 and not r['quantile_filter_applied'] for r in sampler_rows)),
        rank_sampler_sensitivity=rank_sensitivity, runs=runs, sampler_rounds=sampler_rows)
    output=root/'results/evolution_tuning_audit.json'
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'runs','sampler_rounds'}},indent=2))
    for r in primary:
        print(json.dumps({k:r[k] for k in ['run','valid','exact_rank_repeats','near_rank_repeats_0995','selected_nearest_supplied_formula','selected_nearest_supplied_formula_spearman','valid_offspring_above_best_parent_train','valid_offspring_compared_to_parents']},ensure_ascii=False))


if __name__ == '__main__':
    main()
