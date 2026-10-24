"""Separate, fixed source-nine position/length diagnostic; --execute required.

Reproduce the frozen 27B quadratic-sum winner and canonical NLL on every dataset
before any of four predeclared controls are scored. CPU/LP only, no fitting.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'results/overnight20261021'
OUTPUT = DIRECTORY / 'position_length_controls'
SOURCE = DIRECTORY / 'frozen/next27b_evo_diverse_2107__primary/frozen_estimator.py'
SOURCE_SHA = '6a399c68edd3d47e2ad09a611d44204f378514103329a4205422f08a6cb336a3'
PRIMARY = 'new/next27b_evo_diverse_2107__primary'
BASELINE = 'original/SequenceLogProb'
CONTROLS = ('lp_total_length', 'float32_nll_sum', 'float32_nll_mean', 'float32_quadratic_mean')
METHODS = ('canonical_nll', 'frozen_quadratic_sum') + CONTROLS
CONTRASTS = (
    ('float32_nll_sum', 'float32_nll_mean'),
    ('frozen_quadratic_sum', 'float32_quadratic_mean'),
    ('float32_quadratic_mean', 'float32_nll_mean'),
)
BOOTSTRAPS = 500
HARD_STOP = dt.datetime(2026, 10, 22, 2, 0, tzinfo=dt.timezone.utc)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def exact(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def control_scores(logprobs):
    """All four fixed LP-only controls, preserving the reference's n=0/1 cases.

    Quadratic mean divides the same float32 weighted sum by its float32 weight
    mass AFTER summation; it does not renormalize each weight before multiplying.
    """
    import numpy as np
    scores = {key: [] for key in CONTROLS}
    for sequence in logprobs:
        n = len(sequence)
        u = -np.asarray(sequence, dtype=np.float32)
        if u.ndim != 1 or not np.isfinite(u).all():
            raise ValueError('Expected finite one-dimensional token logprobs')
        scores['lp_total_length'].append(float(n))
        if n == 0:
            for key in CONTROLS[1:]:
                scores[key].append(0.0)
            continue
        if n == 1:
            weights = np.array([1.0], dtype=np.float32)
        else:
            indices = np.arange(n, dtype=np.float32)
            weights = (indices / (n - 1)) ** 2
        total = np.sum(u, dtype=np.float32)
        weighted_sum = np.sum(u * weights)
        mass = np.sum(weights, dtype=np.float32)
        scores['float32_nll_sum'].append(float(total))
        scores['float32_nll_mean'].append(float(np.mean(u, dtype=np.float32)))
        scores['float32_quadratic_mean'].append(float(weighted_sum / mass))
    # Same final float32 storage and evaluator promotion as the frozen candidate.
    return {key: np.asarray(value, dtype=np.float32).astype(np.float64)
            for key, value in scores.items()}


def instantiate(meta=None):
    from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
    from packing.evaluate.uncertainty_quantification.found_formulas import SequenceLogProb
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen 27B quadratic source changed')
    namespace = {'Estimator': Estimator}
    exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), namespace)
    context = {'supervised': False, 'meta': {'eval_meta': meta or {}}}
    return namespace['HallucinationDetector'](context), SequenceLogProb(context)


def make_plan():
    import numpy as np
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen 27B quadratic source hash mismatch')
    validation = ROOT / 'results/feature_cache_validation.json'
    archives = json.loads(validation.read_text())['archives']
    files = sorted(p for p in (ROOT / 'data/claims').rglob('*.csv') if 'popqa_train' not in p.name)
    if len(files) != 9:
        raise ValueError('Exactly nine source datasets required')
    datasets = []
    for index, csv in enumerate(files):
        name = csv.parent.name + '__' + csv.stem
        archived = [v for k, v in archives.items()
                    if '/mod1-eval-cache/' in k and Path(k).name == name + '.pt.gz']
        if len(archived) != 1:
            raise ValueError('Ambiguous pinned cache provenance: ' + name)
        expected = archived[0]
        references = []
        for path in sorted((DIRECTORY / 'evaluations').glob('*/' + name + '_scores.npz')):
            with np.load(path) as z:
                if PRIMARY not in z:
                    continue
                arrays = {k: z[k] for k in ('labels', 'claim_group_ids', 'duplicate_disjoint_mask', BASELINE, PRIMARY)}
            if references and not all(exact(arrays[k], references[0][1][k]) for k in arrays):
                raise ValueError('Conflicting saved primary references: ' + name)
            references.append((path, arrays))
        if not references:
            raise ValueError('Missing complete frozen27B primary predictions: ' + name)
        path, arrays = references[0]
        datasets.append(dict(index=index, name=name, csv=str(csv), csv_sha256=sha(csv),
                             cache=str(ROOT / 'data/restored/mod1-eval-cache' / (name + '.pt')),
                             raw_sha256=expected['raw_sha256'], raw_bytes=expected['raw_bytes'],
                             reference_npz=str(path), reference_npz_sha256=sha(path), rows=len(arrays['labels'])))
    training = ROOT / 'data/claims/popqa/popqa_train_filtered_10shot.csv'
    protected = [SOURCE, Path(__file__).resolve(), validation, ROOT / 'scripts/evaluate_cached_claims.py',
                 ROOT / 'src/packing/evaluate/uncertainty_quantification/features.py',
                 ROOT / 'src/packing/evaluate/uncertainty_quantification/found_formulas.py',
                 ROOT / 'src/packing/evaluate/uncertainty_quantification/estimator_base.py']
    context = ROOT / 'results/next_experiments/paper_reporting.json'
    history = json.loads(context.read_text())['methods']
    context_names = ('alignment/ClaimLength', 'alignment/ClaimOnlyMeanSurprisal', 'original/Perplexity',
                     'original/ExpWeightedNegLogProb', 'new/ablation_qwen_linear_position_nll')
    return dict(status='predeclared_exploratory_cpu_diagnostic', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                source=str(SOURCE), source_sha256=SOURCE_SHA, primary=PRIMARY, baseline=BASELINE,
                methods=list(METHODS), controls=list(CONTROLS), contrasts=[list(pair) for pair in CONTRASTS],
                formula_contract={
                    'lp_total_length': 'len(full cached sequence_logprobs); no claim crop',
                    'float32_nll_sum': 'sum(-asarray(lp,float32),dtype=float32)',
                    'float32_nll_mean': 'mean(-asarray(lp,float32),dtype=float32)',
                    'float32_quadratic_mean': 'same frozen float32 quadratic weighted sum divided AFTER summation by sum(same weights,dtype=float32)',
                    'weights': 'n=1:[1]float32; n>1:(arange(n,float32)/(n-1))**2; n=0 score0 for every control',
                    'storage': 'float32 output then float64 evaluator promotion; length follows same output storage'},
                source9_only=True, scoring_chunk_size=512, workers=4, bootstrap_samples=BOOTSTRAPS,
                bootstrap_unit='normalized_claim', bootstrap_seed='441+int(SHA256(dataset_name)[:8],16)',
                bootstrap_protocol='Shared claim-group multiplicities across all six methods and both all-row/strict scopes; fold per dataset per replicate before equal-weight nine-dataset macro. Strict scope uses the same full-cohort multiplicities restricted to its rows.',
                full_reference_gate='All nine frozen-program/NLL/labels/group-ID/strict-mask arrays byte-exact before any controls execute',
                feature_protocol='Existing make_all_stats chunked dependency gate, required_stats=[sequence_logprobs]; no top-k pack, GPU, raw feature extraction or scratch cache',
                maximum_wall_seconds=1800, hard_stop_utc=HARD_STOP.isoformat(), gpu_calls=0, fitting=False,
                new_generation=False, primary_replacement=False, all_variants_retained=True,
                training_csv=str(training), training_csv_sha256=sha(training),
                protected_source_sha256={str(path): sha(path) for path in protected}, datasets=datasets,
                historical_context_file=str(context), historical_context_sha256=sha(context),
                historical_context={name: {k: history[name][k] for k in ('raw_macro_auc_percent', 'direction_invariant_macro_auc_percent')}
                                    for name in context_names},
                limitations=['Designed after viewing existing benchmark scores: exploratory, not independent confirmation.',
                             'No best variant selection, fitting, candidate promotion or new winner replaces any primary.',
                             'Claim-only historical controls use a different token span; historical Perplexity has float16 reduction/exp ties.',
                             'Historical exponential weights sum to one; historical linear-position ablation is accumulated, not normalized.',
                             'The 14-feature head jointly uses length/NLLsum/NLLmean/exponentialmean; it does not establish standalone performance.',
                             'AUROC is nonlinear; the fixed contrasts are sensitivity checks, not a causal additive decomposition.',
                             'Claim bootstrap is conditional on one frozen selected candidate, not writer/search-seed uncertainty.'])


def verify_plan(plan):
    for path, expected in plan['protected_source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Pinned source changed: ' + path)
    if sha(plan['training_csv']) != plan['training_csv_sha256']:
        raise ValueError('Training CSV changed')
    if sha(plan['historical_context_file']) != plan['historical_context_sha256']:
        raise ValueError('Historical context changed')


def reference_arrays(entry):
    import numpy as np
    if sha(entry['reference_npz']) != entry['reference_npz_sha256']:
        raise ValueError('Saved primary changed')
    with np.load(entry['reference_npz']) as z:
        return {k: z[k] for k in ('labels', 'claim_group_ids', 'duplicate_disjoint_mask', BASELINE, PRIMARY)}


def validate_rows(raw, frame, training, reference, expected_rows):
    import numpy as np
    import pandas as pd
    if len(raw['samples']) != expected_rows or len(frame) != expected_rows:
        raise ValueError('Cache/CSV/reference row count mismatch')
    claims = frame.claim.astype(str).str.strip().tolist()
    labels = frame.is_hallu.to_numpy(dtype=int)
    if any(sample['claim'] != claim or float(sample['label']) != float(label)
           for sample, claim, label in zip(raw['samples'], claims, labels)):
        raise ValueError('Cache/CSV claim or label ordering mismatch')
    names = [' '.join(sample['claim'].casefold().split()) for sample in raw['samples']]
    groups, _ = pd.factorize(np.asarray(names, dtype=object), sort=False)
    known = set(training.claim.fillna('').astype(str).str.split().str.join(' ').str.casefold())
    strict = np.asarray([claim not in known for claim in names], dtype=bool)
    arrays = dict(labels=labels, claim_group_ids=groups, duplicate_disjoint_mask=strict)
    if not all(exact(arrays[key], reference[key]) for key in arrays):
        raise ValueError('Labels/groups/strict masks not byte-identical to primary')
    return arrays


def score_raw(raw, phase):
    import numpy as np
    from scripts.evaluate_cached_claims import score_with_timeout
    from packing.evaluate.uncertainty_quantification.features import make_all_stats
    candidate, baseline = instantiate(raw['meta'])
    values = {key: [] for key in ('canonical_nll', 'frozen_quadratic_sum') + (() if phase == 'reference' else CONTROLS)}
    for start in range(0, len(raw['samples']), 512):
        samples = raw['samples'][start:start + 512]
        stats, _, _, _ = make_all_stats(dict(samples=samples, meta=raw['meta']), input_regime='chunked', required_stats=['sequence_logprobs'])
        current = dict(canonical_nll=score_with_timeout(baseline, stats, 120),
                       frozen_quadratic_sum=score_with_timeout(candidate, stats, 120))
        if phase == 'controls':
            current.update(control_scores(stats['sequence_logprobs']))
        for key, scores in current.items():
            if scores.shape != (len(samples),) or not np.isfinite(scores).all():
                raise ValueError('Nonfinite or wrong-shaped scores: ' + key)
            values[key].append(scores)
    return {key: np.concatenate(parts) for key, parts in values.items()}


def paired_bootstraps(arrays):
    import numpy as np
    from scripts.evaluate_cached_claims import bootstrap_counts, weighted_auc_bootstrap
    name = arrays.pop('__dataset_name__')
    labels, groups, strict = arrays['labels'], arrays['claim_group_ids'], arrays['duplicate_disjoint_mask']
    prepared = {}
    for scope, mask in [('all', np.ones(len(labels), dtype=bool)), ('strict', strict)]:
        for key in METHODS:
            scores = arrays[key][mask]
            order = np.argsort(scores, kind='stable')
            starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
            prepared[scope, key] = (mask, order, starts)
    rng = np.random.default_rng(441 + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    results = {scope + '/' + key: [] for scope in ('all', 'strict') for key in METHODS}
    for _ in range(BOOTSTRAPS):
        counts = bootstrap_counts(groups, rng, 'claim')
        for (scope, key), (mask, order, starts) in prepared.items():
            results[scope + '/' + key].append(weighted_auc_bootstrap(labels[mask], arrays[key][mask], counts[mask], order, starts))
    return {key: np.asarray(value) for key, value in results.items()}


def worker(plan, entry, phase):
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    verify_plan(plan)
    if phase == 'controls':
        require_reference(plan)
    if sha(entry['csv']) != entry['csv_sha256']:
        raise ValueError('Dataset CSV changed')
    cache = Path(entry['cache'])
    if cache.stat().st_size != entry['raw_bytes'] or sha(cache) != entry['raw_sha256']:
        raise ValueError('Raw cache size/hash mismatch')
    raw = torch.load(cache, mmap=True, weights_only=False, map_location='cpu')
    reference = reference_arrays(entry)
    identities = validate_rows(raw, pd.read_csv(entry['csv'], usecols=['claim', 'is_hallu']),
                               pd.read_csv(plan['training_csv'], usecols=['claim']), reference, entry['rows'])
    values = score_raw(raw, phase)
    if not exact(values['canonical_nll'], reference[BASELINE]) or not exact(values['frozen_quadratic_sum'], reference[PRIMARY]):
        raise ValueError('Frozen quadratic or canonical NLL prediction byte parity failed')
    folder = OUTPUT / phase
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {**identities, **values}
    np.savez_compressed(folder / (entry['name'] + '_scores.npz'), **arrays)
    if phase == 'controls':
        boots = paired_bootstraps({**arrays, '__dataset_name__': entry['name']})
        np.savez_compressed(folder / (entry['name'] + '_bootstrap.npz'), **boots)
    receipt = dict(status='passed_reference' if phase == 'reference' else 'complete_four_controls',
                   dataset=entry['name'], rows=entry['rows'], source_sha256=SOURCE_SHA, raw_sha256=entry['raw_sha256'],
                   reference_npz_sha256=entry['reference_npz_sha256'], plan_sha256=sha(OUTPUT / 'PLAN.json'),
                   labels_groups_masks_exact=True, frozen_quadratic_sum_byte_exact=True, canonical_nll_byte_exact=True,
                   methods=list(values), bootstrap_samples=BOOTSTRAPS if phase == 'controls' else 0,
                   scores_sha256=sha(folder / (entry['name'] + '_scores.npz')))
    if phase == 'controls':
        receipt['bootstrap_sha256'] = sha(folder / (entry['name'] + '_bootstrap.npz'))
    dump(folder / (entry['name'] + '.json'), receipt)


def require_reference(plan):
    if len(plan['datasets']) != 9:
        raise ValueError('Require all nine reference datasets')
    for entry in plan['datasets']:
        value = json.loads((OUTPUT / 'reference' / (entry['name'] + '.json')).read_text())
        if not (value['status'] == 'passed_reference' and value['source_sha256'] == SOURCE_SHA
                and value['raw_sha256'] == entry['raw_sha256']
                and value['reference_npz_sha256'] == entry['reference_npz_sha256']
                and value['plan_sha256'] == sha(OUTPUT / 'PLAN.json')
                and all(value[key] is True for key in ('labels_groups_masks_exact', 'frozen_quadratic_sum_byte_exact', 'canonical_nll_byte_exact'))
                and value['scores_sha256'] == sha(OUTPUT / 'reference' / (entry['name'] + '_scores.npz'))):
            raise ValueError('All-nine byte-parity gate failed: ' + entry['name'])


def interval(values):
    import numpy as np
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(ci95=np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
                finite_replicates=len(finite), requested_replicates=BOOTSTRAPS)


def summarize_scope(predictions, bootstraps, names, scope):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    per_dataset, draws = [], {key: [] for key in METHODS}
    for name in names:
        z = predictions[name]
        mask = np.ones(len(z['labels']), dtype=bool) if scope == 'all' else z['duplicate_disjoint_mask']
        if len(np.unique(z['labels'][mask])) != 2:
            raise ValueError('Incomplete two-class scope: ' + name + '/' + scope)
        point = {key: 100 * float(roc_auc_score(z['labels'][mask], z[key][mask])) for key in METHODS}
        for key in METHODS:
            b = bootstraps[name][scope + '/' + key]
            if b.shape != (BOOTSTRAPS,):
                raise ValueError('Missing paired500 bootstrap draws')
            draws[key].append(100 * b)
        per_dataset.append(dict(dataset=name, rows=int(mask.sum()), methods={key: dict(directed_auc_percent=point[key], folded_auc_percent=max(point[key], 100 - point[key])) for key in METHODS}))
    stacked = {key: np.stack(value) for key, value in draws.items()}
    valid = np.logical_and.reduce([np.all(np.isfinite(a), axis=0) for a in stacked.values()])
    summary = dict(datasets=len(names), per_dataset=per_dataset, methods=[], contrasts=[], bootstrap_finite_all_method_all_dataset_replicates=int(valid.sum()))
    for key in METHODS:
        item = dict(method=key)
        for measure in ('directed', 'folded'):
            point = np.asarray([row['methods'][key][measure + '_auc_percent'] for row in per_dataset])
            baseline = np.asarray([row['methods']['canonical_nll'][measure + '_auc_percent'] for row in per_dataset])
            frozen = np.asarray([row['methods']['frozen_quadratic_sum'][measure + '_auc_percent'] for row in per_dataset])
            transformed = {k: a if measure == 'directed' else np.maximum(a, 100 - a) for k, a in stacked.items()}
            distribution = np.mean(transformed[key][:, valid], axis=0)
            item[measure] = dict(macro_auc_percent=float(point.mean()), ci95=interval(distribution),
                                 gain_vs_canonical_nll_pp=float((point - baseline).mean()),
                                 gain_vs_canonical_nll_ci95=interval(np.mean((transformed[key] - transformed['canonical_nll'])[:, valid], axis=0)),
                                 gain_vs_frozen_quadratic_sum_pp=float((point - frozen).mean()),
                                 gain_vs_frozen_quadratic_sum_ci95=interval(np.mean((transformed[key] - transformed['frozen_quadratic_sum'])[:, valid], axis=0)))
        summary['methods'].append(item)
    for left, right in CONTRASTS:
        pair = dict(left=left, right=right)
        for measure in ('directed', 'folded'):
            transformed = {k: a if measure == 'directed' else np.maximum(a, 100 - a) for k, a in stacked.items()}
            point = [row['methods'][left][measure + '_auc_percent'] - row['methods'][right][measure + '_auc_percent'] for row in per_dataset]
            pair[measure] = dict(difference_pp=float(np.mean(point)), ci95=interval(np.mean((transformed[left] - transformed[right])[:, valid], axis=0)))
        summary['contrasts'].append(pair)
    return summary


def report(plan):
    import numpy as np
    verify_plan(plan)
    require_reference(plan)
    predictions, bootstraps, files = {}, {}, {}
    for entry in plan['datasets']:
        folder = OUTPUT / 'controls'
        receipt_path = folder / (entry['name'] + '.json')
        receipt = json.loads(receipt_path.read_text())
        scores = folder / (entry['name'] + '_scores.npz')
        boots = folder / (entry['name'] + '_bootstrap.npz')
        if not (receipt['status'] == 'complete_four_controls' and receipt['source_sha256'] == SOURCE_SHA
                and receipt['raw_sha256'] == entry['raw_sha256'] and receipt['plan_sha256'] == sha(OUTPUT / 'PLAN.json')
                and receipt['scores_sha256'] == sha(scores) and receipt['bootstrap_sha256'] == sha(boots)
                and receipt['methods'] == list(METHODS) and receipt['bootstrap_samples'] == BOOTSTRAPS):
            raise ValueError('Incomplete or changed fixed diagnostic: ' + entry['name'])
        with np.load(scores) as z:
            predictions[entry['name']] = {k: z[k] for k in z.files}
        with np.load(boots) as z:
            bootstraps[entry['name']] = {k: z[k] for k in z.files}
        ref = reference_arrays(entry)
        actual = predictions[entry['name']]
        if not all(exact(actual[key], ref[key]) for key in ('labels', 'claim_group_ids', 'duplicate_disjoint_mask')) or not exact(actual['canonical_nll'], ref[BASELINE]) or not exact(actual['frozen_quadratic_sum'], ref[PRIMARY]):
            raise ValueError('Saved controls no longer match primary identities/reference')
        for path in (receipt_path, scores, boots):
            files[str(path)] = sha(path)
    names = [entry['name'] for entry in plan['datasets']]
    final = dict(status='complete_fixed_source9_position_length_diagnostic', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 source_sha256=SOURCE_SHA, plan_sha256=sha(OUTPUT / 'PLAN.json'), source9_only=True,
                 all_nine_primary_reference_byte_exact=True, all_four_controls_retained=True, primary_replacement=False,
                 fitting=False, gpu_calls=0, all_rows=summarize_scope(predictions, bootstraps, names, 'all'),
                 duplicate_disjoint=summarize_scope(predictions, bootstraps, names, 'strict'),
                 historical_context=plan['historical_context'], limitations=plan['limitations'], artifacts_sha256=files)
    dump(OUTPUT / 'POSITION_LENGTH_REPORT.json', final)
    lines = ['# Frozen 27B quadratic-position and length controls', '', '**' + final['status'] + '**. Exactly four predefined source9 CPU controls; all variants retained, no fitting or primary replacement.', '',
             'The original quadratic-sum winner, canonical NLL, labels, normalized-claim group IDs and duplicate-disjoint masks matched all nine saved primary arrays byte-for-byte before any control scoring. All calculations use the complete existing cached LP span and the same 512-row scoring chunks; no top-k pack, GPU calls, new features or model calls.', '',
             '| Method | Source9 directed | Source9 folded | Folded gain vs NLL [95% CI] | Strict folded |', '| --- | ---: | ---: | ---: | ---: |']
    strict = {r['method']: r for r in final['duplicate_disjoint']['methods']}
    for row in final['all_rows']['methods']:
        v = row['folded']; ci = v['gain_vs_canonical_nll_ci95']['ci95']
        interval_text = 'undefined' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        lines.append(f"| {row['method']} | {row['directed']['macro_auc_percent']:.3f} | {v['macro_auc_percent']:.3f} | {v['gain_vs_canonical_nll_pp']:+.3f} {interval_text} | {strict[row['method']]['folded']['macro_auc_percent']:.3f} |")
    lines += ['', 'The quadratic mean divides the exact float32 weighted sum by the float32 sum of the same weights. It preserves n=0 score0 and n=1 weight1. Uniform sum/mean also use float32 reductions; canonical NLL preserves the historical float16 reduction. The length control measures total cached-LP length, whereas historical ClaimLength counts claim tokens.', '',
              'The three predefined contrasts compare uniform sum versus uniform mean (accumulation), quadratic sum versus quadratic mean (weight mass), and quadratic mean versus uniform mean (positional emphasis). These are sensitivity contrasts: AUROC is nonlinear, so their effects are not a causal additive decomposition.', '',
              'Intervals use 500 shared normalized-claim bootstrap multiplicities per dataset. Strict sensitivity restricts those same draws to claims absent from the entire PopQA training CSV. Fold each dataset within each draw, then average all nine equally; only replicates finite for every method and dataset enter intervals. No incomplete dataset macro is reported.', '',
              'Historical context (already available; different spans/precision where stated):', '', '| Historical method | Directed | Folded |', '| --- | ---: | ---: |']
    for name, row in plan['historical_context'].items():
        lines.append(f"| {name} | {row['raw_macro_auc_percent']:.3f} | {row['direction_invariant_macro_auc_percent']:.3f} |")
    lines += ['', 'Historical Perplexity implements -exp(mean(LP)); this is monotonic in mean NLL in real arithmetic, with float16 reduction/exponential ties. Historical exponential-weighted NLL normalizes weights to unit mass. The linear-position ablation retains accumulated length dependence. These are contextual results, not newly scored controls.', '',
              'This diagnostic was designed after benchmark results were viewed and is exploratory. No highest-scoring variant is selected or promoted. Bootstrap uncertainty conditions on this one frozen validation-selected program; it does not estimate writer/search-seed uncertainty. No target1B claim is made. The original primary and all execution/evolution manifests remain unchanged.']
    (OUTPUT / 'POSITION_LENGTH_REPORT.md').write_text('\n'.join(lines) + '\n')
    return final


def cancelled():
    return any((DIRECTORY / name).exists() for name in ('watchdog.cancel', 'manual_stop', 'manual_stop.json', 'position_length_controls.cancel'))


def terminate(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def run_child(entry, phase, deadline, abort):
    if abort.is_set() or cancelled() or dt.datetime.now(dt.timezone.utc) >= deadline:
        raise TimeoutError('No diagnostic launch after cancellation/cutoff')
    folder = OUTPUT / phase
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(entry['index']), '--phase', phase]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': 'src:.', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (folder / (entry['name'] + '.log')).open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                if abort.is_set() or cancelled() or dt.datetime.now(dt.timezone.utc) >= deadline:
                    raise TimeoutError('Diagnostic cancelled/global cutoff')
                time.sleep(.25)
            if proc.returncode:
                raise RuntimeError('Diagnostic worker failed; see ' + str(folder / (entry['name'] + '.log')))
        finally:
            terminate(proc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--worker', type=int)
    ap.add_argument('--phase', choices=['reference', 'controls'])
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.worker is not None:
        if args.phase is None:
            ap.error('--worker requires --phase')
        plan = json.loads((OUTPUT / 'PLAN.json').read_text())
        worker(plan, plan['datasets'][args.worker], args.phase)
        return
    if args.report_only:
        print(report(json.loads((OUTPUT / 'PLAN.json').read_text()))['status'])
        return
    plan = make_plan()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'PLAN.json'
    if path.exists():
        original = json.loads(path.read_text())
        if any(original[key] != plan[key] for key in plan if key != 'created_utc'):
            raise ValueError('Immutable diagnostic PLAN changed; refusing reuse')
        plan = original
    else:
        dump(path, plan)
    if not args.execute:
        print('Prepared PLAN only; --execute required for any scoring.')
        return
    if ROOT != Path('/workspace/mod1'):
        raise ValueError('Real scoring restricted to the authorized pod')
    import fcntl
    lock = (OUTPUT / 'execution.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = OUTPUT / 'execution.json'
    if prior.exists() and json.loads(prior.read_text()).get('status') == 'complete_fixed_source9_position_length_diagnostic':
        print('Already complete; use --report-only to regenerate from saved arrays.')
        return
    verify_plan(plan)
    started = dt.datetime.now(dt.timezone.utc)
    deadline = min(HARD_STOP, started + dt.timedelta(seconds=1800))
    if (deadline - started).total_seconds() < 300 or cancelled():
        raise TimeoutError('Insufficient authorized diagnostic window')
    abort = threading.Event()
    receipt = dict(status='running', started_utc=started.isoformat(), cutoff_utc=deadline.isoformat(), workers=4,
                   gpu_calls=0, plan_sha256=sha(path), phases=[])
    dump(OUTPUT / 'execution.json', receipt)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Diagnostic signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for phase in ('reference', 'controls'):
            if phase == 'controls':
                require_reference(plan)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run_child, entry, phase, deadline, abort) for entry in plan['datasets']]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                except BaseException:
                    abort.set()
                    raise
            receipt['phases'].append(dict(phase=phase, status='complete', datasets=9, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            dump(OUTPUT / 'execution.json', receipt)
        # Bound reporting as well as owned worker groups by the shared cutoff.
        remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining <= 0 or cancelled():
            raise TimeoutError('No report after global cutoff')
        def expired(signum, frame):
            raise TimeoutError('Report global cutoff')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            receipt['status'] = report(plan)['status']
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException as exc:
        abort.set()
        receipt.update(status='failed_no_incomplete_macro', error=f'{type(exc).__name__}: {exc}')
        dump(OUTPUT / 'POSITION_LENGTH_REPORT.json', dict(status=receipt['status'], error=receipt['error'], gpu_calls=0,
                                                       primary_replacement=False, all_rows=None, duplicate_disjoint=None))
        raise
    finally:
        receipt['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        dump(OUTPUT / 'execution.json', receipt)
    print(receipt['status'])


if __name__ == '__main__':
    main()
