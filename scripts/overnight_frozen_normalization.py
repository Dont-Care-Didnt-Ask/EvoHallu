"""Exploratory train-only normalization control for immutable C8741.

Never edits or replaces its parent program/primary. --execute is required.
The private NumPy proxy preserves feature math and changes only median returns.
"""
from __future__ import annotations
import argparse
import ast
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from scripts import overnight_batch_sensitivity as parent

ROOT=parent.ROOT
DIRECTORY=parent.DIRECTORY
OUTPUT=DIRECTORY/'frozen_train_normalization'
CASES=('original_1','original_128','original_512','shuffle451_512')
HARD_STOP=dt.datetime(2026,10,21,23,15,tzinfo=dt.timezone.utc)
CONFIG=ROOT/'out/logs/research20261021/uncertainty_quantification_next_evo_compositional_8741/trial_config.json'
TRAIN_MANIFEST=ROOT/'results/training_feature_cache_manifest.json'
sha=parent.sha
atomic_json=parent.atomic_json

class MedianProxy:
    """Forward all NumPy calls; capture or freeze the two medians per feature."""
    def __init__(self,statistics=None):
        import numpy as np
        self.original=np;self.statistics=statistics;self.calls=0;self.features=[]
    def __getattr__(self,name):return getattr(self.original,name)
    def median(self,arr,*args,**kwargs):
        index=self.calls;self.calls+=1
        if index>=8:raise ValueError('Unexpected additional median call in frozen source')
        feature=index//2
        if index%2==0:self.features.append(self.original.array(arr,copy=True))
        if self.statistics is None:return self.original.median(arr,*args,**kwargs)
        key='median' if index%2==0 else 'mad';item=self.statistics[feature]
        return self.original.dtype(item['dtype']).type(item[key])

class ProxiedEstimator:
    def __init__(self,estimator,statistics=None):self.estimator=estimator;self.statistics=statistics;self.last_features=None
    def __call__(self,stats):
        proxy=MedianProxy(self.statistics);namespace=self.estimator.__call__.__globals__;original=namespace['np'];namespace['np']=proxy
        try:result=self.estimator(stats)
        finally:namespace['np']=original
        if proxy.calls!=8 or len(proxy.features)!=4:raise ValueError('Expected exactly four robust_norm inputs/eight median calls')
        self.last_features=proxy.features
        return result

def source_contract():
    if sha(parent.SOURCE)!=parent.SOURCE_SHA:raise ValueError('Original frozen program changed')
    tree=ast.parse(parent.SOURCE.read_text())
    functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='robust_norm']
    if len(functions)!=1:raise ValueError('Unexpected normalization function structure')
    medians=[n for n in ast.walk(functions[0]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np' and n.func.attr=='median']
    allmedians=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np' and n.func.attr=='median']
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='robust_norm']
    if len(medians)!=2 or len(allmedians)!=2 or len(calls)!=4:raise ValueError('Median-only proxy assumptions fail')
    return dict(original_source_sha256=parent.SOURCE_SHA,nested_normalizer_count=1,median_calls_in_normalizer=2,normalizer_calls=4,other_median_calls=0,
                source_edited=False,ast_rewritten=False,proxy_scope='Private candidate class global np only, restored after each call')

def fit_statistics(feature_vectors):
    import numpy as np
    if len(feature_vectors)!=4:raise ValueError('Exactly four features required')
    statistics=[]
    for i,f in enumerate(feature_vectors):
        if f.ndim!=1 or not len(f) or not np.isfinite(f).all():raise ValueError('Invalid fitting feature')
        med=np.median(f);mad=np.median(np.abs(f-med))
        statistics.append(dict(feature=i+1,dtype=f.dtype.str,median=float(med),mad=float(mad),degenerate=bool(mad<1e-8),
                               minimum=float(np.min(f)),maximum=float(np.max(f)),rows=len(f)))
    return statistics

def prepare():
    contract=source_contract();base_path=DIRECTORY/'batch_sensitivity/PLAN.json';base=json.loads(base_path.read_text())
    complete=json.loads((DIRECTORY/'batch_sensitivity/BATCH_SENSITIVITY.json').read_text())
    if complete['status']!='complete_fixed_five_case_diagnostic' or not complete['original512_byte_parity_all9']:raise ValueError('Require completed parent diagnostic')
    cfg=json.loads(CONFIG.read_text());task=cfg['task'];manifest=json.loads(TRAIN_MANIFEST.read_text())
    if (task['sample_size'],task['val_size'],task['split_seed'],task['split_strategy'],task['chunk_size'])!=(0,.2,441,'stratified',512):raise ValueError('Original split/config changed')
    raw=ROOT/task['train_raw_path']
    if manifest['raw_sha256']!='8d34da6f189e8bfe67c0bca3e1101607f5b0887d5f1de41b4f373ef08f2f5ced' or manifest['raw_file']!=task['train_raw_path']:raise ValueError('Pinned original training-cache identity changed')
    if raw.stat().st_size!=manifest['raw_bytes']:raise ValueError('Training raw size changed')
    protected={**base['protected_source_sha256'],str(Path(__file__).resolve()):sha(__file__),str(CONFIG):sha(CONFIG),str(TRAIN_MANIFEST):sha(TRAIN_MANIFEST),
               str(ROOT/'src/packing/evaluate/uncertainty_quantification/task_uncertainty_quantification.py'):sha(ROOT/'src/packing/evaluate/uncertainty_quantification/task_uncertainty_quantification.py')}
    return dict(status='predeclared_exploratory_train_only_normalization',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),parent_source_sha256=parent.SOURCE_SHA,
                parent_batch_plan=str(base_path),parent_batch_plan_sha256=sha(base_path),prior_batch_report_sha256=sha(DIRECTORY/'batch_sensitivity/BATCH_SENSITIVITY.json'),
                source_contract=contract,protected_source_sha256=protected,datasets=base['datasets'],training_raw=str(raw),training_raw_sha256=manifest['raw_sha256'],
                training_csv=base['training_csv'],training_csv_sha256=base['training_csv_sha256'],fit_rows=5605,excluded_validation_rows=1402,
                split=dict(sample_size=0,val_size=.2,split_seed=441,split_strategy='stratified'),original_train_validation_and_test_chunk_size=512,
                normalization='Four medians/MADs from exact original5605 training rows only, captured by transparent proxy in512-row training chunks; feature formulas/coefs untouched',
                fixed_coefficients=[.4,.25,.2,.15],mad_multiplier=1.4826,degenerate_rule='MAD<1e-8 -> zero entire feature vector',
                evaluation_cases=list(CASES),shuffle='np.random.default_rng(451).permutation(n) per dataset, restore original positions; no labels used',
                invariance_requirement='Byte-identical frozen-control scores/NLL under chunks1/128/512/shuffle451 across all9 source datasets',
                parity_gate='Transparent proxy must reproduce original512 candidate/NLL/labels/groups/masks exactly on all9 before frozen-control cases',
                workers=4,maximum_wall_seconds=1800,hard_stop_utc=HARD_STOP.isoformat(),gpu_calls=0,new_generation=False,coefficient_fitting=False,
                validation_test_target_statistics_used_for_fit=False,labels_used='Only the original stratified split; never normalization fitting',target_evaluation=False,
                primary_replacement=False,scientific_limit='Exploratory control designed after test results were viewed; all cases retained, no test-based variant/normalization selection')

def verify(plan):
    for p,h in plan['protected_source_sha256'].items():
        if sha(p)!=h:raise ValueError('Protected source/config changed: '+p)
    if sha(plan['parent_batch_plan'])!=plan['parent_batch_plan_sha256']:raise ValueError('Parent plan changed')

def load_candidate(meta):
    from scripts.evaluate_cached_claims import load_estimators
    values,_=load_estimators([parent.SOURCE],meta)
    return values[parent.METHOD],values[parent.BASELINE]

def fit(plan):
    import numpy as np
    import torch
    from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _load_raw,_prepare_train_val_indices,_subset_raw
    from packing.evaluate.uncertainty_quantification.features import make_all_stats
    from scripts.evaluate_cached_claims import score_with_timeout
    verify(plan);torch.set_num_threads(1)
    if sha(plan['training_raw'])!=plan['training_raw_sha256'] or sha(plan['training_csv'])!=plan['training_csv_sha256']:raise ValueError('Training raw/CSV hash mismatch')
    raw=_load_raw(plan['training_raw']);train,val=_prepare_train_val_indices(raw,**plan['split'])
    if (len(raw['samples']),len(train),len(val))!=(7007,5605,1402) or len(np.intersect1d(train,val)):raise ValueError('Original row split mismatch')
    subset=_subset_raw(raw,train);candidate,_=load_candidate(raw['meta']);wrapped=ProxiedEstimator(candidate);pieces=[[] for _ in range(4)]
    for start in range(0,len(train),512):
        samples=subset['samples'][start:start+512]
        stats,_,_,_=make_all_stats(dict(samples=samples,meta=raw['meta']),input_regime='chunked',required_stats=['sequence_logprobs'])
        score_with_timeout(wrapped,stats,120)
        for i,x in enumerate(wrapped.last_features):pieces[i].append(x)
    features=[np.concatenate(x) for x in pieces];statistics=fit_statistics(features)
    weights=dict(format='C8741_frozen_training_median_mad_v1',parent_source_sha256=parent.SOURCE_SHA,helper_sha256=sha(__file__),statistics=statistics,
                 coefficients=[.4,.25,.2,.15],mad_multiplier=1.4826,source_fit_rows=5605,excluded_validation_rows=1402,train_only=True,training_raw_sha256=plan['training_raw_sha256'])
    atomic_json(OUTPUT/'normalization.json',weights)
    np.savez_compressed(OUTPUT/'fit_features_and_split.npz',features=np.column_stack(features),train_indices=train,excluded_validation_indices=val)
    atomic_json(OUTPUT/'FIT_RECEIPT.json',dict(status='frozen_train_only_statistics',source_contract=plan['source_contract'],statistics=statistics,
        weights_sha256=sha(OUTPUT/'normalization.json'),feature_and_split_sha256=sha(OUTPUT/'fit_features_and_split.npz'),
        training_raw_sha256=plan['training_raw_sha256'],training_rows=5605,validation_rows_excluded=1402,
        training_indices_sha256=__import__('hashlib').sha256(train.tobytes()).hexdigest(),excluded_validation_indices_sha256=__import__('hashlib').sha256(val.tobytes()).hexdigest(),
        fitting_labels_used=False,labels_used_only_to_reproduce_original_split=True,test_target_data_read_during_fit=False,source_program_rewritten=False,
        coefficient_or_threshold_selection=False,exploratory_after_test_view=True))

def worker(plan,index,phase):
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    verify(plan);torch.set_num_threads(1);e=plan['datasets'][index]
    if sha(e['cache'])!=e['raw_sha256'] or sha(e['csv'])!=e['csv_sha256'] or sha(e['reference_npz'])!=e['reference_npz_sha256']:raise ValueError('Dataset provenance changed')
    raw=torch.load(e['cache'],mmap=True,weights_only=False,map_location='cpu')
    with np.load(e['reference_npz']) as z:ref={k:z[k] for k in ('labels','claim_group_ids','duplicate_disjoint_mask',parent.METHOD,parent.BASELINE)}
    frame=pd.read_csv(e['csv'],usecols=['claim','is_hallu']);labels=frame.is_hallu.to_numpy(dtype=int)
    if len(raw['samples'])!=len(labels) or any(s['claim']!=str(c).strip() or float(s['label'])!=float(y) for s,c,y in zip(raw['samples'],frame.claim,labels)):raise ValueError('Raw/CSV rows differ')
    names=[' '.join(s['claim'].casefold().split()) for s in raw['samples']];groups,_=pd.factorize(np.array(names,dtype=object),sort=False)
    training=pd.read_csv(plan['training_csv'],usecols=['claim']);train_claims=set(training.claim.fillna('').astype(str).str.split().str.join(' ').str.casefold())
    strict=np.array([x not in train_claims for x in names],dtype=bool)
    if not all(parent.exact(a,ref[k]) for a,k in ((labels,'labels'),(groups,'claim_group_ids'),(strict,'duplicate_disjoint_mask'))):raise ValueError('Reference row identity differs')
    candidate,baseline=load_candidate(raw['meta'])
    statistics=None
    if phase=='frozen':
        weights=json.loads((OUTPUT/'normalization.json').read_text());fit_receipt=json.loads((OUTPUT/'FIT_RECEIPT.json').read_text())
        if sha(OUTPUT/'normalization.json')!=fit_receipt['weights_sha256'] or weights['parent_source_sha256']!=parent.SOURCE_SHA or weights['helper_sha256']!=sha(__file__):raise ValueError('Frozen statistics changed')
        statistics=weights['statistics']
    wrapped=ProxiedEstimator(candidate,statistics);cases=('original_512',) if phase=='reference' else CASES
    result=dict(status='running',dataset=e['name'],rows=len(labels),phase=phase,parent_source_sha256=parent.SOURCE_SHA,raw_sha256=e['raw_sha256'],reference_npz_sha256=e['reference_npz_sha256'],labels_groups_masks_exact=True,conditions=[])
    arrays=dict(labels=labels,claim_group_ids=groups,duplicate_disjoint_mask=strict,canonical_nll=ref[parent.BASELINE],original_primary512=ref[parent.METHOD]);first=None
    for case in cases:
        began=time.monotonic();scores,nll=parent.score_case(wrapped,baseline,raw,case,full_pack=phase=='reference')
        if not parent.exact(nll,ref[parent.BASELINE]):raise ValueError('Canonical NLL mismatch: '+case)
        if phase=='reference' and not parent.exact(scores,ref[parent.METHOD]):raise ValueError('Transparent proxy does not reproduce original primary512')
        if phase=='frozen':
            if first is None:first=scores.copy()
            elif not parent.exact(scores,first):raise ValueError('Frozen train normalization not exactly chunk/order invariant')
        auc=100*float(roc_auc_score(labels,scores));rho=float(spearmanr(scores,ref[parent.METHOD]).statistic) if np.ptp(scores)>0 and np.ptp(ref[parent.METHOD])>0 else None
        result['conditions'].append(dict(case=case,directed_auc_percent=auc,paper_auc_percent=max(auc,100-auc),canonical_nll_byte_exact=True,
            original512_byte_exact=parent.exact(scores,ref[parent.METHOD]),frozen_case_invariance_exact=phase=='frozen',spearman_vs_original512=rho,elapsed_seconds=time.monotonic()-began))
        arrays[case]=scores
    result['status']='passed_original_proxy_reference' if phase=='reference' else 'complete_invariant_frozen_control'
    folder=OUTPUT/phase;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/(e['name']+'_scores.npz'),**arrays);atomic_json(folder/(e['name']+'.json'),result)

def require_reference(plan):
    if len(plan['datasets'])!=9:raise ValueError('Nine datasets required')
    for e in plan['datasets']:
        d=json.loads((OUTPUT/'reference'/(e['name']+'.json')).read_text())
        if d['status']!='passed_original_proxy_reference' or d['parent_source_sha256']!=parent.SOURCE_SHA or d['raw_sha256']!=e['raw_sha256'] or not d['labels_groups_masks_exact'] or not d['conditions'][0]['original512_byte_exact'] or not d['conditions'][0]['canonical_nll_byte_exact']:raise ValueError('All9 transparent proxy reference gate failed')

def cancelled():
    return parent.cancelled() or (DIRECTORY/'frozen_train_normalization.cancel').exists()

def child(plan,phase,index,deadline,abort):
    if abort.is_set() or cancelled() or dt.datetime.now(dt.timezone.utc)>=deadline:raise TimeoutError('Control cancelled/cutoff before launch')
    folder=OUTPUT/phase;folder.mkdir(parents=True,exist_ok=True);name='fit' if phase=='fit' else plan['datasets'][index]['name']
    command=[sys.executable,str(Path(__file__).resolve()),'--worker-phase',phase]
    if index is not None:command+=['--dataset-index',str(index)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'src:.','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with (folder/(name+'.log')).open('w') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                if abort.is_set() or cancelled() or dt.datetime.now(dt.timezone.utc)>=deadline:raise TimeoutError('Control cancelled/global cutoff')
                time.sleep(.25)
            if proc.returncode:raise RuntimeError('Control child failed; inspect '+str(folder/(name+'.log')))
        finally:parent.terminate(proc)

def report(plan):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    require_reference(plan);rows=[]
    for e in plan['datasets']:
        d=json.loads((OUTPUT/'frozen'/(e['name']+'.json')).read_text())
        if d['status']!='complete_invariant_frozen_control' or len(d['conditions'])!=4 or not all(x['canonical_nll_byte_exact'] and x['frozen_case_invariance_exact'] for x in d['conditions']):raise ValueError('Incomplete invariant control')
        with np.load(OUTPUT/'frozen'/(e['name']+'_scores.npz')) as z:
            scores=z['original_512'];labels=z['labels'];original=z['original_primary512'];nll=z['canonical_nll']
            for case in CASES:
                if not parent.exact(scores,z[case]):raise ValueError('Saved-case invariance mismatch')
            ao=100*float(roc_auc_score(labels,original));an=100*float(roc_auc_score(labels,nll));af=100*float(roc_auc_score(labels,scores))
        rows.append(dict(dataset=e['name'],rows=e['rows'],frozen_directed=af,frozen_folded=max(af,100-af),original512_directed=ao,original512_folded=max(ao,100-ao),nll_directed=an,nll_folded=max(an,100-an),cases=d['conditions']))
    macro={key:float(np.mean([row[key] for row in rows])) for key in ('frozen_directed','frozen_folded','original512_directed','original512_folded','nll_directed','nll_folded')}
    macro['frozen_minus_original_folded_pp']=macro['frozen_folded']-macro['original512_folded'];macro['frozen_minus_nll_folded_pp']=macro['frozen_folded']-macro['nll_folded']
    final=dict(status='complete_exploratory_train_only_control',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),parent_source_sha256=parent.SOURCE_SHA,
        plan_sha256=sha(OUTPUT/'PLAN.json'),fit_receipt_sha256=sha(OUTPUT/'FIT_RECEIPT.json'),all9_original_proxy_primary_exact=True,all36_case_nll_exact=True,all36_frozen_scores_chunk_order_byte_invariant=True,
        primary_replaced=False,target_evaluated=False,new_generation=False,gpu_calls=0,all_fixed_cases_retained=True,macro=macro,per_dataset=rows,
        limitations=['Control designed after test-viewing; exploratory, no independent confirmatory claim.','Four medians/MADs fitted on exact original5605 training rows only; no coefficients selected.','Original candidate/primary512 remains unchanged; this derived control is separately named.','No interval or best-case selection; no target-model evaluation.'])
    atomic_json(OUTPUT/'FROZEN_NORMALIZATION_REPORT.json',final)
    lines=['# Train-only frozen-normalization control','', '**'+final['status']+'**. One normalization control derived from the immutable C8741 program; all four original coefficients and feature computations retained. Statistics use5605 original PopQA training rows only; validation/test/target feature distributions and labels do not determine normalization. Exploratory study designed after viewing prior test results.','', '|Method|Source9 directed|Source9 folded|','|---|---:|---:|',f"|Canonical NLL|{macro['nll_directed']:.4f}|{macro['nll_folded']:.4f}|",f"|Original C8741,512-row primary|{macro['original512_directed']:.4f}|{macro['original512_folded']:.4f}|",f"|Separate train-normalized control|{macro['frozen_directed']:.4f}|{macro['frozen_folded']:.4f}|",'', '|Dataset|Original512 folded|Frozen train-normalized folded|NLL folded|','|---|---:|---:|---:|']
    for row in rows:lines.append(f"|{row['dataset']}|{row['original512_folded']:.4f}|{row['frozen_folded']:.4f}|{row['nll_folded']:.4f}|")
    lines+=['', 'All9 transparent-proxy original512 predictions/NLL/labels/groups/masks reproduce the saved primary byte-for-byte before control scoring. All36 chunk/order cases have exactly identical frozen-control scores and canonical NLL across1/128/512 and the fixed seed451 shuffle. Original train/validation/benchmark scoring all used512-row chunks; there was no primary train/test chunk mismatch.','', 'The helper only intercepts the eight median calls in the private copy of the original class. It captures the four actual robust_norm input vectors; it does not rewrite feature math or alter original source bytes. Frozen median/MAD returns preserve the existing1.4826 scale and MAD<1e-8 zero-vector rule. The fitting receipt retains exact split indices and source/helper/cache hashes.','', 'No case replaces the primary, no target evaluation or coefficient fitting occurred, and no confidence interval or paper-SOTA claim is made. The control tests whether frozen source-training normalization yields a batch-independent deployable score; its measured quality is exploratory.']
    (OUTPUT/'FROZEN_NORMALIZATION_REPORT.md').write_text('\n'.join(lines)+'\n');return final

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--report-only',action='store_true');ap.add_argument('--worker-phase',choices=['fit','reference','frozen']);ap.add_argument('--dataset-index',type=int);args=ap.parse_args();os.chdir(ROOT)
    if args.worker_phase:
        plan=json.loads((OUTPUT/'PLAN.json').read_text());fit(plan) if args.worker_phase=='fit' else worker(plan,args.dataset_index,args.worker_phase);return
    if args.report_only:print(report(json.loads((OUTPUT/'PLAN.json').read_text()))['status']);return
    plan=prepare();OUTPUT.mkdir(parents=True,exist_ok=True);path=OUTPUT/'PLAN.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=plan[k] for k in plan if k!='created_utc'):raise ValueError('Control plan changed; refusing reuse')
        plan=old
    else:atomic_json(path,plan)
    if not args.execute:print('Prepared only; --execute required.');return
    if ROOT!=Path('/workspace/mod1'):raise ValueError('Real execution restricted to authorized pod')
    import fcntl
    lock=(OUTPUT/'execution.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUTPUT/'execution.json').exists() and json.loads((OUTPUT/'execution.json').read_text()).get('status')=='complete_exploratory_train_only_control':print('Already complete');return
    began=dt.datetime.now(dt.timezone.utc);deadline=min(HARD_STOP,began+dt.timedelta(seconds=1800));abort=threading.Event()
    if (deadline-began).total_seconds()<300 or cancelled():raise TimeoutError('Insufficient authorized control window')
    receipt=dict(status='running',started_utc=began.isoformat(),cutoff_utc=deadline.isoformat(),workers=4,gpu_calls=0,phases=[],plan_sha256=sha(path));atomic_json(OUTPUT/'execution.json',receipt)
    def interrupted(signum,frame):raise KeyboardInterrupt('Control signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        child(plan,'fit',None,deadline,abort);receipt['phases'].append('fit_train_only_complete');atomic_json(OUTPUT/'execution.json',receipt)
        for phase in ('reference','frozen'):
            if phase=='frozen':require_reference(plan)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(child,plan,phase,e['index'],deadline,abort) for e in plan['datasets']]
                try:
                    for f in concurrent.futures.as_completed(futures):f.result()
                except BaseException:abort.set();raise
            receipt['phases'].append(phase+'_all9_complete');atomic_json(OUTPUT/'execution.json',receipt)
        result=report(plan);receipt['status']=result['status']
    except BaseException as exc:
        abort.set();receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}');atomic_json(OUTPUT/'FROZEN_NORMALIZATION_REPORT.json',dict(status='failed_no_control_quality_claim',error=receipt['error'],macro=None));raise
    finally:receipt['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();atomic_json(OUTPUT/'execution.json',receipt)
    print(receipt['status'])

if __name__=='__main__':main()
