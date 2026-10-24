"""Isolated source-nine chunk diagnostic of the already-frozen C8741 program.

Five fixed cases, no fitting/selection, no GPU; --execute required for scoring.
The original 512-row scores must match every saved primary byte exactly first.
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
import time

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'results/overnight20261021'
OUTPUT=DIRECTORY/'batch_sensitivity'
METHOD='new/next_evo_compositional_8741__primary'
SOURCE=DIRECTORY/'frozen/next_evo_compositional_8741__primary/frozen_estimator.py'
SOURCE_SHA='a96fdf5af49a1a6f0f6426d79e2ef60b596be1d7ffd38a672afb6338523cc884'
BASELINE='original/SequenceLogProb'
CASES=('original_128','original_512','original_2048','original_full','shuffle451_512')
HARD_STOP=dt.datetime(2026,10,22,2,0,tzinfo=dt.timezone.utc)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def permutation(n,case):
    import numpy as np
    return np.random.default_rng(451).permutation(n) if case=='shuffle451_512' else np.arange(n)

def chunk_size(n,case):
    return n if case=='original_full' else int(case.rsplit('_',1)[1])

def prevalence(labels,size=512):
    import numpy as np
    rates=[float(np.mean(labels[k:k+size])) for k in range(0,len(labels),size)]
    return dict(chunk_size=size,chunks=len(rates),minimum=min(rates),maximum=max(rates),last_chunk_rows=(len(labels)-1)%size+1,
                prevalence_by_chunk=rates,order='original CSV/cache row order; no label-based ordering')

def score_case(estimator,baseline,raw,case,timeout=600,full_pack=False):
    """Use the existing feature builder and deep-copy scoring wrapper unchanged."""
    import numpy as np
    from scripts.evaluate_cached_claims import score_with_timeout
    from packing.evaluate.uncertainty_quantification.features import make_all_stats
    n=len(raw['samples']);order=permutation(n,case);size=chunk_size(n,case)
    scores=np.empty(n,dtype=np.float64);nll=np.empty(n,dtype=np.float64)
    for start in range(0,n,size):
        indices=order[start:start+size];samples=[raw['samples'][int(j)] for j in indices]
        kwargs=dict(input_regime='logits') if full_pack else dict(input_regime='chunked',required_stats=['sequence_logprobs'])
        stats,y,_,_=make_all_stats(dict(samples=samples,meta=raw['meta']),**kwargs)
        stats['input_tokens']=[s['input_ids'].tolist() for s in samples]
        current=score_with_timeout(estimator,stats,timeout)
        current_nll=score_with_timeout(baseline,stats,timeout)
        if current.shape!=(len(indices),) or not np.isfinite(current).all():raise ValueError('Nonfinite or wrong-shaped candidate output')
        if current_nll.shape!=(len(indices),) or not np.isfinite(current_nll).all():raise ValueError('Nonfinite or wrong-shaped NLL output')
        scores[indices]=current;nll[indices]=current_nll
    return scores,nll

def make_plan():
    import numpy as np
    if sha(SOURCE)!=SOURCE_SHA:raise ValueError('Frozen C8741 source hash mismatch')
    archives=json.loads((ROOT/'results/feature_cache_validation.json').read_text())['archives']
    files=sorted(p for p in (ROOT/'data/claims').rglob('*.csv') if 'popqa_train' not in p.name)
    if len(files)!=9:raise ValueError('Require exactly nine source datasets')
    rows=[]
    for index,csv in enumerate(files):
        name=csv.parent.name+'__'+csv.stem;cache=ROOT/'data/restored/mod1-eval-cache'/(name+'.pt')
        expected=next(v for k,v in archives.items() if '/mod1-eval-cache/' in k and Path(k).name==name+'.pt.gz')
        if not cache.is_file() or cache.stat().st_size!=expected['raw_bytes']:raise ValueError('Missing/wrong cache size: '+str(cache))
        references=[]
        for p in sorted((DIRECTORY/'evaluations').glob('*/'+name+'_scores.npz')):
            with np.load(p) as z:
                if METHOD not in z:continue
                current={k:z[k] for k in ('labels','duplicate_disjoint_mask','claim_group_ids',BASELINE,METHOD)}
            if references and not all(exact(current[k],references[0][1][k]) for k in current):raise ValueError('Conflicting saved primary cohorts: '+name)
            references.append((p,current))
        if not references:raise ValueError('Missing completed original primary: '+name)
        p,z=references[0]
        rows.append(dict(index=index,name=name,csv=str(csv),csv_sha256=sha(csv),cache=str(cache),raw_sha256=expected['raw_sha256'],raw_bytes=expected['raw_bytes'],
                         reference_npz=str(p),reference_npz_sha256=sha(p),rows=len(z['labels'])))
    protected=[SOURCE,Path(__file__).resolve(),ROOT/'scripts/evaluate_cached_claims.py',ROOT/'src/packing/evaluate/uncertainty_quantification/features.py']
    training=ROOT/'data/claims/popqa/popqa_train_filtered_10shot.csv'
    return dict(status='predeclared_cpu_diagnostic',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source=str(SOURCE),source_sha256=SOURCE_SHA,method=METHOD,baseline=BASELINE,
                cases=list(CASES),shuffle='Independent np.random.default_rng(451).permutation(n) per dataset; restore original positions; never use labels',
                workers=4,maximum_wall_seconds=1800,hard_stop_utc=HARD_STOP.isoformat(),gpu_calls=0,refitting=False,new_generation=False,primary_replacement=False,
                feature_protocol='Reference512 uses exact original logits pack/deepcopy wrapper. Other cases use the same make_all_stats chunked dependency gate for sequence_logprobs; formula is untouched. NLL parity required in every case.',
                full_reference_gate='All nine original512 candidate/NLL/labels/group IDs/strict masks must be byte-identical before any variant workers launch',
                training_csv=str(training),training_csv_sha256=sha(training),protected_source_sha256={str(p):sha(p) for p in protected},datasets=rows)

def worker(plan,entry,phase,out):
    import numpy as np
    import pandas as pd
    import torch
    from scripts.evaluate_cached_claims import load_estimators
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    torch.set_num_threads(1)
    for p,h in plan['protected_source_sha256'].items():
        if sha(p)!=h:raise ValueError('Protected source changed: '+p)
    if sha(entry['reference_npz'])!=entry['reference_npz_sha256'] or sha(entry['csv'])!=entry['csv_sha256'] or sha(plan['training_csv'])!=plan['training_csv_sha256']:raise ValueError('Reference/CSV changed')
    if sha(entry['cache'])!=entry['raw_sha256']:raise ValueError('Raw cache SHA mismatch')
    raw=torch.load(entry['cache'],mmap=True,weights_only=False,map_location='cpu')
    with np.load(entry['reference_npz']) as z:reference={k:z[k] for k in ('labels','duplicate_disjoint_mask','claim_group_ids',BASELINE,METHOD)}
    frame=pd.read_csv(entry['csv'],usecols=['claim','is_hallu'])
    if len(raw['samples'])!=entry['rows'] or len(frame)!=entry['rows']:raise ValueError('Row count mismatch')
    claims=frame.claim.astype(str).str.strip().tolist();labels=frame.is_hallu.to_numpy(dtype=int)
    if any(s['claim']!=c or float(s['label'])!=float(y) for s,c,y in zip(raw['samples'],claims,labels)):raise ValueError('Raw/CSV row or label mismatch')
    names=[' '.join(s['claim'].casefold().split()) for s in raw['samples']]
    groups,_=pd.factorize(np.asarray(names,dtype=object),sort=False)
    training=pd.read_csv(plan['training_csv'],usecols=['claim']);train=set(training.claim.fillna('').astype(str).str.split().str.join(' ').str.casefold())
    strict=np.array([c not in train for c in names],dtype=bool)
    if not all(exact(a,reference[k]) for a,k in ((labels,'labels'),(groups,'claim_group_ids'),(strict,'duplicate_disjoint_mask'))):raise ValueError('Labels/group IDs/strict mask not byte-identical')
    estimators,_=load_estimators([Path(plan['source'])],raw['meta'])
    candidate=estimators[METHOD];baseline=estimators[BASELINE]
    result=dict(dataset=entry['name'],rows=entry['rows'],phase=phase,status='running',source_sha256=SOURCE_SHA,raw_sha256=entry['raw_sha256'],reference_npz_sha256=entry['reference_npz_sha256'],
                labels_groups_masks_exact=True,original512_label_prevalence=prevalence(labels),conditions=[])
    arrays=dict(labels=labels,claim_group_ids=groups,duplicate_disjoint_mask=strict,canonical_nll=reference[BASELINE])
    chosen=('original_512',) if phase=='reference' else tuple(c for c in CASES if c!='original_512')
    for case in chosen:
        started=time.monotonic();scores,nll=score_case(candidate,baseline,raw,case,full_pack=phase=='reference')
        if not exact(nll,reference[BASELINE]):raise ValueError('NLL parity failed: '+case)
        if case=='original_512' and not exact(scores,reference[METHOD]):raise ValueError('Original512 candidate parity failed')
        rho=float(spearmanr(scores,reference[METHOD]).statistic) if np.ptp(scores)>0 and np.ptp(reference[METHOD])>0 else None
        auc=100*float(roc_auc_score(labels,scores))
        result['conditions'].append(dict(case=case,chunk_size=chunk_size(len(labels),case),directed_auc_percent=auc,paper_auc_percent=max(auc,100-auc),
                                         spearman_vs_original512=rho,exact_original512_scores=exact(scores,reference[METHOD]),maximum_absolute_score_difference=float(np.max(np.abs(scores-reference[METHOD]))),
                                         nonconstant=bool(np.ptp(scores)>0),canonical_nll_byte_exact=True,elapsed_seconds=time.monotonic()-started))
        arrays[case]=scores
    result['status']='passed_reference' if phase=='reference' else 'complete_conditions'
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/(entry['name']+'_scores.npz'),**arrays);atomic_json(out/(entry['name']+'.json'),result)

def cancelled():
    return any((DIRECTORY/x).exists() for x in ('watchdog.cancel','manual_stop','manual_stop.json','batch_sensitivity.cancel'))

def require_all_reference(plan):
    if len(plan['datasets'])!=9:raise ValueError('Nine reference datasets required')
    for e in plan['datasets']:
        p=OUTPUT/'reference'/(e['name']+'.json');v=json.loads(p.read_text())
        if not (v['status']=='passed_reference' and v['source_sha256']==SOURCE_SHA and v['raw_sha256']==e['raw_sha256']
                and len(v['conditions'])==1 and v['conditions'][0]['case']=='original_512'
                and v['conditions'][0]['exact_original512_scores'] and v['conditions'][0]['canonical_nll_byte_exact']
                and v['labels_groups_masks_exact']):raise ValueError('Nine-dataset reference gate failed: '+e['name'])

def terminate(child):
    if child.poll() is not None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=5)

def run_child(entry,phase,deadline,abort):
    if abort.is_set() or cancelled() or dt.datetime.now(dt.timezone.utc)>=deadline:raise TimeoutError('No diagnostic child launch after cancellation/cutoff')
    folder=OUTPUT/phase;folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(entry['index']),'--phase',phase]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'src:.','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    log=folder/(entry['name']+'.log')
    with log.open('w') as f:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                if abort.is_set() or cancelled() or dt.datetime.now(dt.timezone.utc)>=deadline:raise TimeoutError('Audit cancelled/global cutoff')
                time.sleep(.25)
            if child.returncode:raise RuntimeError('Diagnostic child failed: '+str(log))
        finally:terminate(child)
    return json.loads((folder/(entry['name']+'.json')).read_text())

def report(plan):
    import numpy as np
    import csv
    rows=[];errors=[]
    for entry in plan['datasets']:
        found={}
        for phase in ('reference','variants'):
            p=OUTPUT/phase/(entry['name']+'.json')
            if not p.exists():errors.append(str(p));continue
            data=json.loads(p.read_text())
            if data['source_sha256']!=SOURCE_SHA or data['raw_sha256']!=entry['raw_sha256']:raise ValueError('Stale diagnostic output')
            for row in data['conditions']:found[row['case']]={**row,'dataset':entry['name'],'rows':entry['rows'],'original512_label_prevalence':data['original512_label_prevalence']}
        if set(found)!=set(CASES):errors.append('Missing fixed cases: '+entry['name'])
        rows.extend(found[c] for c in CASES if c in found)
    complete=not errors and len(rows)==45
    macros=[]
    if complete:
        for case in CASES:
            part=[r for r in rows if r['case']==case]
            macros.append(dict(case=case,datasets=9,directed_macro_auc_percent=float(np.mean([r['directed_auc_percent'] for r in part])),paper_macro_auc_percent=float(np.mean([r['paper_auc_percent'] for r in part]))))
    try:require_all_reference(plan);reference_parity=True
    except (ValueError,OSError,KeyError):reference_parity=False
    final=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='complete_fixed_five_case_diagnostic' if complete and reference_parity else 'partial_no_macro',source_sha256=SOURCE_SHA,
               plan_sha256=sha(OUTPUT/'PLAN.json'),all_conditions_retained=True,no_best_condition_selection=True,original512_byte_parity_all9=reference_parity,
               gpu_calls=0,new_generation=False,refit=False,primary_replacement=False,macros=macros,per_dataset=rows,missing=errors,
               limits=['Point sensitivity only; no confidence interval or test-based condition selection.','Fixed512 remains the authoritative primary.','Batch median/MAD is an unsupervised function of current evaluation rows; it uses no labels.','Analytically, a singleton scoring batch has all four MADs zero and produces score0.'])
    atomic_json(OUTPUT/'BATCH_SENSITIVITY.json',final)
    with (OUTPUT/'BATCH_SENSITIVITY.csv').open('w') as f:
        fields=['dataset','case','rows','chunk_size','directed_auc_percent','paper_auc_percent','spearman_vs_original512','maximum_absolute_score_difference','exact_original512_scores','canonical_nll_byte_exact']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Frozen compositional8741 batch sensitivity','',f"Status: **{final['status']}**. All five conditions were fixed before scoring; no condition replaces the original512 primary. Source SHA256 `{SOURCE_SHA}`. No model/GPU calls, new generation or refitting.",'','Original512 must reproduce all nine saved primary predictions, labels, NLL, claim-group IDs and duplicate-disjoint masks byte-for-byte before variant scoring. The original fixed512 protocol remains authoritative.','', '| Condition | Datasets | Directed macro | Folded macro |','| --- | ---: | ---: | ---: |']
    for m in macros:lines.append(f"| {m['case']} |9/9|{m['directed_macro_auc_percent']:.4f}|{m['paper_macro_auc_percent']:.4f}|")
    lines+=['','| Dataset / condition | Directed | Folded | Rank rho vs512 |','| --- | ---: | ---: | ---: |']
    for row in rows:lines.append(f"|{row['dataset']} /{row['case']}|{row['directed_auc_percent']:.4f}|{row['paper_auc_percent']:.4f}|{row['spearman_vs_original512'] if row['spearman_vs_original512'] is not None else 'undefined (constant)'}|")
    lines+=['','## Original-order label composition','', '|Dataset|512chunk prevalence min|max|last chunk rows|','|---|---:|---:|---:|']
    for e in plan['datasets']:
        part=[r for r in rows if r['dataset']==e['name']]
        if part:
            p=part[0]['original512_label_prevalence'];lines.append(f"|{e['name']}|{p['minimum']:.4f}|{p['maximum']:.4f}|{p['last_chunk_rows']}|")
    lines+=['','The class robust-normalizes each of four feature vectors using the current batch median/MAD; for one row every MAD is0 and the score is0 analytically. This is batch-sensitive unsupervised normalization, not target-label fitting. Original order and the seed451 random permutation were specified without inspecting labels. Their label-prevalence ranges are descriptive, not a new sorting rule.','', 'Differences across these fixed cases measure protocol sensitivity of one already-selected program. Do not select the highest-AUROC case, promote another chunk size, or reinterpret these numbers as a new independent detector result. No intervals are asserted. All nine datasets are required for a macro; partial runs have none.']
    (OUTPUT/'BATCH_SENSITIVITY.md').write_text('\n'.join(lines)+'\n')
    return final

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--report-only',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--phase',choices=['reference','variants']);args=ap.parse_args()
    os.chdir(ROOT)
    if args.worker is not None:
        plan=json.loads((OUTPUT/'PLAN.json').read_text());worker(plan,plan['datasets'][args.worker],args.phase,OUTPUT/args.phase);return
    if args.report_only:
        result=report(json.loads((OUTPUT/'PLAN.json').read_text()));print(result['status']);return
    plan=make_plan();OUTPUT.mkdir(parents=True,exist_ok=True)
    existing=OUTPUT/'PLAN.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if any(old[k]!=plan[k] for k in plan if k!='created_utc'):raise ValueError('Existing diagnostic plan differs; refusing reuse')
        plan=old
    else:atomic_json(existing,plan)
    if not args.execute:print('Prepared only; --execute required.');return
    if ROOT!=Path('/workspace/mod1'):raise ValueError('Real execution restricted to the authorized pod')
    import fcntl
    lock=(OUTPUT/'execution.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=OUTPUT/'execution.json'
    if prior.exists() and json.loads(prior.read_text()).get('status')=='complete_fixed_five_case_diagnostic':
        print('Already complete; use --report-only for read-only regeneration.');return
    import threading
    abort=threading.Event();started=dt.datetime.now(dt.timezone.utc);deadline=min(HARD_STOP,started+dt.timedelta(seconds=1800))
    if (deadline-started).total_seconds()<300 or cancelled():raise TimeoutError('Insufficient authorized diagnostic window')
    receipt=dict(status='running',started_utc=started.isoformat(),cutoff_utc=deadline.isoformat(),workers=4,phases=[],plan_sha256=sha(existing),gpu_calls=0)
    atomic_json(OUTPUT/'execution.json',receipt)
    def interrupted(signum,frame):raise KeyboardInterrupt('Audit signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for phase in ('reference','variants'):
            if phase=='variants':require_all_reference(plan)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(run_child,e,phase,deadline,abort) for e in plan['datasets']]
                try:
                    for future in concurrent.futures.as_completed(futures):future.result()
                except BaseException:abort.set();raise
            receipt['phases'].append(dict(phase=phase,status='complete',datasets=9,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()));atomic_json(OUTPUT/'execution.json',receipt)
        final=report(plan);receipt['status']=final['status']
    except BaseException as exc:
        abort.set();receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        atomic_json(OUTPUT/'BATCH_SENSITIVITY.json',dict(status='failed_reference_or_diagnostic',error=receipt['error'],source_sha256=SOURCE_SHA,macros=[],gpu_calls=0,primary_replacement=False))
        (OUTPUT/'BATCH_SENSITIVITY.md').write_text('# Batch sensitivity diagnostic failed\n\n'+receipt['error']+'\n\nNo condition is promoted and no incomplete-coverage macro is reported. See execution.json and individual logs.\n')
        raise
    finally:
        receipt['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();atomic_json(OUTPUT/'execution.json',receipt)
    print(receipt['status'])

if __name__=='__main__':main()
