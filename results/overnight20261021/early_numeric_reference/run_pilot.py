"""One-off isolated 427-row current-runtime check; never completes missing caches."""
from pathlib import Path
from types import SimpleNamespace
import datetime as dt
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import threading
import time

ROOT = Path('/workspace/mod1')
OUT = ROOT/'results/overnight20261021/early_numeric_reference'
LAST = 'next_evo_compositional_8741'
os.chdir(ROOT)
spec=importlib.util.spec_from_file_location('transfer_preparer',ROOT/'scripts/prepare_remaining_transfer.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def proc_snapshot():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: cmd=(p/'cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeError): continue
        if 'src/experiments/main.py' in cmd or ('serve' in cmd and any('vllm' in s for s in cmd)):
            rows.append(dict(pid=int(p.name),command=[x for x in cmd if x]))
    return rows

def gpu_snapshot():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
    return [dict(zip(('index','uuid','free_mib','used_mib','util_percent'),[int(x.strip()) if i!=1 else x.strip() for i,x in enumerate(line.split(','))])) for line in lines.splitlines()]

def check_state():
    state=json.loads((ROOT/'results/next_experiments/parallel_execution_state.json').read_text())
    running=state['running']
    assert len(running)==1 and running[0]['job']==LAST and running[0]['port']==18001, 'Not only last C on GPU1'
    assert all(row['status']=='complete' for key,row in state['jobs'].items() if key!=LAST), 'Source jobs not complete'
    for name in ('sampler_execution_state.json','interface_pilot_state.json'):
        aux=json.loads((ROOT/'results/next_experiments'/name).read_text())
        assert not aux['running'] and all(x['status']=='complete' for x in aux['jobs'].values()), name
    feedback=json.loads((ROOT/'results/overnight20261021/feedback_overlap_receipt.json').read_text())
    assert feedback['status']=='finished', 'Feedback owner not terminal'
    log=ROOT/'out/logs/research20261021'/('uncertainty_quantification_'+LAST)/'candidates.jsonl'
    issued=len([x for x in log.read_text().splitlines() if x.strip()])
    assert 0<issued<30, 'Last C too advanced'
    return dict(running=running,issued_requests=issued,source_finished=sum(x['status']=='complete' for x in state['jobs'].values()),sampler_terminal=True,interface_terminal=True,feedback_terminal=True)

def writer_pids(rows):
    return {x['pid'] for x in rows if 'serve' in x['command']}

OUT.mkdir(parents=True,exist_ok=True)
receipt=dict(status='checking',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='427 historical AVeriTeC rows only; no missing-three extraction or main-receipt writes',gpu_index=0,parent_max_seconds=210,child_max_seconds=180)
stop=threading.Event()
monitor=None
try:
    receipt['state_before']=check_state()
    servers=json.loads((ROOT/'out/overnight20261021/qwen9b_ready_servers.json').read_text())
    assert {(s['gpu'],s['port']) for s in servers['servers']}=={(0,18000),(1,18001)}
    receipt['server_mapping']=[{k:s[k] for k in ('gpu','port','pid')} for s in servers['servers']]
    initial_procs=proc_snapshot(); receipt['filtered_inference_processes_before']=initial_procs
    assert len(writer_pids(initial_procs))==2 and sum('src/experiments/main.py' in x['command'] for x in initial_procs)==1
    assert not (ROOT/'results/overnight20261021/generation_complete.json').exists()
    assets=json.loads((ROOT/'results/overnight20261021/remaining_transfer_download.json').read_text())
    assert assets['status']=='downloaded_verified_no_inference'
    receipt['asset_checks']={}
    for name,folder,model,revision,tok in [('target','Llama3.2-1B',m.TARGET_ID,m.TARGET_REVISION,False),('tokenizer','llama31',m.TOKENIZER_ID,m.TOKENIZER_REVISION,True)]:
        a=assets[name]
        expected=m.pinned_assets(folder,model,revision,Path(a['destination']),tokenizer_only=tok)
        assert a['model']==model and a['revision']==revision and a['files']==expected['files']
        assert m.sha(ROOT/expected['historical_manifest'])==expected['historical_manifest_sha256']
        for f in a['files']:
            p=Path(a['destination'])/f['file']; assert p.stat().st_size==f['bytes'] and m.sha(p)==f['sha256'],str(p)
        receipt['asset_checks'][name]=dict(model=model,revision=revision,all_file_hashes_verified=True,manifest_sha256=expected['historical_manifest_sha256'])
    receipt['tokenizer_parity']=m.tokenizer_parity(Path(assets['tokenizer']['destination']),ROOT/'results/overnight20261021/remaining_transfer_reference_metadata.json')
    validation=json.loads((ROOT/'results/feature_cache_validation.json').read_text())['archives']
    protected=[]
    for name,row in validation.items():
        if '/mod1-transfer-cache/' not in name: continue
        p=ROOT/'data/restored/mod1-transfer-cache'/Path(name).name.removesuffix('.gz')
        assert p.stat().st_size==row['raw_bytes'] and m.sha(p)==row['raw_sha256'],str(p)
        protected.append(dict(path=str(p),raw_sha256=row['raw_sha256'],raw_bytes=row['raw_bytes']))
    assert len(protected)==6
    receipt['historical_caches']=protected
    historical=next(x for x in protected if Path(x['path']).name==m.REFERENCE_NAME)
    import torch
    raw=torch.load(historical['path'],mmap=True,weights_only=False,map_location='cpu')
    assert len(raw['samples'])==427
    max_length=max(len(x['input_ids']) for x in raw['samples'])
    receipt['maximum_reference_input_tokens']=max_length
    # Most pessimistic padded batch logits plus two weights-sized allocations
    # stay below the measured headroom; the extractor moves logits to CPU.
    logits_bytes=min(16*max_length,16384)*128256*4
    conservative_bytes=logits_bytes+3*5*1024**3
    receipt['conservative_reference_gpu_bytes']=conservative_bytes
    assert conservative_bytes < 20*1024**3, 'Insufficient conservative GPU margin'
    del raw
    samples=[]
    for _ in range(2):
        samples.append(gpu_snapshot()); idle=next(x for x in samples[-1] if x['index']==0)
        assert idle['free_mib']>=20480 and idle['util_percent']==0, 'GPU0 not sufficiently idle/free'
        time.sleep(2)
    receipt['gpu_samples_before']=samples
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
    assert len(compute)==2 and {x.split(',')[0].strip() for x in compute}=={x['uuid'] for x in samples[-1]}, 'Other GPU inference detected'
    receipt['gpu_compute_processes_before']=compute
    receipt['state_at_launch']=check_state()
    receipt['code_sha256']={name:m.sha(ROOT/name) for name in ('scripts/prepare_remaining_transfer.py','scripts/evaluate_cached_claims.py','src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py')}
    base_writer_pids=writer_pids(initial_procs)
    def abort_if_changed():
        while not stop.wait(1):
            try:
                if writer_pids(proc_snapshot())!=base_writer_pids or (ROOT/'results/overnight20261021/generation_complete.json').exists():
                    receipt['abort_reason']='Writer processes changed or source generation completed'
                    os.kill(os.getpid(),signal.SIGTERM); return
                if any((ROOT/'results/overnight20261021'/x).exists() for x in ('watchdog.cancel','manual_stop','manual_stop.json')):
                    receipt['abort_reason']='Cancellation marker'; os.kill(os.getpid(),signal.SIGTERM); return
            except BaseException as exc:
                receipt['monitor_error']=repr(exc); os.kill(os.getpid(),signal.SIGTERM); return
    def deadline(signum,frame): raise TimeoutError('Isolated pilot parent210s deadline')
    signal.signal(signal.SIGALRM,deadline); signal.alarm(210)
    args=SimpleNamespace(cache_dir=OUT/'scratch_anchor/cache',data_root=ROOT/'data/claims',model_dir=Path(assets['target']['destination']),tokenizer_dir=Path(assets['tokenizer']['destination']),output=OUT)
    launch=dt.datetime.now(dt.timezone.utc); receipt['launched_utc']=launch.isoformat()
    m.atomic_json(OUT/'pilot_receipt.json',receipt)
    monitor=threading.Thread(target=abort_if_changed,daemon=True); monitor.start()
    receipt['numerical_reference']=m.run_numerical_reference(args,0,launch+dt.timedelta(seconds=200),historical)
    receipt['status']='passed'
except BaseException as exc:
    receipt.update(status='failed' if 'launched_utc' in receipt else 'skipped',error=f'{type(exc).__name__}: {exc}')
finally:
    stop.set(); signal.alarm(0)
    if monitor: monitor.join(timeout=6)
    if 'historical_caches' in receipt:
        receipt['all_six_historical_caches_unchanged']=all(m.sha(x['path'])==x['raw_sha256'] for x in receipt['historical_caches'])
    receipt['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    receipt['gpu_after']=gpu_snapshot()
    m.atomic_json(OUT/'pilot_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','error','started_utc','launched_utc','finished_utc') if k in receipt}),flush=True)
