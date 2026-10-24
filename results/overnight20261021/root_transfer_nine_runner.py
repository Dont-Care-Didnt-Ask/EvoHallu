"""Complete frozen full-nine scoring after the recorded idle-check recovery.

No generation, target fitting, program changes, or selection. The prior six
report and original controller outcome are retained as separate evidence.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path('/workspace/mod1')
os.chdir(root)
out = root / 'results/overnight20261021'
receipt = out / 'root_transfer_nine_execution.json'
assert not receipt.exists(), 'Inspect previous nine execution before any retry'
completion = json.loads((out / 'remaining_transfer_completion.json').read_text())
assert completion['status'] == 'complete_three_missing_caches'
assert completion['all_six_historical_caches_unchanged'] is True
assert len(completion['new_caches']) == 3
assert (out / 'cpu_complete.json').exists() and (out / 'late_extensions_complete.json').exists()
source = root / 'scripts/overnight_transfer_eval.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '62b63c89ad30fb6b4455dd2737d0240d16f66f2ad4ea65386bf40d07a6830462'
assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat('2026-10-22T01:15:00+00:00')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: argv = (proc / 'cmdline').read_bytes().split(b'\0')
    except (OSError, PermissionError): continue
    assert not any(a in [str(source).encode(), b'scripts/overnight_transfer_eval.py'] for a in argv), 'Another main transfer evaluator is active'
command = [str(root / '.venv/bin/python'), str(source), '--execute', '--coverage', 'nine',
           '--workers', '4', '--bootstrap', '500', '--max-seconds', '2400',
           '--cache-dir', 'data/restored/mod1-transfer-cache']
state = dict(status='running', started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    runner_pid=os.getpid(), command=command, evaluator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    completion_receipt_sha256=hashlib.sha256((out / 'remaining_transfer_completion.json').read_bytes()).hexdigest(),
    reason='Guarded three-cache recovery succeeded; prior six scoring is complete. CPU budget accommodates observed 982-second six-dataset workload.',
    all_six_original_results_retained=True, new_model_calls=False, target_fitting=False, selection_changed=False)
def save():
    temporary=receipt.with_suffix('.part')
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace(receipt)
save()
started = time.monotonic()
with (out / 'root_transfer_nine_evaluation.log').open('x') as log:
    child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH='src:.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    state['child_pid']=child.pid
    save()
    try:
        code=child.wait(timeout=2430)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        code=-1
        state['outer_timeout']=True
state.update(status='completed' if code==0 else 'failed_or_partial', exit_code=code,
             finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
save()
print(json.dumps(state), flush=True)
