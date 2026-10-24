"""One-off recovery after the controller's post-shutdown idle refusal.

Uses the unchanged formal preparer/guard. No scientific selection or repair.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/workspace/mod1')
os.chdir(root)
sys.path.insert(0, str(root / 'scripts'))
import overnight_start_glm as glm
out = root / 'results/overnight20261021'
receipt_path = out / 'root_transfer_recovery_launch.json'
assert not receipt_path.exists(), 'Recovery already launched; inspect existing receipt'
assert not (out / 'remaining_transfer_completion.json').exists(), 'Inspect existing formal completion first'
source = root / 'scripts/prepare_remaining_transfer.py'
expected = 'bd35863bd7fbc95fdedeb6d31baa4e2332d3860a2a62a43f75bf44078c49fe8e'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
assert (out / 'late_writer_decision.json').exists()
assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat('2026-10-22T01:20:00+00:00')
samples = []
for _ in range(2):
    sample = glm.telemetry()
    glm.require_idle(sample, [0, 1])
    samples.append(sample)
    time.sleep(2)
command = [str(root / '.venv/bin/python'), str(source), '--execute', '--gpu-indices', '0', '1',
           '--max-minutes', '35', '--deadline-utc', '2026-10-22T02:01:28+00:00',
           '--cache-dir', 'data/restored/mod1-transfer-cache']
with (out / 'root_transfer_recovery.log').open('x') as log:
    child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=dict(os.environ, PYTHONPATH='src:.'))
receipt = dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=child.pid, command=command,
    reason='Original controller refused immediately after writer shutdown; subsequent two snapshots certify both GPUs idle.',
    original_failure_retained=True, original_log_sha256=hashlib.sha256((out / 'late_extensions.log').read_bytes()).hexdigest(),
    preparer_sha256=expected, idle_samples=samples, changed_thresholds=False, changed_scientific_code=False,
    downstream='Existing late controller may finish six; root will separately score nine after successful formal guard/completion.')
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
