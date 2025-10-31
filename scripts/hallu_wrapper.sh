#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to run inside container: starts vLLM detector, runs experiment, ensures cleanup

echo "[wrapper] Reading detector config from task yaml..."
DETECTOR_MODEL=$(python3 - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/task/hallu.yaml')
print(cfg.get('detector_model', 'Qwen/Qwen2.5-0.5B-Instruct'))
PY
)

DETECTOR_PORT=$(python3 - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/task/hallu.yaml')
print(cfg.get('detector_vllm_port', 8088))
PY
)

echo "[wrapper] Starting vLLM detector: model=${DETECTOR_MODEL} port=${DETECTOR_PORT} device=${DETECTOR_DEVICES}"
echo "[wrapper] vLLM logs will be written to /tmp/vllm.log"
# TODO: make gpu memory utilization configurable
# CUDA_VISIBLE_DEVICES="${DETECTOR_DEVICES}" vllm serve "${DETECTOR_MODEL}" --gpu-memory-utilization "0.35" --port "${DETECTOR_PORT}" >/tmp/vllm.log 2>&1 &
CUDA_VISIBLE_DEVICES="${DETECTOR_DEVICES}" vllm serve "${DETECTOR_MODEL}" --port "${DETECTOR_PORT}" >/tmp/vllm.log 2>&1 &
VLLM_PID=$!
echo "[wrapper] vLLM started with PID=${VLLM_PID}"
sleep 2
if ! ps -p ${VLLM_PID} > /dev/null 2>&1; then
    echo "[wrapper] ERROR: vLLM process died immediately. Log tail:"
    tail -n 200 /tmp/vllm.log
    exit 1
fi

# Ensure vLLM is killed on exit (success, failure, or signal)
trap "echo '[wrapper] Stopping vLLM PID=${VLLM_PID}...'; kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true" EXIT

echo "[wrapper] Waiting for vLLM to be ready (this may take several minutes for model download/loading)..."
python3 - <<PY
import time, requests, sys, subprocess
url = f'http://localhost:${DETECTOR_PORT}/v1/chat/completions'
for i in range(300):  # Wait up to 5 minutes
    try:
        r = requests.post(url, json={'model': '${DETECTOR_MODEL}', 'messages': [{'role':'user','content':'ping'}]}, timeout=2)
        if r.status_code == 200:
            print('[wrapper] vLLM is ready!')
            break
    except Exception as e:
        if i % 30 == 0:  # Print every 30 seconds
            print(f'[wrapper] Still waiting... ({i}s elapsed). Last error: {type(e).__name__}', file=sys.stderr)
            # Show last few lines of vLLM log
            try:
                result = subprocess.run(['tail', '-50', '/tmp/vllm.log'], capture_output=True, text=True)
                if result.stdout.strip():
                    print(f'[wrapper] vLLM log tail:\\n{result.stdout}', file=sys.stderr)
            except:
                pass
    time.sleep(1)
else:
    print('[wrapper] vLLM failed to start after 300 seconds', file=sys.stderr)
    print('[wrapper] Full vLLM log:', file=sys.stderr)
    subprocess.run(['cat', '/tmp/vllm.log'])
    sys.exit(1)
PY

echo "[wrapper] Running EvoTune experiment on device=${GENERATOR_DEVICES}..."
CUDA_VISIBLE_DEVICES="${GENERATOR_DEVICES}" python3 -m src.experiments.main

echo "[wrapper] Experiment finished successfully"

