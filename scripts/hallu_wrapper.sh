#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to run inside container: starts vLLM detector, runs experiment, ensures cleanup

echo "[wrapper] ===== Hallucination detector wrapper starting ====="

# Upgrade transformers for newer model support (gpt-oss, qwen3)
echo "[wrapper] Upgrading transformers and vLLM..."
pip install --upgrade transformers vllm

echo "[wrapper] Working directory: $(pwd)"
echo "[wrapper] Detector GPU(s): ${DETECTOR_DEVICES:-<unset>}"
echo "[wrapper] Generator GPU(s): ${GENERATOR_DEVICES:-<unset>}"
echo "[wrapper] HF_HOME=${HF_HOME:-<unset>}"
echo "[wrapper] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[wrapper] ---- nvidia-smi before starting detector ----"
    nvidia-smi || true
    echo "[wrapper] ---------------------------------------------"
fi

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

echo "[wrapper] Detector config: model=${DETECTOR_MODEL} port=${DETECTOR_PORT}"
echo "[wrapper] Overriding detector max model length to 8192 tokens to reduce KV cache size"
echo "[wrapper] Starting vLLM detector: model=${DETECTOR_MODEL} port=${DETECTOR_PORT} device=${DETECTOR_DEVICES:-<unset>}"
echo "[wrapper] vLLM logs will be written to /mnt/extremessd10tb/seleznev/tmp/vllm.log"

# Clear previous vLLM log if it exists to avoid confusion
if [ -f /mnt/extremessd10tb/seleznev/tmp/vllm.log ]; then
    echo "[wrapper] Removing existing /mnt/extremessd10tb/seleznev/tmp/vllm.log"
    rm -f /mnt/extremessd10tb/seleznev/tmp/vllm.log
fi

# TODO: make gpu memory utilization configurable
# CUDA_VISIBLE_DEVICES="${DETECTOR_DEVICES}" vllm serve "${DETECTOR_MODEL}" --gpu-memory-utilization "0.35" --port "${DETECTOR_PORT}" >/mnt/extremessd10tb/seleznev/tmp/vllm.log 2>&1 &
CUDA_VISIBLE_DEVICES="${DETECTOR_DEVICES}" vllm serve "${DETECTOR_MODEL}" \
    --port "${DETECTOR_PORT}" \
    --max-model-len 8192 \
    >/mnt/extremessd10tb/seleznev/tmp/vllm.log 2>&1 &
VLLM_PID=$!
echo "[wrapper] vLLM started with PID=${VLLM_PID}"
sleep 2
if ! ps -p ${VLLM_PID} > /dev/null 2>&1; then
    echo "[wrapper] ERROR: vLLM process died immediately. Log tail:"
    if [ -f /mnt/extremessd10tb/seleznev/tmp/vllm.log ]; then
        tail -n 200 /mnt/extremessd10tb/seleznev/tmp/vllm.log
    else
        echo "[wrapper] /mnt/extremessd10tb/seleznev/tmp/vllm.log not found"
    fi
    exit 1
fi

# Ensure vLLM is killed on exit (success, failure, or signal)
trap "echo '[wrapper] Stopping vLLM PID=${VLLM_PID}...'; kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true" EXIT

echo "[wrapper] Waiting for vLLM to be ready (this may take several minutes for model download/loading)..."
python3 - <<PY
import time, requests, sys, subprocess, os
url = f'http://localhost:${DETECTOR_PORT}/v1/chat/completions'
model = '${DETECTOR_MODEL}'
print(f"[wrapper] Healthcheck URL: {url}, model={model}")
print(f"[wrapper] DETECTOR_DEVICES env: {os.environ.get('DETECTOR_DEVICES')}")
print(f"[wrapper] CUDA_VISIBLE_DEVICES env: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
for i in range(300):  # Wait up to 5 minutes
    try:
        r = requests.post(url, json={'model': model, 'messages': [{'role':'user','content':'ping'}]}, timeout=2)
        if r.status_code == 200:
            print('[wrapper] vLLM is ready!')
            break
        else:
            if i % 30 == 0:
                print(f"[wrapper] Healthcheck HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
    except Exception as e:
        if i % 30 == 0:  # Print every 30 seconds
            print(f'[wrapper] Still waiting... ({i}s elapsed). Last error: {type(e).__name__}: {e}', file=sys.stderr)
            # Show last few lines of vLLM log
            try:
                result = subprocess.run(['tail', '-50', '/mnt/extremessd10tb/seleznev/tmp/vllm.log'], capture_output=True, text=True)
                if result.stdout.strip():
                    print(f'[wrapper] vLLM log tail:\\n{result.stdout}', file=sys.stderr)
            except Exception as log_err:
                print(f'[wrapper] Failed to read vLLM log tail: {log_err}', file=sys.stderr)
    time.sleep(1)
else:
    print('[wrapper] vLLM failed to start after 300 seconds', file=sys.stderr)
    print('[wrapper] Full vLLM log:', file=sys.stderr)
    subprocess.run(['cat', '/mnt/extremessd10tb/seleznev/tmp/vllm.log'])
    sys.exit(1)
PY

echo "[wrapper] Running EvoTune experiment on device=${GENERATOR_DEVICES:-<unset>}..."
# Pass cluster config if specified via environment variable
if [ -n "${CLUSTER:-}" ]; then
    echo "[wrapper] Using cluster config: ${CLUSTER}"
    CUDA_VISIBLE_DEVICES="${GENERATOR_DEVICES}" python3 -m src.experiments.main cluster="${CLUSTER}"
else
    CUDA_VISIBLE_DEVICES="${GENERATOR_DEVICES}" python3 -m src.experiments.main
fi

echo "[wrapper] Experiment finished successfully"

