#!/usr/bin/env bash
set -euo pipefail

# Minimal wrapper script for UQ task - runs Python in foreground
# The key is that bash running a script file properly propagates SIGINT to children

echo "[uq-wrapper] Working directory: $(pwd)"
echo "[uq-wrapper] UQ GPU(s): ${UQ_DEVICES:-<unset>}"
echo "[uq-wrapper] HF_HOME=${HF_HOME:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[uq-wrapper] ---- nvidia-smi ----"
    nvidia-smi || true
    echo "[uq-wrapper] --------------------"
fi

echo "[uq-wrapper] Starting EvoTune UQ experiment (cluster=${CLUSTER:-example})..."

# Run Python in foreground - SIGINT will propagate to Python and its children (vLLM)
if [ -n "${CLUSTER:-}" ]; then
    CUDA_VISIBLE_DEVICES="${UQ_DEVICES}" python3 -m src.experiments.main --config-name config_uq cluster="${CLUSTER}" task=uncertainty_quantification
else
    CUDA_VISIBLE_DEVICES="${UQ_DEVICES}" python3 -m src.experiments.main --config-name config_uq task=uncertainty_quantification
fi

echo "[uq-wrapper] Experiment finished"

