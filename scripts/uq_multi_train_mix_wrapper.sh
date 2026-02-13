#!/usr/bin/env bash
set -euo pipefail

echo "[uq-multi-train-mix] Working directory: $(pwd)"
echo "[uq-multi-train-mix] UQ GPU(s): ${UQ_DEVICES:-<unset>}"
echo "[uq-multi-train-mix] HF_HOME=${HF_HOME:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[uq-multi-train-mix] ---- nvidia-smi ----"
    nvidia-smi || true
    echo "[uq-multi-train-mix] --------------------"
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "[uq-multi-train-mix] Running preflight checks..."
python3 scripts/preflight_uq_multi_train_mix.py \
    --task-config configs/task/uncertainty_quantification_multi_train_mix_generations.yaml

echo "[uq-multi-train-mix] Starting EvoTune UQ multi train-mix run (cluster=${CLUSTER:-example})..."

if [ -n "${CLUSTER:-}" ]; then
    CUDA_VISIBLE_DEVICES="${UQ_DEVICES}" python3 -m src.experiments.main \
        --config-name config_uq_multi_train_mix_generations \
        cluster="${CLUSTER}" \
        task=uncertainty_quantification_multi_train_mix_generations
else
    CUDA_VISIBLE_DEVICES="${UQ_DEVICES}" python3 -m src.experiments.main \
        --config-name config_uq_multi_train_mix_generations \
        task=uncertainty_quantification_multi_train_mix_generations
fi

echo "[uq-multi-train-mix] Run finished"
