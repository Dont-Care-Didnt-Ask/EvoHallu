#!/usr/bin/env bash
set -euo pipefail

echo "[uq-multi-train-mix] Working directory: $(pwd)"
UQ_DEVICES="${UQ_DEVICES:-0}"
PRIMARY_GPU="${UQ_DEVICES%%,*}"
[[ "${PRIMARY_GPU}" =~ ^[0-9]+$ ]] || { echo "[uq-multi-train-mix][ERROR] UQ_DEVICES must start with an integer GPU id, got '${UQ_DEVICES}'" >&2; exit 1; }

echo "[uq-multi-train-mix] UQ GPU(s): ${UQ_DEVICES}"
echo "[uq-multi-train-mix] Primary GPU for vLLM: ${PRIMARY_GPU}"
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
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${UQ_DEVICES}" python3 -m src.experiments.main \
        --config-name config_uq_multi_train_mix_generations \
        cluster="${CLUSTER}" \
        gpu_nums="${PRIMARY_GPU}" \
        task=uncertainty_quantification_multi_train_mix_generations
else
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${UQ_DEVICES}" python3 -m src.experiments.main \
        --config-name config_uq_multi_train_mix_generations \
        gpu_nums="${PRIMARY_GPU}" \
        task=uncertainty_quantification_multi_train_mix_generations
fi

echo "[uq-multi-train-mix] Run finished"
