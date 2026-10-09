#!/usr/bin/env bash
set -euo pipefail

# Minimal wrapper script for UQ task - runs Python in foreground
# The key is that bash running a script file properly propagates SIGINT to children

echo "[uq-wrapper] Working directory: $(pwd)"
UQ_DEVICES="${UQ_DEVICES:-0}"
[[ "${UQ_DEVICES}" =~ ^[0-9]+$ ]] || { echo "[uq-wrapper][ERROR] The default vLLM launcher supports one GPU; set UQ_DEVICES to an integer GPU id, got '${UQ_DEVICES}'" >&2; exit 1; }
PRIMARY_GPU="${UQ_DEVICES}"

HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
while [ "$(basename "${HF_HOME_DIR}")" = "hub" ]; do
    HF_HOME_DIR="$(dirname "${HF_HOME_DIR}")"
done
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"
[ -d "${HUGGINGFACE_HUB_CACHE}" ] || mkdir -p "${HUGGINGFACE_HUB_CACHE}"

echo "[uq-wrapper] UQ GPU(s): ${UQ_DEVICES}"
echo "[uq-wrapper] Primary GPU for vLLM: ${PRIMARY_GPU}"
echo "[uq-wrapper] HF_HOME=${HF_HOME}"
echo "[uq-wrapper] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[uq-wrapper] ---- nvidia-smi ----"
    nvidia-smi || true
    echo "[uq-wrapper] --------------------"
fi

echo "[uq-wrapper] Starting EvoTune UQ experiment (cluster=${CLUSTER:-example})..."

# Run Python in foreground - SIGINT will propagate to Python and its children (vLLM)
# Keep physical GPU IDs intact: start_vllm_server sets its own visibility mask.
# Docker exposes all GPUs; masking here would remap a selected GPU to index 0.
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$(pwd)/src:$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m src.experiments.main --config-name config_uq \
    cluster="${CLUSTER:-example}" gpu_nums="${PRIMARY_GPU}" \
    task=uncertainty_quantification \
    cluster.use_vllm=1 cluster.use_api=0 cluster.use_tgi=0 \
    "$@"

echo "[uq-wrapper] Experiment finished"

