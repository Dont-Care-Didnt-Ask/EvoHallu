#!/usr/bin/env bash
set -euo pipefail

# Clean single-container launcher for uncertainty quantification task.
# This script starts a GPU-enabled container and runs the EvoTune UQ
# optimization loop inside it. No detector vLLM server is required.

IMAGE=${IMAGE:-seleznev/seleznev/packing:amd64-cuda-seleznev-latest}
HOST_REPO=${HOST_REPO:-/disk/4tb/seleznev/EvoHallu}
HOST_DATA=${HOST_DATA:-/disk/4tb/seleznev}
HOST_CACHE=${HOST_CACHE:-/disk/4tb/seleznev/.cache/huggingface}
if [ ! -d "${HOST_CACHE}" ]; then
  echo "[run-uq][ERROR] Persistent model cache directory does not exist: ${HOST_CACHE}" >&2
  echo "[run-uq][ERROR] Please create it manually and rerun this script." >&2
  exit 1
fi
echo "[run-uq] Persistent model (HF/vLLM) cache: ${HOST_CACHE}"
UQ_DEVICES=${UQ_DEVICES:-0}
CLUSTER=${CLUSTER:-example}  # Cluster config to use (e.g., 'example', 'api')

cd "${HOST_REPO}"

echo "[run-uq] Launching UQ task in container..."
echo "[run-uq] Image: ${IMAGE}"
echo "[run-uq] Cluster config: ${CLUSTER}"
echo "[run-uq] UQ GPU device(s): ${UQ_DEVICES}"


# Persistent vLLM cache for compiled kernels (speeds up subsequent startups)
HOST_VLLM_CACHE=${HOST_VLLM_CACHE:-${HOST_CACHE}/vllm_cache}
mkdir -p "${HOST_VLLM_CACHE}"

# Base environment variables for the container
DOCKER_ENV_ARGS=(
  -e UQ_DEVICES="${UQ_DEVICES}"
  -e PROJECT_ROOT_AT="${HOST_REPO}"
  -e HF_HOME="${HOST_CACHE}"
  -e CLUSTER="${CLUSTER}"
  # vLLM startup optimizations
  -e VLLM_CACHE_ROOT="${HOST_VLLM_CACHE}"
  -e TORCH_EXTENSIONS_DIR="${HOST_VLLM_CACHE}/torch_extensions"
  -e CUDA_MODULE_LOADING=LAZY
)

# Pass through API key if set (useful if cluster config relies on an API model)
if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "[run-uq] Passing OPENAI_API_KEY to container"
  DOCKER_ENV_ARGS+=( -e OPENAI_API_KEY="${OPENAI_API_KEY}" )
fi

docker run --rm \
  --gpus all \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "${HOST_REPO}:${HOST_REPO}" \
  -v "${HOST_DATA}:${HOST_DATA}" \
  -v "${HOST_CACHE}:${HOST_CACHE}" \
  -v "${HOST_VLLM_CACHE}:${HOST_VLLM_CACHE}" \
  -w "${HOST_REPO}" \
  --network host \
  --ipc host \
  "${IMAGE}" \
  bash "${HOST_REPO}/scripts/uq_wrapper.sh"

echo "[run-uq] Container exited successfully"


