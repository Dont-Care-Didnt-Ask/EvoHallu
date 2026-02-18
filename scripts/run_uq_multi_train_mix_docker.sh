#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-seleznev/seleznev/packing:amd64-cuda-seleznev-latest}
HOST_REPO=${HOST_REPO:-/mnt/extremessd10tb/seleznev/EvoTune}
HOST_DATA=${HOST_DATA:-/mnt/extremessd10tb/seleznev}
HOST_CACHE=${HOST_CACHE:-/mnt/extremessd10tb/seleznev/.cache/huggingface}
UQ_DEVICES=${UQ_DEVICES:-1}
CLUSTER=${CLUSTER:-example}

if [ ! -d "${HOST_CACHE}" ]; then
  echo "[run-uq-multi][ERROR] Persistent model cache directory does not exist: ${HOST_CACHE}" >&2
  echo "[run-uq-multi][ERROR] Please create it manually and rerun this script." >&2
  exit 1
fi

cd "${HOST_REPO}"

echo "[run-uq-multi] Launching containerized UQ multi train-mix run..."
echo "[run-uq-multi] Image: ${IMAGE}"
echo "[run-uq-multi] Cluster config: ${CLUSTER}"
echo "[run-uq-multi] UQ GPU device(s): ${UQ_DEVICES}"

HOST_VLLM_CACHE=${HOST_VLLM_CACHE:-${HOST_CACHE}/vllm_cache}
mkdir -p "${HOST_VLLM_CACHE}"

DOCKER_ENV_ARGS=(
  -e UQ_DEVICES="${UQ_DEVICES}"
  -e PROJECT_ROOT_AT="${HOST_REPO}"
  -e HF_HOME="${HOST_CACHE}"
  -e CLUSTER="${CLUSTER}"
  -e VLLM_CACHE_ROOT="${HOST_VLLM_CACHE}"
  -e TORCH_EXTENSIONS_DIR="${HOST_VLLM_CACHE}/torch_extensions"
  -e CUDA_MODULE_LOADING=LAZY
)

if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "[run-uq-multi] Passing OPENAI_API_KEY to container"
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
  bash "${HOST_REPO}/scripts/uq_multi_train_mix_wrapper.sh"

echo "[run-uq-multi] Container exited successfully"
