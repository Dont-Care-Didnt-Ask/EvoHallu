#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for verb_checker.py in Docker container
# Uses vLLM as a library (no server setup needed)

IMAGE=${IMAGE:-seleznev/seleznev/packing:amd64-cuda-seleznev-latest}
HOST_REPO=${HOST_REPO:-/mnt/extremessd10tb/seleznev/EvoTune}
HOST_DATA=${HOST_DATA:-/mnt/extremessd10tb/seleznev}
HOST_CACHE=${HOST_CACHE:-/mnt/extremessd10tb/seleznev/.cache}
CUDA_DEVICE=${CUDA_DEVICE:-1}

cd "${HOST_REPO}"

echo "[run] Launching verb_checker.py in container..."
echo "[run] Image: ${IMAGE}"
echo "[run] CUDA Device: ${CUDA_DEVICE}"

# Ensure cache directory exists on host
mkdir -p "${HOST_CACHE}"

# Build docker run command with environment variables
DOCKER_ENV_ARGS=(
  -e CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  -e PROJECT_ROOT_AT="${HOST_REPO}"
  -e HF_HOME="${HOST_CACHE}"
)

# Pass through HF token if set
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[run] Passing HF_TOKEN to container"
  DOCKER_ENV_ARGS+=( -e HF_TOKEN="${HF_TOKEN}" )
fi

docker run --rm -it \
  --gpus all \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "${HOST_REPO}:${HOST_REPO}" \
  -v "${HOST_DATA}:${HOST_DATA}" \
  -v "${HOST_CACHE}:${HOST_CACHE}" \
  -w "${HOST_REPO}/src/experiments" \
  --network host \
  --ipc host \
  "${IMAGE}" \
  python3 verb_checker.py "${HOST_DATA}/longtail-dataset/datasets/popqa/popqa_test_filtered_10shot.csv" verb_checker_repro.csv

echo "[run] Container exited successfully"

