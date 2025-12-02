#!/usr/bin/env bash
set -euo pipefail

# Clean single-container launcher for hallucination task
# Uses existing image, runs wrapper script inside container with trap for cleanup

IMAGE=${IMAGE:-seleznev/seleznev/packing:amd64-cuda-seleznev-latest}
HOST_REPO=${HOST_REPO:-/mnt/extremessd10tb/seleznev/EvoTune}
HOST_DATA=${HOST_DATA:-/mnt/extremessd10tb/seleznev}
HOST_CACHE=${HOST_CACHE:-/mnt/extremessd10tb/seleznev/.cache/huggingface}
DETECTOR_DEVICES=${DETECTOR_DEVICES:-0}
GENERATOR_DEVICES=${GENERATOR_DEVICES:-1}
CLUSTER=${CLUSTER:-api}  # Cluster config to use (e.g., 'example', 'api')

cd "${HOST_REPO}"

echo "[run] Launching hallucination task in container..."
echo "[run] Image: ${IMAGE}"
echo "[run] Cluster config: ${CLUSTER}"
echo "[run] Detector GPU: ${DETECTOR_DEVICES}, Generator GPU: ${GENERATOR_DEVICES}"

# Ensure cache directory exists on host
mkdir -p "${HOST_CACHE}"

# Build docker run command with environment variables
DOCKER_ENV_ARGS=(
  -e DETECTOR_DEVICES="${DETECTOR_DEVICES}"
  -e GENERATOR_DEVICES="${GENERATOR_DEVICES}"
  -e PROJECT_ROOT_AT="${HOST_REPO}"
  -e HF_HOME="${HOST_CACHE}"
  -e CLUSTER="${CLUSTER}"
)

# Pass through API key if set (for API mode)
if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "[run] Passing OPENAI_API_KEY to container"
  DOCKER_ENV_ARGS+=( -e OPENAI_API_KEY="${OPENAI_API_KEY}" )
fi

docker run --rm -it \
  --gpus all \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "${HOST_REPO}:${HOST_REPO}" \
  -v "${HOST_DATA}:${HOST_DATA}" \
  -v "${HOST_CACHE}:${HOST_CACHE}" \
  -w "${HOST_REPO}" \
  --network host \
  --ipc host \
  "${IMAGE}" \
  bash scripts/hallu_wrapper.sh

echo "[run] Container exited successfully"

