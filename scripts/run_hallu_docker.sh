#!/usr/bin/env bash
set -euo pipefail

# Clean single-container launcher for hallucination task
# Uses existing image, runs wrapper script inside container with trap for cleanup

IMAGE=${IMAGE:-seleznev/seleznev/packing:amd64-cuda-seleznev-latest}
HOST_REPO=${HOST_REPO:-/mnt/extremessd10tb/seleznev/EvoTune}
HOST_DATA=${HOST_DATA:-/mnt/extremessd10tb/seleznev}
DETECTOR_DEVICES=${DETECTOR_DEVICES:-0}
GENERATOR_DEVICES=${GENERATOR_DEVICES:-1}

cd "${HOST_REPO}"

echo "[run] Launching hallucination task in container..."
echo "[run] Image: ${IMAGE}"
echo "[run] Detector GPU: ${DETECTOR_DEVICES}, Generator GPU: ${GENERATOR_DEVICES}"

docker run --rm -it \
  --gpus all \
  -e DETECTOR_DEVICES="${DETECTOR_DEVICES}" \
  -e GENERATOR_DEVICES="${GENERATOR_DEVICES}" \
  -e PROJECT_ROOT_AT="${HOST_REPO}" \
  -v "${HOST_REPO}:${HOST_REPO}" \
  -v "${HOST_DATA}:${HOST_DATA}" \
  -w "${HOST_REPO}" \
  --network host \
  --ipc host \
  "${IMAGE}" \
  bash scripts/hallu_wrapper.sh

echo "[run] Container exited successfully"

