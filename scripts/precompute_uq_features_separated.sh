#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

# Override these through environment variables when using another machine/dataset.
CLAIM_DATA=${CLAIM_DATA:-${REPO_ROOT}/../xx_datasets}
OUTPUT_DIR=${OUTPUT_DIR:-data/uncertainty_quantification}
UQ_DEVICES=${UQ_DEVICES:-${devices:-0}}
PYTHON=${PYTHON:-python}
MODEL_ID=${MODEL_ID:-VityaVitalich/Llama3.1-8b-instruct}
BATCH_SIZE=${BATCH_SIZE:-16}
TOP_K=${TOP_K:-512}
LAYER_STRIDE=${LAYER_STRIDE:-1}
CACHE_DIR=${CACHE_DIR:-${HF_HOME:-${HOST_CACHE:-${REPO_ROOT}/../.cache/huggingface}}/hub}

DATASETS=(
    "${CLAIM_DATA}/popqa/popqa_train_filtered_10shot.csv"
    "${CLAIM_DATA}/popqa/popqa_test_filtered_10shot.csv"
)
# Check both inputs before starting an expensive model pass.
for dataset in "${DATASETS[@]}"; do
    [[ -f "${dataset}" ]] || { echo "[precompute][ERROR] Missing CSV: ${dataset}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"

for dataset in "${DATASETS[@]}"; do
    echo "[precompute] Dataset: ${dataset}; GPU(s): ${UQ_DEVICES}"
    CUDA_VISIBLE_DEVICES="${UQ_DEVICES}" "${PYTHON}" src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py \
        --data "${dataset}" \
        --text-col claim --label-col is_hallu \
        --model-id "${MODEL_ID}" --cache-dir "${CACHE_DIR}" \
        --chat-template generate_true_statement \
        --batch-size "${BATCH_SIZE}" --top-k "${TOP_K}" --layer-stride "${LAYER_STRIDE}" \
        --output-path "${OUTPUT_DIR}/$(basename "${dataset}" .csv)_raw_stats.pt" \
        "$@"
done
