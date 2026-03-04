#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
PYTHON_BIN=$(which python)
datasets_source_dir="../xx_datasets"
output_dir="data/uncertainty_quantification/polygraph_fixed2"
log_dir="${output_dir}/logs"

# Clarification: Assuming device 0 and 1. Please update if different.
DEVICE_1=1
DEVICE_2=2

mkdir -p "${output_dir}"
mkdir -p "${log_dir}"

# --- Dataset Lists ---
# Group 1: coqa, wmt
DATASETS_1=(
    "${datasets_source_dir}/coqa/coqa_simple_instruct.csv"
    "${datasets_source_dir}/coqa/coqa_simple_instruct_limit_1536.csv"
    "${datasets_source_dir}/wmt14_deen/wmt14_deen.csv"
    "${datasets_source_dir}/wmt14_deen/wmt14_deen_limit_1536.csv"
)

# Group 2: samsum, triviaqa
DATASETS_2=(
    "${datasets_source_dir}/samsum/samsum_simple_instruct.csv"
    "${datasets_source_dir}/samsum/samsum_simple_instruct_limit_1536.csv"
    "${datasets_source_dir}/triviaqa/triviaqa_simple_instruct.csv"
    "${datasets_source_dir}/triviaqa/triviaqa_simple_instruct_limit_1536.csv"
)

# --- Execution Function ---
run_evaluation() {
    local device_id=$1
    local log_file=$2
    shift 2
    local datasets=("$@")

    echo "Starting on device ${device_id} | Log: ${log_file}" >> "${log_file}"

    for dataset in "${datasets[@]}"; do
        dataset_dir=$(dirname "${dataset}")
        dataset_name=$(basename "${dataset}")

        echo "Dataset: ${dataset}" >> "${log_file}"
        
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${device_id}" "${PYTHON_BIN}" \
            src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_polygraph_chunked.py \
            --input-dir "${dataset_dir}" \
            --pattern "${dataset_name}" \
            --chunk-size 1024 \
            --cache-dir ~/.cache/huggingface/hub \
            --instruct \
            --output-dir "${output_dir}" >> "${log_file}" 2>&1
    done
}

# --- Parallel Execution ---
echo "Launching parallel jobs..."

# Run Job 1 in background
run_evaluation "${DEVICE_1}" "${log_dir}/device_${DEVICE_1}.log" "${DATASETS_1[@]}" &
PID_1=$!

# Run Job 2 in background
run_evaluation "${DEVICE_2}" "${log_dir}/device_${DEVICE_2}.log" "${DATASETS_2[@]}" &
PID_2=$!

# Wait for both to complete. If either fails, script exits due to set -e.
wait ${PID_1}
wait ${PID_2}

echo "All evaluations completed successfully."