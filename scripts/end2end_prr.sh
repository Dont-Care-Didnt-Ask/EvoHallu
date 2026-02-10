echo $(which python)

devices=1
output_dir=Feb2

mkdir -p out/logs/example/EVALS_polygraph/${output_dir}

base_dir="/mnt/extremessd10tb/seleznev/EvoTune/data/uncertainty_quantification/polygraph_generations_metrics"

# Dataset paths ordered by size (smallest first) for quick feedback
# Format: "filename:metric"
datasets=(
    "truthfulqa_simple_instruct.csv:best_rougeL"
    "samsum_simple_instruct.csv:rougeL"
    "babi_qa_continuation.csv:exact_match"
    "gsm8k_simple_instruct.csv:exact_match"
    "wmt19_deen_simple_instruct.csv:bleu"
    "wmt19_ruen_simple_instruct.csv:bleu"
    "wmt14_deen.csv:bleu"
    "wmt14_fren.csv:bleu"
    "mmlu_simple_instruct.csv:exact_match"
    "coqa_simple_instruct.csv:f1"
    "xsum_simple_instruct.csv:rougeL"
    "triviaqa_simple_instruct.csv:f1"
)

for dataset_info in "${datasets[@]}"; do
    # Split by colon
    filename="${dataset_info%%:*}"
    metric="${dataset_info##*:}"
    dataset_path="${base_dir}/${filename}"
    name=$(basename "$filename" .csv)

    echo "=========================================="
    echo "Processing: $name"
    echo "Metric: $metric"
    echo "=========================================="

    # Process with chunked approach to avoid OOM
    # - chunk_size=C: Process C samples at a time, extract scores, discard stats
    # - batch_size=1: Individual processing within chunk to avoid padding artifacts
    CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=/mnt/extremessd10tb/seleznev/EvoTune python src/packing/evaluate/uncertainty_quantification/eval_prr_polygraph.py \
        --data "$dataset_path" \
        --input-col input \
        --text-col model_generation \
        --metric-col "$metric" \
        --output-path "out/logs/example/EVALS_polygraph/${output_dir}/${name}.json" \
        --cache-dir /mnt/extremessd10tb/seleznev/.cache \
        --chunk-size 16 \
        --batch-size 1 \
        --instruct

    echo "Completed: $name"
    echo ""
done

echo "=========================================="
echo "All datasets processed!"
echo "Results saved to: out/logs/example/EVALS_polygraph/${output_dir}/"
echo "=========================================="
