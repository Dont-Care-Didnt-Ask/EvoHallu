echo $(which python)

devices=0
output_dir=Jan27

mkdir -p out/logs/example/EVALS_polygraph/${output_dir}

base_dir="/mnt/extremessd10tb/seleznev/EvoTune/data/uncertainty_quantification/polygraph_generations_metrics"

# Dataset paths and their appropriate conventional metrics
declare -A datasets
datasets["${base_dir}/babi_qa_continuation.csv"]="exact_match"
datasets["${base_dir}/coqa_simple_instruct.csv"]="f1"
datasets["${base_dir}/gsm8k_simple_instruct.csv"]="exact_match"
datasets["${base_dir}/mmlu_simple_instruct.csv"]="exact_match"
datasets["${base_dir}/samsum_simple_instruct.csv"]="rougeL"
datasets["${base_dir}/triviaqa_simple_instruct.csv"]="f1"
datasets["${base_dir}/truthfulqa_simple_instruct.csv"]="best_rougeL"
datasets["${base_dir}/wmt14_deen.csv"]="bleu"
datasets["${base_dir}/wmt14_fren.csv"]="bleu"
datasets["${base_dir}/wmt19_deen_simple_instruct.csv"]="bleu"
datasets["${base_dir}/wmt19_ruen_simple_instruct.csv"]="bleu"
datasets["${base_dir}/xsum_simple_instruct.csv"]="rougeL"

for dataset_path in "${!datasets[@]}"; do
    metric="${datasets[$dataset_path]}"
    name=$(basename "$dataset_path" .csv)

    echo "=========================================="
    echo "Processing: $name"
    echo "Metric: $metric"
    echo "=========================================="

    # Process samples individually (batch_size=1) to avoid padding artifacts
    # This ensures exact alignment between scoring and generation modes
    CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=/mnt/extremessd10tb/seleznev/EvoTune python src/packing/evaluate/uncertainty_quantification/eval_prr_polygraph.py \
        --data "$dataset_path" \
        --input-col input \
        --text-col model_generation \
        --metric-col "$metric" \
        --output-path "out/logs/example/EVALS_polygraph/${output_dir}/${name}.json" \
        --cache-dir /mnt/extremessd10tb/seleznev/.cache \
        --batch-size 1 \
        --instruct

    echo "Completed: $name"
    echo ""
done

echo "=========================================="
echo "All datasets processed!"
echo "Results saved to: out/logs/example/EVALS_polygraph/${output_dir}/"
echo "=========================================="