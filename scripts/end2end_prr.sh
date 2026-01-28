echo $(which python)

devices=0
output_dir=Jan27

mkdir out/logs/example/EVALS_polygraph/${output_dir}

base_dir="/mnt/extremessd10tb/seleznev/EvoTune/data/uncertainty_quantification"

datasets=(
    ${base_dir}/babi_qa_continuation.csv
    ${base_dir}/coqa_simple_instruct.csv
    ${base_dir}/gsm8k_simple_instruct.csv
    ${base_dir}/mmlu_simple_instruct.csv
    ${base_dir}/samsum_simple_instruct.csv
    ${base_dir}/triviaqa_simple_instruct.csv
    ${base_dir}/truthfulqa_simple_instruct.csv
    ${base_dir}/wmt14_deen.csv
    ${base_dir}/wmt14_reen.csv
    ${base_dir}/wmt19_deen_simple_instruct.csv
    ${base_dir}/wmt19_ruen_simple_instruct.csv
    ${base_dir}/xsum_simple_instruct.csv
)

for dataset_path in ${datasets[@]}; do
    name=$(basename $dataset_path .csv)
    echo $name

    # Set batch size based on dataset (memory requirements vary significantly)
    if [[ $dataset_path == *"xsum"* ]]; then
        batch_size=2
    else
        batch_size=64
    fi

    echo "Using batch_size=$batch_size for $name"
    CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=/mnt/extremessd10tb/seleznev/EvoTune python src/packing/evaluate/uncertainty_quantification/eval_prr_polygraph.py \
        --data $dataset_path \
        --input-col input \
        --text-col claim \
        --metric-col alignscore \
        --model-id unsloth/Meta-Llama-3.1-8B \
        --output-path out/logs/example/EVALS_polygraph/${output_dir}/${name}.json \
        --chat-template "" \
        --cache-dir /mnt/extremessd10tb/seleznev/.cache \
        --batch-size $batch_size \
        --instruct
        # --chat-template generate_true_statement \
        # --model-id unsloth/Llama-3.2-1B-Instruct \
done