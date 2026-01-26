echo $(which python)

devices=0
output_dir=Jan19_llama8b_base

mkdir out/logs/example/EVALS/${output_dir}

datasets=(
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/popqa/popqa_test_filtered_10shot.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/AVeriTeC/dev_filtered_10shot.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/x-fact/test_filtered_10shot.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/UHEAD_bio_gemma_annotated_by_mistral/test_filtered_10shot.csv
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/geometry/cities.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/geometry/companies_true_false.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/geometry/common_claim_true_false.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/geometry/counterfact_true_false.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/WildHallucinations/pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot.csv \
)

for dataset_path in ${datasets[@]}; do
    tail_name=$(basename $dataset_path .csv)
    second_tail_name=$(basename $(dirname $dataset_path))
    name="${tail_name}_${second_tail_name}"
    echo $name

    # Set batch size based on dataset (memory requirements vary significantly)
    if [[ $dataset_path == *"x-fact"* ]]; then
        batch_size=4
    elif [[ $dataset_path == *"WildHallucinations"* ]]; then
        batch_size=64
    else
        batch_size=128
    fi

    echo "Using batch_size=$batch_size for $second_tail_name"
    CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=/mnt/extremessd10tb/seleznev/EvoTune python src/packing/evaluate/uncertainty_quantification/end2end_eval.py \
        --data $dataset_path \
        --text-col claim \
        --label-col is_hallu \
        --model-id unsloth/Meta-Llama-3.1-8B \
        --output-path out/logs/example/EVALS/${output_dir}/${name}_eval.json \
        --chat-template "" \
        --batch-size $batch_size
        # --chat-template generate_true_statement \
        # --model-id unsloth/Llama-3.2-1B-Instruct \
done