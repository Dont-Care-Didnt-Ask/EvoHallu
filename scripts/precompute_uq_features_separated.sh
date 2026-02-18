echo $(which python)
devices=1

datasets_source_dir="../xx_datasets"

DATASETS=(
    ${datasets_source_dir}/train_mix_no_wildhall.csv
    ${datasets_source_dir}/test_mix_no_wildhall.csv
    # ${datasets_source_dir}/popqa/popqa_train_filtered_10shot.csv
    # ${datasets_source_dir}/popqa/popqa_test_filtered_10shot.csv
)

for dataset in ${DATASETS[@]}
do
    echo "Dataset:" $dataset
    echo "Devices:" $devices
    CUDA_VISIBLE_DEVICES=${devices} python src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py \
        --data ${dataset} \
        --text-col claim \
        --label-col is_hallu \
        --chat-template "generate_true_statement" \
        --batch-size 16 \
        --output-path data/uncertainty_quantification/$(basename ${dataset} .csv)_raw_stats.pt
        # --max-samples ${max_samples} \
done