echo $(which python)
devices=0

datasets_source_dir="../longtail-dataset/datasets"

DATASETS=(
    ${datasets_source_dir}/popqa/popqa_train_filtered_10shot.csv
    ${datasets_source_dir}/popqa/popqa_test_filtered_10shot.csv
)

# max_samples=6500
for dataset in ${DATASETS[@]}
do
    CUDA_VISIBLE_DEVICES=${devices} python src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py \
        --data ${dataset} \
        --text-col claim \
        --label-col is_hallu \
        --chat-template "generate_true_statement" \
        --layer-stride 2 \
        --output-path data/uncertainty_quantification/$(basename ${dataset} .csv)_raw_stats_layer_stride_2.pt
        # --max-samples ${max_samples} \
done