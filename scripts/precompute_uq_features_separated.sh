echo $(which python)
# devices=1

datasets_source_dir="/disk/4tb/seleznev/longtail-dataset/datasets"

# DATASETS=(
#     ${datasets_source_dir}/popqa/popqa_train_filtered_10shot.csv
#     ${datasets_source_dir}/popqa/popqa_test_filtered_10shot.csv
# )

for dataset in ${datasets_source_dir}/popqa/popqa_train_filtered_10shot.csv
do
    CUDA_VISIBLE_DEVICES=0 python src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py \
        --data ${dataset} \
        --text-col claim \
        --label-col is_hallu \
        --chat-template "generate_true_statement" \
        --output-path data/uncertainty_quantification/$(basename ${dataset} .csv)_raw_stats.pt 
done


# for dataset in ${datasets_source_dir}/popqa/popqa_test_filtered_10shot.csv
# do
#     CUDA_VISIBLE_DEVICES=1 python src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py \
#         --data ${dataset} \
#         --text-col claim \
#         --label-col is_hallu \
#         --chat-template "generate_true_statement" \
#         --output-path data/uncertainty_quantification/$(basename ${dataset} .csv)_raw_stats.pt
# done