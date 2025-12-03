echo $(which python)
devices=1
max_samples=100

datasets_source_dir="/mnt/extremessd10tb/seleznev/longtail-dataset/datasets"

DATASETS=(
    ${datasets_source_dir}/popqa/popqa_train_filtered_10shot.csv
    ${datasets_source_dir}/popqa/popqa_test_filtered_10shot.csv
)

for dataset in ${DATASETS[@]}
do
    CUDA_VISIBLE_DEVICES=${devices} python src/packing/evaluate/uncertainty_quantification/precompute_raw_stats.py \
        --data ${dataset} \
        --text-col claim \
        --label-col is_hallu \
        --max-samples ${max_samples} \
        --output-path data/uncertainty_quantification/$(basename ${dataset} .csv)_raw_stats_${max_samples}samples.pt
done