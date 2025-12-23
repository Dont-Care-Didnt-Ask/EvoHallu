echo $(which python)

datasets=(
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/popqa/popqa_test_filtered_10shot.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/AVeriTeC/dev_filtered_10shot.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/geometry/common_claim_true_false.csv \
    /mnt/extremessd10tb/seleznev/longtail-dataset/datasets/x-fact/test_filtered_10shot.csv \
)

for dataset_path in ${datasets[@]}; do
    tail_name=$(basename $dataset_path .csv)
    PYTHONPATH=/mnt/extremessd10tb/seleznev/EvoTune python src/packing/evaluate/uncertainty_quantification/end2end_eval.py \
        --data $dataset_path \
        --text-col claim \
        --label-col is_hallu \
        --output-path out/logs/example/EVALS/${tail_name}_eval.csv \
        --chat-template generate_true_statement
done