echo $(which python)
devices=0

datasets_source_dir="../xx_datasets"
output_dir="data/uncertainty_quantification/polygraph_fixed"

DATASETS=(
    ${datasets_source_dir}/coqa/coqa_simple_instruct.csv
    ${datasets_source_dir}/coqa/coqa_simple_instruct_limit_1536.csv
    ${datasets_source_dir}/samsum/samsum_simple_instruct.csv
    ${datasets_source_dir}/samsum/samsum_simple_instruct_limit_1536.csv
    ${datasets_source_dir}/triviaqa/triviaqa_simple_instruct.csv
    ${datasets_source_dir}/triviaqa/triviaqa_simple_instruct_limit_1536.csv
    ${datasets_source_dir}/wmt14_deen/wmt14_deen.csv
    ${datasets_source_dir}/wmt14_deen/wmt14_deen_limit_1536.csv
)

mkdir -p ${output_dir}

for dataset in ${DATASETS[@]}
do
    dataset_dir=$(dirname ${dataset})
    dataset_name=$(basename ${dataset})

    echo "Dataset:" ${dataset}
    echo "Devices:" ${devices}
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${devices} python src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_polygraph_chunked.py \
        --input-dir ${dataset_dir} \
        --pattern ${dataset_name} \
        --chunk-size 1024 \
        --cache-dir ~/.cache/huggingface/hub \
        --instruct \
        --output-dir ${output_dir}
done
