GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
PRETRAIN_PATH=$5

source ../.venv/bin/activate

torchrun --nproc_per_node=${GPU_NUM} main.py \
    --output_dir ${OUTPUT_DIR} \
    -c ${CFG} \
    --datasets ${DATASETS}  \
    --pretrain_model_path ${PRETRAIN_PATH} \
    --options text_encoder_type=bert-base-uncased
