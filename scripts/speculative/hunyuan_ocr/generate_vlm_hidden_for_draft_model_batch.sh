#!/bin/bash

DATASET_PATH=train_data
MODEL_NAME=tencent/HunyuanOCR
TARGET_BACKEND=hf
MODEL_MAX_LENGTH=8192
CHAT_TEMPLATE_TYPE=hunyuan_vl
OUTPUT_DIR=train_data_hidden_states

for ((i=0; i<32; i++)); do
    DATASET_PATH=$DATASET_PATH/split_$i.jsonl
    OUTPUT_DIR=$OUTPUT_DIR/split_$i
    torchrun --nproc_per_node=8 \
        tools/generate_hidden_for_draft_model.py \
        --modal_type VLM \
        --dataset_path $DATASET_PATH \
        --model_name $MODEL_NAME \
        --target_backend $TARGET_BACKEND \
        --torch_dtype bfloat16 \
        --model_max_length $MODEL_MAX_LENGTH \
        --chat_template_type $CHAT_TEMPLATE_TYPE \
        --outdir $OUTPUT_DIR \
        --target_model_type hunyuan_vl \
        --num_proc 8
done