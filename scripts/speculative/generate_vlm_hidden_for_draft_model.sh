#!/bin/bash

DATASET_PATH=
MODEL_NAME=
TARGET_BACKEND=hf
MODEL_MAX_LENGTH=2048
CHAT_TEMPLATE_TYPE=qwen3_vl
OUTPUT_DIR=

torchrun --nproc_per_node=8 \
    tools/generate_hidden_for_draft_model.py \
    --modal_type VLM \
    --dataset_path $DATASET_PATH \
    --model_name $MODEL_NAME \
    --target_backend $TARGET_BACKEND \
    --torch_dtype bfloat16 \
    --model_max_length $MODEL_MAX_LENGTH \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --outdir $OUTPUT_DIR
