#!/bin/bash

export TARGET_MODEL_NAME_OR_PATH=Qwen/Qwen3-4B
export DRAFT_MODEL_CONFIG_PATH=./configs/qwen3/eagle3/qwen3-4b-eagle3.json
export TRAIN_DATA_PATH=
export EVAL_DATA_PATH=
export OUTPUT_DIR=
export MODEL_MAX_LENGTH=2048
export CHAT_TEMPLATE_TYPE=qwen3


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 tools/train_eagle3_online.py \
    --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
    --draft_model_config_path $DRAFT_MODEL_CONFIG_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 20 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --model_max_length $MODEL_MAX_LENGTH \
    --deepspeed configs/deepspeed/deepspeed_zero3.json \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --report_to wandb \
    --run_name qwen3-4b-eagle3-angelslim