#!/bin/bash

export CONFIG_DIR=angelslim/compressor/speculative/train/configs
export TARGET_MODEL_NAME_OR_PATH=tencent/HunyuanOCR
export DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/hunyuan_ocr-eagle3.json
export TRAIN_DATA_PATH=
export TRAIN_HIDDEN_PATH=
export EVAL_HIDDEN_PATH=
export OUTPUT_DIR=
export RUN_NAME=hunyuan-ocr-eagle3-angelslim
export MODEL_MAX_LENGTH=8192
export LM_HEAD_KEY="model.embed_tokens.weight"
export CHAT_TEMPLATE_TYPE=hunyuan_vl
export EMBED_WEIGHT_KEY="model.embed_tokens.weight"

torchrun --nproc_per_node=8 tools/train_eagle3_offline.py \
    --modal_type VLM \
    --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
    --draft_model_config_path  $DRAFT_MODEL_CONFIG_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --train_hidden_path $TRAIN_HIDDEN_PATH \
    --eval_hidden_path $EVAL_HIDDEN_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 10000 \
    --eval_steps 20000 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 100 \
    --model_max_length $MODEL_MAX_LENGTH \
    --lm_head_key $LM_HEAD_KEY \
    --embed_weight_key $EMBED_WEIGHT_KEY \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --deepspeed $CONFIG_DIR/deepspeed_zero3.json \
    --report_to wandb \
    --run_name  $RUN_NAME \
    --num_proc 8 \
    --training_time_test_length 4 \
    --bf16
