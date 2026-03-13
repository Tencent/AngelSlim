#!/bin/bash

export CONFIG_DIR=angelslim/compressor/speculative/train/configs
export TARGET_MODEL_NAME_OR_PATH=tencent/HunyuanOCR
export DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/hunyuan_ocr-eagle3.json
export TRAIN_DATA_PATH=
export EVAL_DATA_PATH=
export OUTPUT_DIR=
export EMBED_WEIGHT_KEY="model.embed_tokens.weight"
export MODEL_MAX_LENGTH=8192
export CHAT_TEMPLATE_TYPE=hunyuan_vl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 tools/train_eagle3_online.py \
    --modal_type VLM \
    --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
    --draft_model_config_path $DRAFT_MODEL_CONFIG_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_proc 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --model_max_length $MODEL_MAX_LENGTH \
    --embed_weight_key $EMBED_WEIGHT_KEY \
    --deepspeed $CONFIG_DIR/deepspeed_zero3.json \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --report_to none \
    --run_name hunyuan-ocr-eagle3-angelslim