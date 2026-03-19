#!/bin/bash

# DFlash Online Training Script for Qwen3
# Usage: bash scripts/speculative/run_qwen3_dflash_online.sh [NUM_GPUS] [ATTENTION_BACKEND]

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $(dirname $SCRIPT_DIR))

# Use local source code instead of installed site-packages
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}

# Set paths - modify these to match your environment
TARGET_MODEL_PATH=""
TRAIN_DATA_PATH=""
OUTPUT_DIR="${ROOT_DIR}/outputs/"

export CONFIG_DIR=${ROOT_DIR}/angelslim/compressor/speculative/train/configs

# WandB configuration (mirrors SpecForge's --wandb-project / --wandb-name)
export WANDB_PROJECT=${WANDB_PROJECT:-"angelslim-qwen3-4b-dflash"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"qwen3-4b-dflash"}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/tools/train_dflash_online.py \
    --target_model_name_or_path $TARGET_MODEL_PATH \
    --draft_model_config_path $ROOT_DIR/configs/qwen3_dflash.json \
    --train_data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --modal_type DFlash \
    --training_mode online \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --learning_rate 6e-4 \
    --warmup_ratio 0.04 \
    --max_grad_norm 1.0 \
    --model_max_length 3072 \
    --chat_template_type qwen3 \
    --attention_backend $ATTENTION_BACKEND \
    --block_size 16 \
    --num_anchors 512 \
    --loss_decay_gamma 7.0 \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 2500 \
    --bf16 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --run_name $WANDB_RUN_NAME

