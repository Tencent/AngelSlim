#!/bin/bash
# =============================================================================
# Step 2: Train DFlash draft model in OFFLINE mode.
#
# Prerequisites:
#   Run generate_qwen3_dflash_data.sh first to produce the .ckpt files.
#
# Usage:
#   bash scripts/speculative/run_qwen3_dflash_offline.sh [NUM_GPUS]
# =============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $(dirname $SCRIPT_DIR))

# Use local source code instead of installed site-packages
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

NUM_GPUS=${1:-8}

# ---- Paths -- modify these to match your environment ----
TARGET_MODEL_PATH=""
TRAIN_HIDDEN_PATH=""
OUTPUT_DIR="${ROOT_DIR}/outputs/"

# WandB configuration
export WANDB_PROJECT=${WANDB_PROJECT:-"angelslim-qwen3-4b-dflash"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"qwen3-4b-dflash-offline"}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/tools/train_dflash_offline.py \
    --target_model_name_or_path $TARGET_MODEL_PATH \
    --draft_model_config_path $ROOT_DIR/configs/qwen3_dflash.json \
    --train_hidden_path $TRAIN_HIDDEN_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 12 \
    --per_device_train_batch_size 2 \
    --learning_rate 6e-4 \
    --warmup_ratio 0.04 \
    --max_grad_norm 1.0 \
    --model_max_length 3072 \
    --chat_template_type qwen3 \
    --attention_backend flex_attention \
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
