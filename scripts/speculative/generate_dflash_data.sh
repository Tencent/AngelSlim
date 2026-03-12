#!/bin/bash
# =============================================================================
# Step 1: Pre-generate DFlash training data (hidden states) from target model.
#
# Usage:
#   bash scripts/speculative/generate_qwen3_dflash_data.sh [NUM_GPUS]
#
# Output:
#   One .ckpt file per training sample, saved to OUTPUT_DIR.
#   Each file contains: input_ids, hidden_states (5-layer concat), loss_mask,
#   attention_mask.
# =============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $(dirname $SCRIPT_DIR))

# Use local source code instead of installed site-packages
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

NUM_GPUS=${1:-8}

# ---- Paths -- modify these to match your environment ----
TARGET_MODEL_PATH=""
TRAIN_DATA_PATH=""
OUTPUT_DIR="${ROOT_DIR}/outputs/"  # directory for .ckpt files

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/tools/generate_dflash_data.py \
    --target_model_name_or_path $TARGET_MODEL_PATH \
    --draft_model_config_path $ROOT_DIR/configs/qwen3_dflash.json \
    --train_data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_max_length 3072 \
    --chat_template_type qwen3 \
    --batch_size 1 \
    --num_proc 16 \
    --sample_num 128 \
    --shard_size 10000
