#!/bin/bash

# ==========================================================================
# AngelSlim DFlare Online Training — Fully Aligned Configuration
# ==========================================================================
#
# Recommended training entry for DFlare. DFlare is the enhanced DFlash
# variant with separate context/noise k/v projections and learnable
# per-layer fusion weights. Training-side logic is identical to DFlash, so
# all alignment features below apply unchanged:
#
#   - loss_decay_gamma: 7 (fixed by default; pass --gamma_warmup to enable
#     per-epoch increment via --gamma_warmup_step).
#   - block_size: 16, num_anchors: 512.
#   - batch_size: 2, lr: 6e-4, cosine schedule, warmup_ratio: 0.04.
#   - max_length: 3072, num_epochs: 6.
#   - num_proc: 64 for data preprocessing.
#   - Target model uses flash_attention_2 (matches the inference kernel and
#     avoids train/test attention-backend mismatch).
#   - dataloader_drop_last=True (avoids FSDP shape mismatches on the
#     trailing batch).
#   - FP32 master weights optimizer (fp32 accumulation + fp32 grad clip;
#     only the final copy-back introduces bf16 quantization).
#   - FSDP shard_grad_op + auto_wrap (with configs/fsdp_config.json:
#     NO_WRAP, use_orig_params=True).
#
# Usage:
#   bash scripts/speculative/run_dflare_online.sh [NUM_GPUS] [ATTENTION_BACKEND]
#
# ==========================================================================

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $(dirname $SCRIPT_DIR))

# Use local source code instead of installed site-packages
export PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}

# ==========================================================================
# Paths — set these before running (left empty by default for portability)
# ==========================================================================
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-""}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-"${ROOT_DIR}/configs/qwen3_dflare.json"}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/outputs/qwen3-4b-dflare-aligned"}

if [ -z "$TARGET_MODEL_PATH" ]; then
    echo "[ERROR] TARGET_MODEL_PATH is empty. Export it to your local Qwen3 (or other) HF model dir."
    exit 1
fi
if [ -z "$TRAIN_DATA_PATH" ]; then
    echo "[ERROR] TRAIN_DATA_PATH is empty. Export it to a JSON/JSONL conversation dataset file."
    exit 1
fi

# ==========================================================================
# torch.compile / inductor kernel cache
# ==========================================================================
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${ROOT_DIR}/cache/compiled_kernels}

# ==========================================================================
# Data preprocessing parallelism
# ==========================================================================
DATA_NUM_PROC=${DATA_NUM_PROC:-64}

# ==========================================================================
# WandB configuration
# ==========================================================================
export WANDB_PROJECT=${WANDB_PROJECT:-"angelslim-qwen3-4b-dflare"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"angelslim-qwen3-4b-dflare-fp32master-aligned"}

# ==========================================================================
# Multi-node configuration (optional)
# ==========================================================================
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-12347}

if [ "$NNODES" -gt 1 ]; then
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPUS --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
    echo "[INFO] Multi-node training: nnodes=$NNODES, node_rank=$NODE_RANK, master=$MASTER_ADDR:$MASTER_PORT"
else
    DISTRIBUTED_ARGS="--standalone --nproc_per_node $NUM_GPUS"
    echo "[INFO] Single-node training: $NUM_GPUS GPUs"
fi

# ==========================================================================
# NCCL multi-node communication (for H20 + RoCE 400Gbps); harmless on single node
# ==========================================================================
if [ "$NNODES" -gt 1 ]; then
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_TIMEOUT=23
    export NCCL_IB_RETRY_CNT=7
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_CROSS_NIC=1
    export NCCL_ALGO=Ring
    export NCCL_PROTO=Simple
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_TIMEOUT=1800
fi

echo "[INFO] Draft config: $DRAFT_CONFIG_PATH"
echo "[INFO] Target model: $TARGET_MODEL_PATH"
echo "[INFO] Train data: $TRAIN_DATA_PATH"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Attention backend (draft): $ATTENTION_BACKEND"
echo "[INFO] Target model attn: flash_attention_2 (set in target_model_wrapper.py)"
echo "[INFO] Draft architecture: dflare (--draft_arch dflare)"
echo "[INFO] WandB project: $WANDB_PROJECT, run: $WANDB_RUN_NAME"
echo ""

# ==========================================================================
# Launch training
# ==========================================================================
torchrun $DISTRIBUTED_ARGS \
    $ROOT_DIR/tools/train_dflash_online.py \
    --target_model_name_or_path $TARGET_MODEL_PATH \
    --draft_model_config_path $DRAFT_CONFIG_PATH \
    --draft_arch dflare \
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
    --loss_decay_gamma 7 \
    --num_proc $DATA_NUM_PROC \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 5000 \
    --bf16 \
    --lr_scheduler_type cosine \
    --dataloader_drop_last \
    --fsdp "shard_grad_op auto_wrap" \
    --fsdp_config ${ROOT_DIR}/configs/fsdp_config.json \
    --report_to wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME
