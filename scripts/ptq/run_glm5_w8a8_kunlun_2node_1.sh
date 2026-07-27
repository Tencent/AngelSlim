#!/bin/bash
# =============================================================================
# GLM-5 W8A8 TWO-NODE PTQ launcher — "kunlun" recipe equivalent.
#
# See run_glm5_w8a8_kunlun_2node_0.sh for the full header.  This is the
# NODE 1 (rank 1) counterpart — same YAML, same MASTER_ADDR / MASTER_PORT,
# but NODE_RANK=1.
# =============================================================================

set -euo pipefail

REPO_ROOT="/apdcephfs_sgfd2/share_300532381/harviexu/AngelSlim"
cd "${REPO_ROOT}"

# Make sure the LOCAL AngelSlim (this repo) is what gets imported.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Sensible torch defaults for 16-GPU two-node calibration.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

# ---- torchrun / multi-node rendezvous (override per node) -------------------
# MASTER_ADDR MUST match what run_glm5_w8a8_kunlun_2node_0.sh uses.
export MASTER_ADDR=${MASTER_ADDR:-28.48.117.138}
export MASTER_PORT=${MASTER_PORT:-29556}   # different from w8a8c8 to avoid collision
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Optional: pin visible GPUs (default = all 8 on the local node).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Model + config.
# NOTE: MODEL_PATH intentionally defaults to a SHARED filesystem so both
# nodes see the same tokenizer + weights.  If you point it at a node-local
# path (e.g. ``/dockerdata/...``), you MUST manually pre-load an identical
# copy on every node -- otherwise ``from_pretrained`` on rank>=8 will
# blow up with FileNotFoundError.
MODEL_PATH="${MODEL_PATH:-/apdcephfs_sgfd2/share_300532381/harviexu/chatglm5.2}"
SAVE_PATH="${SAVE_PATH:-/apdcephfs_sgfd2/share_300532381/harviexu/kunlun_harvie/w8a8}"
CONFIG="${CONFIG:-configs/glm5/w8a8_int8/glm5_w8a8_kunlun_2node.yaml}"

mkdir -p "${SAVE_PATH}" logs

python3 tools/run.py \
    -c "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --save-path  "${SAVE_PATH}" \
    2>&1 | tee "logs/run_glm5_w8a8_kunlun_2node_$(date +%Y%m%d_%H%M%S).log"
