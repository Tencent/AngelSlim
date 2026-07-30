#!/bin/bash
# =============================================================================
# GLM-5 W8A8C8 TWO-NODE PTQ launcher (INT8 W/A + INT8 dynamic KV cache).
#
# * Model:  /apdcephfs_zwfy2/share_300532381/harviexu/chatglm5.2
# * Weights  -> INT8 per-channel
# * Activ.   -> INT8 per-token dynamic
# * KV cache -> INT8 dynamic (MLA NoPE latent per-block-128 + indexer K
#              per-token; RoPE tail stays bf16).  Scales are NOT persisted
#              (dynamic), they are recomputed at inference time.
# * TWO-NODE: enable_expert_parallel: true makes tools/run.py auto-relaunch
#   itself under torchrun --multi-nodes (no need to wrap this script in an
#   outer torchrun — run.py does it for you).  16 GPUs hold the model
#   sharded via device_map=auto, so CPU host memory is NOT the bottleneck.
# * SmoothQuant pre-processing enabled (see YAML).
#
# HOW TO RUN ON TWO NODES
#   On EACH node, simply launch this script with bash (run.py will
#   torchrun-relaunch itself once).  Per-node overrides:
#     NODE_RANK=0  on node0      (run_glm5_w8a8c8_2node_0.sh)
#     NODE_RANK=1  on node1      (this file)
#     MASTER_ADDR=<node0_ip>      (must be reachable from both nodes)
#   All other rendezvous env (NNODES / GPUS_PER_NODE / MASTER_PORT) have
#   sensible defaults below.
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
export MASTER_ADDR=28.48.117.138
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29555}
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Optional: pin visible GPUs (default = all 8 on the local node).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Model + config.  Model lives on the shared cephfs so both nodes see it.
MODEL_PATH="${MODEL_PATH:-/dockerdata/chatglm5.2}"
#SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/output_glm5_w8a8c8_2node}"
SAVE_PATH=/dockerdata/w8a8c8_official
CONFIG="${CONFIG:-configs/glm5/w8a8_int8/glm5_w8a8c8_2node.yaml}"

mkdir -p "${SAVE_PATH}" logs

# NOTE: do NOT wrap this in an outer torchrun — tools/run.py sees
# enable_expert_parallel: true and os.execv-relaunches itself under
# torchrun --multi-nodes.  We just exec run.py directly.
python3 tools/run.py \
    -c "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --save-path  "${SAVE_PATH}" \
    2>&1 | tee "logs/run_glm5_w8a8c8_2node_$(date +%Y%m%d_%H%M%S).log"
