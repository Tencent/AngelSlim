#!/bin/bash
# =============================================================================
# GLM-5 W8A8 TWO-NODE PTQ launcher — "kunlun" recipe equivalent.
#
# Strict 1-to-1 match with ``kunlunw8a8/recipe.yaml``:
#   * Weights  -> INT8 per-channel   (symmetric, strategy=channel)
#   * Activ.   -> INT8 per-token     (symmetric, strategy=token, dynamic)
#   * KV cache -> INT8 DYNAMIC       (MLA NoPE per-block-128 + indexer K
#                                     per-token; RoPE tail bf16; NO scales
#                                     persisted).
#
# Ignored (bf16):
#   lm_head / embed_tokens / mlp.gate.* / model.layers.<N>.eh_proj /
#   self_attn.indexer.wq_b / self_attn.indexer.wk /
#   self_attn.indexer.weights_proj / self_attn.indexer.k_norm
#
# Quantized (INT8):
#   q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj /
#   MoE experts.<i>.{gate,up,down}_proj / shared_experts.* / dense MLP
#
# HOW TO RUN ON TWO NODES
#   NODE 0:  bash scripts/ptq/run_glm5_w8a8_kunlun_2node_0.sh
#   NODE 1:  bash scripts/ptq/run_glm5_w8a8_kunlun_2node_1.sh
#   MASTER_ADDR must be reachable from both nodes.
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
# Set MASTER_ADDR to the IP of node 0.  You MUST override this in your
# environment (or edit the line below) if the default is not reachable.
export MASTER_ADDR=${MASTER_ADDR:-28.48.117.138}
export MASTER_PORT=${MASTER_PORT:-29556}   # different from w8a8c8 to avoid collision
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Optional: pin visible GPUs (default = all 8 on the local node).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Model + config.  Model lives on the shared cephfs so both nodes see it.
# NOTE: MODEL_PATH intentionally defaults to a SHARED filesystem so both
# nodes see the same tokenizer + weights.  If you point it at a node-local
# path (e.g. ``/dockerdata/...``), you MUST manually pre-load an identical
# copy on every node -- otherwise ``from_pretrained`` on rank>=8 will
# blow up with FileNotFoundError.
MODEL_PATH="${MODEL_PATH:-/apdcephfs_sgfd2/share_300532381/harviexu/chatglm5.2}"
SAVE_PATH="${SAVE_PATH:-/apdcephfs_sgfd2/share_300532381/harviexu/kunlun_harvie/w8a8}"
CONFIG="${CONFIG:-configs/glm5/w8a8_int8/glm5_w8a8_kunlun_2node.yaml}"

mkdir -p "${SAVE_PATH}" logs

# NOTE: do NOT wrap this in an outer torchrun — tools/run.py sees
# ``enable_expert_parallel: true`` in the YAML and os.execv-relaunches
# itself under torchrun --multi-nodes.  We just exec run.py directly.
python3 tools/run.py \
    -c "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --save-path  "${SAVE_PATH}" \
    2>&1 | tee "logs/run_glm5_w8a8_kunlun_2node_$(date +%Y%m%d_%H%M%S).log"
