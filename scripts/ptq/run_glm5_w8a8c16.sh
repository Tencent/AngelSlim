#!/bin/bash
# =============================================================================
# GLM-5 W8A8C16 low-memory PTQ launcher.
#
# * Model:  /apdcephfs_zwfy2/share_300532381/harviexu/chatglm5.2
# * Weights & activations -> INT8 (per-channel W / per-tensor static A)
# * KV cache               -> bf16 (no quantization)
# * Layer-by-layer streaming calibration on a single 8-GPU node.
# * SmoothQuant pre-processing enabled (see YAML).
#
# Model class ``GLM5`` (angelslim/models/llm/glm5.py) already force-skips:
#   kv_b_proj / DSA indexer / weights_proj / mlp.gate / mtp_block /
#   lm_head / embed_tokens
# so the YAML's ``ignore_layers`` field only needs to hold user extras.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Make sure the local AngelSlim (this repo) is what gets imported.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Sensible torch defaults for CPU-heavy layer streaming on 8x GPU node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

# Optional: pin visible GPUs (default = all 8).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

#MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy2/share_300532381/harviexu/chatglm5.2}"
MODEL_PATH="${MODEL_PATH:-/dockerdata/chatglm5.2}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/output_glm5_w8a8c16}"
CONFIG="${CONFIG:-configs/glm5/w8a8_int8/glm5_w8a8c16_low_memory.yaml}"

mkdir -p "${SAVE_PATH}" logs

python3 tools/run.py \
    -c "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --save-path  "${SAVE_PATH}" \
    2>&1 | tee "logs/run_glm5_w8a8c16_$(date +%Y%m%d_%H%M%S).log"
