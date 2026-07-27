#!/bin/bash
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Quick smoke-test wrapper: load the AngelSlim W8A8C16 quantized
# checkpoint through vLLM and generate a few completions.  Use this to
# answer the question "does the quantized model still speak?" in the
# fastest way possible, without setting up an HTTP server.
#
# Environment overrides:
#   MODEL_PATH    default: ./output_glm5_w8a8c16/glm5_w8a8c16_low_memory
#   TP            tensor-parallel size, default: 8 (single-node 8 GPUs)
#   PP            pipeline-parallel size, default: 1
#   EP            enable expert-parallel (0/1), default: 1
#   GPU_MEM_UTIL  default: 0.9
#   MAX_LEN       default: 4096
#   CHAT          set to 1 to apply the chat template, 0 for raw prompts

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/output_glm5_w8a8c16/glm5_w8a8c16_low_memory}"
TP="${TP:-8}"
PP="${PP:-1}"
EP="${EP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-4096}"
CHAT="${CHAT:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CHAT_FLAG=""
if [ "${CHAT}" = "1" ]; then
    CHAT_FLAG="--apply-chat-template"
fi

EP_FLAG=""
if [ "${EP}" = "1" ]; then
    EP_FLAG="--enable-expert-parallel"
fi

echo "[verify] MODEL_PATH=${MODEL_PATH}"
echo "[verify] TP=${TP} PP=${PP} EP=${EP} GPU_MEM_UTIL=${GPU_MEM_UTIL} MAX_LEN=${MAX_LEN} CHAT=${CHAT}"
echo "[verify] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python3 scripts/deploy/verify_glm5_w8a8.py \
    --model-path "${MODEL_PATH}" \
    --tensor-parallel-size "${TP}" \
    --pipeline-parallel-size "${PP}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    ${EP_FLAG} \
    ${CHAT_FLAG}
