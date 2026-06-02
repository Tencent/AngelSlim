#!/bin/bash
# Phase 1 only: Collect Smooth Stats (+ optional Alpha Search).
# Must be run from the AngelSlim repository root directory.

set -euo pipefail

CONFIG="configs/hy3/ptq/hy3_smooth.yaml"

# -------- Environment Variables --------
# Allow function serialization for apply_model in vLLM v1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Enable MoE expert statistics collection
export VLLM_MOE_COLLECT_STATS=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1
# Disable verbose MoE stats logging

export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF


# -------- Environment Variables --------
export VLLM_MOE_COLLECT_SMOOTH_STATS=1
export VLLM_MOE_COLLECT_ALPHA_SEARCH=1


python3 tools/smooth/run_vllm_smooth.py -c "$CONFIG" "$@"
