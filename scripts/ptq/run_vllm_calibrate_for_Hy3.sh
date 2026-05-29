#!/bin/bash

# Allow function serialization for apply_model in vLLM v1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Enable MoE expert statistics collection
export VLLM_MOE_COLLECT_STATS=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1
# Disable verbose MoE stats logging
export VLLM_MOE_COLLECT_STATS_VERBOSE=0
# Enable per-expert statistics collection
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1

export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

CONFIG=configs/Hy3/ptq/fp8/Hy3_vllm_ptq_per_tensor.yaml

mkdir -p logs

python3 tools/run_vllm_calibrate.py \
    -c $CONFIG \
    2>&1 | tee logs/run_vllm_calibrate_Hy3.log
