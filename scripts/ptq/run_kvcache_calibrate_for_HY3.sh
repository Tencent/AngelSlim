#!/bin/bash

# Allow function serialization for apply_model in vLLM v1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1
# NOTE: VLLM_MOE_COLLECT_STATS is intentionally NOT set here –
#       this script only calibrates kv-cache, not weight/activation/MoE.

export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

CONFIG=configs/HY3/ptq/HY3_kvcache_calibrate.yaml

mkdir -p logs

python3 tools/kvcache/run_kvcache_calibrate.py \
    -c $CONFIG \
    2>&1 | tee logs/run_kvcache_calibrate_HY3.log
