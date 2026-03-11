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

CONFIG=configs/deepseek_r1/w4a8_fp8/deepseek_r1_w4a8_fp8_vllm_calibrate.yaml

mkdir -p logs

python3 tools/run.py \
    -c $CONFIG \
    2>&1 | tee logs/run_vllm_quant_deepseek_v3.log