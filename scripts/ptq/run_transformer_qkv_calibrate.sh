#!/bin/bash
# Transformer-based QKV Scale Calibration Script for HunyuanV3 MoE
# Uses HuggingFace Transformers inference (no vLLM / Ray required)
# Runs on a single node with device_map="auto" (multi-GPU supported)

# ============================================================
# Model & data paths
# ============================================================
# model_path=/apdcephfs_zwfy/share_300532381/bbeiizhang/ckpt/siritao_ckpt/global_step_hf
model_path=/apdcephfs_gy4/share_301053287/bbeiizhang/global_step_hf
# ptq_data_path=/cfs_cloud_code/bbeiizhang/data/hyeval3_math_n5_high.json
ptq_data_path=/apdcephfs_gy8/share_301053287/bbeiizhang/hyeval3_math_n5_high.parquet
output_dir=/cfs_cloud_code/bbeiizhang/code/AngelSlim_github/ckpt/output_transformer_qkv_scales

# Full model output (BF16 weights + injected FP8 QKV scales, vLLM-ready)
# Set to empty string "" to skip full model saving
model_output_dir=/cfs_cloud_code/bbeiizhang/code/AngelSlim_github/ckpt/output_transformer_qkv_model

# ============================================================
# Calibration settings
# ============================================================
num_samples=32
max_length=8192
batch_size=1

# ============================================================
# Scale headroom factors (same as vLLM version)
# ============================================================
qk_headroom=4.0
v_headroom=4.0

# ============================================================
# Run name (for log file)
# ============================================================
run_name=transformer_qkv_calibrate_hyv3

# ============================================================
# Environment
# ============================================================
# Use SDPA to avoid flash_attn dependency
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
# Avoid tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Build optional model-output-dir arg
model_output_dir_arg=""
if [ -n "$model_output_dir" ]; then
    model_output_dir_arg="--model-output-dir $model_output_dir"
fi

mkdir -p logs

python3 tools/run_transformer_qkv_calibrate.py \
    --model-path "$model_path" \
    --ptq-data-path "$ptq_data_path" \
    --output-dir "$output_dir" \
    --num-samples "$num_samples" \
    --max-length "$max_length" \
    --batch-size "$batch_size" \
    --qk-headroom "$qk_headroom" \
    --v-headroom "$v_headroom" \
    $model_output_dir_arg \
    2>&1 | tee "logs/${run_name}.log"
