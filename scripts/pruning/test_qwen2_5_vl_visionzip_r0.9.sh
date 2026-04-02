#!/bin/bash

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
CONFIG_PATH="configs/qwen2_5_vl/pruning/visionzip_r0.9.yaml"

echo "[AngelSlim] Starting Universal Pruning Smoke Test..."
echo "[AngelSlim] Model: $MODEL_PATH"
echo "[AngelSlim] Config: $CONFIG_PATH"

python tools/test_token_pruning.py \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_PATH"

if [ $? -eq 0 ]; then
    echo "[AngelSlim] Test completed successfully."
else
    echo "[AngelSlim] Test failed. Please check the logs."
    exit 1
fi