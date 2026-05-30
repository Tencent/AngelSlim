#!/bin/bash

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
CONFIG_PATH="configs/qwen2_5_vl/pruning/visionzip_r0.9.yaml"
TASKS="textvqa"
OUTPUT_DIR="./results/idpruner_test"

echo "[AngelSlim] Starting Pruning Evaluation..."
echo "[AngelSlim] Model: $MODEL_PATH"
echo "[AngelSlim] Config: $CONFIG_PATH"
echo "[AngelSlim] Tasks: $TASKS"
echo "[AngelSlim] Output Directory: $OUTPUT_DIR"

python tools/run_token_pruning_evaluation.py \
    --model_path "$MODEL_PATH" \
    --configs "$CONFIG_PATH" \
    --tasks "$TASKS" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "[AngelSlim] Evaluation completed successfully."
    echo "[AngelSlim] Results saved to: $OUTPUT_DIR"
else
    echo "[AngelSlim] Evaluation failed. Please check the logs."
    exit 1
fi