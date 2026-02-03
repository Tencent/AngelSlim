#!/bin/bash
# Copyright 2026 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script performs a functional smoke test using the Universal Adapter.
# It verifies that the model architecture can be wrapped and executed for a single inference.

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
CONFIG_PATH="configs/qwen2_5_vl/pruning/visionzip_r0.9.yaml"

echo "[AngelSlim] Starting Universal Pruning Smoke Test..."
echo "[AngelSlim] Model: $MODEL_PATH"
echo "[AngelSlim] Config: $CONFIG_PATH"

python tools/test_universal_pruning.py \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_PATH"

if [ $? -eq 0 ]; then
    echo "[AngelSlim] Test completed successfully."
else
    echo "[AngelSlim] Test failed. Please check the logs."
    exit 1
fi