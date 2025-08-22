#!/bin/bash
set -e
set -x

readonly SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
readonly PROJECT_DIR="${SCRIPT_DIR}/.."

cd "${PROJECT_DIR}/benchmark/pytorch"
python3 generate_eagle_answer.py \
    --eagle-model-path /path/to/eagle/model \
    --base-model-path /path/to/base/model \
    --model-id your-model-id \
    --num-gpus-total 8 \
    --temperature 0.0 \
    --bench-name mt_bench
