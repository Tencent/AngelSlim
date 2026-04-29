#!/usr/bin/env bash
set -euo pipefail

# Smoke-test launcher: Qwen3-4B with DeepSpeed ZeRO-3. Matches the full
# 30B-A3B flow but on a model small enough to iterate quickly.

export PYTORCH_ALLOC_CONF="expandable_segments:True"

NPROC=${NPROC:-2}
CONFIG=${CONFIG:-configs/qwen3/qat/fp8_static/learn_scale/qwen3-4b_fp8_static_end2end_learn_scale_zero3.yaml}

torchrun --nproc_per_node=${NPROC} \
  tools/run.py \
  -c "${CONFIG}"
