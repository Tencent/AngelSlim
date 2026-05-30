#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ALLOC_CONF="expandable_segments:True"

NPROC=${NPROC:-8}
CONFIG=${CONFIG:-configs/qwen3/qad/w4a8_fp8/qwen3-4b_w4a8_fp8_qad_zero2.yaml}

torchrun --nproc_per_node=${NPROC} \
  tools/run.py \
  -c "${CONFIG}"
