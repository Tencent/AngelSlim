#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ALLOC_CONF="expandable_segments:True"

NPROC=${NPROC:-8}
CONFIG=${CONFIG:-configs/hunyuan/qat/fp8_static/learn_scale/hunyuanv3_a20b_fp8_static_end2end_learn_scale_zero3.yaml}

torchrun --nproc_per_node=${NPROC} \
  tools/run.py \
  -c "${CONFIG}"
