#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ALLOC_CONF="expandable_segments:True"

NPROC=${NPROC:-8}
CONFIG=${CONFIG:-configs/qwen3/distill/fp/qwen3-1_7b_fp_distill_cakld_from_qwen3-4b_zero2.yaml}

torchrun --nproc_per_node=${NPROC} \
  tools/run.py \
  -c "${CONFIG}"
