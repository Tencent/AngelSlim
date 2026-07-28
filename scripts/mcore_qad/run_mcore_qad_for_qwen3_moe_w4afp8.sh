#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NPROC=${NPROC:-8}
CONFIG=${CONFIG:-configs/qwen3/mcore_qad/qwen3_moe_w4afp8.yaml}

cd "${ROOT_DIR}"
torchrun --nproc_per_node="${NPROC}" \
  tools/run.py \
  -c "${CONFIG}"
