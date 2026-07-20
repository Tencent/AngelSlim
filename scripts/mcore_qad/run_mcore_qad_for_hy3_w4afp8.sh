#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NPROC=${NPROC:-8}
CONFIG=${CONFIG:-configs/hunyuan/mcore_qad/hy_v3_w4afp8.yaml}

cd "${ROOT_DIR}"
torchrun --nproc_per_node="${NPROC}" \
  tools/run.py \
  -c "${CONFIG}"
