#!/usr/bin/env bash
# Copyright 2025 Tencent Inc. All Rights Reserved.
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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

FORMAT="${FORMAT:-mxfp4}"
SAVE_FORMAT="${SAVE_FORMAT:-fake}"
NPROC="${NPROC:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29514}"
MODEL_PATH="${MODEL_PATH:-}"
SAVE_PATH="${SAVE_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

case "${FORMAT}:${SAVE_FORMAT}" in
  mxfp4:fake)
    DEFAULT_CONFIG="configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_zero3.yaml"
    ;;
  nvfp4:fake)
    DEFAULT_CONFIG="configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_zero3.yaml"
    ;;
  mxfp4:real)
    DEFAULT_CONFIG="configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_real_zero3.yaml"
    ;;
  nvfp4:real)
    DEFAULT_CONFIG="configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_real_zero3.yaml"
    ;;
  *)
    echo "Unsupported FORMAT:SAVE_FORMAT=${FORMAT}:${SAVE_FORMAT}" >&2
    echo "FORMAT must be mxfp4 or nvfp4; SAVE_FORMAT must be fake or real." >&2
    exit 2
    ;;
esac

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${CONFIG}" >&2
  exit 2
fi
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer, got: ${NPROC}" >&2
  exit 2
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_ARGS=(tools/run.py -c "${CONFIG}")
if [[ -n "${MODEL_PATH}" ]]; then
  RUN_ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${SAVE_PATH}" ]]; then
  RUN_ARGS+=(--save-path "${SAVE_PATH}")
fi
RUN_ARGS+=("$@")

echo "FOCUS FP4 QAT"
echo "  format:      ${FORMAT}"
echo "  save format: ${SAVE_FORMAT}"
echo "  config:      ${CONFIG}"
echo "  processes:   ${NPROC}"
[[ -n "${MODEL_PATH}" ]] && echo "  model path:  ${MODEL_PATH}"
[[ -n "${SAVE_PATH}" ]] && echo "  save path:   ${SAVE_PATH}"

COMMAND=(torchrun
  --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${RUN_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "  command:"
  printf " %q" "${COMMAND[@]}"
  printf "\n"
  exit 0
fi

exec "${COMMAND[@]}"
