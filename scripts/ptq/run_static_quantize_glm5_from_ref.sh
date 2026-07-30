#!/usr/bin/env bash
# =============================================================================
# Static INT8 per-channel weight-only quantization for GLM-5.
#
# Reads the BF16 checkpoint at $BF16_DIR, produces an INT8 checkpoint at
# $OUT_DIR whose ``model.safetensors.index.json`` layout is byte-identical
# to $REF_DIR (== ``kunlun_harvie/w8a8/glm5_w8a8_kunlun_2node``).
#
# No calibration.  No activation stats.  No distributed / NCCL.  Single
# process, thread-pool over destination shards.  RAM ~= (workers) *
# max-single-tensor (< 400 MB for GLM-5 MoE experts).
# =============================================================================
set -euo pipefail

BF16_DIR="${BF16_DIR:-/apdcephfs_sgfd2/share_300532381/harviexu/chatglm5.2}"
REF_DIR="${REF_DIR:-/apdcephfs_sgfd2/share_300532381/harviexu/kunlun_harvie/w8a8/glm5_w8a8_kunlun_2node}"
OUT_DIR="${OUT_DIR:-/apdcephfs_sgfd2/share_300532381/harviexu/kunlun_harvie/w8a8/glm5_w8a8_static_int8}"
WORKERS="${WORKERS:-8}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/static_quantize_glm5_from_ref.py"

echo "[cfg] BF16_DIR = ${BF16_DIR}"
echo "[cfg] REF_DIR  = ${REF_DIR}"
echo "[cfg] OUT_DIR  = ${OUT_DIR}"
echo "[cfg] WORKERS  = ${WORKERS}"

mkdir -p "${OUT_DIR}"

# First: dry-run to print stats and validate index sanity fast.
python3 "${PY_SCRIPT}" \
    --bf16 "${BF16_DIR}" \
    --ref  "${REF_DIR}" \
    --out  "${OUT_DIR}" \
    --dry-run

# Real run.
python3 "${PY_SCRIPT}" \
    --bf16 "${BF16_DIR}" \
    --ref  "${REF_DIR}" \
    --out  "${OUT_DIR}" \
    --workers "${WORKERS}"

echo "[done] output at: ${OUT_DIR}"
