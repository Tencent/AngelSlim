#!/usr/bin/env bash
# =============================================================================
# One-click pipeline: bf16 Hy3 model -> vLLM activation calibration
#                                      -> NVFP4 weight-only quantization
#                                      -> merged NVFP4 + FP8 activation/KV scales
#
# Usage:
#   bash scripts/ptq/run_nvfp4_quant_for_Hy3.sh
#   bash scripts/ptq/run_nvfp4_quant_for_Hy3.sh --skip-calibrate
#   bash scripts/ptq/run_nvfp4_quant_for_Hy3.sh --skip-weight-only
#   bash scripts/ptq/run_nvfp4_quant_for_Hy3.sh --skip-merge
#
# Paths can be overridden with environment variables documented in
# scripts/ptq/README.md.
#
# NOTE: Must be run from the AngelSlim repository root directory.
# =============================================================================

set -euo pipefail

do_calibrate=1
do_weight_only=1
do_merge=1
for arg in "$@"; do
    case "${arg}" in
        --skip-calibrate) do_calibrate=0 ;;
        --skip-weight-only) do_weight_only=0 ;;
        --skip-merge) do_merge=0 ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: ${arg}" >&2
            echo "Use --help for usage." >&2
            exit 2
            ;;
    esac
done

# Allow function serialization for apply_model in vLLM v1 engine.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Enable MoE expert statistics collection.
export VLLM_MOE_COLLECT_STATS=1
# Force Ray to reload code (disable code caching).
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache.
export PYTHONDONTWRITEBYTECODE=1
# Disable verbose MoE stats logging.
export VLLM_MOE_COLLECT_STATS_VERBOSE=0
# Enable per-expert statistics collection.
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1

export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

# Pipeline defaults. Override these variables before invoking the script when
# using another config, checkpoint, or output directory.
PTQ_CONFIG="${PTQ_CONFIG:-configs/Hy3/ptq/fp8/Hy3_vllm_ptq_per_tensor.yaml}"
NVFP4_CONFIG="${NVFP4_CONFIG:-configs/Hy3/ptq/nvfp4_weight_only/hunyuan_a20b_nvfp4_weight_only.yaml}"

WORK_DIR="${WORK_DIR:-output/Hy3_nvfp4}"
STATISTICS_PATH="${STATISTICS_PATH:-${WORK_DIR}/statistics}"
NVFP4_W_PATH="${NVFP4_W_PATH:-${WORK_DIR}/nvfp4_weight_only}"
BF16_MODEL_PATH="${BF16_MODEL_PATH:-/path/to/bf16_model}"
OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/moe_nvfp4_merged}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"

mkdir -p "${LOG_DIR}"

# ============================================================================
# Stage 1: run FP8 vLLM activation calibration
# ============================================================================
if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[pipeline] === Stage 1/3: FP8 vLLM activation calibration ==="
    echo "[pipeline] PTQ_CONFIG=${PTQ_CONFIG}"

    python3 tools/run_vllm_calibrate.py \
        -c "${PTQ_CONFIG}" \
        --auto-detect-mtp \
        2>&1 | tee "${LOG_DIR}/run_vllm_calibrate_Hy3.log"

    echo "[pipeline] Stage 1 finished."
else
    echo "[pipeline] --skip-calibrate set, skipping stage 1."
fi

# ============================================================================
# Stage 2: run NVFP4 weight-only quantization
# ============================================================================
if [[ "${do_weight_only}" -eq 1 ]]; then
    echo "[pipeline] === Stage 2/3: NVFP4 weight-only quantization ==="
    echo "[pipeline] NVFP4_CONFIG=${NVFP4_CONFIG}"

    python3 tools/run.py \
        -c "${NVFP4_CONFIG}" \
        2>&1 | tee "${LOG_DIR}/run_nvfp4_weight_only.log"

    echo "[pipeline] Stage 2 finished."
else
    echo "[pipeline] --skip-weight-only set, skipping stage 2."
fi

# ============================================================================
# Stage 3: merge NVFP4 weights with activation and KV-cache calibration
# ============================================================================
if [[ "${do_merge}" -eq 1 ]]; then
    echo "[pipeline] === Stage 3/3: merge NVFP4 weights and calibration ==="
    echo "[pipeline] STATISTICS_PATH=${STATISTICS_PATH}"
    echo "[pipeline] NVFP4_W_PATH=${NVFP4_W_PATH}"
    echo "[pipeline] BF16_MODEL_PATH=${BF16_MODEL_PATH}"
    echo "[pipeline] OUTPUT_PATH=${OUTPUT_PATH}"

    python3 tools/merge_hy3_nvfp4_c8.py \
        --statistics_path "${STATISTICS_PATH}" \
        --nvfp4_w_path "${NVFP4_W_PATH}" \
        --bf16_model_path "${BF16_MODEL_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --mtp-fp8-mode auto \
        2>&1 | tee "${LOG_DIR}/merge_Hy3_nvfp4.log"

    echo "[pipeline] Stage 3 finished."
else
    echo "[pipeline] --skip-merge set, skipping stage 3."
fi

echo "[pipeline] All requested stages completed successfully."
