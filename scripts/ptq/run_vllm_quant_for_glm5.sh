#!/usr/bin/env bash
# =============================================================================
# Unified GLM-5 quantization pipeline.
#
# Select the source checkpoint format with GLM5_SOURCE_FORMAT:
#
#   GLM5_SOURCE_FORMAT=fp8 (default)
#     Stage 1: vLLM MoE activation calibration
#     Stage 2: FP8 blockwise -> routed-expert NVFP4 weight-only +
#              dense FP8 ue8m0 re-quantization
#
#   GLM5_SOURCE_FORMAT=bf16
#     Stage 1: vLLM activation/MoE calibration
#     Stage 2: select exactly one method with BF16_QUANT_METHOD:
#       weight_only (default): routed-expert NVFP4 weight-only
#       gptq:                  routed-expert NVFP4-GPTQ
#     Stage 3: merge NVFP4 MoE weights/scales with the BF16 checkpoint
#
# Common examples:
#   bash scripts/ptq/run_vllm_quant_for_glm5.sh
#   GLM5_SOURCE_FORMAT=bf16 \
#     BF16_QUANT_METHOD=weight_only \
#     SOURCE_MODEL_PATH=/path/to/GLM-5.1 \
#     bash scripts/ptq/run_vllm_quant_for_glm5.sh
#   GLM5_SOURCE_FORMAT=bf16 \
#     BF16_QUANT_METHOD=gptq \
#     SOURCE_MODEL_PATH=/path/to/GLM-5.1 \
#     bash scripts/ptq/run_vllm_quant_for_glm5.sh
#
# NOTE: Run this script from the AngelSlim repository root.
# =============================================================================

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/ptq/run_vllm_quant_for_glm5.sh [options]

Source format:
  GLM5_SOURCE_FORMAT=fp8|bf16   Source checkpoint format (default: fp8).
  BF16_QUANT_METHOD=weight_only|gptq
                                BF16 quantization method (default: weight_only).

Options:
  --skip-calibrate             Skip vLLM calibration for the selected source.
  --skip-fp8-quantize          Skip FP8 -> NVFP4/FP8-ue8m0 conversion.
  --skip-quantize              Skip the selected quantization stage; calibration
                               still runs unless --skip-calibrate is also set.
  --skip-merge                 Skip BF16 NVFP4/BF16 checkpoint merging.
  -h, --help                   Show this help.

Important environment variables:
  SOURCE_MODEL_PATH            Override the source model path in the YAML.
  FP8_PTQ_CONFIG               FP8 calibration/conversion YAML.
  FP8_STATISTICS_PATH          Override the FP8 calibration output directory.
  FP8_OUTPUT_PATH              Override the converted FP8-source output path.
  BF16_CALIB_CONFIG            BF16 vLLM calibration YAML.
  BF16_STATISTICS_PATH         BF16 vLLM calibration output directory.
  BF16_MODEL_PATH              Local BF16 checkpoint used by merge_nvfp4.py.
  NVFP4_WEIGHT_ONLY_CONFIG     BF16 NVFP4 weight-only YAML.
  NVFP4_GPTQ_CONFIG            BF16 NVFP4-GPTQ YAML.
  BF16_WEIGHT_ONLY_SAVE_ROOT   Root passed to tools/run.py --save-path.
  BF16_GPTQ_SAVE_ROOT          Root passed to tools/run.py --save-path.
  BF16_NVFP4_MODEL_PATH        Override the selected Stage-2 checkpoint path.
  BF16_MERGED_OUTPUT_PATH      Final merged Hugging Face checkpoint directory.
  BF16_MERGE_NUM_WORKERS       Merge worker count (default: 8).
  LOG_DIR                      Pipeline log directory.

For tools/run.py outputs, the YAML filename is appended to the configured save
root, for example:
  ${BF16_WEIGHT_ONLY_SAVE_ROOT}/glm5_1_nvfp4_weight_only
  ${BF16_GPTQ_SAVE_ROOT}/glm5_1_nvfp4_gptq
EOF
}

# ----------------------------------------------------------------------------
# CLI flags
# ----------------------------------------------------------------------------
do_calibrate=1
do_fp8_quantize=1
do_bf16_quantize=1
do_merge=1
for arg in "$@"; do
    case "${arg}" in
        --skip-calibrate)    do_calibrate=0 ;;
        --skip-fp8-quantize) do_fp8_quantize=0 ;;
        --skip-quantize)
            do_fp8_quantize=0
            do_bf16_quantize=0
            ;;
        --skip-merge) do_merge=0 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown flag: ${arg}" >&2
            echo "Use --help for usage." >&2
            exit 2
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Runtime / vLLM environment used by both calibration branches.
# ----------------------------------------------------------------------------
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_MOE_COLLECT_STATS=1
export RAY_DEDUP_LOGS=0
export PYTHONDONTWRITEBYTECODE=1
export VLLM_MOE_COLLECT_STATS_VERBOSE=0
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1

export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

# ----------------------------------------------------------------------------
# Pipeline configuration. All values can be overridden by environment.
# ----------------------------------------------------------------------------
GLM5_SOURCE_FORMAT="${GLM5_SOURCE_FORMAT:-fp8}"
GLM5_SOURCE_FORMAT="${GLM5_SOURCE_FORMAT,,}"
BF16_QUANT_METHOD="${BF16_QUANT_METHOD:-weight_only}"
BF16_QUANT_METHOD="${BF16_QUANT_METHOD,,}"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-}"

FP8_PTQ_CONFIG="${FP8_PTQ_CONFIG:-configs/glm5/ptq/nvfp4/glm5_vllm_ptq_moe_fp8.yaml}"
FP8_STATISTICS_PATH="${FP8_STATISTICS_PATH:-}"
FP8_OUTPUT_PATH="${FP8_OUTPUT_PATH:-}"

NVFP4_WEIGHT_ONLY_CONFIG="${NVFP4_WEIGHT_ONLY_CONFIG:-configs/glm5/ptq/nvfp4/glm5_1_nvfp4_weight_only.yaml}"
NVFP4_GPTQ_CONFIG="${NVFP4_GPTQ_CONFIG:-configs/glm5/ptq/nvfp4/glm5_1_nvfp4_gptq.yaml}"

BF16_WORK_DIR="${BF16_WORK_DIR:-output/glm5_bf16}"
BF16_CALIB_CONFIG="${BF16_CALIB_CONFIG:-configs/glm5/ptq/nvfp4/glm5_vllm_calibrate_bf16.yaml}"
BF16_STATISTICS_PATH="${BF16_STATISTICS_PATH:-${BF16_WORK_DIR}/statistics}"
BF16_MODEL_PATH="${BF16_MODEL_PATH:-${SOURCE_MODEL_PATH}}"
BF16_WEIGHT_ONLY_SAVE_ROOT="${BF16_WEIGHT_ONLY_SAVE_ROOT:-${BF16_WORK_DIR}/nvfp4_weight_only}"
BF16_GPTQ_SAVE_ROOT="${BF16_GPTQ_SAVE_ROOT:-${BF16_WORK_DIR}/nvfp4_gptq}"
BF16_NVFP4_MODEL_PATH="${BF16_NVFP4_MODEL_PATH:-}"
BF16_MERGED_OUTPUT_PATH="${BF16_MERGED_OUTPUT_PATH:-${BF16_WORK_DIR}/merged_${BF16_QUANT_METHOD}}"
BF16_MERGE_NUM_WORKERS="${BF16_MERGE_NUM_WORKERS:-8}"
LOG_DIR="${LOG_DIR:-logs}"

case "${GLM5_SOURCE_FORMAT}" in
    fp8|bf16) ;;
    *)
        echo "Invalid GLM5_SOURCE_FORMAT=${GLM5_SOURCE_FORMAT@Q}; expected 'fp8' or 'bf16'." >&2
        exit 2
        ;;
esac

if [[ "${GLM5_SOURCE_FORMAT}" == "bf16" ]]; then
    case "${BF16_QUANT_METHOD}" in
        weight_only|gptq) ;;
        *)
            echo "Invalid BF16_QUANT_METHOD=${BF16_QUANT_METHOD@Q}; expected 'weight_only' or 'gptq'." >&2
            exit 2
            ;;
    esac
fi

mkdir -p "${LOG_DIR}"

if [[ -n "${SOURCE_MODEL_PATH}" && ! -e "${SOURCE_MODEL_PATH}" ]]; then
    if [[ "${GLM5_SOURCE_FORMAT}" == "fp8" ]]; then
        echo "[pipeline] FP8 SOURCE_MODEL_PATH must be a local checkpoint directory: ${SOURCE_MODEL_PATH}" >&2
        exit 2
    fi
    echo "[pipeline] WARNING: SOURCE_MODEL_PATH does not exist locally: ${SOURCE_MODEL_PATH}" >&2
    echo "[pipeline] BF16 calibration/quantization accepts a Hugging Face model ID," >&2
    echo "[pipeline] but Stage 3 still requires a local BF16_MODEL_PATH (or --skip-merge)." >&2
fi

echo "[pipeline] GLM5_SOURCE_FORMAT=${GLM5_SOURCE_FORMAT}"
if [[ "${GLM5_SOURCE_FORMAT}" == "bf16" ]]; then
    echo "[pipeline] BF16_QUANT_METHOD=${BF16_QUANT_METHOD}"
fi
if [[ -n "${SOURCE_MODEL_PATH}" ]]; then
    echo "[pipeline] SOURCE_MODEL_PATH=${SOURCE_MODEL_PATH}"
else
    echo "[pipeline] SOURCE_MODEL_PATH is unset; using model_path from the selected YAML."
fi

# ============================================================================
# FP8 source branch
# ============================================================================
if [[ "${GLM5_SOURCE_FORMAT}" == "fp8" ]]; then
    calibrate_args=(-c "${FP8_PTQ_CONFIG}")
    fp8_quantize_args=(-c "${FP8_PTQ_CONFIG}")
    if [[ -n "${SOURCE_MODEL_PATH}" ]]; then
        calibrate_args+=(--model-path "${SOURCE_MODEL_PATH}")
        fp8_quantize_args+=(--input_path "${SOURCE_MODEL_PATH}")
    fi
    if [[ -n "${FP8_STATISTICS_PATH}" ]]; then
        calibrate_args+=(--output-dir "${FP8_STATISTICS_PATH}")
        fp8_quantize_args+=(--output-dir "${FP8_STATISTICS_PATH}")
    fi
    if [[ -n "${FP8_OUTPUT_PATH}" ]]; then
        fp8_quantize_args+=(--output_path "${FP8_OUTPUT_PATH}")
    fi

    if [[ "${do_calibrate}" -eq 1 ]]; then
        echo "[pipeline] === FP8 Stage 1/2: vLLM MoE activation calibration ==="
        echo "[pipeline] FP8_PTQ_CONFIG=${FP8_PTQ_CONFIG}"

        python3 tools/run_vllm_calibrate.py \
            "${calibrate_args[@]}" \
            2>&1 | tee "${LOG_DIR}/run_vllm_quant_glm5-fp8-calibrate.log"

        echo "[pipeline] FP8 calibration finished."
    else
        echo "[pipeline] --skip-calibrate set, skipping FP8 calibration."
    fi

    if [[ "${do_fp8_quantize}" -eq 1 ]]; then
        echo "[pipeline] === FP8 Stage 2/2: routed-expert NVFP4 + dense FP8 ue8m0 ==="
        echo "[pipeline] FP8_PTQ_CONFIG=${FP8_PTQ_CONFIG}"

        python3 tools/glm5_nvfp4_weight_only_blockwise.py \
            "${fp8_quantize_args[@]}" \
            2>&1 | tee "${LOG_DIR}/run_vllm_quant_glm5-fp8-quantize.log"

        echo "[pipeline] FP8-source quantization finished."
    else
        echo "[pipeline] FP8 quantization disabled, skipping conversion."
    fi

    echo "[pipeline] FP8 source branch completed successfully."
    exit 0
fi

# ============================================================================
# BF16 source branch: calibrate, run one selected method, then merge.
# ============================================================================
case "${BF16_QUANT_METHOD}" in
    weight_only)
        selected_quant_config="${NVFP4_WEIGHT_ONLY_CONFIG}"
        selected_save_root="${BF16_WEIGHT_ONLY_SAVE_ROOT}"
        selected_quant_label="NVFP4 weight-only"
        selected_log_name="weight-only"
        ;;
    gptq)
        selected_quant_config="${NVFP4_GPTQ_CONFIG}"
        selected_save_root="${BF16_GPTQ_SAVE_ROOT}"
        selected_quant_label="NVFP4-GPTQ"
        selected_log_name="gptq"
        ;;
esac

selected_config_name="$(basename "${selected_quant_config}")"
selected_config_stem="${selected_config_name%.*}"
if [[ -z "${BF16_NVFP4_MODEL_PATH}" ]]; then
    BF16_NVFP4_MODEL_PATH="${selected_save_root}/${selected_config_stem}"
fi

bf16_calibrate_args=(
    -c "${BF16_CALIB_CONFIG}"
    --output-dir "${BF16_STATISTICS_PATH}"
)
if [[ -n "${SOURCE_MODEL_PATH}" ]]; then
    bf16_calibrate_args+=(--model-path "${SOURCE_MODEL_PATH}")
fi

if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[pipeline] === BF16 Stage 1/3: vLLM activation/MoE calibration ==="
    echo "[pipeline] BF16_CALIB_CONFIG=${BF16_CALIB_CONFIG}"
    echo "[pipeline] BF16_STATISTICS_PATH=${BF16_STATISTICS_PATH}"

    python3 tools/run_vllm_calibrate.py \
        "${bf16_calibrate_args[@]}" \
        2>&1 | tee "${LOG_DIR}/run_vllm_quant_glm5-bf16-calibrate.log"

    echo "[pipeline] BF16 vLLM calibration finished."
else
    echo "[pipeline] --skip-calibrate set, skipping BF16 vLLM calibration."
fi

if [[ "${do_bf16_quantize}" -eq 1 ]]; then
    quantize_args=(
        -c "${selected_quant_config}"
        --save-path "${selected_save_root}"
    )
    if [[ -n "${SOURCE_MODEL_PATH}" ]]; then
        quantize_args+=(--model-path "${SOURCE_MODEL_PATH}")
    fi

    echo "[pipeline] === BF16 Stage 2/3: ${selected_quant_label} quantization ==="
    echo "[pipeline] SELECTED_QUANT_CONFIG=${selected_quant_config}"
    echo "[pipeline] SELECTED_SAVE_ROOT=${selected_save_root}"
    echo "[pipeline] Expected Stage-2 checkpoint=${BF16_NVFP4_MODEL_PATH}"

    python3 tools/run.py \
        "${quantize_args[@]}" \
        2>&1 | tee "${LOG_DIR}/run_vllm_quant_glm5-bf16-${selected_log_name}.log"

    echo "[pipeline] BF16 ${selected_quant_label} quantization finished."
else
    echo "[pipeline] --skip-quantize set, reusing Stage-2 checkpoint:"
    echo "[pipeline]   ${BF16_NVFP4_MODEL_PATH}"
fi

if [[ "${do_merge}" -eq 1 ]]; then
    if [[ ! -f "${BF16_STATISTICS_PATH}/moe_expert_stats.json" ]]; then
        echo "[pipeline] Missing BF16 MoE calibration statistics:" >&2
        echo "[pipeline]   ${BF16_STATISTICS_PATH}/moe_expert_stats.json" >&2
        echo "[pipeline] Run Stage 1 or set BF16_STATISTICS_PATH to existing statistics." >&2
        exit 2
    fi
    if [[ -z "${BF16_MODEL_PATH}" || ! -d "${BF16_MODEL_PATH}" ]]; then
        echo "[pipeline] BF16_MODEL_PATH must be a local BF16 checkpoint directory for Stage 3." >&2
        echo "[pipeline] Current value: ${BF16_MODEL_PATH@Q}" >&2
        exit 2
    fi
    if [[ ! -d "${BF16_NVFP4_MODEL_PATH}" ]]; then
        echo "[pipeline] Stage-2 NVFP4 checkpoint directory not found: ${BF16_NVFP4_MODEL_PATH}" >&2
        echo "[pipeline] Set BF16_NVFP4_MODEL_PATH when reusing a custom/existing checkpoint." >&2
        exit 2
    fi

    echo "[pipeline] === BF16 Stage 3/3: merge NVFP4 MoE and BF16 weights ==="
    echo "[pipeline] BF16_STATISTICS_PATH=${BF16_STATISTICS_PATH}"
    echo "[pipeline] BF16_NVFP4_MODEL_PATH=${BF16_NVFP4_MODEL_PATH}"
    echo "[pipeline] BF16_MODEL_PATH=${BF16_MODEL_PATH}"
    echo "[pipeline] BF16_MERGED_OUTPUT_PATH=${BF16_MERGED_OUTPUT_PATH}"

    python3 tools/merge_nvfp4.py \
        --statistics_path "${BF16_STATISTICS_PATH}" \
        --nvfp4_modelpath "${BF16_NVFP4_MODEL_PATH}" \
        --bf16_modelpath "${BF16_MODEL_PATH}" \
        --output_path "${BF16_MERGED_OUTPUT_PATH}" \
        --num_workers "${BF16_MERGE_NUM_WORKERS}" \
        2>&1 | tee "${LOG_DIR}/run_vllm_quant_glm5-bf16-merge-${selected_log_name}.log"

    echo "[pipeline] BF16 merged checkpoint finished."
else
    echo "[pipeline] --skip-merge set, skipping BF16 checkpoint merge."
fi

echo "[pipeline] BF16 source branch completed successfully."
