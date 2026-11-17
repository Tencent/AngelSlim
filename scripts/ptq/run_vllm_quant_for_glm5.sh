#!/usr/bin/env bash
# =============================================================================
# Unified GLM-5 quantization pipeline.
#
# Select the source checkpoint format with GLM5_SOURCE_FORMAT:
#
#   GLM5_SOURCE_FORMAT=fp8 (default)
#     Stage 1: vLLM MoE activation calibration
#     Stage 2: tools/run.py mixed RTN: routed-expert NVFP4 + dense FP8 ue8m0,
#              exported as compressed-tensors
#
#   GLM5_SOURCE_FORMAT=bf16
#     Stage 1: vLLM calibration; gptq also collects Hessians
#     Stage 2: tools/run.py mixed, routed-expert NVFP4 with input scales from
#              stage 1 (compressed-tensors, final). Pick the algorithm with
#              BF16_QUANT_METHOD: weight_only (default, RTN) or gptq.
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
  -h, --help                   Show this help.

Important environment variables:
  SOURCE_MODEL_PATH            Override the source model path in the YAML.
  FP8_PTQ_CONFIG               Mixed YAML driving both FP8-source stages.
  FP8_STATISTICS_PATH          Override the FP8 calibration output directory.
  FP8_OUTPUT_PATH              Override the converted FP8-source output path.
  BF16_CALIB_CONFIG            BF16 vLLM calibration YAML (weight_only only).
  BF16_STATISTICS_PATH         BF16 vLLM calibration output directory.
  NVFP4_WEIGHT_ONLY_CONFIG     BF16 NVFP4 RTN mixed YAML.
  NVFP4_GPTQ_CONFIG            BF16 NVFP4-GPTQ mixed YAML.
  BF16_WEIGHT_ONLY_SAVE_ROOT   Root passed to tools/run.py --save-path.
  BF16_GPTQ_SAVE_ROOT          Root passed to tools/run.py --save-path.
  BF16_NVFP4_MODEL_PATH        Override the selected Stage-2 checkpoint path.
  LOG_DIR                      Pipeline log directory.

tools/run.py treats the save root as a run directory: weights go to out/ and the
calibration products to stats/ hessian/ image_embeds/ beside them, for example:
  ${BF16_WEIGHT_ONLY_SAVE_ROOT}/out
  ${BF16_GPTQ_SAVE_ROOT}/out
EOF
}

# ----------------------------------------------------------------------------
# CLI flags
# ----------------------------------------------------------------------------
do_calibrate=1
do_fp8_quantize=1
do_bf16_quantize=1
for arg in "$@"; do
    case "${arg}" in
        --skip-calibrate)    do_calibrate=0 ;;
        --skip-fp8-quantize) do_fp8_quantize=0 ;;
        --skip-quantize)
            do_fp8_quantize=0
            do_bf16_quantize=0
            ;;
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

FP8_PTQ_CONFIG="${FP8_PTQ_CONFIG:-configs/glm5/ptq/mixed/glm5_fp8_nvfp4_rtn.yaml}"
FP8_STATISTICS_PATH="${FP8_STATISTICS_PATH:-}"
FP8_OUTPUT_PATH="${FP8_OUTPUT_PATH:-}"

NVFP4_WEIGHT_ONLY_CONFIG="${NVFP4_WEIGHT_ONLY_CONFIG:-configs/glm5/ptq/mixed/glm5_nvfp4_rtn.yaml}"
NVFP4_GPTQ_CONFIG="${NVFP4_GPTQ_CONFIG:-configs/glm5/ptq/mixed/glm5_nvfp4_gptq.yaml}"

BF16_WORK_DIR="${BF16_WORK_DIR:-output/glm5_bf16}"
BF16_CALIB_CONFIG="${BF16_CALIB_CONFIG:-configs/glm5/ptq/nvfp4/glm5_vllm_calibrate_bf16.yaml}"
BF16_STATISTICS_PATH="${BF16_STATISTICS_PATH:-${BF16_WORK_DIR}/statistics}"
BF16_HESSIAN_PATH="${BF16_HESSIAN_PATH:-${BF16_STATISTICS_PATH}/hessian}"
BF16_WEIGHT_ONLY_SAVE_ROOT="${BF16_WEIGHT_ONLY_SAVE_ROOT:-${BF16_WORK_DIR}/nvfp4_weight_only}"
BF16_GPTQ_SAVE_ROOT="${BF16_GPTQ_SAVE_ROOT:-${BF16_WORK_DIR}/nvfp4_gptq}"
BF16_NVFP4_MODEL_PATH="${BF16_NVFP4_MODEL_PATH:-}"
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
    # tools/run.py mixed conversion reads the safetensors shards directly.
    echo "[pipeline] SOURCE_MODEL_PATH must be a local checkpoint directory: ${SOURCE_MODEL_PATH}" >&2
    exit 2
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
        fp8_quantize_args+=(--model-path "${SOURCE_MODEL_PATH}")
    fi
    if [[ -n "${FP8_STATISTICS_PATH}" ]]; then
        calibrate_args+=(--output-dir "${FP8_STATISTICS_PATH}")
        fp8_quantize_args+=(--stats-dir "${FP8_STATISTICS_PATH}")
    fi
    if [[ -n "${FP8_OUTPUT_PATH}" ]]; then
        fp8_quantize_args+=(--save-path "${FP8_OUTPUT_PATH}")
    fi

    if [[ "${do_calibrate}" -eq 1 ]]; then
        echo "[pipeline] === FP8 Stage 1/2: vLLM MoE activation calibration ==="
        echo "[pipeline] FP8_PTQ_CONFIG=${FP8_PTQ_CONFIG}"

        python3 -m angelslim.compressor.quant.vllm_calib.calibrate \
            "${calibrate_args[@]}" \
            2>&1 | tee "${LOG_DIR}/run_vllm_quant_glm5-fp8-calibrate.log"

        echo "[pipeline] FP8 calibration finished."
    else
        echo "[pipeline] --skip-calibrate set, skipping FP8 calibration."
    fi

    if [[ "${do_fp8_quantize}" -eq 1 ]]; then
        echo "[pipeline] === FP8 Stage 2/2: routed-expert NVFP4 + dense FP8 ue8m0 ==="
        echo "[pipeline] FP8_PTQ_CONFIG=${FP8_PTQ_CONFIG}"

        python3 tools/run.py \
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
# BF16 source branch: calibrate, then convert with the selected algorithm.
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

if [[ -z "${BF16_NVFP4_MODEL_PATH}" ]]; then
    # ckpt_convert treats --save-path as the run root and writes weights to out/.
    BF16_NVFP4_MODEL_PATH="${selected_save_root}/out"
fi

if [[ "${BF16_QUANT_METHOD}" == "gptq" ]]; then
    # GPTQ needs Hessians as well as MoE absmax, so stage 1 runs through
    # tools/run.py, which dispatches to vllm_calib.hessian_collect for a config with
    # calibrate.hessian_dir.
    bf16_calibrate_cmd=(python3 tools/run.py -c "${selected_quant_config}" --stage calib
                        --stats-dir "${BF16_STATISTICS_PATH}"
                        --hessian-dir "${BF16_HESSIAN_PATH}")
else
    bf16_calibrate_cmd=(python3 -m angelslim.compressor.quant.vllm_calib.calibrate
                        -c "${BF16_CALIB_CONFIG}" --output-dir "${BF16_STATISTICS_PATH}")
fi
if [[ -n "${SOURCE_MODEL_PATH}" ]]; then
    bf16_calibrate_cmd+=(--model-path "${SOURCE_MODEL_PATH}")
fi

if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[pipeline] === BF16 Stage 1: vLLM activation/MoE calibration ==="
    echo "[pipeline] BF16_STATISTICS_PATH=${BF16_STATISTICS_PATH}"
    if [[ "${BF16_QUANT_METHOD}" == "gptq" ]]; then
        echo "[pipeline] BF16_HESSIAN_PATH=${BF16_HESSIAN_PATH}"
    else
        echo "[pipeline] BF16_CALIB_CONFIG=${BF16_CALIB_CONFIG}"
    fi

    "${bf16_calibrate_cmd[@]}" \
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
    quantize_args+=(--stats-dir "${BF16_STATISTICS_PATH}")
    if [[ "${BF16_QUANT_METHOD}" == "gptq" ]]; then
        quantize_args+=(--hessian-dir "${BF16_HESSIAN_PATH}")
    fi

    echo "[pipeline] === BF16 Stage 2: ${selected_quant_label} quantization ==="
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

echo "[pipeline] Stage-2 output is final: ${BF16_NVFP4_MODEL_PATH}"

echo "[pipeline] BF16 source branch completed successfully."
