#!/usr/bin/env bash
# =============================================================================
# One-click pipeline:  bf16 model  ->  vLLM activation calibration
#                                   ->  FP8 quantization (HF safetensors)
#
# Stage 1: tools/run_vllm_calibrate.py
#   * Loads the bf16 model with vLLM, runs forward passes on the PTQ dataset,
#     and dumps activation_stats.json / moe_expert_stats.json / kv_cache_*
#     into the directory given by ``output_dir`` in CALIB_CONFIG.
#
# Stage 2: tools/fp8_quant_with_vllm_activation.py
#   * Reads activation_stats.json (+ moe_expert_stats.json if any) plus the
#     original bf16 weights, applies per-tensor FP8 quantization with
#     calibrated input scales, and writes the FP8 HF model into the directory
#     given by ``output_fp8_hf_path`` in QUANT_CONFIG.
#
# IMPORTANT: ``input_vllm_ac_json_path`` in QUANT_CONFIG must equal
# ``output_dir`` in CALIB_CONFIG, otherwise stage 2 cannot find the stats.
#
# Usage:
#   bash run_vllm_quant_for_HY3.sh
#       (run both stages back-to-back)
#
#   bash run_vllm_quant_for_HY3.sh --skip-calibrate
#       (skip stage 1, only quantize using existing stats dir)
#
#   bash run_vllm_quant_for_HY3.sh --skip-quantize
#       (only run stage 1, do not produce the FP8 model)
# =============================================================================

# Strict-mode: stop on first error and propagate failures inside `cmd | tee`.
set -euo pipefail

# ----------------------------------------------------------------------------
# CLI flags
# ----------------------------------------------------------------------------
do_calibrate=1
do_quantize=1
for arg in "$@"; do
    case "${arg}" in
        --skip-calibrate) do_calibrate=0 ;;
        --skip-quantize)  do_quantize=0  ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: ${arg}" >&2
            echo "Use --help for usage." >&2
            exit 2
            ;;
    esac
done

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_MOE_COLLECT_STATS=1
export RAY_DEDUP_LOGS=0
export PYTHONDONTWRITEBYTECODE=1
export VLLM_MOE_COLLECT_STATS_VERBOSE=0
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1

export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

# ----------------------------------------------------------------------------
# YAML configs (one per stage)
# ----------------------------------------------------------------------------
CALIB_CONFIG=configs/HY3/ptq/HY3_vllm_calibrate.yaml
QUANT_CONFIG=configs/HY3/ptq/HY3_vllm_quant_fp8.yaml

mkdir -p logs

# ============================================================================
# Stage 1: activation calibration
# ============================================================================
if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[pipeline] === Stage 1/2: activation calibration ==="
    echo "[pipeline] CALIB_CONFIG=${CALIB_CONFIG}"

    python3 tools/run_vllm_calibrate.py \
        -c "${CALIB_CONFIG}" \
        2>&1 | tee "logs/run_vllm_quant_HY3-calibrate.log"

    echo "[pipeline] Stage 1 finished."
else
    echo "[pipeline] --skip-calibrate set, skipping stage 1."
fi

# ============================================================================
# Stage 2: FP8 quantization (uses calibration outputs)
# ============================================================================
if [[ "${do_quantize}" -eq 1 ]]; then
    echo "[pipeline] === Stage 2/2: FP8 quantization ==="
    echo "[pipeline] QUANT_CONFIG=${QUANT_CONFIG}"

    python3 tools/fp8_quant_with_vllm_activation.py \
        -c "${QUANT_CONFIG}" \
        2>&1 | tee "logs/run_vllm_quant_HY3-quantize.log"

    echo "[pipeline] Stage 2 finished."
else
    echo "[pipeline] --skip-quantize set, skipping stage 2."
fi

echo "[pipeline] All requested stages completed successfully."
