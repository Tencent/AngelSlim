#!/usr/bin/env bash
# =============================================================================
# One-click pipeline:  bf16 model  ->  vLLM activation calibration
#                                   ->  FP8 quantization (HF safetensors)
#
# Stage 1: tools/run_vllm_calibrate.py
#   * Loads the bf16 model with vLLM, runs forward passes on the PTQ dataset,
#     and dumps activation_stats.json / moe_expert_stats.json / kv_cache_*
#     into the directory given by ``output_dir`` in PTQ_CONFIG.
#
# Stage 2: tools/fp8_quant_with_vllm_activation.py
#   * Reads activation_stats.json (+ moe_expert_stats.json if any) plus the
#     original bf16 weights, applies per-tensor FP8 quantization with
#     calibrated input scales, and writes the FP8 HF model into the directory
#     given by ``output_fp8_hf_path`` in PTQ_CONFIG.
#
# Both stages share a SINGLE unified YAML (PTQ_CONFIG); stage 2 reuses stage
# 1's ``model_path`` as ``input_bf16_hf_path`` and ``output_dir`` as
# ``input_vllm_ac_json_path``, so paths only need to be set once.
#
# Usage:
#   bash run_vllm_quant_for_Hy3.sh
#       (run both stages back-to-back)
#
#   bash run_vllm_quant_for_Hy3.sh --skip-calibrate
#       (skip stage 1, only quantize using existing stats dir)
#
#   bash run_vllm_quant_for_Hy3.sh --skip-quantize
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
            sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
# Unified YAML config (drives BOTH stages; each stage's argparse picks up
# only the keys it knows about, and unknown keys are warned-and-ignored).
# ----------------------------------------------------------------------------
PTQ_CONFIG=configs/Hy3/ptq/fp8/Hy3_vllm_ptq_per_tensor.yaml

mkdir -p logs

# ============================================================================
# Stage 1: activation calibration
# ============================================================================
if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[pipeline] === Stage 1/2: activation calibration ==="
    echo "[pipeline] PTQ_CONFIG=${PTQ_CONFIG}"

    python3 tools/run_vllm_calibrate.py \
        -c "${PTQ_CONFIG}" \
        2>&1 | tee "logs/run_vllm_quant_Hy3-calibrate.log"

    echo "[pipeline] Stage 1 finished."
else
    echo "[pipeline] --skip-calibrate set, skipping stage 1."
fi

# ============================================================================
# Stage 2: FP8 quantization (uses calibration outputs)
# ============================================================================
if [[ "${do_quantize}" -eq 1 ]]; then
    echo "[pipeline] === Stage 2/2: FP8 quantization ==="
    echo "[pipeline] PTQ_CONFIG=${PTQ_CONFIG}"

    python3 tools/fp8_quant_with_vllm_activation.py \
        -c "${PTQ_CONFIG}" \
        2>&1 | tee "logs/run_vllm_quant_Hy3-quantize.log"

    echo "[pipeline] Stage 2 finished."
else
    echo "[pipeline] --skip-quantize set, skipping stage 2."
fi

echo "[pipeline] All requested stages completed successfully."
