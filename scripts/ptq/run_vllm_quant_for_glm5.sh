#!/usr/bin/env bash
# =============================================================================
# One-click pipeline:  FP8 GLM-5 model  ->  vLLM activation calibration
#                                       ->  NVFP4 weight-only re-quantization
#                                          (dense layers kept as FP8 ue8m0)
#
# Stage 1: tools/run_vllm_calibrate.py
#   * Loads the FP8 GLM-5 checkpoint with vLLM, runs forward passes on the PTQ
#     dataset, and dumps activation_stats.json / moe_expert_stats.json
#     (and optionally mtp_moe_expert_stats.json) into the directory given by
#     ``output_dir`` in PTQ_CONFIG.
#
# Stage 2: tools/glm5_nvfp4_weight_only_blockwise.py
#   * Streams the source FP8 safetensors shard-by-shard, NVFP4-quantises every
#     MoE expert weight, re-quantises dense FP8 weights to ue8m0, and injects
#     the per-expert ``input_scale`` from moe_expert_stats.json produced by
#     stage 1.
#
# Both stages share a SINGLE unified YAML (PTQ_CONFIG); stage 2 reuses stage
# 1's ``model_path`` as ``input_path`` and ``${output_dir}/moe_expert_stats.json``
# as ``moe_stats_json``, while ``output_nvfp4_hf_path`` controls the
# destination. Paths only need to be set once.
#
# Usage:
#   bash run_vllm_quant_for_glm5.sh
#       (run both stages back-to-back)
#
#   bash run_vllm_quant_for_glm5.sh --skip-calibrate
#       (skip stage 1, only quantize using existing stats dir)
#
#   bash run_vllm_quant_for_glm5.sh --skip-quantize
#       (only run stage 1, do not produce the NVFP4 model)
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

# ----------------------------------------------------------------------------
# Runtime / vLLM environment (mirrors run_vllm_calibrate_for_glm5.sh)
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
# Unified YAML config (drives BOTH stages; each stage's argparse picks up
# only the keys it knows about, and unknown keys are warned-and-ignored).
# ----------------------------------------------------------------------------
PTQ_CONFIG=configs/glm5/ptq/fp8/glm5_vllm_ptq_moe.yaml

mkdir -p logs

# ============================================================================
# Stage 1: activation calibration
# ============================================================================
if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[pipeline] === Stage 1/2: activation calibration ==="
    echo "[pipeline] PTQ_CONFIG=${PTQ_CONFIG}"

    python3 tools/run_vllm_calibrate.py \
        -c "${PTQ_CONFIG}" \
        2>&1 | tee "logs/run_vllm_quant_glm5-calibrate.log"

    echo "[pipeline] Stage 1 finished."
else
    echo "[pipeline] --skip-calibrate set, skipping stage 1."
fi

# ============================================================================
# Stage 2: NVFP4 weight-only re-quantization (dense FP8 ue8m0)
# ============================================================================
if [[ "${do_quantize}" -eq 1 ]]; then
    echo "[pipeline] === Stage 2/2: NVFP4 weight-only quantization ==="
    echo "[pipeline] PTQ_CONFIG=${PTQ_CONFIG}"

    python3 tools/glm5_nvfp4_weight_only_blockwise.py \
        -c "${PTQ_CONFIG}" \
        2>&1 | tee "logs/run_vllm_quant_glm5-quantize.log"

    echo "[pipeline] Stage 2 finished."
else
    echo "[pipeline] --skip-quantize set, skipping stage 2."
fi

echo "[pipeline] All requested stages completed successfully."
