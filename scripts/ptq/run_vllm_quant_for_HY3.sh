#!/usr/bin/env bash
# =============================================================================
# One-click pipeline:  bf16 model  ->  vLLM activation calibration
#                                   ->  FP8 quantization (HF safetensors)
#
# Stage 1: tools/run_vllm_calibrate.py
#   * Loads the bf16 model with vLLM, runs forward passes on the PTQ dataset,
#     and dumps activation_stats.json / moe_expert_stats.json / kv_cache_*
#     into ${act_dir}.
#
# Stage 2: tools/fp8_quant_with_vllm_activation.py
#   * Reads ${act_dir}/activation_stats.json (+ moe_expert_stats.json if any)
#     plus the original bf16 weights, applies per-tensor FP8 quantization
#     with calibrated input scales, and writes the FP8 HF model into
#     ${fp8_path} (including kv_cache_scales.safetensors when per-head KV
#     stats are available).
#
# Usage:
#   bash run_vllm_calibrate_and_quantize_for_HY3_0_622post3.sh
#       (run both stages back-to-back)
#
#   bash run_vllm_calibrate_and_quantize_for_HY3_0_622post3.sh --skip-calibrate
#       (skip stage 1, only quantize using existing ${act_dir})
#
#   bash run_vllm_calibrate_and_quantize_for_HY3_0_622post3.sh --skip-quantize
#       (only run stage 1, do not produce the FP8 model)
#
# NOTE: must be invoked from the AngelSlim repository root, the same way as
# run_vllm_calibrate_for_HY3_0_622post3.sh, because the inner python commands
# use repo-relative paths (tools/...).
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
# Shared configuration (single source of truth for both stages)
# ----------------------------------------------------------------------------
run_name=log_name
model_path=/path/to/input/model
ptq_data_path=/path/to/dataset
act_dir=/path/to/statistics
fp8_path=/path/to/output/fp8_model

# Stage-1 calibration args
tp_size=16
batch_size=4
num_samples=512
max_length=16384

# Boolean flags (non-empty to enable, empty to disable)
skip_weight_loading=""  # set to "--skip-weight-loading" to enable debug mode
verbose=""              # set to "--verbose" to enable



# KV cache scale search settings
search_kv_scale="--search-kv-scale"    # set to "--search-kv-scale" to enable scale search
search_kv_num_samples=64               # number of samples used for the search
search_kv_min_multiplier=0.8           # lower bound of multiplier search range
search_kv_max_multiplier=16.0          # upper bound of multiplier search range
search_kv_num_steps=50                 # number of log-uniform grid points
kv_granularity="per-head"              # KV-cache granularity: none | per-tensor | per-head

mkdir -p logs

# ============================================================================
# Stage 1: activation calibration
# ============================================================================
if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[pipeline] === Stage 1/2: activation calibration ==="
    echo "[pipeline] model_path=${model_path}"
    echo "[pipeline] act_dir   =${act_dir}"

    python3 tools/run_vllm_calibrate.py \
        --model-path "${model_path}" \
        --ptq-data-path "${ptq_data_path}" \
        --output-dir "${act_dir}" \
        --tp-size "${tp_size}" \
        --batch-size "${batch_size}" \
        --num-samples "${num_samples}" \
        --max-length "${max_length}" \
        --kv-granularity "${kv_granularity}" \
        ${skip_weight_loading} \
        ${verbose} \
        ${search_kv_scale} \
        --search-kv-num-samples "${search_kv_num_samples}" \
        --search-kv-min-multiplier "${search_kv_min_multiplier}" \
        --search-kv-max-multiplier "${search_kv_max_multiplier}" \
        --search-kv-num-steps "${search_kv_num_steps}" \
        2>&1 | tee "logs/${run_name}.log"

    echo "[pipeline] Stage 1 finished. Activation stats saved under: ${act_dir}"
else
    echo "[pipeline] --skip-calibrate set, skipping stage 1."
fi

# ============================================================================
# Stage 2: FP8 quantization (uses calibration outputs in ${act_dir})
# ============================================================================
if [[ "${do_quantize}" -eq 1 ]]; then
    echo "[pipeline] === Stage 2/2: FP8 quantization ==="
    echo "[pipeline] input bf16 model = ${model_path}"
    echo "[pipeline] input act stats  = ${act_dir}"
    echo "[pipeline] output FP8 model = ${fp8_path}"

    if [[ ! -f "${act_dir}/activation_stats.json" ]]; then
        echo "[pipeline][ERROR] ${act_dir}/activation_stats.json is missing." >&2
        echo "[pipeline][ERROR] Run stage 1 first (drop --skip-calibrate)." >&2
        exit 1
    fi

    python3 tools/fp8_quant_with_vllm_activation.py \
        --input_bf16_hf_path "${model_path}" \
        --input_vllm_ac_json_path "${act_dir}" \
        --output_fp8_hf_path "${fp8_path}" \
        2>&1 | tee "logs/${run_name}-quantize.log"

    echo "[pipeline] Stage 2 finished. FP8 model saved to: ${fp8_path}"
else
    echo "[pipeline] --skip-quantize set, skipping stage 2."
fi

echo "[pipeline] All requested stages completed successfully."
