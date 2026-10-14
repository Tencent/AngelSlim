#!/usr/bin/env bash
# =============================================================================
# One-click RTN pipeline: BF16 Hy4 -> vLLM activation calibration -> mixed
# NVFP4 (routed experts, trunk + MTP) / FP8-128x128-blockwise (everything else)
#
#   Stage 1  tools/run_vllm_calibrate.py -> moe_expert_stats.json +
#            mtp_moe_expert_stats.json, written to ``output_dir``
#   Stage 2  tools/linear_fp8_moe_nvfp4_quant.py -> quantised checkpoint,
#            written to ``output_nvfp4_hf_path``
#
# Both stages read one YAML (PTQ_CONFIG), so paths are set once: stage 2 reuses
# stage 1's ``model_path`` and finds both stats files under ``output_dir``.
#
# Requires a 4-node * 8-GPU Ray cluster (``ray status`` -> Total GPUs = 32) and
# the HPC -- NOT the community -- vLLM patch on every node:
#   bash tools/vllm_patch_hpc/install.sh install
#
# Usage: pass --skip-calibrate or --skip-quantize to run a single stage.
#        MODEL_PATH / SRC_JSONL / OUTPUT_DIR override the matching YAML keys.
# =============================================================================

set -euo pipefail

do_calibrate=1
do_quantize=1
for arg in "$@"; do
    case "${arg}" in
        --skip-calibrate) do_calibrate=0 ;;
        --skip-quantize)  do_quantize=0  ;;
        -h|--help)
            sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: ${arg}" >&2
            echo "Use --help for usage." >&2
            exit 2
            ;;
    esac
done

export VLLM_MOE_COLLECT_STATS=1
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1
export VLLM_MOE_COLLECT_STATS_VERBOSE=0

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export RAY_DEDUP_LOGS=0
export PYTHONDONTWRITEBYTECODE=1

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

# HPC vLLM engine knobs, aligned with serving (/root/vllm.sh).
export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_ATTENTION_BACKEND=HPC_MLA        # Hy4 = MLA
export MOE_MODE=fused                        # HPC MoE mode
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1

# PTQ_CONFIG=configs/Hy4/ptq/nvfp4/Hy4_template.yaml
PTQ_CONFIG=configs/Hy4/ptq/nvfp4/Hy4_opd_v1.yaml

# CLI flags beat YAML in apply_yaml_config().
cli_overrides=()
[[ -n "${MODEL_PATH:-}" ]]  && cli_overrides+=(--model-path    "${MODEL_PATH}")
[[ -n "${SRC_JSONL:-}" ]]   && cli_overrides+=(--ptq-data-path "${SRC_JSONL}")
[[ -n "${OUTPUT_DIR:-}" ]]  && cli_overrides+=(--output-dir    "${OUTPUT_DIR}")

mkdir -p logs
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${do_calibrate}" -eq 1 ]]; then
    echo "[hy4-pipeline] === Stage 1/2: MoE activation calibration ==="
    echo "[hy4-pipeline] PTQ_CONFIG=${PTQ_CONFIG}"
    [[ ${#cli_overrides[@]} -gt 0 ]] && echo "[hy4-pipeline] CLI overrides: ${cli_overrides[*]}"
    echo "[hy4-pipeline] VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
    echo "[hy4-pipeline] Expecting a 4-node * 8-GPU Ray cluster (Total GPUs = 32)."

    python3 tools/run_vllm_calibrate.py \
        -c "${PTQ_CONFIG}" \
        "${cli_overrides[@]}" \
        2>&1 | tee "logs/run_hy4_nvfp4_quant-calibrate.log"

    echo "[hy4-pipeline] Stage 1 finished."
else
    echo "[hy4-pipeline] --skip-calibrate set, skipping stage 1."
fi

if [[ "${do_quantize}" -eq 1 ]]; then
    echo "[hy4-pipeline] === Stage 2/2: mixed NVFP4 + FP8-blockwise quantisation ==="
    echo "[hy4-pipeline] PTQ_CONFIG=${PTQ_CONFIG}"

    python3 tools/linear_fp8_moe_nvfp4_quant.py \
        -c "${PTQ_CONFIG}" \
        2>&1 | tee "logs/run_hy4_nvfp4_quant-quantize.log"

    echo "[hy4-pipeline] Stage 2 finished."
else
    echo "[hy4-pipeline] --skip-quantize set, skipping stage 2."
fi

echo "[hy4-pipeline] All requested stages completed successfully."
