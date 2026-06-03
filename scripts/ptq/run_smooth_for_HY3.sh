#!/bin/bash
# =============================================================================
# run_smooth_for_HY3.sh — One-click smooth pipeline: calibration + conversion.
#
# Usage:
#   bash run_smooth_for_HY3.sh                    # both phases
#   bash run_smooth_for_HY3.sh --skip-calibrate   # Phase 2 only (reuse stats)
#   bash run_smooth_for_HY3.sh --skip-convert     # Phase 1 only
#   bash run_smooth_for_HY3.sh --help
#
# NOTE: Must be run from the AngelSlim repository root directory.
# =============================================================================

set -euo pipefail

CONFIG="configs/Hy3/ptq/fp8/Hy3_smooth.yaml"

SKIP_CALIBRATE=false
SKIP_CONVERT=false
for arg in "$@"; do
    case "$arg" in
        --skip-calibrate) SKIP_CALIBRATE=true ;;
        --skip-convert)   SKIP_CONVERT=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-calibrate] [--skip-convert] [--help]"
            echo "  --skip-calibrate  Skip Phase 1 (reuse existing smooth stats)"
            echo "  --skip-convert    Skip Phase 2 (calibrate only)"
            exit 0
            ;;
    esac
done

# -------- Environment Variables --------
# Allow function serialization for apply_model in vLLM v1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Enable MoE expert statistics collection
export VLLM_MOE_COLLECT_STATS=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1
# Disable verbose MoE stats logging

export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF


# -------- Environment Variables --------
export VLLM_MOE_COLLECT_SMOOTH_STATS=1
export VLLM_MOE_COLLECT_ALPHA_SEARCH=1

export PYTHONPATH=/cfs_cloud_code/gavinlee/work/open_source_smooth/AngelSlim
# -------- Phase 1: Collect Smooth Stats + Alpha Search --------
if [ "$SKIP_CALIBRATE" = false ]; then
    echo "========================================"
    echo "[Phase 1] Smooth Stats Calibration"
    echo "========================================"
    mkdir -p logs
    python3 tools/smooth/run_vllm_smooth.py -c "$CONFIG"
fi

# -------- Phase 2: Offline Weight Conversion --------
if [ "$SKIP_CONVERT" = false ]; then
    echo "========================================"
    echo "[Phase 2] Offline Weight Conversion"
    echo "========================================"
    python3 tools/smooth/convert_smooth_weights.py -c "$CONFIG"
fi

echo "========================================"
echo "Done."
echo "========================================"
