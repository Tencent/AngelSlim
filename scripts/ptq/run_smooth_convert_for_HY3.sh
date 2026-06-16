#!/bin/bash
# Phase 2 only: Offline Smooth Weight Conversion.
# Reads smooth_stats.json (and optionally smooth_alpha_search.json) from output_dir.
# Must be run from the AngelSlim repository root directory.

set -euo pipefail

CONFIG="configs/Hy3/ptq/fp8/Hy3_smooth.yaml"

python3 tools/smooth/convert_smooth_weights.py -c "$CONFIG" "$@"
