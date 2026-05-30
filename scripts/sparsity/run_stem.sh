#!/usr/bin/env bash
# Stem sparse-attention launch script.
#
# Usage:
#   bash scripts/sparsity/run_stem.sh /path/to/Qwen3-8B prompt.txt stem
#   bash scripts/sparsity/run_stem.sh /path/to/Qwen3-8B prompt.txt dense
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:-Qwen/Qwen3-8B}"
PROMPT_FILE="${2:?Usage: $0 <model-path> <prompt-file> [stem|dense] [extra args...]}"
MODE="${3:-stem}"
shift 3 2>/dev/null || true

cd "$ROOT_DIR"
exec python -u tools/run_stem.py \
  --model-path "$MODEL_PATH" \
  --model-name "Qwen/Qwen3-8B" \
  --prompt-file "$PROMPT_FILE" \
  --mode "$MODE" \
  "$@"
