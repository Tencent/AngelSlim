#!/usr/bin/env bash
# LongBench (v1) accuracy via the NATIVE lm-evaluation-harness — single-GPU,
# all variants sequentially in one process. The 1-GPU sibling of
# run_longbench_lmeval_parallel.sh (same numbers, slower).
#
# GPL DEPENDENCY (eval-only, NOT in requirements): lm-eval's LongBench metrics
# hard-import jieba + fuzzywuzzy (both GPL). Install once before running:
#     pip install jieba fuzzywuzzy python-Levenshtein
#
# DATASET: lm-eval pulls Xnhyacinth/LongBench (HF mirror of THUDM/LongBench).
# Behind a proxy, set LBLM_PROXY=http://host:port for the first-run fetch.
#
# NOTE: lm-eval LEFT-truncates to --max-len (keep tail); numbers match the
# lm-eval LongBench leaderboard.
#
# Usage:
#   bash evaluation/run_longbench_lmeval.sh [extra args to longbench_lmeval.py]
#   bash evaluation/run_longbench_lmeval.sh --variants dense,minference --limit 4

set -u

_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ANGELSLIM_REPO:-$(cd "$_SELF/.." && pwd)}"
_HOME_BASE="${ANGELSLIM_HOME:-$(cd "$REPO/../.." && pwd)}"
PY="$_HOME_BASE/miniconda3/envs/angelslim/bin/python"
MODEL="${LBLM_MODEL:-$_HOME_BASE/weights/Qwen3-8B}"
OUT="${LBLM_OUT:-${REPO}/benchmark_results/longbench_v1_lmeval_qwen3_8b.json}"

if ! "$PY" -c "import jieba, fuzzywuzzy" 2>/dev/null; then
  echo "[lblm] ERROR: lm-eval LongBench needs jieba + fuzzywuzzy (GPL, eval-only)."
  echo "       Install them first: pip install jieba fuzzywuzzy python-Levenshtein"
  exit 2
fi

if [ -n "${LBLM_PROXY:-}" ]; then
  export https_proxy="$LBLM_PROXY" http_proxy="$LBLM_PROXY"
fi

mkdir -p "$(dirname "$OUT")"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$REPO" || exit 2
echo "[lblm] launching lm-eval LongBench…"
"$PY" evaluation/longbench_lmeval.py \
  --model "$MODEL" \
  --out "$OUT" \
  "$@"
rc=$?
echo "[lblm] finished with rc=$rc"
exit $rc
