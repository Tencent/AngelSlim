#!/usr/bin/env bash
# Official LongBench (v1) accuracy run via the NATIVE lm-evaluation-harness,
# fanned out across the visible GPUs: `dense` + EVERY registered sparse algorithm
# (the variant set is DERIVED from the registry, not hardcoded), one variant per
# process pinned to its own GPU, then merge the per-variant JSONs.
#
# This drives the lm-eval `longbench` task group (21 tasks, EN+ZH). Each variant
# runs in its own process (fresh model load + own GPU), so variants share NO
# mutable state — a variant's score is identical whether run in parallel or alone
# (accuracy is GPU-utilization-independent: a busy neighbour changes timing, not
# logits). Independence also avoids any patch/unpatch residue across variants.
#
# GPL DEPENDENCY (eval-only, NOT in requirements): lm-eval's LongBench metrics
# hard-import jieba + fuzzywuzzy (both GPL). Install once before running:
#     pip install jieba fuzzywuzzy python-Levenshtein
#
# DATASET: lm-eval pulls Xnhyacinth/LongBench (HF mirror of THUDM/LongBench).
# Pre-cache it once; behind a proxy export https_proxy/http_proxy first.
#
# NOTE: lm-eval LEFT-truncates to --max-len (keep tail); these numbers match the
# lm-eval LongBench leaderboard.
#
# Env overrides: LBLM_VARIANTS=a,b,c, LBLM_NGPU=N, LBLM_TASKS=t1,t2, LBLM_MAXLEN.
#
# Usage:
#   bash evaluation/run_longbench_lmeval_parallel.sh              # full longbench, all variants
#   bash evaluation/run_longbench_lmeval_parallel.sh --limit 20   # smoke, fewer items/task

set -u

_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ANGELSLIM_REPO:-$(cd "$_SELF/.." && pwd)}"
_HOME_BASE="${ANGELSLIM_HOME:-$(cd "$REPO/../.." && pwd)}"
PY="$_HOME_BASE/miniconda3/envs/angelslim/bin/python"
MODEL="${LBLM_MODEL:-$_HOME_BASE/weights/Qwen3-8B}"
OUT="${LBLM_OUT:-${REPO}/benchmark_results/longbench_v1_lmeval_qwen3_8b.json}"
MAXLEN="${LBLM_MAXLEN:-32768}"
WORKDIR="${LBLM_WORKDIR:-/tmp/lblm_parallel}"

# GPL eval deps must be importable or every variant dies in metrics.py import.
if ! "$PY" -c "import jieba, fuzzywuzzy" 2>/dev/null; then
  echo "[lblm-par] ERROR: lm-eval LongBench needs jieba + fuzzywuzzy (GPL, eval-only)."
  echo "           Install them first: pip install jieba fuzzywuzzy python-Levenshtein"
  exit 2
fi

# Proxy passthrough for the first-run dataset fetch (no-op once cached). Set
# LBLM_PROXY=http://host:port to enable; harmless when the cache is warm.
if [ -n "${LBLM_PROXY:-}" ]; then
  export https_proxy="$LBLM_PROXY" http_proxy="$LBLM_PROXY"
fi

# Variant set DERIVED from the registry — dense first, then every registered
# algorithm. LBLM_VARIANTS overrides for a subset run.
if [ -n "${LBLM_VARIANTS:-}" ]; then
  IFS=',' read -r -a VARIANTS <<< "$LBLM_VARIANTS"
else
  _algos="$("$PY" - <<'PYEOF'
import angelslim.compressor.sparsity.algorithms  # noqa: F401  (register)
from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry
print(" ".join(sorted(SparsityAlgorithmRegistry.available())))
PYEOF
)"
  if [ -z "$_algos" ]; then
    echo "[lblm-par] ERROR: could not enumerate the sparse-algorithm registry"
    exit 2
  fi
  VARIANTS=(dense $_algos)
fi

# Round-robin over the visible GPUs (surplus variants share a card — two 32K
# jobs coexist on one 96 GB H20; OOM is reported as a problem, not a wrong score).
NGPU="${LBLM_NGPU:-$("$PY" -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 8)}"
[ "${NGPU:-0}" -ge 1 ] || NGPU=1

# Optional task subset passed through to every variant.
TASKS_ARG=()
[ -n "${LBLM_TASKS:-}" ] && TASKS_ARG=(--tasks "$LBLM_TASKS")

mkdir -p "$WORKDIR" "$(dirname "$OUT")"
# Clean stale per-variant JSON/logs: a hard-killed variant must not leave a
# previous-run JSON for the merge to fold in as fresh (the merge also re-checks
# git-sha agreement as a second line of defence).
rm -f "$WORKDIR"/*.json "$WORKDIR"/*.log 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[lblm-par] model=$MODEL maxlen=$MAXLEN ngpu=$NGPU variants=${VARIANTS[*]}"
echo "[lblm-par] launching ${#VARIANTS[@]} variants over $NGPU GPU(s) (round-robin)…"

pids=()
g=0
for v in "${VARIANTS[@]}"; do
  gpu=$((g % NGPU))
  log="$WORKDIR/${v}.log"
  vout="$WORKDIR/${v}.json"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" evaluation/longbench_lmeval.py \
    --model "$MODEL" --max-len "$MAXLEN" \
    --variants "$v" --allow-pseudo-sparse --out "$vout" --write-incomplete \
    "${TASKS_ARG[@]}" "$@" > "$log" 2>&1 &
  pids+=($!)
  echo "[lblm-par]   GPU $gpu <- $v  (pid ${pids[-1]}, log $log)"
  g=$((g + 1))
done

echo "[lblm-par] waiting for ${#pids[@]} variant jobs…"
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[lblm-par] WARNING: variant '${VARIANTS[$i]}' exited non-zero"
    fail=1
  fi
done

echo "[lblm-par] merging per-variant JSONs -> $OUT"
"$PY" evaluation/longbench_lmeval_merge.py --workdir "$WORKDIR" --out "$OUT" \
  --variants "$(IFS=,; echo "${VARIANTS[*]}")"
rc=$?
echo "[lblm-par] done (merge rc=$rc, any_variant_fail=$fail)"
exit $((rc | fail))
