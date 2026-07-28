#!/usr/bin/env bash
# Orchestrate the sparse prefill benchmark with safe handling of an optional GPU
# util-keepalive program.
#
# Some shared clusters auto-release cards left idle, so a keepalive process pins
# utilization to hold them. If one is configured (see the env vars below), it
# MUST be killed before measuring (it steals GPU cycles and wrecks latency
# numbers) and restarted after; this script guarantees restart even if the
# benchmark crashes or is interrupted, via a bash EXIT trap. If no keepalive is
# configured, these steps are skipped and the benchmark runs as-is.
#
# Single-node only.
#
# Usage:
#   bash evaluation/run_sparse_benchmark.sh [extra args passed to sparse_benchmark.py]
# e.g.
#   bash evaluation/run_sparse_benchmark.sh --seq-lens 4096,16384,32768,65536,131072

set -u

# Migration-proof roots: derive from this script's own location instead of
# hardcoding an absolute prefix. tools/ -> <repo>; <repo>/../.. -> home base.
# $ANGELSLIM_REPO / $ANGELSLIM_HOME override. Keepalive paths live on a separate
# mount and stay absolute.
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ANGELSLIM_REPO:-$(cd "$_SELF/.." && pwd)}"
_HOME_BASE="${ANGELSLIM_HOME:-$(cd "$REPO/../.." && pwd)}"
PY="$_HOME_BASE/miniconda3/envs/angelslim/bin/python"
MODEL="$_HOME_BASE/weights/Qwen3-8B"
# Optional GPU util-keepalive. Some shared clusters auto-release cards left idle;
# a keepalive process pins utilization to prevent that. It is entirely opt-in and
# cluster-specific: set ANGELSLIM_KEEPALIVE_PY (a torch-enabled interpreter) and
# ANGELSLIM_KEEPALIVE_PY_SCRIPT (the per-GPU keepalive worker) to enable it. If
# they are unset (the default), the keepalive step is skipped and the benchmark
# runs as-is — the keepalive is purely a cluster concern, not part of the
# benchmark itself.
KEEPALIVE_PY_SCRIPT="${ANGELSLIM_KEEPALIVE_PY_SCRIPT:-}"
# A torch-enabled interpreter — the keepalive worker imports torch, so a
# torch-less python3 would make every worker die with ModuleNotFoundError and
# leave the cards unprotected.
KEEPALIVE_PY="${ANGELSLIM_KEEPALIVE_PY:-}"
KEEPALIVE_NGPU="${ANGELSLIM_KEEPALIVE_NGPU:-8}"
OUT="${SPARSE_BENCH_OUT:-${REPO}/benchmark_results/sparse_bench_qwen3_8b.json}"

mkdir -p "$(dirname "$OUT")"

restart_keepalive() {
  # No-op unless a keepalive interpreter + worker script are both configured.
  if [ -z "$KEEPALIVE_PY" ] || [ -z "$KEEPALIVE_PY_SCRIPT" ]; then
    return
  fi
  local match
  match="$(basename "$KEEPALIVE_PY_SCRIPT")"
  echo "[orchestrator] restarting GPU util-keepalive to protect the cards…"
  # Only restart if it isn't already running, to avoid doubling processes.
  if pgrep -f "$match" >/dev/null 2>&1; then
    echo "[orchestrator] keepalive already running; not starting a second copy."
    return
  fi
  if [ ! -x "$KEEPALIVE_PY" ]; then
    echo "[orchestrator] ERROR: keepalive interpreter $KEEPALIVE_PY not found; skipping."
    return
  fi
  # One worker per GPU, arg "<gpu_id> 800".
  for g in $(seq 0 $((KEEPALIVE_NGPU - 1))); do
    nohup "$KEEPALIVE_PY" "$KEEPALIVE_PY_SCRIPT" "$g" 800 \
      > "/tmp/keepalive_${g}.log" 2>&1 &
  done
  sleep 6
  n=$(pgrep -f "$match" | grep -v pgrep | wc -l)
  echo "[orchestrator] keepalive restarted: $n process(es)."
  if [ "$n" -lt "$KEEPALIVE_NGPU" ]; then
    echo "[orchestrator] WARNING: expected $KEEPALIVE_NGPU keepalive workers, got $n."
    echo "[orchestrator] check /tmp/keepalive_*.log — cards may be UNPROTECTED."
  fi
}

# Guarantee the keepalive comes back no matter how we exit.
trap restart_keepalive EXIT INT TERM

if [ -n "$KEEPALIVE_PY_SCRIPT" ]; then
  _ka_match="$(basename "$KEEPALIVE_PY_SCRIPT")"
  echo "[orchestrator] killing GPU util-keepalive before measuring…"
  # Be surgical: only the keepalive, NOT every python3 (the benchmark is python).
  pkill -f "$_ka_match" 2>/dev/null
  sleep 3
  if pgrep -f "$_ka_match" >/dev/null 2>&1; then
    echo "[orchestrator] WARNING: keepalive still present after pkill; forcing -9"
    pkill -9 -f "$_ka_match" 2>/dev/null
    sleep 2
  fi
  echo "[orchestrator] keepalive down. GPU util should now be idle:"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
fi

echo "[orchestrator] launching benchmark…"
cd "$REPO" || exit 2
# expandable_segments avoids CUDA caching-allocator FRAGMENTATION at long context:
# a 36-layer 64K+ prefill interleaves huge activations with each algorithm's
# per-layer transient mask/workspace allocations; with the default allocator that
# fragments the pool and a later large request fails — which surfaced as an async
# illegal-memory-access mid-sweep for flashprefill at 32K/64K (NON-deterministic,
# depends on prior cells). expandable_segments fixes it with zero effect on
# numerics (it only changes how the allocator reserves/returns segments). A true
# capacity OOM (e.g. a variant that genuinely doesn't fit at 128K on one 96 GB
# H20) is still reported by the benchmark as a recorded problem, not a crash.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
"$PY" evaluation/sparse_benchmark.py \
  --model "$MODEL" \
  --out "$OUT" \
  "$@"
rc=$?
echo "[orchestrator] benchmark finished with rc=$rc"

# EXIT trap restarts the keepalive.
exit $rc
