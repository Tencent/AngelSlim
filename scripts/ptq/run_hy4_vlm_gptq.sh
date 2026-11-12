#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
log() { echo "[hy4-vlm-gptq] $*"; }
die() { echo "[hy4-vlm-gptq] ERROR: $*" >&2; exit 1; }

PTQ_CONFIG="${PTQ_CONFIG:-configs/Hy4/ptq/mixed/Hy4_fp8_nvfp4_gptq.yaml}"
[[ -f "${PTQ_CONFIG}" ]] || die "PTQ_CONFIG not found: ${PTQ_CONFIG}"
LOG_PREFIX="logs/hy4_vlm_gptq.$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

export VLLM_ALLOW_INSECURE_SERIALIZATION=1 RAY_DEDUP_LOGS=0 PYTHONDONTWRITEBYTECODE=1
export NCCL_RUNTIME_CONNECT=0 VLLM_ENABLE_CHUNKED_PREFILL=1 MOE_MODE=fused
export ASYNC_SCHEDULING=1 VLLM_ENABLE_PREFIX_CACHING=1
# vllm_calib.calibrate (RTN path) reads these in every worker; the GPTQ collector does not.
export VLLM_MOE_COLLECT_STATS=1 VLLM_MOE_COLLECT_PER_EXPERT_STATS=1
export VLLM_MOE_COLLECT_STATS_VERBOSE="${VLLM_MOE_COLLECT_STATS_VERBOSE:-0}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
# PRECISIONMODE=HF nearly doubles per-rank weights and leaves no room for H.
if [[ "${PRECISION_MODE:-default}" == "default" ]]; then
    unset PRECISIONMODE VLLM_PRECISION_MODE
else
    export PRECISIONMODE="${PRECISION_MODE}"
fi
ulimit -n 1000000 2>/dev/null || true

# offline ViT embeddings (single GPU); image_embeds_dir defaults to <save_path>/image_embeds
read -r embeds_dir is_vlm < <(python3 -c "
import json, os, sys, yaml
from angelslim.utils.config_parser import default_run_dirs
c = yaml.safe_load(open(sys.argv[1])) or {}
calib = ((c.get('compression') or {}).get('calibrate') or {})
save_path = (c.get('global') or {}).get('save_path')
embeds = (calib.get('args') or {}).get('image_embeds_dir')
if not embeds and save_path:
    embeds = default_run_dirs(save_path)['image_embeds_dir']
mp = (c.get('model') or {}).get('model_path') or ''
hf = os.path.join(mp, 'config.json')
vlm = bool(json.load(open(hf)).get('vision_config')) if os.path.isfile(hf) else False
print(embeds or '-', vlm)" "${PTQ_CONFIG}")
if [[ "${is_vlm}" == True && "${embeds_dir}" == "-" ]]; then
    log "WARNING: VL model without calibrate.args.image_embeds_dir."
    log "  The live vision tower then runs on every TP rank on top of the weights AND the"
    log "  H accumulators, which OOMs at TP=32. Set image_embeds_dir to precompute it."
fi
if [[ "${SKIP_EMBEDS:-0}" != 1 && "${is_vlm}" == True && "${embeds_dir}" != "-" ]]; then
    log "image embeds -> ${embeds_dir}"
    python3 -m angelslim.compressor.quant.vllm_calib.image_embeds -c "${PTQ_CONFIG}" 2>&1 | tee "${LOG_PREFIX}.embeds.log"
fi

# Hessians and/or MoE absmax (needs the Ray cluster) 
if [[ "${SKIP_CALIB:-0}" != 1 ]]; then
    python3 tools/run.py -c "${PTQ_CONFIG}" --stage calib 2>&1 | tee "${LOG_PREFIX}.calib.log"
fi

# quantization: node r converts shards[r::N] 
if [[ "${SKIP_QUANT:-0}" == 1 ]]; then
    log "SKIP_QUANT=1, stopping after calibration"
    exit 0
fi
mapfile -t HOSTS < <(echo "${NODE_IP_LIST:-}" | sed 's/:[0-9]*//g; s/,/\n/g' | grep -E '^[0-9]+(\.[0-9]+){3}$' || true)
NUM_NODES="${NUM_NODES:-${#HOSTS[@]}}"
quant_args=(--stage quant)
[[ "${SKIP_EXISTING:-0}" == 1 ]] && quant_args+=(--skip-existing)
if [[ "${NUM_NODES}" -le 1 ]]; then
    python3 tools/run.py -c "${PTQ_CONFIG}" "${quant_args[@]}" 2>&1 | tee "${LOG_PREFIX}.quant.log"
else
    [[ "${NUM_NODES}" -le ${#HOSTS[@]} ]] || die "NUM_NODES=${NUM_NODES} > ${#HOSTS[@]} hosts"
    HOSTS=("${HOSTS[@]:0:${NUM_NODES}}")
    SSH_OPTS=(-n -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=60)
    for h in "${HOSTS[@]}"; do
        timeout 60 ssh "${SSH_OPTS[@]}" "${h}" "test -f '${REPO}/${PTQ_CONFIG}' || test -f '${PTQ_CONFIG}'" \
            || die "${h}: cannot see ${PTQ_CONFIG} under ${REPO}"
    done
    # ssh without -t does not forward signals: kill the remote ranks on interrupt.
    KILL_PAT="[t]ools/run.py -c ${PTQ_CONFIG} --stage quant"
    kill_remote() {
        for h in "${HOSTS[@]}"; do
            timeout 30 ssh "${SSH_OPTS[@]}" "${h}" "pkill -9 -f '${KILL_PAT}'; true" 2>/dev/null || true
        done
        exit 130
    }
    trap kill_remote INT TERM
    pids=()
    for r in $(seq 0 $((NUM_NODES - 1))); do
        ssh "${SSH_OPTS[@]}" "${HOSTS[$r]}" "cd $(printf '%q' "${REPO}") && \
            PYTHONPATH=$(printf '%q' "${REPO}") exec python3 -u tools/run.py -c $(printf '%q' "${PTQ_CONFIG}") \
            ${quant_args[*]} --node-rank ${r} --num-nodes ${NUM_NODES}" > "${LOG_PREFIX}.rank${r}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for r in "${!pids[@]}"; do
        wait "${pids[$r]}" || { fail=1; log "rank ${r} (${HOSTS[$r]}) failed, see ${LOG_PREFIX}.rank${r}.log"; }
    done
    trap - INT TERM
    [[ ${fail} -eq 0 ]] || die "rerun with SKIP_EXISTING=1 to redo only the missing shards"
fi

# finalize: index + compressed-tensors config.json
python3 tools/run.py -c "${PTQ_CONFIG}" --stage finalize 2>&1 | tee "${LOG_PREFIX}.finalize.log"
log "done (logs ${LOG_PREFIX}.*)"
