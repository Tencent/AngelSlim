#!/usr/bin/env bash
# =============================================================================
# install.sh — One-click installer for AngelSlim's vLLM calibration patch.
#
# Usage:
#   bash install.sh              # install (default)
#   bash install.sh install      # install patch + utils
#   bash install.sh uninstall    # restore original vLLM files from .bak
#   bash install.sh check        # verify whether patch is currently active
#   bash install.sh -h | --help  # show this help
#
# Behavior:
#   * Auto-detects the installed vLLM directory via `python3 -c 'import vllm'`.
#   * Backs up original files to *.bak on first install (skipped if backup
#     already exists, so re-running install never destroys the pristine copy).
#   * Copies tools/vllm_patch/{envs.py,fused_moe.py} into the vLLM package and
#     places angelslim/.../vllm_calibrate_utils/ as a Python package at
#     <vllm>/tools/vllm_calibrate_utils/.
#   * Idempotent: re-running install simply refreshes the patched files.
#   * Verifies the patch is active by grepping for known markers.
#
# NOTE: On a multi-node Ray cluster you must run this script on EVERY node
#       that hosts a Ray worker (head + remote workers). It only patches the
#       vLLM install on the local machine.
# =============================================================================

set -euo pipefail

# ---------- locate paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANGELSLIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PATCH_ENVS_SRC="${SCRIPT_DIR}/envs.py"
PATCH_FUSED_MOE_SRC="${SCRIPT_DIR}/fused_moe.py"
# vllm_calibrate_utils is now a *package* (directory) under angelslim/.
CALIB_UTILS_SRC_DIR="${ANGELSLIM_ROOT}/angelslim/compressor/quant/core/vllm_calibrate_utils"

# ---------- pretty logging ----------
log()  { printf '\033[1;32m[install.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install.sh][WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[install.sh][ERR]\033[0m %s\n' "$*" >&2; }

usage() {
    sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

detect_vllm_dir() {
    python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null \
        || { err "Failed to import vllm. Is vLLM installed in this Python env?"; exit 1; }
}

# ---------- subcommands ----------
do_install() {
    local vllm_dir="$1"

    local envs_dst="${vllm_dir}/envs.py"
    local fused_moe_dst="${vllm_dir}/model_executor/layers/fused_moe/fused_moe.py"
    local utils_dst_dir="${vllm_dir}/tools"
    local utils_dst="${utils_dst_dir}/vllm_calibrate_utils"

    # ---- sanity-check sources ----
    for f in "${PATCH_ENVS_SRC}" "${PATCH_FUSED_MOE_SRC}"; do
        [[ -f "${f}" ]] || { err "Missing source file: ${f}"; exit 1; }
    done
    [[ -d "${CALIB_UTILS_SRC_DIR}" ]] || {
        err "Missing source package directory: ${CALIB_UTILS_SRC_DIR}"; exit 1;
    }
    [[ -f "${CALIB_UTILS_SRC_DIR}/__init__.py" ]] || {
        err "Source package is missing __init__.py: ${CALIB_UTILS_SRC_DIR}"; exit 1;
    }

    # ---- sanity-check targets ----
    for f in "${envs_dst}" "${fused_moe_dst}"; do
        [[ -f "${f}" ]] || { err "Expected vLLM file not found: ${f}"; exit 1; }
    done

    # ---- back up originals (only on first install) ----
    if [[ ! -f "${envs_dst}.bak" ]]; then
        cp -p "${envs_dst}" "${envs_dst}.bak"
        log "Backed up envs.py -> envs.py.bak"
    else
        warn "Backup already exists: ${envs_dst}.bak (kept untouched)"
    fi

    if [[ ! -f "${fused_moe_dst}.bak" ]]; then
        cp -p "${fused_moe_dst}" "${fused_moe_dst}.bak"
        log "Backed up fused_moe.py -> fused_moe.py.bak"
    else
        warn "Backup already exists: ${fused_moe_dst}.bak (kept untouched)"
    fi

    # ---- apply patches ----
    cp -p "${PATCH_ENVS_SRC}" "${envs_dst}"
    log "Patched ${envs_dst}"

    cp -p "${PATCH_FUSED_MOE_SRC}" "${fused_moe_dst}"
    log "Patched ${fused_moe_dst}"

    mkdir -p "${utils_dst_dir}"
    # Refresh the whole package directory each install run (idempotent).
    rm -rf "${utils_dst}"
    # Use a copy that follows symlinks and skips __pycache__ to avoid
    # carrying stale .pyc files from the source tree.
    cp -rpL "${CALIB_UTILS_SRC_DIR}" "${utils_dst}"
    rm -rf "${utils_dst}/__pycache__"
    log "Installed ${utils_dst}/ (package)"

    # ---- self-check ----
    do_check "${vllm_dir}"
}

do_uninstall() {
    local vllm_dir="$1"

    local envs_dst="${vllm_dir}/envs.py"
    local fused_moe_dst="${vllm_dir}/model_executor/layers/fused_moe/fused_moe.py"
    local utils_dst="${vllm_dir}/tools/vllm_calibrate_utils"
    # Legacy single-file install layout (pre-split).  Removed if it still
    # exists so the uninstall is exhaustive across both layouts.
    local utils_dst_legacy="${vllm_dir}/tools/vllm_calibrate_utils.py"

    if [[ -f "${envs_dst}.bak" ]]; then
        mv -f "${envs_dst}.bak" "${envs_dst}"
        log "Restored ${envs_dst}"
    else
        warn "No backup found for envs.py — leaving current file in place."
    fi

    if [[ -f "${fused_moe_dst}.bak" ]]; then
        mv -f "${fused_moe_dst}.bak" "${fused_moe_dst}"
        log "Restored ${fused_moe_dst}"
    else
        warn "No backup found for fused_moe.py — leaving current file in place."
    fi

    if [[ -d "${utils_dst}" ]]; then
        rm -rf "${utils_dst}"
        log "Removed ${utils_dst}/"
    fi
    if [[ -f "${utils_dst_legacy}" ]]; then
        rm -f "${utils_dst_legacy}"
        log "Removed legacy ${utils_dst_legacy}"
    fi

    log "Uninstall complete."
}

do_check() {
    local vllm_dir="$1"

    local envs_dst="${vllm_dir}/envs.py"
    local fused_moe_dst="${vllm_dir}/model_executor/layers/fused_moe/fused_moe.py"
    local utils_dst="${vllm_dir}/tools/vllm_calibrate_utils"

    local ok=1

    log "Verifying patch state at: ${vllm_dir}"

    if grep -q "VLLM_MOE_COLLECT_PER_EXPERT_STATS" "${envs_dst}" 2>/dev/null; then
        log "  [OK]   envs.py contains VLLM_MOE_COLLECT_PER_EXPERT_STATS"
    else
        err  "  [FAIL] envs.py does NOT contain VLLM_MOE_COLLECT_PER_EXPERT_STATS"
        ok=0
    fi

    if grep -q "collect_fused_moe_internal_stats" "${fused_moe_dst}" 2>/dev/null; then
        log "  [OK]   fused_moe.py contains collect_fused_moe_internal_stats"
    else
        err  "  [FAIL] fused_moe.py does NOT contain collect_fused_moe_internal_stats"
        ok=0
    fi

    if [[ -f "${utils_dst}/__init__.py" && -f "${utils_dst}/hooks.py" ]]; then
        log "  [OK]   ${utils_dst}/ package is present"
    else
        err  "  [FAIL] ${utils_dst}/ package is missing or incomplete"
        ok=0
    fi

    if [[ "${ok}" -eq 1 ]]; then
        log "Patch is ACTIVE."
        return 0
    else
        err "Patch is NOT fully active. Re-run \`bash install.sh install\`."
        return 1
    fi
}

# ---------- main ----------
ACTION="${1:-install}"

case "${ACTION}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    install|uninstall|check)
        VLLM_DIR="$(detect_vllm_dir)"
        log "Detected vLLM directory: ${VLLM_DIR}"
        log "AngelSlim repo root:     ${ANGELSLIM_ROOT}"
        case "${ACTION}" in
            install)   do_install   "${VLLM_DIR}" ;;
            uninstall) do_uninstall "${VLLM_DIR}" ;;
            check)     do_check     "${VLLM_DIR}" ;;
        esac
        ;;
    *)
        err "Unknown action: ${ACTION}"
        usage
        exit 2
        ;;
esac
