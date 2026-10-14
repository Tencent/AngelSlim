#!/usr/bin/env bash
# =============================================================================
# install.sh — Surgical AngelSlim calibration-hook installer for the internal
#              HPC fork of vLLM (e.g. 0.20.1.dev4xx+g...).
#
# WHY A SEPARATE INSTALLER?
#   The default `tools/vllm_patch/install.sh` was written against community
#   vLLM v0.20 and works by **whole-file** replacement of
#     * <vllm>/envs.py
#     * <vllm>/model_executor/layers/fused_moe/fused_moe.py
#   On the internal HPC fork this is destructive:
#     * The HPC fork's envs.py declares 40+ HPC-only env vars
#       (VLLM_USE_HPC_FUSEMOE, VLLM_MLA_FP8_DYNAMIC_QUANT,
#        VLLM_ENABLE_HPC_FUSED_RMSNORM, VLLM_ENABLE_TPSP, ...);
#       overwriting it kills every downstream HPC feature (Hy4 serving fails
#       to import).
#     * The HPC fork moved `class TritonExperts` out of fused_moe.py into
#       fused_moe/experts/triton_moe.py; whole-file replacement of
#       fused_moe.py drops HPC's swap_ab / SM90 optimisations *and* silently
#       does nothing for the hook (the wrong file is being patched).
#
# WHAT THIS INSTALLER DOES INSTEAD
#   1. Detects the vLLM install directory.
#   2. Locates `class TritonExperts` -- which now lives in
#        <vllm>/model_executor/layers/fused_moe/experts/triton_moe.py
#      -- and does an idempotent, marker-guarded in-place edit that inserts
#      TWO one-liner hook calls into `TritonExperts.apply()`:
#        * before the first  invoke_fused_moe_triton_kernel(...) call
#          -> collect_fused_moe_internal_stats(stage="gate_up_proj", ...)
#        * before the second invoke_fused_moe_triton_kernel(...) call
#          -> collect_fused_moe_internal_stats(stage="down_proj",  ...)
#      Both calls read `_vllm_layer_name` / `_moe_activation_stats_of_model`
#      from `w1` (which is `layer.w13_weight`, and gets those attrs attached
#      by AngelSlim's hooks.py before calibration starts).  The down_proj
#      site additionally fans out to `collect_fused_moe_channel_stats`
#      (per-channel absmax for NVFP4 channel reordering) when
#      `VLLM_MOE_COLLECT_CHANNEL_STATS=1` and hooks.py wired
#      `_moe_channel_stats_of_model`; otherwise it is a no-op.
#   3. Installs `vllm_calibrate_utils/` as a Python package under
#        <vllm>/tools/vllm_calibrate_utils/
#      just like the community installer, so the injected hook can `import
#      collect_fused_moe_internal_stats` at runtime.
#   4. Backs up the *unpatched* triton_moe.py to triton_moe.py.bak on the
#      first install so `uninstall` can restore the exact original file.
#
# WHAT THIS INSTALLER DELIBERATELY DOES **NOT** DO
#   * Does NOT touch <vllm>/envs.py.
#   * Does NOT touch <vllm>/model_executor/layers/fused_moe/fused_moe.py.
#   * Does NOT install smooth_moe_inject.py (Hy4 pipeline does not need
#     smooth stats / alpha search).
#   * Does NOT depend on any AngelSlim-declared env var in envs.py — the
#     hook itself gates on `os.getenv("VLLM_MOE_COLLECT_STATS")`, which is
#     read at runtime and does not require an envs.py declaration.
#
# Usage:
#   bash install.sh              # install (default)
#   bash install.sh install      # install patch + utils
#   bash install.sh uninstall    # restore triton_moe.py from .bak
#   bash install.sh check        # verify whether patch is currently active
#   bash install.sh -h | --help  # show this help
#
# NOTE: This installer must be re-run on EVERY node that hosts a Ray worker
#       (head + all remote workers). It only patches the vLLM installed on
#       the current machine.
# =============================================================================

set -euo pipefail

# ---------- locate paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANGELSLIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# vllm_calibrate_utils lives as a package under angelslim/.
CALIB_UTILS_SRC_DIR="${ANGELSLIM_ROOT}/angelslim/compressor/quant/core/vllm_calibrate_utils"

# Unique markers used to detect / remove the injected hook lines.  Keep
# these EXACTLY in sync with the block emitted by `inject_hook_into()`
# below -- `do_check` and `do_uninstall` grep for them.
BEGIN_MARK="# === AngelSlim MoE calibration hook (BEGIN) ==="
END_MARK="# === AngelSlim MoE calibration hook (END) ==="

# ---------- pretty logging ----------
log()  { printf '\033[1;32m[install-hpc.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install-hpc.sh][WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[install-hpc.sh][ERR]\033[0m %s\n' "$*" >&2; }

usage() {
    sed -n '2,64p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

detect_vllm_dir() {
    python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null \
        || { err "Failed to import vllm. Is vLLM installed in this Python env?"; exit 1; }
}

# ---------- surgical in-place edit of triton_moe.py ----------
#
# We rely on a small inline Python program because the injection points span
# multiple lines and depend on symbolic context (`class TritonExperts:` +
# `def apply(` + the two `invoke_fused_moe_triton_kernel(` calls).  Doing this
# with a sed one-liner is fragile across vLLM point releases; the Python
# variant validates the context before writing anything.
#
# Injection strategy:
#   * The FIRST `invoke_fused_moe_triton_kernel(` call inside `TritonExperts.
#     apply()` consumes the raw gate_up_proj activation -> tag as
#     "gate_up_proj".
#   * The `self.activation(activation, intermediate_cache2, ...)` line marks
#     the boundary between the two GEMMs.
#   * The `qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(...)`
#     call is the down_proj input quantiser.  We insert the down_proj hook
#     right before that quantiser so we capture the *pre-quant* BF16
#     activation (which is what `hooks.py` expects).  This matters because
#     for BF16 (Hy4) `moe_kernel_quantize_input` is a no-op, but keeping the
#     hook upstream of it makes the injection symmetric with the FP8 flow.
#
# The hook payload is a single call that pulls the layer-name / stats-dict
# from `w1` and no-ops when `VLLM_MOE_COLLECT_STATS != "1"`, so it is safe to
# leave installed permanently: normal (non-calibration) serving pays only
# the cost of two attribute lookups + one `os.getenv` per MoE forward.
inject_hook_into() {
    local target_file="$1"
    local begin_mark="$2"
    local end_mark="$3"
    local calib_utils_dst="$4"

    BEGIN_MARK="${begin_mark}" \
    END_MARK="${end_mark}" \
    CALIB_UTILS_DST="${calib_utils_dst}" \
    TARGET_FILE="${target_file}" \
    python3 - <<'PY'
import os
import re
import sys

path = os.environ["TARGET_FILE"]
begin_mark = os.environ["BEGIN_MARK"]
end_mark = os.environ["END_MARK"]
calib_utils_dst = os.environ["CALIB_UTILS_DST"]

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# ---------------------------------------------------------------- idempotency
if begin_mark in src and end_mark in src:
    print(f"[inject] triton_moe.py already patched (markers present); leaving untouched.")
    sys.exit(0)

# ------------------------------------------------------------ sanity-check
# We look for a specific structural anchor: `class TritonExperts` followed
# somewhere by a `def apply(` and then two `invoke_fused_moe_triton_kernel(`
# calls plus a `moe_kernel_quantize_input(` call.  If any of these is
# missing, the file layout is not what we expect and we bail out
# LOUDLY -- overwriting silently would be worse than failing.
class_re = re.compile(r"^class\s+TritonExperts\b", re.M)
if not class_re.search(src):
    print(f"[inject][ERR] `class TritonExperts` not found in {path}", file=sys.stderr)
    sys.exit(2)

# Split the file into three chunks:
#   [0]  everything up to (and including) the "def apply(" line of TritonExperts
#   [1]  the body of `apply` up to the FIRST `invoke_fused_moe_triton_kernel(`
#   [2]  everything after
# We only ever edit within TritonExperts.apply, and both injection anchors
# are inside `apply` so a naive `str.replace` on the first-match anchors is
# safe here.
FIRST_KERNEL_ANCHOR = "invoke_fused_moe_triton_kernel(\n            hidden_states,"
QUANT_ANCHOR = "qintermediate_cache2, a2q_scale = moe_kernel_quantize_input("

if FIRST_KERNEL_ANCHOR not in src:
    # Fall back to a looser regex in case the codebase reformats to a single
    # line: `invoke_fused_moe_triton_kernel(hidden_states,`.
    m = re.search(r"invoke_fused_moe_triton_kernel\(\s*\n?\s*hidden_states\s*,", src)
    if not m:
        print(f"[inject][ERR] cannot locate gate_up_proj kernel invocation anchor in {path}",
              file=sys.stderr)
        sys.exit(2)
    # Rewrite the anchor to whatever actually appears in the source so the
    # replace() below succeeds.
    FIRST_KERNEL_ANCHOR = m.group(0)

if QUANT_ANCHOR not in src:
    print(f"[inject][ERR] cannot locate down_proj quantise-input anchor in {path}", file=sys.stderr)
    sys.exit(2)

# ------------------------------------------------------------- build blocks
# The injected snippet must:
#   * bootstrap `collect_fused_moe_internal_stats` from
#     <vllm>/tools/vllm_calibrate_utils/  (added to sys.path once at import
#     time via the file-level bootstrap below), and
#   * be a no-op if the utility isn't installed (so serving with the patch
#     applied but without vllm_calibrate_utils never crashes).
# We stash the boilerplate at the very top of the file (marker-guarded) and
# only insert one-liners at the two call sites.

FILE_LEVEL_BOOTSTRAP = f'''{begin_mark}
# The block below is inserted by tools/vllm_patch_hpc/install.sh.
# It wires AngelSlim's `collect_fused_moe_internal_stats` into
# `TritonExperts.apply()` so a running vLLM engine can produce
# per-layer MoE input-activation `scale_amax` statistics for
# downstream NVFP4 global-scale calibration.  When
# `VLLM_MOE_COLLECT_STATS != "1"` the hook returns immediately and
# adds ~microseconds of overhead per forward.
import os as _angelslim_os
import sys as _angelslim_sys

_ANGELSLIM_CALIB_UTILS_DIR = {calib_utils_dst!r}
if _ANGELSLIM_CALIB_UTILS_DIR and _ANGELSLIM_CALIB_UTILS_DIR not in _angelslim_sys.path:
    _angelslim_sys.path.insert(0, _ANGELSLIM_CALIB_UTILS_DIR)

try:
    from vllm_calibrate_utils import collect_fused_moe_internal_stats  # noqa: E402
except Exception as _e:  # pragma: no cover — fall back to no-op
    def collect_fused_moe_internal_stats(*_a, **_kw):
        return None

# Per-channel down_proj-input absmax collector (feeds NVFP4 channel
# reordering).  Imported separately so a missing symbol degrades to a no-op
# instead of breaking the scalar-stats path above.
try:
    from vllm_calibrate_utils import collect_fused_moe_channel_stats  # noqa: E402
except Exception as _e:  # pragma: no cover — fall back to no-op
    def collect_fused_moe_channel_stats(*_a, **_kw):
        return None

# Raw-token sampler (real signed down_proj input tokens). Separate import so a
# missing symbol degrades to no-op.
try:
    from vllm_calibrate_utils import collect_fused_moe_rawtok  # noqa: E402
except Exception as _e:  # pragma: no cover
    def collect_fused_moe_rawtok(*_a, **_kw):
        return None

_ANGELSLIM_WARNED = set()


def _angelslim_moe_stats_call(stage, hidden_states, topk_ids, global_num_experts, w1):
    """Thin wrapper: pull layer-name / stats-dict from w1 and forward."""
    layer_name = getattr(w1, "_vllm_layer_name", None)
    stats_dict = getattr(w1, "_moe_activation_stats_of_model", None)
    if layer_name is None or stats_dict is None:
        # Only warn once per attribute to keep logs quiet.
        for _attr in ("_vllm_layer_name", "_moe_activation_stats_of_model"):
            if getattr(w1, _attr, None) is None and _attr not in _ANGELSLIM_WARNED:
                _ANGELSLIM_WARNED.add(_attr)
        return
    collect_fused_moe_internal_stats(
        stage=stage,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        global_num_experts=global_num_experts,
        layer_name=layer_name,
        global_moe_activation_stats=stats_dict,
    )
    # Per-channel down_proj-input absmax (only fires for stage=="down_proj"
    # and when VLLM_MOE_COLLECT_CHANNEL_STATS=1 + the stats dict was wired).
    channel_dict = getattr(w1, "_moe_channel_stats_of_model", None)
    if channel_dict is not None:
        collect_fused_moe_channel_stats(
            stage=stage,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            layer_name=layer_name,
            global_moe_channel_stats=channel_dict,
        )
    # Raw-token sampling (only stage=="down_proj" + VLLM_MOE_COLLECT_RAWTOK=1
    # + the buffer was wired). No-op otherwise.
    rawtok_dict = getattr(w1, "_moe_rawtok_of_model", None)
    if rawtok_dict is not None:
        collect_fused_moe_rawtok(
            stage=stage,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            layer_name=layer_name,
            global_moe_rawtok=rawtok_dict,
        )


{end_mark}
'''

# The two per-site hook call blocks.  Body lines carry 8-space indent
# because they live inside `TritonExperts.apply(...)`.  IMPORTANT: the
# FIRST line intentionally has NO leading whitespace -- the `str.replace`
# below substitutes only the bare kernel-call token, so the 8 spaces that
# originally preceded that token in the source are preserved and become
# the leading indent of the injected block's first line.
GATE_UP_HOOK = (
    "# === AngelSlim: gate_up_proj input capture ===\n"
    "        _angelslim_moe_stats_call(\n"
    "            stage=\"gate_up_proj\",\n"
    "            hidden_states=hidden_states,\n"
    "            topk_ids=topk_ids,\n"
    "            global_num_experts=global_num_experts,\n"
    "            w1=w1,\n"
    "        )\n"
    "        # === /AngelSlim ===\n"
)

DOWN_HOOK = (
    "# === AngelSlim: down_proj input capture ===\n"
    "        _angelslim_moe_stats_call(\n"
    "            stage=\"down_proj\",\n"
    "            hidden_states=intermediate_cache2,\n"
    "            topk_ids=topk_ids,\n"
    "            global_num_experts=global_num_experts,\n"
    "            w1=w1,\n"
    "        )\n"
    "        # === /AngelSlim ===\n"
)

# ------------------------------------------------------------- do the edit
# Insert file-level bootstrap right after the last `from ...` import block.
# We locate the position of the *last* top-level `from vllm.` import and
# insert a blank line + bootstrap block right after it.  If we can't find
# one (unlikely), fall back to end-of-imports heuristic.
last_import_match = None
for m in re.finditer(r"^from\s+\S+\s+import\s+.*$", src, re.M):
    last_import_match = m
if last_import_match is None:
    # Extremely defensive fallback: prepend before the first `class ` line.
    class_match = re.search(r"^class\s", src, re.M)
    insert_pos = class_match.start() if class_match else 0
    new_src = src[:insert_pos] + FILE_LEVEL_BOOTSTRAP + "\n" + src[insert_pos:]
else:
    insert_pos = last_import_match.end()
    new_src = src[:insert_pos] + "\n\n" + FILE_LEVEL_BOOTSTRAP + src[insert_pos:]

# Insert per-site hooks.  We do them in order and each `replace` uses count=1
# to guarantee we only touch the FIRST occurrence (which is inside
# TritonExperts.apply -- subclasses like TritonWNA16Experts have their own
# copies but we intentionally don't hook those; W-only quant paths are not
# what Hy4 BF16 calibration exercises).
if FIRST_KERNEL_ANCHOR not in new_src:
    print(f"[inject][ERR] gate_up_proj anchor vanished after bootstrap insertion", file=sys.stderr)
    sys.exit(2)
new_src = new_src.replace(FIRST_KERNEL_ANCHOR, GATE_UP_HOOK + "\n        " + FIRST_KERNEL_ANCHOR, 1)

if QUANT_ANCHOR not in new_src:
    print(f"[inject][ERR] down_proj anchor vanished after bootstrap insertion", file=sys.stderr)
    sys.exit(2)
new_src = new_src.replace(QUANT_ANCHOR, DOWN_HOOK + "\n        " + QUANT_ANCHOR, 1)

# ---------------------------------------------------------------- write out
with open(path, "w", encoding="utf-8") as f:
    f.write(new_src)

print(f"[inject] injected 2 hook sites + bootstrap into {path}")
PY
}

# ---------- subcommands ----------
do_install() {
    local vllm_dir="$1"

    local triton_moe_dst="${vllm_dir}/model_executor/layers/fused_moe/experts/triton_moe.py"
    local utils_dst_dir="${vllm_dir}/tools"
    local utils_dst="${utils_dst_dir}/vllm_calibrate_utils"

    # ---- sanity-check sources ----
    [[ -d "${CALIB_UTILS_SRC_DIR}" ]] || {
        err "Missing source package directory: ${CALIB_UTILS_SRC_DIR}"; exit 1;
    }
    [[ -f "${CALIB_UTILS_SRC_DIR}/__init__.py" ]] || {
        err "Source package is missing __init__.py: ${CALIB_UTILS_SRC_DIR}"; exit 1;
    }

    # ---- sanity-check target ----
    [[ -f "${triton_moe_dst}" ]] || {
        err "Expected HPC-vLLM file not found: ${triton_moe_dst}"
        err "This installer only supports the internal HPC fork of vLLM"
        err "(where TritonExperts lives in fused_moe/experts/triton_moe.py)."
        err "For community vLLM v0.20, use tools/vllm_patch/install.sh instead."
        exit 1;
    }

    # ---- back up original (only on first install) ----
    if [[ ! -f "${triton_moe_dst}.bak" ]]; then
        cp -p "${triton_moe_dst}" "${triton_moe_dst}.bak"
        log "Backed up triton_moe.py -> triton_moe.py.bak"
    else
        warn "Backup already exists: ${triton_moe_dst}.bak (kept untouched)"
        # Re-install: revert to backup FIRST so we always inject into a
        # pristine source tree. This makes the installer fully idempotent
        # even if a prior injection was interrupted mid-write.
        cp -p "${triton_moe_dst}.bak" "${triton_moe_dst}"
        log "Restored triton_moe.py from backup before re-injecting."
    fi

    # ---- install vllm_calibrate_utils package ----
    mkdir -p "${utils_dst_dir}"
    rm -rf "${utils_dst}"
    cp -rpL "${CALIB_UTILS_SRC_DIR}" "${utils_dst}"
    rm -rf "${utils_dst}/__pycache__"
    log "Installed ${utils_dst}/ (package)"

    # ---- inject the hooks ----
    inject_hook_into "${triton_moe_dst}" "${BEGIN_MARK}" "${END_MARK}" "${utils_dst_dir}"
    log "Patched ${triton_moe_dst}"

    # ---- self-check ----
    do_check "${vllm_dir}"
}

do_uninstall() {
    local vllm_dir="$1"

    local triton_moe_dst="${vllm_dir}/model_executor/layers/fused_moe/experts/triton_moe.py"
    local utils_dst="${vllm_dir}/tools/vllm_calibrate_utils"

    if [[ -f "${triton_moe_dst}.bak" ]]; then
        mv -f "${triton_moe_dst}.bak" "${triton_moe_dst}"
        log "Restored ${triton_moe_dst}"
    else
        warn "No backup found for triton_moe.py — leaving current file in place."
    fi

    if [[ -d "${utils_dst}" ]]; then
        rm -rf "${utils_dst}"
        log "Removed ${utils_dst}/"
    fi

    log "Uninstall complete."
}

do_check() {
    local vllm_dir="$1"

    local triton_moe_dst="${vllm_dir}/model_executor/layers/fused_moe/experts/triton_moe.py"
    local utils_dst="${vllm_dir}/tools/vllm_calibrate_utils"

    local ok=1

    log "Verifying patch state at: ${vllm_dir}"

    if [[ -f "${triton_moe_dst}" ]]; then
        log "  [OK]   ${triton_moe_dst} present"
    else
        err "  [FAIL] ${triton_moe_dst} missing"
        return 1
    fi

    if grep -qF "${BEGIN_MARK}" "${triton_moe_dst}" && grep -qF "${END_MARK}" "${triton_moe_dst}"; then
        log "  [OK]   AngelSlim hook markers present in triton_moe.py"
    else
        err "  [FAIL] AngelSlim hook markers NOT present in triton_moe.py"
        ok=0
    fi

    if grep -q "_angelslim_moe_stats_call(" "${triton_moe_dst}"; then
        local count
        count="$(grep -c '_angelslim_moe_stats_call(' "${triton_moe_dst}" || true)"
        # Expected: 1 definition + 2 call sites = 3 occurrences.
        if [[ "${count}" -ge 3 ]]; then
            log "  [OK]   Found ${count} _angelslim_moe_stats_call occurrences (>=3)"
        else
            err "  [FAIL] Only ${count} _angelslim_moe_stats_call occurrences found (expected >=3)"
            ok=0
        fi
    else
        err "  [FAIL] triton_moe.py does NOT contain _angelslim_moe_stats_call"
        ok=0
    fi

    if [[ -f "${utils_dst}/__init__.py" && -f "${utils_dst}/hooks.py" ]]; then
        log "  [OK]   ${utils_dst}/ package present"
        if grep -q "^def collect_fused_moe_internal_stats" "${utils_dst}/hooks.py" 2>/dev/null; then
            log "  [OK]   hooks.py defines collect_fused_moe_internal_stats"
        else
            err "  [FAIL] hooks.py missing collect_fused_moe_internal_stats definition"
            ok=0
        fi
    else
        err "  [FAIL] ${utils_dst}/ package missing or incomplete"
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
