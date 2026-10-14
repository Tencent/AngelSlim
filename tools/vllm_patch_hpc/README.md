# AngelSlim vLLM Calibration Hook — HPC fork installer

This directory ships an **HPC-fork-aware** installer for AngelSlim's MoE
activation calibration hook. It is a companion of, **not a replacement
for**, the community-vLLM installer under [`../vllm_patch/`](../vllm_patch/).

Pick the right installer based on which vLLM you are running:

| Your vLLM build | Installer to use |
| --- | --- |
| Community vLLM (e.g. `pip install vllm==0.10.x` / `0.20.x`) — `class TritonExperts` still lives in `<vllm>/model_executor/layers/fused_moe/fused_moe.py` | [`tools/vllm_patch/install.sh`](../vllm_patch/install.sh) |
| **Internal HPC fork** (e.g. `0.20.1.dev4xx+g<hash>`) — with 40+ `VLLM_USE_HPC_*` / `VLLM_ENABLE_HPC_*` env vars in `envs.py`, and `TritonExperts` moved to `<vllm>/model_executor/layers/fused_moe/experts/triton_moe.py` | **`tools/vllm_patch_hpc/install.sh`** (this dir) |

Running the wrong installer against an HPC fork will silently kill Hy4
serving because the community installer whole-file-overwrites `envs.py`,
losing every `VLLM_USE_HPC_FUSEMOE` / `VLLM_MLA_FP8_DYNAMIC_QUANT` /
`VLLM_PRECISION_MODE` / `VLLM_ENABLE_TPSP` / ... declaration that HPC
requires. It also patches the wrong `fused_moe.py` (the HPC fork's hook
target is `triton_moe.py`), so no MoE stats are collected.

## What this installer does

`bash install.sh install` performs a **surgical, marker-guarded in-place
edit** of the HPC fork's `triton_moe.py`, and installs
`vllm_calibrate_utils/` as a Python package alongside it. Concretely, it:

1. Auto-detects the vLLM install directory via `python3 -c 'import vllm'`.
2. Locates
   `<vllm>/model_executor/layers/fused_moe/experts/triton_moe.py`.
3. Backs it up to `triton_moe.py.bak` (first install only; re-installs
   restore from this backup before re-injecting, so the installer is
   fully idempotent even after interrupted runs).
4. Injects:
   * a file-level bootstrap block right after the last `from ...` import
     line — this block adds `<vllm>/tools/` to `sys.path` and imports
     `collect_fused_moe_internal_stats`, falling back to a no-op if the
     package isn't installed;
   * a `stage="gate_up_proj"` hook call **before** the first
     `invoke_fused_moe_triton_kernel(...)` in `TritonExperts.apply()` (so
     it captures the raw pre-quant `hidden_states` of the gate_up GEMM);
   * a `stage="down_proj"` hook call **before**
     `moe_kernel_quantize_input(...)` (so it captures the post-activation
     `intermediate_cache2` — the input to the down_proj GEMM — in its
     un-quantised BF16 form).
5. Copies
   `angelslim/compressor/quant/core/vllm_calibrate_utils/` to
   `<vllm>/tools/vllm_calibrate_utils/`.

Both hooks are **no-ops when `VLLM_MOE_COLLECT_STATS != "1"`**, so the
patched `triton_moe.py` is safe to leave installed permanently — regular
(non-calibration) Hy4 serving pays only two attribute lookups plus one
`os.getenv` per MoE forward.

## What this installer does **NOT** do

Compared with `tools/vllm_patch/install.sh`, this installer deliberately:

* **Does NOT touch `<vllm>/envs.py`.** The HPC fork's `envs.py` declares
  40+ HPC-only env vars; overwriting it kills every downstream HPC
  feature and Hy4 serving stops importing.
* **Does NOT touch `<vllm>/model_executor/layers/fused_moe/fused_moe.py`.**
  On the HPC fork, `class TritonExperts` no longer lives there (it was
  moved to `experts/triton_moe.py`), and `fused_moe.py` now contains HPC
  SM90 `swap_ab` GEMM optimisations that whole-file replacement would
  destroy.
* **Does NOT install `smooth_moe_inject.py`.** Hy4's NVFP4 MoE
  calibration only needs the global input `scale_amax` (i.e.
  `collect_fused_moe_internal_stats`); smooth stats / alpha search are
  not part of the Hy4 pipeline.
* **Does NOT depend on any AngelSlim-added `envs.py` field.** The hook
  gates on `os.getenv("VLLM_MOE_COLLECT_STATS")` at runtime, which does
  not require declaration in `envs.py`.

## Usage

Run once on **every node** that will host a vLLM worker (Ray head + every
remote worker) — the installer only patches the local machine's vLLM.

```bash
# from the AngelSlim repo root
bash tools/vllm_patch_hpc/install.sh              # install (default)
bash tools/vllm_patch_hpc/install.sh install      # explicit install
bash tools/vllm_patch_hpc/install.sh check        # verify patch is active
bash tools/vllm_patch_hpc/install.sh uninstall    # restore triton_moe.py from .bak
bash tools/vllm_patch_hpc/install.sh --help
```

Post-install, `check` should report all `[OK]` lines including:

```
[OK]   AngelSlim hook markers present in triton_moe.py
[OK]   Found 3 _angelslim_moe_stats_call occurrences (>=3)
[OK]   <vllm>/tools/vllm_calibrate_utils/ package present
[OK]   hooks.py defines collect_fused_moe_internal_stats
```

The `>=3` is expected: one wrapper definition (`_angelslim_moe_stats_call`
at file-level) + two call sites (gate_up_proj + down_proj) = 3 grep hits.

## Runtime activation

At calibration time, export **all three** of these on the launcher shell
before invoking `tools/run_vllm_calibrate.py`:

```bash
export VLLM_MOE_COLLECT_STATS=1
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1
export VLLM_MOE_COLLECT_STATS_VERBOSE=0
```

Serving (i.e. `bash /root/vllm.sh tp`) does **not** need to set these —
without them the hook returns immediately.

## Verifying the hook fires (sanity script)

```python
python3 - <<'PY'
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts, collect_fused_moe_internal_stats, _angelslim_moe_stats_call,
)
print("TritonExperts:", TritonExperts)
print("collect_fused_moe_internal_stats:", collect_fused_moe_internal_stats)
print("_angelslim_moe_stats_call:", _angelslim_moe_stats_call)
PY
```

All three imports must succeed. The `collect_fused_moe_internal_stats`
callable comes from the freshly-installed
`<vllm>/tools/vllm_calibrate_utils/` package.
