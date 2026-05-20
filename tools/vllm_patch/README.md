# vLLM Patch for AngelSlim Calibration

This directory contains patch files that need to be applied to an installed
vLLM package to enable AngelSlim's PTQ calibration features (especially MoE
expert statistics collection on `FusedMoE` layers).

## What's in this directory

| File                | Purpose                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| `fused_moe.py`      | Patched version of `vllm/model_executor/layers/fused_moe/fused_moe.py` with AngelSlim hooks injected.  |
| `envs.py`           | Patched version of `vllm/envs.py` that adds `VLLM_MOE_COLLECT_STATS*` environment variables.           |
| `README.md`         | This file.                                                                                             |

These patches are aligned with the **current** vLLM version installed in the
calibration environment. If your vLLM version differs, the patch files may
need to be regenerated against your specific vLLM source.

## Required companion package: `vllm_calibrate_utils/`

`fused_moe.py` imports `collect_fused_moe_internal_stats` from a module named
`vllm_calibrate_utils`. The lookup logic walks up from the patched
`fused_moe.py` location and appends `vllm/tools/`, so the calibration utils
package **must** be placed inside the installed vLLM package as:

```
<vllm_install_dir>/tools/vllm_calibrate_utils/
├── __init__.py        # re-exports every public symbol (incl.
│                      #   collect_fused_moe_internal_stats)
├── _common.py         # shared low-level helpers
├── hooks.py           # all forward-hook based calibration:
│                      #   activations, KV (per-tensor / per-head /
│                      #   KV-only), MoE stats (collect_fused_moe_internal_stats),
│                      #   MTP draft model
└── search.py          # KV-cache FP8 scale grid-search
                       #   (per-tensor + per-head)
```

The single source of truth for this package lives at:

```
angelslim/compressor/quant/core/vllm_calibrate_utils/
```

`install.sh` copies this whole directory in one shot; `uninstall` removes it
(and also cleans up the legacy single-file `vllm_calibrate_utils.py` from
pre-split installations if it is still around).

## Deployment

Assuming `VLLM_DIR` points to your installed vLLM package directory (e.g.
`/usr/local/lib/python3.12/dist-packages/vllm` or your editable-install
checkout), run:

```bash
bash tools/vllm_patch/install.sh install
```


## Reverting

To restore the original vLLM files:

```bash
bash tools/vllm_patch/install.sh uninstall
```
