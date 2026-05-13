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

## Required companion file: `vllm_calibrate_utils.py`

`fused_moe.py` imports `collect_fused_moe_internal_stats` from a module named
`vllm_calibrate_utils`. The lookup logic walks up from the patched
`fused_moe.py` location and appends `vllm/tools/`, so the calibration utils
file **must** be placed inside the installed vLLM package as:

```
<vllm_install_dir>/tools/vllm_calibrate_utils.py
```

The single source of truth for this file lives at:

```
angelslim/compressor/quant/core/vllm_calibrate_utils.py
```

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
