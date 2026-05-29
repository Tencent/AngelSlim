# `tools/kvcache/` — KV-cache calibration & scale post-processing

This directory groups the **standalone KV-cache** utilities that work on
top of an already-trained / already-quantized model.  Both tools are
plain Python CLI entrypoints — invoke them from the **AngelSlim repo
root** (the same convention as everything else under `tools/`).

| File | Role | Stage |
| --- | --- | --- |
| [`run_kvcache_calibrate.py`](./run_kvcache_calibrate.py) | KV-only calibration with vLLM (no weight / activation / MoE hooks). Optionally searches for the best per-layer or per-head KV scale multiplier. | Calibration |
| [`replace_kv_scales.py`](./replace_kv_scales.py) | Patch an existing FP8 model's `kv_cache_scales.safetensors` with calibrated scales coming from the JSON files produced by `run_kvcache_calibrate.py` (or by `tools/run_vllm_calibrate.py`). Also rewrites `attn_quant_config` inside `config.json`. | Post-quant |

The two tools form a **complete KV-only pipeline** that can be run on
top of a model that has already been quantized for weights / activations
(e.g. via `tools/run_vllm_calibrate.py` + `tools/fp8_quant_with_vllm_activation.py`).
They share the same JSON layout, so the calibrator's outputs feed
directly into the patcher.

> ℹ️  These are **standalone** scripts — they do **not** import each
> other. The heavy lifting (hooks, scale search) lives in
> `angelslim.compressor.quant.core.vllm_calibrate_utils`.

---

## 1. `run_kvcache_calibrate.py` — KV-only calibration

Compared with `tools/run_vllm_calibrate.py`, this script registers
**only** the Attention min/max hooks. It is intended for fast,
KV-focused experiments (e.g. sweeping search ranges) when you don't
need fresh weight / activation / MoE statistics.

### Outputs

Per granularity, written under `--output-dir`:

| Granularity | Stats file | Search-multipliers | Final tuned scales |
| --- | --- | --- | --- |
| `per-tensor` (default) | `activation_stats.json` | `kv_scale_multipliers.json` | `kv_cache_tuned_scales.json` |
| `per-head` (`--per-head`) | `activation_stats_per_head.json` | `kv_scale_multipliers_per_head.json` | `kv_cache_tuned_scales_per_head.json` |

The search step is opt-in (`--search-kv-scale`); without it only the
raw min/max stats file is produced.

### Typical invocation

```bash
python3 tools/kvcache/run_kvcache_calibrate.py \
    --model-path     /path/to/bf16_model \
    --ptq-data-path  /path/to/ptq_dataset.json \
    --output-dir     /path/to/kv_stats \
    --tp-size        16 \
    --batch-size     4 \
    --num-samples    512 \
    --max-length     32768 \
    --per-head \
    --search-kv-scale \
    --search-kv-num-samples   32 \
    --search-kv-min-multiplier 0.4 \
    --search-kv-max-multiplier 8.0 \
    --search-kv-num-steps      50
```

A ready-to-run wrapper for Hy3 lives at
[`scripts/ptq/run_kvcache_calibrate_for_Hy3.sh`](../../scripts/ptq/run_kvcache_calibrate_for_Hy3.sh).

> ⚠️  Requires the AngelSlim vLLM patch
> ([`tools/vllm_patch/`](../vllm_patch/)) to be installed in the active
> vLLM environment, same as the W8A8C8 calibration flow.

---

## 2. `replace_kv_scales.py` — patch FP8 model with tuned scales

After calibration you have a JSON of calibrated KV scales, but your
quantized model still ships its original (often suboptimal) ones inside
`kv_cache_scales.safetensors`. This tool patches them in-place.

It supports both granularities and selects the JSON layout via
`--granularity`:

* `per-tensor` — one scalar scale per layer-slot (key:
  `model.layers.N.self_attn.{k,v}_cache.scale`).
* `per-head`   — one scale per KV head per layer (key:
  `model.layers.N.self_attn.{k,v}_cache.head_H.scale`); legacy JSON
  files where heads are TP-replicated are auto-deduplicated using
  `--tp-size` / `--num-kv-heads`.

It additionally rewrites `attn_quant_config` inside the **`config.json`
sitting next to `--output`** so the model is loadable by inference
runtimes that read this field.

### Per-head usage (default)

```bash
python3 tools/kvcache/replace_kv_scales.py \
    --granularity per-head \
    --json   /path/to/kv_stats/kv_cache_tuned_scales_per_head.json \
    --src    /path/to/fp8_model/kv_cache_scales.safetensors \
    --output /path/to/fp8_model/kv_cache_scales.safetensors \
    --num-kv-heads 8
```

### Per-tensor usage

```bash
python3 tools/kvcache/replace_kv_scales.py \
    --granularity per-tensor \
    --json   /path/to/kv_stats/kv_cache_tuned_scales.json \
    --src    /path/to/fp8_model/kv_cache_scales.safetensors \
    --output /path/to/fp8_model/kv_cache_scales.safetensors
```

If `--output` is omitted, the source file is overwritten in place and a
`*.bak` backup is created automatically.

---

## End-to-end pattern (KV-only refinement of an FP8 model)

```text
  bf16 model + dataset
          │
          ▼
  tools/kvcache/run_kvcache_calibrate.py
          │     (writes kv_cache_tuned_scales*.json)
          ▼
  tools/kvcache/replace_kv_scales.py
          │     (patches fp8_model/kv_cache_scales.safetensors,
          │      rewrites fp8_model/config.json's attn_quant_config)
          ▼
  ready-to-serve FP8 model with refined KV scales
```

For the full W8A8C8 flow (weights + activations + MoE + KV in one
shot), see [`scripts/ptq/README.md`](../../scripts/ptq/README.md).
