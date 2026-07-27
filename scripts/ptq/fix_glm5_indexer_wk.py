#!/usr/bin/env python3
"""Dequantize ``indexer.wk`` weights from INT8 back to BF16 in-place.

vLLM's ``deepseek_v2.py`` fuses ``indexer.wk`` + ``indexer.weights_proj``
into a single ``MergedColumnParallelLinear(wk_weights_proj)``.  Because
``weights_proj`` is in our quantization ``ignore`` list (bf16), the fused
module is treated as *unquantized* by ``CompressedTensorsW8A8Int8`` --
i.e. it has no ``weight_scale`` parameter.  Our checkpoint however
contains ``indexer.wk.weight_scale`` (since we DID quantize wk under W8A8C8).
vLLM's stacked-params mapping renames ``indexer.wk.weight_scale`` ->
``indexer.wk_weights_proj.weight_scale``, then looks it up in
``params_dict`` -> ``KeyError``.

Fix: for every ``*.indexer.wk.weight`` INT8 tensor in the sharded
checkpoint, dequantize back to BF16 (``bf16 = int8.float() * scale``),
DELETE the corresponding ``weight_scale`` from the same shard, and drop
the ``weight_scale`` entry from ``model.safetensors.index.json``.  The
shard file is rewritten in place.  Also expands the ``ignore`` list in
``config.json`` and ``hf_quant_config.json`` to include every
``indexer.wk`` so vLLM won't try to look up any residual scale.

Usage:
    python3 scripts/ptq/fix_glm5_indexer_wk.py \
        --save-path /path/to/output_glm5_w8a8c8_2node/glm5_w8a8c8_2node
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequantize_int8_per_channel(w_int8: torch.Tensor, scale: torch.Tensor,
                                out_dtype=torch.bfloat16) -> torch.Tensor:
    """Per-channel INT8 dequantization: bf16 = int8.float() * scale.

    ``w_int8`` shape is [out_features, in_features] (torch.int8).
    ``scale`` shape is [out_features] or [out_features, 1] (bf16/float).
    """
    if scale.ndim == 1:
        scale = scale.view(-1, 1)
    return (w_int8.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", required=True)
    args = ap.parse_args()

    dst = args.save_path
    idx_path = os.path.join(dst, "model.safetensors.index.json")
    cfg_path = os.path.join(dst, "config.json")
    hf_quant_path = os.path.join(dst, "hf_quant_config.json")

    with open(idx_path, "r") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # ----- discover every indexer.wk key we need to dequantize -----------
    wk_weight_keys = [k for k in weight_map if k.endswith(".indexer.wk.weight")]
    wk_scale_keys = [k for k in weight_map if k.endswith(".indexer.wk.weight_scale")]
    print(f"[info] found {len(wk_weight_keys)} indexer.wk.weight tensors, "
          f"{len(wk_scale_keys)} indexer.wk.weight_scale tensors")
    if not wk_weight_keys:
        print("[warn] no indexer.wk.weight found; nothing to do")
        return

    # ----- group by shard file so we rewrite each shard once -------------
    # For every shard we need to know:
    #   * which tensors currently live inside it
    #   * which of them are indexer.wk.weight  (int8 -> replace with bf16)
    #   * which of them are indexer.wk.weight_scale (drop entirely)
    # Wk weight & scale may live in DIFFERENT shards (unlikely but possible),
    # so we first collect all scales into a dict.
    shard_to_wk_weights = defaultdict(list)   # shard -> [wk keys]
    shard_to_wk_scales = defaultdict(list)    # shard -> [wk scale keys]
    for k in wk_weight_keys:
        shard_to_wk_weights[weight_map[k]].append(k)
    for k in wk_scale_keys:
        shard_to_wk_scales[weight_map[k]].append(k)

    # Load all scales into memory (small: 78 tensors of shape [head_dim]).
    scale_cache = {}
    for shard_file, keys in shard_to_wk_scales.items():
        p = os.path.join(dst, shard_file)
        with safe_open(p, framework="pt") as r:
            for k in keys:
                scale_cache[k] = r.get_tensor(k)
    print(f"[info] cached {len(scale_cache)} weight_scale tensors")

    # ----- rewrite every affected shard ----------------------------------
    affected_shards = set(shard_to_wk_weights.keys()) | set(shard_to_wk_scales.keys())
    print(f"[info] will rewrite {len(affected_shards)} shard files")

    dequant_count = 0
    dropped_scales = 0

    for shard_file in sorted(affected_shards):
        p = os.path.join(dst, shard_file)
        keys_in_shard = [k for k, v in weight_map.items() if v == shard_file]
        new_tensors = {}
        with safe_open(p, framework="pt") as r:
            for k in keys_in_shard:
                if k.endswith(".indexer.wk.weight"):
                    scale_k = k[: -len(".weight")] + ".weight_scale"
                    if scale_k not in scale_cache:
                        print(f"[warn] no scale for {k}; keeping int8 as-is")
                        new_tensors[k] = r.get_tensor(k)
                        continue
                    w_int8 = r.get_tensor(k)
                    scale = scale_cache[scale_k]
                    if w_int8.dtype != torch.int8:
                        # Already dequantized (idempotent run).
                        new_tensors[k] = w_int8
                        continue
                    w_bf16 = dequantize_int8_per_channel(w_int8, scale)
                    new_tensors[k] = w_bf16.contiguous()
                    dequant_count += 1
                elif k.endswith(".indexer.wk.weight_scale"):
                    # Drop from shard; also drop from weight_map below.
                    dropped_scales += 1
                    continue
                else:
                    new_tensors[k] = r.get_tensor(k)
        save_file(new_tensors, p)
        print(f"[ok]  rewrote {shard_file}: kept {len(new_tensors)} tensors")

    # ----- drop weight_scale keys from the index -------------------------
    for k in wk_scale_keys:
        weight_map.pop(k, None)
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"[ok]  updated model.safetensors.index.json "
          f"(dropped {len(wk_scale_keys)} weight_scale entries; "
          f"{len(weight_map)} keys remain)")

    # ----- extend the ignore list in config.json + hf_quant_config.json --
    wk_module_names = sorted({
        k[: -len(".weight")] for k in wk_weight_keys
    })  # e.g. "model.layers.0.self_attn.indexer.wk"

    for p, top_key in ((cfg_path, "quantization_config"),
                       (hf_quant_path, "quantization")):
        if not os.path.isfile(p):
            continue
        with open(p, "r") as f:
            j = json.load(f)
        section = j.get(top_key, {})
        ignore_key = "ignore" if top_key == "quantization_config" else "exclude_modules"
        current = list(section.get(ignore_key, []))
        added = [n for n in wk_module_names if n not in current]
        if added:
            current.extend(added)
            current.sort()
            section[ignore_key] = current
            j[top_key] = section
            with open(p, "w") as f:
                json.dump(j, f, indent=2)
            print(f"[ok]  {os.path.basename(p)}: added "
                  f"{len(added)} indexer.wk entries to '{ignore_key}' "
                  f"(total {len(current)})")

    print()
    print(f"[done] dequantized {dequant_count} indexer.wk tensors, "
          f"dropped {dropped_scales} weight_scale tensors")


if __name__ == "__main__":
    main()
