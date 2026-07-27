#!/usr/bin/env python3
"""Patch a GLM-5 EP-saved INT8 checkpoint so vLLM can load it.

The ``Glm5EPQuantSaver`` writes per-rank ``model-r{RR}-{SSSSS}.safetensors``
shards + ``model.safetensors.index.json`` + ``hf_quant_config.json`` +
tokenizer, but forgot to emit ``config.json`` (which vLLM demands) and
``generation_config.json``.  This script fills those in without touching
weights:

  * Copies ``config.json`` from the original model directory
  * Injects a ``quantization_config`` field derived from
    ``hf_quant_config.json`` (compressed-tensors / vLLM INT8 W8A8C8 recipe)
  * Copies ``generation_config.json`` if present
  * Leaves everything else (weights, index, tokenizer) untouched

Usage:
    python3 scripts/ptq/fix_glm5_vllm_meta.py \
        --orig-model /path/to/original/chatglm5.2 \
        --save-path  /path/to/output_glm5_w8a8c8_2node/glm5_w8a8c8_2node
"""
import argparse
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-model", required=True,
                    help="Path to the original bf16 GLM-5.2 checkpoint dir "
                         "(source of config.json / generation_config.json).")
    ap.add_argument("--save-path", required=True,
                    help="Path to the quantized EP-saved dir "
                         "(destination for config.json + generation_config.json).")
    ap.add_argument("--activation-scheme", default="dynamic",
                    choices=("dynamic", "static"))
    ap.add_argument("--kv-cache-scheme", default="static",
                    choices=("static", "none"),
                    help="Set to 'static' for W8A8C8, 'none' for W8A8.")
    args = ap.parse_args()

    orig = args.orig_model
    dst = args.save_path

    orig_cfg_p = os.path.join(orig, "config.json")
    dst_cfg_p = os.path.join(dst, "config.json")
    hf_quant_p = os.path.join(dst, "hf_quant_config.json")

    if not os.path.isfile(orig_cfg_p):
        print(f"[ERR] missing {orig_cfg_p}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(hf_quant_p):
        print(f"[ERR] missing {hf_quant_p}", file=sys.stderr)
        sys.exit(1)

    with open(orig_cfg_p, "r") as f:
        cfg = json.load(f)
    with open(hf_quant_p, "r") as f:
        trtllm = json.load(f)

    exclude_modules = trtllm["quantization"]["exclude_modules"]

    # compressed-tensors / vLLM INT8 W8A8(C8) quantization_config
    quantization_config = {
        "quant_method": "compressed-tensors",
        "ignore": exclude_modules,
        "activation_scheme": args.activation_scheme,
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                    "type": "int",
                    "symmetric": True,
                },
                "weights": {
                    "num_bits": 8,
                    "strategy": "channel",
                    "dynamic": False,
                    "type": "int",
                    "symmetric": True,
                },
                "targets": ["Linear"],
            }
        },
        "format": "int-quantized",
    }
    if args.kv_cache_scheme == "static":
        quantization_config["kv_cache_scheme"] = {
            "num_bits": 8,
            "strategy": "tensor",
            "type": "float",
            "dynamic": True,
        }

    # Overwrite any existing quantization_config field.
    cfg.pop("quantization_config", None)
    cfg["quantization_config"] = quantization_config

    # ------------------------------------------------------------------
    # Head-dim repair (mirrors ``GLM5._fix_hf_config`` on the loader side).
    # ------------------------------------------------------------------
    # ``configuration_glm_moe_dsa.py`` declares ``"head_dim": "qk_rope_head_dim"``
    # in its HF ``attribute_map``.  Since the released ``config.json`` sets
    # ``head_dim: 192`` (== qk_nope_head_dim) as a *display* alias, HF
    # silently overrides ``qk_rope_head_dim`` to 192, which then makes
    # vLLM compute ``head_size = kv_lora_rank + qk_rope_head_dim =
    # 512 + 192 = 704``.  No MLA backend supports 704 -- FLASHMLA_SPARSE
    # accepts only [512, 576] (flashmla_sparse.py:121).  The remedy is to
    # write ``head_dim = qk_rope_head_dim = 64`` in the shipped config so
    # HF's attribute_map produces the correct 64 at load time; the
    # resulting head_size = 512 + 64 = 576 is supported.
    if int(cfg.get("head_dim", -1)) != 64 or int(cfg.get("qk_rope_head_dim", -1)) != 64:
        old_hd = cfg.get("head_dim")
        cfg["head_dim"] = 64
        cfg["qk_rope_head_dim"] = 64
        # Recompute qk_head_dim from the corrected fields.
        qnope = int(cfg.get("qk_nope_head_dim", 192))
        cfg["qk_nope_head_dim"] = qnope
        cfg["qk_head_dim"] = qnope + 64
        print(f"[OK] head-dim repair: head_dim {old_hd} -> 64, "
              f"qk_head_dim -> {cfg['qk_head_dim']} "
              f"(vLLM head_size = kv_lora_rank + qk_rope_head_dim = "
              f"{int(cfg.get('kv_lora_rank', 512))} + 64 = "
              f"{int(cfg.get('kv_lora_rank', 512)) + 64})")

    with open(dst_cfg_p, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[OK] wrote {dst_cfg_p} "
          f"(exclude_modules={len(exclude_modules)}, "
          f"activation={args.activation_scheme}, "
          f"kv_cache={args.kv_cache_scheme})")

    # generation_config.json (best-effort copy).
    orig_gen = os.path.join(orig, "generation_config.json")
    dst_gen = os.path.join(dst, "generation_config.json")
    if os.path.isfile(orig_gen) and not os.path.isfile(dst_gen):
        shutil.copy2(orig_gen, dst_gen)
        print(f"[OK] copied generation_config.json")
    elif os.path.isfile(dst_gen):
        print(f"[skip] generation_config.json already exists")

    # Sanity summary.
    have = {
        "config.json": os.path.isfile(dst_cfg_p),
        "model.safetensors.index.json": os.path.isfile(
            os.path.join(dst, "model.safetensors.index.json")
        ),
        "tokenizer.json": os.path.isfile(os.path.join(dst, "tokenizer.json")),
        "tokenizer_config.json": os.path.isfile(
            os.path.join(dst, "tokenizer_config.json")
        ),
        "generation_config.json": os.path.isfile(dst_gen),
        "hf_quant_config.json": os.path.isfile(hf_quant_p),
    }
    print("[sanity] files present:")
    for k, v in have.items():
        print(f"   {'✓' if v else '✗'}  {k}")


if __name__ == "__main__":
    main()
