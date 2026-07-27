#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Standalone offline diagnostic for the GLM-5 MTP (Multi-Token-Prediction)
# draft layer in a source HuggingFace checkpoint.
#
# Usage:
#   python scripts/ptq/diag_glm5_mtp.py \
#       /apdcephfs_zwfy2/share_300532381/harviexu/chatglm5.2
#
# What it does (all read-only, seconds to run, no torch import needed):
#   1. Loads ``config.json`` and reports ``num_hidden_layers`` +
#      ``num_nextn_predict_layers`` -> derives the MTP layer id
#      (= num_hidden_layers).
#   2. Loads ``model.safetensors.index.json`` (or ``pytorch_model.bin.index.json``)
#      and buckets every key by its ``model.layers.<lid>.`` prefix.
#      Prints per-layer key counts plus a full listing of every
#      ``model.layers.<MTP>.*`` key found on disk (this is the *ground
#      truth* set of MTP tensors we need to reproduce in the exported
#      checkpoint).
#   3. Cross-checks a few kunlun-recipe expectations:
#      * mtp_block.eh_proj / enorm / hnorm / shared_head.*  (dense sub-mods)
#      * mtp_block.self_attn.*  (attention sub-module)
#      * mtp_block.mlp.experts.<i>.{gate,up,down}_proj.weight  (MoE)
#   4. Reports which of ``indexer_types`` / ``mlp_layer_types`` schedule
#      entries apply to the MTP layer (needed to decide whether MTP is a
#      full-indexer sparse-MoE layer, which the AngelSlim padding assumes).
#
# The script deliberately avoids any torch / transformers import so it can
# be run on a login node or from a Jupyter cell without materialising the
# model.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _pick_index_json(model_dir: str) -> str | None:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    return None


def _layer_id(key: str) -> int | None:
    m = re.match(r"model\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _sub_pattern(key: str, mtp_lid: int) -> str:
    """Return the "sub-key" under model.layers.<MTP>., anonymised so
    per-expert indices collapse into a single bucket."""
    prefix = f"model.layers.{mtp_lid}."
    tail = key[len(prefix):] if key.startswith(prefix) else key
    # collapse experts.<eid>. -> experts.<*>.
    tail = re.sub(r"experts\.\d+\.", "experts.<*>.", tail)
    return tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "model_dir",
        help="Path to the source HF checkpoint (e.g. /apdcephfs_zwfy2/.../chatglm5.2)",
    )
    args = ap.parse_args()
    model_dir = args.model_dir

    print(f"[diag] model_dir = {model_dir}")

    # -------- (1) config.json --------
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"[diag] ERROR: no config.json at {cfg_path}", file=sys.stderr)
        return 2
    cfg = _load_json(cfg_path)
    n_main = int(cfg.get("num_hidden_layers", 0))
    n_mtp = int(cfg.get("num_nextn_predict_layers", 0) or 0)
    mtp_lid = n_main  # GLM-5 packs MTP as model.layers.<num_hidden_layers>
    print(f"[diag] num_hidden_layers        = {n_main}")
    print(f"[diag] num_nextn_predict_layers = {n_mtp}")
    print(f"[diag] MTP layer id (= n_main)  = {mtp_lid}")
    print(f"[diag] hidden_size              = {cfg.get('hidden_size')}")
    print(f"[diag] intermediate_size (MoE)  = {cfg.get('moe_intermediate_size')}")
    print(f"[diag] num_experts (routed)     = {cfg.get('n_routed_experts') or cfg.get('num_experts')}")

    # Schedule lists that AngelSlim pads to n_main + n_mtp so MTP
    # construction is safe.  The MTP layer's entries tell us what it
    # actually IS on disk.
    for lname in ("indexer_types", "mlp_layer_types",
                  "moe_layer_freq", "layer_types"):
        lval = cfg.get(lname)
        if isinstance(lval, list):
            entry = lval[mtp_lid] if mtp_lid < len(lval) else "<OUT-OF-RANGE>"
            print(f"[diag] cfg.{lname:<20s} len={len(lval):>3d}  "
                  f"MTP entry (idx {mtp_lid}) = {entry!r}")

    # -------- (2) index.json --------
    idx_path = _pick_index_json(model_dir)
    if idx_path is None:
        print(f"[diag] ERROR: no index.json found under {model_dir}", file=sys.stderr)
        return 3
    idx = _load_json(idx_path)
    weight_map = idx.get("weight_map", {})
    all_keys = list(weight_map.keys())
    print(f"[diag] index.json = {os.path.relpath(idx_path, model_dir)}"
          f"  ({len(all_keys)} keys)")

    # -------- (3) bucket by layer id --------
    by_layer: dict[int | None, list[str]] = defaultdict(list)
    for k in all_keys:
        by_layer[_layer_id(k)].append(k)
    print(f"[diag] top-level (non-layer) keys: {len(by_layer.get(None, []))}")
    for k in sorted(by_layer.get(None, []))[:20]:
        print(f"[diag]   top-level  {k}")
    if len(by_layer.get(None, [])) > 20:
        print(f"[diag]   ...({len(by_layer[None]) - 20} more top-level keys)")

    _layer_ids_sorted = sorted(k for k in by_layer.keys() if k is not None)
    print(f"[diag] layers present in index.json: {_layer_ids_sorted}")

    # -------- (4) MTP-layer key listing --------
    mtp_keys = sorted(by_layer.get(mtp_lid, []))
    print(f"\n[diag] === MTP LAYER {mtp_lid}: {len(mtp_keys)} keys ===")
    if not mtp_keys:
        print(f"[diag] WARNING: no MTP keys found in index.json.  Either "
              f"this checkpoint has no MTP block, or MTP is stored under "
              f"a different key prefix.  Searching for likely alt prefixes...")
        for probe in ("mtp", "mtp_block", "nextn", "draft", "eh_proj",
                      "enorm", "hnorm", "shared_head"):
            hits = [k for k in all_keys if probe in k]
            if hits:
                print(f"[diag]   probe '{probe}' matched {len(hits)} keys, "
                      f"first: {hits[0]}")

    # Anonymised sub-pattern histogram: this is the schema we need to
    # reproduce in the exported checkpoint.
    sub_hist: Counter[str] = Counter()
    for k in mtp_keys:
        sub_hist[_sub_pattern(k, mtp_lid)] += 1
    print(f"[diag] MTP anonymised sub-keys ({len(sub_hist)} unique):")
    for sub, cnt in sorted(sub_hist.items()):
        print(f"[diag]   [{cnt:>3d}x] model.layers.{mtp_lid}.{sub}")

    # Also dump full listing (verbose) so we can grep for specific tensors
    # (eh_proj / enorm / hnorm / shared_head / experts.<i>).
    print(f"\n[diag] --- full MTP key list (verbose) ---")
    for k in mtp_keys:
        shard = weight_map.get(k, "?")
        print(f"[diag]   {k}   -> {shard}")

    # -------- (5) targeted cross-checks --------
    checks = {
        "eh_proj":           r"\.eh_proj\.",
        "enorm":             r"\.enorm\.",
        "hnorm":             r"\.hnorm\.",
        "shared_head":       r"\.shared_head\.",
        "mtp_block.self_attn": r"\.mtp_block\.self_attn\.",
        "mtp_block.mlp.experts": r"\.mtp_block\.mlp\.experts\.",
        "mtp_block.mlp.gate":    r"\.mtp_block\.mlp\.gate\.",
        "mtp_block.mlp.shared_experts": r"\.mtp_block\.mlp\.shared_experts\.",
        "mtp_block.indexer": r"\.mtp_block\..*indexer\.",
        "post_attention_layernorm (mtp)":
                                r"\.mtp_block\..*post_attention_layernorm",
        "input_layernorm (mtp)":
                                r"\.mtp_block\..*input_layernorm",
    }
    print(f"\n[diag] === TARGETED PRESENCE CHECKS ===")
    prefix_mtp = f"model.layers.{mtp_lid}."
    for label, pat in checks.items():
        rgx = re.compile(pat)
        hits = [k for k in mtp_keys if rgx.search(k)]
        # keep only the mtp-layer scope; some patterns (eh_proj etc.) may
        # also live top-level in some releases so also probe those.
        top_hits = [k for k in all_keys
                    if not k.startswith(prefix_mtp) and rgx.search(k)]
        print(f"[diag]   [{'OK ' if hits else '!! '}] {label:<40s} "
              f"MTP-scoped={len(hits)}  top-level={len(top_hits)}"
              f"{'   e.g. ' + hits[0] if hits else ''}"
              f"{'   [top] e.g. ' + top_hits[0] if top_hits else ''}")

    # -------- (6) compare MTP schema to a normal MoE layer as reference --------
    ref_lid = None
    for lid in sorted(k for k in by_layer.keys() if k is not None):
        if lid is None or lid == mtp_lid:
            continue
        # pick a layer that clearly has MoE experts (not layer 0/1/2 dense)
        keys = by_layer[lid]
        if any(".mlp.experts." in k for k in keys):
            ref_lid = lid
            break
    if ref_lid is not None:
        ref_sub: Counter[str] = Counter()
        for k in by_layer[ref_lid]:
            tail = k[len(f"model.layers.{ref_lid}."):]
            tail = re.sub(r"experts\.\d+\.", "experts.<*>.", tail)
            ref_sub[tail] += 1
        only_in_mtp = sorted(set(sub_hist) - set(ref_sub))
        only_in_ref = sorted(set(ref_sub) - set(sub_hist))
        print(f"\n[diag] === MTP vs. reference MoE layer {ref_lid} diff ===")
        print(f"[diag] keys unique to MTP ({len(only_in_mtp)}):")
        for s in only_in_mtp:
            print(f"[diag]   + model.layers.<MTP>.{s}")
        print(f"[diag] keys unique to reference layer {ref_lid} "
              f"({len(only_in_ref)}):")
        for s in only_in_ref:
            print(f"[diag]   - model.layers.<ref>.{s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
