# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-
"""
Offline SmoothQuant weight conversion.

Usage:
    python convert_smooth_weights.py \
        --model-path /path/to/model \
        --smooth-stats /path/to/smooth_stats.json \
        --save-path /path/to/output \
        [--alpha-qk 0.6] [--alpha-vo 0.5] \
        [--use-ema]

Smooth stats JSON format (from run_vllm_smooth.py):
    {
      "<attn_layer>.q":        {"absmax": [...], "ema": [...], "call_count": N},
      "<attn_layer>.k":        {...},
      "<attn_layer>.attn_out": {...},
      ...
    }
  where <attn_layer> is the vLLM Attention layer name,
  e.g. "model.layers.0.self_attn.attn"
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Smooth helpers / appliers — re-exported from the angelslim
# transform/smooth package so this CLI script becomes a thin orchestration
# layer.  Imported here (before parse_args) because PREDEFINED_KEY_MAPS is
# referenced by argparse choices.
from angelslim.compressor.transform.smooth.convert import (
    PREDEFINED_KEY_MAPS,
    apply_down_proj_smooth,
    apply_down_proj_smooth_from_search,
    apply_qk_smooth,
    apply_vo_smooth,
    snapshot_attn_output_before,
    snapshot_mlp_outputs_before,
    verify_attn_output_diff,
    verify_mlp_output_diff,
)
from angelslim.compressor.transform.smooth.core import load_smooth_stats

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Offline SmoothQuant weight conversion")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the pretrained model directory (safetensors format)",
    )
    parser.add_argument(
        "--smooth-stats",
        type=str,
        default=None,
        help="Path to smooth stats JSON file. If not set, auto-detected from output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory containing smooth_stats.json and smooth_alpha_search.json "
        "Used for auto-detection when --smooth-stats is not set.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Directory to save the transformed model",
    )
    parser.add_argument(
        "--alpha-qk",
        type=float,
        default=0.6,
        help="Alpha for QK smooth weight formula: smooth = k_absmax^alpha",
    )
    parser.add_argument(
        "--alpha-vo",
        type=float,
        default=0.5,
        help="Alpha for VO smooth: smooth = attn_out^alpha / o_proj_max^(1-alpha)",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA stats instead of absmax for smooth weight computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on (default: cpu). Use 'cuda' or 'auto' for GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Dtype to load the model in (default: auto)",
    )
    parser.add_argument(
        "--smooth-qk",
        action="store_true",
        help="Apply QK smooth (q_proj/k_proj weight transformation). "
        "Requires attn.q and attn.k stats in smooth-stats.",
    )
    parser.add_argument(
        "--smooth-vo",
        action="store_true",
        help="Apply VO smooth (v_proj/o_proj weight transformation). "
        "Requires attn_out stats in smooth-stats.",
    )
    parser.add_argument(
        "--smooth-down",
        action="store_true",
        help="Apply down_proj smooth (down_proj/up_proj weight transformation). "
        "Requires down_proj stats in smooth-stats.",
    )
    parser.add_argument(
        "--alpha-down",
        type=float,
        default=0.5,
        help="Alpha for down_proj smooth: "
        "smooth = absmax^alpha / weight_absmax^(1-alpha) (default: 0.5)",
    )
    parser.add_argument(
        "--enable-alpha-search",
        action="store_true",
        help="Enable alpha search result loading. When set, auto-detects "
        "smooth_alpha_search.json from output-dir. Per-key: if key has search "
        "result, use precomputed smooth_weight; otherwise use --alpha-down.",
    )
    parser.add_argument(
        "--alpha-search-results",
        type=str,
        default=None,
        help="Path to smooth_alpha_search.json (output of alpha grid search). "
        "When provided, down_proj smooth uses pre-computed smooth_weight "
        "from search results instead of computing from smooth-stats + fixed alpha.",
    )
    parser.add_argument(
        "--down-workers",
        type=int,
        default=1,
        help="Number of parallel threads for down_proj smooth (default: 128). ",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=16,
        help="Number of parallel threads for writing safetensors shards (default: 16). "
        "Increase for fast NVMe/parallel NFS; each thread writes one shard file independently.",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=8.0,
        help="Target size (GiB) of each safetensors shard file (default: 8.0).",
    )
    parser.add_argument(
        "--patch-mlp-layers",
        type=int,
        default=3,
        help="Number of MLP Linear layers to patch for logit loss verification",
    )
    parser.add_argument(
        "--verify-seq-len",
        type=int,
        default=8,
        help="Sequence length for smooth verification forward passes (default: 8)",
    )

    # Architecture key map
    parser.add_argument(
        "--arch",
        type=str,
        default="hy_v3",
        choices=list(PREDEFINED_KEY_MAPS.keys()) + ["custom"],
        help="Architecture key map to use (default: hy_v3). "
        "Use 'custom' with --key-map-file for user-defined mapping.",
    )
    parser.add_argument(
        "--key-map-file",
        type=str,
        default=None,
        help="Custom key_map JSON file path (used when --arch=custom).",
    )

    # YAML config support
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML config file path. Values override argparse defaults; "
        "explicit CLI flags still take final precedence.",
    )

    args = parser.parse_args()

    # Lazy-import _yaml_args (located in tools/). Done here instead of at
    # module top so flake8 doesn't trip on a sys.path mutation between
    # imports.
    import sys

    _tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from _yaml_args import apply_yaml_config

    apply_yaml_config(parser, args)

    # Auto-resolve smooth_stats path from output_dir
    if args.smooth_stats is None and args.output_dir is not None:
        args.smooth_stats = os.path.join(args.output_dir, "smooth_stats.json")

    if args.smooth_stats is None:
        parser.error("Either --smooth-stats or --output-dir must be specified.")

    # Auto-resolve alpha_search_results from output_dir or enable_alpha_search
    if args.alpha_search_results is None and args.enable_alpha_search:
        if args.output_dir is not None:
            candidate = os.path.join(args.output_dir, "smooth_alpha_search.json")
            if os.path.isfile(candidate):
                args.alpha_search_results = candidate

    return args


# ---------------------------------------------------------------------------
# Post-save: append source keys that save_pretrained dropped (e.g. MTP layers)
# ---------------------------------------------------------------------------


def _read_weight_map(model_dir: str) -> tuple[dict, bool]:
    """
    Read the safetensors weight_map of a model directory.

    Returns ``(weight_map, is_sharded)`` where ``weight_map`` maps
    ``tensor_key -> shard_filename``. Handles both the sharded layout
    (``model.safetensors.index.json``) and the single-file layout
    (``model.safetensors``). Returns ``({}, False)`` if neither is found.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        return dict(index["weight_map"]), True

    if os.path.exists(single_path):
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as f_st:
            return {k: "model.safetensors" for k in f_st.keys()}, False

    return {}, False


def append_missing_keys_from_source(
    model_path: str, save_path: str, shard_size_gb: float = 8.0
) -> None:
    """
    Compare the ORIGINAL checkpoint's index with the just-saved index and
    append any keys that ``save_pretrained`` dropped back into ``save_path``.

    Typical case: Multi-Token-Prediction (MTP) layers (e.g.
    ``model.layers.80.*`` for HY_V3) that the HF architecture never registers,
    so they are absent from ``model.state_dict()`` and therefore not written by
    ``save_pretrained``.

    The missing tensors are loaded from the source shards (each containing
    source shard is opened once) into memory, then written into one or more
    ``model-appended-from-source-XYZ.safetensors`` files inside ``save_path``.
    When the total appended size exceeds ``shard_size_gb``, the output is split
    across multiple files so no single appended shard exceeds that budget. The
    saved ``model.safetensors.index.json`` is then updated to reference them
    (an index is synthesized if the save produced a single-file layout).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    # 1. Read source & saved weight maps
    src_weight_map, _ = _read_weight_map(model_path)
    if not src_weight_map:
        print("  [Append][WARNING] No safetensors found in model_path, skipping")
        return

    saved_weight_map, saved_is_sharded = _read_weight_map(save_path)
    if not saved_weight_map:
        print("  [Append][WARNING] No safetensors found in save_path, skipping")
        return

    # 2. Compute keys present in source but missing from the saved model
    missing_keys = set(src_weight_map.keys()) - set(saved_weight_map.keys())
    if not missing_keys:
        print("  [Append] No missing keys; saved model matches source key set.")
        return

    print(
        f"  [Append] {len(missing_keys)} key(s) in source but missing from saved "
        f"model (e.g. MTP layers), appending:"
    )
    for k in sorted(missing_keys)[:8]:
        print(f"    {k}")
    if len(missing_keys) > 8:
        print(f"    ... total {len(missing_keys)}")

    # 3. Load the missing tensors from the source shards (one open per shard).
    shard_to_keys: dict[str, list[str]] = {}
    for k in missing_keys:
        shard_to_keys.setdefault(src_weight_map[k], []).append(k)

    missing_tensors: dict[str, torch.Tensor] = {}
    for shard_file, keys in sorted(shard_to_keys.items()):
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f_st:
            for k in keys:
                t = f_st.get_tensor(k)
                if not t.is_contiguous():
                    t = t.contiguous()
                missing_tensors[k] = t
        print(f"    Loaded {len(keys)} missing key(s) from {shard_file}")

    # 4. Split the missing tensors into size-bounded groups, then write one
    #    safetensors file per group (a single tensor larger than the budget
    #    still goes into its own file on its own).
    size_budget = int(shard_size_gb * 1024**3)
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for k in sorted(missing_tensors.keys()):
        t = missing_tensors[k]
        t_bytes = t.numel() * t.element_size()
        if current and current_bytes + t_bytes > size_budget:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(k)
        current_bytes += t_bytes
    if current:
        groups.append(current)

    n_groups = len(groups)
    appended_weight_map: dict[str, str] = {}
    appended_bytes = 0
    for gi, keys in enumerate(groups):
        # 1-based, zero-padded index in the classic HF sharding style.
        out_shard = f"model-appended-{gi + 1:05d}-of-{n_groups:05d}.safetensors"
        group_tensors = {k: missing_tensors[k] for k in keys}
        save_file(group_tensors, os.path.join(save_path, out_shard), metadata={"format": "pt"})
        group_bytes = sum(t.numel() * t.element_size() for t in group_tensors.values())
        appended_bytes += group_bytes
        for k in keys:
            appended_weight_map[k] = out_shard
        print(f"    Wrote {len(keys)} key(s) ({group_bytes / 1024**3:.2f} GiB) -> {out_shard}")

    print(
        f"  [Append] Wrote {len(appended_weight_map)} tensor(s) "
        f"({appended_bytes / 1024**3:.2f} GiB) across {n_groups} appended file(s)"
    )

    # 5. Update / synthesize model.safetensors.index.json so the appended keys
    #    are discoverable. weight_map must reference every tensor file.
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    if saved_is_sharded:
        with open(index_path, "r") as f:
            index = json.load(f)
    else:
        # Single-file save -> build an index that maps existing keys to
        # model.safetensors plus the new keys to the appended shards. Compute
        # the existing total_size from the single file.
        single_path = os.path.join(save_path, "model.safetensors")
        existing_bytes = 0
        with safe_open(single_path, framework="pt", device="cpu") as f_st:
            for k in f_st.keys():
                t = f_st.get_slice(k)
                shape = t.get_shape()
                n = 1
                for d in shape:
                    n *= d
                existing_bytes += (
                    n
                    * torch.empty(
                        0, dtype=getattr(torch, t.get_dtype().lower(), torch.float32)
                    ).element_size()
                )
        index = {
            "metadata": {"total_size": existing_bytes},
            "weight_map": dict(saved_weight_map),
        }

    index["weight_map"].update(appended_weight_map)
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = int(index["metadata"].get("total_size", 0)) + appended_bytes

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  [Append] Updated index: {index_path} ({len(index['weight_map'])} keys)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print("=" * 70)
    print("[Config]")
    print(f"  model-path:      {args.model_path}")
    print(f"  smooth-stats:    {args.smooth_stats}")
    print(f"  save-path:       {args.save_path}")
    print(f"  alpha-qk:        {args.alpha_qk}")
    print(f"  alpha-vo:        {args.alpha_vo}")
    print(f"  use-ema:         {args.use_ema}")
    print(f"  device:          {args.device}")
    print(f"  dtype:           {args.dtype}")
    print(f"  smooth-qk:       {args.smooth_qk}")
    print(f"  smooth-vo:       {args.smooth_vo}")
    print(f"  smooth-down:     {args.smooth_down}")
    print(f"  alpha-down:      {args.alpha_down}")
    print(f"  down-workers:    {args.down_workers}")
    print(f"  save-workers:    {args.save_workers}")
    print(f"  shard-size-gb:   {args.shard_size_gb}")
    print(f"  patch-mlp-layers:{args.patch_mlp_layers}")
    print(f"  verify-seq-len:  {args.verify_seq_len}")
    print(f"  alpha-search:    {args.alpha_search_results or '(none, use fixed alpha)'}")
    print("=" * 70)

    # Resolve key_map
    if args.arch == "custom" and args.key_map_file:
        with open(args.key_map_file, "r") as f:
            km = json.load(f)
        print(f"  [KeyMap] Loaded custom key_map from {args.key_map_file}")
    else:
        km = PREDEFINED_KEY_MAPS[args.arch]
        print(f"  [KeyMap] Using predefined key_map: {args.arch}")

    if not args.smooth_qk and not args.smooth_vo and not args.smooth_down:
        print(
            "\n[WARNING] Neither --smooth-qk, --smooth-vo, nor --smooth-down is set. "
            "No weight transformation will be applied. "
            "Use --smooth-qk / --smooth-vo / --smooth-down to enable conversion."
        )
        return

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Loading model from {args.model_path} ...")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Model loaded: {type(model).__name__}")

    # Load tokenizer (optional, for saving alongside model)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print(f"  Tokenizer loaded: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"  [WARNING] Could not load tokenizer: {e}")

    # Infer head_dim from model config
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        num_heads = getattr(cfg, "num_attention_heads", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        if num_heads and hidden_size:
            head_dim = hidden_size // num_heads
    print(f"  head_dim inferred: {head_dim}")

    # ------------------------------------------------------------------
    # Step 2: Load smooth stats
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Loading smooth stats from {args.smooth_stats} ...")
    smooth_stats = load_smooth_stats(args.smooth_stats, use_ema=args.use_ema)
    print(f"  Loaded {len(smooth_stats)} stat entries")
    sample_keys = list(smooth_stats.keys())[:6]
    for k in sample_keys:
        sc = smooth_stats[k]["scale"]
        print(f"    {k!r:70s}  shape={list(sc.shape) if sc is not None else None}")
    if len(smooth_stats) > 6:
        print(f"    ... and {len(smooth_stats) - 6} more")

    # ------------------------------------------------------------------
    # Step 2.5: Record pre-transform attn output (for verification)
    # ------------------------------------------------------------------
    verify_snap = snapshot_attn_output_before(
        model, smooth_stats, seq_len=args.verify_seq_len, seed=42, km=km
    )

    # ------------------------------------------------------------------
    # Step 2.6: Record pre-transform MLP module output (for verification)
    # ------------------------------------------------------------------
    mlp_snap = snapshot_mlp_outputs_before(
        model,
        patch_n=args.patch_mlp_layers,
        seq_len=args.verify_seq_len,
        seed=42,
        km=km,
    )

    # ------------------------------------------------------------------
    # Step 3: Apply smooth transforms
    # ------------------------------------------------------------------
    print("\n[Step 3] Applying offline smooth transforms ...")

    # QK smooth
    if args.smooth_qk:
        has_qk = any(k.endswith(km["stat_k"]) for k in smooth_stats) and any(
            k.endswith(km["stat_q"]) for k in smooth_stats
        )
        if has_qk:
            apply_qk_smooth(model, smooth_stats, alpha=args.alpha_qk, head_dim=head_dim, km=km)
        else:
            print("  [SKIP] QK smooth: no attn.k / attn.q stats found in smooth-stats")
    else:
        print("  [SKIP] QK smooth: --smooth-qk not set")

    # VO smooth
    if args.smooth_vo:
        has_vo = any(k.endswith(km["stat_attn_out"]) for k in smooth_stats)
        if has_vo:
            apply_vo_smooth(model, smooth_stats, alpha=args.alpha_vo, head_dim=head_dim, km=km)
        else:
            print("  [SKIP] VO smooth: no attn_out stats found in smooth-stats")
    else:
        print("  [SKIP] VO smooth: --smooth-vo not set")

    # Down proj smooth (two-step: alpha search first, then fixed alpha fallback)
    if args.smooth_down:
        search_processed_keys = set()

        # Step 1: If alpha search results provided, apply them first
        if args.alpha_search_results:
            print(f"\n[Down Smooth] Using alpha search results: {args.alpha_search_results}")
            with open(args.alpha_search_results, "r") as f:
                alpha_search = json.load(f)
            search_processed_keys = apply_down_proj_smooth_from_search(
                model, alpha_search, num_workers=args.down_workers, km=km
            )

        # Step 2: Fallback -- for keys NOT in alpha search, use fixed alpha
        import re as _re

        _DOWN_PATTERNS = [_re.compile(p) for p in km["down_patterns"]]
        has_down = any(any(pat.search(k) for pat in _DOWN_PATTERNS) for k in smooth_stats)
        if has_down:
            apply_down_proj_smooth(
                model,
                smooth_stats,
                alpha=args.alpha_down,
                num_workers=args.down_workers,
                exclude_keys=search_processed_keys,
                km=km,
            )
        elif not search_processed_keys:
            print("  [SKIP] Down smooth: no down_proj stats found in smooth-stats")
    else:
        print("  [SKIP] Down smooth: --smooth-down not set")

    # ------------------------------------------------------------------
    # Step 3.5: Verify attn output diff
    # ------------------------------------------------------------------
    verify_attn_output_diff(verify_snap, model, atol=1e-3, rtol=1e-3, km=km)

    # ------------------------------------------------------------------
    # Step 3.6: Verify MLP output diff
    # ------------------------------------------------------------------
    verify_mlp_output_diff(mlp_snap, atol=1e-2, rtol=1e-2)

    # ------------------------------------------------------------------
    # Step 4: Save transformed model
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Saving transformed model to {args.save_path} ...")
    os.makedirs(args.save_path, exist_ok=True)

    # Save with transformers' standard save_pretrained. It serializes to
    # safetensors, shards according to max_shard_size, and writes
    # model.safetensors.index.json + config.json automatically. By default
    # (save_original_format=True) it also reverts the load-time weight
    # conversion, so fused params like experts.gate_up_proj are split back to
    # the original per-expert checkpoint layout.
    model.save_pretrained(
        args.save_path,
        max_shard_size=f"{args.shard_size_gb}GB",
        safe_serialization=True,
    )
    print(f"  [Save] Model saved via save_pretrained to {args.save_path}")

    # ------------------------------------------------------------------
    # Step 4.1: Append keys present in the source but dropped on save
    # ------------------------------------------------------------------
    # save_pretrained only writes tensors that exist in model.state_dict().
    # Weights the HF architecture never registers (e.g. MTP layers such as
    # model.layers.80.* for HY_V3) are therefore absent from the saved model.
    # Compare the source vs saved safetensors index and append the gap.
    print("\n[Step 4.1] Appending source keys missing from the saved model ...")
    append_missing_keys_from_source(
        args.model_path, args.save_path, shard_size_gb=args.shard_size_gb
    )

    if tokenizer is not None:
        tokenizer.save_pretrained(args.save_path)
    print(f"  Done. Saved to {args.save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
