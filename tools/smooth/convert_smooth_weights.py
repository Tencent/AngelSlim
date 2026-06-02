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
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# MTP / extra weight recovery: load tensors dropped by AutoModelForCausalLM
# ---------------------------------------------------------------------------


def load_missing_tensors(model_path: str, loaded_keys: set) -> dict:
    """
    Load weights that were NOT loaded by AutoModelForCausalLM (e.g. MTP layers).

    Typical case: models with Multi-Token Prediction heads whose extra layers
    are not registered in the HF architecture and get dropped with
    "Some weights of the model checkpoint were not used" warning.

    Args:
        model_path:   original model directory path
        loaded_keys:  model.state_dict().keys() (keys already loaded by HF)

    Returns:
        dict[str, torch.Tensor] -- missing key -> CPU tensor mapping;
        empty dict if nothing is missing.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("  [WARNING] safetensors not available, cannot supplement missing weights")
        return {}

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f_st:
            weight_map = {k: "model.safetensors" for k in f_st.keys()}
    else:
        print("  [WARNING] No safetensors index/file found in model_path, skipping")
        return {}

    missing_keys = set(weight_map.keys()) - loaded_keys
    if not missing_keys:
        print("  [MTP] No missing keys, model parameters are complete.")
        return {}

    print(
        f"  [MTP] Found {len(missing_keys)} weight keys not loaded by HF "
        f"(e.g. MTP layers), supplementing:"
    )
    for k in sorted(missing_keys)[:8]:
        print(f"    {k}")
    if len(missing_keys) > 8:
        print(f"    ... total {len(missing_keys)}")

    # Group by shard file, open each shard only once
    shard_to_keys: dict[str, list[str]] = {}
    for k in missing_keys:
        shard_to_keys.setdefault(weight_map[k], []).append(k)

    missing_tensors: dict[str, torch.Tensor] = {}
    for shard_file, keys in sorted(shard_to_keys.items()):
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f_st:
            for k in keys:
                t = f_st.get_tensor(k)
                if not t.is_contiguous():
                    t = t.contiguous()
                missing_tensors[k] = t
        print(f"    Loaded {len(keys)} missing keys from {shard_file}")

    return missing_tensors


# ---------------------------------------------------------------------------
# Parallel safetensors save
# ---------------------------------------------------------------------------


def save_model_parallel(
    model: torch.nn.Module,
    save_path: str,
    shard_size_gb: float = 4.0,
    num_workers: int = 4,
    extra_state_dict: dict | None = None,
) -> None:
    """
    Save model weights as safetensors with parallel shard writing.

    Much faster than save_pretrained (which serializes sequentially).

    Flow:
      1. Collect state_dict, split into shards by shard_size_gb
      2. ThreadPoolExecutor writes shards concurrently
      3. Generate model.safetensors.index.json
      4. Single shard -> write model.safetensors directly (no index)

    Args:
        model:             transformed model
        save_path:         output directory (must exist)
        shard_size_gb:     target size per shard (GiB)
        num_workers:       concurrent writer threads
        extra_state_dict:  extra weights to merge (e.g. MTP layers from
                           load_missing_tensors())
    """
    try:
        from safetensors.torch import save_file as st_save_file
    except ImportError:
        raise ImportError(
            "safetensors is required for parallel saving. " "Install with: pip install safetensors"
        )

    shard_size_bytes = int(shard_size_gb * 1024**3)

    # 1. Collect state_dict (all contiguous cpu tensors)
    print("  [Save] Collecting state_dict ...")
    raw_state_dict = model.state_dict()
    state_dict: dict[str, torch.Tensor] = {}
    for name, t in raw_state_dict.items():
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        state_dict[name] = t

    # Merge extra_state_dict (e.g. MTP layers)
    if extra_state_dict:
        n_extra = 0
        for name, t in extra_state_dict.items():
            if name in state_dict:
                print(f"  [Save][WARNING] key {name!r} already in state_dict, extra ignored")
                continue
            if t.device.type != "cpu":
                t = t.cpu()
            if not t.is_contiguous():
                t = t.contiguous()
            state_dict[name] = t
            n_extra += 1
        print(f"  [Save] Supplemented {n_extra} extra weights (MTP etc.) into state_dict")

    total_bytes = sum(t.nbytes for t in state_dict.values())
    total_gb = total_bytes / 1024**3
    n_shards = max(1, math.ceil(total_bytes / shard_size_bytes))
    print(
        f"  [Save] Total params: {total_gb:.2f} GiB, splitting into {n_shards} shards "
        f"(each <= {shard_size_gb} GiB)"
    )

    # 2. Assign tensors to shards (greedy, no cross-shard splitting)
    shard_dicts: list[dict[str, torch.Tensor]] = []
    cur_shard: dict[str, torch.Tensor] = {}
    cur_bytes = 0

    for name, tensor in state_dict.items():
        nb = tensor.nbytes
        if cur_bytes + nb > shard_size_bytes and cur_shard:
            shard_dicts.append(cur_shard)
            cur_shard = {}
            cur_bytes = 0
        cur_shard[name] = tensor
        cur_bytes += nb

    if cur_shard:
        shard_dicts.append(cur_shard)

    n_shards = len(shard_dicts)

    # 3. Write shard files concurrently
    if n_shards == 1:
        out_file = os.path.join(save_path, "model.safetensors")
        print(f"  [Save] Single shard -> {out_file}")
        st_save_file(shard_dicts[0], out_file)
        print(f"  [Save] Done: {out_file}")
        return

    pad = len(str(n_shards))
    shard_filenames = [
        f"model-{str(i + 1).zfill(pad)}-of-{str(n_shards).zfill(pad)}.safetensors"
        for i in range(n_shards)
    ]

    def _write_shard(idx: int) -> tuple[int, str]:
        fname = shard_filenames[idx]
        out_file = os.path.join(save_path, fname)
        shard_bytes = sum(t.nbytes for t in shard_dicts[idx].values())
        print(
            f"  [Save] [{idx+1}/{n_shards}] Writing {fname}  "
            f"({shard_bytes / 1024**3:.2f} GiB, "
            f"{len(shard_dicts[idx])} tensors) ..."
        )
        st_save_file(shard_dicts[idx], out_file)
        return idx, fname

    effective_workers = min(num_workers, n_shards)
    print(f"  [Save] Writing {n_shards} shards concurrently, workers={effective_workers}")

    results_list: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_write_shard, i): i for i in range(n_shards)}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                raise RuntimeError(f"Shard {futures[fut]} write failed: {exc}") from exc
            results_list.append(fut.result())

    results_list.sort()

    # 4. Generate model.safetensors.index.json
    weight_map: dict[str, str] = {}
    for idx, fname in results_list:
        for key in shard_dicts[idx]:
            weight_map[key] = fname

    index = {
        "metadata": {"total_size": str(total_bytes)},
        "weight_map": weight_map,
    }
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  [Save] Index written: {index_path}")
    print(f"  [Save] All {n_shards} shards written successfully")


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

    # Supplement MTP / extra weights dropped by AutoModelForCausalLM
    print("\n[Step 4.1] Detecting and supplementing extra weights (MTP etc.)...")
    loaded_keys = set(model.state_dict().keys())
    extra_state_dict = load_missing_tensors(args.model_path, loaded_keys)

    # Parallel safetensors shard writing
    save_model_parallel(
        model,
        args.save_path,
        shard_size_gb=args.shard_size_gb,
        num_workers=args.save_workers,
        extra_state_dict=extra_state_dict if extra_state_dict else None,
    )

    # Copy config / tokenizer / non-weight files from original model directory
    _NON_WEIGHT_EXTS = {".json", ".txt", ".py", ".model", ".tiktoken"}
    _SKIP_FILES = {"model.safetensors.index.json"}
    copied = []
    for fname in os.listdir(args.model_path):
        if fname in _SKIP_FILES:
            continue
        if fname.startswith("model-") and fname.endswith(".safetensors"):
            continue
        if fname == "model.safetensors":
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in _NON_WEIGHT_EXTS:
            src = os.path.join(args.model_path, fname)
            dst = os.path.join(args.save_path, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(fname)
    if copied:
        print(f"  [Save] Copied config files: {copied}")

    if tokenizer is not None:
        tokenizer.save_pretrained(args.save_path)
    print(f"  Done. Saved to {args.save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
