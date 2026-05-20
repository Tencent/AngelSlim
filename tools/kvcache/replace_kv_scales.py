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

"""
Replace kv_cache_scales.safetensors with per-head scales from
kv_cache_tuned_scales_per_head.json.

Supports two granularities:

* ``per-tensor``  – one scalar scale per layer-slot.
* ``per-head``    – one scale per KV head.

Usage
-----
Per-head (default)::

    python tools/kvcache/replace_kv_scales.py \\
        --granularity per-head \\
        --json   /path/to/kv_cache_tuned_scales_per_head.json \\
        --src    /path/to/model_dir/kv_cache_scales.safetensors \\
        --output /path/to/output_dir/kv_cache_scales.safetensors

Per-tensor::

    python tools/kvcache/replace_kv_scales.py \\
        --granularity per-tensor \\
        --json   /path/to/kv_cache_tuned_scales.json \\
        --src    /path/to/model_dir/kv_cache_scales.safetensors \\
        --output /path/to/output_dir/kv_cache_scales.safetensors

If --output is omitted the source file is overwritten in-place (a .bak
backup is created automatically).
"""

import argparse
import json
import os
import re
import shutil

import safetensors.torch as st
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace kv_cache_scales.safetensors with calibrated scales "
        "(supports per-tensor and per-head granularities).",
    )
    parser.add_argument(
        "--granularity",
        choices=["per-tensor", "per-head"],
        default="per-head",
        help="Calibration granularity that matches the JSON layout. "
        "'per-tensor' expects scalar scales per layer-slot; "
        "'per-head' expects one scale per KV head (default: per-head).",
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the tuned-scales JSON file "
        "(per-tensor or per-head depending on --granularity).",
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to the existing kv_cache_scales.safetensors to be updated.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the new safetensors file.  "
        "Defaults to overwriting --src (a .bak backup is kept).",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=16,
        help="TP size used during calibration (used only for legacy per-head JSON "
        "files that still contain replicated heads; default: 16).",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Actual number of KV heads in the model (per-head only; default: 8).",
    )
    return parser.parse_args()


def load_json_scales_pertensor(json_path: str, ref_tensors: dict) -> dict:
    """
    Load per-tensor scalar scales from JSON and return a dict mapping each
    key to a 1-D torch.Tensor with the same dtype as the corresponding
    tensor in ``ref_tensors`` (falling back to float32 if the key is new).

    JSON keys are expected to be of the form
    ``"model.layers.N.self_attn.{k,v}_cache.scale"`` with float values.
    """
    with open(json_path) as f:
        raw = json.load(f)

    out: dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        ref = ref_tensors.get(key)
        dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float32
        out[key] = torch.tensor([val], dtype=dtype)
    return out


def load_json_scales_perhead(json_path: str, tp_size: int, num_kv_heads: int) -> dict:
    """
    Load per-head scales from JSON and return a dict mapping
    ``"model.layers.N.self_attn.{k,v}_cache.scale"`` to a float32 torch.Tensor
    of shape ``(num_kv_heads,)``.

    Two JSON layouts are supported:

    1. **New layout** (K/V-split calibration): each slot has exactly
       ``num_kv_heads`` entries (head_0 … head_{H-1}).  We read them in
       order.

    2. **Legacy layout** (pre-split calibration): each slot has ``tp_size``
       replicated entries where adjacent heads inside each replication group
       are identical.  We de-duplicate by picking the primary replica of
       each global head (index ``h * replication``).
    """
    with open(json_path) as f:
        raw = json.load(f)

    # Group by (layer_idx, kv_slot): layer_key -> {head_idx: scale}
    # JSON key format: "model.layers.N.self_attn.{k,v}_cache.head_H.scale"
    pattern = re.compile(r"^(model\.layers\.\d+\.self_attn\.[kv]_cache)\.head_(\d+)\.scale$")
    groups: dict[str, dict[int, float]] = {}
    for key, val in raw.items():
        m = pattern.match(key)
        if not m:
            print(f"  WARNING: unrecognised key format, skipping: {key}")
            continue
        base = m.group(1)  # e.g. "model.layers.0.self_attn.k_cache"
        head_idx = int(m.group(2))
        groups.setdefault(base, {})[head_idx] = val

    out: dict[str, torch.Tensor] = {}
    for base, head_dict in groups.items():
        n_entries = len(head_dict)

        if n_entries == num_kv_heads:
            # New layout: one entry per real KV head.
            scales = [head_dict[h] for h in range(num_kv_heads)]
        elif n_entries == tp_size and tp_size % num_kv_heads == 0:
            # Legacy layout: deduplicate replicated heads.
            replication = tp_size // num_kv_heads
            scales = [head_dict[h * replication] for h in range(num_kv_heads)]
            print(
                f"  NOTE: {base} has {n_entries} heads in JSON (legacy "
                f"layout, replication={replication}); de-duplicating."
            )
        else:
            print(
                f"  WARNING: {base} has {n_entries} heads in JSON "
                f"(expected {num_kv_heads} new-layout or {tp_size} legacy), "
                f"skipping."
            )
            continue

        # Save as float32 tensor (the safetensors file uses bfloat16 but we
        # write float32 for precision; the loader will cast as needed).
        out[f"{base}.scale"] = torch.tensor(scales, dtype=torch.float32)

    return out


def merge_and_save(existing: dict, new_scales: dict, src_path: str, output_path: str) -> dict:
    """
    Merge ``new_scales`` into ``existing`` (replacing matching keys, adding
    new ones, warning on shape mismatch / missing keys), then save to
    ``output_path``.  If the output overwrites the source, a .bak backup
    is created.  Returns the merged dict.
    """
    updated = dict(existing)
    replaced = 0
    for key, tensor in new_scales.items():
        if key not in updated:
            print(f"  WARNING: key not found in safetensors, will be added: {key}")
        else:
            old_shape = updated[key].shape
            if old_shape != tensor.shape:
                print(
                    f"  WARNING: shape mismatch for {key}: "
                    f"existing={old_shape}, new={tensor.shape}. Replacing anyway."
                )
        updated[key] = tensor
        replaced += 1
    print(f"  Replaced/added {replaced} keys.")

    # Check for keys in the original file that were NOT updated.
    missing = [k for k in existing if k not in new_scales]
    if missing:
        print(
            f"  NOTE: {len(missing)} keys in original file were NOT updated "
            f"(no corresponding entry in JSON): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    if output_path == src_path and os.path.exists(src_path):
        bak = src_path + ".bak"
        shutil.copy2(src_path, bak)
        print(f"\nBackup saved to: {bak}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    st.save_file(updated, output_path)
    print(f"Saved updated safetensors to: {output_path}")
    return updated


def update_config_json(output_path: str, granularity: str) -> None:
    """
    Write/overwrite ``attn_quant_config`` in the ``config.json`` next to
    ``output_path`` according to the requested calibration granularity.
    """
    cfg_granularity = "per_tensor" if granularity == "per-tensor" else "per_head"
    config_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["attn_quant_config"] = {
        "kv_cache_quant": {
            "dtype": "fp8_e4m3",
            "k_quant": {"scheme": "static", "granularity": cfg_granularity},
            "v_quant": {"scheme": "static", "granularity": cfg_granularity},
        },
        "q_quant": {"dtype": "fp8_e4m3", "scheme": "dynamic", "granularity": "per_token_per_head"},
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated config.json: {config_path}")


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load existing safetensors (needed early for per-tensor dtype     #
    #    inference)                                                       #
    # ------------------------------------------------------------------ #
    print(f"Loading existing safetensors from: {args.src}")
    existing = st.load_file(args.src)
    print(f"  Existing keys: {len(existing)}")

    # ------------------------------------------------------------------ #
    # 2. Load scales from JSON (layout depends on --granularity)          #
    # ------------------------------------------------------------------ #
    print(f"\nLoading {args.granularity} scales from: {args.json}")
    if args.granularity == "per-tensor":
        new_scales = load_json_scales_pertensor(args.json, existing)
        print(
            f"  Loaded {len(new_scales)} layer-slot entries "
            f"(each is a scalar tensor of shape [1])"
        )
    else:
        print(f"  num_kv_heads={args.num_kv_heads} (legacy-fallback tp_size={args.tp_size})")
        new_scales = load_json_scales_perhead(args.json, args.tp_size, args.num_kv_heads)
        print(
            f"  Loaded {len(new_scales)} layer-slot entries "
            f"(each is a tensor of shape [{args.num_kv_heads}])"
        )

    # ------------------------------------------------------------------ #
    # 3. Merge and save                                                   #
    # ------------------------------------------------------------------ #
    output_path = args.output if args.output else args.src
    merge_and_save(existing, new_scales, args.src, output_path)

    # ------------------------------------------------------------------ #
    # 4. Update config.json with attn_quant_config                        #
    # ------------------------------------------------------------------ #
    update_config_json(output_path, args.granularity)

    # ------------------------------------------------------------------ #
    # 5. Quick sanity check                                               #
    # ------------------------------------------------------------------ #
    verify = st.load_file(output_path)
    sample_key = next(iter(new_scales))
    print(f"\nSanity check – {sample_key}:")
    print(f"  shape : {verify[sample_key].shape}")
    print(f"  values: {verify[sample_key]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
