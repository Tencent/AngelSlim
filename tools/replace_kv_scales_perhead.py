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

Background
----------
The JSON is produced by ``tools/run_kvcache_calibrate.py`` in per-head mode.
After the K/V-split optimisation (see ``vllm_calibrate_utils.py``), each
layer's K/V slot has exactly ``num_kv_heads`` entries in the JSON – one per
real KV head.  We simply load them in head-index order and write a float32
tensor of shape ``(num_kv_heads,)`` into the safetensors file.

(Legacy JSON files produced *before* the K/V-split optimisation contained
``tp_size`` replicated entries per slot.  If such a file is encountered and
``len(head_dict) == tp_size`` with ``tp_size > num_kv_heads``, we
automatically de-duplicate by taking the primary replica.)

Usage
-----
    python tools/replace_kv_scales_perhead.py \
        --json   /path/to/kv_cache_tuned_scales_per_head.json \
        --src    /path/to/model_dir/kv_cache_scales.safetensors \
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
        description="Replace kv_cache_scales.safetensors with calibrated per-head scales."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to kv_cache_tuned_scales_per_head.json.",
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
        help="TP size used during calibration (used only for legacy JSON "
        "files that still contain replicated heads; default: 16).",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Actual number of KV heads in the model (default: 8).",
    )
    return parser.parse_args()


def load_json_scales(json_path: str, tp_size: int, num_kv_heads: int) -> dict:
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


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load and de-duplicate per-head scales from JSON                  #
    # ------------------------------------------------------------------ #
    print(f"Loading per-head scales from: {args.json}")
    print(f"  num_kv_heads={args.num_kv_heads} " f"(legacy-fallback tp_size={args.tp_size})")
    new_scales = load_json_scales(args.json, args.tp_size, args.num_kv_heads)
    print(
        f"  Loaded {len(new_scales)} layer-slot entries "
        f"(each is a tensor of shape [{args.num_kv_heads}])"
    )

    # ------------------------------------------------------------------ #
    # 2. Load existing safetensors                                        #
    # ------------------------------------------------------------------ #
    print(f"\nLoading existing safetensors from: {args.src}")
    existing = st.load_file(args.src)
    print(f"  Existing keys: {len(existing)}")

    # ------------------------------------------------------------------ #
    # 3. Merge: replace matching keys; warn on missing ones               #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 4. Save                                                             #
    # ------------------------------------------------------------------ #
    output_path = args.output if args.output else args.src
    if output_path == args.src and os.path.exists(args.src):
        bak = args.src + ".bak"
        shutil.copy2(args.src, bak)
        print(f"\nBackup saved to: {bak}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    st.save_file(updated, output_path)
    print(f"Saved updated safetensors to: {output_path}")

    # ------------------------------------------------------------------ #
    # 5. Update config.json with attn_quant_config                        #
    # ------------------------------------------------------------------ #
    config_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["attn_quant_config"] = {
        "kv_cache_quant": {
            "dtype": "fp8_e4m3",
            "k_quant": {"scheme": "static", "granularity": "per_head"},
            "v_quant": {"scheme": "static", "granularity": "per_head"},
        },
        "q_quant": {"dtype": "fp8_e4m3", "scheme": "dynamic", "granularity": "per_token_per_head"},
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated config.json: {config_path}")

    # ------------------------------------------------------------------ #
    # 6. Quick sanity check                                               #
    # ------------------------------------------------------------------ #
    verify = st.load_file(output_path)
    sample_key = next(iter(new_scales))
    print(f"\nSanity check – {sample_key}:")
    print(f"  shape : {verify[sample_key].shape}")
    print(f"  values: {verify[sample_key]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
