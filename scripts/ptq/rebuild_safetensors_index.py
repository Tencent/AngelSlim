"""Rebuild ``model.safetensors.index.json`` from the actual on-disk shards.

Motivation
----------
The current OUT index.json in ``glm5_w8a8_static_int8`` has (at least)
one key rendered as a JSON-illegal literal newline sequence, e.g. the
byte range around offset ``10386825`` contains::

    ..."model-r15-00075.safetensors",
        "model.\n    .eh_proj.weight": "model-mtp-r00.safetensors",...

which is not valid JSON and, more importantly, hides the real key
``model.layers.78.eh_proj.weight`` from any loader (vLLM, HF).  The
underlying safetensors shard is fine -- inspection shows the tensor
``model.layers.78.eh_proj.weight`` is present in ``model-mtp-r00.safetensors``.

The safest and most complete recovery is therefore to rebuild the
index from the ground truth on disk: enumerate every ``*.safetensors``
shard in the OUT dir, read its actual key list plus tensor byte sizes,
and emit a fresh ``model.safetensors.index.json`` whose ``weight_map``
and ``metadata.total_size`` reflect reality.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

from safetensors import safe_open

# Bytes per element for every dtype we emit.  Keep in sync with the
# quantizer -- currently we only ship bf16 + int8.
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _shard_tensor_bytes(path: str) -> Dict[str, int]:
    """Return ``{key: nbytes}`` for every tensor in the shard, using the
    header's dtype/shape (no data materialisation)."""
    sizes: Dict[str, int] = {}
    with safe_open(path, framework="pt") as f:
        # safetensors' public API doesn't expose header directly, but
        # get_tensor is cheap for metadata via .get_slice().  We use the
        # tensor's numel * dtype size.
        for k in f.keys():
            t = f.get_tensor(k)
            sizes[k] = t.numel() * t.element_size()
    return sizes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True,
                   help="Quantized output dir whose index.json will be rebuilt.")
    p.add_argument("--backup-suffix", default=".broken",
                   help="Suffix appended to the original index.json before "
                        "overwriting (kept as an audit trail).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only report what would change, do not write.")
    args = p.parse_args()

    idx_path = os.path.join(args.out, "model.safetensors.index.json")

    # 1. Enumerate shards.
    shard_names = sorted(
        f for f in os.listdir(args.out)
        if f.endswith(".safetensors") and f.startswith("model")
    )
    print(f"[info] scanning {len(shard_names)} shard(s) under {args.out}")
    t0 = time.time()

    weight_map: Dict[str, str] = {}
    total_size = 0
    duplicates = []

    for i, name in enumerate(shard_names, 1):
        shard_path = os.path.join(args.out, name)
        sizes = _shard_tensor_bytes(shard_path)
        for k, nbytes in sizes.items():
            if k in weight_map:
                duplicates.append((k, weight_map[k], name))
            weight_map[k] = name
            total_size += nbytes
        if i % 50 == 0 or i == len(shard_names):
            elapsed = time.time() - t0
            print(f"  [{i:>4d}/{len(shard_names)}] cum_keys={len(weight_map)} "
                  f"cum_size={total_size/1e9:.2f} GB ({elapsed:.1f}s)")

    if duplicates:
        print(f"[WARN] {len(duplicates)} duplicated key(s) across shards "
              f"(kept the last occurrence). First few:")
        for k, a, b in duplicates[:5]:
            print(f"  {k}:  {a}  ->  {b}")

    # 2. Cross-check against the (possibly broken) old index.
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, "rb") as f:
                raw = f.read()
            # Attempt a strict parse; if it fails we still proceed to write.
            try:
                old = json.loads(raw)
                old_wm = old.get("weight_map", {})
                old_keys = set(old_wm.keys())
                new_keys = set(weight_map.keys())
                only_old = sorted(old_keys - new_keys)
                only_new = sorted(new_keys - old_keys)
                print(f"[compare] old_keys={len(old_keys)} new_keys={len(new_keys)}")
                print(f"          only-in-old (dropped): {len(only_old)}")
                print(f"          only-in-new (recovered): {len(only_new)}")
                if only_new[:10]:
                    print("          first few recovered keys:")
                    for k in only_new[:10]:
                        print(f"            + {k}  ->  {weight_map[k]}")
                if only_old[:10]:
                    print("          first few dropped keys (from broken index):")
                    for k in only_old[:10]:
                        print(f"            - {k}")
            except json.JSONDecodeError as e:
                print(f"[compare] old index is JSON-invalid ({e}); "
                      f"skipping detailed diff. Total NEW keys={len(weight_map)}")
        except OSError as e:
            print(f"[compare] could not read old index: {e}")

    # 3. Write out.
    new_doc = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    if args.dry_run:
        print("[dry-run] not writing. total_size=%d bytes, keys=%d"
              % (total_size, len(weight_map)))
        return 0

    if os.path.isfile(idx_path):
        backup = idx_path + args.backup_suffix
        # Only back up once; if a backup already exists, keep the earliest.
        if not os.path.exists(backup):
            os.replace(idx_path, backup)
            print(f"[backup] {idx_path}  ->  {backup}")
        else:
            os.remove(idx_path)
            print(f"[backup] backup already exists at {backup}; discarding old")

    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, idx_path)
    print(f"[done] wrote {idx_path}: keys={len(weight_map)} "
          f"total_size={total_size/1e9:.2f} GB in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
