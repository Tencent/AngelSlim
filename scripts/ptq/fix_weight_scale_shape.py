"""In-place fix: reshape every ``*.weight_scale`` from 1-D ``[N]`` to
2-D ``[N, 1]`` inside the OUT directory's shards.

Why: vLLM's compressed-tensors INT8 W8A8 per-channel loader registers
``weight_scale`` with the parameter shape ``[out_features, 1]`` (a 2-D
column vector).  Our quantizer had saved it as 1-D ``[out_features]``,
which makes ``_assert_and_load`` fail with a plain ``AssertionError``:
the shapes do not match and the 1-D-scalar bypass does not apply.

The reference kunlun checkpoint stores 2-D ``(N, 1)`` scales, which is
what we now match here.  Only the ``weight_scale`` tensors are rewritten
(and only those whose current shape is 1-D); ``weight`` tensors are
untouched, so the fix is fast and byte-identical for the INT8 payload.
"""

import argparse
import json
import os
import shutil
import time
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _load_shard(path: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _rewrite_shard(path: str, verbose: bool = False) -> int:
    """Rewrite one shard, returning the number of scales upgraded."""
    tensors = _load_shard(path)
    upgraded = 0
    for k, t in list(tensors.items()):
        if not k.endswith(".weight_scale"):
            continue
        if t.dim() == 2 and t.shape[-1] == 1:
            continue  # already correct
        if t.dim() != 1:
            raise RuntimeError(f"{path}::{k} has unexpected shape {tuple(t.shape)}")
        tensors[k] = t.unsqueeze(-1).contiguous()
        upgraded += 1
        if verbose:
            print(f"    {k}: {tuple(t.shape)} -> {tuple(tensors[k].shape)}")
    if upgraded == 0:
        return 0
    tmp = path + ".tmp"
    save_file(tensors, tmp)
    os.replace(tmp, path)
    return upgraded


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True,
                   help="Quantized output dir whose weight_scale tensors "
                        "should be reshaped in place.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, process only the first N shards (for smoke tests).")
    args = p.parse_args()

    idx_path = os.path.join(args.out, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx["weight_map"]

    # Collect shards that actually contain at least one weight_scale.
    shards_to_fix: List[str] = []
    seen = set()
    for k, shard in weight_map.items():
        if not k.endswith(".weight_scale"):
            continue
        if shard in seen:
            continue
        seen.add(shard)
        shards_to_fix.append(shard)

    shards_to_fix.sort()
    if args.limit > 0:
        shards_to_fix = shards_to_fix[: args.limit]

    print(f"[info] {len(shards_to_fix)} shards contain weight_scale tensors.")
    t0 = time.time()
    total_upgraded = 0
    for i, name in enumerate(shards_to_fix):
        p_path = os.path.join(args.out, name)
        if not os.path.isfile(p_path):
            print(f"  [skip] {name} (missing on disk)")
            continue
        n = _rewrite_shard(p_path, verbose=args.verbose)
        total_upgraded += n
        if (i + 1) % 10 == 0 or n == 0 or args.verbose:
            elapsed = time.time() - t0
            print(f"  [{i+1:>4d}/{len(shards_to_fix)}] {name}: +{n} scales "
                  f"(cum={total_upgraded}, {elapsed:.1f}s elapsed)")

    print(f"[done] upgraded {total_upgraded} weight_scale tensors "
          f"across {len(shards_to_fix)} shards in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
