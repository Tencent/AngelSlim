#!/usr/bin/env python3
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

"""Static INT8 per-channel weight-only quantization for GLM-5, aligned to a
reference ``model.safetensors.index.json``.

Design notes
------------
1. No calibration, no activation stats, no smooth-quant, no distributed
   NCCL, no HF model object.  Purely a tensor-level rewrite of the
   BF16 checkpoint into per-channel symmetric INT8 weights.
2. The set of tensors to quantize is derived from the *reference* index
   file: every source ``foo.weight`` whose paired ``foo.weight_scale``
   exists in the reference is quantized; every other tensor is copied
   through as bf16.  This guarantees the resulting layout is 1:1 with
   the reference (same file naming, same key set, same shard grouping),
   including all MTP layer-78 keys and the odd per-layer indexer opt-outs.
3. Reads source shards lazily via ``safetensors.safe_open`` (mmap, no
   full-model materialization).  RAM footprint is bounded by the size of
   the largest single tensor (< 400 MB for GLM-5 MoE experts).
4. Shard grouping is copied from the reference verbatim.  We stream each
   destination shard by iterating its key list, pulling tensors from the
   correct source shard on demand, and dumping into
   ``safetensors.torch.save_file``.

Quantization formula
--------------------
    scale = |W|.amax(dim=1, keepdim=False) / 127.0     # per output channel
    W_int8 = clamp(round(W / scale[:, None]), -127, 127).to(int8)
    stored: {key}.weight       -> int8   [out, in]
    {key}.weight_scale -> bf16   [out, 1]

This matches the vLLM ``compressed-tensors`` INT8 layout consumed by the
kunlun deployment stack: ``config.json.quantization_config`` declares
``strategy=channel``, ``symmetric=true``, and activations run in dynamic
per-token mode at inference time (no static input_scale on disk).

Usage
-----
    python scripts/ptq/static_quantize_glm5_from_ref.py \
        --bf16 /apdcephfs_sgfd2/share_300532381/harviexu/chatglm5.2 \
        --ref  /apdcephfs_sgfd2/share_300532381/harviexu/kunlun_harvie/w8a8/glm5_w8a8_kunlun_2node \
        --out  /apdcephfs_sgfd2/share_300532381/harviexu/kunlun_harvie/w8a8/glm5_w8a8_static_int8 \
        --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as _fut
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Reference index parsing
# ---------------------------------------------------------------------------

def _load_index(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wm = data.get("weight_map")
    if not isinstance(wm, dict):
        raise ValueError(f"{path} does not look like a HF safetensors index (no weight_map)")
    return wm


def _derive_quant_targets(ref_weight_map: Dict[str, str]) -> Tuple[set, set, Dict[str, List[str]]]:
    """Split reference keys into three views.

    Returns
    -------
    to_quantize:
        set of ``{key}.weight`` names that must be quantized (INT8 [out,in]
        + bf16 [out] weight_scale).  A source key is quantizable iff the
        reference also contains its sibling ``{key}.weight_scale``.
    passthrough:
        set of names that must be copied verbatim (bf16 or original dtype).
        This is every reference key that is NOT one of the two roles above
        (i.e. neither the quantized-weight nor the derived scale).
    shard_to_keys:
        destination shard filename -> list of keys living in that shard,
        preserving the reference's ordering (Python dict is insertion-
        ordered as of 3.7+, and json.load preserves file order).
    """
    scale_keys = {k for k in ref_weight_map if k.endswith(".weight_scale")}
    to_quantize = set()
    for sk in scale_keys:
        # foo.weight_scale -> foo.weight is the paired quantized tensor
        base = sk[: -len(".weight_scale")]
        wkey = base + ".weight"
        if wkey in ref_weight_map:
            to_quantize.add(wkey)
        # else: dangling scale (should not happen); ignore.

    passthrough = set(ref_weight_map.keys()) - to_quantize - scale_keys

    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for k, shard in ref_weight_map.items():
        shard_to_keys[shard].append(k)

    return to_quantize, passthrough, shard_to_keys


# ---------------------------------------------------------------------------
# Source shard resolution
# ---------------------------------------------------------------------------

class _SourceReader:
    """Lazy mmap wrapper over the BF16 checkpoint.

    ``safe_open`` handles are cached per shard file to amortize the
    open cost across the thousands of tensors we pull per shard.
    """

    def __init__(self, bf16_dir: str) -> None:
        self.bf16_dir = bf16_dir
        idx_path = os.path.join(bf16_dir, "model.safetensors.index.json")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(idx_path)
        self.weight_map = _load_index(idx_path)
        # Per-thread cache: dict {shard_file: safe_open handle}.  We
        # avoid true multi-threading concerns by using thread-local
        # caches in the workers.
        self._handles: Dict[str, "safe_open"] = {}

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        self._handles.clear()

    def _get_handle(self, shard_file: str):
        h = self._handles.get(shard_file)
        if h is None:
            path = os.path.join(self.bf16_dir, shard_file)
            h = safe_open(path, framework="pt", device="cpu")
            h.__enter__()
            self._handles[shard_file] = h
        return h

    def get_tensor(self, key: str) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"Source BF16 index has no key: {key}")
        h = self._get_handle(shard)
        return h.get_tensor(key)


# ---------------------------------------------------------------------------
# INT8 per-channel symmetric quantization core
# ---------------------------------------------------------------------------

_INT8_MAX = 127
_EPS = 1e-8


def quantize_per_channel_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8 quantization.

    Assumes ``weight`` is 2-D ``[out_features, in_features]`` (the HF
    ``nn.Linear.weight`` convention).  For non-2-D tensors the caller
    must decide whether they belong to the quantizable set at all.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"Expected 2-D linear weight for per-channel quant; got shape {tuple(weight.shape)}"
        )
    # Work in fp32 for numeric stability, then cast the scale back to bf16.
    w_f32 = weight.detach().to(torch.float32)
    amax = w_f32.abs().amax(dim=1)                     # [out]
    scale = (amax / float(_INT8_MAX)).clamp_min(_EPS)  # [out]  bf16-safe
    w_int = torch.round(w_f32 / scale[:, None])
    w_int = w_int.clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    # vLLM's compressed-tensors INT8 W8A8 per-channel loader expects the
    # weight_scale param to be 2-D ``[out_features, 1]`` (a column
    # vector).  Saving a bare 1-D ``[out_features]`` tensor makes
    # ``_assert_and_load`` fail with an AssertionError at load time.
    scale_bf16 = scale.to(torch.bfloat16).unsqueeze(-1).contiguous()  # [out, 1]
    return w_int.contiguous(), scale_bf16


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------

def _write_one_shard(
    shard_name: str,
    keys: List[str],
    bf16_dir: str,
    out_dir: str,
    to_quantize: set,
    passthrough: set,
    scale_keys: set,
) -> Tuple[str, int, int, int]:
    """Materialize one destination shard on disk.

    Returns ``(shard_name, n_quantized, n_passthrough, n_scale)`` for
    the top-level progress log.
    """
    reader = _SourceReader(bf16_dir)
    try:
        out_tensors: Dict[str, torch.Tensor] = {}
        n_q = n_p = n_s = 0

        # First pass: quantize every weight that is scheduled to be
        # quantized in this shard, and produce its scale side-by-side.
        # We also cover passthrough tensors here so the shard-writer
        # sees a single flat dict.
        pending_scales = {k for k in keys if k in scale_keys}
        pending_weights_to_quant = {k for k in keys if k in to_quantize}
        pending_passthrough = {k for k in keys if k in passthrough}

        # Sanity: every scale in the shard must have its paired weight
        # in the SAME shard (verified against reference index; we do
        # NOT relax this).  We produce the scale from the source
        # weight, then also emit the int8 weight for the same key.
        for wkey in pending_weights_to_quant:
            src = reader.get_tensor(wkey)
            if src.dim() != 2:
                # Non-2-D "weight" that the reference chose to quantize
                # should not happen for GLM-5, but be strict rather than
                # silently corrupt: fall back to passthrough and drop
                # the scale.
                print(
                    f"[warn] {wkey}: shape {tuple(src.shape)} is not 2-D; "
                    f"falling back to bf16 passthrough (no scale emitted).",
                    file=sys.stderr,
                )
                out_tensors[wkey] = src.to(torch.bfloat16).contiguous()
                n_p += 1
                continue

            w_int8, scale_bf16 = quantize_per_channel_int8(src)
            out_tensors[wkey] = w_int8
            scale_name = wkey[: -len(".weight")] + ".weight_scale"
            if scale_name in pending_scales:
                out_tensors[scale_name] = scale_bf16
                pending_scales.discard(scale_name)
                n_s += 1
            else:
                # Reference did not co-locate the scale here.  This is a
                # loud error because it means our shard partition is
                # inconsistent with the reference; abort rather than
                # ship a broken checkpoint.
                raise RuntimeError(
                    f"Reference index puts {wkey} in {shard_name} but its "
                    f"scale {scale_name} is missing from the same shard."
                )
            n_q += 1

        # Any leftover scales in the shard have no paired weight -> bug.
        if pending_scales:
            raise RuntimeError(
                f"Shard {shard_name} contains dangling scales without "
                f"paired weights: {sorted(pending_scales)[:5]}..."
            )

        # Passthrough tensors: keep original dtype (bf16 in GLM-5) and
        # push straight through.
        for pkey in pending_passthrough:
            src = reader.get_tensor(pkey)
            out_tensors[pkey] = src.contiguous()
            n_p += 1

        # Write.  Metadata format="pt" avoids the "no metadata" warning
        # some HF loaders emit.
        out_path = os.path.join(out_dir, shard_name)
        save_file(out_tensors, out_path, metadata={"format": "pt"})
        return shard_name, n_q, n_p, n_s
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Metadata / config emission
# ---------------------------------------------------------------------------

def _emit_hf_quant_config(out_dir: str, ref_weight_map: Dict[str, str]) -> None:
    """Emit ``hf_quant_config.json`` = trtllm-style INT8 recipe.

    ``exclude_modules`` lists every quantizable-looking leaf that the
    reference decided NOT to quantize (i.e. no paired ``weight_scale``).
    We approximate this by scanning reference for names matching typical
    quantizable leaves; anything without a scale entry is excluded.
    """
    _QUANT_LEAVES = (
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
        "wq_b", "wk", "gate_proj", "up_proj", "down_proj",
    )
    scale_bases = {
        k[: -len(".weight_scale")]
        for k in ref_weight_map
        if k.endswith(".weight_scale")
    }
    excluded: set = set()
    for k in ref_weight_map:
        if not k.endswith(".weight"):
            continue
        base = k[: -len(".weight")]
        leaf = base.rsplit(".", 1)[-1]
        if leaf not in _QUANT_LEAVES:
            continue
        if base in scale_bases:
            continue
        excluded.add(base)

    # Always-excluded top-level modules that never carry scales anyway.
    excluded_common = ["lm_head"]

    payload = {
        "quantization": {
            "quant_algo": "INT8",
            "kv_cache_quant_algo": None,
            "exclude_modules": excluded_common + sorted(excluded),
        }
    }
    with open(os.path.join(out_dir, "hf_quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _emit_config_json(
    bf16_dir: str,
    ref_dir: str,
    out_dir: str,
    ref_weight_map: Dict[str, str],
) -> None:
    """Emit ``config.json`` for the INT8 checkpoint.

    We use the REFERENCE ``config.json`` as the base, NOT the BF16 one.
    Reason: kunlun's serving stack patches a handful of MLA-critical
    fields on top of the vanilla HF config (most importantly
    ``head_dim`` which the BF16 dump sets to ``qk_nope_head_dim`` (192)
    but the vLLM MLA attention backend expects to equal
    ``qk_rope_head_dim`` (64) for GlmMoeDsa; leaving it at 192 makes
    vLLM compute a bogus ``head_size=704`` and no attention backend
    accepts it).  Other REF-only fields (``bos_token_id``, ``mlp_bias``,
    ``num_experts``) are similarly required.

    Any BF16-only additions are preserved via a shallow merge (REF wins
    on collision), so future BF16 metadata additions won't silently
    disappear.
    """
    with open(os.path.join(bf16_dir, "config.json"), "r", encoding="utf-8") as f:
        bf16_cfg = json.load(f)
    with open(os.path.join(ref_dir, "config.json"), "r", encoding="utf-8") as f:
        ref_cfg = json.load(f)

    # Shallow merge: BF16 first (baseline), then REF wins on every key
    # (except ``quantization_config`` which we will rebuild ourselves).
    cfg = dict(bf16_cfg)
    for k, v in ref_cfg.items():
        if k == "quantization_config":
            continue
        cfg[k] = v

    # Build ignore list = same base names as hf_quant_config.exclude_modules,
    # minus the "leaf substring" shortcuts.  These are FQNs so vLLM's
    # compressed-tensors ignore-matcher hits them exactly.
    _QUANT_LEAVES = (
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
        "wq_b", "wk", "gate_proj", "up_proj", "down_proj",
    )
    scale_bases = {
        k[: -len(".weight_scale")]
        for k in ref_weight_map
        if k.endswith(".weight_scale")
    }
    ignore = ["lm_head"]
    for k in ref_weight_map:
        if not k.endswith(".weight"):
            continue
        base = k[: -len(".weight")]
        leaf = base.rsplit(".", 1)[-1]
        if leaf not in _QUANT_LEAVES:
            continue
        if base in scale_bases:
            continue
        ignore.append(base)

    cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": None,
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": None,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.15.1",
    }

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _copy_side_files(bf16_dir: str, out_dir: str) -> None:
    """Copy tokenizer / chat template / generation config from BF16 dir.

    We do NOT copy the source ``model.safetensors.index.json`` -- the
    output has its own (built from the reference).  We also do NOT copy
    source shards.
    """
    _CANDIDATES = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    )
    for name in _CANDIDATES:
        src = os.path.join(bf16_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))


def _write_output_index(out_dir: str, ref_weight_map: Dict[str, str]) -> None:
    """Write ``model.safetensors.index.json`` == reference (verbatim)."""
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": ref_weight_map}, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bf16", required=True, help="Path to BF16 GLM-5 checkpoint directory.")
    p.add_argument("--ref",  required=True, help="Path to reference W8A8 checkpoint directory (used only for index.json layout).")
    p.add_argument("--out",  required=True, help="Output directory to write the INT8 checkpoint.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel shard writers (each holds one full tensor in RAM at a time).")
    p.add_argument("--shard-glob", type=str, default=None,
                   help="If set, only process destination shards whose name matches this substring "
                        "(handy for restart / partial runs).")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse index and print stats, but do not write any tensor.")
    args = p.parse_args()

    bf16_dir = os.path.abspath(args.bf16)
    ref_dir  = os.path.abspath(args.ref)
    out_dir  = os.path.abspath(args.out)

    if not os.path.isdir(bf16_dir):
        print(f"[fatal] BF16 dir not found: {bf16_dir}", file=sys.stderr); return 2
    if not os.path.isdir(ref_dir):
        print(f"[fatal] reference dir not found: {ref_dir}",   file=sys.stderr); return 2
    os.makedirs(out_dir, exist_ok=True)

    ref_index = os.path.join(ref_dir, "model.safetensors.index.json")
    if not os.path.isfile(ref_index):
        print(f"[fatal] reference index missing: {ref_index}", file=sys.stderr); return 2
    ref_wm = _load_index(ref_index)

    to_quantize, passthrough, shard_to_keys = _derive_quant_targets(ref_wm)
    scale_keys = {k for k in ref_wm if k.endswith(".weight_scale")}

    # Sanity: every quantizable weight must exist in the BF16 source
    # index; otherwise the reference lists a key we cannot produce.
    src_index_path = os.path.join(bf16_dir, "model.safetensors.index.json")
    if not os.path.isfile(src_index_path):
        print(f"[fatal] BF16 index missing: {src_index_path}", file=sys.stderr); return 2
    src_wm = _load_index(src_index_path)
    missing_q = sorted(k for k in to_quantize if k not in src_wm)
    missing_p = sorted(k for k in passthrough if k not in src_wm)
    if missing_q or missing_p:
        print(
            f"[warn] {len(missing_q)} quantize-keys and {len(missing_p)} "
            f"passthrough-keys present in reference but MISSING in BF16 source. "
            f"First few missing quantize: {missing_q[:5]}",
            file=sys.stderr,
        )
        # These keys will fail hard at shard-write time.  We surface
        # them here so the user can decide (typically MTP-only keys
        # from an outdated BF16 dump).

    print(
        f"[info] reference keys        : {len(ref_wm):>8}\n"
        f"[info]   to-quantize weight  : {len(to_quantize):>8}\n"
        f"[info]   paired scale (bf16) : {len(scale_keys):>8}\n"
        f"[info]   passthrough (bf16)  : {len(passthrough):>8}\n"
        f"[info] destination shards    : {len(shard_to_keys):>8}\n"
        f"[info] BF16 src keys         : {len(src_wm):>8}"
    )

    if args.shard_glob:
        shard_to_keys = {
            s: ks for s, ks in shard_to_keys.items() if args.shard_glob in s
        }
        print(f"[info] shard-glob filter applied -> {len(shard_to_keys)} shards")

    if args.dry_run:
        print("[dry-run] no tensors written; exiting.")
        return 0

    # Emit sidecar metadata first (cheap) so partial runs still land a
    # valid config.json / tokenizer.
    _write_output_index(out_dir, ref_wm)
    _emit_config_json(bf16_dir, ref_dir, out_dir, ref_wm)
    _emit_hf_quant_config(out_dir, ref_wm)
    _copy_side_files(bf16_dir, out_dir)
    print(f"[info] wrote sidecar metadata under {out_dir}")

    # Dispatch shard writers.  Each worker owns its own _SourceReader
    # (== its own safe_open cache) so they never share mmap handles.
    t0 = time.time()
    tasks: List[Tuple[str, List[str]]] = sorted(shard_to_keys.items())
    total = len(tasks)

    def _run_one(item: Tuple[str, List[str]]):
        shard_name, keys = item
        return _write_one_shard(
            shard_name, keys, bf16_dir, out_dir,
            to_quantize, passthrough, scale_keys,
        )

    done = 0
    if args.workers <= 1:
        for item in tasks:
            r = _run_one(item)
            done += 1
            _log_shard(r, done, total, t0)
    else:
        with _fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_run_one, tasks):
                done += 1
                _log_shard(r, done, total, t0)

    dt = time.time() - t0
    print(f"[done] {total} shards written in {dt/60.0:.1f} min -> {out_dir}")
    return 0


def _log_shard(r: Tuple[str, int, int, int], done: int, total: int, t0: float) -> None:
    shard, nq, np_, ns = r
    dt = time.time() - t0
    rate = done / max(dt, 1e-6)
    eta = (total - done) / max(rate, 1e-6)
    print(
        f"[shard {done:>4d}/{total}] {shard:<32s} "
        f"quantized={nq:<5d} passthrough={np_:<5d} scales={ns:<5d} "
        f"elapsed={dt/60.0:6.1f}m  ETA={eta/60.0:6.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
