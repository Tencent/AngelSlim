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

"""KV-cache min/max hooks on vLLM ``Attention`` layers: per-tensor and per-head."""

import torch

from ....utils import all_reduce_minmax, new_minmax, rank_world
from ....utils.utils import find_modules

__all__ = [
    "KVCacheHook",
    "KVCachePerHeadHook",
    "setup_kvcache_pertensor_hooks",
    "setup_kvcache_perhead_hooks",
    "get_kvcache_perhead_stats",
    "remove_kvcache_perhead_hooks",
    "setup_kvcache_only_hooks",
    "get_kvcache_only_stats",
]


def attention_cls():
    try:
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        from vllm.attention.layer import Attention
    return Attention


class KVCacheHook:
    """Per-tensor K/V min/max on an ``Attention`` forward (pickle-safe)."""

    def __init__(self, layer_name, kcache_stats, vcache_stats):
        self.layer_name = layer_name
        self.kcache_stats = kcache_stats
        self.vcache_stats = vcache_stats
        self.call_count = 0

    def __call__(self, module, input, output):
        self.call_count += 1
        _, k, v = input[0], input[1], input[2]
        if not isinstance(k, torch.Tensor):
            return
        with torch.no_grad():
            k_stats = self.kcache_stats[self.layer_name]
            k_stats["min"] = torch.minimum(k_stats["min"], k.min().detach().cpu())
            k_stats["max"] = torch.maximum(k_stats["max"], k.max().detach().cpu())
            v_stats = self.vcache_stats[self.layer_name]
            v_stats["min"] = torch.minimum(v_stats["min"], v.min().detach().cpu())
            v_stats["max"] = torch.maximum(v_stats["max"], v.max().detach().cpu())
            k_stats["call_count"] = self.call_count
            v_stats["call_count"] = self.call_count


def setup_kvcache_pertensor_hooks(model):
    """Per-layer K/V min/max into ``model._kcache_stats`` / ``_vcache_stats``."""
    kvcache_layers = find_modules(model, [attention_cls()])
    if not hasattr(model, "_kcache_stats"):
        model._kcache_stats = {name: new_minmax() for name in kvcache_layers}
        model._vcache_stats = {name: new_minmax() for name in kvcache_layers}
    if not hasattr(model, "_kvcache_hooks"):
        model._kvcache_hooks = [
            layer.register_forward_hook(
                KVCacheHook(name, model._kcache_stats, model._vcache_stats)
            )
            for name, layer in kvcache_layers.items()
        ]
    return f"Registered {len(model._kvcache_hooks)} kv-pertensor hooks"


# Per-head: when TP replicates KV heads (tp_size > num_kv_heads), odd ranks of a
# replication group track K only and even ranks V only; otherwise every rank
# tracks both.  A rank's "role" is "k", "v" or "both".


def get_kv_role(rank, world_size, num_kv_heads_total):
    if world_size <= 1 or num_kv_heads_total is None or num_kv_heads_total <= 0:
        return "both"
    replication = world_size // num_kv_heads_total
    if replication < 2 or world_size % num_kv_heads_total != 0:
        return "both"
    return "k" if (rank % 2 == 1) else "v"


def compute_perhead_layout(rank, world_size, num_kv_heads_total):
    """``(role, heads_per_rank, global_head_offset, replication)`` for this rank."""
    if num_kv_heads_total is None or num_kv_heads_total <= 0 or world_size <= 1:
        return "both", num_kv_heads_total or 0, 0, 1

    if world_size % num_kv_heads_total == 0 and world_size >= num_kv_heads_total:
        replication = world_size // num_kv_heads_total
        global_head_offset = rank // replication
        heads_per_rank = 1
    elif num_kv_heads_total % world_size == 0:
        replication = 1
        heads_per_rank = num_kv_heads_total // world_size
        global_head_offset = rank * heads_per_rank
    else:
        replication = 1
        heads_per_rank = max(1, num_kv_heads_total // max(1, world_size))
        global_head_offset = rank * heads_per_rank

    role = get_kv_role(rank, world_size, num_kv_heads_total)
    return role, heads_per_rank, global_head_offset, replication


def num_heads_from_tensor(k, module):
    """Local KV head count: from the attention module if it says, else the shape."""
    impl = getattr(module, "impl", None) or getattr(module, "attn", None)
    for attr in ("num_kv_heads", "num_heads"):
        for obj in (module, impl):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    if k.ndim == 4:
        return k.shape[-2]
    return 1


def infer_num_kv_heads_total(model):
    """Global (pre-TP) KV head count from the model config, or ``None``."""
    for cfg_attr in ("config", "model_config", "hf_config"):
        cfg = getattr(model, cfg_attr, None)
        if cfg is None:
            continue
        for val_attr in ("num_key_value_heads", "num_kv_heads"):
            for c in (cfg, getattr(cfg, "text_config", None)):
                v = getattr(c, val_attr, None) if c is not None else None
                if isinstance(v, int) and v > 0:
                    return v
    return None


class KVCachePerHeadHook:
    """Per-head K/V min/max; stats become ``(num_heads,)`` vectors on first call."""

    def __init__(self, layer_name, kcache_stats, vcache_stats, num_kv_heads_total=None):
        self.layer_name = layer_name
        self.kcache_stats = kcache_stats
        self.vcache_stats = vcache_stats
        self._num_heads = None
        self._num_kv_heads_total = num_kv_heads_total
        self._role = None

    def __call__(self, module, input, output):
        _, k, v = input[0], input[1], input[2]
        if not isinstance(k, torch.Tensor):
            return

        with torch.no_grad():
            if self._num_heads is None:
                self._num_heads = num_heads_from_tensor(k, module)
            if self._role is None:
                rank, world_size = rank_world()
                total = self._num_kv_heads_total or self._num_heads * world_size
                self._role = get_kv_role(rank, world_size, total)

            H = self._num_heads

            def per_head_minmax(t):
                if t.ndim >= 3 and t.shape[-2] == H:
                    t_heads, actual_H = t.reshape(-1, H, t.shape[-1]), H
                elif H > 0 and t.shape[-1] % H == 0:
                    t_heads, actual_H = t.reshape(-1, H, t.shape[-1] // H), H
                else:
                    t_heads, actual_H = t.reshape(1, -1, 1), 1
                flat = t_heads.reshape(t_heads.shape[0], actual_H, -1).float()
                return (
                    flat.min(dim=0).values.min(dim=-1).values.cpu(),
                    flat.max(dim=0).values.max(dim=-1).values.cpu(),
                )

            for side, t, stats_dict in (("k", k, self.kcache_stats), ("v", v, self.vcache_stats)):
                if self._role not in ("both", side):
                    continue
                h_min, h_max = per_head_minmax(t)
                stats = stats_dict[self.layer_name]
                if stats["min"].shape != h_min.shape:
                    stats["min"], stats["max"] = h_min.clone(), h_max.clone()
                else:
                    stats["min"] = torch.minimum(stats["min"], h_min)
                    stats["max"] = torch.maximum(stats["max"], h_max)


def _all_gather_stats_perhead(stats_dict, stats_type, num_kv_heads_total=None):
    """Reduce per-head stats into full ``(num_kv_heads_total,)`` vectors on every rank.
    Each rank writes its owned head slice into a neutral vector before MIN/MAX; under
    the K/V split ``stats_type`` ends in ``-k``/``-v`` so non-owners stay neutral."""
    import torch.distributed as dist

    rank, world_size = rank_world()
    if world_size <= 1:
        return
    done_key = f"__gathered_{stats_type}__"
    if stats_dict.get(done_key, False):
        return

    side = "k" if stats_type.endswith("-k") else "v" if stats_type.endswith("-v") else None

    local_h = 0
    for name, stats in stats_dict.items():
        if name.startswith("__gathered_"):
            continue
        t = stats["min"]
        if t.ndim == 1 and not (t.numel() == 1 and torch.isinf(t).all()):
            local_h = t.numel()
            break
    h_tensor = torch.tensor([local_h], dtype=torch.long, device="cuda")
    dist.all_reduce(h_tensor, op=dist.ReduceOp.MAX)
    H_local = int(h_tensor.item())
    if H_local <= 0:
        stats_dict[done_key] = True
        return

    if num_kv_heads_total is None or num_kv_heads_total <= 0:
        num_kv_heads_total = H_local * world_size
    role, _, global_head_offset, _ = compute_perhead_layout(rank, world_size, num_kv_heads_total)
    have_data = role == "both" or side is None or role == side

    for name, stats in stats_dict.items():
        if name.startswith("__gathered_"):
            continue
        for key, op, neutral in (
            ("min", dist.ReduceOp.MIN, float("inf")),
            ("max", dist.ReduceOp.MAX, float("-inf")),
        ):
            full = torch.full((num_kv_heads_total,), neutral, dtype=torch.float32, device="cuda")
            if have_data:
                t = stats[key].to(device="cuda", dtype=torch.float32)
                end = global_head_offset + H_local
                if t.numel() == H_local and end <= num_kv_heads_total:
                    full[global_head_offset:end] = t
            dist.all_reduce(full, op=op)
            stats[key] = full.cpu()
        torch.cuda.empty_cache()

    dist.barrier()
    stats_dict[done_key] = True


def setup_kvcache_perhead_hooks(model):
    """Per-head K/V min/max into ``model._kvcache_perhead_stats``."""
    kvcache_layers = find_modules(model, [attention_cls()])
    # The hook only sees the per-rank head count; the K/V role needs the global one.
    num_kv_heads_total = infer_num_kv_heads_total(model)
    model._kvcache_num_kv_heads_total = num_kv_heads_total

    if not hasattr(model, "_kvcache_perhead_stats"):
        model._kvcache_perhead_stats = {
            "k": {name: new_minmax() for name in kvcache_layers},
            "v": {name: new_minmax() for name in kvcache_layers},
        }
    if not hasattr(model, "_kvcache_perhead_hooks"):
        model._kvcache_perhead_hooks = [
            layer.register_forward_hook(
                KVCachePerHeadHook(
                    name,
                    model._kvcache_perhead_stats["k"],
                    model._kvcache_perhead_stats["v"],
                    num_kv_heads_total=num_kv_heads_total,
                )
            )
            for name, layer in kvcache_layers.items()
        ]
    return (
        f"Registered {len(model._kvcache_perhead_hooks)} kv-perhead hooks "
        f"(num_kv_heads_total={num_kv_heads_total})"
    )


def get_kvcache_perhead_stats(model):
    """``{"<attn>.k_cache"|".v_cache": {"min": [float]*H, "max": [...]}}`` or ``None``."""
    if not hasattr(model, "_kvcache_perhead_stats"):
        return None
    total = getattr(model, "_kvcache_num_kv_heads_total", None)
    for side in ("k", "v"):
        _all_gather_stats_perhead(
            model._kvcache_perhead_stats[side], f"kv-perhead-{side}", num_kv_heads_total=total
        )

    out = {}
    for side in ("k", "v"):
        for name, stats in model._kvcache_perhead_stats[side].items():
            if name.startswith("__gathered_"):
                continue
            out[f"{name}.{side}_cache"] = {k: stats[k].tolist() for k in ("min", "max")}
    return out


def remove_kvcache_perhead_hooks(model):
    for h in getattr(model, "_kvcache_perhead_hooks", []):
        h.remove()
    for attr in ("_kvcache_perhead_hooks", "_kvcache_perhead_stats"):
        if hasattr(model, attr):
            delattr(model, attr)
    return "KV-perhead hooks removed"


# KV-only (no Linear / MoE hooks): fp8 KV cache without weight quantization.


def setup_kvcache_only_hooks(model):
    kvcache_layers = find_modules(model, [attention_cls()])
    if not hasattr(model, "_kvcache_only_stats"):
        model._kvcache_only_stats = {
            "k": {name: new_minmax() for name in kvcache_layers},
            "v": {name: new_minmax() for name in kvcache_layers},
        }
    if not hasattr(model, "_kvcache_only_hooks"):
        model._kvcache_only_hooks = [
            layer.register_forward_hook(
                KVCacheHook(name, model._kvcache_only_stats["k"], model._kvcache_only_stats["v"])
            )
            for name, layer in kvcache_layers.items()
        ]
    return f"Registered {len(model._kvcache_only_hooks)} kv-only hooks"


def get_kvcache_only_stats(model):
    """Same format as ``get_activation_stats``' KV entries (``KVScaleSearcher`` input)."""
    if not hasattr(model, "_kvcache_only_stats"):
        return None
    out = {}
    for side in ("k", "v"):
        all_reduce_minmax(model._kvcache_only_stats[side], f"kv-only-{side}")
        for name, s in model._kvcache_only_stats[side].items():
            out[f"{name}.{side}_cache"] = {"min": s["min"].item(), "max": s["max"].item()}
    return out
