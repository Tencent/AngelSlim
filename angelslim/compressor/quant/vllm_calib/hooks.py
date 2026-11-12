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


"""Runtime observation hooks for vLLM calibration: the fused-MoE kernel wrapper, the MoE
expert-input stats it feeds, and the per-module activation min/max hooks.

The MoE kernel is wrapped in place, not patched on disk. gate_up input:
``TritonExperts.apply``'s ``hidden_states``; down_proj input: the first argument of
``moe_kernel_quantize_input``, called inside ``apply``."""

import os
import threading

import torch

from ....utils import all_reduce_minmax, new_minmax, rank_world
from ....utils.utils import find_modules
from .image_embeds import patch_image_embeds
from .kvcache import (
    KVCacheHook,
    attention_cls,
    setup_kvcache_perhead_hooks,
    setup_kvcache_pertensor_hooks,
)

__all__ = [
    "ActivationHook",
    "collect_fused_moe_channel_stats",
    "collect_fused_moe_internal_stats",
    "collect_fused_moe_scale_stats",
    "get_activation_stats",
    "get_moe_channel_stats",
    "get_moe_stats",
    "install",
    "is_fp8_scale_only",
    "moe_observer",
    "moe_scale_observer",
    "register_observer",
    "register_scale_observer",
    "setup_activation_hooks",
    "setup_mtp_activation_hooks",
    "tag_moe_layers",
]

_ctx = threading.local()
_observers = []
_scale_observers = []
_installed = False


def register_observer(fn):
    """``fn(stage, hidden_states, topk_ids, global_num_experts, w1)``; w1 is ``w13_weight``."""
    if fn not in _observers:
        _observers.append(fn)


def register_scale_observer(fn):
    """``fn(stage, scale, w1)`` with the dynamic FP8 activation scale (None for BF16 MoE)."""
    if fn not in _scale_observers:
        _scale_observers.append(fn)


def _noop(*_args, **_kwargs):
    return None


def _triton_module():
    """The module defining ``TritonExperts``: HPC fork, else community vLLM."""
    try:
        from vllm.model_executor.layers.fused_moe.experts import triton_moe
    except ImportError:
        from vllm.model_executor.layers.fused_moe import fused_moe as triton_moe
    return triton_moe


def install():
    """Wrap the kernel once per process; every vLLM worker has to call it."""
    global _installed
    if _installed:
        return
    triton_moe = _triton_module()

    # Retired on-disk patches call these module globals; a no-op stops double counting.
    for attr in ("_gptq_hessian_moe_call", "_angelslim_moe_stats_call"):
        if hasattr(triton_moe, attr):
            setattr(triton_moe, attr, _noop)
            print(f"[moe_hook] WARNING: disabled retired patch {attr} in {triton_moe.__file__}")
    src = ""
    if getattr(triton_moe, "__file__", None):
        with open(triton_moe.__file__, encoding="utf-8") as f:
            src = f.read()
    if "AngelSlim MoE calibration hook" in src or "# === AngelSlim ===" in src:
        raise RuntimeError(
            f"{triton_moe.__file__} still carries a retired AngelSlim vllm_patch, which would "
            "count MoE stats twice: restore the original from its .bak backup"
        )

    orig_apply = triton_moe.TritonExperts.apply
    orig_quant = triton_moe.moe_kernel_quantize_input

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, *args, **kwargs):  # fmt: skip
        if _observers or _scale_observers:
            # Resolved the same way apply() does before using it.
            n = w1.shape[0] if global_num_experts == -1 else global_num_experts
            for obs in _observers:
                obs("gate_up_proj", hidden_states, topk_ids, n, w1)
            a1q_scale = args[1] if len(args) > 1 else kwargs.get("a1q_scale")
            for obs in _scale_observers:
                obs("gate_up_proj", a1q_scale, w1)
            _ctx.call = (topk_ids, n, w1)
        try:
            return orig_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                              activation, global_num_experts, *args, **kwargs)  # fmt: skip
        finally:
            _ctx.call = None

    def quantize_input(A, *args, **kwargs):
        call = getattr(_ctx, "call", None)
        if call is not None:
            for obs in _observers:
                obs("down_proj", A, *call)
        out = orig_quant(A, *args, **kwargs)
        if call is not None:
            for obs in _scale_observers:
                obs("down_proj", out[1], call[2])
        return out

    triton_moe.TritonExperts.apply = apply
    # triton_moe binds this name at import, so it has to be replaced there.
    triton_moe.moe_kernel_quantize_input = quantize_input
    _installed = True


_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
FP8_E4M3_MAX = 448.0  # vLLM's dynamic FP8 scale is amax / 448


def _enabled(env, layer_name, stats):
    return (
        os.getenv(env, "0") == "1"
        and stats is not None
        and layer_name is not None
        and "experts" in layer_name.lower()
    )


def collect_fused_moe_internal_stats(
    stage, hidden_states, topk_ids, global_num_experts, layer_name=None,
    global_moe_activation_stats=None,
):  # fmt: skip
    """Layer-level (and with ``VLLM_MOE_COLLECT_PER_EXPERT_STATS=1`` per-expert) min/max.
    FP8 inputs are skipped: their amax comes from ``collect_fused_moe_scale_stats``."""
    stats_dict = global_moe_activation_stats
    if not _enabled("VLLM_MOE_COLLECT_STATS", layer_name, stats_dict):
        return
    if hidden_states.dtype in _FP8:
        return

    key = f"{layer_name}.{stage}"
    with torch.no_grad():
        if key in stats_dict:
            stats = stats_dict[key]
            stats["min"] = torch.minimum(stats["min"], hidden_states.min().detach().cpu())
            stats["max"] = torch.maximum(stats["max"], hidden_states.max().detach().cpu())
            stats["call_count"] = stats.get("call_count", 0) + 1

        if os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") != "1":
            return
        rows, (n_tok, top_k) = hidden_states.shape[0], topk_ids.shape
        if rows == n_tok:  # gate_up: one row per token, repeated for each routed expert
            flat = hidden_states.unsqueeze(1).expand(-1, top_k, -1)
            flat = flat.reshape(-1, hidden_states.shape[-1])
        elif rows == n_tok * top_k:  # down_proj: already one row per (token, expert)
            flat = hidden_states
        else:
            return
        ids = topk_ids.reshape(-1)
        for e in ids.unique():
            if e < 0:  # padding slot
                continue
            rows_e = flat[ids == e]
            if rows_e.numel() == 0:
                continue
            stats = stats_dict.setdefault(f"{layer_name}.{e.item()}.{stage}", new_minmax())
            stats["min"] = torch.minimum(stats["min"], rows_e.min().detach().cpu())
            stats["max"] = torch.maximum(stats["max"], rows_e.max().detach().cpu())
            stats["call_count"] = stats.get("call_count", 0) + 1


def collect_fused_moe_scale_stats(stage, scale, layer_name=None, global_moe_activation_stats=None):
    """Dynamic FP8 activation scale under ``<layer>.<stage>.scale_amax``; max * 448 = BF16 amax."""
    stats_dict = global_moe_activation_stats
    if scale is None or not _enabled("VLLM_MOE_COLLECT_STATS", layer_name, stats_dict):
        return
    with torch.no_grad():
        s = scale.detach().to(torch.float32)
        s_min, s_max = s.min().cpu(), s.max().cpu()
        stats = stats_dict.setdefault(
            f"{layer_name}.{stage}.scale_amax",
            {**new_minmax(), "bf16_amax_recovered": torch.tensor(float("-inf")), "call_count": 0},
        )
        stats["min"] = torch.minimum(stats["min"], s_min)
        stats["max"] = torch.maximum(stats["max"], s_max)
        stats["bf16_amax_recovered"] = torch.maximum(
            stats["bf16_amax_recovered"], (s_max * FP8_E4M3_MAX).cpu()
        )
        stats["call_count"] += 1


def collect_fused_moe_channel_stats(
    stage, hidden_states, topk_ids, layer_name=None, global_moe_channel_stats=None
):
    """Per-expert per-channel absmax of the down_proj input, for NVFP4 channel reordering.
    Stored as ``{"<layer>.<e>.down_proj": {"absmax": Tensor[I_shard], "call_count": int}}``."""
    stats_dict = global_moe_channel_stats
    if stage != "down_proj":
        return
    if not _enabled("VLLM_MOE_COLLECT_CHANNEL_STATS", layer_name, stats_dict):
        return
    if hidden_states.dtype in _FP8:
        return
    with torch.no_grad():
        act = hidden_states if hidden_states.dim() > 1 else hidden_states.unsqueeze(0)
        if act.shape[0] != topk_ids.numel():
            return
        ids = topk_ids.reshape(-1)
        for e in ids.unique():
            if e < 0:  # padding slot
                continue
            rows_e = act[ids == e]
            if rows_e.numel() == 0:
                continue
            cur = rows_e.abs().max(dim=0).values.detach().cpu()
            stats = stats_dict.get(f"{layer_name}.{e.item()}.down_proj")
            if stats is None:
                stats_dict[f"{layer_name}.{e.item()}.down_proj"] = {"absmax": cur, "call_count": 1}
            else:
                stats["absmax"] = torch.maximum(stats["absmax"], cur)
                stats["call_count"] += 1


def moe_observer(stage, hidden_states, topk_ids, global_num_experts, w1):
    name = getattr(w1, "_vllm_layer_name", None)
    if not getattr(w1, "_moe_scale_only", False):  # FP8 MoE: see moe_scale_observer
        collect_fused_moe_internal_stats(
            stage, hidden_states, topk_ids, global_num_experts, name,
            getattr(w1, "_moe_activation_stats_of_model", None),
        )  # fmt: skip
    collect_fused_moe_channel_stats(
        stage, hidden_states, topk_ids, name, getattr(w1, "_moe_channel_stats_of_model", None)
    )


def moe_scale_observer(stage, scale, w1):
    collect_fused_moe_scale_stats(
        stage, scale, getattr(w1, "_vllm_layer_name", None),
        getattr(w1, "_moe_activation_stats_of_model", None),
    )  # fmt: skip


def tag_moe_layers(model, moe_layers, *, is_fp8=False, per_channel=False):
    """Allocate ``model._moe_activation_stats`` and tag every layer's ``w13_weight``.
    FP8 models get no BF16 keys; their ``*.scale_amax`` entries are created lazily."""
    per_expert = os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") == "1"
    if per_channel and not hasattr(model, "_moe_channel_stats"):
        model._moe_channel_stats = {}
    if hasattr(model, "_moe_activation_stats"):
        return
    stats = model._moe_activation_stats = {}
    for name, layer in moe_layers.items():
        n_experts = layer.global_num_experts
        for stage in () if is_fp8 else ("gate_up_proj", "down_proj"):
            stats[f"{name}.{stage}"] = new_minmax()
            for e in range(n_experts) if per_expert else ():
                stats[f"{name}.{e}.{stage}"] = new_minmax()
        w13 = layer.w13_weight
        w13._vllm_layer_name = name
        w13._moe_activation_stats_of_model = stats
        w13._moe_scale_only = is_fp8
        if per_channel:
            w13._moe_channel_stats_of_model = model._moe_channel_stats


def get_moe_stats(model):
    if not hasattr(model, "_moe_activation_stats"):
        return None
    all_reduce_minmax(model._moe_activation_stats, "MoE")
    return {
        n: {"min": s["min"].item(), "max": s["max"].item()}
        for n, s in model._moe_activation_stats.items()
    }


def get_moe_channel_stats(model):
    """Per-channel absmax gathered across TP to the full intermediate dim, or ``None``."""
    if not getattr(model, "_moe_channel_stats", None):
        return None
    import torch.distributed as dist

    _, world_size = rank_world()
    if world_size > 1:
        # Ranks must issue identical all_gathers: group by shard size, sort keys.
        groups = {}
        for key, s in model._moe_channel_stats.items():
            groups.setdefault(s["absmax"].shape[0], []).append(key)
        for keys in groups.values():
            keys = sorted(keys)
            stacked = torch.stack([model._moe_channel_stats[k]["absmax"] for k in keys]).cuda()
            gathered = [torch.zeros_like(stacked) for _ in range(world_size)]
            dist.all_gather(gathered, stacked)
            full = torch.cat(gathered, dim=1).cpu()
            for i, k in enumerate(keys):
                model._moe_channel_stats[k]["absmax"] = full[i]
    return {
        k: {"absmax": s["absmax"].tolist(), "call_count": s["call_count"]}
        for k, s in model._moe_channel_stats.items()
    }


class ActivationHook:
    """Input min/max of one Linear layer (pickle-safe)."""

    def __init__(self, layer_name, activation_stats):
        self.layer_name = layer_name
        self.activation_stats = activation_stats
        self.call_count = 0

    def __call__(self, module, input, output):
        self.call_count += 1
        act = input[0] if isinstance(input, tuple) else input
        if not isinstance(act, torch.Tensor):
            return
        with torch.no_grad():
            stats = self.activation_stats[self.layer_name]
            stats["min"] = torch.minimum(stats["min"], act.min().detach().cpu())
            stats["max"] = torch.maximum(stats["max"], act.max().detach().cpu())
            stats["call_count"] = self.call_count


def _vllm_classes():
    from vllm.model_executor.layers.linear import LinearBase

    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE as moe_cls
    except ImportError:
        # Builds where FusedMoEFactory returns a MoERunner owning a RoutedExperts.
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts as moe_cls,
        )
    return [torch.nn.Linear, LinearBase], moe_cls


def find_moe_layers(model):
    """``{name: layer}`` of the modules owning ``w13_weight``, named like the checkpoint
    (``...mlp.experts``) even where the weights sit on ``...mlp.experts.routed_experts``."""
    _, moe_cls = _vllm_classes()
    found = find_modules(model, [moe_cls])
    return {name.removesuffix(".routed_experts"): layer for name, layer in found.items()}


def _is_fp8_model(model, moe_layers):
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    get = qcfg.get if isinstance(qcfg, dict) else lambda k: getattr(qcfg, k, None)
    qm = get("quant_method")
    if isinstance(qm, str) and qm.lower() == "fp8":
        return True
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    for layer in list(moe_layers.values())[:1]:
        return any(getattr(layer, a).dtype in fp8 for a in ("w13_weight", "w2_weight"))
    return False


def _register_linear_hooks(model, linear_classes):
    layers = find_modules(model, linear_classes)
    if not hasattr(model, "_activation_stats"):
        model._activation_stats = {name: new_minmax() for name in layers}
    if not hasattr(model, "_activation_hooks"):
        model._activation_hooks = [
            layer.register_forward_hook(ActivationHook(name, model._activation_stats))
            for name, layer in layers.items()
        ]


def setup_activation_hooks(model, kv_granularity="per-tensor"):
    """Arm a worker for Linear / KV (``none|per-tensor|per-head``) / MoE min/max.
    FP8 checkpoints only collect MoE activation scales: their inputs are already quantized."""
    linear_classes, _ = _vllm_classes()
    embeds_msg = patch_image_embeds(model)
    moe_layers = find_moe_layers(model)

    is_fp8 = _is_fp8_model(model, moe_layers)
    model._angelslim_fp8_scale_only = is_fp8
    if is_fp8:
        model._activation_stats = getattr(model, "_activation_stats", {})
        model._activation_hooks = getattr(model, "_activation_hooks", [])
    else:
        _register_linear_hooks(model, linear_classes)
        if kv_granularity == "per-tensor":
            setup_kvcache_pertensor_hooks(model)
        elif kv_granularity == "per-head":
            setup_kvcache_perhead_hooks(model)

    if moe_layers:
        per_channel = os.getenv("VLLM_MOE_COLLECT_CHANNEL_STATS", "0") == "1"
        tag_moe_layers(model, moe_layers, is_fp8=is_fp8, per_channel=per_channel)
        install()
        register_observer(moe_observer)
        register_scale_observer(moe_scale_observer)
    return (
        f"{len(model._activation_hooks)} linear hooks, {len(moe_layers)} MoE layers, "
        f"kv={kv_granularity}, fp8_scale_only={is_fp8}; {embeds_msg}"
    )


def is_fp8_scale_only(model):
    return bool(getattr(model, "_angelslim_fp8_scale_only", False))


def get_activation_stats(model):
    """Globally reduced Linear min/max plus per-tensor KV as ``<attn>.k_cache|.v_cache``."""
    if not hasattr(model, "_activation_stats"):
        return None
    out = {}
    for attr, suffix in (
        ("_activation_stats", ""),
        ("_kcache_stats", ".k_cache"),
        ("_vcache_stats", ".v_cache"),
    ):
        stats = getattr(model, attr, None)
        if stats is None:
            continue
        all_reduce_minmax(stats, attr)
        for n, s in stats.items():
            out[n + suffix] = {"min": s["min"].item(), "max": s["max"].item()}
    return out


def setup_mtp_activation_hooks(draft_model):
    """MTP draft block (reached via ``collective_rpc``): Linear, per-tensor KV and MoE min/max."""
    linear_classes, _ = _vllm_classes()
    _register_linear_hooks(draft_model, linear_classes)

    kv_layers = find_modules(draft_model, [attention_cls()])
    if not hasattr(draft_model, "_kcache_stats"):
        draft_model._kcache_stats = {name: new_minmax() for name in kv_layers}
        draft_model._vcache_stats = {name: new_minmax() for name in kv_layers}
    if not hasattr(draft_model, "_kvcache_hooks"):
        draft_model._kvcache_hooks = [
            layer.register_forward_hook(
                KVCacheHook(name, draft_model._kcache_stats, draft_model._vcache_stats)
            )
            for name, layer in kv_layers.items()
        ]

    moe_layers = find_moe_layers(draft_model)
    if moe_layers:
        tag_moe_layers(draft_model, moe_layers)
        install()
        register_observer(moe_observer)
        register_scale_observer(moe_scale_observer)
    n_linear = len(draft_model._activation_hooks)
    return f"[MTP] {n_linear} linear, {len(kv_layers)} kv, {len(moe_layers)} MoE"
