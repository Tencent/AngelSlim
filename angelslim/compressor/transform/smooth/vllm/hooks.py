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

"""Online smooth-stat collection: forward hooks, setup/teardown, and
TP-aware stats retrieval.

Combines:

* Forward-hook classes for online smooth-stat collection:
    - :class:`SmoothAttnHook` — captures ``q`` / ``k`` inputs and the
      attention output (= ``o_proj`` input) on a vLLM ``Attention`` module.
    - :class:`SmoothDownProjInputHook` — captures the input of a dense MLP
      ``down_proj`` (which is ``silu(gate) * up``).
    - :class:`SmoothAlphaValueHook` — captures the *raw* down_proj input
      tensors needed by the alpha grid search.
* Setup / teardown for the above hooks and alpha-search hooks (registered
  via ``llm.apply_model(...)`` from the Phase 1 driver
  ``tools/smooth/run_vllm_smooth.py``).
* TP-aware smooth-stats retrieval & summary printing.

The hook classes were migrated verbatim from
``angelslim/compressor/quant/core/vllm_calibrate_utils/{hooks,search}.py``;
the only change is the import of ``per_channel_absmax`` from the
shared ``smooth/core`` layer instead of being defined locally.
"""

import os
from collections import defaultdict

import torch

from angelslim.compressor.quant.core.vllm_calibrate_utils._common import (
    _find_layers,
    _get_dist_info,
)

from ..core.tensor_math import per_channel_absmax
from .moe_inject import set_moe_collect_alpha_search, set_moe_collect_smooth

__all__ = [
    # Hook classes
    "SmoothAttnHook",
    "SmoothDownProjInputHook",
    "SmoothAlphaValueHook",
    # Setup / teardown
    "setup_smooth_hooks",
    "remove_smooth_hooks",
    "setup_smooth_alpha_search_hooks",
    "remove_smooth_alpha_search_hooks",
    # Stats retrieval
    "get_smooth_stats",
    "print_smooth_stats",
]


# ===========================================================================
# Forward-hook classes
# ===========================================================================


class SmoothAttnHook:
    """Hook for collecting per-channel (last-dim) absmax and EMA on
    Attention ``q`` / ``k`` inputs and the attention output.

    ``v`` is intentionally skipped.

    EMA update rule::
        ema = ema_momentum * ema + (1 - ema_momentum) * current_absmax

    Args:
        layer_name: Attention layer name.
        smooth_stats: Shared dict that stores per-key running absmax/ema.
        ema_momentum: EMA decay factor.
        token_clip: Optional percentile-based clipping for the per-channel
            absmax. See :func:`per_channel_absmax` for accepted values.
            ``<=0`` (default) preserves the original absolute-max behaviour.
    """

    def __init__(self, layer_name, smooth_stats, ema_momentum=0.9, token_clip=-1):
        self.layer_name = layer_name
        self.smooth_stats = smooth_stats
        self.ema_momentum = ema_momentum
        self.token_clip = token_clip
        self.call_count = 0

    def _update(self, key, tensor):
        """Update absmax and EMA for a named tensor slot."""
        with torch.no_grad():
            cur_absmax = per_channel_absmax(tensor, token_clip=self.token_clip)

            stats = self.smooth_stats[key]
            if stats["absmax"] is None:
                stats["absmax"] = cur_absmax
            else:
                stats["absmax"] = torch.maximum(stats["absmax"], cur_absmax)
            if stats["ema"] is None:
                stats["ema"] = cur_absmax.clone()
            else:
                stats["ema"] = (
                    self.ema_momentum * stats["ema"] + (1.0 - self.ema_momentum) * cur_absmax
                )
            stats["call_count"] = self.call_count

    def __call__(self, module, input, output):
        self.call_count += 1
        # --- inputs: q, k (v skipped) ---
        q = input[0] if len(input) > 0 else None
        k = input[1] if len(input) > 1 else None
        for tag, tensor in [("q", q), ("k", k)]:
            if tensor is not None and isinstance(tensor, torch.Tensor):
                self._update(f"{self.layer_name}.{tag}", tensor)
        # --- output: attention result = o_proj input ---
        attn_out = output[0] if isinstance(output, tuple) else output
        if attn_out is not None and isinstance(attn_out, torch.Tensor):
            self._update(f"{self.layer_name}.attn_out", attn_out)


class SmoothDownProjInputHook:
    """Hook for collecting per-channel (last-dim) absmax and EMA on the
    INPUT of a dense MLP ``down_proj`` layer.

    This captures ``silu(gate) * up`` — the true activation that
    ``down_proj`` sees and the correct signal for SmoothQuant calibration.

    Input shape: ``[num_tokens, intermediate_size]``.
    """

    def __init__(self, layer_name, smooth_stats, ema_momentum=0.9, token_clip=-1):
        self.layer_name = layer_name
        self.smooth_stats = smooth_stats
        self.ema_momentum = ema_momentum
        self.token_clip = token_clip
        self.call_count = 0

    def __call__(self, module, input, output):
        self.call_count += 1
        act = input[0] if isinstance(input, tuple) else input
        if not isinstance(act, torch.Tensor):
            return

        with torch.no_grad():
            cur_absmax = per_channel_absmax(act, token_clip=self.token_clip)

            stats = self.smooth_stats[self.layer_name]
            if stats["absmax"] is None:
                stats["absmax"] = cur_absmax
            else:
                stats["absmax"] = torch.maximum(stats["absmax"], cur_absmax)
            if stats["ema"] is None:
                stats["ema"] = cur_absmax.clone()
            else:
                stats["ema"] = (
                    self.ema_momentum * stats["ema"] + (1.0 - self.ema_momentum) * cur_absmax
                )
            stats["call_count"] = self.call_count


class SmoothAlphaValueHook:
    """Capture raw down_proj input activation for alpha grid search.

    Stores up to ``max_tokens`` token-rows per layer via CPU offload
    with uniform sub-sampling when the cap is exceeded.

    Works alongside :class:`SmoothDownProjInputHook` (which collects
    absmax / ema statistics); this hook stores the actual tensor values.

    Args:
        layer_name: down_proj module name (same key used in
            ``_smooth_stats``).
        alpha_search_values: shared dict
            ``{layer_name: {"tokens": Tensor | None}}``.
        max_tokens: maximum token-rows to keep per layer.
    """

    def __init__(self, layer_name, alpha_search_values, max_tokens=4096):
        self.layer_name = layer_name
        self.alpha_search_values = alpha_search_values
        self.max_tokens = max_tokens

    def __call__(self, module, input, output):
        act = input[0] if isinstance(input, tuple) else input
        if not isinstance(act, torch.Tensor):
            return

        with torch.no_grad():
            act_cpu = act.detach().float().cpu()
            stored = self.alpha_search_values[self.layer_name]
            if stored["tokens"] is None:
                stored["tokens"] = act_cpu
            else:
                stored["tokens"] = torch.cat([stored["tokens"], act_cpu], dim=0)
            # Uniform sub-sample when cap exceeded
            if stored["tokens"].shape[0] > self.max_tokens:
                indices = torch.linspace(0, stored["tokens"].shape[0] - 1, self.max_tokens).long()
                stored["tokens"] = stored["tokens"][indices]


# ===========================================================================
# Setup / teardown
# ===========================================================================


def setup_smooth_hooks(
    model,
    ema_momentum=0.9,
    collect_attn=True,
    collect_down_proj=True,
    collect_moe=True,
    token_clip=-1,
):
    """Register smooth hooks on the model to collect per-channel absmax
    and EMA stats.

    Args:
        model: The nn.Module model instance (called inside ``llm.apply_model``).
        ema_momentum: EMA decay factor (default 0.9).
        collect_attn: If True, register hooks on
            ``vllm.attention.layer.Attention`` to collect q / k inputs and
            attn output (= o_proj input).
        collect_down_proj: If True, register hooks on dense MLP
            ``down_proj`` layers to collect ``silu(gate)*up``.
        collect_moe: If True, wire FusedMoE layers for kernel-injected
            smooth stats.  Requires ``VLLM_MOE_COLLECT_SMOOTH_STATS=1``
            env var AND the corresponding call to
            :func:`collect_fused_moe_smooth_stats` inside the vLLM
            FusedMoE kernel source.
        token_clip: Optional percentile-based clipping for per-channel
            absmax collection.
            * ``<= 0`` (default): disabled, use absolute max.
            * In ``(0, 1]``: quantile fraction (e.g. 0.999).
            * In ``(1, 100)``: percentile value (e.g. 99.9).

    Statistics are stored in ``model._smooth_stats``::

        {
          "<attn_layer>.q":        {"absmax": Tensor, "ema": Tensor, "call_count": int},
          "<attn_layer>.k":        {...},
          "<attn_layer>.attn_out": {...},
          "<down_proj_layer>":     {...},
          "<moe_layer>.<i>.down_proj": {...},
        }
    """
    try:
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        from vllm.attention.layer import Attention
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase

    # ------------------------------------------------------------------
    # 1. Discover layers for each requested scope
    # ------------------------------------------------------------------
    attn_layers = _find_layers(model, layers=[Attention]) if collect_attn else {}
    down_proj_layers = {}
    if collect_down_proj:
        all_linear = _find_layers(model, layers=[torch.nn.Linear, LinearBase])
        down_proj_layers = {
            name: module
            for name, module in all_linear.items()
            if name.split(".")[-1] == "down_proj"
        }
    moe_layers = _find_layers(model, layers=[FusedMoE]) if collect_moe else {}

    # ------------------------------------------------------------------
    # EP (Expert Parallelism) check: smooth stat collection is NOT
    # compatible with EP — abort early to prevent silent wrong results.
    # ------------------------------------------------------------------
    if collect_moe and moe_layers:
        os.environ["VLLM_MOE_COLLECT_SMOOTH_STATS"] = "1"
        set_moe_collect_smooth(True)
        ep_layers = {
            name: layer
            for name, layer in moe_layers.items()
            if getattr(layer, "use_ep", False) and getattr(layer, "ep_size", 1) > 1
        }
        if ep_layers:
            ep_example = next(iter(ep_layers))
            ep_size = getattr(ep_layers[ep_example], "ep_size", "?")
            raise RuntimeError(
                f"[setup_smooth_hooks] Expert Parallelism (EP) is NOT supported for "
                f"smooth stat calibration. Detected {len(ep_layers)} FusedMoE layer(s) "
                f"with ep_size={ep_size} (e.g. '{ep_example}'). "
                f"Please re-run calibration with pure TP (set ep_size=1 / "
                f"disable expert parallelism)."
            )

    collect_moe_smooth = collect_moe and os.getenv("VLLM_MOE_COLLECT_SMOOTH_STATS", "0") == "1"

    print(
        f"---------Smooth hooks scope: "
        f"attn={'ON' if collect_attn else 'OFF'} ({len(attn_layers)} layers), "
        f"down_proj={'ON' if collect_down_proj else 'OFF'} ({len(down_proj_layers)} layers), "
        f"moe={'ON' if collect_moe else 'OFF'} ({len(moe_layers)} layers, "
        f"kernel_inject={'ON' if collect_moe_smooth else 'OFF'})---------"
    )
    if collect_moe and moe_layers and not collect_moe_smooth:
        print(
            "---------[WARNING] MoE scope enabled but VLLM_MOE_COLLECT_SMOOTH_STATS"
            " is not set to 1 — MoE stats will NOT be collected---------"
        )

    # ------------------------------------------------------------------
    # 2. Initialise stats storage
    # ------------------------------------------------------------------
    if not hasattr(model, "_smooth_stats"):
        model._smooth_stats = {}
    if not hasattr(model, "_smooth_stats_meta"):
        model._smooth_stats_meta = {}

    for name, _attn_module in attn_layers.items():
        # Read kv_head replica count from the qkv_proj inside the parent
        # LlamaAttention (or equivalent).
        kv_replicas = 1
        try:
            from vllm.model_executor.layers.linear import QKVParallelLinear

            parent_name = ".".join(name.split(".")[:-1])  # drop ".attn" suffix
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            qkv_proj = getattr(parent, "qkv_proj", None)
            if isinstance(qkv_proj, QKVParallelLinear):
                kv_replicas = qkv_proj.num_kv_head_replicas
        except Exception:
            pass

        for tag in ("q", "k", "attn_out"):
            key = f"{name}.{tag}"
            if key not in model._smooth_stats:
                model._smooth_stats[key] = {"absmax": None, "ema": None, "call_count": 0}
            model._smooth_stats_meta[key] = {
                "kv_replicas": kv_replicas if tag == "k" else 1,
            }

    for name in down_proj_layers:
        if name not in model._smooth_stats:
            model._smooth_stats[name] = {"absmax": None, "ema": None, "call_count": 0}

    if collect_moe_smooth:
        for name, layer in moe_layers.items():
            num_experts = getattr(layer, "global_num_experts", None) or getattr(
                layer, "local_num_experts", None
            )
            if num_experts is None:
                print(
                    f"[WARNING] Cannot determine num_experts for {name}, "
                    f"skipping pre-allocation"
                )
                continue
            for expert_id in range(num_experts):
                key = f"{name}.{expert_id}.down_proj"
                if key not in model._smooth_stats:
                    model._smooth_stats[key] = {"absmax": None, "ema": None, "call_count": 0}

    # ------------------------------------------------------------------
    # 3. Register PyTorch forward hooks (idempotent guard)
    # ------------------------------------------------------------------
    if not hasattr(model, "_smooth_hooks"):
        model._smooth_hooks = []

        for name, layer in attn_layers.items():
            hook = SmoothAttnHook(
                name,
                model._smooth_stats,
                ema_momentum=ema_momentum,
                token_clip=token_clip,
            )
            handle = layer.register_forward_hook(hook)
            model._smooth_hooks.append(handle)

        for name, layer in down_proj_layers.items():
            hook = SmoothDownProjInputHook(
                name,
                model._smooth_stats,
                ema_momentum=ema_momentum,
                token_clip=token_clip,
            )
            handle = layer.register_forward_hook(hook)
            model._smooth_hooks.append(handle)

    # ------------------------------------------------------------------
    # 4. Wire FusedMoE layers for kernel-injected smooth stats
    # ------------------------------------------------------------------
    if collect_moe_smooth:
        for name, layer in moe_layers.items():
            if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                layer.w13_weight._vllm_layer_name_smooth = name
                layer.w13_weight._smooth_stats_of_model = model._smooth_stats
                layer.w13_weight._smooth_ema_momentum = ema_momentum
            else:
                print(
                    f"[DEBUG] Cannot set smooth attrs on w13_weight for {name}: "
                    f"hasattr={hasattr(layer, 'w13_weight')}, "
                    f"is_none={getattr(layer, 'w13_weight', None) is None}"
                )

    print(f"---------Smooth hooks registered: {len(model._smooth_hooks)} total---------")
    return f"Registered {len(model._smooth_hooks)} smooth hooks"


def remove_smooth_hooks(model):
    """Remove smooth hooks registered by :func:`setup_smooth_hooks`."""
    if hasattr(model, "_smooth_hooks"):
        for handle in model._smooth_hooks:
            handle.remove()
        model._smooth_hooks.clear()
    os.environ.pop("VLLM_MOE_COLLECT_SMOOTH_STATS", None)
    set_moe_collect_smooth(False)
    return "Smooth hooks removed"


def setup_smooth_alpha_search_hooks(
    model,
    max_tokens=4096,
    collect_moe=True,
):
    """Register hooks to capture raw down_proj activations for alpha
    grid search.

    Must be called via ``llm.apply_model()`` AFTER
    :func:`setup_smooth_hooks`.

    Dense MLP: registers :class:`SmoothAlphaValueHook` on each
    ``down_proj`` layer.
    MoE: wires ``w13_weight`` attributes so that
    :func:`collect_fused_moe_alpha_search_values` (called from the kernel
    injection point) stores raw tensors per expert.
    """
    from vllm.model_executor.layers.linear import LinearBase

    if not hasattr(model, "_alpha_search_values"):
        model._alpha_search_values = {}
    if not hasattr(model, "_alpha_search_hooks"):
        model._alpha_search_hooks = []

    # --- Dense MLP down_proj ---
    all_linear = _find_layers(model, layers=[torch.nn.Linear, LinearBase])
    down_proj_layers = {
        name: module for name, module in all_linear.items() if name.split(".")[-1] == "down_proj"
    }

    for name, layer in down_proj_layers.items():
        if name not in model._alpha_search_values:
            model._alpha_search_values[name] = {"tokens": None}
        hook = SmoothAlphaValueHook(name, model._alpha_search_values, max_tokens)
        handle = layer.register_forward_hook(hook)
        model._alpha_search_hooks.append(handle)

    # --- MoE (kernel injection) ---
    moe_count = 0
    if collect_moe:
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE

            moe_layers = _find_layers(model, layers=[FusedMoE])
        except ImportError:
            moe_layers = {}

        if moe_layers:
            os.environ["VLLM_MOE_COLLECT_ALPHA_SEARCH"] = "1"
            set_moe_collect_alpha_search(True)

        per_expert_max = max(1, max_tokens // 8)  # for topk=8
        for name, layer in moe_layers.items():
            if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                layer.w13_weight._alpha_search_values_of_model = model._alpha_search_values
                layer.w13_weight._vllm_layer_name_smooth = name
                layer.w13_weight._alpha_search_max_tokens = per_expert_max
                moe_count += 1
            else:
                print(
                    f"[DEBUG] Cannot set smooth attrs on w13_weight for {name}: "
                    f"hasattr={hasattr(layer, 'w13_weight')}, "
                    f"is_none={getattr(layer, 'w13_weight', None) is None}"
                )
    total = len(model._alpha_search_hooks)
    msg = (
        f"Registered {total} alpha-search hooks (dense down_proj), "
        f"{moe_count} MoE layers wired"
    )
    print(f"[Alpha Search] {msg}")
    return msg


def remove_smooth_alpha_search_hooks(model):
    """Remove alpha-search hooks and free stored activation tensors."""
    if hasattr(model, "_alpha_search_hooks"):
        for handle in model._alpha_search_hooks:
            handle.remove()
        model._alpha_search_hooks.clear()
    if hasattr(model, "_alpha_search_values"):
        del model._alpha_search_values
    os.environ.pop("VLLM_MOE_COLLECT_ALPHA_SEARCH", None)
    set_moe_collect_alpha_search(False)
    return "Alpha-search hooks removed"


# ===========================================================================
# TP-aware smooth-stats retrieval & summary printing
# ===========================================================================


def get_smooth_stats(model):
    """Retrieve smooth statistics from the model.

    Performs all-reduce (MAX for absmax, mean for ema) across all
    workers.  Returns a serialisable dict::

        {
          "<key>": {"absmax": [float, ...], "ema": [float, ...], "call_count": int},
          ...
        }

    TP-aware gather:
        Each rank holds one shard of the full channel dimension.
        For ``k`` under GQA + TP (``kv_replicas > 1``), only every
        ``kv_replicas``-th rank carries a unique shard; the rest are
        duplicates and are skipped.  ``setup_smooth_hooks`` raises
        ``RuntimeError`` for EP, so EP never reaches here.

    Performance optimisation:
        Keys are grouped by ``(tensor_size, kv_replicas)``; each group
        runs **one** ``dist.all_gather`` instead of one per key.
    """
    if not hasattr(model, "_smooth_stats"):
        return None

    import torch.distributed as dist

    rank, world_size = _get_dist_info()

    if world_size > 1 and dist.is_initialized():
        meta = getattr(model, "_smooth_stats_meta", {})

        # Step 1: group keys by (tensor_size, kv_replicas)
        groups = defaultdict(list)  # (size, kv_reps) -> [(key, has_ema), ...]
        for key, stats in model._smooth_stats.items():
            if stats["absmax"] is None:
                continue
            kv_replicas = meta.get(key, {}).get("kv_replicas", 1)
            size = stats["absmax"].shape[0]
            has_ema = stats["ema"] is not None
            groups[(size, kv_replicas)].append((key, has_ema))

        # Step 2: per-group batched all_gather
        for (_size, kv_replicas), keys_info in groups.items():
            rows = []
            row_map = []  # tracks (key, field) for each row
            for key, has_ema in keys_info:
                rows.append(model._smooth_stats[key]["absmax"])
                row_map.append((key, "absmax"))
                if has_ema:
                    rows.append(model._smooth_stats[key]["ema"])
                    row_map.append((key, "ema"))

            stacked = torch.stack(rows, dim=0).cuda()  # [num_rows, size]
            gathered = [torch.zeros_like(stacked) for _ in range(world_size)]
            dist.all_gather(gathered, stacked)

            # De-duplicate replicated shards (GQA k heads)
            unique = gathered[::kv_replicas]

            full = torch.cat(unique, dim=1).cpu()  # [num_rows, full_size]

            for i, (key, field) in enumerate(row_map):
                model._smooth_stats[key][field] = full[i]

            del stacked, gathered, full

    # Serialise to plain Python lists (JSON-friendly)
    result = {}
    for key, stats in model._smooth_stats.items():
        result[key] = {
            "absmax": stats["absmax"].tolist() if stats["absmax"] is not None else None,
            "ema": stats["ema"].tolist() if stats["ema"] is not None else None,
            "call_count": stats.get("call_count", 0),
        }
    return result


def print_smooth_stats(model, max_rows=20):
    """Pretty-print smooth statistics summary (rank-0 only)."""
    if not hasattr(model, "_smooth_stats"):
        print("No smooth statistics available")
        return

    rank, _ = _get_dist_info()
    if rank != 0:
        return

    print("\n" + "=" * 90)
    print("Smooth Statistics  (per-channel absmax / ema — showing scalar max over channels)")
    print("=" * 90)
    rows = 0
    for key, stats in model._smooth_stats.items():
        if stats["absmax"] is None:
            absmax_repr = "N/A"
            ema_repr = "N/A"
        else:
            absmax_repr = f"{stats['absmax'].max().item():.6f}"
            ema_repr = f"{stats['ema'].max().item():.6f}" if stats["ema"] is not None else "N/A"
        print(
            f"{key:70s} | absmax_max: {absmax_repr:>12} | ema_max: {ema_repr:>12}"
            f" | calls: {stats.get('call_count', 0):4d}"
        )
        rows += 1
        if rows >= max_rows:
            remaining = len(model._smooth_stats) - max_rows
            if remaining > 0:
                print(f"  ... and {remaining} more entries (use get_smooth_stats() for full data)")
            break
    print("=" * 90 + "\n")
