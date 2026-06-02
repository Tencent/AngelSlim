"""All forward-hook based calibration logic.

Combines five originally separate sub-modules:

* Linear-layer activation hooks
* Per-tensor KV-cache hooks
* Per-head KV-cache hooks
* KV-cache only (lightweight) hooks
* FusedMoE expert-internal stats (vLLM patch entry point
  ``collect_fused_moe_internal_stats``)
* MTP draft-model hook setup

Definition order is bottom-up: hook classes and helpers come first, then
the public ``setup_*`` / ``get_*`` / ``print_*`` / ``remove_*`` entry points.
"""

import os

import torch

from ._common import (
    _all_reduce_stats,
    _compute_perhead_layout,
    _find_layers,
    _get_dist_info,
    _get_kv_role,
    _print_stats_table,
)

__all__ = [
    # Activation hooks (linear layers)
    "ActivationHook",
    "setup_activation_hooks",
    "get_activation_stats",
    "print_activation_stats",
    # KV-cache hooks (per-tensor)
    "KVCacheHook",
    "setup_kvcache_pertensor_hooks",
    # KV-cache hooks (per-head)
    "KVCachePerHeadHook",
    "setup_kvcache_perhead_hooks",
    "get_kvcache_perhead_stats",
    "print_kvcache_perhead_stats",
    "remove_kvcache_perhead_hooks",
    # KV-cache only (no Linear / MoE)
    "setup_kvcache_only_hooks",
    "get_kvcache_only_stats",
    "print_kvcache_only_stats",
    "remove_kvcache_only_hooks",
    # MoE stats (vLLM patch entry point)
    "collect_fused_moe_internal_stats",
    "get_moe_stats",
    "print_moe_stats",
    # MTP draft model
    "setup_mtp_activation_hooks",
    "get_mtp_activation_stats",
    "print_mtp_activation_stats",
    "get_mtp_moe_stats",
    "print_mtp_moe_stats",
]


# =============================================================================
# Activation hook & per-tensor KV-cache hook (definitions)
# =============================================================================


class ActivationHook:
    """Hook class for collecting activation statistics (pickle-safe)."""

    def __init__(self, layer_name, activation_stats):
        self.layer_name = layer_name
        self.activation_stats = activation_stats
        self.call_count = 0  # Track how many times this hook is called

    def __call__(self, module, input, output):
        self.call_count += 1

        # Get the input activation
        if isinstance(input, tuple):
            act = input[0]
        else:
            act = input

        if isinstance(act, torch.Tensor):
            # if act.numel() == 0:
            #     #print("Empty tensor", module)
            #     act = torch.tensor([0.0])
            # Use tensor operations to avoid graph breaks
            with torch.no_grad():
                act_min = act.min().detach().cpu()
                act_max = act.max().detach().cpu()

                # Update global min/max using tensor operations
                stats = self.activation_stats[self.layer_name]
                stats["min"] = torch.minimum(stats["min"], act_min)
                stats["max"] = torch.maximum(stats["max"], act_max)
                stats["call_count"] = self.call_count  # Store call count


class KVCacheHook:
    """Hook class for collecting kv cache statistics (pickle-safe)."""

    def __init__(self, layer_name, kcache_stats, vcache_stats):
        self.layer_name = layer_name
        self.kcache_stats = kcache_stats
        self.vcache_stats = vcache_stats
        self.call_count = 0  # Track how many times this hook is called

    def __call__(self, module, input, output):
        self.call_count += 1

        # Get the input activation
        _, k, v = input[0], input[1], input[2]

        if isinstance(k, torch.Tensor):
            # Use tensor operations to avoid graph breaks
            with torch.no_grad():
                k_act_min = k.min().detach().cpu()
                k_act_max = k.max().detach().cpu()
                v_act_min = v.min().detach().cpu()
                v_act_max = v.max().detach().cpu()

                # Update global min/max using tensor operations
                k_stats = self.kcache_stats[self.layer_name]
                k_stats["min"] = torch.minimum(k_stats["min"], k_act_min)
                k_stats["max"] = torch.maximum(k_stats["max"], k_act_max)
                v_stats = self.vcache_stats[self.layer_name]
                v_stats["min"] = torch.minimum(v_stats["min"], v_act_min)
                v_stats["max"] = torch.maximum(v_stats["max"], v_act_max)
                k_stats["call_count"] = self.call_count  # Store call count
                v_stats["call_count"] = self.call_count  # Store call count


def setup_activation_hooks(model, kv_granularity="per-tensor"):
    """
    Setup activation hooks on the model to collect min/max statistics.
    This function is applied to each worker's model instance.

    Args:
        kv_granularity: Controls KV-cache hook registration.
            'none'       – skip KV hooks entirely (only Linear/MoE hooks registered).
            'per-tensor' – register per-layer (per-tensor) KV min/max hooks (default).
            'per-head'   – register per-head KV min/max hooks (calls
                           setup_kvcache_perhead_hooks internally).
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase

    # Find all linear layers to monitor
    layers_to_monitor = _find_layers(model, layers=[torch.nn.Linear, LinearBase])
    print(f"---------Found {len(layers_to_monitor)} layers to monitor---------")
    for name in list(layers_to_monitor.keys())[:5]:  # Print first 5
        print(f"  {name}")
    if len(layers_to_monitor) > 5:
        print(f"  ... and {len(layers_to_monitor) - 5} more activation layers")

    # Initialize activation statistics storage
    if not hasattr(model, "_activation_stats"):
        model._activation_stats = {}
        for name in layers_to_monitor.keys():
            model._activation_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    # Register hooks for all linear layers
    if not hasattr(model, "_activation_hooks"):
        model._activation_hooks = []
        for name, layer in layers_to_monitor.items():
            hook = ActivationHook(name, model._activation_stats)
            hook_handle = layer.register_forward_hook(hook)
            model._activation_hooks.append(hook_handle)

    # KV-cache hooks: behaviour controlled by kv_granularity
    if kv_granularity == "none":
        print("---------KV-cache hooks skipped (kv_granularity=none)---------")

    elif kv_granularity == "per-tensor":
        # Delegate to the dedicated per-tensor hook setup
        setup_kvcache_pertensor_hooks(model)
        print("---------Per-tensor KV-cache hooks registered via setup_activation_hooks---------")

    elif kv_granularity == "per-head":
        # Delegate to the dedicated per-head hook setup
        setup_kvcache_perhead_hooks(model)
        print("---------Per-head KV-cache hooks registered via setup_activation_hooks---------")

    # Register MoE statistics storage and hooks
    moe_layers = _find_layers(model, layers=[FusedMoE])
    if moe_layers:
        print(f"---------Found {len(moe_layers)} MoE layers to monitor---------")
        for name in list(moe_layers.keys())[:5]:  # Print first 5
            print(f"  {name}")
        if len(moe_layers) > 5:
            print(f"  ... and {len(moe_layers) - 5} more")

        # Check if per-expert stats collection is enabled
        per_expert = os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") == "1"
        print(
            f"---------Per-expert stats collection: {'ENABLED' if per_expert else 'DISABLED'}---------"  # noqa: E501
        )

        # Initialize MoE activation statistics storage
        if not hasattr(model, "_moe_activation_stats"):
            model._moe_activation_stats = {}
            for name, layer in moe_layers.items():
                # Get the number of experts from the FusedMoE layer
                num_experts = getattr(layer, "global_num_experts", None)
                if num_experts is None:
                    num_experts = getattr(layer, "num_experts", 256)
                    print(
                        f"[WARNING] Could not find global_num_experts "
                        f"for {name}, using {num_experts}"
                    )

                for stage in ["gate_up_proj", "down_proj"]:
                    # Layer-level stats (overall)
                    model._moe_activation_stats[f"{name}.{stage}"] = {
                        "min": torch.tensor(float("inf")),
                        "max": torch.tensor(float("-inf")),
                    }
                    # Per-expert stats (only when enabled)
                    if per_expert:
                        for expert_id in range(num_experts):
                            model._moe_activation_stats[f"{name}.{expert_id}.{stage}"] = {
                                "min": torch.tensor(float("inf")),
                                "max": torch.tensor(float("-inf")),
                            }

                # Set layer name attribute on weights for statistics collection
                if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                    layer.w13_weight._vllm_layer_name = name
                    layer.w13_weight._moe_activation_stats_of_model = model._moe_activation_stats
                else:
                    print(
                        f"[DEBUG] Cannot set w13_weight._vllm_layer_name: "
                        f"hasattr={hasattr(layer, 'w13_weight')}, "
                        f"is_none={getattr(layer, 'w13_weight', None) is None}"
                    )

    print("---------Activation hooks registered---------")
    return f"Registered {len(model._activation_hooks)} hooks"


def get_activation_stats(model):
    """
    Retrieve activation statistics from the model.
    Performs all-reduce across all workers to get global min/max.
    """
    if not hasattr(model, "_activation_stats"):
        return None

    # Perform all-reduce to get global min/max across all workers
    try:
        _all_reduce_stats(model._activation_stats, stats_type="activation")
        if hasattr(model, "_kcache_stats"):
            _all_reduce_stats(model._kcache_stats, stats_type="kcache")
            _all_reduce_stats(model._vcache_stats, stats_type="vcache")
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")

    # Convert tensors to Python scalars for easier use
    stats_dict = {}
    for name, stats in model._activation_stats.items():
        stats_dict[name] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    if hasattr(model, "_kcache_stats"):
        kcache_stats_dict = {}
        for name, stats in model._kcache_stats.items():
            kcache_stats_dict[name + ".k_cache"] = {
                "min": (
                    stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"]
                ),
                "max": (
                    stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"]
                ),
            }
        stats_dict.update(kcache_stats_dict)
    if hasattr(model, "_vcache_stats"):
        vcache_stats_dict = {}
        for name, stats in model._vcache_stats.items():
            vcache_stats_dict[name + ".v_cache"] = {
                "min": (
                    stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"]
                ),
                "max": (
                    stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"]
                ),
            }
        stats_dict.update(vcache_stats_dict)
    return stats_dict


def print_activation_stats(model):
    """
    Print activation statistics in a readable format.
    Performs all-reduce to get global statistics across all workers.
    """
    if not hasattr(model, "_activation_stats"):
        print("No activation statistics available")
        return

    # Perform all-reduce to get global min/max
    try:
        rank, world_size = _all_reduce_stats(model._activation_stats, stats_type="activation")
        if hasattr(model, "_kcache_stats"):
            _all_reduce_stats(model._kcache_stats, stats_type="kcache")
            _all_reduce_stats(model._vcache_stats, stats_type="vcache")
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")
        rank, world_size = 0, 1

    # Only rank 0 prints the statistics (or single process)
    if rank != 0:
        return

    # Print statistics
    if world_size > 1:
        print(f"\n[Global statistics across {world_size} workers]")
    _print_stats_table(model._activation_stats, "Activation Statistics")
    if hasattr(model, "_kcache_stats"):
        _print_stats_table(model._kcache_stats, "K-cache Statistics")
        _print_stats_table(model._vcache_stats, "V-cache Statistics")


# =============================================================================
# Per-Head KV-Cache Calibration
# =============================================================================


def _get_num_heads_from_tensor(k: torch.Tensor, module) -> int:
    """
    Infer the number of KV heads from the tensor and (optionally) the module.

    vLLM passes k with shape  (total_tokens, num_kv_heads * head_dim)  for the
    PagedAttention path, or sometimes (batch, seq, num_kv_heads, head_dim).
    We try the module attribute first, then fall back to shape inspection.
    """
    # 1) Try to read directly from the Attention implementation object.
    for attr in ("num_kv_heads", "num_heads"):
        if hasattr(module, attr):
            val = getattr(module, attr)
            if isinstance(val, int) and val > 0:
                return val
        impl = getattr(module, "impl", None) or getattr(module, "attn", None)
        if impl is not None and hasattr(impl, attr):
            val = getattr(impl, attr)
            if isinstance(val, int) and val > 0:
                return val

    # 2) Fall back: if k is 4-D the head dimension is dim[-2].
    if k.ndim == 4:
        return k.shape[-2]

    # 3) Cannot determine – treat as single head (per-tensor).
    return 1


def _infer_num_kv_heads_total(model) -> int | None:
    """
    Try hard to discover the model's **global** (pre-TP-replication) number
    of KV heads by probing common config locations.  Returns ``None`` when
    the value cannot be determined – callers should then fall back to
    ``H_local * world_size`` which is correct when there is no TP
    replication.

    Probed locations (first hit wins):
        1. ``model.config.num_key_value_heads`` (HuggingFace)
        2. ``model.model_config.num_key_value_heads``
        3. ``model.config.num_kv_heads``
        4. ``model.hf_config.num_key_value_heads``
        5. ``model.config.text_config.num_key_value_heads`` (VLM with nested)
    """
    candidates = [
        ("config", "num_key_value_heads"),
        ("config", "num_kv_heads"),
        ("model_config", "num_key_value_heads"),
        ("model_config", "num_kv_heads"),
        ("hf_config", "num_key_value_heads"),
        ("hf_config", "num_kv_heads"),
    ]
    for cfg_attr, val_attr in candidates:
        cfg = getattr(model, cfg_attr, None)
        if cfg is None:
            continue
        if hasattr(cfg, val_attr):
            v = getattr(cfg, val_attr)
            if isinstance(v, int) and v > 0:
                return v
        # Try nested text_config (used by some VLMs).
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, val_attr):
            v = getattr(text_cfg, val_attr)
            if isinstance(v, int) and v > 0:
                return v
    return None


class KVCachePerHeadHook:
    """
    Forward hook on vLLM ``Attention`` layers that tracks per-head min/max
    statistics for both K and V tensors.

    The hook reshapes k/v from ``(T, num_heads * head_dim)`` (or any layout
    that contains a head axis) to ``(T, num_heads, head_dim)``, then reduces
    over the token and head-dim axes so the result has shape ``(num_heads,)``.

    Stats are stored as 1-D tensors so the existing ``_all_reduce_stats``
    (which calls ``dist.all_reduce`` on arbitrary-shape tensors) works
    transparently.
    """

    def __init__(
        self,
        layer_name: str,
        kcache_stats: dict,
        vcache_stats: dict,
        num_kv_heads_total: int | None = None,
    ):
        self.layer_name = layer_name
        self.kcache_stats = kcache_stats
        self.vcache_stats = vcache_stats
        self._num_heads: int | None = None  # resolved on first call
        # Total KV-head count across all TP ranks (before replication).
        # Passed in from ``setup_kvcache_perhead_hooks``; used to detect
        # the replicated-TP case correctly (see _get_kv_role).
        self._num_kv_heads_total: int | None = num_kv_heads_total
        # Role determines whether this rank tracks K, V or both.  Resolved
        # lazily on first call once we know the real head count, so we can
        # detect TP-replication (world_size > num_kv_heads_total).
        self._role: str | None = None

    def __call__(self, module, input, output):
        _, k, v = input[0], input[1], input[2]
        if not isinstance(k, torch.Tensor):
            return

        with torch.no_grad():
            # Resolve head count on first call and cache it.
            if self._num_heads is None:
                self._num_heads = _get_num_heads_from_tensor(k, module)

            # Resolve K/V role once.  Prefer the externally-provided
            # ``num_kv_heads_total`` (known accurately from model config);
            # fall back to ``H_local * world_size`` only if unset.
            if self._role is None:
                rank, world_size = _get_dist_info()
                if self._num_kv_heads_total is not None and self._num_kv_heads_total > 0:
                    num_kv_heads_total = self._num_kv_heads_total
                else:
                    num_kv_heads_total = self._num_heads * world_size
                self._role = _get_kv_role(rank, world_size, num_kv_heads_total)

            H = self._num_heads

            def _per_head_minmax(t: torch.Tensor):
                """Return (min_vec, max_vec) of shape (num_heads,)."""
                # t shape: (..., H * head_dim)  OR  (..., H, head_dim)
                if t.ndim >= 3 and t.shape[-2] == H:
                    # Already in (..., H, head_dim) layout
                    t_heads = t.reshape(-1, H, t.shape[-1])  # (T, H, D)
                    actual_H = H
                else:
                    # Assume last dim = H * head_dim
                    last = t.shape[-1]
                    if H > 0 and last % H == 0:
                        head_dim = last // H
                        t_heads = t.reshape(-1, H, head_dim)  # (T, H, D)
                        actual_H = H
                    else:
                        # Cannot reshape cleanly; treat as single pseudo-head
                        t_heads = t.reshape(1, -1, 1)  # (1, N, 1)
                        actual_H = 1

                # Reduce over (T, D) → shape (actual_H,)
                t_flat = t_heads.reshape(t_heads.shape[0], actual_H, -1).float()  # (T, H, D)
                h_min = t_flat.min(dim=0).values.min(dim=-1).values  # (H,)
                h_max = t_flat.max(dim=0).values.max(dim=-1).values  # (H,)
                return h_min.cpu(), h_max.cpu()

            # Role-based selective computation:
            # - "both": compute K and V (default / no-replication / single-GPU)
            # - "k"   : compute K only, skip V
            # - "v"   : compute V only, skip K
            if self._role in ("both", "k"):
                k_min, k_max = _per_head_minmax(k)
                k_stats = self.kcache_stats[self.layer_name]
                if k_stats["min"].shape != k_min.shape:
                    # First call – initialise running stats to match head count.
                    k_stats["min"] = k_min.clone()
                    k_stats["max"] = k_max.clone()
                else:
                    k_stats["min"] = torch.minimum(k_stats["min"], k_min)
                    k_stats["max"] = torch.maximum(k_stats["max"], k_max)

            if self._role in ("both", "v"):
                v_min, v_max = _per_head_minmax(v)
                v_stats = self.vcache_stats[self.layer_name]
                if v_stats["min"].shape != v_min.shape:
                    v_stats["min"] = v_min.clone()
                    v_stats["max"] = v_max.clone()
                else:
                    v_stats["min"] = torch.minimum(v_stats["min"], v_min)
                    v_stats["max"] = torch.maximum(v_stats["max"], v_max)


def _all_gather_stats_perhead(
    stats_dict, stats_type="statistics", num_kv_heads_total: int | None = None
):
    """
    Collect per-head KV stats from all TP workers into a full global
    ``(num_kv_heads_total,)`` vector on every rank.

    Two modes are supported transparently:

    1. **No K/V split** (``replication < 2``): each rank owns ``H_local``
       distinct KV heads and holds valid min/max for them.  We scatter
       each rank's slice into its owned head range and do a global
       min/max all_reduce.

    2. **K/V split** (``replication >= 2``, the efficient path for GQA
       with ``tp_size > num_kv_heads``): within each replication group
       only one rank holds valid K stats and another holds valid V stats.
       ``stats_type`` must end with ``"-k"`` or ``"-v"`` so we know which
       side this dict represents.  Ranks that don't own this side
       contribute neutral values (``+inf`` for min, ``-inf`` for max) and
       the opposite-side rank's valid slice wins through the reduce.

    Args:
        stats_dict: per-layer dict of ``{"min": tensor, "max": tensor}``
        stats_type: string label; suffix ``"-k"`` or ``"-v"`` selects role
        num_kv_heads_total: the model's global KV head count.  If ``None``
            we fall back to ``H_local * world_size`` (correct only when
            there is no replication).

    Falls back gracefully to no-op when ``world_size == 1``.

    **Idempotent**: uses a ``_gathered`` flag on ``stats_dict`` to ensure
    the reduce is performed at most once.
    """
    import torch.distributed as dist

    rank, world_size = _get_dist_info()
    if world_size <= 1:
        return rank, world_size

    # Idempotency guard.
    _gathered_key = f"__gathered_{stats_type}__"
    if stats_dict.get(_gathered_key, False):
        if rank == 0:
            print(f"Per-head {stats_type} all-reduce already done, skipping.")
        return rank, world_size

    # Determine which side this dict represents from stats_type suffix.
    side = None
    if stats_type.endswith("-k"):
        side = "k"
    elif stats_type.endswith("-v"):
        side = "v"

    # Probe H_local from this rank's first non-sentinel layer tensor.
    local_H_local = 0
    for name, stats in stats_dict.items():
        if name.startswith("__gathered_"):
            continue
        t = stats["min"]
        if (
            isinstance(t, torch.Tensor)
            and t.ndim == 1
            and t.numel() >= 1
            and not (t.numel() == 1 and torch.isinf(t).all())
        ):
            local_H_local = t.numel()
            break

    # Agree on H_local across all ranks (MAX handles ranks that didn't
    # collect this side under the K/V-split scheme).
    h_local_tensor = torch.tensor([local_H_local], dtype=torch.long, device="cuda")
    dist.all_reduce(h_local_tensor, op=dist.ReduceOp.MAX)
    H_local = int(h_local_tensor.item())
    if H_local <= 0:
        stats_dict[_gathered_key] = True
        return rank, world_size

    # Derive num_kv_heads_total & replication.
    if num_kv_heads_total is None or num_kv_heads_total <= 0:
        # No info: assume no replication.
        num_kv_heads_total = H_local * world_size
        replication = 1
    else:
        replication = (
            max(1, world_size // num_kv_heads_total)
            if num_kv_heads_total > 0 and world_size % num_kv_heads_total == 0
            else 1
        )

    # Decide role & layout for this rank.
    role, heads_per_rank, global_head_offset, _ = _compute_perhead_layout(
        rank, world_size, num_kv_heads_total
    )

    have_data = (role == "both") or (side is None) or (role == side)

    if rank == 0:
        print(
            f"Performing per-head {stats_type} all-reduce across {world_size} workers "
            f"(H_local={H_local}, H_global={num_kv_heads_total}, "
            f"replication={replication}, side={side}, role={role})..."
        )

    for name, stats in stats_dict.items():
        if name.startswith("__gathered_"):  # skip sentinel keys
            continue
        for key in ["min", "max"]:
            neutral = float("inf") if key == "min" else float("-inf")
            full = torch.full((num_kv_heads_total,), neutral, dtype=torch.float32, device="cuda")
            if have_data:
                t = stats[key]
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, dtype=torch.float32)
                t_gpu = t.to(device="cuda", dtype=torch.float32)
                if t_gpu.numel() == H_local:
                    end = global_head_offset + H_local
                    if end <= num_kv_heads_total:
                        full[global_head_offset:end] = t_gpu
                del t_gpu

            op = dist.ReduceOp.MIN if key == "min" else dist.ReduceOp.MAX
            dist.all_reduce(full, op=op)

            stats[key] = full.cpu()
        torch.cuda.empty_cache()

    dist.barrier()
    stats_dict[_gathered_key] = True
    if rank == 0:
        print(f"Per-head {stats_type} all-reduce completed.")
    return rank, world_size


def setup_kvcache_pertensor_hooks(model):
    """
    Register per-tensor (per-layer) kv-cache min/max hooks on all vLLM
    Attention layers.

    Collects a single scalar min/max for each KV cache tensor per layer
    via ``KVCacheHook``.  Stats are stored on the model as
    ``_kcache_stats`` / ``_vcache_stats`` and the registered handles on
    ``_kvcache_hooks``.

    Designed to be passed directly to ``llm.apply_model()``.
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = _find_layers(model, layers=[Attention])
    print(
        f"---------Found {len(kvcache_layers)} kv cache layers to monitor (per-tensor)---------"  # noqa: E501
    )
    for name in list(kvcache_layers.keys())[:5]:
        print(f"  {name}")
    if len(kvcache_layers) > 5:
        print(f"  ... and {len(kvcache_layers) - 5} more kv cache layers")

    if not hasattr(model, "_kcache_stats"):
        model._kcache_stats = {}
        model._vcache_stats = {}
        for name in kvcache_layers.keys():
            model._kcache_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }
            model._vcache_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    if not hasattr(model, "_kvcache_hooks"):
        model._kvcache_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCacheHook(name, model._kcache_stats, model._vcache_stats)
            hook_handle = layer.register_forward_hook(hook)
            model._kvcache_hooks.append(hook_handle)

    return f"Registered {len(model._kvcache_hooks)} kv-pertensor hooks"


def setup_kvcache_perhead_hooks(model):
    """
    Register per-head kv-cache min/max hooks on all vLLM Attention layers.

    Like ``setup_kvcache_only_hooks`` but collects one min/max per KV head
    instead of one per layer.  The stats tensors start as scalar ``inf`` /
    ``-inf`` and are replaced with ``(num_heads,)`` vectors on the first
    forward call (when the actual head count is known).

    Designed to be passed directly to ``llm.apply_model()``.
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = _find_layers(model, layers=[Attention])
    print(f"[KVPerHead] Found {len(kvcache_layers)} Attention layers to monitor")

    # Try to discover the model's total KV-head count so hooks can make an
    # informed K/V-split decision.  This is needed because inside the hook
    # we only see the *local* (per-TP-rank) head count, which under
    # replication (tp_size > num_kv_heads_total) is smaller than the true
    # global value.
    num_kv_heads_total = _infer_num_kv_heads_total(model)
    model._kvcache_num_kv_heads_total = num_kv_heads_total
    print(f"[KVPerHead] Inferred num_kv_heads_total = {num_kv_heads_total}")

    if not hasattr(model, "_kvcache_perhead_stats"):
        model._kvcache_perhead_stats = {"k": {}, "v": {}}
        for name in kvcache_layers:
            # Scalar sentinels – will be replaced by (H,) vectors on first hook call.
            model._kvcache_perhead_stats["k"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }
            model._kvcache_perhead_stats["v"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    if not hasattr(model, "_kvcache_perhead_hooks"):
        model._kvcache_perhead_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCachePerHeadHook(
                name,
                model._kvcache_perhead_stats["k"],
                model._kvcache_perhead_stats["v"],
                num_kv_heads_total=num_kv_heads_total,
            )
            handle = layer.register_forward_hook(hook)
            model._kvcache_perhead_hooks.append(handle)

    return f"Registered {len(model._kvcache_perhead_hooks)} kv-perhead hooks"


def get_kvcache_perhead_stats(model):
    """
    Retrieve per-head kv-cache statistics.

    Returns a dict with keys like::

        "model.layers.0.self_attn.attn.k_cache"

    and values::

        {"min": [float, ...],   # list of length num_kv_heads
         "max": [float, ...]}

    This is intentionally different from the per-tensor format so that
    downstream tools can distinguish the two.  Returns ``None`` if no
    stats are available.
    """
    if not hasattr(model, "_kvcache_perhead_stats"):
        return None

    num_kv_heads_total = getattr(model, "_kvcache_num_kv_heads_total", None)
    try:
        _all_gather_stats_perhead(
            model._kvcache_perhead_stats["k"],
            stats_type="kv-perhead-k",
            num_kv_heads_total=num_kv_heads_total,
        )
        _all_gather_stats_perhead(
            model._kvcache_perhead_stats["v"],
            stats_type="kv-perhead-v",
            num_kv_heads_total=num_kv_heads_total,
        )
    except Exception as e:
        print(f"[KVPerHead] Warning: Could not perform all-gather: {e}")

    def _to_list(t):
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return [float(t)]

    stats_dict = {}
    for name, stats in model._kvcache_perhead_stats["k"].items():
        if name.startswith("__gathered_"):  # skip sentinel keys
            continue
        stats_dict[f"{name}.k_cache"] = {
            "min": _to_list(stats["min"]),
            "max": _to_list(stats["max"]),
        }
    for name, stats in model._kvcache_perhead_stats["v"].items():
        if name.startswith("__gathered_"):  # skip sentinel keys
            continue
        stats_dict[f"{name}.v_cache"] = {
            "min": _to_list(stats["min"]),
            "max": _to_list(stats["max"]),
        }
    return stats_dict


def print_kvcache_perhead_stats(model):
    """
    Print per-head kv-cache statistics.  Only rank-0 prints.
    Designed to be passed to ``llm.apply_model()``.
    """
    if not hasattr(model, "_kvcache_perhead_stats"):
        print("[KVPerHead] No per-head kv statistics available.")
        return

    num_kv_heads_total = getattr(model, "_kvcache_num_kv_heads_total", None)
    try:
        rank, world_size = _all_gather_stats_perhead(
            model._kvcache_perhead_stats["k"],
            stats_type="kv-perhead-k",
            num_kv_heads_total=num_kv_heads_total,
        )
        _all_gather_stats_perhead(
            model._kvcache_perhead_stats["v"],
            stats_type="kv-perhead-v",
            num_kv_heads_total=num_kv_heads_total,
        )
    except Exception as e:
        print(f"[KVPerHead] Warning: Could not perform all-gather: {e}")
        rank, world_size = 0, 1

    if rank != 0:
        return

    if world_size > 1:
        print(f"\n[KVPerHead] Global statistics across {world_size} workers")

    def _fmt(stats, label):
        print(f"\n  === {label} ===")
        for name, s in stats.items():
            if name.startswith("__gathered_"):  # skip sentinel keys
                continue
            mn = s["min"]
            mx = s["max"]
            if isinstance(mn, torch.Tensor):
                mn_str = f"[{mn.min().item():.4f} .. {mn.max().item():.4f}] ({mn.numel()} heads)"
                mx_str = f"[{mx.min().item():.4f} .. {mx.max().item():.4f}]"
            else:
                mn_str = str(mn)
                mx_str = str(mx)
            print(f"    {name}: min={mn_str}  max={mx_str}")

    _fmt(model._kvcache_perhead_stats["k"], "Per-Head K-cache Statistics")
    _fmt(model._kvcache_perhead_stats["v"], "Per-Head V-cache Statistics")


def remove_kvcache_perhead_hooks(model):
    """
    Remove hooks registered by ``setup_kvcache_perhead_hooks``.
    Designed to be passed to ``llm.apply_model()``.
    """
    if hasattr(model, "_kvcache_perhead_hooks"):
        for h in model._kvcache_perhead_hooks:
            h.remove()
        del model._kvcache_perhead_hooks
    if hasattr(model, "_kvcache_perhead_stats"):
        del model._kvcache_perhead_stats
    return "KV-perhead hooks removed"


# =============================================================================
# KV-cache only calibration (lightweight: no Linear / MoE hooks)
# =============================================================================


def setup_kvcache_only_hooks(model):
    """
    Register **only** kv-cache min/max statistic hooks on vLLM Attention layers.

    This is a lightweight alternative to ``setup_activation_hooks`` for the
    use-case where you only need kv-cache scales (e.g. fp8 kv-cache without
    weight quantisation).  No linear-layer or MoE hooks are registered, so
    the memory footprint and hook overhead are minimal.

    Designed to be passed directly to ``llm.apply_model()``.

    Returns:
        A human-readable status string (e.g. "Registered 80 kv-only hooks").
    """
    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    kvcache_layers = _find_layers(model, layers=[Attention])
    print(f"[KVOnly] Found {len(kvcache_layers)} Attention layers to monitor")

    if not hasattr(model, "_kvcache_only_stats"):
        model._kvcache_only_stats = {"k": {}, "v": {}}
        for name in kvcache_layers:
            model._kvcache_only_stats["k"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }
            model._kvcache_only_stats["v"][name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    if not hasattr(model, "_kvcache_only_hooks"):
        model._kvcache_only_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCacheHook(
                name,
                model._kvcache_only_stats["k"],
                model._kvcache_only_stats["v"],
            )
            handle = layer.register_forward_hook(hook)
            model._kvcache_only_hooks.append(handle)

    return f"Registered {len(model._kvcache_only_hooks)} kv-only hooks"


def get_kvcache_only_stats(model):
    """
    Retrieve kv-cache min/max statistics collected by ``setup_kvcache_only_hooks``.

    Performs an all-reduce across workers so every rank holds the global
    min/max before rank-0 returns the final dict.

    Returns:
        dict with keys like ``"model.layers.0.self_attn.attn.k_cache"`` and
        values ``{"min": float, "max": float}``, matching the format produced
        by ``get_activation_stats()`` so that ``KVScaleSearcher`` can consume
        it directly.  Returns ``None`` if no stats are available.
    """
    if not hasattr(model, "_kvcache_only_stats"):
        return None

    try:
        _all_reduce_stats(model._kvcache_only_stats["k"], stats_type="kv-only-k")
        _all_reduce_stats(model._kvcache_only_stats["v"], stats_type="kv-only-v")
    except Exception as e:
        print(f"[KVOnly] Warning: Could not perform all-reduce: {e}")

    stats_dict = {}
    for name, stats in model._kvcache_only_stats["k"].items():
        stats_dict[f"{name}.k_cache"] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    for name, stats in model._kvcache_only_stats["v"].items():
        stats_dict[f"{name}.v_cache"] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    return stats_dict


def print_kvcache_only_stats(model):
    """
    Print kv-cache-only statistics.  Only rank-0 prints.
    Designed to be passed to ``llm.apply_model()``.
    """
    if not hasattr(model, "_kvcache_only_stats"):
        print("[KVOnly] No kv-only statistics available.")
        return

    try:
        rank, world_size = _all_reduce_stats(
            model._kvcache_only_stats["k"], stats_type="kv-only-k"
        )
        _all_reduce_stats(model._kvcache_only_stats["v"], stats_type="kv-only-v")
    except Exception as e:
        print(f"[KVOnly] Warning: Could not perform all-reduce: {e}")
        rank, world_size = 0, 1

    if rank != 0:
        return

    if world_size > 1:
        print(f"\n[KVOnly] Global statistics across {world_size} workers")
    _print_stats_table(model._kvcache_only_stats["k"], "KV-Only K-cache Statistics")
    _print_stats_table(model._kvcache_only_stats["v"], "KV-Only V-cache Statistics")


def remove_kvcache_only_hooks(model):
    """
    Remove hooks registered by ``setup_kvcache_only_hooks``.
    Designed to be passed to ``llm.apply_model()``.
    """
    if hasattr(model, "_kvcache_only_hooks"):
        for h in model._kvcache_only_hooks:
            h.remove()
        del model._kvcache_only_hooks
    if hasattr(model, "_kvcache_only_stats"):
        del model._kvcache_only_stats
    return "KV-only hooks removed"


# =============================================================================
# MoE statistics collection (vLLM patch entry point lives here)
# =============================================================================


def collect_fused_moe_internal_stats(
    stage,
    hidden_states,
    topk_ids,
    global_num_experts,
    layer_name=None,
    global_moe_activation_stats=None,
):
    """
    Collect FusedMoE internal activation statistics and accumulate in global dictionary.
    Only collects stats during actual generation (skips CUDA graph capture phase).

    Args:
        stage: "gate_up_proj" or "down_proj"
        hidden_states: Input tensor [num_tokens, hidden_size] or [num_tokens*top_k, hidden_size]
        topk_ids: Expert IDs [num_tokens, top_k]
        global_num_experts: Total number of experts
        layer_name: Layer name for identification (if None, will try to get from context)
        global_moe_activation_stats: Global dictionary to store statistics

    Environment Variables:
        VLLM_MOE_COLLECT_STATS: Set to "1" to enable statistics collection
        VLLM_MOE_COLLECT_STATS_VERBOSE: Set to "1" to enable verbose debug output
    """
    # Use os.getenv directly instead of vllm.envs to avoid caching issues in Ray workers
    # Check if MoE stats collection is enabled
    if os.getenv("VLLM_MOE_COLLECT_STATS", "0") != "1":
        return

    # Check verbose flag (default off to avoid hang in distributed setting)
    verbose = os.getenv("VLLM_MOE_COLLECT_STATS_VERBOSE", "0") == "1"

    #
    if global_moe_activation_stats is None:
        return

    # Skip if layer_name is not provided (weight not properly initialized yet)
    if layer_name is None:
        return

    # Only collect stats for MoE layers (should contain "experts" in the name)
    if "experts" not in layer_name.lower():
        return

    # Get rank information
    rank, world_size = _get_dist_info()

    # Collect statistics
    key = f"{layer_name}.{stage}"
    with torch.no_grad():
        # --- Layer-level (overall) stats ---
        if key in global_moe_activation_stats:
            stats = global_moe_activation_stats[key]
            act_min = hidden_states.min().detach().cpu()
            act_max = hidden_states.max().detach().cpu()
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: Collected MoE stats "
                    f"for {key}, min: {act_min.item()}, max: {act_max.item()}"
                )
            stats["min"] = torch.minimum(stats["min"], act_min)
            stats["max"] = torch.maximum(stats["max"], act_max)
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: "
                    f"Updated MoE stats for {key}, min: {stats['min'].item()}, "
                    f"max: {stats['max'].item()}"
                )
            stats["call_count"] = stats.get("call_count", 0) + 1

        # --- Per-expert stats (only when enabled) ---
        if os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") != "1":
            return

        # topk_ids shape: [num_tokens, top_k], hidden_states shape: [num_tokens, hidden_size]
        # For down_proj stage, hidden_states may be [num_tokens * top_k, hidden_size]
        num_tokens_hs = hidden_states.shape[0]
        num_tokens_topk = topk_ids.shape[0]
        top_k = topk_ids.shape[1]

        if num_tokens_hs == num_tokens_topk:
            # gate_up_proj: hidden_states is [num_tokens, hidden_size]
            # Each token may be assigned to multiple experts, use the same hidden_state for each
            flat_expert_ids = topk_ids.reshape(-1)  # [num_tokens * top_k]
            flat_hidden = (
                hidden_states.unsqueeze(1)
                .expand(-1, top_k, -1)
                .reshape(-1, hidden_states.shape[-1])
            )  # [num_tokens * top_k, hidden_size]
        elif num_tokens_hs == num_tokens_topk * top_k:
            # down_proj: hidden_states is [num_tokens * top_k, hidden_size]
            flat_expert_ids = topk_ids.reshape(-1)  # [num_tokens * top_k]
            flat_hidden = hidden_states  # already [num_tokens * top_k, hidden_size]
        else:
            # Fallback: skip per-expert stats if shape doesn't match
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: Skipping per-expert "
                    f"stats for {key}, shape mismatch: "
                    f"hidden_states={hidden_states.shape}, topk_ids={topk_ids.shape}"
                )
            return

        # Iterate over each unique expert in the current batch
        unique_experts = flat_expert_ids.unique()
        for expert_id_tensor in unique_experts:
            expert_id = expert_id_tensor.item()
            if expert_id < 0:
                continue  # Skip invalid expert ids (e.g., -1 padding)
            expert_key = f"{layer_name}.{expert_id}.{stage}"
            if expert_key not in global_moe_activation_stats:
                # Dynamically create entry if not pre-allocated
                global_moe_activation_stats[expert_key] = {
                    "min": torch.tensor(float("inf")),
                    "max": torch.tensor(float("-inf")),
                }
            expert_stats = global_moe_activation_stats[expert_key]
            mask = flat_expert_ids == expert_id_tensor
            expert_hidden = flat_hidden[mask]
            if expert_hidden.numel() == 0:
                continue
            e_min = expert_hidden.min().detach().cpu()
            e_max = expert_hidden.max().detach().cpu()
            expert_stats["min"] = torch.minimum(expert_stats["min"], e_min)
            expert_stats["max"] = torch.maximum(expert_stats["max"], e_max)
            expert_stats["call_count"] = expert_stats.get("call_count", 0) + 1
            if verbose:
                print(
                    f"[VERBOSE] Rank {rank}/{world_size}: Expert {expert_id} "
                    f"stats for {key}, min: {e_min.item()}, max: {e_max.item()}"
                )


def get_moe_stats(model):
    """
    Retrieve moe statistics from the model.
    Performs all-reduce across all workers to get global min/max.
    """
    if not hasattr(model, "_moe_activation_stats"):
        return None

    # Perform all-reduce to get global min/max across all workers
    try:
        _all_reduce_stats(model._moe_activation_stats, stats_type="MoE")
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")

    # Convert tensors to Python scalars for easier use
    stats_dict = {}
    for name, stats in model._moe_activation_stats.items():
        stats_dict[name] = {
            "min": stats["min"].item() if isinstance(stats["min"], torch.Tensor) else stats["min"],
            "max": stats["max"].item() if isinstance(stats["max"], torch.Tensor) else stats["max"],
        }
    return stats_dict


def print_moe_stats(model, verbose=False):
    """
    Print MoE activation statistics in a readable format.
    Performs all-reduce to get global statistics across all workers.

    Args:
        model: The model containing MoE activation statistics
        verbose: If True, print detailed debug information during all-reduce
    """
    if not hasattr(model, "_moe_activation_stats"):
        print("No MoE activation statistics available")
        return

    # Perform all-reduce to get global min/max
    try:
        rank, world_size = _all_reduce_stats(
            model._moe_activation_stats, stats_type="MoE", verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not perform all-reduce: {e}")
        rank, world_size = 0, 1

    # Only rank 0 prints the statistics (or single process)
    if rank != 0:
        return

    # Print statistics
    if world_size > 1:
        print(f"\n[Global statistics across {world_size} workers]")
    _print_stats_table(model._moe_activation_stats, "MoE gate_up and down Statistics")


# =============================================================================
# MTP (Multi-Token Prediction) draft model activation hooks
# =============================================================================


def setup_mtp_activation_hooks(draft_model):
    """
    Setup activation hooks on the MTP draft model to collect min/max statistics.
    This function should be applied via llm.apply_draft_model(setup_mtp_activation_hooks).

    The MTP draft model (e.g., HYV3MTP) is a separate nn.Module from the main model,
    containing its own Linear, Attention, and FusedMoE layers that need independent
    hook registration.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase

    try:
        # vLLM ≥ 0.20 (Tencent custom): Attention moved under model_executor.
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        # Older vLLM layout.
        from vllm.attention.layer import Attention

    # Find all linear layers in the MTP model
    layers_to_monitor = _find_layers(draft_model, layers=[torch.nn.Linear, LinearBase])
    kvcache_layers = _find_layers(draft_model, layers=[Attention])
    print(f"---------[MTP] Found {len(layers_to_monitor)} layers to monitor---------")
    print(f"---------[MTP] Found {len(kvcache_layers)} kv cache layers to monitor---------")
    for name in list(layers_to_monitor.keys())[:5]:
        print(f"  [MTP] {name}")
    if len(layers_to_monitor) > 5:
        print(f"  ... and {len(layers_to_monitor) - 5} more MTP activation layers")

    for name in list(kvcache_layers.keys())[:5]:
        print(f"  [MTP] {name}")
    if len(kvcache_layers) > 5:
        print(f"  ... and {len(kvcache_layers) - 5} more MTP kv cache layers")

    # Initialize activation statistics storage (prefix with "mtp." to distinguish)
    if not hasattr(draft_model, "_activation_stats"):
        draft_model._activation_stats = {}
        for name in layers_to_monitor.keys():
            draft_model._activation_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    if not hasattr(draft_model, "_kcache_stats"):
        draft_model._kcache_stats = {}
        draft_model._vcache_stats = {}
        for name in kvcache_layers.keys():
            draft_model._kcache_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }
            draft_model._vcache_stats[name] = {
                "min": torch.tensor(float("inf")),
                "max": torch.tensor(float("-inf")),
            }

    # Register hooks for all linear layers
    if not hasattr(draft_model, "_activation_hooks"):
        draft_model._activation_hooks = []
        for name, layer in layers_to_monitor.items():
            hook = ActivationHook(name, draft_model._activation_stats)
            hook_handle = layer.register_forward_hook(hook)
            draft_model._activation_hooks.append(hook_handle)

    if not hasattr(draft_model, "_kvcache_hooks"):
        draft_model._kvcache_hooks = []
        for name, layer in kvcache_layers.items():
            hook = KVCacheHook(name, draft_model._kcache_stats, draft_model._vcache_stats)
            hook_handle = layer.register_forward_hook(hook)
            draft_model._kvcache_hooks.append(hook_handle)

    # Register MoE statistics storage and hooks
    moe_layers = _find_layers(draft_model, layers=[FusedMoE])
    if moe_layers:
        print(f"---------[MTP] Found {len(moe_layers)} MoE layers to monitor---------")
        for name in list(moe_layers.keys())[:5]:
            print(f"  [MTP] {name}")
        if len(moe_layers) > 5:
            print(f"  ... and {len(moe_layers) - 5} more")

        per_expert = os.getenv("VLLM_MOE_COLLECT_PER_EXPERT_STATS", "0") == "1"

        if not hasattr(draft_model, "_moe_activation_stats"):
            draft_model._moe_activation_stats = {}
            for name, layer in moe_layers.items():
                num_experts = getattr(layer, "global_num_experts", None)
                if num_experts is None:
                    num_experts = getattr(layer, "num_experts", 256)

                for stage in ["gate_up_proj", "down_proj"]:
                    draft_model._moe_activation_stats[f"{name}.{stage}"] = {
                        "min": torch.tensor(float("inf")),
                        "max": torch.tensor(float("-inf")),
                    }
                    if per_expert:
                        for expert_id in range(num_experts):
                            draft_model._moe_activation_stats[f"{name}.{expert_id}.{stage}"] = {
                                "min": torch.tensor(float("inf")),
                                "max": torch.tensor(float("-inf")),
                            }

                if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                    layer.w13_weight._vllm_layer_name = name
                    layer.w13_weight._moe_activation_stats_of_model = (
                        draft_model._moe_activation_stats
                    )

    print("---------[MTP] Activation hooks registered---------")
    return f"[MTP] Registered {len(draft_model._activation_hooks)} hooks"


def get_mtp_activation_stats(draft_model):
    """
    Retrieve activation statistics from the MTP draft model.
    Performs all-reduce across all workers to get global min/max.
    """
    return get_activation_stats(draft_model)


def print_mtp_activation_stats(draft_model):
    """Print MTP draft model activation statistics."""
    if not hasattr(draft_model, "_activation_stats"):
        print("[MTP] No activation statistics available")
        return
    print_activation_stats(draft_model)


def get_mtp_moe_stats(draft_model):
    """Retrieve MoE statistics from the MTP draft model."""
    return get_moe_stats(draft_model)


def print_mtp_moe_stats(draft_model, verbose=False):
    """Print MTP draft model MoE statistics."""
    if not hasattr(draft_model, "_moe_activation_stats"):
        print("[MTP] No MoE activation statistics available")
        return
    print_moe_stats(draft_model, verbose=verbose)
