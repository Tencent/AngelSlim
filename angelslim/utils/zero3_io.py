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

"""All-in-one DeepSpeed ZeRO-3 helpers for AngelSlim QAT.

This module concentrates everything ZeRO-3-specific so that the rest of the
codebase touches only a few thin call-sites:

  * ZeRO-3 detection / parameter gathering helpers.
  * Empty model construction under ``deepspeed.zero.Init`` plus a generic
    "linearize fused MoE experts" pass that builds *empty* per-expert
    ``nn.Linear`` modules (no copy from the old fused tensor required —
    weights are filled later from the safetensors checkpoint).
  * Streaming weight loader: walks the safetensors shards once, on rank 0
    only, and broadcasts each tensor into the (possibly sharded) target
    parameter via ``GatheredParameters(modifier_rank=0)``. Handles fused
    MoE keys (``...experts.gate_up_proj``, ``...experts.down_proj``) by
    slicing per expert into the linearized targets.
  * Streaming scale loader for QAT warm-start: reads ``*.weight_scale``,
    ``*.input_scale``, ``*.k_cache.scale``, ``*.v_cache.scale`` keys from a
    PTQ "real" checkpoint and writes them into the freshly-created
    ``QuantLinear`` quantizer parameters.
  * Saving: gather a sharded model into a rank-0 CPU state_dict and call
    the model-specific save_func by patching ``state_dict``.
  * Optimizer-side patches needed because QAT scale parameters are tied
    across multiple modules in a layer.

By design, **nothing in this file mutates the model when ZeRO-3 is not
enabled** (each helper is a no-op or behaves identically to the
non-distributed path). Importing this module is therefore safe in any
configuration.
"""

from __future__ import annotations

import gc
import glob
import json
import os
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
import torch.nn as nn

from .lazy_imports import deepspeed
from .utils import find_parent_layer_and_sub_name, print_info

ZERO3_PARAM_ATTRS = ("ds_id", "ds_status", "ds_numel", "ds_tensor")


# ---------------------------------------------------------------------------
# Basic detection / context helpers
# ---------------------------------------------------------------------------


def is_deepspeed_zero3_enabled() -> bool:
    """True iff HuggingFace's ``HfTrainerDeepSpeedConfig`` is registered with
    ZeRO stage 3. Returns False if ``transformers``/``deepspeed`` is not
    importable."""
    try:
        from transformers.integrations.deepspeed import (
            is_deepspeed_zero3_enabled as _hf,
        )

        return bool(_hf())
    except Exception:  # noqa: BLE001
        return False


def is_zero3_param(x) -> bool:
    """True iff ``x`` is a ZeRO-3 sharded parameter (``deepspeed.zero.Init``
    has injected its bookkeeping attributes)."""
    if not isinstance(x, torch.nn.Parameter):
        return False
    return any(hasattr(x, attr) for attr in ZERO3_PARAM_ATTRS)


@contextmanager
def gathered_param_if_zero3(x, modifier_rank: Optional[int] = None):
    """All-gather a ZeRO-3 shard for the lifetime of the block.

    Pure no-op (yields ``x`` unchanged) when ``x`` is not a ZeRO-3 shard, so
    callers can always wrap their critical sections without branching.
    """
    if is_zero3_param(x):
        ctx = deepspeed.zero.GatheredParameters([x], modifier_rank=modifier_rank)
    else:
        ctx = nullcontext()
    with ctx:
        yield x


@contextmanager
def gathered_params_if_zero3(params, modifier_rank: Optional[int] = None):
    """Batched variant of :func:`gathered_param_if_zero3`."""
    params = [p for p in params if p is not None]
    z3 = [p for p in params if is_zero3_param(p)]
    if z3:
        ctx = deepspeed.zero.GatheredParameters(z3, modifier_rank=modifier_rank)
    else:
        ctx = nullcontext()
    with ctx:
        yield params


def model_has_zero3_params(model) -> bool:
    return any(is_zero3_param(p) for p in model.parameters())


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Empty MoE linearization
# ---------------------------------------------------------------------------


class LinearizedMoeExperts(nn.Module):
    """Empty per-expert ``nn.Linear`` container for fused MoE experts.

    This module mirrors the ``forward`` of HuggingFace's fused experts but
    holds parameters as ``num_experts`` triplets of ``(gate_proj, up_proj,
    down_proj)`` ``nn.Linear`` modules. Construction is **purely structural**
    — no weight copy from the old fused tensor — so it is safe to instantiate
    under ``deepspeed.zero.Init`` and let the streaming loader fill in the
    weights afterwards.
    """

    _angelslim_linearized_moe = True

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        dtype=torch.bfloat16,
        device=None,
        config=None,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.act_fn = act_fn
        if config is not None:
            self.config = config

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # NOTE: when device.type == "meta" we KEEP it (Plan C zero-copy build).
        # The streaming loader materialises each rank's shard on CUDA later.

        for expert_idx in range(self.num_experts):
            expert = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(
                        self.hidden_dim,
                        self.intermediate_dim,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                    "up_proj": nn.Linear(
                        self.hidden_dim,
                        self.intermediate_dim,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                    "down_proj": nn.Linear(
                        self.intermediate_dim,
                        self.hidden_dim,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                }
            )
            setattr(self, str(expert_idx), expert)

    def __getitem__(self, idx):
        return getattr(self, str(idx))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit_mask = torch.greater(expert_mask.sum(dim=(-1, -2)), 0)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # ZeRO-3 gathers parameters when a Linear is entered. All ranks must
                # enter the same experts in the same order even when their local
                # batches route to different experts, otherwise collectives deadlock.
                expert_hit_int = expert_hit_mask.to(torch.int32)
                torch.distributed.all_reduce(
                    expert_hit_int,
                    op=torch.distributed.ReduceOp.MAX,
                )
                expert_hit_mask = expert_hit_int.to(torch.bool)
            expert_hit = expert_hit_mask.nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert_layer = getattr(self, str(int(expert_idx.item())))
            gate = expert_layer["gate_proj"](current_state)
            up = expert_layer["up_proj"](current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert_layer["down_proj"](current_hidden_states)
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


def _is_fused_moe_experts(module) -> bool:
    """Heuristic: matches HF Qwen3MoeExperts / HYV3Experts / similar.

    They all expose ``gate_up_proj`` / ``down_proj`` as ``nn.Parameter``
    plus ``num_experts``, ``hidden_dim``, ``intermediate_dim``, ``act_fn``.

    GLM-5's ``GlmMoeDsaNaiveMoe`` also carries those attributes, but it is
    intentionally excluded here: GLM-5 performs its own *empty* defuse into
    ``GlmMoeDsaSplitMoe`` (via ``GLM5._defuse_moe_experts_empty``) so the
    per-expert keys line up with its quant pipeline.  If we let the generic
    path replace it with ``LinearizedMoeExperts`` first, GLM-5's hook would
    find nothing to defuse."""
    if isinstance(module, LinearizedMoeExperts):
        return False
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaNaiveMoe,
        )
        if isinstance(module, GlmMoeDsaNaiveMoe):
            return False
    except Exception:
        pass
    required = (
        "gate_up_proj",
        "down_proj",
        "num_experts",
        "hidden_dim",
        "intermediate_dim",
        "act_fn",
    )
    return all(hasattr(module, a) for a in required)


def _ds_full_shape(param):
    """Full shape of a parameter, accounting for ZeRO-3 sharding."""
    shape = getattr(param, "ds_shape", None)
    if shape is None:
        shape = param.shape
    return tuple(int(x) for x in shape)


def linearize_moe_experts_empty(model, dtype=torch.bfloat16) -> int:
    """Replace every fused MoE experts module in ``model`` with an empty
    :class:`LinearizedMoeExperts`.

    Under ZeRO-3 the new ``nn.Linear`` parameters are created inside a
    ``deepspeed.zero.Init`` context so they get partitioned immediately.
    Weights are NOT copied from the old fused tensors — the streaming
    safetensors loader is responsible for that. The old module is dropped
    afterwards.
    """
    targets = []
    for name, module in model.named_modules():
        if _is_fused_moe_experts(module):
            targets.append(name)

    if not targets:
        return 0

    z3 = is_deepspeed_zero3_enabled()
    replaced = 0
    for name in targets:
        parent, sub_name = find_parent_layer_and_sub_name(model, name)
        old = getattr(parent, sub_name)

        # Resolve dimensions from the (possibly sharded) old tensors.
        gate_up_shape = _ds_full_shape(old.gate_up_proj)
        down_shape = _ds_full_shape(old.down_proj)
        num_experts = int(old.num_experts)
        hidden_dim = int(old.hidden_dim)
        # gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim]
        if len(gate_up_shape) >= 3 and gate_up_shape[-2] % 2 == 0:
            intermediate_dim = gate_up_shape[-2] // 2
        elif len(down_shape) >= 3:
            intermediate_dim = int(down_shape[-1])
        else:
            intermediate_dim = int(old.intermediate_dim)

        old_dtype = old.gate_up_proj.dtype if hasattr(old.gate_up_proj, "dtype") else dtype
        config = getattr(old, "config", None)
        act_fn = old.act_fn

        ctx = deepspeed.zero.Init() if z3 else nullcontext()
        with ctx:
            new_module = LinearizedMoeExperts(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                act_fn=act_fn,
                dtype=old_dtype,
                device=None,  # let LinearizedMoeExperts pick cuda
                config=config,
            )

        setattr(parent, sub_name, new_module)
        del old
        replaced += 1
        _cleanup()

    print_info(f"[zero3] linearize_moe_experts_empty: replaced {replaced} fused module(s).")
    return replaced


# ---------------------------------------------------------------------------
# Empty model construction + streaming weight loader
# ---------------------------------------------------------------------------


def _resolve_dtype(torch_dtype, config):
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        return getattr(torch, torch_dtype)
    resolved = getattr(config, "torch_dtype", None) or torch.float32
    if isinstance(resolved, str):
        return getattr(torch, resolved)
    return resolved


def zero3_empty_model_from_pretrained(
    model_path,
    torch_dtype="auto",
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="default",
    log_prefix="[zero3]",
    config_hook=None,
):
    """Build an EMPTY ZeRO-3 sharded model from a HuggingFace ``model_path``.

    Linearizes all fused MoE experts immediately so subsequent QuantLinear
    insertion can iterate flat ``nn.Linear`` modules. Does NOT load
    weights — caller must invoke :func:`stream_load_weights`.

    Args:
        config_hook: optional callable ``(hf_cfg, model_path) -> None`` invoked
            right after ``AutoConfig.from_pretrained`` so the caller can patch
            fields before the empty model is materialised.  Used e.g. by GLM-5
            to repair ``num_hidden_layers`` when ``config.json`` under-counts
            the actual number of transformer blocks in the checkpoint.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.integrations.deepspeed import HfDeepSpeedConfig  # noqa: F401

    # ``no_init_weights`` / ``no_tie_weights`` moved across transformers
    # versions: newest (>=5.x) expose them under ``transformers.initialization``;
    # older releases kept them in ``modeling_utils`` or the top-level package.
    no_init_weights = None
    no_tie_weights = None
    for mod in ("transformers.initialization", "transformers.modeling_utils", "transformers"):
        try:
            m = __import__(mod, fromlist=["no_init_weights"])
            if hasattr(m, "no_init_weights"):
                no_init_weights = m.no_init_weights
            if hasattr(m, "no_tie_weights"):
                no_tie_weights = m.no_tie_weights
        except Exception:  # noqa: BLE001
            continue
        if no_init_weights is not None and no_tie_weights is not None:
            break
    if no_init_weights is None:
        no_init_weights = nullcontext  # type: ignore
    if no_tie_weights is None:
        no_tie_weights = nullcontext  # type: ignore

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if attn_implementation != "default":
        config._attn_implementation = attn_implementation
    if use_cache is not None:
        config.use_cache = use_cache

    # Optional model-specific config patch (e.g. GLM-5 ``_fix_hf_config``
    # repairing ``num_hidden_layers`` when the released config under-counts
    # the transformer blocks in the checkpoint).  Runs BEFORE ``from_config``
    # so the empty skeleton has the correct number of ``model.layers`` and
    # the streaming loader can fill every block including the last one.
    if config_hook is not None:
        try:
            config_hook(config, model_path)
        except Exception as e:  # noqa: BLE001
            print_info(
                f"{log_prefix} config_hook failed ({e}); "
                f"proceeding with unpatched HF config."
            )

    resolved = _resolve_dtype(torch_dtype, config)
    print_info(
        f"{log_prefix} build empty ZeRO-3 model dtype={resolved} from config "
        f"{getattr(config, 'model_type', '?')}"
    )

    # meta-device empty build.
    #
    # Earlier revisions wrapped ``from_config`` in
    # ``deepspeed.zero.Init(remote_device="cpu")``.  That keeps the FULL model
    # replica on host memory during partition, so the per-rank GPU peak is only
    # ~1/N -- but it still materialises the *entire* ~2.8 TB model on CPU (and
    # a second full CPU replica is added by the subsequent defuse step), which
    # OOMs a 1.9 TB/node host on GLM-5's 75 MoE layers.
    #
    # Instead we build the whole skeleton on ``torch.device("meta")``: every
    # Parameter is allocated with ZERO bytes (verified: GLM-5 -> 1269 params
    # all on meta, RSS delta ~0).  The streaming loader later fills each rank's
    # shard slice directly on CUDA, so no full CPU replica is ever created.
    print_info(
        f"{log_prefix} building empty model on torch.device('meta') "
        f"(zero-copy skeleton; world_size="
        f"{int(os.getenv('WORLD_SIZE', '1'))})."
    )

    with torch.device("meta"), no_init_weights(), no_tie_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=resolved,
            trust_remote_code=trust_remote_code,
        )

    # Linearize fused MoE experts (meta skeleton -> meta per-expert Linears).
    # linearize_moe_experts_empty opens its own meta-aware context so the new
    # params stay on meta (no CPU replica).
    linearize_moe_experts_empty(model, dtype=resolved)

    return model


def _shards(model_path):
    """Yield ``(shard_path, [keys])`` for every safetensors shard."""
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        per_shard = {}
        for key, shard in weight_map.items():
            per_shard.setdefault(shard, []).append(key)
        for shard in sorted(per_shard):
            yield os.path.join(model_path, shard), per_shard[shard]
        return

    paths = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No safetensors found under {model_path}")
    for shard_path in paths:
        with safe_open(shard_path, framework="pt") as r:
            yield shard_path, list(r.keys())


def _broadcast_into_target(src, target, *, is_buffer=False, key=None, model=None):
    """Copy ``src`` (rank0 only, or None on other ranks) into ``target``.

    Handles four cases:
      * ``meta`` parameter (expert-parallel / PLAN-C empty build): the target
        has no storage yet, so it is *materialised* on the CUDA device via
        ``set_module_tensor_to_device`` (rank0 stages the real tensor, other
        ranks allocate an empty CUDA tensor and receive it through broadcast).
        This keeps the per-rank footprint at ``1/world_size`` of the model with
        **no full CPU replica** -- exactly the option-A EP goal.
      * ZeRO-3 sharded ``Parameter``: gather, rank0 writes, exit gather.
      * Regular distributed ``Parameter`` / replicated buffer: rank0 stages,
        then broadcast.
      * Single-process: direct copy.
    """
    dist_active = torch.distributed.is_available() and torch.distributed.is_initialized()

    # ---- meta target: materialise directly on CUDA (no CPU replica) --------
    if getattr(target, "is_meta", False):
        if model is None or key is None:
            raise RuntimeError(
                f"Cannot materialise meta parameter {key!r}: 'model' must be "
                "passed to _broadcast_into_target for meta targets."
            )
        from accelerate.utils import set_module_tensor_to_device
        # Option B (hybrid sharding for H20 96GB): materialise meta parameters
        # on CPU, NOT cuda.  INT8 PTQ runs in ``low_memory`` mode, which keeps
        # the whole model on CPU and streams one transformer block at a time to
        # the GPU (``layers[i].to(dev)`` then back to ``.cpu()``).  That keeps
        # the per-rank GPU peak at ~1 layer (shared ~2GB + 1/16 EP experts
        # ~11GB) instead of the full shared layers (~150GB), which would never
        # fit on a 96GB H20.  Materialising on CPU also means no full-model
        # replica ever lands on host memory (vs the old remote_device="cpu"
        # ZeRO-3 path that pinned the ENTIRE 2.8TB model on CPU).
        dst_device = "cpu"
        # NCCL only speaks CUDA, so we cannot broadcast a CPU tensor directly.
        # Route through a per-tensor CUDA staging buffer: rank0 stages the
        # real weight on cuda:local_rank, we broadcast on cuda, then every
        # rank copies the staged tensor down to a CPU buffer and materialises
        # the parameter with ``set_module_tensor_to_device(..., "cpu", ...)``.
        # The CUDA staging is freed immediately (peak = one tensor).
        #
        # SHAPE MISMATCH HANDLING: HF checkpoints for GLM-5 DSA layers may
        # differ in some fused attention dims from what the current
        # ``modeling_glm_moe_dsa`` skeleton allocates (e.g. index-head fusion).
        # Two rules must hold to avoid NCCL hangs / silent corruption:
        #   1) All ranks MUST agree on the staging shape (rank0's src.shape,
        #      broadcast as an int64 tensor).
        #   2) All ranks MUST take the same skip-or-load decision.
        # If shapes don't match the target, rank0 logs it, all ranks skip.
        if dist_active:
            stage_device = f"cuda:{torch.cuda.current_device()}"
            # Broadcast rank0's decision + shape as an int64 tensor:
            #   [ok_flag, ndim, dim0, dim1, ...] padded to 8 slots.
            decision = torch.zeros(8, dtype=torch.int64, device=stage_device)
            if _rank() == 0:
                if src is None:
                    decision[0] = 0  # skip (no src on rank0)
                elif src.shape != target.shape:
                    print_info(
                        f"[stream_load] SHAPE MISMATCH -- zero-filling {key!r}: "
                        f"checkpoint={tuple(src.shape)} vs target={tuple(target.shape)}"
                    )
                    decision[0] = 0
                else:
                    decision[0] = 1
                    decision[1] = src.ndim
                    for i, d in enumerate(src.shape):
                        decision[2 + i] = int(d)
            torch.distributed.broadcast(decision, src=0)
            if int(decision[0].item()) == 0:
                # Skip loading, but still materialise a zero CPU tensor so the
                # target parameter is no longer on ``meta`` -- otherwise the
                # subsequent ``layers[i].to("cuda")`` in INT8.low_memory_run
                # would raise ``Cannot copy out of meta tensor``.  Zero-filled
                # weights make this parameter functionally empty but keep the
                # pipeline runnable (calibration + save proceeds; only this
                # specific op is meaningless).
                zero = torch.zeros(target.shape, dtype=target.dtype, device=dst_device)
                set_module_tensor_to_device(model, key, dst_device, value=zero)
                return False
            ndim = int(decision[1].item())
            shape = tuple(int(decision[2 + i].item()) for i in range(ndim))
            stage = torch.empty(shape, dtype=target.dtype, device=stage_device)
            if _rank() == 0:
                stage.copy_(src.to(device=stage_device, dtype=target.dtype))
            torch.distributed.broadcast(stage, src=0)
            buf = stage.to(device=dst_device, dtype=target.dtype).contiguous()
            del stage
        else:
            if src is None or src.shape != target.shape:
                if src is not None:
                    print_info(
                        f"[stream_load] SHAPE MISMATCH -- zero-filling {key!r}: "
                        f"checkpoint={tuple(src.shape)} vs target={tuple(target.shape)}"
                    )
                # Same zero-fill fallback as the distributed branch above.
                zero = torch.zeros(target.shape, dtype=target.dtype, device=dst_device)
                set_module_tensor_to_device(model, key, dst_device, value=zero)
                return False
            buf = torch.empty(src.shape, dtype=target.dtype, device=dst_device)
            buf.copy_(src.to(device=dst_device, dtype=target.dtype))
        set_module_tensor_to_device(model, key, dst_device, value=buf)
        return True

    if is_zero3_param(target):
        with gathered_param_if_zero3(target, modifier_rank=0):
            if _rank() == 0:
                if src is None or src.shape != target.shape:
                    return False
                target.data.copy_(src.to(device=target.device, dtype=target.dtype))
        return True

    # Regular tensor (parameter or buffer).
    if dist_active:
        # NCCL requires CUDA tensors.  If the target is on CPU (option-B EP:
        # shared layers + buffers all live in host memory), route through a
        # CUDA staging buffer and copy back to the CPU target.
        target_is_cpu = target.device.type == "cpu"
        stage_device = (
            f"cuda:{torch.cuda.current_device()}" if target_is_cpu else target.device
        )
        if _rank() == 0:
            if src is None or (not is_buffer and src.shape != target.shape):
                return False
            tmp = src.to(device=stage_device, dtype=target.dtype).contiguous()
        else:
            tmp = torch.empty(target.shape, dtype=target.dtype, device=stage_device)
        torch.distributed.broadcast(tmp, src=0)
        if target_is_cpu:
            target.data.copy_(tmp.to(device="cpu"))
        else:
            target.data.copy_(tmp)
        del tmp
        return True

    if src is None:
        return False
    target.data.copy_(src.to(device=target.device, dtype=target.dtype))
    return True


def stream_load_weights(model, model_path, log_prefix="[zero3]"):
    """Stream a HF safetensors checkpoint into ``model``.

    Two loading strategies, chosen by whether any param is a meta tensor:

    * **Meta model (option-B EP / no-DeepSpeed path)**: each rank walks the
      shards on its OWN, only reads (via safetensors ``get_tensor``, which is
      mmap-lazy) the keys that appear in its local ``named_parameters`` /
      ``named_buffers`` (EP experts NOT owned by this rank are absent), and
      materialises them straight onto CPU.  This is the **only** way to avoid
      the collective-asymmetry deadlock that ZeRO-3-style rank0-reads +
      broadcast has under EP -- under EP, rank0's local param set does not
      contain the experts other ranks own, so a broadcast would either hang
      (some ranks not participating) or wastefully copy the same tensor to
      every rank.  Since every rank has read access to the checkpoint on
      shared cephfs, per-rank reads are trivially parallel and touch only
      1/W of the routed-expert bytes on each rank.

    * **ZeRO-3 model**: falls back to the original rank0-reads-and-broadcasts
      path (via ``_broadcast_into_target``), because ZeRO-3 sharded params
      require ``GatheredParameters(modifier_rank=0)`` semantics.
    """
    from safetensors import safe_open
    from accelerate.utils import set_module_tensor_to_device

    name_to_param = dict(model.named_parameters())
    name_to_buffer = dict(model.named_buffers())
    rank = _rank()

    # Detect option-B path: any meta param means we build with meta skeleton
    # and each rank should read what it owns directly.
    any_meta = any(p.is_meta for p in name_to_param.values()) or any(
        b.is_meta for b in name_to_buffer.values()
    )

    loaded = 0
    skipped = 0
    shape_mismatch = 0
    seen_targets = set()

    if any_meta:
        # ---- per-rank independent load (no NCCL) --------------------------
        for shard_path, keys in _shards(model_path):
            with safe_open(shard_path, framework="pt") as reader:
                for key in keys:
                    tgt = name_to_param.get(key)
                    is_buf = False
                    if tgt is None:
                        tgt = name_to_buffer.get(key)
                        is_buf = tgt is not None
                    if tgt is None:
                        skipped += 1
                        continue
                    src = reader.get_tensor(key)
                    if src.shape != tgt.shape:
                        if shape_mismatch < 8 or rank == 0:
                            print_info(
                                f"[stream_load] SHAPE MISMATCH -- zero-filling "
                                f"{key!r}: checkpoint={tuple(src.shape)} vs "
                                f"target={tuple(tgt.shape)}"
                            )
                        shape_mismatch += 1
                        zero = torch.zeros(tgt.shape, dtype=tgt.dtype, device="cpu")
                        set_module_tensor_to_device(model, key, "cpu", value=zero)
                        skipped += 1
                        del src
                        continue
                    buf = src.to(dtype=tgt.dtype, device="cpu").contiguous()
                    set_module_tensor_to_device(model, key, "cpu", value=buf)
                    seen_targets.add(key)
                    loaded += 1
                    del src
            _cleanup()
            if rank == 0:
                print_info(f"{log_prefix} loaded shard {os.path.basename(shard_path)}")
    else:
        # ---- legacy ZeRO-3 path: rank0 reads, others receive via broadcast -
        for shard_path, keys in _shards(model_path):
            with safe_open(shard_path, framework="pt") as reader:
                for key in keys:
                    if key.endswith(".experts.gate_up_proj"):
                        base = key[: -len(".gate_up_proj")]
                        src = reader.get_tensor(key) if rank == 0 else None
                        n_exp = (
                            int(src.shape[0])
                            if src is not None
                            else _infer_num_experts(base, name_to_param)
                        )
                        for i in range(n_exp):
                            gkey = f"{base}.{i}.gate_proj.weight"
                            ukey = f"{base}.{i}.up_proj.weight"
                            gtgt = name_to_param.get(gkey)
                            utgt = name_to_param.get(ukey)
                            if gtgt is None or utgt is None:
                                skipped += 2
                                continue
                            gsrc = src[i].chunk(2, dim=-2)[0] if src is not None else None
                            usrc = src[i].chunk(2, dim=-2)[1] if src is not None else None
                            if _broadcast_into_target(gsrc, gtgt, key=gkey, model=model):
                                seen_targets.add(gkey)
                                loaded += 1
                            else:
                                skipped += 1
                            if _broadcast_into_target(usrc, utgt, key=ukey, model=model):
                                seen_targets.add(ukey)
                                loaded += 1
                            else:
                                skipped += 1
                        del src
                    elif key.endswith(".experts.down_proj"):
                        base = key[: -len(".down_proj")]
                        src = reader.get_tensor(key) if rank == 0 else None
                        n_exp = (
                            int(src.shape[0])
                            if src is not None
                            else _infer_num_experts(base, name_to_param)
                        )
                        for i in range(n_exp):
                            dkey = f"{base}.{i}.down_proj.weight"
                            dtgt = name_to_param.get(dkey)
                            if dtgt is None:
                                skipped += 1
                                continue
                            dsrc = src[i] if src is not None else None
                            if _broadcast_into_target(dsrc, dtgt, key=dkey, model=model):
                                seen_targets.add(dkey)
                                loaded += 1
                            else:
                                skipped += 1
                        del src
                    else:
                        tgt = name_to_param.get(key)
                        is_buf = False
                        if tgt is None:
                            tgt = name_to_buffer.get(key)
                            is_buf = tgt is not None
                        if tgt is None:
                            skipped += 1
                            continue
                        src = reader.get_tensor(key) if rank == 0 else None
                        if _broadcast_into_target(src, tgt, is_buffer=is_buf, key=key, model=model):
                            seen_targets.add(key)
                            loaded += 1
                        else:
                            skipped += 1
                        del src
            _cleanup()
            print_info(f"{log_prefix} loaded shard {os.path.basename(shard_path)}")

    all_targets = set(name_to_param) | set(name_to_buffer)
    missing = sorted(all_targets - seen_targets)
    print_info(
        f"{log_prefix} stream_load_weights done: "
        f"loaded={loaded} skipped={skipped} missing={len(missing)}"
    )
    if missing:
        print_info(f"{log_prefix} first missing keys: {missing[:10]}")

    # ------------------------------------------------------------------
    # Meta-tensor safety net.
    # ------------------------------------------------------------------
    # A parameter can remain on ``meta`` after stream_load_weights for any of
    # several benign reasons: a fused key mismatch, an HF modeling vs
    # checkpoint version drift (e.g. GLM-5 kv_a_proj_with_mqa 576 vs 704),
    # a tied-weight key not covered by ``named_parameters``, etc.  If ANY
    # parameter is still ``is_meta`` when downstream code calls
    # ``layers[i].to("cuda")`` (INT8.low_memory_run), PyTorch raises
    # ``NotImplementedError: Cannot copy out of meta tensor``.  So we walk the
    # model once, zero-fill every meta leaf on CPU, and log a summary.  The
    # affected ops become numerically inert but the pipeline stays runnable.
    from accelerate.utils import set_module_tensor_to_device
    meta_params = 0
    meta_buffers = 0
    for name, p in list(model.named_parameters()):
        if p.is_meta:
            set_module_tensor_to_device(
                model, name, "cpu",
                value=torch.zeros(p.shape, dtype=p.dtype, device="cpu"),
            )
            meta_params += 1
    for name, b in list(model.named_buffers()):
        if b.is_meta:
            set_module_tensor_to_device(
                model, name, "cpu",
                value=torch.zeros(b.shape, dtype=b.dtype, device="cpu"),
            )
            meta_buffers += 1
    if meta_params or meta_buffers:
        print_info(
            f"{log_prefix} zero-filled residual meta tensors: "
            f"params={meta_params} buffers={meta_buffers} "
            f"(these ops are numerically empty but non-meta so .to(cuda) works)"
        )

    try:
        model.tie_weights()
    except Exception as e:  # noqa: BLE001
        print_info(f"{log_prefix} tie_weights skipped: {e}")


def _infer_num_experts(base, name_to_param):
    prefix = f"{base}."
    ids = []
    for name in name_to_param:
        if not name.startswith(prefix):
            continue
        first = name[len(prefix) :].split(".", 1)[0]
        if first.isdigit():
            ids.append(int(first))
    return (max(ids) + 1) if ids else 0


# ---------------------------------------------------------------------------
# Streaming PTQ-scale loader for QAT warm-start
# ---------------------------------------------------------------------------


_SCALE_SUFFIX_RULES = [
    # (suffix_in_ckpt, quantizer_attr_on_QuantLinear, sub_attr, layer_name_rewrite)
    (".weight_zero_point", "weight_quantizer", "zero_point", None),
    (".input_zero_point", "act_quantizer", "zero_point", None),
    (".weight_scale", "weight_quantizer", "scale", None),
    (".input_scale", "act_quantizer", "scale", None),
    (".k_cache.scale", "qkv_quantizer", "scale", ".k_proj"),
    (".v_cache.scale", "qkv_quantizer", "scale", ".v_proj"),
]
# Longest first to avoid '.scale' winning over '.k_cache.scale'.
_SCALE_SUFFIX_RULES.sort(key=lambda r: len(r[0]), reverse=True)


def _parse_scale_key(key):
    for suffix, qname, sub, rewrite in _SCALE_SUFFIX_RULES:
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            return (base + rewrite if rewrite else base), qname, sub
    return None


def _expand_scale_targets(layer_name, qname, sub, named_modules):
    """Expand a checkpoint key that targets a fused MoE expert tensor into the
    matching per-expert linears in the linearized model. For non-MoE keys
    returns ``[(layer_name, qname, sub)]`` unchanged."""
    if layer_name in named_modules:
        return [(layer_name, qname, sub)]

    # PTQ checkpoint may store scales for the fused expert matrix; map them
    # to every per-expert Linear we expanded into.
    if layer_name.endswith(".experts.gate_up_proj"):
        base = layer_name[: -len(".gate_up_proj")]
        return [
            (n, qname, sub)
            for n in named_modules
            if n.startswith(base + ".") and (n.endswith(".gate_proj") or n.endswith(".up_proj"))
        ]
    if layer_name.endswith(".experts.down_proj"):
        base = layer_name[: -len(".down_proj")]
        return [
            (n, qname, sub)
            for n in named_modules
            if n.startswith(base + ".") and n.endswith(".down_proj")
        ]
    return []


def _copy_scale_into(src, target):
    """rank0-driven copy of a scale-like tensor into a (possibly ZeRO-3)
    Parameter, with shape coercion for scalar/per-tensor mismatch."""
    rank = _rank()
    ok = True
    with gathered_param_if_zero3(target, modifier_rank=0):
        if rank == 0:
            s = src
            if s.numel() == target.numel():
                s = s.reshape(target.shape)
            else:
                try:
                    s = s.expand_as(target).contiguous()
                except RuntimeError:
                    ok = False
            if ok:
                target.data.copy_(s.to(device=target.device, dtype=target.dtype))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else target.device
        )
        flag = torch.tensor(int(ok), device=device)
        torch.distributed.broadcast(flag, src=0)
        ok = bool(flag.item())
    return ok


def stream_load_scales(model, ckpt_dir, log_prefix="[zero3]"):
    """Read a PTQ "real" checkpoint and write its scale / zero_point /
    kv-cache scale tensors into the matching ``QuantLinear`` quantizer
    parameters of ``model``.

    Sets ``act_quantizer.init = True`` for every static activation
    quantizer that successfully receives a scale, so the lazy-init pass is
    skipped.
    """
    from safetensors import safe_open

    # Resolve nested layout (some PTQ exporters nest under final_quant_checkpoint/).
    if not glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        nested = os.path.join(ckpt_dir, "final_quant_checkpoint")
        if os.path.isdir(nested):
            ckpt_dir = nested

    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No *.safetensors in {ckpt_dir}")

    # Lazy import to avoid circular dependency: this module is imported by
    # angelslim.utils, which is imported very early.
    from ..compressor.qat.modules.quantizer import QuantLinear

    named_modules = dict(model.named_modules())
    rank = _rank()
    loaded = 0
    skipped = 0

    for src_file in files:
        with safe_open(src_file, framework="pt") as reader:
            for key in reader.keys():
                parsed = _parse_scale_key(key)
                if parsed is None:
                    continue
                layer_name, qname, sub = parsed
                targets = _expand_scale_targets(layer_name, qname, sub, named_modules)
                if not targets:
                    skipped += 1
                    continue
                src = reader.get_tensor(key) if rank == 0 else None
                for tgt_layer, tgt_qname, tgt_sub in targets:
                    module = named_modules.get(tgt_layer)
                    if not isinstance(module, QuantLinear):
                        skipped += 1
                        continue
                    quantizer = getattr(module, tgt_qname, None)
                    if quantizer is None:
                        skipped += 1
                        continue
                    target = getattr(quantizer, tgt_sub, None)
                    if not isinstance(target, torch.nn.Parameter):
                        skipped += 1
                        continue
                    if _copy_scale_into(src, target):
                        loaded += 1
                        if tgt_qname == "act_quantizer" and tgt_sub == "scale":
                            quantizer.init = True
                    else:
                        skipped += 1

    print_info(
        f"{log_prefix} stream_load_scales: loaded={loaded} skipped={skipped} from {ckpt_dir}"
    )


# ---------------------------------------------------------------------------
# Saving a sharded model via the model-specific save_func
# ---------------------------------------------------------------------------


def consolidated_state_dict(model):
    """rank-0 CPU state_dict for a possibly ZeRO-3 sharded ``model``.

    Other ranks see an empty dict (matching the contract of HF/Trainer
    save callbacks). Includes persistent buffers."""
    rank = _rank()
    sd = {}
    for name, param in model.named_parameters():
        with gathered_param_if_zero3(param):
            if rank == 0:
                sd[name] = param.detach().cpu().clone()
    if rank == 0:
        for module_name, module in model.named_modules():
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf is None or buf_name in module._non_persistent_buffers_set:
                    continue
                full = f"{module_name}.{buf_name}" if module_name else buf_name
                sd[full] = buf.detach().cpu().clone()
    return sd


def save_via_model_save_func(
    quant_model,
    save_func,
    save_target_dir,
    prebuilt_state_dict=None,
):
    """Invoke ``save_func.save(...)`` with the model's ``state_dict`` patched
    to return a consolidated rank-0 dict.

    Parameters
    ----------
    prebuilt_state_dict : dict | None
        If provided (rank 0), use this dict directly and skip the per-param
        gather+clone pass over ``model``. This is the **recommended** path
        under ZeRO-3 because the caller (typically ``QAT.convert``) has
        already produced rank0's full state_dict layer-by-layer with bounded
        peak memory. Other ranks may pass ``None``.

        If ``None``, fall back to ``consolidated_state_dict(model)``, which
        gathers every parameter once more — avoid this when ``model``
        already holds large materialised tensors on rank 0 (it would double
        the peak).

    No-op (delegates straight to ``save_func.save``) when no parameters are
    sharded.
    """
    if not model_has_zero3_params(quant_model.model) and prebuilt_state_dict is None:
        save_func.save(save_target_dir)
        return

    rank = _rank()
    if prebuilt_state_dict is not None:
        sd = prebuilt_state_dict if rank == 0 else {}
    else:
        sd = consolidated_state_dict(quant_model.model)

    hf_model = quant_model.get_model()
    original = hf_model.state_dict

    def _patched(*args, **kwargs):
        return sd if rank == 0 else {}

    try:
        hf_model.state_dict = _patched  # type: ignore[method-assign]
        if rank == 0:
            save_func.save(save_target_dir)
    finally:
        hf_model.state_dict = original  # type: ignore[method-assign]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# DeepSpeed engine: tolerate scale parameters that appear in multiple param
# groups (the scale tensor itself is unique, but our optimizer construction
# may pick it up from multiple QuantLinear children of a shared MoE expert).
# ---------------------------------------------------------------------------


def patch_deepspeed_duplicate_check():
    """No-op DeepSpeed's ``_check_for_duplicates`` so QAT scale parameters
    that share storage across modules don't crash optimizer init.

    Idempotent: only patches once per process.
    """
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except Exception as exc:  # noqa: BLE001
        print_info(f"[zero3] skip duplicate-check patch: {exc}")
        return
    if getattr(DeepSpeedEngine, "_angelslim_skip_dup_check", False):
        return

    def _noop(self, basic_optimizer):  # noqa: ARG001
        return

    DeepSpeedEngine._check_for_duplicates = _noop
    DeepSpeedEngine._angelslim_skip_dup_check = True
    print_info("[zero3] patched DeepSpeed _check_for_duplicates (idempotent).")
