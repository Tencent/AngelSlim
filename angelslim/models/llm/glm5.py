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

"""GLM-5 (``GlmMoeDsaForCausalLM``) model adapter for AngelSlim PTQ.

Architecture highlights (see ``config.json``):

* MLA (DeepSeek-style latent attention) 

* DSA (DeepSeek Sparse Attention) indexer:
      ``self_attn.indexer.wq_b``, ``self_attn.indexer.wk``,
      ``self_attn.indexer.weights_proj``, ``self_attn.indexer.k_norm``

* MoE (78 layers, first 3 dense + 75 sparse, 256 routed + 1 shared expert):
      ``mlp.experts.<idx>.gate_proj / up_proj / down_proj``
      ``mlp.shared_experts.gate_proj / up_proj / down_proj``
      ``mlp.gate.weight``          <- router, DO NOT quantize
      ``mlp.gate.e_score_correction_bias``

* MTP (Multi-Token Prediction) draft blocks live under
  ``model.layers.<N>.mtp_block.*`` (N == num_hidden_layers, i.e. layer 78 in
  the reference config).  Their **regular** linears (q_a_proj / q_b_proj /
  kv_a_proj_with_mqa / o_proj / gate_proj / up_proj / down_proj) are
  quantized alongside the main stack; MTP-internal ``kv_b_proj`` /
  ``indexer.*`` / ``weights_proj`` / ``mlp.gate.`` remain skipped for the
  same reasons as the main stack.
"""

import re
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ...compressor.quant.core import PTQSaveVllmHF
from ...compressor.quant.core import Glm5EPQuantSaver
from ...utils.utils import find_layers
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory
# Re-export the torch-free layer-selection constants so unit tests can
# also import them from the lightweight ``_glm5_skip_lists`` module (no
# torch dependency), while adapter callers can import them from here.
from ._glm5_skip_lists import (  # noqa: F401  (re-export)
    _QUANTIZABLE_LEAF_NAMES,
    _FORCED_SKIP_SUBSTRINGS,
)


# ==========================================================================
# MLA / DSA / Indexer dimension constants (W8A8C8 recipe)
# ==========================================================================
# These mirror the reference ``chatglm5.2/config.json``.  At runtime
# ``GLM5._read_mla_dims`` re-reads them from ``self.model.config`` so the
# same adapter also works for future GLM-5 variants with different sizes.
#
#  Latent KV cache layout produced by ``kv_a_proj_with_mqa`` (per token):
#      [0                 : kv_lora_rank)          <- NoPE latent, INT8 per-block-128 dynamic
#      [kv_lora_rank       : kv_lora_rank + qk_rope_head_dim)
#                                                   <- RoPE, KEPT bf16 (no quant)
#
#  Indexer K cache produced by ``indexer.wk`` (per token):
#      [0                 : index_head_dim)         <- INT8 per-token dynamic
_MLA_KV_LORA_RANK = 512
_MLA_QK_ROPE_HEAD_DIM = 64
_MLA_LATENT_BLOCK_SIZE = 128
_MLA_LATENT_NUM_BLOCKS = _MLA_KV_LORA_RANK // _MLA_LATENT_BLOCK_SIZE  # = 4
_INDEXER_HEAD_DIM = 128


# ==========================================================================
# Layer-selection constants (whitelist + hard/optional skip lists) are
# imported from ``_glm5_skip_lists`` above so that unit tests can exercise
# them on a torch-free environment.  The imports are kept as
# ``noqa: F401`` re-exports; see the top of this file.
# ==========================================================================


# ==========================================================================
# GLM-5 fused-experts -> per-expert nn.Linear rewrite
# ==========================================================================
#
# Upstream ``GlmMoeDsaNaiveMoe`` stores routed experts as two 3-D
# ``nn.Parameter``:
#
#     gate_up_proj  shape=[num_experts, 2*inter, hidden]
#     down_proj     shape=[num_experts, hidden,  inter]
#
# AngelSlim's ``find_layers`` only picks up ``nn.Linear`` leaves, so those
# fused parameters silently escape quantization and stay bf16 -> ~1.4 TB
# checkpoints (the exact symptom the user hit).
#
# ``GlmMoeDsaSplitMoe`` below is a drop-in replacement that:
#   * stores each expert as three real ``nn.Linear`` (gate_proj / up_proj /
#     down_proj), so the standard int8 QDQ pipeline works out of the box;
#   * reproduces the naive-MoE forward exactly (same expert-mask, same
#     ``index_add_`` accumulation, same dtype casting);
#   * loads its weights from a pre-existing ``GlmMoeDsaNaiveMoe`` by
#     un-fusing the 3-D parameters (chunks the gate/up columns).
#
# The saved checkpoint keys are ``experts.<idx>.{gate,up,down}_proj.weight``
# (+ ``.weight_scale``) - the exact format vLLM's
# ``CompressedTensorsWNA8MoEMethod`` expects.
# ==========================================================================


def _normalize_dtype(dtype):
    """Coerce a dtype that may arrive as a string (e.g. ``"torch.bfloat16"``
    parsed from YAML) or the ``"auto"`` sentinel into a real ``torch.dtype``
    (or ``None``).  ``nn.Linear(dtype="torch.bfloat16")`` would raise
    ``TypeError: empty() received an invalid combination of arguments``.
    """
    if dtype is None or dtype == "auto":
        return None
    if isinstance(dtype, str):
        name = dtype.split(".")[-1]          # "torch.bfloat16" -> "bfloat16"
        return getattr(torch, name, None)
    return dtype


class _GlmSplitExpertMLP(nn.Module):
    """One expert = three plain ``nn.Linear``.  bias=False matches upstream."""

    def __init__(self, hidden_dim, intermediate_dim, act_fn, dtype=None, device=None):
        # Normalize because ``dtype`` can be a YAML string ("torch.bfloat16")
        # or the "auto" sentinel, both of which break nn.Linear directly.
        dtype = _normalize_dtype(dtype)
        device = None if device in (None, "auto") else device
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, **factory)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, **factory)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, **factory)
        self.act_fn = act_fn

    def forward(self, x):  # standard SwiGLU
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _GlmZeroExpert(nn.Module):
    """Placeholder for experts NOT owned by the current EP rank.

    Used by the expert-parallel (EP) pre-shard path: when the model is sharded
    across ``world_size`` ranks, each rank only materialises its local experts
    (``[experts_start_idx, experts_end_idx)``) as real ``_GlmSplitExpertMLP``
    modules.  Experts outside that range are replaced by this zero-cost stub so
    that ``GlmMoeDsaSplitMoe.experts`` stays a contiguous ``nn.ModuleList`` of
    length ``num_experts`` (keeping ``self.experts[expert_idx]`` indexing valid)
    without holding any parameters.  The EP ``forward`` skips these indices
    entirely, so ``_GlmZeroExpert.forward`` is never actually invoked.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Should never be called: EP forward skips non-local experts.  Return a
        # zero tensor of the right shape as a defensive fallback.
        return torch.zeros_like(x)


class GlmMoeDsaSplitMoe(nn.ModuleList):
    """Drop-in replacement for ``GlmMoeDsaNaiveMoe`` with per-expert Linears.

    Inherits from ``nn.ModuleList`` so that when this module is set as an
    attribute called ``experts`` on the parent ``GlmMoeDsaMoE`` (i.e.
    ``mlp.experts = GlmMoeDsaSplitMoe(...)``), the child linears' parameter
    names collapse to ``mlp.experts.<i>.gate_proj.weight`` -- exactly the key
    layout the released GLM-5 safetensors checkpoints use.  Wrapping the
    experts inside a *second* ``self.experts = ModuleList(...)`` would
    duplicate the prefix to ``mlp.experts.experts.<i>...`` and silently miss
    every expert weight during ``stream_load_weights``.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, act_fn,
                 dtype=None, device=None, ep_rank=0, ep_world_size=1):
        # ``nn.ModuleList.__init__`` MUST run first: it sets up ``_modules``
        # and other bookkeeping dicts that ``nn.Module.__setattr__`` looks
        # up.  Assigning any attribute (especially an ``nn.Module`` like
        # ``act_fn`` when it happens to be a submodule) before this raises
        # ``AttributeError: cannot assign module before Module.__init__()
        # call``.
        nn.ModuleList.__init__(self, [])
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        # CRITICAL: ``act_fn`` is typically ``nn.SiLU()`` -- an ``nn.Module``.
        # ``nn.Module.__setattr__`` would auto-register it to
        # ``self._modules['act_fn']``, polluting the ModuleList's expert
        # slot count: ``append`` uses ``str(len(self))`` as key, so after
        # ``act_fn`` is registered, ``len(self) == 1`` and the first
        # ``append(expert)`` stores at key ``'1'`` instead of ``'0'`` -- every
        # subsequent expert index is shifted by 1 (``self[0]`` -> KeyError,
        # ``self[128]`` -> off-by-one _GlmZeroExpert).  Store it in the
        # plain ``__dict__`` via ``object.__setattr__`` so the ModuleList
        # only contains the 256 experts.
        object.__setattr__(self, "act_fn", act_fn)
        self._init_ep_attrs(ep_rank, ep_world_size)
        for i in range(num_experts):
            self.append(self._make_expert(
                i, hidden_dim, intermediate_dim, act_fn,
                dtype=dtype, device=device,
            ))

    def _init_ep_attrs(self, ep_rank, ep_world_size):
        """Set expert-parallel (EP) bookkeeping attributes.

        Mirrors ``hunyuan_v3_moe.HYV3ExpertsWithLinear``: when ``ep_world_size
        > 1`` each rank owns a contiguous slice ``[experts_start_idx,
        experts_end_idx)`` of the full expert set; experts outside that slice
        are zero-cost placeholders.  When ``ep_world_size == 1`` (no EP, e.g.
        the standard single-node ``from_naive`` path) ``expert_parallel_enabled``
        is False and every expert is materialised.
        """
        self.expert_parallel_enabled = bool(ep_world_size > 1)
        self.rank = int(ep_rank)
        self.world_size = int(ep_world_size)
        if self.expert_parallel_enabled:
            if num_experts := int(self.num_experts):
                if num_experts % self.world_size != 0:
                    raise ValueError(
                        f"num_experts {num_experts} must be divisible by "
                        f"ep_world_size {self.world_size} for expert parallel."
                    )
            self.n_local_experts = num_experts // self.world_size
            self.experts_start_idx = self.rank * self.n_local_experts
            self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        else:
            self.n_local_experts = int(self.num_experts)
            self.experts_start_idx = 0
            self.experts_end_idx = int(self.num_experts)

    def _make_expert(self, expert_idx, hidden_dim, intermediate_dim, act_fn,
                     dtype=None, device=None):
        """Build the expert module for ``expert_idx``.

        Under EP, only experts in ``[experts_start_idx, experts_end_idx)`` get
        a real ``_GlmSplitExpertMLP``; the rest become ``_GlmZeroExpert``
        placeholders (no parameters).  When EP is disabled every expert is real.
        """
        if (self.expert_parallel_enabled
                and not (self.experts_start_idx <= expert_idx
                         < self.experts_end_idx)):
            return _GlmZeroExpert()
        return _GlmSplitExpertMLP(hidden_dim, intermediate_dim, act_fn,
                                  dtype=dtype, device=device)

    @classmethod
    def empty(cls, num_experts, hidden_dim, intermediate_dim, act_fn,
              dtype=None, device=None, ep_rank=0, ep_world_size=1):
        """Build a ``GlmMoeDsaSplitMoe`` *without* copying any weights.

        Used by the expert-parallel / ZeRO-3 pre-shard path.  When
        ``ep_world_size > 1`` only the local expert slice
        ``[ep_rank*E/W, (ep_rank+1)*E/W)`` is built as (meta) ``nn.Linear``
        leaves; the rest are ``_GlmZeroExpert`` placeholders, so the per-rank
        parameter footprint is ``1/ep_world_size`` of the MoE weights (zero CPU
        replica of the full model).  The streaming safetensors loader
        (``zero3_io.stream_load_weights``) only fills the local experts because
        the other names are absent from ``model.named_parameters()``.  Parameter
        names (``experts.<i>.gate_proj/up_proj/down_proj.weight``) are identical
        to ``LinearizedMoeExperts``, so the loader needs no special-casing.
        """
        split = cls.__new__(cls)
        # ``nn.ModuleList.__init__`` first (see note in ``__init__``): it
        # installs ``_modules`` etc. before any attribute assignment.
        nn.ModuleList.__init__(split, [])
        split.num_experts = int(num_experts)
        split.hidden_dim = int(hidden_dim)
        split.intermediate_dim = int(intermediate_dim)
        # See CRITICAL comment in __init__: bypass nn.Module.__setattr__ so
        # act_fn (an nn.Module) does NOT get registered under _modules
        # ["act_fn"] and shift every expert index by one.
        object.__setattr__(split, "act_fn", act_fn)
        split._init_ep_attrs(ep_rank, ep_world_size)
        for i in range(int(num_experts)):
            split.append(split._make_expert(
                i, hidden_dim, intermediate_dim, act_fn,
                dtype=dtype, device=device,
            ))
        return split

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Numerically identical to ``GlmMoeDsaNaiveMoe.forward``.

        Under expert-parallel (``expert_parallel_enabled``) each rank only
        computes its locally-owned experts and the partial sums are reduced
        across ranks at the end (every rank processes the *same* input, so a
        plain ``all_reduce`` recovers the full MoE output).  Experts outside
        ``[experts_start_idx, experts_end_idx)`` are skipped.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            if expert_idx == self.num_experts:
                continue
            if (self.expert_parallel_enabled
                    and (expert_idx < self.experts_start_idx
                         or expert_idx >= self.experts_end_idx)):
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = self[expert_idx]
            # Defensive check: after EP skip we MUST land on a real
            # _GlmSplitExpertMLP.  Landing on a _GlmZeroExpert means the EP
            # bookkeeping (expert_parallel_enabled / experts_start_idx /
            # experts_end_idx) is out of sync with the actual expert modules
            # -- e.g. the module was rebuilt without EP or attributes were
            # dropped by a later ``setattr``.  Log once per rank and skip;
            # this yields a correct-but-partial forward (partial sums get
            # all_reduced below) instead of a hard AttributeError.
            if not isinstance(expert, _GlmSplitExpertMLP):
                if not getattr(self, "_ep_mismatch_logged", False):
                    _r = dist.get_rank() if (
                        dist.is_available() and dist.is_initialized()
                    ) else 0
                    print(
                        f"[GLM5][rank {_r}] EP MISMATCH: expert_idx={expert_idx} "
                        f"landed on {type(expert).__name__}, but "
                        f"expert_parallel_enabled={self.expert_parallel_enabled} "
                        f"local_range=[{self.experts_start_idx},"
                        f"{self.experts_end_idx}) world_size={self.world_size} "
                        f"num_experts={self.num_experts}. Skipping.",
                        flush=True,
                    )
                    object.__setattr__(self, "_ep_mismatch_logged", True)
                continue
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = expert.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)
            current_hidden_states = current_hidden_states * \
                top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        if self.expert_parallel_enabled and self.world_size > 1:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(final_hidden_states)
        return final_hidden_states

    @classmethod
    def from_naive(cls, naive_moe):
        """Build a ``GlmMoeDsaSplitMoe`` from an existing ``GlmMoeDsaNaiveMoe``.

        Weights are moved (not copied) into the new expert Linears to keep
        peak host memory bounded.  The source module is left with empty
        tensors after the transfer.
        """
        num_experts = naive_moe.num_experts
        hidden_dim = naive_moe.hidden_dim
        intermediate_dim = naive_moe.intermediate_dim
        act_fn = naive_moe.act_fn

        # Reference tensors (shape: [E, 2*I, H] and [E, H, I]).
        gate_up = naive_moe.gate_up_proj  # nn.Parameter
        down = naive_moe.down_proj        # nn.Parameter
        dtype = gate_up.dtype
        device = gate_up.device

        # Build the empty container on the same device / dtype but with
        # ``meta`` placeholders so we don't double the memory before we can
        # steal the fused tensors.
        split = cls.__new__(cls)
        nn.ModuleList.__init__(split, [])
        split.num_experts = num_experts
        split.hidden_dim = hidden_dim
        split.intermediate_dim = intermediate_dim
        # Bypass nn.Module.__setattr__ so act_fn (an nn.Module) is not
        # auto-registered in _modules -- see CRITICAL note in __init__.
        object.__setattr__(split, "act_fn", act_fn)
        # Under ``from_naive`` there is no EP shard (single-node path); mark
        # every expert as local so ``_make_expert`` builds a real Linear.
        split._init_ep_attrs(0, 1)

        # gate_up rows are laid out as ``[gate_row_0..I-1, up_row_0..I-1]``
        # per expert (upstream uses ``.chunk(2, dim=-1)`` on the *output*).
        # Confirm: ``F.linear(x, gate_up[i]).chunk(2, dim=-1)`` returns
        # ``(out[..., :I], out[..., I:])``. Since ``F.linear`` computes
        # ``x @ W.T`` with W shaped ``[out, in]``, the *rows* of
        # ``gate_up[i]`` are the output channels.  Row 0..I-1 -> gate,
        # row I..2I-1 -> up.
        for i in range(num_experts):
            gate_w = gate_up[i, :intermediate_dim, :].contiguous()
            up_w = gate_up[i, intermediate_dim:, :].contiguous()
            down_w = down[i, :, :].contiguous()

            expert = _GlmSplitExpertMLP(
                hidden_dim, intermediate_dim, act_fn,
                dtype=dtype, device=device,
            )
            # Assign without triggering .to()/copy - we already match dtype/device.
            with torch.no_grad():
                expert.gate_proj.weight.copy_(gate_w)
                expert.up_proj.weight.copy_(up_w)
                expert.down_proj.weight.copy_(down_w)
            split.append(expert)
            # Free the intermediate contiguous copies eagerly.
            del gate_w, up_w, down_w

        # Nuke the fused parameters so peak memory drops back to O(N*W).
        naive_moe.gate_up_proj = None
        naive_moe.down_proj = None
        return split


@SlimModelFactory.register
class GLM5(BaseLLMModel):
    """AngelSlim model adapter for ``GlmMoeDsaForCausalLM`` (GLM-5)."""

    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.block_name = "model.layers"
        # ---- W8A8C8 kvcache observer state (populated by apply_kvcache_observers) ----
        # Each entry maps ``attn_name`` -> dict with per-block NoPE observers
        # for ``kv_a_proj_with_mqa`` output and one per-token observer for
        # ``indexer.wk`` output.  ``_kvcache_hook_handles`` stores the
        # ``RemovableHandle`` objects returned by ``register_forward_hook``
        # so ``remove_kvcache_observers`` can restore the model cleanly.
        self.kv_cache_observers = {}
        self._kvcache_hook_handles = []
        # ---- Effective MLA / indexer dims (filled lazily from HF config) ----
        self._mla_kv_lora_rank = _MLA_KV_LORA_RANK
        self._mla_qk_rope_head_dim = _MLA_QK_ROPE_HEAD_DIM
        self._mla_latent_block_size = _MLA_LATENT_BLOCK_SIZE
        self._mla_latent_num_blocks = _MLA_LATENT_NUM_BLOCKS
        self._indexer_head_dim = _INDEXER_HEAD_DIM
        # W8A8C8 saver metadata (set by ``get_save_func`` when a KV-C8 recipe
        # is active; picked up by ``PTQSaveVllmHF`` in ``core/save.py``).
        self._extra_kv_cache_scheme = None

    # ------------------------------------------------------------------
    # Weight loading -- override to de-fuse routed experts right after
    # HF finishes ``AutoModelForCausalLM.from_pretrained``.
    # ------------------------------------------------------------------
    def _fix_hf_config(self, hf_cfg, model_path):
        """Repair fields corrupted by HF ``PretrainedConfig.attribute_map``.

        The upstream ``configuration_glm_moe_dsa.py`` declares
        ``"head_dim": "qk_rope_head_dim"`` in its attribute_map -- so when
        the released ``config.json`` has BOTH ``head_dim`` and
        ``qk_rope_head_dim`` keys, HF silently overwrites
        ``qk_rope_head_dim`` with the value of ``head_dim`` (192 instead of
        64).  That in turn makes ``from_config`` allocate
        ``kv_a_proj_with_mqa`` as ``kv_lora_rank + 192 = 704`` (checkpoint
        is 576) and blows up the attention forward's
        ``torch.split(query_states, [192, 192])`` (real qk_head_dim is
        192+64=256).

        We re-read the *raw* JSON here and force the correct values.
        """
        import json
        import os as _os
        cfg_json_path = _os.path.join(model_path, "config.json")
        try:
            with open(cfg_json_path, "r") as f:
                raw = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[GLM5] _fix_hf_config: cannot read {cfg_json_path}: {e}",
                  flush=True)
            return
        # Only override when the JSON explicitly provides these keys.
        corrections = {}
        for name in (
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "qk_head_dim",
            "v_head_dim",
            "kv_lora_rank",
            "index_head_dim",
            "q_lora_rank",
        ):
            if name in raw and getattr(hf_cfg, name, None) != raw[name]:
                corrections[name] = (getattr(hf_cfg, name, None), raw[name])
                setattr(hf_cfg, name, raw[name])
        # Recompute derived qk_head_dim once qk_nope/qk_rope are correct.
        if hasattr(hf_cfg, "qk_nope_head_dim") and hasattr(hf_cfg, "qk_rope_head_dim"):
            derived = int(hf_cfg.qk_nope_head_dim) + int(hf_cfg.qk_rope_head_dim)
            if getattr(hf_cfg, "qk_head_dim", derived) != derived:
                corrections["qk_head_dim"] = (
                    getattr(hf_cfg, "qk_head_dim", None), derived,
                )
                hf_cfg.qk_head_dim = derived
        if corrections:
            print(
                f"[GLM5] _fix_hf_config: overriding HF-attribute_map "
                f"corruption with values from config.json: {corrections}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Repair per-layer schedule lists so MTP construction can index
        # them safely.
        #
        # GLM-5 packs the MTP (Multi-Token-Prediction) draft block(s) as
        # ``model.layers.<N>`` with N == num_hidden_layers (78 in
        # chatglm5.2, controlled by ``num_nextn_predict_layers``).
        # Even though those MTP blocks are NOT counted in
        # ``num_hidden_layers`` (the reference kunlunw8a8 config keeps it
        # at 78 too — confirming the correct architecture), the modeling
        # code still consults per-layer schedule lists such as
        # ``mlp_layer_types[layer_idx]`` and ``indexer_types[layer_idx]``
        # when it constructs the MTP block with ``layer_idx == 78``.  The
        # released ``config.json`` only fills those lists to length 78, so
        # the MTP block build blows up with ``IndexError: list index out
        # of range``.
        #
        # ⚠️  Do NOT bump ``num_hidden_layers`` to fix this — that turns
        # layer 78 into a *plain* ``GlmMoeDsaDecoderLayer`` instead of the
        # real MTP block, causing modeling to drop
        # ``eh_proj / enorm / hnorm / shared_head`` sub-modules and the
        # matching tensors get silently discarded as "unexpected keys"
        # at ``from_pretrained`` time.
        #
        # Correct fix: keep ``num_hidden_layers`` untouched (matches
        # ``kunlunw8a8/config.json``) and only pad the per-layer schedule
        # lists to ``num_hidden_layers + num_nextn_predict_layers``.
        try:
            n_main = int(getattr(hf_cfg, "num_hidden_layers", 0) or 0)
            n_mtp = int(getattr(hf_cfg, "num_nextn_predict_layers", 0) or 0)
            target_len = n_main + n_mtp
            # For chatglm5.2: target_len = 78 + 1 = 79.  For the MTP layer
            # (layer 78) we mirror the last full block's schedule: it's a
            # sparse-MoE layer (its safetensors carry
            # ``mlp.experts.*.{gate,up,down}_proj``) with its own "full"
            # indexer (verified against the reference kunlunw8a8 index).
            _per_layer_defaults = {
                "indexer_types": "full",
                "mlp_layer_types": "sparse",
            }
            for _lname in list(vars(hf_cfg).keys()):
                _lval = getattr(hf_cfg, _lname, None)
                if not isinstance(_lval, list):
                    continue
                # only touch lists whose length matches n_main (i.e. the
                # per-transformer-block schedule lists) and are short.
                if len(_lval) != n_main or len(_lval) >= target_len:
                    continue
                _fill = _per_layer_defaults.get(
                    _lname, _lval[-1] if _lval else None
                )
                pad = [_fill] * (target_len - len(_lval))
                setattr(hf_cfg, _lname, list(_lval) + pad)
                print(
                    f"[GLM5] _fix_hf_config: padded {_lname} from "
                    f"{n_main} -> {target_len} with {pad} to make the "
                    f"MTP-block ({n_mtp}) constructor safe.",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            print(
                f"[GLM5] _fix_hf_config: per-layer schedule padding "
                f"failed ({e}); leaving config unchanged.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # IndexShare helper -- which layers own an independent ("full")
    # indexer, and which share the previous "full" layer's topk.
    # ------------------------------------------------------------------
    def _shared_indexer_layer_ids(self):
        """Return the set of layer indices whose ``self_attn.indexer`` is
        a *shared* placeholder (GLM-5 IndexShare: every four sparse
        attention layers reuse the same indexer weights).

        Authority = ``config.indexer_types``: entries equal to ``"shared"``
        mean this layer skips its own indexer computation and reuses the
        preceding "full" layer's topk indices.  In the released
        chatglm5.2 checkpoint those layers have NO indexer weights on disk
        (the ``nn.Module`` is instantiated by modeling code but its
        parameters were never trained / saved).  We must therefore:
          * NOT quantize their indexer sub-linears (garbage in, garbage out);
          * NOT emit their indexer keys in the exported checkpoint (they'd
            just be random-init noise polluting the vLLM state_dict).
        Returns an empty set when the config lacks ``indexer_types`` (e.g.
        older architectures) so callers become no-ops.
        """
        try:
            cfg = self.model.config
        except Exception:
            return set()
        itypes = getattr(cfg, "indexer_types", None)
        if not isinstance(itypes, (list, tuple)):
            return set()
        return {i for i, t in enumerate(itypes) if t == "shared"}

    def from_pretrained(self, model_path, *args, **kwargs):
        super().from_pretrained(model_path, *args, **kwargs)
        # Record the source HF checkpoint path on the underlying model
        # object so downstream savers can reach back for tensors that
        # ``from_pretrained`` silently dropped.  The MTP draft block --
        # ``model.layers.<num_hidden_layers>`` -- is one such case: the
        # upstream ``GlmMoeDsaModel.__init__`` sizes ``self.layers`` off
        # ``num_hidden_layers`` alone, so every disk key belonging to
        # layer N is discarded as "unexpected" at load time.  The saver
        # (``Glm5EPQuantSaver._maybe_emit_mtp_shard_from_source``) reads
        # those tensors back straight from ``ori_model_path`` and
        # offline-quantizes them.  Mirrors the same convention set by
        # ``modeling_deepseek.py``.
        try:
            self.model.ori_model_path = model_path
        except Exception:
            # Fallback: some HF model classes forbid arbitrary attribute
            # assignment via ``__setattr__``.  Store on the adapter
            # instead; the saver checks both.
            self.ori_model_path = model_path
        self._defuse_moe_experts()
        self._promote_ep_state()

    def _promote_ep_state(self):
        """Lift expert-parallel (EP) status from the model's ``GlmMoeDsaSplitMoe``
        modules onto this adapter, so the saver can decide whether to use the
        EP-merge path (``Glm5EPQuantSaver``) without re-walking the graph.
        """
        for _, module in self.model.named_modules():
            if isinstance(module, GlmMoeDsaSplitMoe):
                self.expert_parallel_enabled = bool(
                    getattr(module, "expert_parallel_enabled", False)
                )
                self.world_size = int(getattr(module, "world_size", 1) or 1)
                self.rank = int(getattr(module, "rank", 0) or 0)
                return
        self.expert_parallel_enabled = False
        self.world_size = 1
        self.rank = 0

    def _defuse_moe_experts(self):
        """Replace every ``GlmMoeDsaNaiveMoe`` with ``GlmMoeDsaSplitMoe``.

        This makes routed experts visible to ``find_layers`` (they become
        real ``nn.Linear`` leaves), which is the pre-requisite for the
        int8 QDQ pipeline to actually quantize them.  Idempotent: if no
        fused experts are found (already de-fused, or a different arch),
        this is a no-op.
        """
        import time

        try:
            from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
                GlmMoeDsaNaiveMoe,
            )
        except Exception:
            # Not a GLM-5 install; nothing to do.
            return

        print("[GLM5] Scanning model for fused MoE layers...", flush=True)
        # Walk once to collect targets; mutate in a second pass so we don't
        # invalidate ``named_modules`` iterator.
        targets = []
        for name, module in self.model.named_modules():
            if isinstance(module, GlmMoeDsaNaiveMoe):
                targets.append(name)

        n_total = len(targets)
        if n_total == 0:
            print("[GLM5] No fused MoE layers found; nothing to de-fuse.",
                  flush=True)
            return

        print(f"[GLM5] Found {n_total} fused MoE layers; starting "
              f"de-fuse (this expands each layer's 256 experts into "
              f"individual nn.Linear leaves so int8 PTQ can see them).",
              flush=True)

        t_start = time.time()
        for idx, name in enumerate(targets):
            t_layer = time.time()
            parent_name, _, child_name = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name \
                else self.model
            naive_moe = getattr(parent, child_name)
            n_experts = naive_moe.num_experts
            split_moe = GlmMoeDsaSplitMoe.from_naive(naive_moe)
            setattr(parent, child_name, split_moe)

            dt_layer = time.time() - t_layer
            dt_total = time.time() - t_start
            eta = dt_total / (idx + 1) * (n_total - idx - 1)
            print(
                f"[GLM5] [{idx + 1:>3d}/{n_total}] de-fused {name} "
                f"({n_experts} experts) in {dt_layer:.2f}s  "
                f"total={dt_total:.1f}s  ETA={eta:.1f}s",
                flush=True,
            )

        print(
            f"[GLM5] All {n_total} MoE layers de-fused in "
            f"{time.time() - t_start:.1f}s; routed experts are now standard "
            f"nn.Linear leaves and will be quantized by the int8 pipeline.",
            flush=True,
        )

    def _defuse_moe_experts_empty(self):
        """ZeRO-3 pre-shard variant of :meth:`_defuse_moe_experts`.

        Replace every ``GlmMoeDsaNaiveMoe`` with an *empty*
        ``GlmMoeDsaSplitMoe`` built via :meth:`GlmMoeDsaSplitMoe.empty`.
        Must be called while ``deepspeed.zero.Init`` is active (i.e. inside
        ``zero3_empty_model_from_pretrained``) so the per-expert ``nn.Linear``
        parameters are partitioned across ranks immediately.  No weights are
        copied here -- :func:`angelslim.utils.zero3_io.stream_load_weights`
        streams the shards in afterwards, keyed by
        ``experts.<i>.gate_proj/up_proj/down_proj.weight`` (names identical to
        ``LinearizedMoeExperts`` so the generic loader works unchanged).
        """
        import time

        try:
            from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
                GlmMoeDsaNaiveMoe,
            )
        except Exception:
            return

        print("[GLM5] Scanning model for fused MoE layers (ZeRO-3 empty "
              "defuse)...", flush=True)
        targets = []
        for name, module in self.model.named_modules():
            if isinstance(module, GlmMoeDsaNaiveMoe):
                targets.append(name)

        n_total = len(targets)
        if n_total == 0:
            print("[GLM5] No fused MoE layers found; nothing to de-fuse.",
                  flush=True)
            return

        print(f"[GLM5] Found {n_total} fused MoE layers; starting empty "
              f"de-fuse (structure-only, weights streamed later).", flush=True)

        # Expert-parallel (EP) shard parameters.  When running under torchrun
        # with world_size > 1 we shard the routed experts across ranks: each
        # rank materialises only its local slice [rank*E/W, (rank+1)*E/W) as
        # real (meta) nn.Linear leaves; the rest become zero-cost placeholders.
        # This is the "option A" EP path (mirrors hunyuan_v3_moe): no full CPU
        # replica of the MoE experts, per-rank footprint = 1/world_size.
        ep_world_size = 1
        ep_rank = 0
        if dist.is_available() and dist.is_initialized():
            ep_world_size = dist.get_world_size()
            ep_rank = dist.get_rank()
        ep_enabled = ep_world_size > 1
        if ep_enabled:
            print(f"[GLM5] Expert-parallel defuse enabled: rank={ep_rank}/"
                  f"{ep_world_size} (each rank owns 1/{ep_world_size} of the "
                  f"routed experts).", flush=True)

        t_start = time.time()
        for idx, name in enumerate(targets):
            t_layer = time.time()
            parent_name, _, child_name = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name \
                else self.model
            naive_moe = getattr(parent, child_name)
            n_experts = int(naive_moe.num_experts)
            hidden_dim = int(naive_moe.hidden_dim)
            intermediate_dim = int(naive_moe.intermediate_dim)
            act_fn = naive_moe.act_fn
            dtype = getattr(naive_moe.gate_up_proj, "dtype", torch.bfloat16) \
                if hasattr(naive_moe, "gate_up_proj") else torch.bfloat16

            # Build the split experts on torch.device("meta") so NO full CPU
            # replica is ever materialised.  The skeleton is already meta from
            # from_config; keep new per-expert Linears on meta too.  Under EP
            # only the local expert slice is materialised (as meta); the rest
            # are _GlmZeroExpert placeholders with no parameters.
            # stream_load_weights materialises only this rank's local experts
            # on CUDA afterwards (names absent from named_parameters are skipped
            # by the loader, so the full-model CPU replica never exists).
            with torch.device("meta"):
                split_moe = GlmMoeDsaSplitMoe.empty(
                    num_experts=n_experts,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    act_fn=act_fn,
                    dtype=dtype,
                    device="meta",
                    ep_rank=ep_rank if ep_enabled else 0,
                    ep_world_size=ep_world_size if ep_enabled else 1,
                )
            # Swap the reference FIRST, then drop the old fused module.  The old
            # naive_moe is a meta skeleton (zero bytes) so del is cheap and there
            # is no accumulated replica to OOM on.
            setattr(parent, child_name, split_moe)
            del naive_moe
            torch.cuda.empty_cache()

            dt_layer = time.time() - t_layer
            dt_total = time.time() - t_start
            eta = dt_total / (idx + 1) * (n_total - idx - 1)
            print(
                f"[GLM5] [{idx + 1:>3d}/{n_total}] empty-de-fused {name} "
                f"({n_experts} experts) in {dt_layer:.2f}s  "
                f"total={dt_total:.1f}s  ETA={eta:.1f}s",
                flush=True,
            )

        print(
            f"[GLM5] All {n_total} MoE layers empty-de-fused in "
            f"{time.time() - t_start:.1f}s; routed experts are now "
            f"partitioned nn.Linear leaves (weights pending stream load).",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Observer selection
    # ------------------------------------------------------------------
    def get_observer_layers(self):
        """Select the set of nn.Linear layers to observe / quantize.

        The base ``PTQ`` pipeline uses this to:
          * install the activation-observer hook on every returned layer,
          * later swap each layer for a ``QDQModule`` at ``convert()`` time.

        Anything NOT in the returned dict is silently kept in the original
        (bf16) precision, which is exactly what we want for
        ``kv_b_proj`` / indexer / router / MTP / lm_head / embed.
        MTP 层被当作正常一层， 根据safetensors.index.json 被继续量化
        """
        obs_layer_classes = [nn.Linear]
        layers_dict = find_layers(self.model, layers=obs_layer_classes)

        # Start with any user-supplied ignore list, then append our forced
        # skips.  ``skip_layer_names()`` returns a reference to
        # ``quant_config.quant_algo_info['ignore_layers']`` so mutating it
        # here also updates PTQ's downstream views.
        ignore_layers = self.skip_layer_names()
        # Snapshot the ORIGINAL user-supplied patterns before we start
        # appending fully-qualified names to ``ignore_layers``.  Otherwise
        # each freshly-appended FQN would immediately participate in
        # substring matches for the next iteration and could poison the
        # decision (e.g. ``"model.layers.0.self_attn.kv_b_proj"`` becoming
        # a "pattern" that matches nothing useful but wastes cycles).
        user_ignore_patterns = tuple(p for p in ignore_layers if p)

        # ---- IndexShare: skip ``self_attn.indexer.*`` on shared layers ----
        # GLM-5 layers marked ``indexer_types[l] == "shared"`` reuse the
        # preceding "full" layer's topk indices.  The released checkpoint
        # has NO indexer weights for those layers (only the "full" ones at
        # 0/1/2/6/10/14/.../78), so we must (a) not observe them (they are
        # random-init noise; quantizing that would be garbage) and (b) let
        # the saver drop them from the exported state_dict.  Building the
        # exclusion rule from ``config.indexer_types`` mirrors the modeling
        # code's own authority and stays correct across model versions.
        shared_indexer_lids = self._shared_indexer_layer_ids()
        shared_indexer_re = re.compile(
            r"^model\.layers\.(\d+)\.self_attn\.indexer\."
        )

        observer_layers_dict = {}
        for name, module in layers_dict.items():
            # (1) must live inside a transformer block
            if not name.startswith(self.block_name):
                ignore_layers.append(name)
                continue

            # (2) leaf name must be in the whitelist
            leaf = name.split(".")[-1]
            if leaf not in _QUANTIZABLE_LEAF_NAMES:
                ignore_layers.append(name)
                continue

            # (3) hard-skip substrings (router / lm_head / embed / MTP fuse
            # / indexer RMSNorm).  ALWAYS excluded, no YAML override.
            if any(pat in name for pat in _FORCED_SKIP_SUBSTRINGS):
                ignore_layers.append(name)
                continue

            # (4) user-supplied YAML ``ignore_layers`` patterns.  Substring
            # semantics identical to what gptq / daq / save downstream use.
            # This is how the kunlun recipe opts out of ``.indexer.wq_b`` /
            # ``.indexer.wk`` (and how any future recipe can opt out of
            # ``kv_b_proj`` by simply listing it here).
            if any(pat in name for pat in user_ignore_patterns):
                ignore_layers.append(name)
                continue

            # (5) IndexShare: drop indexer sub-linears of shared layers.
            m = shared_indexer_re.match(name)
            if m and int(m.group(1)) in shared_indexer_lids:
                ignore_layers.append(name)
                continue

            observer_layers_dict[name] = module

        # De-dup and stable-sort the ignore list.
        ignore_layers[:] = sorted(list(set(ignore_layers)))
        self.quant_config.quant_algo_info["ignore_layers"] = ignore_layers

        # Optional user-scoped filter (kept for parity with ``GLM``).
        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in list(observer_layers_dict.keys()):
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)

        return observer_layers_dict

    # ------------------------------------------------------------------
    # SmoothQuant mapping — must reflect MLA, not q/k/v_proj
    # ------------------------------------------------------------------
    def get_smooth_mapping_layers(self, smooth_config, mappings=None):
        """Return the ``{norm_layer: (norm, [linear, ...])}`` mapping used by
        SmoothQuant.

        For GLM-5 MLA the two smoothable groups are:

          * ``input_layernorm`` -> [``q_a_proj``, ``kv_a_proj_with_mqa``]
            (the first attention linears that consume the norm output;
            ``q_b_proj`` / ``kv_b_proj`` are behind their own ``q_a_layernorm`` /
            ``kv_a_layernorm`` and are not directly smoothable here.)

          * ``post_attention_layernorm`` -> [``gate_proj``, ``up_proj``]
            for dense MLP and every MoE ``experts.*`` / ``shared_experts.*``
            block.  ``BaseLLMModel.get_smooth_mapping_layers`` walks
            ``named_modules()`` with a longest-common-prefix rule, which
            correctly picks up all expert copies.
        """
        if mappings is None:
            mappings = [
                (["q_a_proj", "kv_a_proj_with_mqa"], "input_layernorm"),
                (["gate_proj", "up_proj"], "post_attention_layernorm"),
            ]
        print(f"[GLM5] smooth mappings={mappings}")
        assert len(mappings) == 2
        assert smooth_config.smooth_first_linears or smooth_config.smooth_last_linears
        return super().get_smooth_mapping_layers(smooth_config, mappings)

    # ------------------------------------------------------------------
    # MoE experts share the same parent module (``mlp.experts``); this
    # collapses per-expert observer keys onto that parent so the saver
    # can fuse expert scales correctly.  Copied verbatim from GLM.
    # ------------------------------------------------------------------
    def get_parent_dict(self, observer_layers_dict):
        parent_mapping = {r"experts\.\d+": "experts"}
        parent_dict = {}
        for layer_name in observer_layers_dict.keys():
            parent_name = layer_name
            for k, v in parent_mapping.items():
                parent_name = re.sub(k, v, layer_name)
            if parent_name != layer_name:
                parent_dict[layer_name] = parent_name
        return parent_dict

    # ------------------------------------------------------------------
    # Fuse qkv / gate-up scales just like the base ``GLM`` adapter.
    # For MLA the natural fuse group is (q_a_proj, kv_a_proj_with_mqa)
    # sharing ``input_layernorm``; keep gate/up fused for MLP.
    # ------------------------------------------------------------------
    def fuse_observer_amax(self, sub_layer, name):
        if "q_a_proj" in name or "kv_a_proj_with_mqa" in name:
            prefix = name.rsplit(".", 1)[0]
            q_name = f"{prefix}.q_a_proj"
            kv_name = f"{prefix}.kv_a_proj_with_mqa"

            weight_scales = []
            act_scales = []
            for key in [q_name, kv_name]:
                if key in self.weight_observer_amax_dict:
                    weight_scales.append(self.weight_observer_amax_dict[key])
                if key in self.input_observer_amax_dict:
                    act_scales.append(self.input_observer_amax_dict[key])
            weight_observer_amax = max(weight_scales) if weight_scales else \
                self.weight_observer_amax_dict[name]
            input_observer_amax = max(act_scales) if act_scales else \
                self.input_observer_amax_dict[name]
        elif "gate_proj" in name or "up_proj" in name:
            prefix = name.rsplit(".", 1)[0]
            gate_name = f"{prefix}.gate_proj"
            up_name = f"{prefix}.up_proj"

            weight_scales = []
            act_scales = []
            for key in [gate_name, up_name]:
                if key in self.weight_observer_amax_dict:
                    weight_scales.append(self.weight_observer_amax_dict[key])
                if key in self.input_observer_amax_dict:
                    act_scales.append(self.input_observer_amax_dict[key])
            weight_observer_amax = max(weight_scales) if weight_scales else \
                self.weight_observer_amax_dict[name]
            input_observer_amax = max(act_scales) if act_scales else \
                self.input_observer_amax_dict[name]
        else:
            weight_observer_amax = self.weight_observer_amax_dict[name]
            input_observer_amax = self.input_observer_amax_dict[name]

        return weight_observer_amax, input_observer_amax

    # ==================================================================
    # W8A8C8 KV cache path (MLA latent NoPE + DSA indexer K)
    # ==================================================================
    #
    # Design notes
    # ------------
    # The upstream GLM-5 (``GlmMoeDsaAttention``) forward is intricate:
    # q_lora / kv_lora split, indexer top-k with cross-layer
    # ``prev_topk_indices`` state, RoPE interleave, MTP branch, custom
    # attention interface dispatch.  Re-implementing it inside a
    # monkey-patch (as Hunyuan-V3 does) would be brittle.
    #
    # Instead we attach ``register_forward_hook`` observers directly on
    # the two Linear modules whose OUTPUTS are exactly the tensors we
    # want to quantize at inference time:
    #
    #   * ``self_attn.kv_a_proj_with_mqa`` -> shape ``[..., 576]``
    #       - channels ``[:512]`` = NoPE latent (INT8 per-block-128 dynamic)
    #       - channels ``[512:]`` = RoPE (KEPT bf16, no observer)
    #   * ``self_attn.indexer.wk``         -> shape ``[..., 128]``
    #       - full tensor = indexer latent K (INT8 per-token dynamic)
    #
    # Because both target flavours are DYNAMIC (per-block-128 or
    # per-token, scale is computed at inference time), we don't need to
    # persist static scales into the checkpoint -- ``get_kvcache_scales``
    # therefore returns ``{}``.  The observers still run so that (a) we
    # get calibration-time distribution stats for debugging and (b) the
    # saver knows the KV path was actually quantized (via
    # ``self.kv_cache_observers`` being non-empty).
    # ==================================================================

    def _read_mla_dims(self):
        """Refresh MLA / indexer dims from the loaded HF config.

        Falls back to the module-level constants if a field is missing.
        Called lazily from ``apply_kvcache_observers`` so it's safe even
        when ``self.model`` hasn't been populated yet (e.g. during unit
        tests that stub ``BaseLLMModel``).
        """
        cfg = getattr(self.model, "config", None) if self.model is not None else None
        if cfg is None:
            return
        self._mla_kv_lora_rank = int(getattr(cfg, "kv_lora_rank", _MLA_KV_LORA_RANK))
        self._mla_qk_rope_head_dim = int(
            getattr(cfg, "qk_rope_head_dim", _MLA_QK_ROPE_HEAD_DIM)
        )
        self._indexer_head_dim = int(getattr(cfg, "index_head_dim", _INDEXER_HEAD_DIM))
        # Block size stays fixed by design (matches GPU FP8 layout); recompute
        # ``num_blocks_per_token`` if the latent rank changes.
        assert self._mla_kv_lora_rank % self._mla_latent_block_size == 0, (
            f"[GLM5] kv_lora_rank={self._mla_kv_lora_rank} is not divisible by "
            f"block_size={self._mla_latent_block_size}; per-block layout won't line up."
        )
        self._mla_latent_num_blocks = (
            self._mla_kv_lora_rank // self._mla_latent_block_size
        )

    def get_kvcache_observer_layers_names(self, observe_names):
        """Suppress default k_proj / v_proj observation.

        GLM-5 MLA has neither: the K/V streams are reconstructed on the
        fly from a low-rank latent produced by ``kv_a_proj_with_mqa``.
        We install our own hooks in ``apply_kvcache_observers`` below,
        so the default observer wiring in ``core/hook.py`` must stay
        empty for this model.
        """
        return []

    def apply_kvcache_observers(self, kv_cache_observer_class, quant_bits=8):
        """Attach W8A8C8 KV observers to every attention module.

        For each ``self_attn`` (main stack + MTP branch):

          * hook ``kv_a_proj_with_mqa`` output; slice ``[..., :kv_lora_rank]``
            into ``num_blocks_per_token`` chunks of ``block_size=128`` and
            feed each chunk to its own tensor-wise observer.  The RoPE
            tail ``[..., kv_lora_rank:]`` is IGNORED (bf16 passthrough).
          * hook ``indexer.wk`` output; feed the whole tensor to one
            per-token / tensor-wise observer.

        Falls back to a per-layer C16 no-op if the module tree is
        missing either target (unusual GLM-5 variants).

        Args:
            kv_cache_observer_class: observer class (defaults to
                ``AbsmaxPertensorObserver``).  Must be a callable
                ``nn.Module`` in the same shape as Hunyuan-V3 uses.
            quant_bits: forwarded to the observer.
        """
        from ...utils import print_info
        from ...compressor.quant.observers import AbsmaxPertensorObserver

        if kv_cache_observer_class is None:
            kv_cache_observer_class = AbsmaxPertensorObserver

        # Sync ``self._mla_*`` with the HF config; use snapshot locals so
        # the hook closures see stable values even if the config is
        # mutated later.
        self._read_mla_dims()
        kv_lora_rank = self._mla_kv_lora_rank
        block_size = self._mla_latent_block_size
        num_blocks = self._mla_latent_num_blocks

        n_hooked = 0
        n_fallback = 0

        for attn_name, attn_module in self._iter_self_attn_modules():
            kv_a = getattr(attn_module, "kv_a_proj_with_mqa", None)
            indexer = getattr(attn_module, "indexer", None)
            indexer_wk = getattr(indexer, "wk", None) if indexer is not None else None

            if kv_a is None:
                # No MLA latent path -- e.g. a stripped MTP block or a
                # variant we don't recognize.  Skip cleanly; the layer
                # falls back to C16 for its KV path.
                print_info(
                    f"[GLM5] [C8] skip {attn_name}: no kv_a_proj_with_mqa "
                    f"(layer stays bf16 for KV cache)"
                )
                n_fallback += 1
                continue

            # ---- NoPE latent observers (4 x per-block-128, per layer) ----
            nope_observers = [
                kv_cache_observer_class(layer=kv_a, quant_bits=quant_bits)
                for _ in range(num_blocks)
            ]

            def _make_kv_a_hook(_observers, _kv_lora_rank, _block_size, _num_blocks):
                def _hook(_module, _inputs, output):
                    # ``output`` has shape ``[..., kv_lora_rank + qk_rope_head_dim]``.
                    # Only quantize the NoPE portion.
                    try:
                        nope = output[..., :_kv_lora_rank]
                        for b in range(_num_blocks):
                            start = b * _block_size
                            end = start + _block_size
                            _observers[b](nope[..., start:end])
                    except Exception:  # noqa: BLE001 -- observer must never break forward
                        pass
                    return output
                return _hook

            handle = kv_a.register_forward_hook(
                _make_kv_a_hook(nope_observers, kv_lora_rank, block_size, num_blocks)
            )
            self._kvcache_hook_handles.append(handle)

            # ---- Indexer K observer (per-token, one per layer) ----
            indexer_observer = None
            if indexer_wk is not None:
                indexer_observer = kv_cache_observer_class(
                    layer=indexer_wk, quant_bits=quant_bits
                )

                def _make_indexer_hook(_observer):
                    def _hook(_module, _inputs, output):
                        try:
                            _observer(output)
                        except Exception:  # noqa: BLE001
                            pass
                        return output
                    return _hook

                handle_idx = indexer_wk.register_forward_hook(
                    _make_indexer_hook(indexer_observer)
                )
                self._kvcache_hook_handles.append(handle_idx)

            self.kv_cache_observers[attn_name] = {
                "nope_observers": nope_observers,
                "indexer_observer": indexer_observer,
                "num_blocks": num_blocks,
                "block_size": block_size,
                "kv_lora_rank": kv_lora_rank,
                "has_indexer": indexer_wk is not None,
            }
            n_hooked += 1

        print_info(
            f"[GLM5] hooked {n_hooked} attention modules for KV C8 "
            f"(NoPE per-block-{block_size} x {num_blocks} + indexer "
            f"per-token); fallback-C16 layers={n_fallback}"
        )

    def remove_kvcache_observers(self):
        """Detach all forward hooks installed by ``apply_kvcache_observers``.

        Safe to call multiple times.  Leaves ``self.kv_cache_observers``
        populated so ``get_save_func`` still knows a KV-C8 recipe was
        active (for saver-side metadata emission).
        """
        from ...utils import print_info

        n = len(self._kvcache_hook_handles)
        for handle in self._kvcache_hook_handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._kvcache_hook_handles.clear()
        print_info(f"[GLM5] removed {n} KV C8 forward hooks")

    def get_kvcache_scales(self):
        """Return an empty scale dict.

        The W8A8C8 recipe uses DYNAMIC quantization on both the NoPE
        latent (per-block-128) and the indexer K stream (per-token), so
        scales are computed at inference time on the fly and MUST NOT
        be persisted into the checkpoint.  Calibration-time observers
        are still useful for debug prints (see the ``print_info`` in
        ``remove_kvcache_observers``) but their scales aren't saved.
        """
        # Deliberate no-op to keep the saver from emitting stale static scales.
        return {}

    def _iter_self_attn_modules(self):
        """Yield ``(name, module)`` for every ``self_attn`` in the model.

        Covers both the 78 main-stack transformer blocks and the MTP
        draft block(s) (``model.layers.<N>.mtp_block.self_attn``).
        """
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn") and hasattr(module, "kv_a_proj_with_mqa"):
                yield name, module

    # ------------------------------------------------------------------
    # Saver — reuse the standard vLLM / HF compressed-tensors saver.
    # ------------------------------------------------------------------
    def get_save_func(self):
        # If the KV cache path has been quantized via ``apply_kvcache_observers``
        # (i.e. W8A8C8 recipe), attach a rich scheme descriptor so the saver
        # can emit it into ``config.json``'s ``quantization_config``.  vLLM
        # / downstream kernels use this to reconstruct the per-block-128
        # dynamic INT8 layout on the NoPE latent stream and per-token INT8
        # dynamic layout on the DSA indexer K stream.
        if self.kv_cache_observers:
            self._extra_kv_cache_scheme = {
                "kv_cache_scheme": "int8_dynamic_per_block",
                "block_size": int(self._mla_latent_block_size),
                "num_blocks_per_token": int(self._mla_latent_num_blocks),
                "rope_head_dim": int(self._mla_qk_rope_head_dim),
                "rope_quantized": False,
                "indexer_k_scheme": "int8_dynamic_per_token",
                "indexer_head_dim": int(self._indexer_head_dim),
            }
        if self.deploy_backend in ["vllm", "huggingface"]:
            if getattr(self, "expert_parallel_enabled", False) and \
                    getattr(self, "world_size", 1) > 1:
                return Glm5EPQuantSaver
            return PTQSaveVllmHF
        raise NotImplementedError(
            f"deploy_backend {self.deploy_backend} is not supported for saving."
        )
