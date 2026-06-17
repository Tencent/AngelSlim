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

import gc
import glob
import json
import os
import re
import types

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.hy_v3.modeling_hy_v3 import (
    ALL_ATTENTION_FUNCTIONS,
    HYV3Experts,
    HYV3TopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ...compressor.quant.core import PTQSaveVllmHF
from ...utils import is_deepspeed_zero3_enabled
from ...utils.utils import find_layers, find_parent_layer_and_sub_name, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


def _patch_hyv3_router_for_zero3():
    if getattr(HYV3TopKRouter, "_angelslim_zero3_dtype_patch", False):
        return

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> tuple:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = nn.functional.linear(
            hidden_states.to(self.weight.dtype),
            self.weight,
        ).to(torch.float32)
        routing_weights = torch.sigmoid(router_logits)

        scores_for_choice = routing_weights + e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)

        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        top_k_weights = top_k_weights * self.router_scaling_factor

        return router_logits, top_k_weights, top_k_index

    HYV3TopKRouter.forward = patched_forward
    HYV3TopKRouter._angelslim_zero3_dtype_patch = True


def _is_hyv3_parameter_experts(module):
    if HYV3Experts is not None and isinstance(module, HYV3Experts):
        return True
    required_attrs = (
        "gate_up_proj",
        "down_proj",
        "num_experts",
        "hidden_dim",
        "intermediate_dim",
        "act_fn",
    )
    return all(hasattr(module, attr) for attr in required_attrs) and isinstance(
        getattr(module, "gate_up_proj", None), nn.Parameter
    )


class _HYV3ZeroExpert(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x.new_zeros((x.shape[0], x.shape[-1]))


HYV3ZeroExpert = _HYV3ZeroExpert


class HYV3ExpertsWithLinear(HYV3Experts):
    """Wrapper around HYV3Experts that exposes per-expert weights as nn.Linear modules.

    HYV3Experts stores all expert weights as 3-D nn.Parameter tensors, which are
    invisible to AngelSlim's find_layers() and PTQ hook (both only recognise
    nn.Linear).  This wrapper splits those tensors into individual nn.Linear
    modules at construction time so that the standard quantisation pipeline can
    observe and quantise them.

    Weight shape mapping
    --------------------
    gate_up_proj : [num_experts, 2*intermediate_dim, hidden_dim]
        gate_up_proj[i]  →  chunk(2, dim=0)
            gate_proj[i].weight : [intermediate_dim, hidden_dim]
            up_proj[i].weight   : [intermediate_dim, hidden_dim]
    down_proj : [num_experts, hidden_dim, intermediate_dim]
        down_proj[i] → down_proj[i].weight : [hidden_dim, intermediate_dim]
    """

    def __init__(self, experts_layer):
        # Bypass HYV3Experts.__init__ to avoid allocating large empty Parameter
        # tensors that we would immediately overwrite.  HYV3Experts does not
        # store self.config, so we copy the required scalar attributes directly.
        nn.Module.__init__(self)
        self.num_experts = experts_layer.num_experts
        self.hidden_dim = experts_layer.hidden_dim
        self.intermediate_dim = experts_layer.intermediate_dim
        self.act_fn = experts_layer.act_fn

        for expert_idx in range(self.num_experts):
            expert = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "up_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "down_proj": nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False),
                }
            )
            # gate_up_proj[i]: [2*intermediate_dim, hidden_dim]
            # chunk on dim=0 → [intermediate_dim, hidden_dim] each
            gate_weight, up_weight = experts_layer.gate_up_proj[expert_idx].chunk(2, dim=0)
            expert["gate_proj"].weight.data = gate_weight
            expert["up_proj"].weight.data = up_weight
            # down_proj[i]: [hidden_dim, intermediate_dim]
            expert["down_proj"].weight.data = experts_layer.down_proj[expert_idx]
            setattr(self, f"{expert_idx}", expert)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_parallel_enabled = getattr(self, "expert_parallel_enabled", False)
        experts_start_idx = getattr(self, "experts_start_idx", 0)
        experts_end_idx = getattr(self, "experts_end_idx", self.num_experts)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            if expert_idx == self.num_experts:
                continue
            if expert_parallel_enabled and (
                expert_idx < experts_start_idx or expert_idx >= experts_end_idx
            ):
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert_layer = getattr(self, f"{expert_idx}")
            if not isinstance(expert_layer, nn.ModuleDict):
                continue
            expert_scores = top_k_weights[token_idx, top_k_pos, None]
            for child_name in ("gate_proj", "up_proj", "down_proj"):
                child = expert_layer[child_name]
                child._angelslim_moe_token_idx = token_idx.detach()
                child._angelslim_moe_expert_scores = expert_scores.detach()
                object.__setattr__(child, "_angelslim_moe_parent_expert", expert_layer)
            gate = expert_layer["gate_proj"](current_state)
            up = expert_layer["up_proj"](current_state)
            current_hidden_states = (self.act_fn(gate).float() * up.float()).to(
                expert_layer["down_proj"].weight.dtype
            )
            current_hidden_states = expert_layer["down_proj"](current_hidden_states)
            current_hidden_states = current_hidden_states.float() * expert_scores.float()
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        if expert_parallel_enabled and dist.is_available() and dist.is_initialized():
            dist.all_reduce(final_hidden_states)

        return final_hidden_states


class HYV3LocalExpertsWithLinear(HYV3ExpertsWithLinear):
    def __init__(self, experts_layer, rank, world_size, dtype=torch.bfloat16, device="cpu"):
        nn.Module.__init__(self)
        self.num_experts = int(experts_layer.num_experts)
        self.hidden_dim = int(experts_layer.hidden_dim)
        self.intermediate_dim = int(experts_layer.intermediate_dim)
        self.act_fn = experts_layer.act_fn

        if self.num_experts % world_size != 0:
            raise ValueError(
                f"num_experts {self.num_experts} must be divisible by world_size {world_size} "
                "for expert parallel."
            )

        self.rank = rank
        self.world_size = world_size
        self.n_local_experts = self.num_experts // self.world_size
        self.experts_start_idx = self.rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.expert_parallel_enabled = True

        for expert_idx in range(self.num_experts):
            if self.experts_start_idx <= expert_idx < self.experts_end_idx:
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
            else:
                expert = _HYV3ZeroExpert()
            setattr(self, f"{expert_idx}", expert)


@SlimModelFactory.register
class HYV3MoE(BaseLLMModel):
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
        # Store original forward methods for restoration
        self._original_attn_forwards = {}
        # Store KV cache observers: {attn_layer_name: {"key_observer": ..., "value_observer": ...}}
        self.kv_cache_observers = {}
        self.using_multi_nodes = False
        self.rank = 0
        self.world_size = 1

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        using_multi_nodes=False,
    ):
        # Attention implementation. Default "eager" so KV-cache / activation
        # calibration can hook the explicit attention weights. For weight-only
        # GPTQ (no attention hooks) "eager" materializes a [heads, seq, seq]
        # fp32 matrix (16 GiB at seq=8192) and OOMs; set ANGELSLIM_ATTN_IMPL=sdpa
        # to use the flash-style SDPA path that never materializes it.
        attn_implementation = os.environ.get("ANGELSLIM_ATTN_IMPL", "eager")
        torch_dtype = torch.bfloat16
        if is_deepspeed_zero3_enabled():
            _patch_hyv3_router_for_zero3()
        self.using_multi_nodes = (
            using_multi_nodes
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        self.rank = dist.get_rank() if self.using_multi_nodes else 0
        self.world_size = dist.get_world_size() if self.using_multi_nodes else 1

        if self.using_multi_nodes:
            self._from_pretrained_expert_parallel(
                model_path=model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                use_cache=use_cache,
                attn_implementation=attn_implementation,
            )
        else:
            super().from_pretrained(
                model_path=model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_cache=use_cache,
                using_multi_nodes=using_multi_nodes,
                attn_implementation=attn_implementation,
            )
            self._restore_router_fp32_from_checkpoint(model_path)

        if self.using_multi_nodes:
            self._enable_expert_parallel()

    def _is_router_fp32_name(self, name):
        return name.endswith(".mlp.gate.weight") or name.endswith(".mlp.e_score_correction_bias")

    def _restore_router_fp32_from_checkpoint(self, model_path):
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open

        name_to_param = dict(self.model.named_parameters())
        name_to_buffer = dict(self.model.named_buffers())
        target_state_dict = {}
        target_state_dict.update(name_to_param)
        target_state_dict.update(name_to_buffer)
        weight_renamings, weight_converters = self.get_checkpoint_key_conversions()

        restored = 0
        for shard_path, keys in self._iter_checkpoint_shards(model_path):
            with safe_open(shard_path, framework="pt") as reader:
                if keys is None:
                    keys = list(reader.keys())
                for key in keys:
                    model_key = self.resolve_checkpoint_key_for_model(
                        key,
                        target_state_dict=target_state_dict,
                        weight_renamings=weight_renamings,
                        weight_converters=weight_converters,
                    )
                    if not self._is_router_fp32_name(model_key):
                        continue
                    target = target_state_dict.get(model_key)
                    if target is None:
                        continue
                    value = reader.get_tensor(key).to(torch.float32)
                    set_module_tensor_to_device(
                        self.model,
                        model_key,
                        target.device,
                        value=value,
                        dtype=torch.float32,
                    )
                    restored += 1
                    del value
            gc.collect()

        print_info(f"HYV3 restored {restored} router tensor(s) in float32.")

    def _resolve_torch_dtype(self, torch_dtype, config):
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            return getattr(torch, torch_dtype)
        resolved = getattr(config, "torch_dtype", None) or torch.bfloat16
        if isinstance(resolved, str):
            return getattr(torch, resolved)
        return resolved

    def _from_pretrained_expert_parallel(
        self,
        model_path,
        torch_dtype,
        trust_remote_code,
        use_cache,
        attn_implementation,
    ):
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from tqdm import tqdm
        from transformers import GenerationConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if attn_implementation != "default":
            config._attn_implementation = attn_implementation
        if use_cache is not None:
            config.use_cache = use_cache

        resolved_dtype = self._resolve_torch_dtype(torch_dtype, config)
        print_info(
            "HYV3 expert-parallel loading: "
            f"rank={self.rank}, world_size={self.world_size}, dtype={resolved_dtype}"
        )

        with init_empty_weights(include_buffers=False):
            self.model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=resolved_dtype,
                trust_remote_code=trust_remote_code,
            )

        self._replace_moe_with_local_experts_before_load(resolved_dtype)
        self._stream_load_local_rank_weights(
            model_path=model_path,
            set_tensor=set_module_tensor_to_device,
            safe_open_fn=safe_open,
            progress_cls=tqdm,
        )

        try:
            self.model.tie_weights()
        except Exception as exc:
            print_info(f"HYV3 expert-parallel loading: tie_weights skipped: {exc}")

        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception:
            self.model.generation_config = GenerationConfig.from_model_config(self.model.config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

    def _replace_moe_with_local_experts_before_load(self, dtype):
        replaced = 0
        for name, module in tuple(self.model.named_modules()):
            if isinstance(module, HYV3ExpertsWithLinear):
                continue
            if not _is_hyv3_parameter_experts(module):
                continue
            parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
            local_experts = HYV3LocalExpertsWithLinear(
                module,
                rank=self.rank,
                world_size=self.world_size,
                dtype=dtype,
                device="cpu",
            )
            setattr(parent_layer, sub_name, local_experts)
            replaced += 1
            del module
            gc.collect()

        print_info(
            "HYV3 expert-parallel loading: replaced "
            f"{replaced} fused expert module(s) with local-only experts on rank {self.rank}."
        )

    def _iter_checkpoint_shards(self, model_path):
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
            yield shard_path, None

    def _local_expert_range(self):
        num_experts = int(getattr(self.model.config, "num_experts", 0))
        if num_experts <= 0:
            return 0, 0
        if num_experts % self.world_size != 0:
            raise ValueError(
                f"num_experts {num_experts} must be divisible by world_size {self.world_size}"
            )
        n_local_experts = num_experts // self.world_size
        start = self.rank * n_local_experts
        return start, start + n_local_experts

    def _stream_load_local_rank_weights(
        self,
        model_path,
        set_tensor,
        safe_open_fn,
        progress_cls,
    ):
        local_start, local_end = self._local_expert_range()
        name_to_param = dict(self.model.named_parameters())
        name_to_buffer = dict(self.model.named_buffers())
        target_state_dict = {}
        target_state_dict.update(name_to_param)
        target_state_dict.update(name_to_buffer)
        target_names = set(target_state_dict)
        weight_renamings, _ = self.get_checkpoint_key_conversions(include_converters=False)
        if weight_renamings:
            self.model._weight_conversions = weight_renamings

        shards = list(self._iter_checkpoint_shards(model_path))
        loaded = 0
        skipped_unavailable = 0
        seen_targets = set()

        desc = (
            f"Loading checkpoint shards rank {self.rank}/{self.world_size} "
            f"experts[{local_start},{local_end})"
        )
        for shard_path, keys in progress_cls(shards, desc=desc, disable=self.rank != 0):
            with safe_open_fn(shard_path, framework="pt") as reader:
                if keys is None:
                    keys = list(reader.keys())
                for key in keys:
                    model_key = self.resolve_checkpoint_key_for_model(
                        key,
                        target_state_dict=target_state_dict,
                        weight_renamings=weight_renamings,
                        weight_converters=[],
                    )
                    target = target_state_dict.get(model_key)
                    if target is None:
                        skipped_unavailable += 1
                        continue

                    value = reader.get_tensor(key)
                    dtype = None
                    if torch.is_floating_point(value) and torch.is_floating_point(target):
                        if self._is_router_fp32_name(model_key):
                            value = value.to(dtype=torch.float32)
                            dtype = torch.float32
                        else:
                            value = value.to(dtype=target.dtype)
                    set_tensor(self.model, model_key, "cpu", value=value, dtype=dtype)
                    seen_targets.add(model_key)
                    loaded += 1
                    del value

            gc.collect()

        meta_params = [name for name, param in self.model.named_parameters() if param.is_meta]
        meta_buffers = [name for name, buf in self.model.named_buffers() if buf.is_meta]
        if meta_params or meta_buffers:
            raise RuntimeError(
                "HYV3 expert-parallel loading left tensors on meta device: "
                f"params={meta_params[:10]}, buffers={meta_buffers[:10]}"
            )

        missing_targets = sorted(target_names - seen_targets)
        print_info(
            "HYV3 expert-parallel loading done: "
            f"rank={self.rank}, loaded={loaded}, "
            f"skipped_unavailable_checkpoint_weights={skipped_unavailable}, "
            f"missing_targets={len(missing_targets)}"
        )
        if missing_targets:
            print_info(
                "HYV3 expert-parallel loading first missing targets: " f"{missing_targets[:10]}"
            )

    def _configure_linearized_expert_parallel(self, experts_layer, layer_name):
        if not self.using_multi_nodes:
            return

        if experts_layer.num_experts % self.world_size != 0:
            raise ValueError(
                f"num_experts {experts_layer.num_experts} must be divisible by "
                f"world_size {self.world_size} for expert parallel."
            )

        n_local_experts = experts_layer.num_experts // self.world_size
        experts_start_idx = self.rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts
        experts_layer.n_local_experts = n_local_experts
        experts_layer.experts_start_idx = experts_start_idx
        experts_layer.experts_end_idx = experts_end_idx
        experts_layer.rank = self.rank
        experts_layer.world_size = self.world_size
        experts_layer.expert_parallel_enabled = True

        for expert_idx in range(experts_layer.num_experts):
            if expert_idx < experts_start_idx or expert_idx >= experts_end_idx:
                setattr(experts_layer, f"{expert_idx}", _HYV3ZeroExpert())

        print_info(
            f"Enable HYV3 expert parallel for {layer_name}: "
            f"rank={self.rank}, world_size={self.world_size}, "
            f"local_experts=[{experts_start_idx}, {experts_end_idx})"
        )

    def replace_moe(self):
        """Replace HYV3Experts instances with HYV3ExpertsWithLinear.

        This must be called before init_ptq() so that find_layers() can discover
        the per-expert nn.Linear modules and register them with the PTQ hook.
        """
        for name, module in tuple(self.model.named_modules()):
            if isinstance(module, HYV3ExpertsWithLinear):
                continue
            if not _is_hyv3_parameter_experts(module):
                continue
            parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
            moe_linear = HYV3ExpertsWithLinear(module)
            self._configure_linearized_expert_parallel(moe_linear, name)
            setattr(parent_layer, sub_name, moe_linear)

    def init_ptq(self, slim_config):
        self.replace_moe()
        super().init_ptq(slim_config)

    def _enable_expert_parallel(self):
        num_experts = getattr(self.model.config, "num_experts", 0)
        if num_experts <= 0:
            return
        assert (
            num_experts % self.world_size == 0
        ), f"num_experts {num_experts} must be divisible by world_size {self.world_size}"

        print_info(
            f"Enable HYV3 expert parallel: rank={self.rank}, "
            f"world_size={self.world_size}, num_experts={num_experts}"
        )
        for layer_idx, layer in enumerate(self.model.model.layers):
            moe_module = getattr(layer, "mlp", None)
            if moe_module is None or not hasattr(moe_module, "experts"):
                continue

            n_local_experts = moe_module.experts.num_experts // self.world_size
            experts_start_idx = self.rank * n_local_experts
            experts_end_idx = experts_start_idx + n_local_experts
            moe_module.n_local_experts = n_local_experts
            moe_module.experts_start_idx = experts_start_idx
            moe_module.experts_end_idx = experts_end_idx
            moe_module.world_size = self.world_size
            moe_module.rank = self.rank
            moe_module.ep_enabled = True

            moe_module.forward = types.MethodType(self._build_ep_forward(), moe_module)
            print_info(
                f"Layer {layer_idx} local experts: [{experts_start_idx}, {experts_end_idx})"
            )

    def _build_ep_forward(self):
        def ep_forward(moe_module, hidden_states: torch.Tensor):
            batch_size, seq_len, hidden_dim = hidden_states.shape
            x = hidden_states.view(-1, hidden_dim)

            _, top_k_weights, top_k_index = moe_module.gate(x, moe_module.e_score_correction_bias)
            expert_output = torch.zeros_like(x)
            experts_start_idx = moe_module.experts_start_idx
            experts_end_idx = moe_module.experts_end_idx

            if getattr(moe_module, "_angelslim_collect_native_full_input", False):
                native_full_input = x.detach()
                for expert_idx in range(experts_start_idx, experts_end_idx):
                    expert = getattr(moe_module.experts, f"{expert_idx}", None)
                    if expert is None:
                        continue
                    for child_name in ("gate_proj", "up_proj", "down_proj"):
                        child = (
                            expert[child_name]
                            if isinstance(expert, nn.ModuleDict)
                            else getattr(expert, child_name, None)
                        )
                        if child is not None:
                            child._angelslim_moe_native_full_input = native_full_input

            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=moe_module.experts.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            for expert_idx in range(experts_start_idx, experts_end_idx):
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.numel() == 0:
                    continue

                expert_input = x[token_idx]
                expert_scores = top_k_weights[token_idx, top_k_pos].reshape(-1, 1)
                expert = getattr(moe_module.experts, f"{expert_idx}")
                expert._angelslim_moe_token_idx = token_idx.detach()
                for child_name in ("gate_proj", "up_proj", "down_proj"):
                    child = (
                        expert[child_name]
                        if isinstance(expert, nn.ModuleDict)
                        else getattr(expert, child_name, None)
                    )
                    if child is not None:
                        child._angelslim_moe_token_idx = token_idx.detach()
                        child._angelslim_moe_expert_scores = expert_scores.detach()
                        object.__setattr__(
                            child,
                            "_angelslim_moe_parent_expert",
                            expert,
                        )

                gate = expert["gate_proj"]
                up = expert["up_proj"]
                current_hidden_states = moe_module.experts.act_fn(gate(expert_input)) * up(
                    expert_input
                )
                current_hidden_states = expert["down_proj"](current_hidden_states)
                expert_output.index_add_(
                    0,
                    token_idx,
                    (current_hidden_states * expert_scores).to(expert_output.dtype),
                )

            if dist.is_available() and dist.is_initialized() and moe_module.world_size > 1:
                dist.all_reduce(expert_output)

            shared_output = moe_module.shared_experts(x)
            if moe_module.enable_moe_fp32_combine:
                out = (expert_output.float() + shared_output.float()).to(hidden_states.dtype)
            else:
                out = expert_output + shared_output

            return out.reshape(batch_size, seq_len, hidden_dim)

        return ep_forward

    def get_observer_layers(self):
        names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "shared_experts.gate_proj",
            "shared_experts.up_proj",
            "shared_experts.down_proj",
        ]
        expert_pattern = [
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.up_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.down_proj",
        ]

        obs_layers = [nn.Linear]
        layers_dict = find_layers(self.model, layers=obs_layers)

        compiled_patterns = [re.compile(pattern) for pattern in expert_pattern]

        ignore_patterns = self.skip_layer_names()
        ignore_layers = []
        observer_layers_dict = {}
        for k, v in layers_dict.items():
            if k.startswith(self.block_name) and (
                any(name in k for name in names)
                or any(pattern.search(k) for pattern in compiled_patterns)
            ):
                # Check if this layer should be ignored based on ignore_layers config
                if any(pattern in k for pattern in ignore_patterns):
                    ignore_layers.append(k)
                else:
                    observer_layers_dict[k] = v
            else:
                ignore_layers.append(k)

        ignore_layers = sorted(list(set(ignore_layers)))
        self.quant_config.quant_algo_info["ignore_layers"] = ignore_layers

        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in observer_layers_dict.keys():
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)
        return observer_layers_dict

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

    def get_kvcache_observer_layers_names(self, observe_names):
        """Return empty list since we use attention-level patching for KV cache."""
        # Return empty list to disable the default k_proj/v_proj output observation
        # We will use apply_kvcache_observers() instead for RoPE-after key/value states
        return []

    def get_attention_layers(self):
        """Get all attention layers in the model."""
        attention_layers = {}
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn") and hasattr(module, "forward"):
                # Verify it has k_proj and v_proj attributes
                if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                    attention_layers[name] = module
        return attention_layers

    def apply_kvcache_observers(self, kv_cache_observer_class, quant_bits=8):
        """
        Apply KV cache observers to attention layers using monkey patching.
        This observes key_states and value_states AFTER RoPE is applied.

        Args:
            kv_cache_observer_class: The observer class to use (e.g., AbsmaxPertensorObserver)
            quant_bits: Quantization bits for the observer
        """
        from ...compressor.quant.observers import AbsmaxPertensorObserver

        if kv_cache_observer_class is None:
            kv_cache_observer_class = AbsmaxPertensorObserver

        attention_layers = self.get_attention_layers()

        for attn_name, attn_module in attention_layers.items():
            # Create observers for key and value states
            key_observer = kv_cache_observer_class(
                layer=attn_module.k_proj,
                quant_bits=quant_bits,
            )
            value_observer = kv_cache_observer_class(
                layer=attn_module.v_proj,
                quant_bits=quant_bits,
            )

            # Store observers
            self.kv_cache_observers[attn_name] = {
                "key_observer": key_observer,
                "value_observer": value_observer,
            }

            # Save original forward
            self._original_attn_forwards[attn_name] = attn_module.forward

            # Create patched forward
            self._patch_attention_forward(attn_module, attn_name)

    def _patch_attention_forward(self, attn_module, attn_name):
        """
        Patch the attention module's forward method to observe KV cache after RoPE.

        Adapted to the new transformers ``HYV3Attention.forward`` signature, where
        rotary embeddings are pre-computed and passed in as ``position_embeddings``
        (a ``(cos, sin)`` tuple), ``q_norm``/``k_norm`` are applied unconditionally
        on the pre-transpose view, and attention dispatch goes through
        ``ALL_ATTENTION_FUNCTIONS``.
        """
        key_observer = self.kv_cache_observers[attn_name]["key_observer"]
        value_observer = self.kv_cache_observers[attn_name]["value_observer"]

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)

            query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(key_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # === OBSERVE KV CACHE AFTER RoPE ===
            key_observer(key_states)
            value_observer(value_states)
            # === END OBSERVE ===

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn_module.layer_idx
                )

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_module.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                attn_module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
                scaling=attn_module.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, attn_weights

        # Replace the forward method
        attn_module.forward = patched_forward

    def remove_kvcache_observers(self):
        """Remove patched forward methods and restore original ones."""
        for attn_name, original_forward in self._original_attn_forwards.items():
            # Find the attention module and restore its forward
            parts = attn_name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)
            module.forward = original_forward

        self._original_attn_forwards.clear()

    def get_kvcache_scales(self):
        """
        Get KV cache scales from observers.
        Returns dict with format: {"layer_name.k_cache.scale": scale,
                                   "layer_name.v_cache.scale": scale}
        """
        kv_scales = {}
        for attn_name, observers in self.kv_cache_observers.items():
            key_scale = observers["key_observer"].scales()
            value_scale = observers["value_observer"].scales()
            kv_scales[f"{attn_name}.k_cache.scale"] = key_scale
            kv_scales[f"{attn_name}.v_cache.scale"] = value_scale
        return kv_scales

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
