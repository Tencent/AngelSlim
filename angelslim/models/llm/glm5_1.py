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

"""AngelSlim model adapter for GLM-5 (``glm_moe_dsa``).

GLM-5 combines a DeepSeek-V3.2-style MLA attention + DSA indexer with an
HYV3-style MoE whose expert weights are stored as fused 3-D ``nn.Parameter``
tensors. This adapter targets **weight-only NVFP4-GPTQ quantization of the
routed experts only** (gate/up/down per expert); attention, the DSA indexer,
the router, shared experts, and the leading dense MLP layers are left intact.

The input checkpoint is expected to be a plain **bf16** model

The fused-expert linearization, the GPTQ-MoE token-routing hooks, and the
optional expert-parallel streaming loader are lifted almost verbatim from
``hunyuan_v3_moe.py`` because ``GlmMoeDsaNaiveMoe`` and ``HYV3Experts`` share
an identical weight layout.
"""

import gc
import glob
import json
import os
import re

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaNaiveMoe

from ...compressor.quant.core import PTQSaveVllmHF
from ...utils.utils import find_parent_layer_and_sub_name, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


def _is_glm_naive_moe(module):
    """True for a GlmMoeDsaNaiveMoe (or a structural look-alike) holding fused experts."""
    if GlmMoeDsaNaiveMoe is not None and isinstance(module, GlmMoeDsaNaiveMoe):
        return True
    required_attrs = ("gate_up_proj", "down_proj", "num_experts", "hidden_dim", "intermediate_dim")
    return all(hasattr(module, attr) for attr in required_attrs) and isinstance(
        getattr(module, "gate_up_proj", None), nn.Parameter
    )


class _GlmZeroExpert(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x.new_zeros((x.shape[0], x.shape[-1]))


class GlmExpertsWithLinear(GlmMoeDsaNaiveMoe):
    """Expose ``GlmMoeDsaNaiveMoe`` fused 3-D expert weights as per-expert nn.Linear.

    ``GlmMoeDsaNaiveMoe`` stores all expert weights as 3-D ``nn.Parameter``
    tensors, invisible to ``find_layers()`` and the PTQ hook (both only
    recognise ``nn.Linear``). This wrapper splits them into individual
    ``nn.Linear`` modules so the standard quantisation pipeline can observe and
    quantise each expert projection.

    Weight shape mapping (identical to HYV3):
        gate_up_proj : [num_experts, 2*intermediate_dim, hidden_dim]
            gate_up_proj[i] -> chunk(2, dim=0)
                gate_proj[i].weight : [intermediate_dim, hidden_dim]
                up_proj[i].weight   : [intermediate_dim, hidden_dim]
        down_proj : [num_experts, hidden_dim, intermediate_dim]
            down_proj[i] -> down_proj[i].weight : [hidden_dim, intermediate_dim]
    """

    def __init__(self, experts_layer):
        # Bypass GlmMoeDsaNaiveMoe.__init__ to avoid allocating large empty
        # Parameter tensors we would immediately overwrite.
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
            # chunk on dim=0 -> [intermediate_dim, hidden_dim] each
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
        """Mirror GlmMoeDsaNaiveMoe.forward over the linearized experts.

        Injects the per-expert token-routing metadata that the GPTQ-MoE hook
        consumes (``_angelslim_moe_token_idx`` / ``_angelslim_moe_expert_scores``)
        so each expert's Hessian only accumulates over the tokens routed to it.
        """
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


class GlmLocalExpertsWithLinear(GlmExpertsWithLinear):
    """Expert-parallel variant: only this rank's expert slice gets real Linears."""

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
                expert = _GlmZeroExpert()
            setattr(self, f"{expert_idx}", expert)


@SlimModelFactory.register
class GLM5_1(BaseLLMModel):
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
        # The DSA indexer builds a custom additive attention mask and is only
        # validated against eager attention; force eager so the calibration
        # forward matches the reference math.
        attn_implementation = "eager"
        torch_dtype = torch.bfloat16

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
            self._enable_expert_parallel()
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

    # ------------------------------------------------------------------
    # MoE linearization
    # ------------------------------------------------------------------
    def replace_moe(self):
        """Replace GlmMoeDsaNaiveMoe instances with GlmExpertsWithLinear.

        Must run before init_ptq() so find_layers() can discover the per-expert
        nn.Linear modules and register them with the PTQ hook.
        """
        from tqdm import tqdm

        moe_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if not isinstance(module, GlmExpertsWithLinear) and _is_glm_naive_moe(module)
        ]
        for name, module in tqdm(moe_modules, desc="Replacing MoE layers"):
            parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
            moe_linear = GlmExpertsWithLinear(module)
            self._configure_linearized_expert_parallel(moe_linear, name)
            setattr(parent_layer, sub_name, moe_linear)

    def init_ptq(self, slim_config):
        self.replace_moe()
        super().init_ptq(slim_config)

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
                setattr(experts_layer, f"{expert_idx}", _GlmZeroExpert())
        print_info(
            f"Enable GLM5 expert parallel for {layer_name}: rank={self.rank}, "
            f"world_size={self.world_size}, local_experts=[{experts_start_idx}, {experts_end_idx})"
        )

    # ------------------------------------------------------------------
    # Expert-parallel streaming loader (optional; for multi-node low-mem load)
    # ------------------------------------------------------------------
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
        config._attn_implementation = attn_implementation
        if use_cache is not None:
            config.use_cache = use_cache

        resolved_dtype = self._resolve_torch_dtype(torch_dtype, config)
        print_info(
            "GLM5 expert-parallel loading: "
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
            print_info(f"GLM5 expert-parallel loading: tie_weights skipped: {exc}")

        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception:
            self.model.generation_config = GenerationConfig.from_model_config(self.model.config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def _replace_moe_with_local_experts_before_load(self, dtype):
        replaced = 0
        for name, module in tuple(self.model.named_modules()):
            if isinstance(module, GlmExpertsWithLinear):
                continue
            if not _is_glm_naive_moe(module):
                continue
            parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
            local_experts = GlmLocalExpertsWithLinear(
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
            f"GLM5 expert-parallel loading: replaced {replaced} fused expert module(s) "
            f"with local-only experts on rank {self.rank}."
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
        num_experts = int(getattr(self.model.config, "n_routed_experts", 0)) or int(
            getattr(self.model.config, "num_local_experts", 0)
        )
        if num_experts <= 0:
            return 0, 0
        if num_experts % self.world_size != 0:
            raise ValueError(
                f"num_experts {num_experts} must be divisible by world_size {self.world_size}"
            )
        n_local_experts = num_experts // self.world_size
        start = self.rank * n_local_experts
        return start, start + n_local_experts

    def _is_router_fp32_name(self, name):
        return name.endswith(".mlp.gate.weight") or name.endswith(
            ".mlp.gate.e_score_correction_bias"
        )

    def _stream_load_local_rank_weights(self, model_path, set_tensor, safe_open_fn, progress_cls):
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
        local_start, local_end = self._local_expert_range()
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
                "GLM5 expert-parallel loading left tensors on meta device: "
                f"params={meta_params[:10]}, buffers={meta_buffers[:10]}"
            )
        missing_targets = sorted(target_names - seen_targets)
        print_info(
            f"GLM5 expert-parallel loading done: rank={self.rank}, loaded={loaded}, "
            f"skipped_unavailable_checkpoint_weights={skipped_unavailable}, "
            f"missing_targets={len(missing_targets)}"
        )
        if missing_targets:
            print_info(f"GLM5 expert-parallel first missing targets: {missing_targets[:10]}")

    def _enable_expert_parallel(self):
        num_experts = int(getattr(self.model.config, "n_routed_experts", 0))
        if num_experts <= 0:
            return
        assert (
            num_experts % self.world_size == 0
        ), f"num_experts {num_experts} must be divisible by world_size {self.world_size}"
        print_info(
            f"Enable GLM5 expert parallel: rank={self.rank}, "
            f"world_size={self.world_size}, num_experts={num_experts}"
        )

    # ------------------------------------------------------------------
    # Observation / quantization config
    # ------------------------------------------------------------------
    def get_observer_layers(self):
        """Collect only the routed-expert projections; ignore everything else.

        Weight-only NVFP4-GPTQ targets the MoE experts. Attention (MLA), the
        DSA indexer, the router gate, shared experts, the leading dense MLP
        layers, embeddings, and lm_head are all routed to ignore_layers.
        """
        from ...utils.utils import find_layers

        expert_pattern = [
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.up_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.down_proj",
        ]
        compiled_patterns = [re.compile(p) for p in expert_pattern]

        layers_dict = find_layers(self.model, layers=[nn.Linear])
        ignore_patterns = self.skip_layer_names()
        ignore_layers = []
        observer_layers_dict = {}
        for k, v in layers_dict.items():
            is_expert = k.startswith(self.block_name) and any(
                p.search(k) for p in compiled_patterns
            )
            if is_expert and not any(pat in k for pat in ignore_patterns):
                observer_layers_dict[k] = v
            else:
                ignore_layers.append(k)

        ignore_layers = sorted(list(set(ignore_layers)))
        self.quant_config.quant_algo_info["ignore_layers"] = ignore_layers

        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in list(observer_layers_dict.keys()):
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
        # Weight-only: no KV-cache quantization.
        return []

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQSaveVllmHF
        raise NotImplementedError(
            f"deploy_backend {self.deploy_backend} is not supported for saving."
        )
