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

import re

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeTopKRouter,
)

from ...compressor.qat.modules.quantizer import fp8_cast_ste
from ...compressor.quant.core import PTQSaveVllmHF
from ...utils.utils import find_layers, find_parent_layer_and_sub_name, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


class QwenMoeExpertsWithLinear(Qwen3MoeExperts):

    def __init__(self, experts_layer):
        super().__init__(experts_layer.config)
        self.gate_up_proj = experts_layer.gate_up_proj
        self.down_proj = experts_layer.down_proj
        for expert_idx in range(self.num_experts):
            expert = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "up_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "down_proj": nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False),
                }
            )
            expert["gate_proj"].weight.data, expert["up_proj"].weight.data = self.gate_up_proj[
                expert_idx
            ].chunk(2, dim=-2)
            expert["down_proj"].weight.data = self.down_proj[expert_idx]
            setattr(self, f"{expert_idx}", expert)
        del self.gate_up_proj
        del self.down_proj

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
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert_layer = getattr(self, f"{expert_idx}")
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


@SlimModelFactory.register
class Qwen(BaseLLMModel):
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
        self.observer_layer_classes = [torch.nn.Linear, Qwen3MoeTopKRouter]
        self.observed_names = [
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def replace_moe(self):
        for name, module in self.model.named_modules():
            if isinstance(module, Qwen3MoeExperts) and not isinstance(
                module, QwenMoeExpertsWithLinear
            ):
                print(name)
                parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
                moe_linear = QwenMoeExpertsWithLinear(module)
                del module
                setattr(parent_layer, sub_name, moe_linear)

    def init_ptq(self, slim_config):
        self.replace_moe()
        super().init_ptq(slim_config)

    def get_observer_layers(self):
        observer_layers_dict = {}
        layers_dict = find_layers(self.model, layers=self.observer_layer_classes)

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            # todo: shared_experts
            if name.startswith(self.block_name) and name.split(".")[-1] in self.observed_names:
                observer_layers_dict[name] = module
            else:
                ignore_layers.append(name)
        self.quant_config.quant_algo_info["ignore_layers"] = ignore_layers

        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in observer_layers_dict.keys():
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)
        return observer_layers_dict

    def get_smooth_mapping_layers(self, smooth_config, mappings=None):
        if mappings is None:
            mappings = [
                (["q_proj", "k_proj", "v_proj"], "input_layernorm"),
                (["gate_proj", "up_proj"], "post_attention_layernorm"),
            ]
        print(f"smooth mappings={mappings}")
        assert len(mappings) == 2
        assert smooth_config.smooth_first_linears or smooth_config.smooth_last_linears
        # TODO: support smooth_last_linears
        return super().get_smooth_mapping_layers(smooth_config, mappings)

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

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )

    def fuse_observer_amax(self, sub_layer, name):
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            prefix = name.rsplit(".", 1)[0]
            q_name = f"{prefix}.q_proj"
            k_name = f"{prefix}.k_proj"
            v_name = f"{prefix}.v_proj"

            weight_scales = []
            for key in [q_name, k_name, v_name]:
                tensor = self.weight_observer_amax_dict[key]
                weight_scales.append(tensor)
            weight_observer_amax = max(weight_scales)

            act_scales = []
            for key in [q_name, k_name, v_name]:
                tensor = self.input_observer_amax_dict[key]
                act_scales.append(tensor)
            input_observer_amax = max(act_scales)
        elif "gate_proj" in name or "up_proj" in name:
            prefix = name.rsplit(".", 1)[0]
            gate_name = f"{prefix}.gate_proj"
            up_name = f"{prefix}.up_proj"

            weight_scales = []
            for key in [gate_name, up_name]:
                tensor = self.weight_observer_amax_dict[key]
                weight_scales.append(tensor)
            weight_observer_amax = max(weight_scales)

            act_scales = []
            for key in [gate_name, up_name]:
                tensor = self.input_observer_amax_dict[key]
                act_scales.append(tensor)
            input_observer_amax = max(act_scales)
        else:
            weight_observer_amax = self.weight_observer_amax_dict[name]
            input_observer_amax = self.input_observer_amax_dict[name]

        return weight_observer_amax, input_observer_amax

    def get_attention_layers(self):
        """Get all attention layers in the model."""
        attention_layers = {}
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn") and hasattr(module, "forward"):
                # Verify it has k_proj and v_proj attributes
                if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                    attention_layers[name] = module
        return attention_layers

    def patch_fp8_attention(self):
        attention_layers = self.get_attention_layers()

        for attn_name, attn_module in attention_layers.items():
            original_forward = attn_module.forward
            self._original_attn_forwards.setdefault(attn_name, original_forward)

            def _make_patched(attn_mod):
                def patched_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask=None,
                    past_key_values=None,
                    cache_position=None,
                    **kwargs,
                ):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn_mod.head_dim)

                    query_states = attn_mod.q_norm(
                        attn_mod.q_proj(hidden_states).view(hidden_shape)
                    ).transpose(1, 2)
                    key_states = attn_mod.k_norm(
                        attn_mod.k_proj(hidden_states).view(hidden_shape)
                    ).transpose(1, 2)
                    value_states = (
                        attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    )

                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin
                    )

                    if past_key_values is not None:
                        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                        key_states, value_states = past_key_values.update(
                            key_states, value_states, attn_mod.layer_idx, cache_kwargs
                        )

                    dropout = 0.0 if not attn_mod.training else attn_mod.attention_dropout
                    key_states = repeat_kv(key_states, attn_mod.num_key_value_groups)
                    value_states = repeat_kv(value_states, attn_mod.num_key_value_groups)

                    attn_weights = (
                        torch.matmul(query_states, key_states.transpose(2, 3)) * attn_mod.scaling
                    )
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask

                    attn_weights = nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)
                    attn_weights = nn.functional.dropout(
                        attn_weights, p=dropout, training=attn_mod.training
                    )
                    # === FP8 cast simulation ===
                    attn_weights = fp8_cast_ste(attn_weights)
                    attn_output = torch.matmul(attn_weights, value_states)
                    attn_output = attn_output.transpose(1, 2).contiguous()

                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = attn_mod.o_proj(attn_output)
                    return attn_output, attn_weights

                return patched_forward

            attn_module.forward = _make_patched(attn_module)

        print_info(f"FP8 attention enabled for {len(attention_layers)} layers (patch forward)")
