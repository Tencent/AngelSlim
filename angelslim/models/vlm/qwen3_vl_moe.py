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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLMoeForConditionalGeneration,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextExperts,
    Qwen3VLMoeTextTopKRouter,
)

from angelslim.compressor.quant.core.quant_func import get_fp_maxval
from angelslim.compressor.quant.observers import (
    MoEAbsmaxPertensorObserver,
    ParentObserver,
)

from ...compressor.quant.core import LossFilter, PTQVLMSaveVllmHF
from ...compressor.quant.modules import MoEQDQModule
from ...utils import find_layers, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


def moe_observer_forward(
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
        self.gateupobservers[expert_idx](current_state.reshape(1, -1))
        gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(
            2, dim=-1
        )
        current_hidden_states = self.act_fn(gate) * up
        self.downobservers[expert_idx](current_hidden_states.reshape(1, -1))
        current_hidden_states = nn.functional.linear(
            current_hidden_states, self.down_proj[expert_idx]
        )
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


@SlimModelFactory.register
class Qwen3VLMoE(BaseLLMModel):
    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.modal_type = "VLM"
        self.block_name = "model.language_model.layers"
        self.vit_block_name = "model.visual.blocks"
        self.pre_transformer_module_names = [
            "visual",
            "language_model.embed_tokens",
            "language_model.norm",
            "language_model.rotary_emb",
        ]
        self.observer_layer_classes = [nn.Linear, Qwen3VLMoeTextExperts, Qwen3VLMoeTextTopKRouter]

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
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def init_ptq(self, slim_config):
        for _, module in self.model.named_modules():
            if isinstance(module, Qwen3VLMoeTextExperts):
                module.forward = moe_observer_forward.__get__(module, Qwen3VLMoeTextExperts)
        super().init_ptq(slim_config)

    def get_observer_layers(self):
        names = [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
        ]

        if hasattr(self.quant_config, "quant_vit") and self.quant_config.quant_vit:
            vit_names = ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]
            names.extend(vit_names)

        observer_layers_dict = {}
        layers_dict = find_layers(self.model, layers=self.observer_layer_classes)

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            block_condition = name.startswith(self.block_name) or (
                hasattr(self.quant_config, "quant_vit")
                and self.quant_config.quant_vit
                and name.startswith(self.vit_block_name)
            )
            parts = name.split(".")
            result = ".".join(parts[-2:])
            if result == "mlp.experts":
                if not hasattr(module, "gateupobservers"):
                    parent = ParentObserver()
                    module.gateupobservers = nn.ModuleList(
                        [
                            MoEAbsmaxPertensorObserver(
                                layer_name=f"{name}.gate_up.expert_{i}", parent_observer=parent
                            )
                            for i in range(module.num_experts)
                        ]
                    )
                if not hasattr(module, "downobservers"):
                    parent = ParentObserver()
                    module.downobservers = nn.ModuleList(
                        [
                            MoEAbsmaxPertensorObserver(
                                layer_name=f"{name}.down.expert_{i}", parent_observer=parent
                            )
                            for i in range(module.num_experts)
                        ]
                    )
            else:
                if block_condition and result in names:
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

    def get_moe_qdq_module(self, sub_layer, name):
        if not isinstance(sub_layer, Qwen3VLMoeTextExperts):
            return sub_layer
        maxval = get_fp_maxval(bits=8)

        # Stack per-expert activation scales: [num_experts, 1]
        gate_up_act_max = torch.stack(
            [obs.scales() for obs in sub_layer.gateupobservers], dim=0
        ).view(-1, 1)
        down_act_max = torch.stack([obs.scales() for obs in sub_layer.downobservers], dim=0).view(
            -1, 1
        )
        gate_up_act_dtype = gate_up_act_max.dtype
        down_act_dtype = down_act_max.dtype
        gate_up_act_scale = gate_up_act_max / maxval.type(gate_up_act_dtype)
        down_act_scale = down_act_max / maxval.type(down_act_dtype)

        # gate_up_proj shape from nn.functional.linear: [num_experts, out, in]
        # transpose to [num_experts, in, out] to match old format (used with x @ w)
        gate_up_proj_t = sub_layer.gate_up_proj.transpose(1, 2).contiguous()
        down_proj_t = sub_layer.down_proj.transpose(1, 2).contiguous()

        gate_proj, up_proj = gate_up_proj_t.chunk(2, dim=-1)
        abs_inputs = torch.abs(gate_proj)
        batch_size = abs_inputs.shape[0]
        abs_inputs_flat = abs_inputs.view(batch_size, -1)
        gate_weight_max, _ = torch.max(abs_inputs_flat, dim=1, keepdim=True)

        abs_inputs = torch.abs(up_proj)
        batch_size = abs_inputs.shape[0]
        abs_inputs_flat = abs_inputs.view(batch_size, -1)
        up_weight_max, _ = torch.max(abs_inputs_flat, dim=1, keepdim=True)

        abs_inputs = torch.abs(down_proj_t)
        batch_size = abs_inputs.shape[0]
        abs_inputs_flat = abs_inputs.view(batch_size, -1)
        down_weight_max, _ = torch.max(abs_inputs_flat, dim=1, keepdim=True)

        gate_weight_dtype = gate_proj.dtype
        up_weight_dtype = up_proj.dtype
        down_weight_dtype = down_proj_t.dtype
        gate_weight_scale = gate_weight_max / maxval.type(gate_weight_dtype)
        up_weight_scale = up_weight_max / maxval.type(up_weight_dtype)
        down_weight_scale = down_weight_max / maxval.type(down_weight_dtype)

        q_linear = MoEQDQModule(
            gate_proj=gate_proj.cpu(),
            up_proj=up_proj.cpu(),
            down_proj=down_proj_t.cpu(),
            gate_proj_weight_scale=gate_weight_scale.cpu(),
            up_proj_weight_scale=up_weight_scale.cpu(),
            down_proj_weight_scale=down_weight_scale.cpu(),
            gate_up_proj_input_scale=gate_up_act_scale.cpu(),
            down_proj_input_scale=down_act_scale.cpu(),
        )
        return q_linear

    def model_forward(self, dataloader, **kwargs):
        self.model.use_cache = False

        calibrated_cnt = 0
        if (
            "gptq" in self.quant_config.quant_algo
            or "awq" in self.quant_config.quant_algo
            or "gptaq" in self.quant_config.quant_algo
        ):
            device = "cuda:0"
        else:
            device = self.model.device
        print_info(f"device is {device}")
        if dataloader is not None:
            loss_filter = LossFilter(processor=self.processor)
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="calibrating...", total=len(dataloader)):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    inputs["use_cache"] = False
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    try:
                        outputs = self.model(**inputs)
                        logits = outputs.logits.float()

                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            reduction="none",
                        )

                        attention_mask = attention_mask.view(-1).to(logits.device).float()
                        loss = loss * attention_mask
                        loss = loss_filter.filter_loss(
                            loss=loss, labels=labels, model_type="Qwen3VL"
                        )
                        avg_loss = loss.mean()
                        ppl = torch.exp(avg_loss)

                        print_info(f"ppl is : {ppl:.4f}")

                        calibrated_cnt += 1
                    except ValueError:
                        calibrated_cnt += 1
                        pass
                    inputs = {
                        k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    attention_mask = attention_mask.to("cpu")
                    labels = labels.to("cpu")
                    del outputs, inputs
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

    def get_quant_module(self):
        """
        Returns the module that will be quantized.
        This is typically the main transformer module of the model.
        """
        return self.model.model.language_model.layers

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQVLMSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )
