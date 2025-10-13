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

import torch

from .....utils import get_op_by_name, get_op_name, print_info, set_op_by_name
from ...core import mse_loss, per_block_weight_quant, weight_dequant
from ...modules.helper_layer import SmoothHelpModule
from .search import AWQSearch


class AutoLayerScale:
    def __init__(
        self,
        weight_bits=4,
        weight_quant_method="groupwise",
        group_size=64,
        n_grid=20,
        smooth_all_linears=False,
        loss_function=mse_loss,
        merge_samples=True,
        model_type="dense",
        observer_layer_classes=None,
        low_memory=False,
    ):
        """
        The implementation from AWQ(https://arxiv.org/pdf/2306.00978.pdf).
        """
        self.weight_bits = weight_bits
        self.weight_quant_method = weight_quant_method
        self.group_size = group_size
        self.n_grid = n_grid
        self.smooth_all_linears = smooth_all_linears
        self.loss_function = loss_function
        self.merge_samples = merge_samples
        self.model_type = model_type
        self.layer_count = 0
        self.observer_layer_classes = observer_layer_classes
        self.low_memory = low_memory
        self.search_function = AWQSearch(
            n_grid=n_grid,
            bits_length=weight_bits,
            weight_quant_method=weight_quant_method,
            group_size=group_size,
            merge_samples=merge_samples,
            observer_layer_classes=observer_layer_classes,
            low_memory=low_memory,
        )

    def apply_scale(self, module, scales_list, input_feat_dict=None):
        print_info("[apply scale] start")
        for prev_op_name, layer_names, scales in scales_list:
            print_info(
                f"[apply scale] {prev_op_name} -> {layer_names},scales:{scales[0]}"
            )
            prev_op = get_op_by_name(module, prev_op_name)
            layers = [get_op_by_name(module, name) for name in layer_names]

            if "norm" in prev_op_name:
                scales = scales.to(prev_op.weight.device)
                prev_op.weight.div_(scales)

                for p in prev_op.parameters():
                    assert torch.isnan(p).sum() == 0, f"nan in {prev_op_name} weight"

                for layer in layers:
                    if layer.weight.dtype == torch.float8_e4m3fn:
                        weight = weight_dequant(layer.weight, layer.weight_scale_inv)
                        weight.mul_(scales.view(1, -1))
                        weight, _ = per_block_weight_quant(weight)
                        layer.weight.data.copy_(weight)
                    else:
                        layer.weight.mul_(scales.view(1, -1))
                    for p in layer.parameters():
                        assert torch.isnan(p).sum() == 0, f"nan in {layer_names} weight"

            elif "activation_func" in prev_op_name:
                scales = scales.to(prev_op.weight.device)
                new_module = SmoothHelpModule(layers[0])
                new_module.convert_weight(scales)
                set_op_by_name(module, prev_op_name, new_module)

            if input_feat_dict is not None:
                for layer_name in layer_names:
                    if layer_name not in input_feat_dict.keys():
                        continue
                    inp = input_feat_dict[layer_name].to(scales.device)
                    if inp.dim() == 2:
                        inp = inp.reshape(-1, inp.shape[0], scales.shape[-1])
                    inp.div_(scales.view(1, -1))
                    inp = inp.cpu()

            scales = scales.cpu()
        print_info("[apply scale] end")

    def auto_scale(self, module, input_feat, cache):
        print_info("[auto scale] start")

        def _auto_get_scale(
            layer_name, prev_op, layers, inp, module2inspect=None, cache=None
        ):
            if module2inspect is None:
                assert len(layers) == 1
                module2inspect = layers[0]
            if not self.low_memory:
                inp = inp.to(prev_op.weight.device)
            if self.merge_samples:
                act_abs_max = (
                    inp.abs().reshape(-1, inp.shape[-1]).mean(0).reshape(1, -1)
                )
            else:
                all_inp = inp
                act_abs_max = (
                    all_inp.abs().reshape(-1, all_inp.shape[-1]).mean(0).reshape(1, -1)
                )
                del all_inp

            print_info(f"[auto scale] {layer_name} act_abs_max: {act_abs_max}")
            scales = self.search_function.search_by_block(
                layer_name,
                inp,
                act_abs_max,
                layers,
                module2inspect,
                cache,
                self.layer_count,
            )
            scales = scales.detach().cpu()
            print_info(f"[auto scale] {layer_name} scales: {scales}")
            inp = inp.cpu()
            torch.cuda.empty_cache()

            # prev_op_name, [layer_name], scale
            return (
                get_op_name(module, prev_op),
                tuple([get_op_name(module, m) for m in layers]),
                scales,
            )

        scales_list = []
        print_info(input_feat.keys())
        if self.model_type == "deepseek_v3":
            scales_list.append(
                _auto_get_scale(
                    layer_name="attn.qkv",
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_a_proj,
                        module.self_attn.kv_a_proj_with_mqa,
                    ],
                    inp=input_feat["self_attn.q_a_proj"],
                    module2inspect=module.self_attn,
                    cache=cache,
                )
            )

            # attention output
            scales_list.append(
                _auto_get_scale(
                    layer_name="attn.o",
                    prev_op=module.self_attn.kv_b_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        else:
            scales_list.append(
                _auto_get_scale(
                    layer_name="attn.qkv",
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    cache=cache,
                )
            )

            # attention output
            if (
                module.self_attn.v_proj.weight.shape
                == module.self_attn.o_proj.weight.shape
            ):
                scales_list.append(
                    _auto_get_scale(
                        layer_name="attn.o",
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

        if hasattr(module.mlp, "gate"):
            print_info("auto scale -> MoeAWQ")
            if self.model_type == "hunyuan_v1_moe":
                # share_mlp fc1
                scales_list.append(
                    _auto_get_scale(
                        layer_name="shared_mlp.gate_proj",
                        prev_op=module.post_attention_layernorm,
                        layers=[
                            module.mlp.shared_mlp.gate_proj,
                            module.mlp.shared_mlp.up_proj,
                        ],
                        inp=input_feat["mlp"],
                        module2inspect=module.mlp,
                        cache=cache,
                    )
                )
                # share_mlp fc2
                scales_list.append(
                    _auto_get_scale(
                        layer_name="shared_mlp.down_proj",
                        prev_op=module.mlp.shared_mlp.up_proj,
                        layers=[module.mlp.shared_mlp.down_proj],
                        inp=input_feat["mlp.shared_mlp.down_proj"],
                    )
                )
                # fc1
                scales_list.append(
                    _auto_get_scale(
                        layer_name="expert.gate_proj",
                        prev_op=module.post_attention_layernorm,
                        layers=[
                            w
                            for expert in module.mlp.experts
                            for w in [expert.gate_proj, expert.up_proj]
                        ],
                        inp=input_feat["mlp"],
                        module2inspect=module.mlp,
                        cache=cache,
                    )
                )
                # fc2
                for i, expert in enumerate(module.mlp.experts):
                    scales_list.append(
                        _auto_get_scale(
                            layer_name="mlp.down_proj",
                            prev_op=expert.up_proj,
                            layers=[expert.down_proj],
                            inp=input_feat[f"mlp.experts.{i}.down_proj"],
                        )
                    )
            elif self.model_type == "deepseek_v3":
                # share_mlp fc1
                scales_list.append(
                    _auto_get_scale(
                        layer_name="shared_experts.gate_proj",
                        prev_op=module.post_attention_layernorm,
                        layers=[
                            module.mlp.shared_experts.gate_proj,
                            module.mlp.shared_experts.up_proj,
                        ],
                        inp=input_feat["mlp"],
                        module2inspect=module.mlp,
                        cache=cache,
                    )
                )
                # share_mlp fc2
                scales_list.append(
                    _auto_get_scale(
                        layer_name="shared_experts.down_proj",
                        prev_op=module.mlp.shared_experts.up_proj,
                        layers=[module.mlp.shared_experts.down_proj],
                        inp=input_feat["mlp.shared_experts.down_proj"].view(
                            input_feat["mlp"].shape[0], input_feat["mlp"].shape[1], -1
                        ),
                    )
                )
                # fc1
                scales_list.append(
                    _auto_get_scale(
                        layer_name="expert.gate_proj",
                        prev_op=module.post_attention_layernorm,
                        layers=[
                            w
                            for expert in module.mlp.experts
                            for w in [expert.gate_proj, expert.up_proj]
                        ]
                        + [module.mlp.gate],
                        inp=input_feat["mlp"],
                        module2inspect=module.mlp,
                        cache=cache,
                    )
                )
                # fc2
                for i, expert in enumerate(module.mlp.experts):
                    scales_list.append(
                        _auto_get_scale(
                            layer_name=f"expert.{i}.down_proj",
                            prev_op=expert.up_proj,
                            layers=[expert.down_proj],
                            inp=input_feat[f"mlp.experts.{i}.down_proj"].unsqueeze(0),
                        )
                    )
            else:
                # fc1
                scales_list.append(
                    _auto_get_scale(
                        layer_name="expert.gate_proj",
                        prev_op=module.post_attention_layernorm,
                        layers=[
                            w
                            for expert in module.mlp.experts
                            for w in [expert.gate_proj, expert.up_proj]
                        ],
                        inp=input_feat["mlp"],
                        module2inspect=module.mlp,
                        cache=cache,
                    )
                )
                # fc2
                for i, expert in enumerate(module.mlp.experts):
                    scales_list.append(
                        _auto_get_scale(
                            layer_name="mlp.down_proj",
                            prev_op=expert.up_proj,
                            layers=[expert.down_proj],
                            inp=input_feat[f"mlp.experts.{i}.down_proj"].unsqueeze(0),
                        )
                    )
        else:
            print_info("auto scale -> DenseAWQ")
            # fc1
            scales_list.append(
                _auto_get_scale(
                    layer_name="mlp.gate_proj",
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                    cache=cache,
                )
            )
            # fc2
            scales_list.append(
                _auto_get_scale(
                    layer_name="mlp.down_proj",
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )

        self.layer_count += 1
        print_info("[auto scale] end")
        return scales_list
