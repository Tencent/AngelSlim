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


import functools

import torch

from .....utils import get_op_name, print_info
from ...core import get_fp_maxval, mse_loss
from ...core.quant_func import quantize_weight_per_tensor_fp8, tensor_quant_dequant_fp8


class AutoLayerScale:
    def __init__(
        self,
        loss_function=mse_loss,
        merge_samples=True,
        model_type="dense",
        observer_layer_classes=None,
    ):
        """ """
        self.loss_function = loss_function
        self.merge_samples = merge_samples
        self.model_type = model_type
        self.layer_count = 0
        self.observer_layer_classes = observer_layer_classes
        self.n_exponent = 5
        self.search_step = 10

    def auto_scale(self, ptq_hook, module, input_feat, cache):
        print_info("[auto scale] start")

        def _auto_get_scale(layer_name, layers, inp, module2inspect=None, cache=None):
            if module2inspect is None:
                assert len(layers) == 1
                module2inspect = layers[0]

            inp = inp.to(layers[0].weight.device)
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

            scales = self.search_by_block(
                layer_name,
                inp,
                act_abs_max,
                layers,
                module2inspect,
                cache,
            )
            scales = scales.detach().cpu()
            print_info(f"[auto scale] {layer_name} scales: {scales}")
            inp = inp.cpu()
            torch.cuda.empty_cache()

            # prev_op_name, [layer_name], scale
            return (
                tuple([get_op_name(module, m) for m in layers]),
                scales,
            )

        scales_list = []
        print_info(input_feat.keys())
        scales_list.append(
            _auto_get_scale(
                layer_name="attn.qkv",
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
        scales_list.append(
            _auto_get_scale(
                layer_name="attn.o",
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                cache=cache,
            )
        )

        print_info("auto scale -> Denselepto")
        # fc1
        scales_list.append(
            _auto_get_scale(
                layer_name="mlp.gate_proj",
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
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
                cache=cache,
            )
        )
        self.layer_count += 1
        print_info("[auto scale] end")
        return scales_list

    def _get_out(self, layer_name, act, block, cache):
        if "att" in layer_name:
            return block(act, **cache)[0].squeeze(1)
        else:
            return block(act)[0].squeeze(1)

    def lepto_qdq_fp8_tensor(self, tensor, ratio):
        assert len(tensor.shape) == 1, f"tensor.device:{tensor.device}"
        w_scale = tensor.abs().max() / get_fp_maxval(bits=8)

        orig_fp8w, _ = quantize_weight_per_tensor_fp8(tensor, w_scale)

        outlier_point = 0.999 + 0.00005 * ratio
        n = min(round(len(tensor) * outlier_point), len(tensor) - 1)  # 0.001%0.001
        sorted_indices = torch.argsort(tensor.abs())
        closest_indices = sorted_indices[n]

        cut_np_fp8w1 = orig_fp8w[closest_indices].float().abs()

        adapt_scale = tensor.abs().max() / cut_np_fp8w1.type(tensor.dtype)
        print_info(
            f"w_scale:{w_scale.item()}, adapt_scale:{adapt_scale.item()},"
            f" cut_np_fp8w1: {cut_np_fp8w1.item()}"
        )
        return adapt_scale.to(tensor.dtype)

    def lepto_qdq_fp8_tensor_v2(self, tensor, ratio):
        assert len(tensor.shape) == 1, f"tensor.device:{tensor.device}"
        w_scale = tensor.abs().max() / get_fp_maxval(bits=8)

        orig_fp8w, _ = quantize_weight_per_tensor_fp8(tensor, w_scale)

        outlier_point = 0.999
        n = min(round(len(tensor) * outlier_point), len(tensor) - 1)  # 0.001%0.001
        sorted_indices = torch.argsort(tensor.abs())
        closest_indices = sorted_indices[n]

        cut_np_fp8w1 = max(
            orig_fp8w[closest_indices].float().abs(), get_fp_maxval(bits=8) / 7
        )

        step = (get_fp_maxval(bits=8) - cut_np_fp8w1) / self.search_step
        break_point = min(cut_np_fp8w1 + (step * (ratio + 1)), get_fp_maxval(bits=8))

        # FP8-list
        # r_list = [22, 30, 44, 60, 88, 120, 176, 224, 288, 320, 352, 384, 416, 448]
        # break_point = torch.tensor(r_list[ratio])

        adapt_scale = tensor.abs().max() / break_point.type(tensor.dtype)
        print_info(
            f"{w_scale.item()}, {adapt_scale.item()}, "
            f"{tensor.abs().max().item()}, "
            f"{break_point.type(tensor.dtype).item()}"
        )
        return adapt_scale.to(tensor.dtype)

    def lepto_input_hook(self, module, input, scale):
        modified_input = tensor_quant_dequant_fp8(input[0], scale)
        new_input = [modified_input]
        for i in range(len(input) - 1):
            new_input.append(input[1 + i])
            exit()
        return tuple(new_input)

    def search_by_block(
        self,
        layer_name,
        act_input,
        act_abs_max,
        layers,
        block,
        cache,
    ):
        print_info(f"inp.shape:{act_input.shape}")
        print_info(f"block:{block}")
        print_info(f"act_abs_max.shape:{act_abs_max.shape}")
        act = act_input
        print_info("[lepto search] search input of %s" % layer_name)
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        with torch.no_grad():
            if cache is not None:
                origin_out = torch.ones_like(act)
                new_out = torch.ones_like(act)
            else:
                origin_out = torch.ones(
                    (act.shape[0], act.shape[1], layers[0].weight.shape[0]),
                    dtype=act.dtype,
                    device=act.device,
                )
                new_out = torch.ones(
                    (act.shape[0], act.shape[1], layers[0].weight.shape[0]),
                    dtype=act.dtype,
                    device=act.device,
                )

            for j in range(act.shape[0]):
                origin_out[j, :, :] = self._get_out(
                    layer_name, act[j, :, :].unsqueeze(0), block, cache
                )
            print_info(f"origin_out.shape:{origin_out.shape}")
            org_w = []
            for layer in layers:
                org_w.append(layer.weight.clone().cpu())

            for ratio in range(8, 21):
                adapt_scale = self.lepto_qdq_fp8_tensor(
                    act.unsqueeze(0).view(-1), ratio
                ).unsqueeze(0)
                handles = []
                for layer in layers:
                    handles.append(
                        layer.register_forward_pre_hook(
                            functools.partial(self.lepto_input_hook, scale=adapt_scale)
                        )
                    )

                for j in range(act.shape[0]):
                    new_act = act[j, :, :].unsqueeze(0)
                    new_out[j, :, :] = self._get_out(layer_name, new_act, block, cache)

                loss = self.loss_function(origin_out, new_out).to(torch.float32)
                print_info(
                    "ratio: {}, adscale: {}, loss: {}".format(ratio, adapt_scale, loss)
                )
                if loss < best_error:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = adapt_scale

                for layer, w in zip(layers, org_w):
                    layer.weight.data.copy_(w)

                for h in handles:
                    h.remove()

        origin_out = origin_out.detach().cpu()
        new_out = w.detach().cpu()
        del origin_out
        del new_out
        for w in org_w:
            w = w.detach().cpu()
            del w

        if best_scales is None:
            best_scales = torch.ones(adapt_scale.shape, dtype=act.dtype)
            print_info("Cannot find better ratio.")
        else:
            print_info(
                "Best ratio :{}, minimal loss : {}, best_scales:{}.".format(
                    best_ratio, best_error, best_scales
                )
            )
        return best_scales.detach().cpu()
