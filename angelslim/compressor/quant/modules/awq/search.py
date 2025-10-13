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
from torch.nn import Linear

from .....utils import get_best_device, print_info
from ...core import (
    mse_loss,
    per_block_weight_quant,
    pseudo_quantize_tensor,
    weight_dequant,
)

print_func = print_info


class AWQSearch:
    def __init__(
        self,
        n_grid=20,
        bits_length=4,
        weight_quant_method="groupwise",
        group_size=128,
        loss_function=mse_loss,
        merge_samples=True,
        observer_layer_classes=None,
        low_memory=False,
    ):
        """
        The implementation of AutoScale from AWQ(https://arxiv.org/pdf/2306.00978.pdf).
        """
        self.n_grid = n_grid
        self.bits_length = bits_length
        self.weight_quant_method = weight_quant_method
        self.bnt = (1 << (bits_length - 1)) - 1
        self.group_size = group_size
        self.loss_function = loss_function
        self.merge_samples = merge_samples
        self.observer_layer_classes = observer_layer_classes
        self.low_memory = low_memory

    def _get_out(self, layer_name, act, block, cache):
        if "qkv" in layer_name:
            return block(act, **cache)[0].squeeze(1)
        else:
            return block(act)[0].squeeze(1)

    # search by block
    @torch.no_grad()
    def search_by_block(
        self, layer_name, act_input, act_abs_max, layers, block, cache, layer_count
    ):
        act = act_input
        print_func("[awq search] act device: %s" % act.device)
        print_func("[awq search] search input of %s" % layer_name)
        best_error = float("inf")
        best_ratio = -1
        best_scales = None
        dev = get_best_device()
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
                    layer_name, act[j, :, :].unsqueeze(0).to(dev), block, cache
                ).to(act.device)

            org_w = []
            for layer in layers:
                org_w.append(layer.weight.clone().cpu())

            for ratio in range(self.n_grid):
                ratio = ratio * 1 / self.n_grid
                act_abs_max_tmp = act_abs_max.detach().clone().to(dev)
                scales = act_abs_max_tmp.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for layer in layers:
                    if layer.weight.dtype == torch.float8_e4m3fn:
                        weight = weight_dequant(layer.weight, layer.weight_scale_inv)
                        weight.mul_(scales.view(1, -1))
                        weight, _ = per_block_weight_quant(weight)
                        layer.weight.data.copy_(weight)
                    else:
                        layer.weight.mul_(scales.view(1, -1))
                    if type(layer) in [Linear]:
                        quant_dequant_weight = pseudo_quantize_tensor(
                            layer.weight,
                            w_bit=self.bits_length,
                            q_group_size=self.group_size,
                        )
                        layer.weight.data.copy_(quant_dequant_weight)

                for j in range(act.shape[0]):
                    new_act = act[j, :, :].unsqueeze(0).to(dev) / scales
                    new_out[j, :, :] = self._get_out(
                        layer_name, new_act, block, cache
                    ).to(act.device)

                loss = self.loss_function(origin_out, new_out).to(torch.float32)

                if loss < best_error:
                    print_func("find better ratio: {}, loss: {}".format(ratio, loss))
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales

                for layer, w in zip(layers, org_w):
                    layer.weight.data.copy_(w)

        origin_out = origin_out.detach().cpu()
        new_out = w.detach().cpu()
        del origin_out
        del new_out
        for w in org_w:
            w = w.detach().cpu()
            del w

        if best_scales is None:
            best_scales = torch.ones(scales.shape, dtype=act.dtype)
            print_func("Cannot find better ratio.")
        else:
            print_func(
                "Best ratio :{}, minimal loss : {}.".format(best_ratio, best_error)
            )
        return best_scales.detach().cpu()
