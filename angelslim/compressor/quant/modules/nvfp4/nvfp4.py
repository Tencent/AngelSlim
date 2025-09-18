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

from .....utils import print_info

__all__ = ["NVFP4"]


class NVFP4:
    def __init__(
        self,
        model,
    ):
        """
        Args:
            model(nn.Module, required): The model to be quanted.
        """
        super(NVFP4, self).__init__()
        self.model = model
        self.block_size = self.model.quant_config.quant_algo_info["block_size"]

    @torch.no_grad()
    def run(self, dataloader):
        print_info("Use NVFP4 fast forward")
        self.model.model_forward(dataloader)

    def get_activation_scaling_factor(self, input_observer_amax):
        """Returns the activation scaling factor for export."""
        activation_scaling_factor = input_observer_amax.float() / 6.0 / 448.0

        assert torch.all(
            activation_scaling_factor > 0
        ), f" activation scaling factor {activation_scaling_factor} not positive."

        return activation_scaling_factor

    def get_weights_scaling_factor_2(self, weight_observer_amax):
        """Returns per tensor weight scaling factor."""
        return weight_observer_amax.float() / 6.0 / 448.0

    def get_weights_scaling_factor(
        self,
        weight: torch.Tensor,
        block_size: int,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ):
        """Returns quantized per block weight scaling factor."""
        # Get per_block amax
        [n, k] = weight.shape[-2:]
        assert (
            block_size != 0
        ), "Block size is zero. Cannot return per_block amax for given weight."

        assert (
            k % block_size == 0
        ), "Weight shape is not divisible for block size for block quantiation."

        weight = weight.reshape(
            (*tuple(weight.shape[:-2]), n, k // block_size, block_size)
        )
        # Get per block amax
        per_block_amax = weight.abs().amax(dim=-1).float()
        # Get per-block-scale
        per_block_scale = per_block_amax / 6.0
        # Quantize per_block_scale to FP8
        q_per_block_scale = per_block_scale / weights_scaling_factor_2
        # Set all zero values in scale to 1.0
        q_per_block_scale[per_block_scale == 0] = 1.0
        # Convert to torch.float8_e4m3fn
        if not keep_high_precision:
            finfo = torch.finfo(torch.float8_e4m3fn)
            q_per_block_scale = q_per_block_scale.clamp(min=finfo.min, max=finfo.max)
            q_per_block_scale = q_per_block_scale.to(torch.float8_e4m3fn)
        return q_per_block_scale

    def post_process(self, sub_layer, name):
        # TODO:Fuse observer amax because TRT-LLM requires the qkv,
        # gate and up to share the weight_scale2
        weight_observer_amax, input_observer_amax = self.model.fuse_observer_amax(
            sub_layer, name
        )

        weight_scale_2 = self.get_weights_scaling_factor_2(weight_observer_amax)
        self.model.weight_scales_dict_2[name] = weight_scale_2

        weight_scale = self.get_weights_scaling_factor(
            weight=sub_layer.weight.detach(),
            weights_scaling_factor_2=weight_scale_2,
            block_size=self.block_size,
        )
        self.model.weight_scales_dict[name] = weight_scale

        input_scale = self.get_activation_scaling_factor(input_observer_amax)
        self.model.act_scales_dict[name] = input_scale
