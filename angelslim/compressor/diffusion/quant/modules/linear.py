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

from ..quant_func import (
    fp8_gemm,
    fp8_per_block_quant,
    fp8_per_tensor_quant,
    fp8_per_token_group_quant,
    fp8_weight_only_gemm,
)


# modified from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
class FP8DynamicLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,
        quant_type: str = "fp8-per-tensor",
        block_size: int = 128,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support
        self.quant_type = quant_type
        self.block_size = block_size

    @torch.compiler.disable(recursive=True)
    def forward(self, x):
        ori_dtype = x.dtype
        assert ori_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "x.dtype must be float32, bfloat16, or float16"

        if ori_dtype == torch.float32:
            x = x.to(torch.bfloat16)

        if self.quant_type == "fp8-per-tensor":
            origin_shape = None
            qinput, x_scale = fp8_per_tensor_quant(x)
        elif self.quant_type == "fp8-per-token":
            origin_shape = None
            x_2d = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_token_group_quant(x_2d, x_2d.shape[-1])
        elif self.quant_type == "fp8-per-block" and self.native_fp8_support:
            origin_shape = x.shape
            x = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_token_group_quant(
                x, group_size=128, column_major_scales=True, scale_tma_aligned=True
            )
        elif self.quant_type == "fp8-per-block" and not self.native_fp8_support:
            origin_shape = None
            qinput, x_scale = fp8_per_block_quant(x, block_size=128)
        else:
            raise ValueError(f"Invalid quant_type: {self.quant_type}")

        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            native_fp8_support=self.native_fp8_support,
            quant_type=self.quant_type,
            origin_shape=origin_shape,
        )

        if self.quant_type == "fp8-per-token" and x.dim() == 3 and output.dim() == 2:
            output = output.unsqueeze(0)

        return output


class FP8WeightOnlyLinear(torch.nn.Module):
    """
    FP8 Weight-Only Quantized Linear Layer.

    This layer quantizes only the weights to FP8 while keeping activations
    in higher precision (bfloat16/float16). This provides a good balance
    between memory savings and accuracy.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,  # not used
        quant_type: str = "fp8-per-tensor-weight-only",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support  # not used
        self.quant_type = quant_type

    @torch.compiler.disable(recursive=True)
    def forward(self, x):
        ori_dtype = x.dtype
        assert ori_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "x.dtype must be float32, bfloat16, or float16"

        if ori_dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # For weight-only quantization, we don't quantize activations
        # Just use the original activations with quantized weights
        output = fp8_weight_only_gemm(
            A=x,  # Keep activations in original precision
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        return output
