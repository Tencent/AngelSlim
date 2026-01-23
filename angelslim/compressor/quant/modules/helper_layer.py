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
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from ....utils import get_best_device
from ..core import (
    QuantConfig,
    dequantize_gemm,
    fake_quant_dequant,
    gemm_fp8,
    pack_weight_to_int8,
    quantize_activation_per_tensor_fp8,
    quantize_weight_int,
    quantize_weight_per_tensor_fp8,
    reduce_block_padding,
    tensor_quant_dequant_fp8,
    tensor_quant_dequant_int,
)


def flush():
    gc.collect()
    torch.cuda.empty_cache()


# Adapted from https://github.com/compressa-ai/AutoAWQ/tree/dev
class WQLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
    ):
        # The forward pass can use ctx.
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # global user_has_been_warned
        # if not user_has_been_warned:
        #     warnings.warn("Using naive (slow) implementation." + msg)
        #     user_has_been_warned = True
        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        # always want 3D tensor if tensor is 2D
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out


class WQLinearGEMM(nn.Module):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev, training=False
    ):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)

        best_device = get_best_device()

        if "mps" in best_device:
            intweight = intweight.to("cpu")

        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device,
        )

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32, device=best_device)

        if "mps" in best_device:
            zeros = zeros.to("cpu")

        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        return awq_linear

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        with torch.no_grad():
            out = WQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )


class GPTQQuantLinear(nn.Module):
    QUANT_TYPE = "cuda"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        weight_dtype=torch.float16,
    ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor(
                [i // self.group_size for i in range(infeatures)], dtype=torch.int32
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(
                list(range(0, 32, self.bits)), dtype=torch.int32
            ).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, g_idx=None):
        w = linear.weight.data.clone()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (w[:, idx] + scale_zeros[self.g_idx[idx]])
                    / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32
        )
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)

        if self.bits in [2, 4, 8]:
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                self.wf.unsqueeze(0),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        elif self.bits == 3:
            zeros = self.qzeros.reshape(
                self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1
            ).expand(-1, -1, -1, 12)
            zeros = zeros >> self.wf.unsqueeze(0)
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
                (zeros[:, :, 1, 0] << 2) & 0x4
            )
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
                (zeros[:, :, 2, 0] << 1) & 0x6
            )
            zeros = zeros & 0x7
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            )

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = self.qweight.reshape(
                self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
            ).expand(-1, -1, 12, -1)
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat(
                [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
            )
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (
                weight - zeros[self.g_idx.long()]
            )
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim]
                weights.append(
                    scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()])
                )
            weights = torch.cat(weights, dim=1)
        out = torch.matmul(x, weights)
        out = out.to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


class SmoothHelpModule(nn.Module):
    def __init__(self, layer):
        super(SmoothHelpModule, self).__init__()
        self.weight = layer.weight
        self.weight.all_gather()
        self.layer = layer
        module_shape = self.weight.shape[-1]
        smooth_data = torch.ones(module_shape, dtype=self.weight.dtype).to(
            self.weight.device
        )
        self.smooth_weight = nn.Parameter(smooth_data)
        self.register_parameter("smooth_weight", self.smooth_weight)

    def forward(self, input):
        new_input = input
        # multiply
        smooth_input = torch.mul(new_input, self.smooth_weight)
        return self.layer(smooth_input)

    def convert_weight(self, smooth_weight):
        self.smooth_weight.data.copy_(1 / smooth_weight.squeeze())
        self.weight.all_gather()
        self.weight.data.copy_(self.weight * smooth_weight)
        self.weight.partition(has_been_updated=True)


class QDQSingleModule(nn.Module):
    def __init__(self, layer, act_scales, weight_scales, quant_algo="int8"):
        super(QDQSingleModule, self).__init__()
        assert act_scales is not None and weight_scales is not None
        self.layer = layer
        self.weight = layer.weight
        self.act_scales = act_scales
        self.weight_scales = weight_scales
        self.quant_algo = quant_algo
        for param_name, params in layer.named_parameters():
            if "weight" in param_name:
                if self.quant_algo == "int8":
                    qdq_weight = tensor_quant_dequant_int(
                        params, self.weight_scales, bits=8
                    )
                elif self.quant_algo == "fp8":
                    qdq_weight = tensor_quant_dequant_fp8(
                        params, self.weight_scales, bits=8
                    )
                params.data.copy_(qdq_weight)

    def forward(self, input):
        new_input = input
        if self.quant_algo == "int8":
            qdq_inp = tensor_quant_dequant_int(new_input, self.act_scales)
        elif self.quant_algo == "fp8":
            qdq_inp = tensor_quant_dequant_fp8(new_input, self.act_scales)
        return self.layer(qdq_inp)


class QDQModule(torch.nn.Module):
    def __init__(
        self,
        quant_algo: QuantConfig,
        weight: torch.nn.Parameter,
        weight_scale: torch.nn.Parameter,
        bias: torch.nn.Parameter,
        group_size: int = 128,
        input_scale: Optional[torch.nn.Parameter] = None,
        output_scale: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()
        self.quant_algo = quant_algo
        weight_scale = weight_scale.to(weight.device)
        if "fp8" in quant_algo:
            if "w4a8" in self.quant_algo:
                max_value_group_wise = weight_scale.clone()
                tensor_wise_scale = max_value_group_wise.max() / 448.0
                quant_weight, _ = quantize_weight_per_tensor_fp8(
                    weight, tensor_wise_scale
                )
                new_weight_bf16 = quant_weight.to(torch.bfloat16) * tensor_wise_scale

                new_weight_bf16_qdq = fake_quant_dequant(
                    new_weight_bf16, method="groupwise", bits=4, group_size=group_size
                )
                quant_weight, _ = quantize_weight_int(
                    new_weight_bf16_qdq, max_value_group_wise, bits=4
                )
                quant_weight = pack_weight_to_int8(quant_weight)
                del new_weight_bf16_qdq, new_weight_bf16
                self.weight_scale_int4 = torch.nn.Parameter(
                    max_value_group_wise / 8, requires_grad=False
                )
                weight_scale = tensor_wise_scale
            else:
                quant_weight, weight_scale = quantize_weight_per_tensor_fp8(
                    weight, weight_scale
                )
        elif "int8" in self.quant_algo:
            quant_weight, weight_scale = quantize_weight_int(
                weight, weight_scale, bits=8
            )
            quant_weight = quant_weight.to(torch.int8)
        else:
            raise ValueError(f"Unsupported quantization algorithm: {self.quant_algo}")

        if "w4a8" in self.quant_algo:
            self.qweight = torch.nn.Parameter(quant_weight, requires_grad=False)
        else:
            self.weight = torch.nn.Parameter(quant_weight, requires_grad=False)
        weight_scale = weight_scale.view(-1) if weight_scale.ndim == 0 else weight_scale
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.output_scale = output_scale
        if input_scale is not None:
            input_scale = input_scale.view(-1) if input_scale.ndim == 0 else input_scale
            self.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)
        else:
            self.input_scale = None
        if self.output_scale:
            self.output_scale = torch.nn.Parameter(
                self.output_scale, requires_grad=False
            )

    def forward(self, x):
        if self.input_scale:
            if "fp8" in self.quant_algo:
                qinput = quantize_activation_per_tensor_fp8(x, self.input_scale)
            elif "int8" in self.quant_algo:
                qinput = tensor_quant_dequant_int(x, self.input_scale, bits=8)
            else:
                raise ValueError(
                    f"Unsupported quantization algorithm: {self.quant_algo}"
                )

        if "fp8" in self.quant_algo:
            output = gemm_fp8(
                act=qinput,
                act_scale=self.input_scale,
                weight=self.weight,
                weight_scale=self.weight_scale,
                bias=self.bias,
                out_dtype=x.dtype,
            )
        elif "int8" in self.quant_algo:
            output = torch.nn.functional.linear(
                x, self.weight * self.weight_scale, bias=self.bias
            )
        else:
            raise ValueError(f"Unsupported quantization algorithm: {self.quant_algo}")

        if self.output_scale:
            if "fp8" in self.quant_algo:
                qoutput = quantize_activation_per_tensor_fp8(output, self.output_scale)
                output = qoutput.to(output.dtype) * self.output_scale
        return output

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        keys_to_rename = [k for k in state_dict.keys() if "weight_scale_int4" in k]
        for old_key in keys_to_rename:
            new_key = old_key.replace("weight_scale_int4", "weight_scale.int4")
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict


class QLinear(torch.nn.Module):
    def __init__(
        self,
        quant_algo: QuantConfig,
        weight: torch.nn.Parameter,
        weight_scale: torch.nn.Parameter,
        bias: torch.nn.Parameter,
        input_scale: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()
        self.quant_algo = quant_algo
        self.weight = weight

        self.weight_scale = (
            weight_scale.view(-1) if weight_scale.ndim == 0 else weight_scale
        )
        self.bias = bias
        if input_scale is not None:
            self.input_scale = (
                input_scale.view(-1) if input_scale.ndim == 0 else input_scale
            )
        else:
            self.input_scale = None

    def forward(self, x):
        if self.input_scale:
            if "fp8" in self.quant_algo:
                qinput = quantize_activation_per_tensor_fp8(x, self.input_scale)
            elif "int8" in self.quant_algo:
                qinput = tensor_quant_dequant_int(x, self.input_scale, bits=8)
            else:
                raise ValueError(
                    f"Unsupported quantization algorithm: {self.quant_algo}"
                )

        if "fp8" in self.quant_algo:
            output = gemm_fp8(
                act=qinput,
                act_scale=self.input_scale,
                weight=self.weight,
                weight_scale=self.weight_scale,
                bias=self.bias,
                out_dtype=x.dtype,
            )
        elif "int8" in self.quant_algo:
            output = torch.nn.functional.linear(
                x, self.weight * self.weight_scale, bias=self.bias
            )
        else:
            raise ValueError(f"Unsupported quantization algorithm: {self.quant_algo}")

        return output


class NVFP4QDQModule(torch.nn.Module):
    def __init__(
        self,
        weight: torch.nn.Parameter,
        weight_scale: torch.nn.Parameter,
        weight_scale_2: torch.nn.Parameter,
        bias: torch.nn.Parameter,
        block_size: int = 16,
        input_scale: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()
        # Define conversion tables
        self.e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
        self.e2m1_values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        self.e2m1_values_on_device = {}
        self.shape = weight.shape
        self.dtype = weight.dtype
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.weight_scale_2 = torch.nn.Parameter(weight_scale_2, requires_grad=False)
        self.bias = bias
        self.block_size = block_size
        if input_scale is not None:
            self.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)
        else:
            self.input_scale = None

        quant_weight = self.to_quantized_weight(
            weight,
            weight_scale,
            weight_scale_2,
            block_size,
        )
        self.weight = torch.nn.Parameter(quant_weight, requires_grad=False)

    def get_e2m1_values(self, device):
        """Returns the e2m1 values on the device."""
        if device not in self.e2m1_values_on_device:
            self.e2m1_values_on_device[device] = self.e2m1_values.to(device)
        return self.e2m1_values_on_device[device]

    def _cast_fp4(self, weight: torch.Tensor):
        """Converts tensor to uint4."""
        # Get device
        device = weight.device

        # Define mask to perform rounding
        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
        mask_shape = list(weight.shape)
        mask = mask.expand([*mask_shape, 7])

        sign_bit = (weight < 0).to(torch.uint8)

        weight_abs = weight.abs_()
        # Calculate the ordinal value based on the bounds
        ord = torch.searchsorted(
            self.e2m1_bounds.to(device), weight_abs, out_int32=True
        ).to(torch.uint8)
        # All values equal to e2m1_bounds at odd indices are rounded up
        # and even indices are rounded down
        round = torch.any(
            (weight_abs.unsqueeze(-1) == self.e2m1_bounds.to(device)) * mask, dim=-1
        )
        fp4_val = (sign_bit * 0b1000 + ord + round).to(torch.uint8)
        return fp4_val

    def quantize(
        self,
        weight: torch.Tensor,
        block_size: int,
        weights_scaling_factor: torch.Tensor | None = None,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ):
        """Converting a tensor to a quantized format based on NVFP4 quantization.

        Args:
            weight (torch.Tensor): The weight tensor to be quantized.
            block_size (int): The size of each block for quantization.
            weights_scaling_factor (torch.Tensor): The scaling factor for the weights.
            weights_scaling_factor_2 (torch.Tensor): The scaling factor for the weights.
            keep_high_precision (bool): Whether to keep output scales at high precision.

        Returns:
        Quantized data.
        """
        # pad the weight if needed
        weight = reduce_block_padding(weight, block_sizes={-1: block_size})

        # Reshape the weight and scale factors
        weight = weight.view((*tuple(weight.shape[:-1]), -1, block_size))

        # Scale weights
        scaled_weight = weight / (
            (
                weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2
            ).unsqueeze(-1)
        )

        # Reshape weights to original
        scaled_weight = scaled_weight.view((*tuple(scaled_weight.shape[:-2]), -1))

        if keep_high_precision:
            return scaled_weight
        # Cast weights to fp4
        q_weight = self._cast_fp4(scaled_weight)
        # Pack weights
        packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]

        return packed_weight

    def to_quantized_weight(
        self,
        weight: torch.Tensor,
        weights_scaling_factor: torch.Tensor,
        weights_scaling_factor2: torch.Tensor | None = None,
        block_size: int | None = None,
    ):
        """Converts the weight to the quantized (packed) format."""
        if weights_scaling_factor is not None:
            weights_scaling_factor = weights_scaling_factor.to(weight.device)

        if weights_scaling_factor2 is not None:
            weights_scaling_factor2 = weights_scaling_factor2.to(weight.device)

        assert (
            block_size is not None
        ), "Block size not passed. Unable to quantize to NVFP4 format."
        assert (
            weights_scaling_factor2 is not None
        ), "Weights scaling factor 2 not passed. Unable to quantize to NVFP4 format"
        # If MoE reshape weights_scaling_factor2 to enable quantize operations
        return self.quantize(
            weight,
            block_size,
            weights_scaling_factor,
            (
                weights_scaling_factor2.view(-1, 1, 1)
                if weights_scaling_factor2.dim() != 0
                else weights_scaling_factor2
            ),
        )

    def get_input_scaling_factor(
        self,
        inputs: torch.Tensor,
        block_size: int,
        inputs_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ):
        """Returns quantized per block input scaling factor."""
        # Get per_block amax
        [n, k] = inputs.shape[-2:]
        assert (
            block_size != 0
        ), "Block size is zero. Cannot return per_block amax for given input."

        assert (
            k % block_size == 0
        ), "input shape is not divisible for block size for block quantiation."

        inputs = inputs.reshape(
            (*tuple(inputs.shape[:-2]), n, k // block_size, block_size)
        )
        # Get per block amax
        per_block_amax = inputs.abs().amax(dim=-1).float()
        # Get per-block-scale
        per_block_scale = per_block_amax / 6.0
        # Quantize per_block_scale to FP8
        q_per_block_scale = per_block_scale / inputs_scaling_factor_2
        # Set all zero values in scale to 1.0
        q_per_block_scale[per_block_scale == 0] = 1.0
        # Convert to torch.float8_e4m3fn
        if not keep_high_precision:
            finfo = torch.finfo(torch.float8_e4m3fn)
            q_per_block_scale = q_per_block_scale.clamp(min=finfo.min, max=finfo.max)
            q_per_block_scale = q_per_block_scale.to(torch.float8_e4m3fn)
        return q_per_block_scale

    def quantize_input(
        self,
        inputs: torch.Tensor,
        block_size: int,
        inputs_scaling_factor: torch.Tensor | None = None,
        inputs_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ):
        """Converting a tensor to a quantized format based on NVFP4 quantization.

        Args:
            weight (torch.Tensor): The weight tensor to be quantized.
            block_size (int): The size of each block for quantization.
            weights_scaling_factor (torch.Tensor): The scaling factor for the weights.
            weights_scaling_factor_2 (torch.Tensor): The scaling factor for the weights.
            keep_high_precision (bool): Whether to keep output scales at high precision.

        Returns:
        tuple: Contains quantized data, quantized per block scaling factor,
        and per tensor scaling factor.
        """
        # pad the weight if needed
        inputs = reduce_block_padding(inputs, block_sizes={-1: block_size})

        # Reshape the weight and scale factors
        inputs = inputs.view((*tuple(inputs.shape[:-1]), -1, block_size))

        # Scale weights
        scaled_inputs = inputs / (
            (
                inputs_scaling_factor.to(torch.float32) * inputs_scaling_factor_2
            ).unsqueeze(-1)
        )

        # Reshape weights to original
        scaled_inputs = scaled_inputs.view((*tuple(scaled_inputs.shape[:-2]), -1))

        if keep_high_precision:
            return scaled_inputs
        # Cast weights to fp4
        cast_inputs = self._cast_fp4(scaled_inputs)
        qinputs = self.get_e2m1_values(cast_inputs.device)[cast_inputs.long()]

        return qinputs

    def forward(self, x):
        qdqweight = self.dequantize(
            self.weight, self.block_size, self.weight_scale, self.weight_scale_2
        )

        if self.input_scale is None:
            input_amax = x.abs().amax()
            input_scale_2 = input_amax.float() / 6.0 / 448.0
        else:
            input_scale_2 = self.input_scale

        input_scale = self.get_input_scaling_factor(
            inputs=x.detach(),
            inputs_scaling_factor_2=input_scale_2,
            block_size=self.block_size,
        )

        qinput = self.quantize_input(
            x,
            self.block_size,
            input_scale,
            input_scale_2.view(-1, 1, 1) if input_scale_2.dim() != 0 else input_scale_2,
        )

        qdqinput = qinput.view(
            qinput.shape[0], qinput.shape[1], qinput.shape[2] // self.block_size, -1
        ) * (input_scale.to(torch.float32) * input_scale_2).unsqueeze(-1)
        qdqinput = qdqinput.view(-1)[: np.prod(x.shape)].reshape(x.shape).to(x.dtype)

        output = torch.nn.functional.linear(
            qdqinput.to(self.dtype),
            qdqweight,
            bias=self.bias,
        )

        return output

    def dequantize(
        self,
        weight: torch.Tensor,
        block_size: int,
        weights_scaling_factor: torch.Tensor | None = None,
        weights_scaling_factor_2: torch.Tensor | None = None,
    ):
        """Dequantze NVFP4 packed tensor to a target dtype."""
        dtype = self.dtype

        def _unpack_tensor(input: torch.Tensor):
            # Initalize storage for unpacked tensor
            unpacked = torch.empty(
                [input.shape[0], input.shape[1] * 2], dtype=dtype, device=input.device
            )
            unpacked_shape = unpacked.shape

            unpacked[..., 1::2] = input >> 4
            unpacked[..., 0::2] = input & 0x0F

            unpacked = unpacked.reshape(-1)
            unpacked = self.get_e2m1_values(input.device)[unpacked.long()]

            return unpacked.reshape(unpacked_shape)

        q_per_block_scale = weights_scaling_factor.to(torch.float32)
        per_block_quant_scale = weights_scaling_factor_2

        # Dequantize scales
        per_block_scale = q_per_block_scale * per_block_quant_scale

        # Unpack and unscale weights
        deq_data = _unpack_tensor(weight)

        deq_data = deq_data.view(
            deq_data.shape[0], deq_data.shape[1] // block_size, -1
        ) * per_block_scale.unsqueeze(-1)
        return deq_data.view(-1)[: np.prod(self.shape)].reshape(self.shape).to(dtype)


class MoEQDQModule(torch.nn.Module):
    def __init__(
        self,
        gate_proj: torch.nn.Parameter,
        up_proj: torch.nn.Parameter,
        down_proj: torch.nn.Parameter,
        gate_proj_weight_scale: torch.nn.Parameter,
        up_proj_weight_scale: torch.nn.Parameter,
        down_proj_weight_scale: torch.nn.Parameter,
        gate_up_proj_input_scale: torch.nn.Parameter,
        down_proj_input_scale: torch.nn.Parameter,
    ):
        super().__init__()
        quant_gate_weight, _ = quantize_weight_per_tensor_fp8(
            gate_proj, gate_proj_weight_scale
        )
        quant_up_weight, _ = quantize_weight_per_tensor_fp8(
            up_proj, up_proj_weight_scale
        )
        quant_down_weight, _ = quantize_weight_per_tensor_fp8(
            down_proj, down_proj_weight_scale
        )
        quant_gate_up_weight = torch.cat([quant_gate_weight, quant_up_weight], dim=-1)

        self.gate_up_proj = torch.nn.Parameter(
            quant_gate_up_weight, requires_grad=False
        )
        self.down_proj = torch.nn.Parameter(quant_down_weight, requires_grad=False)

        gate_proj_weight_scale = (
            gate_proj_weight_scale.view(-1)
            if gate_proj_weight_scale.ndim == 0
            else gate_proj_weight_scale
        )
        up_proj_weight_scale = (
            up_proj_weight_scale.view(-1)
            if up_proj_weight_scale.ndim == 0
            else up_proj_weight_scale
        )
        down_proj_weight_scale = (
            down_proj_weight_scale.view(-1)
            if down_proj_weight_scale.ndim == 0
            else down_proj_weight_scale
        )
        gate_up_proj_weight_scale = torch.cat(
            [gate_proj_weight_scale, up_proj_weight_scale], dim=-1
        )

        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            gate_up_proj_weight_scale, requires_grad=False
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            down_proj_weight_scale, requires_grad=False
        )

        down_proj_input_scale = (
            down_proj_input_scale.view(-1)
            if down_proj_input_scale.ndim == 0
            else down_proj_input_scale.squeeze()
        )
        gate_up_proj_input_scale = (
            gate_up_proj_input_scale.view(-1)
            if gate_up_proj_input_scale.ndim == 0
            else gate_up_proj_input_scale.squeeze()
        )

        self.gate_up_proj_input_scale = torch.nn.Parameter(
            gate_up_proj_input_scale, requires_grad=False
        )
        self.down_proj_input_scale = torch.nn.Parameter(
            down_proj_input_scale, requires_grad=False
        )

    def forward(self, x):
        pass


class W4A8Int8QuantLinear(nn.Module):
    """
    W4A8 per-channel symmetric quantized Linear layer for compressed-tensors format.
    Weights are quantized to 4-bit and packed into int8 tensors.
    """

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        weight_dtype=torch.float16,
    ):
        super().__init__()
        if bits != 4:
            raise NotImplementedError("Only 4-bit weights are supported for W4A8Int8.")
        if group_size != -1:
            raise RuntimeError(
                "Only per-channel quantization are supported for W4A8Int8."
            )

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        # Register buffers for packed weight and scale only
        self.register_buffer(
            "weight_packed",
            torch.zeros((outfeatures, infeatures // 2), dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros((outfeatures, 1), dtype=torch.float32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None

    @torch.no_grad()
    def pack(self, linear, scales, zeros, g_idx=None):
        """
        Per-output-channel symmetric quantization.
        """
        w = linear.weight  # [out, in]

        # bias
        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias)

        # ---- normalize shapes ----
        # scales: [out, 1] -> [out]
        if scales.dim() == 2:
            scales = scales.squeeze(1)
        # zeros: [out, 1] or scalar
        if zeros.dim() == 2:
            zeros = zeros.squeeze(1)

        assert (
            scales.shape[0] == w.shape[0]
        ), f"scale shape mismatch: {scales.shape} vs {w.shape}"

        # store scale
        self.weight_scale.copy_(scales.float().unsqueeze(1))  # [out, 1]

        # ---- quantize ----
        q = torch.round(w / scales[:, None])
        q = torch.clamp(q, -8, 7).to(torch.int8)

        # ---- pack ----
        # q must be [out, in]
        pack_w = self._pack_to_int8(q)

        assert (
            pack_w.shape == self.weight_packed.shape
        ), f"packed shape mismatch: {pack_w.shape} vs {self.weight_packed.shape}"

        self.weight_packed.copy_(pack_w)
        linear.weight = None
        gc.collect()

    def _pack_to_int8(self, t: torch.Tensor) -> torch.Tensor:
        """
        Pack int4 tensor to int8 by storing two int4 values in one int8.

        Args:
            t: int8 tensor with values in range [-8, 7],
              shape [out_features, in_features]

        Returns:
            Packed int8 tensor with shape [out_features, in_features // 2]
        """
        assert t.dtype == torch.int8
        assert t.shape[-1] % 2 == 0, "Last dimension must be even for packing"

        # Flatten and pack
        # Take two adjacent int4 values and pack them into one int8
        # Low nibble: first value, High nibble: second value

        out_features = t.shape[0]
        in_features = t.shape[1]

        # Reshape to separate pairs
        t_reshaped = t.view(out_features, in_features // 2, 2)

        # Extract low and high nibbles
        # Low nibble (first int4)
        low = t_reshaped[:, :, 0] & 0x0F
        # High nibble (second int4)
        high = t_reshaped[:, :, 1] & 0x0F

        # Pack: (high << 4) | low
        packed = (high << 4) | low

        return packed.view(out_features, in_features // 2)

    def forward(self, x):
        # For compressed-tensors format, forward is handled by the inference engine
        # This is mainly for compatibility
        raise NotImplementedError(
            "W4A8Int8QuantLinear is for export only. "
            "Use compressed-tensors runtime for inference."
        )
