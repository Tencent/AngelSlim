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

import math
import time

import torch

from .....utils import get_tensor_item, print_info
from ...core import compute_scales_with_zero

__all__ = ["GPTQModule"]


class GPTQModule:
    def __init__(self, layer, quant_bits=4):
        """
        GPTQ quantization wrapper for neural network layers.

        Args:
            layer: Full-precision torch.nn.Module to quantize (Linear)
            quant_bits: Quantization bitwidth (2-8 bits, default=4)
        """
        super(GPTQModule, self).__init__()
        self.layer = layer
        self.dev = self.layer.weight.device
        self.w = layer.weight.data.clone()
        self.rows = self.w.shape[0]
        self.columns = self.w.shape[1]
        self.h = torch.zeros(
            (self.columns, self.columns), device=self.dev, dtype=torch.float32
        )
        self.nsamples = 0
        self.quant_bits = quant_bits

    def add_batch(self, inp, out):
        if len(inp.shape) == 4:
            inp = inp[0, 0, :, :]
        inp = inp.squeeze()
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.float()
        inp = inp.t()
        self.h *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.h += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=True,
        sym=True,
    ):
        w_weight = self.w.float()

        tick = time.time()

        hessian = self.h
        if torch.isnan(hessian).any():
            print_info("[error] Hessian contains nan!")
            exit()
        self.h.detach().cpu()
        del self.h
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        w_weight[:, dead] = 0

        g_idx = []
        scale = []
        zero = []
        now_idx = 1
        static_groups = True

        if static_groups:
            for i in range(0, self.columns, group_size):
                weight_scale, weight_zero = compute_scales_with_zero(
                    w_weight[:, i : (i + group_size)], bits=self.quant_bits, sym=sym
                )
                scale.append(weight_scale)
                zero.append(weight_zero)

        if actorder:
            perm = torch.argsort(torch.diag(hessian), descending=True)
            w_weight = w_weight[:, perm]
            hessian = hessian[perm][:, perm]
            invperm = torch.argsort(perm)

        losses = torch.zeros_like(w_weight)
        q_weight = torch.zeros_like(w_weight)

        while 1 > percdamp > 0:
            try:
                damp = percdamp * torch.mean(torch.diag(hessian))
                diag = torch.arange(self.columns, device=self.dev)
                hessian[diag, diag] += damp
                hessian = torch.linalg.cholesky(hessian)
                hessian = torch.cholesky_inverse(hessian)
                hessian = torch.linalg.cholesky(hessian, upper=True)
                hinv = hessian
                break
            except torch._C._LinAlgError as e:
                print_info(e)
                print_info(f"Cholesky failed with percdamp={percdamp:.5f}")
                percdamp += 0.01

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            w1 = w_weight[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            hinv1 = hinv[i1:i2, i1:i2]

            for i in range(count):
                w = w1[:, i]
                d = hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            weight_scale, weight_zero = compute_scales_with_zero(
                                w_weight[:, (i1 + i) : (i1 + i + group_size)],
                                bits=self.quant_bits,
                                sym=sym,
                            )

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(weight_scale)
                            zero.append(weight_zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        weight_scale = scale[idx // group_size]
                        weight_zero = zero[idx // group_size]

                maxq = torch.tensor(2**self.quant_bits - 1)
                q = torch.clamp(
                    torch.round(w.unsqueeze(1) / weight_scale) + weight_zero, 0, maxq
                )
                q = weight_scale * (q - weight_zero)
                q = q.flatten()
                q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d**2

                err = (w - q) / d
                w1[:, i:] -= err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                err1[:, i] = err

            q_weight[:, i1:i2] = q1
            losses[:, i1:i2] = losses1 / 2

            w_weight[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print_info(f" duration: {(time.time() - tick)}")

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=q_weight.device)
        if actorder:
            q_weight = q_weight[:, invperm]
            g_idx = g_idx[invperm]

        norm_loss = torch.norm(
            q_weight.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
            - self.layer.weight.data
        )

        all_norm_loss = [norm_loss]

        print_info(f" norm loss: {list(map(get_tensor_item, all_norm_loss))}")

        self.layer.weight.data.copy_(
            q_weight.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
        )

        if scale == []:
            scale = weight_scale
            zero = torch.zeros_like(weight_scale)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        losses = losses.cpu()
        q_weight = q_weight.cpu()
        w_weight = w_weight.cpu()
        hessian = hessian.cpu()
        hinv = hinv.cpu()
        del losses, q_weight, w_weight, hessian, hinv
        self.w = self.w.cpu()
        del self.w
        torch.cuda.empty_cache()
        return scale, zero, g_idx

    def free(self):
        self.h = None
        self.w = None
        self.losses = None
        torch.cuda.empty_cache()
