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

import torch
import torch.nn as nn


def _round_ste(x):
    return (x.round() - x).detach() + x


def _clamp_ste(x, min_val, max_val):
    return (x.clamp(min_val, max_val) - x).detach() + x


def _normalize_granularity(value):
    return str(value).replace("-", "_")


def _reshape_by_granularity(weight, granularity, group_size):
    original_shape = weight.shape
    if len(original_shape) != 2:
        raise ValueError("Special weight quantization expects a 2D weight tensor.")
    if granularity == "per_tensor":
        return weight.reshape(1, -1), original_shape
    if granularity == "per_channel":
        return weight.reshape(original_shape[0], -1), original_shape
    if granularity == "per_group":
        if group_size <= 0 or original_shape[1] % group_size != 0:
            raise ValueError("per_group quantization requires a valid group_size.")
        return (
            weight.reshape(original_shape[0], original_shape[1] // group_size, group_size),
            original_shape,
        )
    raise ValueError(f"Unsupported special quantizer granularity: {granularity}")


def _signed_quant_bounds(bits):
    if bits <= 1:
        return -1, 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


class SherryNMQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, granularity, group_size, n, m):
        original_shape = input.shape
        if len(original_shape) != 2:
            raise ValueError("Sherry N:M quantization expects a 2D weight tensor.")
        if original_shape[1] % m != 0:
            raise ValueError(f"Input dimension {original_shape[1]} is not divisible by M={m}.")

        weight = input.reshape(original_shape[0], original_shape[1] // m, m)
        _, topk_indices = torch.topk(torch.abs(weight), n, dim=-1)
        mask = torch.zeros_like(weight, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        sparse_weight = (weight * mask).reshape(original_shape)

        x, _ = _reshape_by_granularity(sparse_weight, granularity, group_size)
        signed = torch.sign(x)
        denom = max(float(x.shape[-1]) / float(m) * float(n), 1.0)
        scale = x.abs().sum(dim=-1, keepdim=True) / denom
        return (signed * scale).reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class SpecialWeightQuantizer(nn.Module):
    def __init__(self, config, weight_shape=None, quant_method=None):
        super().__init__()
        special_cfg = config.get("special", {}) if isinstance(config, dict) else {}
        sherry_cfg = config.get("sherry", {}) if isinstance(config, dict) else {}
        method = quant_method or special_cfg.get("quant_method") or sherry_cfg.get("quant_method")
        self.quant_method = str(method or "sherry").lower()
        method_cfg = sherry_cfg if self.quant_method == "sherry" and sherry_cfg else special_cfg

        self.granularity = _normalize_granularity(
            method_cfg.get("granularity", config.get("granularity", "per_group"))
        )
        self.group_size = int(method_cfg.get("group_size", config.get("group_size", 128)))
        self.w_bits = int(method_cfg.get("w_bits", config.get("w_bits", 1)))
        self.n = int(method_cfg.get("N", method_cfg.get("n", config.get("N", 3))))
        self.m = int(method_cfg.get("M", method_cfg.get("m", config.get("M", 4))))
        self.warmup_step = int(method_cfg.get("warmup_step", config.get("warmup_step", 0)))
        self.total_steps = int(method_cfg.get("total_steps", config.get("total_steps", 0)))
        self.max_eps = float(method_cfg.get("max_eps", config.get("max_eps", 0.0)))
        self.steps = 0
        self.eps = 0.0

        scale_shape = self._scale_shape(weight_shape)
        scale = torch.ones(scale_shape, dtype=torch.float32)
        if self.quant_method in ("lsq", "seq", "dlt"):
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        if self.quant_method == "dlt":
            self.gamma = nn.Parameter(torch.zeros(scale_shape, dtype=torch.float32))
        else:
            self.gamma = None
        self.zero_point = None

    def _scale_shape(self, weight_shape):
        if self.granularity == "per_tensor" or weight_shape is None:
            return (1, 1)
        out_dim, in_dim = int(weight_shape[0]), int(weight_shape[1])
        if self.granularity == "per_channel":
            return (out_dim, 1)
        if self.granularity == "per_group":
            return (out_dim, max(in_dim // max(self.group_size, 1), 1), 1)
        raise ValueError(f"Unsupported special quantizer granularity: {self.granularity}")

    def _current_eps(self):
        if not self.training or self.max_eps <= 0.0 or self.total_steps <= 0:
            return 0.0
        if self.warmup_step > 0 and self.steps <= self.warmup_step:
            return self.max_eps * self.steps / self.warmup_step
        if self.steps < self.total_steps:
            denom = max(self.total_steps - self.warmup_step, 1)
            progress = (self.steps - self.warmup_step) / denom
            return self.max_eps * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.0

    def _reshape_param(self, param, x):
        if self.granularity == "per_tensor":
            return param.reshape(1, 1)
        if self.granularity == "per_channel":
            return param.reshape(x.shape[0], 1)
        if self.granularity == "per_group":
            return param.reshape(x.shape[0], x.shape[1], 1)
        raise ValueError(f"Unsupported special quantizer granularity: {self.granularity}")

    def _absmean_or_twn(self, weight):
        x, original_shape = _reshape_by_granularity(weight, self.granularity, self.group_size)
        mean_abs = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = mean_abs / 2 if self.quant_method == "absmean" else 0.75 * mean_abs
        mask_pos = x >= delta
        mask_neg = x <= -delta
        ternary = torch.zeros_like(x)
        ternary[mask_pos] = 1
        ternary[mask_neg] = -1
        if self.quant_method == "twn":
            count = ternary.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_abs = (x * ternary).sum(dim=-1, keepdim=True).abs() / count
        quant = (ternary * mean_abs).reshape(original_shape)
        return weight + (quant - weight).detach()

    def _lsq_like(self, weight):
        x, original_shape = _reshape_by_granularity(weight, self.granularity, self.group_size)
        scale = (
            self._reshape_param(self.scale, x).to(device=x.device, dtype=x.dtype).clamp_min(1e-6)
        )
        qmin, qmax = _signed_quant_bounds(self.w_bits)
        q = x / scale
        if self.quant_method == "seq":
            clip_val = 1.0 - 1e-2
            levels = 1.5 if self.w_bits == 0 else 2 ** max(self.w_bits - 1, 0)
            shift = 0.0 if self.w_bits == 0 else 0.5
            q = (_round_ste(_clamp_ste(q, -clip_val, clip_val) * levels - shift) + shift) / levels
        else:
            q = _clamp_ste(_round_ste(q), qmin, qmax)
        quant = q * scale
        if self.quant_method == "dlt":
            gamma = self._reshape_param(self.gamma, x).to(device=x.device, dtype=x.dtype)
            quant = quant + gamma
        return quant.reshape(original_shape)

    def forward(self, weight):
        if self.quant_method == "sherry":
            quant_weight = SherryNMQuant.apply(
                weight,
                self.granularity,
                self.group_size,
                self.n,
                self.m,
            )
            self.steps += 1
            self.eps = self._current_eps()
            if self.eps:
                quant_weight = quant_weight + self.eps * weight
            return quant_weight
        if self.quant_method in ("absmean", "twn"):
            return self._absmean_or_twn(weight)
        if self.quant_method in ("lsq", "seq", "dlt"):
            return self._lsq_like(weight)
        raise ValueError(f"Unsupported special quantization method: {self.quant_method}")


SherryWeightQuantizer = SpecialWeightQuantizer
