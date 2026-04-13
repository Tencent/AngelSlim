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
import torch.nn.functional as F

FP8_E4M3_QMIN = -448
FP8_E4M3_QMAX = 448


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min_val, max_val):
    return (x.clamp(min_val, max_val) - x).detach() + x


def fp8_cast_ste(x: torch.Tensor):
    """Simulate FP8 E4M3 cast with STE for gradient pass-through."""
    x_fp8 = x.to(torch.float8_e4m3fn)
    return (x_fp8.to(x.dtype) - x).detach() + x


def _parse_bits_and_dtype(qtype_str):
    match = re.search(r"\d+", qtype_str)
    if match is None:
        raise ValueError(f"Cannot parse bit-width from: {qtype_str}")
    bits = int(match.group())
    if "fp8" in qtype_str:
        return bits, "fp8"
    elif "int" in qtype_str:
        return bits, "int"
    raise ValueError(f"Unsupported dtype in: {qtype_str}")


class Quantizer(nn.Module):
    def __init__(self, config, quant_info, x=None, is_act=False, resume=False, num_heads=-1):
        super().__init__()
        self.is_act = is_act
        self.num_heads = num_heads
        info = quant_info.quant_algo_info["w"]
        self.group_size = quant_info.quant_algo_info.get("w_group_size", -1)
        rewrite_conf = config.get("weight", {})

        self.is_w4a8_fp8 = (
            not self.is_act and not rewrite_conf and "w4a8_fp8" in quant_info.quant_algo
        )

        if self.is_act:
            info = quant_info.quant_algo_info["a"]
            rewrite_conf = config.get("activation", {})
            self.resume = resume

        self._apply_settings(info, rewrite_conf)
        self._set_quant_range()
        self._init_quant_params(x)

    def _apply_settings(self, info, rewrite_conf):
        if rewrite_conf:
            self.bits, self.dtype = _parse_bits_and_dtype(rewrite_conf["qtype"])
            self.granularity = rewrite_conf["granularity"]
            self.group_size = rewrite_conf.get("group_size", -1)
            self.is_sym = rewrite_conf.get("is_sym", True)
            self.dynamic = rewrite_conf.get("dynamic", False)
        else:
            self.bits, self.dtype = _parse_bits_and_dtype(info)
            self.is_sym = True
            self.dynamic = "dynamic" in info
            parts = info.split("_")
            if len(parts) < 2:
                raise ValueError(f"Cannot parse granularity from quant info: {info}")
            sub_parts = parts[1].rsplit("-")
            self.granularity = "-".join(sub_parts[0:2])

        if self.dtype == "fp8":
            self.is_sym = True
        if self.granularity == "per-token":
            self.dynamic = True

    def _set_quant_range(self):
        if self.dtype == "fp8":
            self.qmin, self.qmax = FP8_E4M3_QMIN, FP8_E4M3_QMAX
        elif self.dtype == "int" and self.is_sym:
            self.qmin = -(2 ** (self.bits - 1))
            self.qmax = 2 ** (self.bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**self.bits - 1

    def _set_quant_parameters(self, scale, zero_point=None):
        self.scale = nn.Parameter(scale)
        self.zero_point = nn.Parameter(zero_point) if zero_point is not None else None

    def _init_quant_params(self, x):
        with torch.no_grad():
            if self.is_act:
                if self.dynamic:
                    self.init = True
                    return
                self.init = False
                self.scale = self.zero_point = None
                if self.resume:
                    self.init = True
                    zp = torch.empty(1) if not self.is_sym else None
                    self._set_quant_parameters(torch.empty(1), zp)
                return

            if self.is_sym:
                self._set_quant_parameters(
                    self._compute_scales(x, self.granularity, self.group_size)
                )
            else:
                scale, zp = self._compute_scales_and_zero_points(
                    x, self.granularity, self.group_size
                )
                self._set_quant_parameters(scale, zp.round())

    def _compute_scales(self, x, granularity="per-tensor", group_size=-1):
        if granularity == "per-tensor":
            s = torch.clamp(torch.max(torch.abs(x.flatten())), min=1e-8)

        elif granularity == "per-channel":
            if len(x.shape) > 2:
                x = x.flatten(1)
            s = torch.clamp(x.abs().max(dim=-1)[0], min=1e-8)
            s = s.unsqueeze(1)  # shape: [out_channels, 1]

        elif granularity == "per-group":
            if x.shape[1] % group_size != 0:
                raise ValueError(
                    f"dim 1 ({x.shape[1]}) not divisible by group_size ({group_size})"
                )
            x_g = x.view(x.shape[0], x.shape[1] // group_size, group_size)
            s = torch.clamp(x_g.abs().max(dim=-1)[0], min=1e-8)  # shape: [out_channels, n_groups]

        elif granularity == "per-token":
            init_shape = x.shape
            rx = x.reshape(-1, x.shape[-1])
            tmp = torch.zeros(rx.shape[0], device=x.device, dtype=x.dtype)
            xmax = torch.maximum(
                torch.abs(torch.minimum(rx.min(1)[0], tmp)),
                torch.maximum(rx.max(1)[0], tmp),
            )
            s = xmax.unsqueeze(1)  # shape: [n_tokens, 1]
            s[xmax == 0] = 1
            # Reshape scale back to match original input dims, e.g. [B, S, 1]
            if len(init_shape) > 2:
                s = s.reshape(*init_shape[:-1], 1)

        elif granularity == "per-head":
            # Per-head: reshape [..., num_heads * head_dim] -> [..., num_heads, head_dim]
            # then reduce over all dims except num_heads to get scale shape (num_heads,)
            assert self.num_heads > 0, "num_heads must be set for per-head granularity"
            head_dim = x.shape[-1] // self.num_heads
            x_heads = x.view(*x.shape[:-1], self.num_heads, head_dim)
            # Flatten all dims except the num_heads dim, then take max per head
            # x_heads: [..., num_heads, head_dim] -> reduce all except dim -2
            s = (
                torch.clamp(
                    x_heads.abs().flatten(0, -3).amax(dim=(0, -1)),  # shape: (num_heads,)
                    min=1e-8,
                )
                if x_heads.dim() > 2
                else torch.clamp(
                    x_heads.abs().amax(dim=-1),  # shape: (num_heads,)
                    min=1e-8,
                )
            )

        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

        return s / self.qmax

    def _compute_scales_and_zero_points(self, x, granularity="per-tensor", group_size=-1):
        if granularity == "per-tensor":
            xmin = min(torch.min(x.flatten()), 0.0)
            xmax = max(torch.max(x.flatten()), 0.0)
            if xmin == xmax:
                xmin, xmax = -1.0, 1.0
            s = max((xmax - xmin) / (self.qmax - self.qmin), 1e-8)
            zp = torch.round(-xmin / s) + self.qmin

        elif granularity == "per-channel":
            if len(x.shape) > 2:
                x = x.flatten(1)
            tmp = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            xmin = torch.minimum(x.min(dim=-1)[0], tmp)
            xmax = torch.maximum(x.max(dim=-1)[0], tmp)
            mask = xmin == xmax
            xmin[mask], xmax[mask] = -1.0, 1.0
            s = torch.clamp((xmax - xmin) / (self.qmax - self.qmin), min=1e-8)
            zp = torch.round(-xmin / s) + self.qmin
            s = s.unsqueeze(1)
            zp = zp.unsqueeze(1)

        elif granularity == "per-group":
            if x.shape[1] % group_size != 0:
                raise ValueError(
                    f"dim 1 ({x.shape[1]}) not divisible by group_size ({group_size})"
                )
            x_g = x.view(x.shape[0], x.shape[1] // group_size, group_size)
            tmp = torch.zeros(x_g.shape[0], x_g.shape[1], device=x.device, dtype=x.dtype)
            xmin = torch.minimum(x_g.min(dim=-1)[0], tmp)
            xmax = torch.maximum(x_g.max(dim=-1)[0], tmp)
            mask = xmin == xmax
            xmin[mask], xmax[mask] = -1.0, 1.0
            s = torch.clamp((xmax - xmin) / (self.qmax - self.qmin), min=1e-8)
            zp = torch.round(-xmin / s) + self.qmin

        elif granularity == "per-token":
            rx = x.reshape(-1, x.shape[-1])
            tmp = torch.zeros(rx.shape[0], device=x.device, dtype=x.dtype)
            xmin = torch.minimum(rx.min(dim=1)[0], tmp)
            xmax = torch.maximum(rx.max(dim=1)[0], tmp)
            mask = xmin == xmax
            xmin[mask], xmax[mask] = -1.0, 1.0
            s = torch.clamp((xmax - xmin) / (self.qmax - self.qmin), min=1e-8)
            zp = torch.round(-xmin / s) + self.qmin
            s = s.unsqueeze(1)
            zp = zp.unsqueeze(1)

        elif granularity == "per-head":
            # Per-head: reshape [..., num_heads * head_dim] -> [..., num_heads, head_dim]
            # then reduce over all dims except num_heads to get scale/zp shape (num_heads,)
            assert self.num_heads > 0, "num_heads must be set for per-head granularity"
            head_dim = x.shape[-1] // self.num_heads
            x_heads = x.view(*x.shape[:-1], self.num_heads, head_dim)
            if x_heads.dim() > 2:
                flat = x_heads.flatten(0, -3)  # (N, num_heads, head_dim)
                reduce_dims = (0, -1)
            else:
                flat = x_heads  # (num_heads, head_dim)
                reduce_dims = (-1,)
            tmp = torch.zeros(self.num_heads, device=x.device, dtype=x.dtype)
            xmin = torch.minimum(flat.amin(dim=reduce_dims), tmp)  # (num_heads,)
            xmax = torch.maximum(flat.amax(dim=reduce_dims), tmp)  # (num_heads,)
            mask = xmin == xmax
            xmin[mask], xmax[mask] = -1.0, 1.0
            s = torch.clamp((xmax - xmin) / (self.qmax - self.qmin), min=1e-8)
            zp = torch.round(-xmin / s) + self.qmin

        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

        zp = torch.clamp(
            zp if isinstance(zp, torch.Tensor) else torch.tensor(zp),
            self.qmin,
            self.qmax,
        )
        return s, zp

    def _lazy_init(self, x):
        if not hasattr(self, "calib_count"):
            self.calib_count = 0
            self.overall_scale = []
            self.overall_zero_point = []

        if len(x.shape) == 2:  # for MoE
            x = x.unsqueeze(0)

        if self.is_sym:
            self.overall_scale.append(self._compute_scales(x, self.granularity, self.group_size))
        else:
            scale, zp = self._compute_scales_and_zero_points(x, self.granularity, self.group_size)
            self.overall_scale.append(scale)
            self.overall_zero_point.append(zp)
        self.calib_count += x.shape[0]

    def _expand_scale_zp(self, scale, zero_point, x):
        def _expand(t, target_shape):
            if t is None:
                return None
            return t.expand(target_shape)

        if self.granularity == "per-channel":
            # scale: [out_channels, 1] -> [out_channels, in_features]
            target = x.shape if len(x.shape) == 2 else (x.shape[0], x.flatten(1).shape[1])
            scale = _expand(scale, target)
            zero_point = _expand(zero_point, target)

        elif self.granularity == "per-group":
            # scale: [out_channels, n_groups] -> [out_channels, in_features]
            group_size = self.group_size
            scale = (
                scale.unsqueeze(-1).expand(*scale.shape, group_size).reshape(scale.shape[0], -1)
            )
            if zero_point is not None:
                zero_point = (
                    zero_point.unsqueeze(-1)
                    .expand(*zero_point.shape, group_size)
                    .reshape(zero_point.shape[0], -1)
                )

        elif self.granularity == "per-token":
            # scale: [n_tokens, 1] -> [n_tokens, in_features] then reshape to x.shape
            init_shape = x.shape
            rx = x.reshape(-1, x.shape[-1])
            scale = _expand(scale, rx.shape).reshape(init_shape)
            zero_point = (
                _expand(zero_point, rx.shape).reshape(init_shape)
                if zero_point is not None
                else None
            )

        return scale, zero_point

    def fake_quant(self, x):
        scale = clamp_ste(self.scale, 1e-4, 1e4)
        round_zero_point = (
            None if self.is_sym else clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        )
        scale, round_zero_point = self._expand_scale_zp(scale, round_zero_point, x)

        if self.is_w4a8_fp8:
            x_int4 = round_ste(x / scale)
            x_int4 = clamp_ste(x_int4, self.qmin, self.qmax).mul(scale)
            fp8_scale = scale.max() * self.qmax / FP8_E4M3_QMAX
            weight_fp8 = (x_int4 / fp8_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            return weight_fp8.to(torch.bfloat16) * fp8_scale

        if self.dtype == "fp8":
            weight_fp8 = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            return weight_fp8.to(torch.bfloat16) * scale

        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = clamp_ste(x_int, self.qmin, self.qmax)
        if round_zero_point is not None:
            x_int = x_int.sub(round_zero_point)
        return x_int.mul(scale)

    def forward(self, x: torch.Tensor):
        if self.bits >= 16:
            return x

        if self.is_act and not self.dynamic and not self.init:
            self._lazy_init(x)
            return x

        if self.dynamic:
            if self.is_sym:
                self.scale = self._compute_scales(x, self.granularity, self.group_size)
            else:
                self.scale, self.zero_point = self._compute_scales_and_zero_points(
                    x, self.granularity, self.group_size
                )

        return self.fake_quant(x)


class QuantLinear(nn.Module):
    def __init__(
        self,
        org_module,
        config,
        quant_info,
        use_weight_quant,
        use_act_quant,
        resume=False,
        qkv_config=None,
    ):
        super().__init__()
        self.fwd_func = F.linear
        self.register_parameter("weight", org_module.weight)
        self.bias = None
        if org_module.bias is not None:
            self.register_parameter("bias", org_module.bias)
        self.use_weight_quant = use_weight_quant
        self.use_act_quant = use_act_quant
        if self.use_weight_quant:
            self.weight_quantizer = Quantizer(config, quant_info, x=org_module.weight)
        if self.use_act_quant:
            self.act_quantizer = Quantizer(config, quant_info, is_act=True, resume=resume)

        # QKV output quantization for KV cache compression simulation
        self.use_qkv_quant = qkv_config is not None
        if self.use_qkv_quant:
            qkv_cfg = {**config, "activation": qkv_config}
            num_heads = qkv_config.get("num_heads", -1) if isinstance(qkv_config, dict) else -1
            self.qkv_quantizer = Quantizer(
                qkv_cfg, quant_info, is_act=True, resume=resume, num_heads=num_heads
            )

    def forward(self, input: torch.Tensor):
        if input.shape[0] == 0:
            return self.fwd_func(input, self.weight, self.bias)

        weight = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        if self.use_act_quant:
            input = self.act_quantizer(input)
        output = self.fwd_func(input, weight, self.bias)
        if self.use_qkv_quant:
            output = self.qkv_quantizer(output)
        return output

    def set_quant_state(self, weight_quant=False, act_quant=False, qkv_quant=None):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        if qkv_quant is not None:
            # Only enable qkv_quant if this module actually has a qkv_quantizer
            self.use_qkv_quant = qkv_quant and hasattr(self, "qkv_quantizer")
