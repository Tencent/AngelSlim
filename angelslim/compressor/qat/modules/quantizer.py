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

from ....utils import is_deepspeed_zero3_enabled, is_zero3_param

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
    def __init__(
        self,
        config,
        quant_info,
        x=None,
        is_act=False,
        resume=False,
        num_heads=-1,
        weight_shape=None,
    ):
        super().__init__()
        self.is_act = is_act
        self.num_heads = num_heads
        # ``weight_shape`` lets the caller pre-declare the (out_features,
        # in_features) of the parent Linear so we can size weight-side
        # quantizer Parameters without ever touching the (possibly ZeRO-3
        # sharded) weight tensor.
        self.weight_shape = (
            (int(weight_shape[0]), int(weight_shape[1])) if weight_shape is not None else None
        )
        # Configurable initial values used when ZeRO-3 is active and we
        # cannot depend on the weight data.
        self.weight_scale_init_value = float(config.get("weight_scale_init_value", 1.0))
        self.activation_scale_init_value = float(config.get("activation_scale_init_value", 1.0))
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
        self._init_lwc_params(x, config)

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
                    init_val = self.activation_scale_init_value
                    scale = torch.full((1,), init_val, dtype=torch.float32)
                    zp = torch.zeros(1, dtype=torch.float32) if not self.is_sym else None
                    self._set_quant_parameters(scale, zp)
                return

            # Weight-side path. If we cannot use ``x`` (ZeRO-3 sharded,
            # meta, or simply not provided), allocate Parameters by shape
            # and ``weight_scale_init_value``.
            if self._needs_external_weight_init(x):
                shape = self._weight_scale_shape_from_meta()
                init_val = self.weight_scale_init_value
                scale = torch.full(shape, init_val, dtype=torch.float32)
                zp = torch.zeros(shape, dtype=torch.float32) if not self.is_sym else None
                self._set_quant_parameters(scale, zp)
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

    def _needs_external_weight_init(self, x):
        """True when weight-side init must skip data-dependent computation
        and instead allocate Parameters from shape + init_value.

        Triggered by:
          * DeepSpeed ZeRO-3 active (HF integration registered)
          * ``x`` is a ZeRO-3 sharded Parameter
          * ``x`` is None / on meta device / empty
        """
        if is_deepspeed_zero3_enabled():
            return True
        if x is None:
            return True
        if is_zero3_param(x):
            return True
        if hasattr(x, "device") and x.device.type == "meta":
            return True
        if hasattr(x, "numel") and x.numel() == 0:
            return True
        return False

    def _weight_2d_shape(self):
        """Resolve (out_features, in_features) for the underlying Linear.
        Callers must have passed ``weight_shape`` via ``QuantLinear``."""
        if self.weight_shape is not None:
            return self.weight_shape
        raise RuntimeError(
            "Quantizer needs ``weight_shape`` to size weight scale without a "
            "concrete tensor (set in QuantLinear.__init__)."
        )

    def _weight_scale_shape_from_meta(self):
        out_dim, in_dim = self._weight_2d_shape()
        if self.granularity == "per-channel":
            return (out_dim, 1)
        if self.granularity == "per-group":
            if not self.group_size or self.group_size <= 0:
                raise ValueError("per-group quantization requires positive group_size.")
            if in_dim % self.group_size != 0:
                raise ValueError(
                    f"dim 1 ({in_dim}) not divisible by group_size ({self.group_size})"
                )
            return (out_dim, in_dim // self.group_size)
        # per-tensor and any reduce-to-scalar variant
        return (1,)

    def _init_lwc_params(self, x, config):
        lwc_cfg = config.get("lwc", {})
        if isinstance(lwc_cfg, dict):
            self.lwc = (not self.is_act) and bool(lwc_cfg.get("enable_lwc", False))
            self.lwc_init_value = float(lwc_cfg.get("lwc_init_value", 4.0))
        else:
            self.lwc = (not self.is_act) and bool(lwc_cfg)
            self.lwc_init_value = 4.0

        if self.lwc:
            # Resolve (out_dim, in_dim) without depending on ``x`` data.
            if self._needs_external_weight_init(x):
                out_dim, in_dim = self._weight_2d_shape()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                if x.dim() != 2:
                    x_for_shape = x.flatten(1)
                else:
                    x_for_shape = x
                out_dim, in_dim = x_for_shape.shape
                device = x.device

            if self.granularity == "per-group":
                if not self.group_size or self.group_size <= 0:
                    raise ValueError("per-group quantization requires positive group_size.")
                assert in_dim % self.group_size == 0
                n_groups = in_dim // self.group_size
                dim1 = out_dim * n_groups
            elif self.granularity == "per-channel":
                dim1 = out_dim
            else:
                dim1 = 1

            init = torch.ones((dim1, 1), device=device, dtype=torch.float32) * self.lwc_init_value
            self.clip_factor_w_max = nn.Parameter(init.clone(), requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(init.clone(), requires_grad=True)
            self.sigmoid = nn.Sigmoid()

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
            scale = _expand(scale, x.shape)
            zero_point = _expand(zero_point, x.shape) if zero_point is not None else None

        elif self.granularity == "per-head":
            if self.num_heads <= 0:
                raise ValueError("num_heads must be set for per-head granularity.")
            if x.shape[-1] % self.num_heads != 0:
                raise ValueError(
                    f"last dim ({x.shape[-1]}) must be divisible by num_heads ({self.num_heads})"
                )
            head_dim = x.shape[-1] // self.num_heads
            head_shape = (*x.shape[:-1], self.num_heads, head_dim)

            def _expand_per_head(t):
                if t is None:
                    return None
                # Broadcast one scale per head across that head's contiguous feature slice.
                view_shape = (1,) * (x.dim() - 1) + (self.num_heads, 1)
                return t.reshape(view_shape).expand(head_shape).reshape(x.shape)

            scale = _expand_per_head(scale)
            zero_point = _expand_per_head(zero_point)

        return scale, zero_point

    def _expand_scale_zp_lwc(self, scale, zero_point, x):
        def _expand(t, target_shape):
            if t is None:
                return None
            return t.expand(target_shape)

        if self.granularity == "per-channel":
            scale = _expand(scale, x.shape)
            zero_point = _expand(zero_point, x.shape)

        elif self.granularity == "per-group":
            group_size = self.group_size
            scale = scale.unsqueeze(-1).expand(*scale.shape, group_size).reshape(x.shape)
            if zero_point is not None:
                zero_point = (
                    zero_point.unsqueeze(-1).expand(*zero_point.shape, group_size).reshape(x.shape)
                )

        return scale, zero_point

    def _lwc_fake_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        # Weight-only LWC path (OmniQuant-style):
        # compute scale/zp from (possibly grouped) xmin/xmax
        # with learnable bound factors, then quantize with STE.
        x_dtype = x.dtype
        x_work = x
        if x_work.dim() != 2:
            x_work = x_work.flatten(1)

        if self.granularity == "per-group":
            out_dim, in_dim = x_work.shape
            x_reduce = x_work.reshape(out_dim, in_dim // self.group_size, self.group_size)
            xmin = x_reduce.amin(dim=-1)
            xmax = x_reduce.amax(dim=-1)
        elif self.granularity == "per-channel":
            xmin = x_work.amin(dim=-1, keepdim=True)
            xmax = x_work.amax(dim=-1, keepdim=True)
        else:
            # per-tensor (default)
            xmin = x_work.amin().view(1, 1)
            xmax = x_work.amax().view(1, 1)

        xmax = self.sigmoid(self.clip_factor_w_max).reshape_as(xmax) * xmax
        xmin = self.sigmoid(self.clip_factor_w_min).reshape_as(xmin) * xmin
        if self.is_sym:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = (abs_max / self.qmax).to(dtype=x_work.dtype)
            round_zero_point = None
        else:
            range_ = xmax - xmin
            scale = (range_ / (self.qmax - self.qmin)).to(dtype=x_work.dtype)
            zero_point = (-xmin) / range_ * (self.qmax - self.qmin) + self.qmin
            round_zero_point = clamp_ste(round_ste(zero_point), self.qmin, self.qmax)
        scale, round_zero_point = self._expand_scale_zp_lwc(scale, round_zero_point, x_work)
        x_dequant = self._fake_quant_with_params(x_work, scale, round_zero_point)
        return x_dequant.to(dtype=x_dtype).reshape(x.shape)

    def _w4a8_fp8_ste_from_dequant(
        self, x_dequant: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        fp8_scale = scale.max() * self.qmax / FP8_E4M3_QMAX
        weight_fp8 = x_dequant / fp8_scale
        weight_fp8_q = weight_fp8.clamp(FP8_E4M3_QMIN, FP8_E4M3_QMAX).to(torch.float8_e4m3fn)
        weight_fp8_q = (weight_fp8_q.to(torch.bfloat16) - weight_fp8).detach() + weight_fp8
        return weight_fp8_q * fp8_scale

    def _fake_quant_with_params(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        round_zero_point: torch.Tensor | None,
    ) -> torch.Tensor:
        scale = clamp_ste(scale, 1e-4, 1e4)
        if round_zero_point is not None:
            round_zero_point = clamp_ste(round_zero_point, self.qmin, self.qmax)

        if self.is_w4a8_fp8:
            x_int4 = round_ste(x / scale)
            x_int4 = clamp_ste(x_int4, self.qmin, self.qmax)
            x_dequant = x_int4.mul(scale)
            return self._w4a8_fp8_ste_from_dequant(x_dequant, scale)

        if self.dtype == "fp8":
            weight_fp8 = x / scale
            weight_fp8 = clamp_ste(weight_fp8, FP8_E4M3_QMIN, FP8_E4M3_QMAX)
            weight_fp8_q = weight_fp8.to(torch.float8_e4m3fn).to(torch.bfloat16)
            weight_fp8_q = (weight_fp8_q - weight_fp8).detach() + weight_fp8
            return weight_fp8_q * scale

        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = clamp_ste(x_int, self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        return x_dequant

    def fake_quant(self, x):
        scale = clamp_ste(self.scale, 1e-4, 1e4)
        round_zero_point = (
            None if self.is_sym else clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        )
        scale, round_zero_point = self._expand_scale_zp(scale, round_zero_point, x)
        out = self._fake_quant_with_params(x, scale, round_zero_point)
        # Scale is kept in fp32 for numerical stability, but multiplying by
        # a bf16/fp16 activation upcasts the result. Cast back to the input
        # dtype so downstream F.linear / DeepSpeed autocast wrappers see a
        # consistent dtype.
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out

    def forward(self, x: torch.Tensor):
        if self.bits >= 16:
            return x

        if self.lwc:
            return self._lwc_fake_quant_weight(x)

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
        # Under ZeRO-3 the weight Parameter ``org_module.weight`` may be a
        # zero-numel shard. Pass an explicit (out, in) shape so the weight
        # quantizer can size its scale Parameter from the Linear shape
        # rather than inspecting the (possibly sharded) tensor.
        weight_shape = (org_module.out_features, org_module.in_features)
        if self.use_weight_quant:
            self.weight_quantizer = Quantizer(
                config,
                quant_info,
                x=org_module.weight,
                weight_shape=weight_shape,
            )
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
        weight = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        if self.use_act_quant:
            input = self.act_quantizer(input)
        # Defensive dtype alignment: upstream (DeepSpeed ZeRO-3 / HF
        # autocast) may have cast ``input`` to fp16 even though we run in
        # bf16. Align to the (fake-quantised) weight dtype so F.linear
        # stays consistent.
        output = self.fwd_func(
            input.to(self.weight.dtype), weight.to(self.weight.dtype), self.bias
        )
        if self.use_qkv_quant:
            output = self.qkv_quantizer(output)
        return output

    def set_quant_state(self, weight_quant=False, act_quant=False, qkv_quant=None):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        if qkv_quant is not None:
            # Only enable qkv_quant if this module actually has a qkv_quantizer
            self.use_qkv_quant = qkv_quant and hasattr(self, "qkv_quantizer")
