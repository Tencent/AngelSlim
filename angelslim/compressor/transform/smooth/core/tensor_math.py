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

"""Backend-agnostic smooth tensor math.

Combines the pure-tensor building blocks shared by the vLLM online
calibration path and the HuggingFace offline conversion path:

* Smooth-weight formulas (``smooth_default`` / ``smooth_per_tensor_act_first``).
* Grouped-Query Attention head expansion / pooling helpers.
* RoPE-aware QK geometry utilities.
* Low-level tensor utilities (per-channel absmax, in-place fp32 scaling).
* Quantize-Dequantize helpers used by the alpha grid search.

All functions accept torch tensors and return torch tensors with no
awareness of TP / Linear / FusedMoE.
"""

import math
import os

import torch

__all__ = [
    "power_of_two_smooth_scale",
    "fp8_group_qdq_act",
    "fp8_block_qdq_weight",
    "norm_scale",
    "apply_smooth_operations",
    "consumer_error",
    "bf16_equivalence_error",
    "producer_equivalence_error",
    # formulas
    "smooth_default",
    "smooth_per_tensor_act_first",
    # gqa
    "expand_kv_smooth_to_q",
    "pool_attn_out_smooth_to_v",
    # rope
    "pair_rope_kv_absmax",
    "rope_kv_smooth_weight",
    "fold_smooth_into_qk_norm",
    # tensor ops
    "per_channel_absmax",
    "inplace_mul_fp32",
    "inplace_div_fp32",
    "set_percentile_subsample",
    "get_percentile_subsample",
    # qdq
    "smooth_qdq_act",
    "smooth_qdq_weight",
]


# ===========================================================================
# Smooth-weight formulas
# ===========================================================================


def smooth_default(
    act_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    alpha: float,
    smooth_min: float = 1e-6,
    smooth_max: float = 1e6,
) -> torch.Tensor:
    """Classic SmoothQuant per-channel formula.

    ``smooth_j = max|x_j|^alpha / max|w_j|^(1 - alpha)``

    Args:
        act_absmax: ``[C]`` per-channel activation absmax (or EMA).
        weight_absmax: ``[C]`` per-channel weight absmax along the
            input dimension.
        alpha: smoothing strength in ``[0, 1]``.  Larger values bias the
            scale toward activation distribution.
        smooth_min / smooth_max: clamp bounds against degenerate values.

    Returns:
        ``[C]`` smooth-weight tensor on the same device / dtype as the
        inputs (after `.pow` they are floating-point).
    """
    act = act_absmax.clamp(min=1e-8)
    weight = weight_absmax.clamp(min=1e-8)
    smooth = act.pow(alpha) / weight.pow(1.0 - alpha)
    return smooth.clamp(smooth_min, smooth_max)


def smooth_per_tensor_act_first(
    act_absmax: torch.Tensor,
    mul: float,
    smooth_min: float = 1e-6,
    smooth_max: float = 1e6,
) -> torch.Tensor:
    """Per-tensor-act-first smooth formula.

    Pick a global target absmax ``mul * max(act_absmax)`` and rescale
    each channel so its activation magnitude approaches that target.

    ``tgt = mul * act_absmax.max()``
    ``smooth = act_absmax / tgt``

    Args:
        act_absmax: ``[C]`` per-channel activation absmax.
        mul: multiplier searched in the grid (``act_mul_min..act_mul_max``).
        smooth_min / smooth_max: clamp bounds.

    Returns:
        ``[C]`` smooth-weight tensor.
    """
    act = act_absmax.clamp(min=1e-8)
    per_tensor_absmax = act.max()
    tgt = mul * per_tensor_absmax
    smooth = act / tgt
    return smooth.clamp(smooth_min, smooth_max)


# ===========================================================================
# Grouped-Query Attention helpers
# ===========================================================================


def expand_kv_smooth_to_q(
    smooth_kv: torch.Tensor,
    head_dim: int,
    num_q_heads: int,
) -> torch.Tensor:
    """Expand a ``[num_kv_heads * head_dim]`` smooth weight to
    ``[num_q_heads * head_dim]`` via per-head repeat_interleave.

    Each KV head is shared by ``n_groups = num_q_heads // num_kv_heads``
    consecutive Q heads, so we tile its smooth weight that many times.

    Args:
        smooth_kv: 1-D, length ``num_kv_heads * head_dim``.
        head_dim: per-head dimension.
        num_q_heads: target Q-head count.

    Returns:
        1-D tensor of length ``num_q_heads * head_dim``.
    """
    kv_dim = smooth_kv.numel()
    num_kv_heads = kv_dim // head_dim
    if kv_dim % head_dim != 0:
        raise ValueError(f"smooth_kv len {kv_dim} not divisible by head_dim {head_dim}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}")
    n_groups = num_q_heads // num_kv_heads
    sw_kv = smooth_kv.reshape(num_kv_heads, head_dim)
    sw_q = sw_kv.repeat_interleave(n_groups, dim=0)
    return sw_q.reshape(num_q_heads * head_dim)


def pool_attn_out_smooth_to_v(
    smooth_attn_out: torch.Tensor,
    head_dim: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Reduce a ``[num_q_heads * head_dim]`` attn-output smooth weight
    to a ``[num_kv_heads * head_dim]`` V-output smooth weight by per-head
    channel-wise max within each KV-head group.

    Args:
        smooth_attn_out: 1-D, length ``num_q_heads * head_dim``.
        head_dim: per-head dimension.
        num_kv_heads: target KV-head count.

    Returns:
        1-D tensor of length ``num_kv_heads * head_dim``.
    """
    out_dim = smooth_attn_out.numel()
    num_q_heads = out_dim // head_dim
    if out_dim % head_dim != 0:
        raise ValueError(f"smooth len {out_dim} not divisible by head_dim {head_dim}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}")
    n_groups = num_q_heads // num_kv_heads
    sw_q = smooth_attn_out.reshape(num_kv_heads, n_groups, head_dim)
    sw_v = sw_q.max(dim=1).values  # [num_kv_heads, head_dim]
    return sw_v.reshape(num_kv_heads * head_dim)


# ===========================================================================
# RoPE-aware QK geometry utilities
# ===========================================================================


def pair_rope_kv_absmax(
    k_absmax: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE-aware symmetrisation to a per-channel K absmax vector.

    For each head ``h`` and dimension ``d`` in ``[0, head_dim/2)``:
        ``paired[h, d] = paired[h, d + head_dim/2]
            = max(k_absmax[h, d], k_absmax[h, d + head_dim/2])``

    Args:
        k_absmax: 1-D tensor of shape ``[num_kv_heads * head_dim]``.
        head_dim: per-head dimension (must divide ``k_absmax.numel()``).

    Returns:
        ``[num_kv_heads, head_dim]`` symmetrised absmax (each row's
        first half equals its second half).
    """
    if k_absmax.dim() != 1:
        raise ValueError(f"Expected 1-D k_absmax, got shape {tuple(k_absmax.shape)}")
    D = k_absmax.numel()
    if D % head_dim != 0:
        raise ValueError(f"k_absmax len {D} not divisible by head_dim {head_dim}")
    num_kv_heads = D // head_dim
    half_hd = head_dim // 2

    k_per_head = k_absmax.reshape(num_kv_heads, head_dim)
    paired = torch.empty(num_kv_heads, head_dim, dtype=k_per_head.dtype, device=k_per_head.device)
    paired_first_half = torch.maximum(k_per_head[:, :half_hd], k_per_head[:, half_hd:])
    paired[:, :half_hd] = paired_first_half
    paired[:, half_hd:] = paired_first_half
    return paired


def rope_kv_smooth_weight(
    k_absmax: torch.Tensor,
    head_dim: int,
    alpha: float,
) -> torch.Tensor:
    """Compute the per-head RoPE-aware smooth weight from K absmax.

    ``smooth[h, d] = paired[h, d]^alpha``, where ``paired`` comes from
    :func:`pair_rope_kv_absmax`.

    Returns:
        ``[num_kv_heads, head_dim]`` float32 smooth weight.
    """
    paired = pair_rope_kv_absmax(k_absmax.float(), head_dim)
    return paired.pow(alpha)


def fold_smooth_into_qk_norm(
    smooth_per_head: torch.Tensor,
) -> torch.Tensor:
    """Reduce a per-head ``[num_kv_heads, head_dim]`` smooth weight to a
    head-shared ``[head_dim]`` weight.

    For models where ``Q = q_norm(q_proj(x))`` (RMSNorm after proj), the
    proj-level scale is annihilated by the norm, so the smooth weight
    must be folded into the norm weight instead.  Since the norm weight
    has shape ``[head_dim]`` and is shared across heads, we collapse the
    per-head smooth weight by taking the channel-wise max across heads.

    Args:
        smooth_per_head: ``[num_kv_heads, head_dim]``.

    Returns:
        ``[head_dim]`` float32 norm-level smooth weight.
    """
    if smooth_per_head.dim() != 2:
        raise ValueError(f"Expected 2-D smooth_per_head, got shape {tuple(smooth_per_head.shape)}")
    return smooth_per_head.float().max(dim=0).values


# ===========================================================================
# Low-level tensor utilities
# ===========================================================================


# ---------------------------------------------------------------------------
# Stride-sampling configuration for percentile estimation.
# ---------------------------------------------------------------------------
_SAMPLE_STRIDE = 683  # prime, coprime with all common hidden_dims
_SAMPLE_MIN_KEEP = 1024  # numel <= STRIDE * MIN_KEEP → skip sampling

# Initial value taken from env var; can be flipped at runtime via
# set_percentile_subsample().
_SAMPLE_ENABLE = os.getenv("VLLM_CALIB_PERCENTILE_SUBSAMPLE", "0") == "1"


def set_percentile_subsample(enable: bool) -> None:
    """Enable / disable stride sub-sampling for percentile estimation.

    Affects :func:`per_channel_absmax`. Takes effect on the next call
    (safe to flip between calibration batches).
    """
    global _SAMPLE_ENABLE
    _SAMPLE_ENABLE = bool(enable)


def get_percentile_subsample() -> bool:
    """Return the current sub-sampling flag."""
    return _SAMPLE_ENABLE


def per_channel_absmax(tensor: torch.Tensor, token_clip: float = -1.0) -> torch.Tensor:
    """Per-channel absmax over ``dim=0``, with optional percentile clipping.

    Args:
        tensor: shape ``[num_tokens, dim]`` (1-D input is first unsqueezed).
        token_clip: clip mode for the per-channel statistic.
            * ``<= 0`` (default): absolute max along ``dim=0``.
            * In ``(0, 1]``: quantile fraction (e.g. 0.999).
            * In ``(1, 100)``: percentile value (e.g. 99.9).

    Implementation notes:
        * Uses ``torch.topk(..., dim=0)`` instead of ``torch.kthvalue`` —
          much faster on GPU when ``k`` is small (always the case for
          tail percentiles like 0.999).
        * Sub-samples rows with a coprime stride (683) when
          ``num_tokens`` is large.  683 is prime and coprime with all
          common hidden dims, so the sampling is uniform across channels.
        * Returns a CPU tensor for compatibility with running stats that
          live on the host side.

    Returns:
        CPU tensor of shape ``[dim]``.
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    abs_t = tensor.abs()

    use_percentile = token_clip is not None and token_clip > 0
    if not use_percentile:
        return abs_t.max(dim=0).values.detach().cpu()

    num_tokens = abs_t.shape[0]
    if num_tokens == 0:
        return abs_t.max(dim=0).values.detach().cpu()

    # Normalise token_clip to a quantile fraction in (0, 1].
    tc = float(token_clip)
    q = tc if tc <= 1.0 else tc / 100.0
    q = max(0.5, min(q, 1.0 - 1e-6))

    if not abs_t.is_floating_point():
        abs_t = abs_t.float()

    if _SAMPLE_ENABLE and num_tokens > _SAMPLE_STRIDE * _SAMPLE_MIN_KEEP:
        abs_t = abs_t[::_SAMPLE_STRIDE]
        num_tokens = abs_t.shape[0]

    try:
        k_high_raw = int(round((1.0 - q) * num_tokens))
        if k_high_raw <= 1:
            return abs_t.max(dim=0).values.detach().cpu()

        k_high = min(num_tokens, k_high_raw)
        top = torch.topk(abs_t, k_high, dim=0, largest=True, sorted=False).values
        return top.min(dim=0).values.detach().cpu()
    except Exception as e:  # pragma: no cover - defensive fallback
        print(
            f"[per_channel_absmax] token_clip topk failed (shape="
            f"{tuple(abs_t.shape)}, q={q}): {e}. Fallback to absolute max."
        )
        return abs_t.max(dim=0).values.detach().cpu()


def inplace_mul_fp32(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Multiply ``weight`` in-place by ``scale`` at fp32 precision.

    The operation is computed in float32 to avoid bfloat16 / float16
    accumulation errors and the result is written back into the original
    storage with the original dtype preserved.
    """
    orig_dtype = weight.dtype
    weight.data.copy_(weight.data.float().mul_(scale.float()).to(orig_dtype))


def inplace_div_fp32(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Divide ``weight`` in-place by ``scale`` at fp32 precision."""
    orig_dtype = weight.dtype
    weight.data.copy_(weight.data.float().div_(scale.float()).to(orig_dtype))


# ===========================================================================
# Quantize-Dequantize helpers (used by the alpha grid search)
# ===========================================================================
#
# These are intentionally backend-agnostic — both the vLLM online searcher
# and any offline simulator can call them with plain torch tensors.


def smooth_qdq_act(
    x: torch.Tensor,
    method: str = "per_token",
    quant_type: str = "int8",
    bits: int = 8,
) -> torch.Tensor:
    """Quantize-dequantize activation (already divided by smooth_weight).

    Args:
        x: ``[N, in_features]``, float32.
        method: ``"per_tensor"`` | ``"per_token"``.
        quant_type: ``"int8"`` | ``"int4"`` | ``"int"`` | ``"fp8"``.
        bits: bit width for INT.

    Returns:
        ``x_qdq``: ``[N, in_features]``, float32.
    """
    if quant_type in ("int8", "int4", "int"):
        bnt = (1 << (bits - 1)) - 1  # 127 for int8
        if method == "per_tensor":
            scale = x.abs().max().clamp(min=1e-8) / bnt
        else:  # per_token
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / bnt
        return (x / scale).round().clamp(-bnt - 1, bnt) * scale

    elif quant_type == "fp8":
        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        if method == "per_tensor":
            scale = x.abs().max().clamp(min=1e-8) / fp8_max
        else:  # per_token
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
        q = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return q.float() * scale

    else:
        raise ValueError(f"Unknown act quant_type: {quant_type}")


def smooth_qdq_weight(
    w: torch.Tensor,
    method: str = "per_channel",
    quant_type: str = "int8",
    bits: int = 8,
    group_size: int = 128,
    block_size: int = 128,
) -> torch.Tensor:
    """Quantize-dequantize weight (already multiplied by smooth_weight).

    Args:
        w: ``[out_features, in_features]``, float32.
        method: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"`` | ``"per_block"``.
        quant_type: ``"int8"`` | ``"int4"`` | ``"int"`` | ``"fp8"``.
        bits: bit width for INT.
        group_size: for ``per_group`` (along ``dim=1``).
        block_size: for ``per_block`` fp8.

    Returns:
        ``w_qdq``: ``[out_features, in_features]``, float32.
    """
    if quant_type in ("int8", "int4", "int"):
        bnt = (1 << (bits - 1)) - 1

        if method == "per_tensor":
            scale = w.abs().max().clamp(min=1e-8) / bnt
            return (w / scale).round().clamp(-bnt - 1, bnt) * scale

        elif method == "per_channel":
            # per output-channel (per row)
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / bnt
            return (w / scale).round().clamp(-bnt - 1, bnt) * scale

        elif method == "per_group":
            # group along in_features (dim=1)
            out_feat, in_feat = w.shape
            gs = group_size if group_size > 0 else in_feat
            if in_feat % gs != 0:
                # pad to group_size multiple
                pad_len = gs - in_feat % gs
                w_padded = torch.nn.functional.pad(w, (0, pad_len))
            else:
                w_padded = w
                pad_len = 0
            w_grouped = w_padded.reshape(out_feat, -1, gs)
            scale = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / bnt
            w_qdq = (w_grouped / scale).round().clamp(-bnt - 1, bnt) * scale
            w_qdq = w_qdq.reshape(out_feat, -1)
            if pad_len > 0:
                w_qdq = w_qdq[:, :in_feat]
            return w_qdq

        else:
            raise ValueError(f"Unknown weight method for INT: {method}")

    elif quant_type == "fp8":
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        if method == "per_tensor":
            scale = w.abs().max().clamp(min=1e-8) / fp8_max
            q = (w / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            return q.float() * scale

        elif method == "per_channel":
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
            q = (w / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            return q.float() * scale

        elif method == "per_block":
            out_feat, in_feat = w.shape
            # pad to block_size multiples
            pad_r = (block_size - in_feat % block_size) % block_size
            pad_b = (block_size - out_feat % block_size) % block_size
            if pad_r > 0 or pad_b > 0:
                w_padded = torch.nn.functional.pad(w, (0, pad_r, 0, pad_b))
            else:
                w_padded = w
            po, pi = w_padded.shape
            w_blocks = w_padded.reshape(po // block_size, block_size, pi // block_size, block_size)
            w_blocks = w_blocks.permute(0, 2, 1, 3)  # [bo, bi, bs, bs]
            scale = w_blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8) / fp8_max
            q = (w_blocks / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            w_qdq = (q.float() * scale).permute(0, 2, 1, 3).reshape(po, pi)
            return w_qdq[:out_feat, :in_feat]

        elif method == "per_group":
            # FP8 per_group: group along dim=1
            out_feat, in_feat = w.shape
            gs = group_size if group_size > 0 else in_feat
            if in_feat % gs != 0:
                pad_len = gs - in_feat % gs
                w_padded = torch.nn.functional.pad(w, (0, pad_len))
            else:
                w_padded = w
                pad_len = 0
            w_grouped = w_padded.reshape(out_feat, -1, gs)
            scale = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
            q = (w_grouped / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            w_qdq = (q.float() * scale).reshape(out_feat, -1)
            if pad_len > 0:
                w_qdq = w_qdq[:, :in_feat]
            return w_qdq

        else:
            raise ValueError(f"Unknown weight method for FP8: {method}")

    else:
        raise ValueError(f"Unknown weight quant_type: {quant_type}")


# Offline channel compensation.
def power_of_two_smooth_scale(act, weight, alpha, lower=1 / 12, upper=2):
    """Power-of-two channel scales; alpha=0 means identity."""
    if not 0 <= alpha <= 1 or not 0 < lower <= upper or not math.isfinite(upper):
        raise ValueError("invalid alpha or scale bounds")
    if act.shape != weight.shape or act.ndim != 1 or not act.numel():
        raise ValueError("expected matching nonempty activation/weight channel vectors")
    if (
        not torch.isfinite(act).all()
        or not torch.isfinite(weight).all()
        or (act < 0).any()
        or (weight < 0).any()
    ):
        raise ValueError("expected finite nonnegative activation/weight statistics")
    lo, hi = math.ceil(math.log2(lower)), math.floor(math.log2(upper))
    if lo > hi:
        raise ValueError("scale interval contains no power of two")
    if alpha == 0:
        return torch.ones_like(act, dtype=torch.float32)
    scale = smooth_default(act.float(), weight.float(), alpha, lower, upper)
    return torch.pow(2.0, scale.log2().round().clamp(lo, hi))


def _positive_scale(scale):
    if (
        scale.ndim != 1
        or not scale.numel()
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("expected a finite positive channel scale vector")


def norm_scale(norm, requested):
    """Return rounded Norm weights and the actual x/x' ratio."""
    requested = requested.to(device=norm.device, dtype=torch.float32)
    _positive_scale(requested)
    if norm.shape != requested.shape or not torch.isfinite(norm).all():
        raise ValueError("invalid Norm compensation shape or values")
    replacement = (norm.float() / requested).to(norm.dtype)
    if not torch.isfinite(replacement).all() or ((replacement == 0) & (norm != 0)).any():
        raise ValueError("Norm compensation overflowed or underflowed")
    actual = torch.where(replacement != 0, norm.float() / replacement.float(), requested)
    _positive_scale(actual)
    return replacement, actual


def apply_smooth_operations(name, tensor, operations, scales):
    """Apply compensation without mutating the source or changing dtype.
    KV_b compensation touches V rows only; nonparticipants pass through.
    """
    ops = operations.get(name, [])
    if not ops:
        return tensor
    out = tensor.float().clone()
    for op in ops:
        scale = scales[op["scale"]].to(device=out.device, dtype=torch.float32)
        if op["op"] == "replace":
            if scale.shape != out.shape or not torch.isfinite(scale).all():
                raise ValueError(f"invalid replacement: {name}")
            out = scale.clone()
            continue
        _positive_scale(scale)
        selector = op.get("selector")
        if selector:
            if selector["kind"] != "head_strided_rows" or op["axis"] != 0:
                raise ValueError(f"invalid selector: {op}")
            heads, width, offset, length = [
                selector[k] for k in ("heads", "head_width", "offset", "length")
            ]
            if (
                min(heads, width, length) <= 0
                or offset < 0
                or offset + length > width
                or out.ndim != 2
                or out.shape[0] != heads * width
                or scale.numel() != heads * length
            ):
                raise ValueError(f"selector shape mismatch: {name}")
            target = out.reshape(heads, width, -1)[:, offset : offset + length]
            factor = scale.reshape(heads, length, 1)
        else:
            target = out
            if op["axis"] not in range(out.ndim) or out.shape[op["axis"]] != scale.numel():
                raise ValueError(f"scale shape mismatch: {name}")
            shape = [1] * out.ndim
            shape[op["axis"]] = scale.numel()
            factor = scale.reshape(shape)
        if op["op"] == "mul":
            target.mul_(factor)
        elif op["op"] == "div":
            target.div_(factor)
        else:
            raise ValueError(f"unknown operation: {op}")
    out = out.to(tensor.dtype).contiguous()
    if not torch.isfinite(out).all():
        raise ValueError(f"nonfinite Smooth tensor: {name}")
    return out


def fp8_group_qdq_act(x, group_size=128):
    """E4M3 dynamic per-token/group QDQ with FP32 scales (vLLM eps=1e-10)."""
    if x.ndim != 2 or not x.numel() or group_size <= 0:
        raise ValueError("expected a nonempty activation matrix and positive group size")
    rows, cols = x.shape
    padded = torch.nn.functional.pad(x.float(), (0, (-cols) % group_size))
    blocks = padded.reshape(rows, -1, group_size)
    scale = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448.0
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape_as(padded)[:, :cols]


def fp8_block_qdq_weight(weight):
    """Gate/O QDQ via FP8Block RTN with FP32 scales."""
    from ....quant.ckpt_convert.formats.fp8 import FP8Block

    if weight.ndim != 2 or not weight.numel():
        raise ValueError("expected a nonempty weight matrix")
    quantized, scale = FP8Block(block_size=(128, 128), legacy=False).rtn(weight)
    rows, cols = weight.shape
    scale = scale.repeat_interleave(min(128, rows), 0).repeat_interleave(min(128, cols), 1)
    return quantized.float() * scale[:rows, :cols]


def consumer_error(x, weights, scale, device="cuda:0"):
    """FP8 output SSE normalized by reference energy."""
    x, scale = x.to(device).float(), scale.to(device).float()
    _positive_scale(scale)
    if x.ndim != 2 or not x.numel() or x.shape[1] != scale.numel() or not weights:
        raise ValueError("invalid consumer samples, weights or scale")
    xq = fp8_group_qdq_act((x / scale).bfloat16().float())
    error, energy = 0.0, 0.0
    for weight in weights:
        weight = weight.to(device)
        if weight.ndim != 2 or weight.shape[1] != scale.numel():
            raise ValueError("consumer weight/input scale mismatch")
        # Keep chunks aligned to 128-row FP8 tiles.
        for begin in range(0, weight.shape[0], 1024):
            w = weight[begin : begin + 1024].float()
            reference = x @ w.T
            ws = (w * scale).to(weight.dtype)
            candidate = xq @ fp8_block_qdq_weight(ws).T
            error += float((candidate - reference).square().sum())
            energy += float(reference.square().sum())
    return error / max(energy, 1e-30)


def bf16_equivalence_error(x, weights, scale, norm=None):
    """Consumer compensation relative L2 error after BF16 rounding."""
    scale = scale.to(x.device)
    actual = norm_scale(norm.to(x.device), scale)[1] if norm is not None else scale
    _positive_scale(actual)
    xs = (x.float() / actual).bfloat16().float()
    worst = 0.0
    for weight in weights:
        weight = weight.to(x.device)
        for begin in range(0, weight.shape[0], 1024):
            w = weight[begin : begin + 1024].float()
            before = x.float() @ w.T
            after = xs @ (w * actual).to(weight.dtype).float().T
            error = (
                (before - after).square().mean() / before.square().mean().clamp(min=1e-30)
            ).sqrt()
            if not torch.isfinite(error):
                return float("inf")
            worst = max(worst, float(error))
    return worst


def producer_equivalence_error(x, weight, scale, selector=None):
    """Relative L2 error of the V-row producer; selector is (heads,width,offset,length)."""
    weight, scale = weight.to(x.device), scale.to(x.device)
    _positive_scale(scale)
    if selector:
        heads, width, offset, length = selector
        if (
            min(heads, width, length) <= 0
            or offset < 0
            or offset + length > width
            or weight.shape[0] != heads * width
        ):
            raise ValueError("invalid producer head selector")
        weight = weight.reshape(heads, width, -1)[:, offset : offset + length].reshape(
            -1, weight.shape[-1]
        )
    if weight.ndim != 2 or weight.shape[0] != scale.numel():
        raise ValueError("producer weight/output scale mismatch")
    before = x.float() @ weight.float().T
    after = x.float() @ (weight.float() / scale[:, None]).to(weight.dtype).float().T
    expected = before / scale
    return float(
        ((after - expected).square().mean() / expected.square().mean().clamp(min=1e-30)).sqrt()
    )
