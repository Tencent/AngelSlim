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
"""
Hy4 (HYV4ForCausalLM / hy_v4_internal) mixed-precision offline re-quantiser.

Input
-----
A **BF16** Hy4 HuggingFace checkpoint (e.g. ``/dockerdata/hy4-bf16``) that
stores MoE experts as **batched** tensors::

    model.layers.{L}.mlp.experts.gate_up_proj   shape (E, 2*I, H)  bf16
    model.layers.{L}.mlp.experts.down_proj      shape (E, H,   I)  bf16

together with the usual per-linear ``.weight`` tensors for attention, dense
MLP (layer 0), shared_experts, router gate, etc.

Output
------
A new HF checkpoint directory where::

    MoE experts (layers >= first_k_dense_replace, < num_hidden_layers) ->
        NVFP4 weight-only, split into per-expert keys to match the reference
        GLM-5.1 NVFP4 layout -- for every ``{gate,up,down}_proj`` under
        ``model.layers.{L}.mlp.experts.{E}.``::
            .weight          uint8 packed FP4   (I, H/2)
            .weight_scale    fp8 e4m3           (I, H/16)
            .weight_scale_2  fp32 scalar
            .input_scale     fp32 scalar        <- from the calibration dump

    MTP routed MoE experts (``model.mtp_layers.{L}.mlp.experts``) ->
        the SAME NVFP4 layout, matching ``hy4_mtp_quant.py --mtp_format
        fp8_nvfp4``.

    Everything else that maps to an `nn.Linear` weight (attention QKV/O,
    linear_gate, dense-layer-0 MLP, shared_experts, MTP linears) ->
        FP8 128x128 blockwise (matches ``tools/fp8_quant_blockwise.py`` and
        the GLM-5.1-FP8 layout).
"""

import json
import math
import multiprocessing as mp
import os
import re
import shutil
from argparse import ArgumentParser

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

_FP8_FINFO = torch.finfo(torch.float8_e4m3fn)
_FP8_E4M3_MAX = float(_FP8_FINFO.max)  # 448.0

_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

_EXPERT_STATS_RE = re.compile(
    r"^model\.(?:"
    r"layers\.(?P<layer>\d+)(?P<block>\.mtp_block)?"  # model.layers.{N}[.mtp_block]
    r"|mtp_layers\.(?P<mtp>\d+)"  # model.mtp_layers.{L}
    r")\.mlp\.experts\."
    r"(?:(?P<expert>\d+)\.)?(?P<stage>gate_up_proj|down_proj)$"
)

NVFP4_PATTERNS = [
    r"model\.layers\.\d+\.mlp\.experts",
    r"model\.mtp_layers\.\d+\.mlp\.experts",
]

FP8_PATTERNS = [
    r"model\.layers\.\d+\.self_attn\.q_a_proj",
    r"model\.layers\.\d+\.self_attn\.q_b_proj",
    r"model\.layers\.\d+\.self_attn\.kv_a_proj_with_mqa",
    r"model\.layers\.\d+\.self_attn\.kv_b_proj",
    r"model\.layers\.\d+\.self_attn\.o_proj",
    r"model\.layers\.\d+\.self_attn\.linear_gate",
    r"model\.layers\.\d+\.self_attn\.indexer\.wk",
    r"model\.layers\.\d+\.self_attn\.indexer\.wq_b",
    r"model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)",
    r"model\.layers\.\d+\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)",
    r"model\.mtp_layers\.\d+\.self_attn\.q_a_proj",
    r"model\.mtp_layers\.\d+\.self_attn\.q_b_proj",
    r"model\.mtp_layers\.\d+\.self_attn\.kv_a_proj_with_mqa",
    r"model\.mtp_layers\.\d+\.self_attn\.kv_b_proj",
    r"model\.mtp_layers\.\d+\.self_attn\.o_proj",
    r"model\.mtp_layers\.\d+\.self_attn\.linear_gate",
    r"model\.mtp_layers\.\d+\.self_attn\.indexer\.wk",
    r"model\.mtp_layers\.\d+\.self_attn\.indexer\.wq_b",
    r"model\.mtp_layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)",
    r"model\.mtp_layers\.\d+\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)",
]

NVFP4_PATTERNS_COMPILED = [re.compile(p) for p in NVFP4_PATTERNS]
FP8_PATTERNS_COMPILED = [re.compile(p) for p in FP8_PATTERNS]

BATCHED_EXPERT_GATE_UP = ".mlp.experts.gate_up_proj"
BATCHED_EXPERT_DOWN = ".mlp.experts.down_proj"


def _cast_fp4(weight_abs_signed: torch.Tensor) -> torch.Tensor:
    """Signed float tensor (already scaled into the FP4 dynamic range) ->
    uint8 codes in [0, 15] (sign bit + 3 magnitude bits, packed later).
    """
    device = weight_abs_signed.device
    bounds = _E2M1_BOUNDS.to(device)
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8, device=device)
    mask = mask.expand([*weight_abs_signed.shape, 7])

    sign_bit = (weight_abs_signed < 0).to(torch.uint8)
    weight_abs = weight_abs_signed.abs()
    ord_ = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    round_ = torch.any((weight_abs.unsqueeze(-1) == bounds) * mask, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord_ + round_).to(torch.uint8)
    return fp4_val


def nvfp4_quantize_weight_only(
    weight_bf16: torch.Tensor,
    block_size: int = 16,
    weight_scale_2: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """NVFP4 weight-only quantisation. Returns (packed_uint8, weight_scale_fp8,
    weight_scale_2_fp32). ``weight_bf16`` must be 2-D with the last dim
    divisible by ``block_size``.

    If ``weight_scale_2`` is given it is used verbatim as the per-tensor global
    scale instead of deriving it from this tensor's own amax. This lets callers
    share one ``weight_scale_2`` across several weights (e.g. a fused gate/up
    pair uses ``max(gate.amax, up.amax)`` so both experts halves get an
    identical ``weight_scale_2``).
    """
    assert weight_bf16.dim() == 2, f"expected 2-D tensor, got {tuple(weight_bf16.shape)}"
    n, k = weight_bf16.shape
    assert k % block_size == 0, (
        f"last dim {k} not divisible by block_size {block_size}; "
        f"Hy4 MoE expert dims are multiples of 128 so this should never trigger."
    )

    w = weight_bf16.detach()
    if w.dtype != torch.bfloat16:
        w = w.to(torch.bfloat16)

    # per-tensor weight_scale_2 = amax / (6 * 448); optionally shared by caller.
    if weight_scale_2 is None:
        amax = w.float().abs().max()
        weight_scale_2 = (amax / 6.0 / 448.0).to(torch.float32)
    else:
        weight_scale_2 = weight_scale_2.to(torch.float32).to(w.device)
    if not torch.isfinite(weight_scale_2) or weight_scale_2.item() == 0.0:
        weight_scale_2 = torch.tensor(1.0e-6, dtype=torch.float32, device=w.device)

    # per-block scale
    blocks_view = w.view(n, k // block_size, block_size)
    per_block_amax = blocks_view.abs().amax(dim=-1).float()
    per_block_scale = per_block_amax / 6.0
    q_per_block_scale = per_block_scale / weight_scale_2
    q_per_block_scale[per_block_scale == 0] = 1.0
    q_per_block_scale = q_per_block_scale.clamp(min=_FP8_FINFO.min, max=_FP8_FINFO.max)
    weight_scale_fp8 = q_per_block_scale.to(torch.float8_e4m3fn)

    # weight -> FP4 codes -> packed uint8
    eff_scale = (weight_scale_fp8.to(torch.float32) * weight_scale_2).unsqueeze(-1)
    scaled = blocks_view / eff_scale
    scaled = scaled.view(n, k)
    fp4_codes = _cast_fp4(scaled)
    packed = (fp4_codes[..., 1::2] << 4) | fp4_codes[..., 0::2]

    return packed.contiguous(), weight_scale_fp8.contiguous(), weight_scale_2.contiguous()


def fp8_blockwise_quantise(
    weight_bf16: torch.Tensor,
    block_size: "tuple[int, int]" = (128, 128),
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Quantise a 2-D BF16 (or FP32) tensor to FP8 E4M3 with 128x128 block
    ``ue8m0`` power-of-2 scale, matching the xiaojingqi HY4 ``linear-fp8``
    reference release::

        bf16 = fp8 * 2^(scale_u8 - 127)   (scale_u8 shape = ceil(M/128), ceil(N/128))

    Behaviour:
      * pads with zeros on the last two dims so the tile grid is uniform;
      * the per-tile scale is a **ue8m0 biased exponent** stored as ``uint8``
        (``scale_u8 = ceil(log2(amax / fp8_max)) + 127``), i.e. the reference's
        ``.scale`` tensor.  The actual multiplicative scale is a pure power of
        two ``2^(scale_u8 - 127)`` (round-up / ceil convention, so the quantised
        weight never saturates).  This mirrors ``hy4_mxfp8_quant.py`` but keeps
        the 128x128 2-D tile grid instead of a 1-D 32-block layout.
      * a ``-1`` block dim means "tensor-wide" scale in that axis.
    """
    fp8_min = _FP8_FINFO.min
    fp8_max = _FP8_FINFO.max

    block_size_m, block_size_n = block_size
    rows, cols = weight_bf16.shape[-2:]

    if block_size_m == -1 or block_size_m > rows:
        block_size_m = rows
    if block_size_n == -1 or block_size_n > cols:
        block_size_n = cols

    param = weight_bf16
    # Pad along last two dims so we can tile evenly.
    if rows % block_size_m != 0:
        pad = torch.zeros(
            [*param.shape[:-2], block_size_m - rows % block_size_m, cols],
            dtype=param.dtype,
            device=param.device,
        )
        param = torch.concat([param, pad], dim=-2)
    if cols % block_size_n != 0:
        pad = torch.zeros(
            [*param.shape[:-2], param.shape[-2], block_size_n - cols % block_size_n],
            dtype=param.dtype,
            device=param.device,
        )
        param = torch.concat([param, pad], dim=-1)
    padded_shape = param.shape

    param_value = (
        param.float()
        .reshape(
            -1,
            math.ceil(rows / block_size_m),
            block_size_m,
            math.ceil(cols / block_size_n),
            block_size_n,
        )
        .permute(0, 1, 3, 2, 4)
    )

    max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
    # ue8m0 power-of-2 scale (round-up / ceil convention so the quantised
    # weight never saturates). Guard all-zero tiles -> exponent 0 (scale 2^-127).
    zero = max_abs == 0
    safe_max_abs = torch.where(zero, torch.ones_like(max_abs), max_abs)
    cont_scale = (safe_max_abs / fp8_max).clamp(min=2.0**-127)
    exponent = torch.ceil(torch.log2(cont_scale))  # e s.t. 2^e >= amax/fp8_max
    scale_u8 = (exponent + 127.0).clamp(0, 254).to(torch.uint8)

    # Actual multiplicative scale = 2^(scale_u8 - 127); divide weight by it.
    actual_scale = torch.pow(2.0, scale_u8.float() - 127.0)
    inv_scale = (1.0 / actual_scale).unsqueeze(-1).unsqueeze(-1)

    quantized = torch.clamp(param_value * inv_scale, min=fp8_min, max=fp8_max).to(
        torch.float8_e4m3fn
    )
    quantized = quantized.permute(0, 1, 3, 2, 4).reshape(padded_shape)[..., :rows, :cols]

    # scale_u8 is [1, grid_m, grid_n] (leading batch from the -1 reshape); drop
    # the batch dim to the reference's 2-D [ceil(M/128), ceil(N/128)] grid.
    # NEVER squeeze() – a grid dim of 1 (e.g. indexer.wk -> [1, 48]) must survive.
    grid_m = math.ceil(rows / block_size_m)
    grid_n = math.ceil(cols / block_size_n)
    scale_u8 = scale_u8.reshape(grid_m, grid_n).contiguous()

    return quantized.contiguous(), scale_u8


def _stats_key_layer(match, mtp: bool, num_hidden_layers: int) -> "int | None":
    """Layer index within the requested stack, or None for a foreign key.

    Returning None for every key of the *other* stack is what lets both maps
    be built from either file: pointing ``--mtp_moe_stats_json`` at a trunk
    dump yields an empty map rather than silently mis-scaling a tensor.
    """
    if match.group("mtp") is not None:  # model.mtp_layers.{L}
        return int(match.group("mtp")) if mtp else None
    layer = int(match.group("layer"))
    if match.group("block") is not None:  # model.layers.{N}.mtp_block
        return layer - num_hidden_layers if mtp else None
    if not mtp:
        return layer
    # Plain model.layers.{N} counts as MTP only past the end of the trunk.
    return layer - num_hidden_layers if layer >= num_hidden_layers > 0 else None


def load_input_scales_from_stats(
    stats_json_path: str, mtp: bool = False, num_hidden_layers: int = 0
) -> "dict[tuple[int, str], float]":
    """Return ``(layer_idx, stage) -> input_scale`` for one expert stack.

    ``mtp=False`` reads the trunk entries, ``mtp=True`` the MTP ones (which
    need ``num_hidden_layers`` to resolve their layer index).  A
    layer-aggregated entry wins over the max of that layer's per-expert ones.
    """
    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    aggregated: "dict[tuple[int, str], float]" = {}
    per_expert: "dict[tuple[int, str], float]" = {}

    for name, entry in stats.items():
        if not isinstance(entry, dict) or "max" not in entry:
            continue
        match = _EXPERT_STATS_RE.match(name)
        if match is None:
            continue
        layer = _stats_key_layer(match, mtp, num_hidden_layers)
        if layer is None:
            continue

        # Fold in |min|: activations are signed and NVFP4 saturates
        # symmetrically around zero.
        amax = abs(float(entry["max"]))
        low = entry.get("min")
        if isinstance(low, (int, float)):
            amax = max(amax, abs(float(low)))
        if not math.isfinite(amax) or amax <= 0.0:
            continue

        target = aggregated if match.group("expert") is None else per_expert
        key = (layer, match.group("stage"))
        target[key] = max(amax, target.get(key, 0.0))

    return {
        key: aggregated.get(key, per_expert.get(key, 0.0)) / (6.0 * _FP8_E4M3_MAX)
        for key in aggregated.keys() | per_expert.keys()
    }


def load_expert_input_scales(
    stats_json_path: "str | None",
    flag: str,
    required: bool,
    mtp: bool = False,
    num_hidden_layers: int = 0,
) -> "dict[tuple[int, str], float]":
    """:func:`load_input_scales_from_stats` plus path handling and warnings.

    A missing path or an empty result is only a problem when ``required`` --
    i.e. when that stack's experts really do land in NVFP4.  Otherwise the
    quantiser would fall back to ``input_scale=1.0``, which loads but has
    undefined accuracy, so say so loudly instead of failing late.
    """
    if not stats_json_path:
        if required:
            print(
                f"[warn] {flag} not provided; those experts will get "
                "input_scale=1.0 (model will load but accuracy is undefined).",
                flush=True,
            )
        return {}

    if not os.path.exists(stats_json_path):
        raise FileNotFoundError(f"{flag} not found: {stats_json_path}")

    scales = load_input_scales_from_stats(stats_json_path, mtp, num_hidden_layers)
    print(
        f"Loaded {len(scales)} ({'mtp_' if mtp else ''}layer, stage) "
        f"input_scale entries from {stats_json_path}",
        flush=True,
    )
    if required and not scales:
        print(
            f"[warn] {stats_json_path} holds no matching expert entries. For "
            "the MTP dump this usually means stage 1 ran without both "
            "`enable_mtp: true` and `mtp_calib_max_tokens` > 1, so the draft "
            "model was never exercised.",
            flush=True,
        )
    return scales


def classify_module(prefix: str) -> "str | None":
    """Return ``'nvfp4'`` / ``'fp8'`` / ``None`` for a module prefix.

    ``prefix`` has no trailing ``.weight``. NVFP4 wins if a prefix somehow
    matches both lists.
    """
    if any(p.fullmatch(prefix) for p in NVFP4_PATTERNS_COMPILED):
        return "nvfp4"
    if any(p.fullmatch(prefix) for p in FP8_PATTERNS_COMPILED):
        return "fp8"
    return None


def _is_fp8_linear(weight_name: str) -> bool:
    """True when a non-expert ``.weight`` tensor should go FP8 blockwise."""
    if not weight_name.endswith(".weight"):
        return False
    return classify_module(weight_name[: -len(".weight")]) == "fp8"


def _parse_batched_expert(weight_name: str):
    """Return ``(layer_idx, stage, mode)`` for a batched expert weight, else None.

    ``stage`` is ``'gate_up_proj'`` | ``'down_proj'``; ``mode`` is the
    classification of the batched ``...mlp.experts`` prefix -- ``'nvfp4'``,
    ``'fp8'``, or ``None`` when the patterns leave these experts un-quantised.
    """
    for suf, stage in (
        (BATCHED_EXPERT_GATE_UP, "gate_up_proj"),
        (BATCHED_EXPERT_DOWN, "down_proj"),
    ):
        if weight_name.endswith(suf):
            # ``suf`` already carries ``.mlp``, so this is "model.(mtp_)layers.{L}".
            layer_prefix = weight_name[: -len(suf)]
            m = re.fullmatch(r"model\.(?:layers|mtp_layers)\.(\d+)", layer_prefix)
            if m is None:
                return None
            mode = classify_module(f"{layer_prefix}.mlp.experts")
            return int(m.group(1)), stage, mode
    return None


# ===========================================================================
# Per-shard worker
# ===========================================================================
def process_shard(
    rank: int,
    file_name: str,
    input_path: str,
    output_path: str,
    first_k_dense_replace: int,
    num_hidden_layers: int,
    nvfp4_block_size: int,
    fp8_block_size: "tuple[int, int]",
    use_gpu: bool,
    input_scale_map: "dict[tuple[int, str], float]",
    mtp_input_scale_map: "dict[tuple[int, str], float] | None" = None,
):
    """Convert a single safetensors shard.

    Returns the shard's ``weight_map`` fragment (key -> file_name).
    """
    state_dict: "dict[str, torch.Tensor]" = {}
    index: "dict[str, str]" = {}
    device = f"cuda:{rank}" if use_gpu else "cpu"

    src = os.path.join(input_path, file_name)
    with safe_open(src, framework="pt", device="cpu") as f:
        for weight_name in f.keys():
            # ----------------------------------------------------------------
            # Case A: batched MoE experts -> split + NVFP4 + input_scale
            # ----------------------------------------------------------------
            parsed = _parse_batched_expert(weight_name)
            if parsed is not None:
                layer_idx, stage, mode = parsed
                is_mtp = weight_name.startswith("model.mtp_layers.")
                # Range check applies only to the main trunk; MTP (mtp_layers.N)
                # is a separate namespace and is not range-checked.
                out_of_range = not is_mtp and (
                    layer_idx < first_k_dense_replace or layer_idx >= num_hidden_layers
                )
                if mode is None or out_of_range:
                    # Either the patterns leave these experts un-quantised, or
                    # it is an out-of-range MoE layer (shouldn't happen for Hy4).
                    # Keep the batched tensor verbatim.
                    state_dict[weight_name] = f.get_tensor(weight_name)
                    index[weight_name] = file_name
                    continue

                weight = f.get_tensor(weight_name)  # bf16
                assert weight.dim() == 3, (
                    f"Expected batched expert weight to be 3-D, "
                    f"got {tuple(weight.shape)} for {weight_name}"
                )
                num_experts = weight.shape[0]
                prefix = weight_name[: -len(f".experts.{stage}")]  # "...{L}.mlp"

                # NVFP4 needs the calibrated per-tensor input_scale; FP8
                # per-expert does not (dynamic activation scaling).
                is_val = None
                if mode == "nvfp4":
                    # MTP experts are calibrated into their own dump, under a
                    # different key naming -- so they get their own map.
                    scale_src = (mtp_input_scale_map or {}) if is_mtp else input_scale_map
                    is_val = scale_src.get((layer_idx, stage))
                    if is_val is None:
                        print(
                            f"[warn] no input_scale stat for "
                            f"{'mtp_' if is_mtp else ''}layer={layer_idx} "
                            f"stage={stage}; falling back to 1.0",
                            flush=True,
                        )
                        is_val = 1.0

                def _emit_expert(
                    base: str, sub_w: torch.Tensor, w_scale_2=None, mode=mode, is_val=is_val
                ):
                    """Write one per-expert weight: FP8 ue8m0 or NVFP4 two-level
                    according to ``mode``. ``sub_w`` is a 2-D bf16 tensor.
                    ``w_scale_2`` (NVFP4 only) forces a shared per-tensor scale
                    so a fused gate/up pair gets an identical value."""
                    w2d = sub_w.contiguous()
                    if use_gpu:
                        w2d = w2d.to(device)
                    if mode == "fp8":
                        fp8_w, scale_u8 = fp8_blockwise_quantise(w2d, block_size=fp8_block_size)
                        state_dict[base + ".weight"] = fp8_w.cpu()
                        state_dict[base + ".scale"] = scale_u8.cpu()
                        for suf in (".weight", ".scale"):
                            index[base + suf] = file_name
                        del fp8_w, scale_u8
                    else:
                        packed, w_scale, w_scale_2 = nvfp4_quantize_weight_only(
                            w2d, block_size=nvfp4_block_size, weight_scale_2=w_scale_2
                        )
                        state_dict[base + ".weight"] = packed.cpu()
                        state_dict[base + ".weight_scale"] = w_scale.cpu()
                        # clone(): a fused gate/up pair is handed the SAME
                        # weight_scale_2 object, and safetensors >= 0.6 refuses
                        # to serialise two keys that share storage.
                        state_dict[base + ".weight_scale_2"] = w_scale_2.cpu().clone()
                        state_dict[base + ".input_scale"] = torch.tensor(
                            is_val, dtype=torch.float32
                        )
                        for suf in (
                            ".weight",
                            ".weight_scale",
                            ".weight_scale_2",
                            ".input_scale",
                        ):
                            index[base + suf] = file_name
                        del packed, w_scale, w_scale_2
                    del w2d

                if stage == "gate_up_proj":
                    # weight: (E, 2*I, H) -> split into (E, I, H) gate/up.
                    gate_w, up_w = weight.chunk(2, dim=1)
                    for e in range(num_experts):
                        g_e, u_e = gate_w[e], up_w[e]
                        # Share weight_scale_2 across the fused gate/up pair:
                        # derive it from max(gate.amax, up.amax) so both halves
                        # store an identical value (NVFP4 only; the FP8 path
                        # ignores the override). amax is permutation-invariant.
                        shared_w2 = None
                        if mode == "nvfp4":
                            gu_amax = torch.maximum(
                                g_e.float().abs().max(),
                                u_e.float().abs().max(),
                            )
                            shared_w2 = (gu_amax / 6.0 / 448.0).to(torch.float32)
                        _emit_expert(f"{prefix}.experts.{e}.gate_proj", g_e, shared_w2)
                        _emit_expert(f"{prefix}.experts.{e}.up_proj", u_e, shared_w2)
                    del gate_w, up_w, weight
                else:  # down_proj
                    # weight: (E, H, I).
                    for e in range(num_experts):
                        d_e = weight[e]
                        _emit_expert(f"{prefix}.experts.{e}.down_proj", d_e)
                    del weight

                if use_gpu:
                    torch.cuda.empty_cache()
                continue

            # ----------------------------------------------------------------
            # Case B: regular linear weight -> FP8 128x128 blockwise
            # ----------------------------------------------------------------
            if _is_fp8_linear(weight_name):
                weight = f.get_tensor(weight_name)
                assert weight.dim() == 2, (
                    f"Expected 2-D linear weight, got {tuple(weight.shape)} " f"for {weight_name}"
                )
                if use_gpu:
                    weight = weight.to(device)
                fp8_w, scale_u8 = fp8_blockwise_quantise(weight, block_size=fp8_block_size)
                base = weight_name[: -len(".weight")]
                state_dict[weight_name] = fp8_w.cpu()
                # ue8m0 scale stored as ``.scale`` (uint8), matching the
                # xiaojingqi HY4 linear-fp8 reference layout.
                state_dict[base + ".scale"] = scale_u8.cpu()
                index[base + ".scale"] = file_name
                index[weight_name] = file_name
                del weight, fp8_w, scale_u8
                if use_gpu:
                    torch.cuda.empty_cache()
                continue

            # ----------------------------------------------------------------
            # Case C: everything else -> passthrough (layernorms, router
            # gate, learnable_sink_param, hc_* tensors, embed_tokens,
            # lm_head, biases, etc.)
            # ----------------------------------------------------------------
            state_dict[weight_name] = f.get_tensor(weight_name)
            index[weight_name] = file_name

    dst = os.path.join(output_path, file_name)
    save_file(state_dict, dst)
    if use_gpu:
        torch.cuda.empty_cache()
    return index


def worker(
    i,
    file_names,
    input_path,
    output_path,
    first_k_dense_replace,
    num_hidden_layers,
    nvfp4_block_size,
    fp8_block_size,
    use_gpu,
    input_scale_map,
    return_dict,
    mtp_input_scale_map=None,
):
    world_size = max(1, torch.cuda.device_count())
    rank = i % world_size
    if use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(rank)
        # Force CUDA primary-context init before any tensor op.
        torch.empty(1, device=f"cuda:{rank}")
    for file_name in tqdm(file_names, desc=f"Worker {i}"):
        idx = process_shard(
            rank=rank,
            file_name=file_name,
            input_path=input_path,
            output_path=output_path,
            first_k_dense_replace=first_k_dense_replace,
            num_hidden_layers=num_hidden_layers,
            nvfp4_block_size=nvfp4_block_size,
            fp8_block_size=fp8_block_size,
            use_gpu=use_gpu,
            input_scale_map=input_scale_map,
            mtp_input_scale_map=mtp_input_scale_map,
        )
        return_dict[file_name] = idx


# ===========================================================================
# modules_to_not_convert / NVFP4 exclude-modules builders
# ===========================================================================
def build_modules_to_not_convert(
    output_index: "dict[str, str]",
    num_hidden_layers: int,
    first_k_dense_replace: int,
) -> "list[str]":
    """Walk the produced weight map and record every non-quantised tensor as a
    bare module name, matching the xiaojingqi HY4 ``linear-fp8`` reference::

      * FP8-blockwise linears (trunk + MTP) own a companion ``.scale`` -> excluded.
      * Routed MoE experts (trunk NVFP4 via sidecar, MTP FP8) -> excluded.
      * Every other tensor -> recorded with a trailing ``.weight`` / ``.bias``
        stripped to the module name; leaf params (``learnable_sink_param``,
        ``hc_base``, ``e_score_correction_bias``, ...) keep their full name.
      * No duplicate ``.weight``/``.bias``-suffixed entries are emitted.
    """
    # FP8-blockwise modules own a companion ``.scale`` (ue8m0). Covers trunk/MTP
    # FP8 linears AND MTP per-expert FP8 weights.
    fp8_module_names: "set[str]" = {
        k[: -len(".scale")] for k in output_index if k.endswith(".scale")
    }
    # Per-expert routed MoE weights are quantised (trunk = NVFP4 in the sidecar,
    # MTP = FP8) and must never appear in modules_to_not_convert.
    expert_re = re.compile(
        r"^model\.(?:layers|mtp_layers)\.\d+\.mlp\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj)\."
    )
    quant_companions = (".scale", ".weight_scale", ".weight_scale_2", ".input_scale")

    excluded: "set[str]" = set()
    for k in output_index.keys():
        if k.endswith(quant_companions):
            continue  # quantisation scale/companion tensor
        if expert_re.match(k):
            continue  # quantised routed-expert weight
        if k.endswith(".weight") and k[: -len(".weight")] in fp8_module_names:
            continue  # FP8-blockwise linear weight

        # Non-quantised tensor: strip a trailing .weight/.bias to the bare module
        # name; everything else (leaf params) keeps its full name.
        if k.endswith(".weight") or k.endswith(".bias"):
            excluded.add(k.rsplit(".", 1)[0])
        else:
            excluded.add(k)

    # Defence-in-depth: never list a real FP8 module as excluded.
    excluded -= fp8_module_names

    return sorted(excluded)


def build_nvfp4_exclude_globs(
    output_index: "dict[str, str]",
) -> "list[str]":
    """modelopt-style ``exclude_modules`` for the NVFP4 sidecar.

    Derived from what was actually emitted rather than from a hardcoded layer
    layout, so it stays correct whatever NVFP4_PATTERNS / FP8_PATTERNS select:
    a module is NVFP4 iff it owns a ``.weight_scale_2`` companion, and every
    other module in the checkpoint is excluded by exact name.
    """
    nvfp4_modules = {
        k[: -len(".weight_scale_2")] for k in output_index if k.endswith(".weight_scale_2")
    }
    quant_companions = (".scale", ".weight_scale", ".weight_scale_2", ".input_scale")

    excluded: "set[str]" = set()
    for k in output_index:
        if k.endswith(quant_companions):
            module = k.rsplit(".", 1)[0]
        elif k.endswith(".weight") or k.endswith(".bias"):
            module = k[: k.rfind(".")]
        else:
            module = k
        if module not in nvfp4_modules:
            excluded.add(module)
    return sorted(excluded)


# ===========================================================================
# Top-level driver
# ===========================================================================
def collect_shards(input_path: str) -> "list[str]":
    idx_file = os.path.join(input_path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        with open(idx_file) as f:
            wm = json.load(f)["weight_map"]
        return sorted(set(wm.values()))
    if os.path.exists(os.path.join(input_path, "model.safetensors")):
        return ["model.safetensors"]
    raise FileNotFoundError(f"No safetensors found under {input_path}")


def main(
    input_path: str,
    output_path: str,
    nvfp4_block_size: int,
    fp8_block_size: "tuple[int, int]",
    num_workers: int,
    use_gpu: bool,
    moe_stats_json: "str | None",
    mtp_moe_stats_json: "str | None" = None,
):
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(input_path, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    # ``first_k_dense_replace`` may be null in Hy4 configs -> treat as 0.
    first_k_dense_replace = int(config.get("first_k_dense_replace") or 0)
    num_hidden_layers = int(config.get("num_hidden_layers") or 0)
    num_nextn_predict_layers = int(config.get("num_nextn_predict_layers") or 0)
    print(
        f"first_k_dense_replace = {first_k_dense_replace}, "
        f"num_hidden_layers = {num_hidden_layers}, "
        f"num_nextn_predict_layers = {num_nextn_predict_layers}",
        flush=True,
    )

    if "quantization_config" in config:
        raise AssertionError(
            "Input checkpoint already has a quantization_config – this "
            "script expects a **BF16** Hy4 checkpoint."
        )

    # The trunk and the MTP head are calibrated into separate dumps with
    # different key namings, so each NVFP4 stack gets its own scale map.  The
    # MTP one is only consulted when NVFP4_PATTERNS actually puts the MTP
    # experts in NVFP4; FP8 / BF16 MTP experts need no activation scale.
    input_scale_map = load_expert_input_scales(moe_stats_json, "--moe_stats_json", required=True)
    mtp_input_scale_map = load_expert_input_scales(
        mtp_moe_stats_json,
        "--mtp_moe_stats_json",
        required=num_nextn_predict_layers > 0
        and classify_module("model.mtp_layers.0.mlp.experts") == "nvfp4",
        mtp=True,
        num_hidden_layers=num_hidden_layers,
    )

    shards = collect_shards(input_path)
    print(f"Found {len(shards)} safetensor shards to process.", flush=True)

    if use_gpu and torch.cuda.device_count() == 0:
        print("[warn] --use_gpu requested but no CUDA device visible; falling back to CPU.")
        use_gpu = False

    num_workers = max(1, min(num_workers, len(shards)))
    file_subsets = [shards[i::num_workers] for i in range(num_workers)]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    procs = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                i,
                file_subsets[i],
                input_path,
                output_path,
                first_k_dense_replace,
                num_hidden_layers,
                nvfp4_block_size,
                fp8_block_size,
                use_gpu,
                input_scale_map,
                return_dict,
                mtp_input_scale_map,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=None)
        if p.exitcode != 0:
            raise RuntimeError(f"worker pid={p.pid} exited with code {p.exitcode}")

    # Merge per-shard weight maps.
    weight_map: "dict[str, str]" = {}
    for shard_index in return_dict.values():
        weight_map.update(shard_index)
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    # ------------------------------------------------------------------
    # Copy auxiliary files (tokenizer, generation config, modeling code).
    # We deliberately skip the source config.json – it is rewritten below.
    # ------------------------------------------------------------------
    for fname in os.listdir(input_path):
        if fname == "config.json":
            continue
        if fname.endswith((".py", ".json", ".md", ".txt", ".jinja")):
            src = os.path.join(input_path, fname)
            dst = os.path.join(output_path, fname)
            if os.path.exists(dst) and fname == "model.safetensors.index.json":
                continue
            if os.path.exists(dst):
                continue
            print(f"cp {src} {dst}", flush=True)
            shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # Rewrite config.json:
    #   * top-level ``quantization_config`` describes the FP8-blockwise
    #     portion (matches GLM-5.1-FP8 layout so vLLM's fp8 backend picks
    #     it up correctly);
    #   * ``modules_to_not_convert`` lists every non-FP8 tensor (routed
    #     experts, layernorms, embed, lm_head, hc_*, router gate, biases,
    #     learnable_sink_param, ...).  Routed experts are excluded here
    #     and are instead described in the NVFP4 sidecar.
    # ------------------------------------------------------------------
    modules_to_not_convert = build_modules_to_not_convert(
        weight_map,
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=first_k_dense_replace,
    )

    new_config = dict(config)
    quant_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": list(fp8_block_size),
        "modules_to_not_convert": modules_to_not_convert,
    }
    if fp8_block_size[0] == -1 and fp8_block_size[1] == -1:
        # Tensor-wise fallback: drop weight_block_size.
        quant_config.pop("weight_block_size", None)
    new_config["quantization_config"] = quant_config
    # Report what the patterns actually did to the MTP routed experts rather
    # than assuming FP8: they may now be NVFP4, or left in BF16.
    _mtp_mode = classify_module("model.mtp_layers.0.mlp.experts")
    if _mtp_mode is not None and _mtp_mode == "fp8":
        new_config["mtp_quant_algo"] = "fp8"
    else:
        new_config.pop("mtp_quant_algo", None)
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)
    print(
        f"Wrote rewritten config.json (fp8 blockwise, "
        f"modules_to_not_convert={len(modules_to_not_convert)}).",
        flush=True,
    )

    # ------------------------------------------------------------------
    # NVFP4 sidecar (modelopt-style). Rename to hf_quant_config.json to
    # switch inference backend to NVFP4.
    # ------------------------------------------------------------------
    exclude_globs = build_nvfp4_exclude_globs(weight_map)
    sidecar = {
        "producer": {
            "name": "angelslim",
            "version": "hy4_nvfp4_weight_only_blockwise",
        },
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": nvfp4_block_size,
            "exclude_modules": exclude_globs,
        },
    }
    sidecar_path = os.path.join(output_path, "hf_quant_config.json.nvfp4")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(
        f"Wrote NVFP4 sidecar: {sidecar_path} " f"(exclude_modules={len(exclude_globs)})",
        flush=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (same style as the GLM-5 script).",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
        help="BF16 Hy4 HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Destination directory for the mixed NVFP4 / FP8-blockwise ckpt.",
    )
    parser.add_argument(
        "--moe_stats_json",
        type=str,
        default="",
        help="moe_expert_stats.json produced by tools/run_vllm_calibrate.py.",
    )
    parser.add_argument(
        "--mtp_moe_stats_json",
        type=str,
        default="",
        help="mtp_moe_expert_stats.json produced by tools/run_vllm_calibrate.py "
        "when run with `enable_mtp: true`. Supplies the per-tensor NVFP4 "
        "input_scale for the MTP routed experts; only needed while "
        "NVFP4_PATTERNS keeps the MTP experts in NVFP4.",
    )
    parser.add_argument("--nvfp4_block_size", type=int, default=16)
    parser.add_argument(
        "--fp8_block_size",
        type=int,
        nargs=2,
        default=(128, 128),
        help="FP8 blockwise tile size (rows cols). Use -1 -1 for tensor-wise.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Run quantisation on GPU (each worker pins to one device).",
    )
    # YAML-only fallbacks (declared for _yaml_args.apply_yaml_config).
    import argparse as _argparse

    parser.add_argument("--model-path", type=str, default="", help=_argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default="", help=_argparse.SUPPRESS)
    parser.add_argument("--output-nvfp4-hf-path", type=str, default="", help=_argparse.SUPPRESS)
    args = parser.parse_args()

    # Optional YAML config parsing (mirrors glm5_nvfp4_weight_only_blockwise).
    if args.config:
        import sys

        _tools_dir = os.path.dirname(os.path.abspath(__file__))
        if _tools_dir not in sys.path:
            sys.path.insert(0, _tools_dir)
        try:
            from _yaml_args import apply_yaml_config

            apply_yaml_config(parser, args)
        except ImportError:
            print("[warn] tools/_yaml_args.py not found; ignoring --config.")

    if not args.input_path and getattr(args, "model_path", ""):
        args.input_path = args.model_path
    if not args.output_path and getattr(args, "output_nvfp4_hf_path", ""):
        args.output_path = args.output_nvfp4_hf_path
    if not args.moe_stats_json and getattr(args, "output_dir", ""):
        cand = os.path.join(args.output_dir, "moe_expert_stats.json")
        if os.path.exists(cand):
            args.moe_stats_json = cand
    if not args.mtp_moe_stats_json and getattr(args, "output_dir", ""):
        cand = os.path.join(args.output_dir, "mtp_moe_expert_stats.json")
        if os.path.exists(cand):
            args.mtp_moe_stats_json = cand
    missing = [n for n in ("input_path", "output_path") if not getattr(args, n, "")]
    if missing:
        parser.error(
            "the following arguments are required (CLI or YAML): "
            + ", ".join("--" + m for m in missing)
        )

    print(args, flush=True)
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        nvfp4_block_size=args.nvfp4_block_size,
        fp8_block_size=tuple(args.fp8_block_size),
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        moe_stats_json=args.moe_stats_json or None,
        mtp_moe_stats_json=args.mtp_moe_stats_json or None,
    )
