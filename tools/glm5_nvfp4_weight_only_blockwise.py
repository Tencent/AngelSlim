"""
Stream-style NVFP4 weight-only re-quantization for GLM-5 FP8 checkpoints.

Why this script exists
----------------------
The standard ``tools/run.py -c <yaml>`` path needs to instantiate the model in
GPU memory first (``AutoModelForCausalLM.from_pretrained``), which OOMs on
GLM-5.1 (78 layers x 256 experts ~= 360GB FP8). For *weight-only* NVFP4 we
don't actually need a working forward graph at all – every weight tensor can
be re-quantized in isolation. This script does exactly that, processing the
checkpoint shard-by-shard using multiprocessing, mirroring the structure of
``tools/fp8_quant_blockwise.py``.

Output layout (to match the reference real checkpoint
``GLM-5.1-moe_nvfp4-dense_fp8_ue8m0``):

    expert MoE layers (model.layers.{L>=first_k_dense_replace}.mlp.experts.E):
        gate_proj.weight          uint8 packed FP4   (out, in/2)
        gate_proj.weight_scale    fp8 e4m3 per-block (out, in/16)
        gate_proj.weight_scale_2  fp32 scalar
        gate_proj.input_scale     fp32 scalar  ★ from --moe_stats_json
        (same for up_proj / down_proj)

    dense FP8 layers (attn / dense MLP layer 0..first_k_dense_replace-1 /
                      shared_experts / indexer projections):
        weight                    fp8 e4m3 (kept from source)
        scale                     uint8 ue8m0 per-block (out/128, in/128)
        ★ ue8m0 is computed by re-quantising the BF16 weight (recovered from
          source FP8 + weight_scale_inv) so that the FP8 representation
          remains valid w.r.t. the new ue8m0 scale (no overflow).

    other (router gate / layernorms / embed / lm_head / e_score_correction_bias):
        kept verbatim (BF16/FP32 as in the source).

After all shards are written, ``config.json`` is rewritten with
``quantization_config = source.quantization_config + {scale_fmt: "ue8m0"}``
(byte-identical ``modules_to_not_convert`` to the source FP8 ckpt – same 712
entries as the reference NVFP4 release).  A modelopt-style sidecar
``hf_quant_config.json.nvfp4`` is also dropped to match the reference layout.
"""

import json
import multiprocessing as mp
import os
import re
import shutil
from argparse import ArgumentParser

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

from angelslim.compressor.quant.core import weight_dequant

# ===========================================================================
# Layer-name selection helpers
# ===========================================================================
EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def parse_expert_weight(
    weight_name: str,
    first_k_dense_replace: int,
    num_hidden_layers: int | None = None,
):
    """Return ``(layer_idx, expert_idx, proj_name)`` if ``weight_name`` is an
    MoE expert projection at layer in
    ``[first_k_dense_replace, num_hidden_layers)``; otherwise return ``None``.

    The ``num_hidden_layers`` upper bound matters because GLM-5.1 has an
    extra **MTP layer** at index ``num_hidden_layers`` (78 in the FP8 release)
    that the reference NVFP4 checkpoint keeps in the dense FP8 path -- not
    NVFP4. If ``num_hidden_layers`` is ``None`` the upper bound is disabled
    (legacy behaviour).
    """
    m = EXPERT_PATTERN.match(weight_name)
    if m is None:
        return None
    layer_idx = int(m.group("layer"))
    if layer_idx < first_k_dense_replace:
        return None
    if num_hidden_layers is not None and layer_idx >= num_hidden_layers:
        return None
    return layer_idx, int(m.group("expert")), m.group("proj")


def is_expert_weight(
    weight_name: str,
    first_k_dense_replace: int,
    num_hidden_layers: int | None = None,
) -> bool:
    return parse_expert_weight(weight_name, first_k_dense_replace, num_hidden_layers) is not None


# ===========================================================================
# NVFP4 weight-only quantization (CPU/GPU agnostic)
# ===========================================================================
# E2M1 reference table (values that FP4 can represent), copied from
# angelslim/compressor/quant/modules/helper_layer.py::NVFP4QDQModule. Kept
# in module scope so each worker reuses the same constants.
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_FP8_FINFO = torch.finfo(torch.float8_e4m3fn)
# Maximum absolute value representable by FP8 E4M3 (used both in the NVFP4
# global-scale and in the dense-FP8 ue8m0 re-quantisation).
_FP8_E4M3_MAX = float(_FP8_FINFO.max)  # 448.0


def _cast_fp4(weight_abs_signed: torch.Tensor) -> torch.Tensor:
    """Cast a signed float tensor (already scaled into the fp4 dynamic range)
    to packed-but-not-yet-packed uint4 codes. Output dtype is uint8 with
    values in [0, 15] (sign bit + 3 magnitude bits).
    """
    device = weight_abs_signed.device
    bounds = _E2M1_BOUNDS.to(device)
    # Mask to perform "round half to even"-style rounding at odd indices
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8, device=device)
    mask = mask.expand([*weight_abs_signed.shape, 7])

    sign_bit = (weight_abs_signed < 0).to(torch.uint8)
    weight_abs = weight_abs_signed.abs()
    ord_ = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    round_ = torch.any((weight_abs.unsqueeze(-1) == bounds) * mask, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord_ + round_).to(torch.uint8)
    return fp4_val


def nvfp4_quantize_weight_only(
    weight_bf16: torch.Tensor, block_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 weight-only quantization following AngelSlim's convention.

    Args:
        weight_bf16: a 2-D weight tensor (typically BF16 or FP32). Last dim
            must be divisible by ``block_size``.
        block_size: NVFP4 group size along the last dimension (default 16).

    Returns:
        (packed_weight, weight_scale, weight_scale_2) where:
          * ``packed_weight`` is uint8 of shape (n, k // 2);
          * ``weight_scale`` is FP8 e4m3 of shape (n, k // block_size);
          * ``weight_scale_2`` is FP32 scalar = ``W.abs().max() / 6 / 448``.
    """
    assert weight_bf16.dim() == 2, f"expected 2-D tensor, got {tuple(weight_bf16.shape)}"
    n, k = weight_bf16.shape
    assert k % block_size == 0, (
        f"last dim {k} not divisible by block_size {block_size}; "
        "GLM-5 expert dims are multiples of 128 so this should never trigger."
    )

    weight_orig = weight_bf16.detach()
    if weight_orig.dtype != torch.bfloat16:
        weight_orig = weight_orig.to(torch.bfloat16)

    # ---- per-tensor weight_scale_2 ----
    amax = weight_orig.float().abs().max()
    weight_scale_2 = (amax / 6.0 / 448.0).to(torch.float32)
    if not torch.isfinite(weight_scale_2) or weight_scale_2.item() == 0.0:
        weight_scale_2 = torch.tensor(1.0e-6, dtype=torch.float32, device=weight_orig.device)

    # ---- per-block weight_scale (fp8) ----
    blocks_view = weight_orig.view(n, k // block_size, block_size)
    per_block_amax = blocks_view.abs().amax(dim=-1).float()
    per_block_scale = per_block_amax / 6.0
    q_per_block_scale = per_block_scale / weight_scale_2
    q_per_block_scale[per_block_scale == 0] = 1.0
    q_per_block_scale = q_per_block_scale.clamp(min=_FP8_FINFO.min, max=_FP8_FINFO.max)
    weight_scale_fp8 = q_per_block_scale.to(torch.float8_e4m3fn)

    # ---- quantize weight to FP4 ----
    eff_scale = (weight_scale_fp8.to(torch.float32) * weight_scale_2).unsqueeze(-1)
    scaled = blocks_view / eff_scale
    scaled = scaled.view(n, k)
    fp4_codes = _cast_fp4(scaled)
    packed = (fp4_codes[..., 1::2] << 4) | fp4_codes[..., 0::2]

    return packed.contiguous(), weight_scale_fp8.contiguous(), weight_scale_2.contiguous()


# ===========================================================================
# Dense FP8 ue8m0 re-quantization
# ===========================================================================
def fp8_requantize_to_ue8m0(
    bf16_weight: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize a BF16 weight tensor as FP8 E4M3 with **ue8m0** per-block
    scales (every scale value is a power of two, encoded as ``uint8`` with
    ``scale_fp32 = 2 ** (uint8 - 127)``).

    The output matches the format used by
    ``GLM-5.1-moe_nvfp4-dense_fp8_ue8m0``::

        weight: float8_e4m3fn       shape (M, N)
        scale : uint8 (ue8m0)       shape (ceil(M/block_size), ceil(N/block_size))

    Note: GLM-5 has at least one tensor (``self_attn.kv_a_proj_with_mqa``,
    shape ``(576, 6144)``) whose row count is **not** a multiple of
    ``block_size`` -- the source ckpt and the reference NVFP4 release both
    handle that with a *padded* block layout (576 -> 5 row-blocks, the last
    block covering only 64 valid rows). This implementation matches that
    convention.

    Convention: ``bf16 = fp8 * scale``  (note: this is a *forward* scale, not
    ``scale_inv``; that matches the on-disk layout of the reference real
    checkpoint, not the original GLM-5.1-FP8 source layout that uses
    ``weight_scale_inv``).

    Algorithm:
        1. compute per-block BF16 amax (with the partial last block treated
           correctly via a masked padded view);
        2. ideal block scale = ``amax / 448``  (so |fp8|<=448 after divide);
        3. ue8m0 quantise the scale by **rounding the exponent up** (so the
           fp8 representation never overflows ±448);
        4. divide the BF16 block by the rounded scale and cast to FP8 E4M3.
           The output ``weight`` keeps the original (M, N) shape; only the
           ``scale`` grid is ceil-rounded.
    """
    assert bf16_weight.dim() == 2, f"expected 2-D tensor, got {tuple(bf16_weight.shape)}"
    M, N = bf16_weight.shape
    bs = block_size
    m_blocks = (M + bs - 1) // bs
    n_blocks = (N + bs - 1) // bs
    pad_m = m_blocks * bs - M
    pad_n = n_blocks * bs - N

    w = bf16_weight.detach().to(torch.float32)
    if pad_m or pad_n:
        # Right-pad with zeros so we can reshape into uniform (bs, bs) tiles.
        # Zeros do not affect the per-block amax computation, and the padded
        # output region is sliced away before returning.
        w_padded = torch.nn.functional.pad(w, (0, pad_n, 0, pad_m), value=0.0)
    else:
        w_padded = w
    Mp, Np = w_padded.shape

    # Reshape to (m_blocks, bs, n_blocks, bs) to compute per-block amax.
    blocks = w_padded.view(m_blocks, bs, n_blocks, bs)
    amax = blocks.abs().amax(dim=(1, 3))  # (m_blocks, n_blocks)

    # Avoid log2(0): treat all-zero blocks as scale 1.0 (uint8 == 127).
    zero_mask = amax == 0
    safe_amax = torch.where(zero_mask, torch.ones_like(amax), amax)
    ideal_scale = safe_amax / _FP8_E4M3_MAX  # forward scale: bf16 = fp8 * scale
    # Round the exponent UP so the resulting fp8 magnitudes stay <= 448.
    exp = torch.ceil(torch.log2(ideal_scale))  # may be negative
    # ue8m0 with bias 127, clamp to representable range [0, 255].
    ue8m0 = (exp + 127.0).clamp_(0.0, 255.0).to(torch.uint8)
    # Force zero-blocks to bias=127 (scale = 1.0) – inert sentinel.
    ue8m0[zero_mask] = torch.tensor(127, dtype=torch.uint8, device=ue8m0.device)

    # Recover the rounded scale as fp32 and quantise the weight.
    rounded_scale = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=w.device),
        ue8m0.to(torch.float32) - 127.0,
    )  # (m_blocks, n_blocks)
    # Broadcast to (m_blocks, 1, n_blocks, 1) so it divides every block element.
    eff_scale = rounded_scale.view(m_blocks, 1, n_blocks, 1)
    fp8_blocks = (blocks / eff_scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    fp8_padded = fp8_blocks.view(Mp, Np)
    # Slice off the padded rows/cols so the on-disk weight retains its
    # original (M, N) shape.
    if pad_m or pad_n:
        fp8_padded = fp8_padded[:M, :N]
    fp8 = fp8_padded.to(torch.float8_e4m3fn)
    return fp8.contiguous(), ue8m0.contiguous()


# ===========================================================================
# FP8 -> BF16 dequantisation (deepseek-style 128x128 weight_scale_inv)
# ===========================================================================
def _bf16_dequant_fp8(weight_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize an FP8 block-wise weight to BF16 using the deepseek-style
    128x128 inverse-scale tensor. Always returns a contiguous BF16 tensor.

    ``weight_dequant`` outputs ``torch.get_default_dtype()`` so we set BF16
    for the duration of the call.
    """
    assert weight_fp8.dtype == torch.float8_e4m3fn
    assert (
        weight_fp8.dim() == 2
    ), f"_bf16_dequant_fp8 expects 2-D weight, got {tuple(weight_fp8.shape)}"
    weight_fp8 = weight_fp8.contiguous()
    scale_inv = scale_inv.to(torch.float32)

    M, N = weight_fp8.shape
    BS = 128
    m_blocks = (M + BS - 1) // BS
    n_blocks = (N + BS - 1) // BS
    if scale_inv.dim() == 0:
        scale_inv = scale_inv.view(1, 1)
    elif scale_inv.dim() == 1:
        if scale_inv.numel() == m_blocks and n_blocks == 1:
            scale_inv = scale_inv.view(m_blocks, 1)
        elif scale_inv.numel() == n_blocks and m_blocks == 1:
            scale_inv = scale_inv.view(1, n_blocks)
        else:
            raise ValueError(
                f"Cannot infer 2-D scale layout for shape {tuple(scale_inv.shape)} "
                f"with weight {tuple(weight_fp8.shape)} and block_size={BS}"
            )
    elif scale_inv.dim() != 2:
        raise ValueError(
            f"Unexpected scale_inv shape {tuple(scale_inv.shape)} for weight "
            f"{tuple(weight_fp8.shape)}"
        )
    scale_inv = scale_inv.contiguous()

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        out = weight_dequant(weight_fp8, scale_inv, block_size=BS)
    finally:
        torch.set_default_dtype(prev_dtype)
    return out.to(torch.bfloat16).contiguous()


# ===========================================================================
# moe_expert_stats.json -> per-layer input_scale lookup
# ===========================================================================
# Pattern for keys produced by run_vllm_calibrate's scale-only mode:
#   model.layers.{L}.mlp.experts.{stage}.scale_amax
# where stage is "gate_up_proj" or "down_proj".  The corresponding
# ``input_scale`` for every expert in that layer is::
#
#     input_scale = scale_amax_max / 6
#
# because the calibration records ``scale_amax_max`` = max over a forward of
# ``activation_amax / 448``  (the FP8-E4M3 dynamic-quant scale), so the
# activation amax is ``scale_amax_max * 448``, and the modelopt NVFP4 global
# input scale is ``activation_amax / (6 * 448) = scale_amax_max / 6``.
_MOE_STATS_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<stage>gate_up_proj|down_proj)\.scale_amax$"
)


def load_input_scales_from_stats(
    stats_json_path: str,
) -> dict[tuple[int, str], float]:
    """Load the calibration JSON and return a mapping
    ``(layer_idx, stage) -> input_scale (float32)`` where ``stage`` is one of
    ``"gate_up_proj"`` (shared by gate_proj and up_proj) or ``"down_proj"``.
    """
    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    out: dict[tuple[int, str], float] = {}
    for k, v in stats.items():
        m = _MOE_STATS_KEY_RE.match(k)
        if m is None:
            continue
        if not isinstance(v, dict) or "max" not in v:
            continue
        scale_max = float(v["max"])
        if not (scale_max > 0.0):
            # ``inf``, ``-inf``, ``nan`` or ``0`` -> treat as missing.
            continue
        layer_idx = int(m.group("layer"))
        stage = m.group("stage")
        out[(layer_idx, stage)] = scale_max / 6.0
    return out


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
    fp8_block_size: int,
    use_gpu: bool,
    input_scale_map: dict[tuple[int, str], float],
    quantize_dense_fp8: bool,
):
    """Process a single safetensor shard.

    Returns the local weight_map fragment for this shard (key -> file_name).
    """
    state_dict = {}
    index = {}
    device = f"cuda:{rank}" if use_gpu else "cpu"

    src = os.path.join(input_path, file_name)
    with safe_open(src, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        weight_keys = [k for k in all_keys if k.endswith(".weight")]
        scale_keys = set(k for k in all_keys if k.endswith(".weight_scale_inv"))
        non_pair_keys = [
            k
            for k in all_keys
            if not k.endswith(".weight") and not k.endswith(".weight_scale_inv")
        ]

        # 1) Pass-through everything that is neither weight nor scale_inv.
        for k in non_pair_keys:
            state_dict[k] = f.get_tensor(k)
            index[k] = file_name

        # 2) Process weight tensors.
        for weight_name in weight_keys:
            base = weight_name[: -len(".weight")]
            scale_name = base + ".weight_scale_inv"
            has_scale = scale_name in scale_keys
            weight = f.get_tensor(weight_name)

            # ----------------------------------------------------------------
            # Case A: expert projection -> dequant FP8 -> NVFP4 (+ input_scale)
            # ----------------------------------------------------------------
            parsed = parse_expert_weight(weight_name, first_k_dense_replace, num_hidden_layers)
            if parsed is not None:
                layer_idx, expert_idx, proj_name = parsed
                if not has_scale:
                    raise RuntimeError(
                        f"Expected FP8 + weight_scale_inv for expert key "
                        f"{weight_name}, but no scale found."
                    )
                scale_inv = f.get_tensor(scale_name)
                if use_gpu:
                    weight = weight.to(device)
                    scale_inv = scale_inv.to(device)
                bf16_w = _bf16_dequant_fp8(weight, scale_inv)
                packed, w_scale, w_scale_2 = nvfp4_quantize_weight_only(
                    bf16_w, block_size=nvfp4_block_size
                )

                state_dict[weight_name] = packed.cpu()
                state_dict[base + ".weight_scale"] = w_scale.cpu()
                state_dict[base + ".weight_scale_2"] = w_scale_2.cpu()
                index[weight_name] = file_name
                index[base + ".weight_scale"] = file_name
                index[base + ".weight_scale_2"] = file_name

                # Inject per-tensor input_scale derived from calibration.
                # gate_proj and up_proj share the gate_up_proj statistic.
                stat_stage = "down_proj" if proj_name == "down_proj" else "gate_up_proj"
                lookup_key = (layer_idx, stat_stage)
                if lookup_key in input_scale_map:
                    input_scale_val = input_scale_map[lookup_key]
                else:
                    # Sentinel: ``1.0`` collapses ``alpha = input_scale *
                    # weight_scale_2`` to just ``weight_scale_2`` at runtime,
                    # which means the activation is treated as if its amax
                    # equals ``6 * 448 = 2688`` -- nearly never tight, so
                    # accuracy may suffer.  Surface a clear warning so the
                    # user knows calibration data was incomplete.
                    print(
                        f"[warn] no input_scale stat for layer={layer_idx} "
                        f"stage={stat_stage}; falling back to 1.0"
                    )
                    input_scale_val = 1.0
                state_dict[base + ".input_scale"] = torch.tensor(
                    input_scale_val, dtype=torch.float32
                )
                index[base + ".input_scale"] = file_name

                del weight, scale_inv, bf16_w, packed, w_scale, w_scale_2
                if use_gpu:
                    torch.cuda.empty_cache()
                continue

            # ----------------------------------------------------------------
            # Case B: non-expert FP8 weight
            # ----------------------------------------------------------------
            if has_scale:
                scale_inv = f.get_tensor(scale_name)
                if quantize_dense_fp8:
                    # Dense path matching the real reference checkpoint:
                    # keep FP8, but re-quantise the per-block scale to
                    # ue8m0 (uint8) and store as ``...scale`` (forward scale,
                    # not scale_inv).
                    if use_gpu:
                        weight = weight.to(device)
                        scale_inv = scale_inv.to(device)
                    bf16_w = _bf16_dequant_fp8(weight, scale_inv)
                    fp8_w, ue8m0_scale = fp8_requantize_to_ue8m0(bf16_w, block_size=fp8_block_size)
                    state_dict[weight_name] = fp8_w.cpu()
                    state_dict[base + ".scale"] = ue8m0_scale.cpu()
                    index[weight_name] = file_name
                    index[base + ".scale"] = file_name
                    del weight, scale_inv, bf16_w, fp8_w, ue8m0_scale
                else:
                    # Legacy path: dequant to BF16 and drop the scale.
                    if use_gpu:
                        weight = weight.to(device)
                        scale_inv = scale_inv.to(device)
                    bf16_w = _bf16_dequant_fp8(weight, scale_inv)
                    state_dict[weight_name] = bf16_w.cpu()
                    index[weight_name] = file_name
                    del weight, scale_inv, bf16_w
                if use_gpu:
                    torch.cuda.empty_cache()
                continue

            # ----------------------------------------------------------------
            # Case C: already-non-FP8 weight (bf16/fp32/embedding/router/etc.)
            # ----------------------------------------------------------------
            state_dict[weight_name] = weight
            index[weight_name] = file_name

    dst = os.path.join(output_path, file_name)
    save_file(state_dict, dst)
    del state_dict
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
    quantize_dense_fp8,
    return_dict,
):
    # In a ``spawn`` child the CUDA context is uninitialised at start; an
    # implicit ``.to("cuda:R")`` may succeed but leave the runtime in an
    # inconsistent state for subsequent triton kernel launches ("Pointer
    # argument cannot be accessed from Triton (cpu tensor?)"). Forcing the
    # device early triggers the lazy init under our control.
    world_size = max(1, torch.cuda.device_count())
    rank = i % world_size
    if use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(rank)
        # Touch a tensor so the CUDA primary context is fully initialised
        # before any triton kernel is launched.
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
            quantize_dense_fp8=quantize_dense_fp8,
        )
        return_dict[file_name] = idx


# ===========================================================================
# Top-level driver
# ===========================================================================
def collect_shards(input_path: str) -> list[str]:
    idx_file = os.path.join(input_path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        with open(idx_file) as f:
            wm = json.load(f)["weight_map"]
        return sorted(set(wm.values()))
    if os.path.exists(os.path.join(input_path, "model.safetensors")):
        return ["model.safetensors"]
    raise FileNotFoundError(f"No safetensors found under {input_path}")


def build_nvfp4_exclude_globs(
    num_hidden_layers: int, first_k_dense_replace: int, num_nextn_predict_layers: int
) -> list[str]:
    """Build the modelopt-style ``exclude_modules`` glob list that mirrors
    the reference real checkpoint's ``hf_quant_config.json.nvfp4``::

        lm_head
        model.layers.0*           # for every layer in [0, first_k_dense_replace)
                                  # (covers dense MLP + attn + indexer + layernorms)
        model.layers.{L}.mlp.shared_experts*   # for every MoE layer
        model.layers.{L}.self_attn*            # for every MoE layer
        model.layers.{L}.eh_proj* ... etc      # for the MTP layer (last one)
    """
    out = ["lm_head"]
    # Dense layers (0..first_k_dense_replace-1): exclude entirely with one glob.
    for L in range(first_k_dense_replace):
        out.append(f"model.layers.{L}.*")
    # MoE layers: exclude shared_experts and self_attn (router stays excluded
    # as well via ``modules_to_not_convert`` in the parent fp8 config; for
    # the nvfp4 view only experts.* are actually targets, everything else is
    # excluded). The MTP layer at index ``num_hidden_layers`` is handled
    # below.
    for L in range(first_k_dense_replace, num_hidden_layers):
        out.append(f"model.layers.{L}.mlp.shared_experts*")
        out.append(f"model.layers.{L}.self_attn*")
    # MTP layer (model.layers.{num_hidden_layers}) – only when present.
    if num_nextn_predict_layers > 0:
        for d in range(num_nextn_predict_layers):
            L = num_hidden_layers + d
            out.append(f"model.layers.{L}.*")
    return sorted(set(out))


def main(
    input_path: str,
    output_path: str,
    nvfp4_block_size: int,
    fp8_block_size: int,
    num_workers: int,
    use_gpu: bool,
    moe_stats_json: str | None,
    quantize_dense_fp8: bool,
):
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(input_path, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    first_k_dense_replace = int(config.get("first_k_dense_replace", 0))
    num_hidden_layers = int(config.get("num_hidden_layers", 0))
    num_nextn_predict_layers = int(config.get("num_nextn_predict_layers", 0))
    print(
        f"first_k_dense_replace = {first_k_dense_replace}, "
        f"num_hidden_layers = {num_hidden_layers}, "
        f"num_nextn_predict_layers = {num_nextn_predict_layers}"
    )

    if "quantization_config" not in config:
        raise AssertionError(
            "Input checkpoint has no quantization_config – this script expects "
            "an FP8 GLM-5 checkpoint. Use it on the FP8 release."
        )
    src_qcfg = config["quantization_config"]
    if str(src_qcfg.get("quant_method", "")).lower() != "fp8":
        raise AssertionError(
            f"Expected source quant_method='fp8', got {src_qcfg.get('quant_method')!r}"
        )

    # Load calibration stats → per-layer input_scale.
    if moe_stats_json:
        if not os.path.exists(moe_stats_json):
            raise FileNotFoundError(f"--moe_stats_json not found: {moe_stats_json}")
        input_scale_map = load_input_scales_from_stats(moe_stats_json)
        print(
            f"Loaded {len(input_scale_map)} (layer, stage) input_scale entries "
            f"from {moe_stats_json}"
        )
    else:
        input_scale_map = {}
        print(
            "[warn] --moe_stats_json not provided; every expert will get "
            "input_scale=1.0 (model will load but accuracy is undefined)."
        )

    shards = collect_shards(input_path)
    print(f"Found {len(shards)} safetensor shards to process.")

    if use_gpu and torch.cuda.device_count() == 0:
        print("[warn] --use_gpu requested but no CUDA device visible, falling back to CPU.")
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
                quantize_dense_fp8,
                return_dict,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=None)
        if p.exitcode != 0:
            raise RuntimeError(f"worker pid={p.pid} exited with code {p.exitcode}")

    # Merge per-shard weight maps.
    weight_map: dict[str, str] = {}
    for shard_index in return_dict.values():
        weight_map.update(shard_index)

    out_idx = {"metadata": {}, "weight_map": weight_map}
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(out_idx, f, indent=2)

    # Copy auxiliary files (tokenizer, generation config, modeling code, etc.)
    for fname in os.listdir(input_path):
        if fname.endswith((".py", ".json", ".md", ".txt", ".jinja")):
            src = os.path.join(input_path, fname)
            dst = os.path.join(output_path, fname)
            if os.path.exists(dst) and fname == "model.safetensors.index.json":
                continue
            if os.path.exists(dst):
                continue
            print(f"cp {src} {dst}")
            shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # Rewrite config.json – mirrors the reference real checkpoint:
    #   * top-level fp8 quantization_config (same modules_to_not_convert as
    #     the source ckpt), with ``scale_fmt`` upgraded to ``"ue8m0"`` when
    #     ``quantize_dense_fp8`` is enabled;
    #   * the NVFP4 description goes into ``hf_quant_config.json.nvfp4``.
    # ------------------------------------------------------------------
    new_config_path = os.path.join(output_path, "config.json")
    with open(new_config_path, encoding="utf-8") as f:
        out_config = json.load(f)

    # Preserve the exact ordering of the source quantization_config and
    # only insert/overwrite ``scale_fmt`` when ue8m0 is in use.
    new_qcfg = dict(out_config.get("quantization_config", {}))
    if quantize_dense_fp8:
        new_qcfg["scale_fmt"] = "ue8m0"
    else:
        # Dense path is BF16 -> drop the dense fp8 metadata that no longer
        # applies; the field name ``modules_to_not_convert`` is irrelevant in
        # a pure-bf16 checkpoint, but we keep it since vLLM tolerates it.
        new_qcfg.pop("scale_fmt", None)
    out_config["quantization_config"] = new_qcfg

    with open(new_config_path, "w", encoding="utf-8") as f:
        json.dump(out_config, f, indent=2)
    print(
        f"Wrote rewritten config.json (quant_method={new_qcfg.get('quant_method')}, "
        f"scale_fmt={new_qcfg.get('scale_fmt')!r}, "
        f"modules_to_not_convert={len(new_qcfg.get('modules_to_not_convert', []))})."
    )

    # ------------------------------------------------------------------
    # NVFP4 sidecar (modelopt format) – matches the reference's
    # ``hf_quant_config.json.nvfp4`` layout, named with the ``.nvfp4``
    # suffix so it does NOT shadow the active fp8 ``hf_quant_config.json``.
    # The user can rename it to ``hf_quant_config.json`` when switching the
    # backend to nvfp4 inference.
    # ------------------------------------------------------------------
    exclude_globs = build_nvfp4_exclude_globs(
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=first_k_dense_replace,
        num_nextn_predict_layers=num_nextn_predict_layers,
    )
    nvfp4_sidecar = {
        "producer": {
            "name": "angelslim",
            "version": "glm5_nvfp4_weight_only_blockwise",
        },
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": nvfp4_block_size,
            "exclude_modules": exclude_globs,
        },
    }
    sidecar_path = os.path.join(output_path, "hf_quant_config.json.nvfp4")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(nvfp4_sidecar, f, indent=2)
    print(f"Wrote NVFP4 sidecar: {sidecar_path} " f"(exclude_modules={len(exclude_globs)})")


if __name__ == "__main__":
    parser = ArgumentParser()
    # ------------------------------------------------------------------
    # YAML config support (matches tools/fp8_quant_with_vllm_activation.py
    # style). The same unified PTQ YAML can drive both stage 1
    # (run_vllm_calibrate.py) and stage 2 (this script). Explicit CLI
    # flags still take final precedence over YAML values.
    # ------------------------------------------------------------------
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML config file path. Values override argparse defaults; "
        "explicit CLI flags still take final precedence.",
    )
    # All these are normally driven by the YAML file, but kept as CLI
    # flags for ad-hoc overrides / backward compatibility.
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
        help="Path to the source FP8 GLM-5 checkpoint. "
        "Falls back to YAML ``model_path`` if unset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Where to write the NVFP4 weight-only checkpoint. "
        "Falls back to YAML ``output_nvfp4_hf_path`` if unset.",
    )
    parser.add_argument(
        "--moe_stats_json",
        type=str,
        default="",
        help="Path to the calibration JSON produced by "
        "run_vllm_calibrate.py (e.g. moe_expert_stats.json). "
        "Used to populate per-expert ``input_scale``. "
        "Falls back to ``${output_dir}/moe_expert_stats.json`` "
        "when set via YAML.",
    )
    parser.add_argument(
        "--nvfp4_block_size",
        type=int,
        default=16,
        help="NVFP4 group size along the last dim (default 16).",
    )
    parser.add_argument(
        "--fp8_block_size", type=int, default=128, help="Dense FP8 ue8m0 block size (default 128)."
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Run dequant + quant on GPU (uses triton). " "If unset, runs entirely on CPU.",
    )
    parser.add_argument(
        "--no_quantize_dense_fp8",
        action="store_true",
        help="Legacy path: dequantise dense layers to BF16 "
        "instead of re-quantising to FP8 ue8m0. The "
        "default (and recommended) path produces "
        "FP8 ue8m0 dense, matching the reference "
        "GLM-5.1-moe_nvfp4-dense_fp8_ue8m0 checkpoint.",
    )
    # YAML-only fallbacks (declared so apply_yaml_config can populate
    # them without WARNING: unknown keys). Hidden from --help.
    import argparse as _argparse

    parser.add_argument("--model-path", type=str, default="", help=_argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default="", help=_argparse.SUPPRESS)
    parser.add_argument(
        "--output-nvfp4-hf-path",
        type=str,
        default="",
        help=_argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # Lazy-import _yaml_args (sibling module in tools/). Done here instead of
    # at module top so flake8 doesn't trip on a sys.path mutation between
    # imports.
    import sys

    _tools_dir = os.path.dirname(os.path.abspath(__file__))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from _yaml_args import apply_yaml_config

    apply_yaml_config(parser, args)

    # Path fallbacks: when running with the unified GLM-5 YAML, stage 2
    # reuses stage 1's ``model_path`` as the FP8 input dir, and the per-
    # expert stats live at ``${output_dir}/moe_expert_stats.json``. The
    # destination is taken from the dedicated ``output_nvfp4_hf_path`` key.
    if not getattr(args, "input_path", "") and getattr(args, "model_path", ""):
        args.input_path = args.model_path
        print(
            f"[yaml-config] input_path not set; falling back to " f"model_path={args.input_path!r}"
        )
    if not getattr(args, "output_path", "") and getattr(args, "output_nvfp4_hf_path", ""):
        args.output_path = args.output_nvfp4_hf_path
        print(
            f"[yaml-config] output_path not set; falling back to "
            f"output_nvfp4_hf_path={args.output_path!r}"
        )
    if not getattr(args, "moe_stats_json", "") and getattr(args, "output_dir", ""):
        args.moe_stats_json = os.path.join(args.output_dir, "moe_expert_stats.json")
        print(
            f"[yaml-config] moe_stats_json not set; falling back to "
            f"${{output_dir}}/moe_expert_stats.json={args.moe_stats_json!r}"
        )

    # Validate required paths (may come from CLI or YAML).
    missing = [name for name in ("input_path", "output_path") if not getattr(args, name, "")]
    if missing:
        parser.error(
            "the following arguments are required (via CLI or YAML config): "
            + ", ".join("--" + n for n in missing)
        )

    print(args)

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        nvfp4_block_size=args.nvfp4_block_size,
        fp8_block_size=args.fp8_block_size,
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        moe_stats_json=args.moe_stats_json or None,
        quantize_dense_fp8=not args.no_quantize_dense_fp8,
    )
