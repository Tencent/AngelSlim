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


"""HPC (C++ extension) backend for Stem sparse prefill.

Provides three execution paths:

* **bf16 dense** — calls ``hpc.attention_prefill_bf16`` directly.
* **fp8 varlen** — Triton scoring → ``hpc.stem_tpd`` mask →
  ``hpc.attention_blocksparse_prefill_fp8``.
* **fp8 paged** — ``hpc.stem_paged_kv`` mask →
  ``hpc.attention_with_kvcache_blocksparse_prefill_fp8``.
"""

from __future__ import annotations

import hpc
import torch

from .torch_impl import _compute_triton_block_logits, stem_forward_torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FP8_DTYPE = torch.float8_e4m3fn
HPC_FP8_BLOCK_SIZE = 128
HPC_INITIAL_BLOCKS = 4
HPC_WINDOW_SIZE = 4


# ---------------------------------------------------------------------------
# Tensor packing / quantisation helpers
# ---------------------------------------------------------------------------


def _pack_bhld_to_varlen(x: torch.Tensor) -> torch.Tensor:
    """Reshape ``(B, H, L, D)`` → ``(B*L, H, D)`` for variable-length HPC kernels."""
    B, H, L, D = x.shape
    return x.transpose(1, 2).reshape(B * L, H, D).contiguous()


def _uniform_seq_metadata(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(seqlens, cu_seqlens)`` for a uniform-length batch.

    Parameters
    ----------
    batch_size : int
    seq_len : int
    device : torch.device

    Returns
    -------
    seqlens : torch.Tensor  — ``(batch_size,)``, all equal to *seq_len*.
    cu_seqlens : torch.Tensor — ``(batch_size + 1,)``, cumulative offsets.
    """
    seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        step=seq_len,
        dtype=torch.int32,
        device=device,
    )
    return seqlens, cu_seqlens


def _quantize_per_tensor_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor FP8 (E4M3) quantisation.

    Returns
    -------
    x_fp8 : torch.Tensor   — quantised tensor (same shape, dtype ``float8_e4m3fn``).
    scale  : torch.Tensor   — scalar scale factor, shape ``(1,)``.
    """
    fp8_max = torch.finfo(FP8_DTYPE).max
    scale = x.abs().amax().float().clamp(min=1e-12) / fp8_max
    x_fp8 = torch.clamp(x.float() / scale, min=-fp8_max, max=fp8_max).to(FP8_DTYPE)
    return x_fp8.contiguous(), scale.view(1)


def _quantize_query_for_paged_fp8(
    query_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise the query tensor for the paged FP8 attention kernel.

    Uses HPC's per-token-group quantisation (``group_size = head_dim``).

    Args:
        query_states: Query tensor of shape ``(B, H, L, D)``.

    Returns:
        A tuple of ``(q_fp8, q_scale)`` where *q_fp8* has shape
        ``(B*L, H, D)`` in ``float8_e4m3fn`` and *q_scale* has shape
        ``(B, H, L_padded)`` with per-token scales.

    Raises:
        ValueError: If ``head_dim != 128``.
    """
    B, H, L, D = query_states.shape
    if D != HPC_FP8_BLOCK_SIZE:
        raise ValueError(f"HPC fp8 query quant only supports head_dim=128, got {D}.")

    q_rows = query_states.transpose(1, 2).reshape(B * L, H * D).contiguous()
    q_fp8_rows, q_scale_rows = hpc.quant.per_token_group_fp8_quant(q_rows, group_size=D)

    q_fp8 = q_fp8_rows.view(B * L, H, D).contiguous()
    q_scale = q_scale_rows.view(B, L, H).permute(0, 2, 1).contiguous()

    # Pad the sequence dimension to a multiple of the block size.
    padded_L = ((L + HPC_FP8_BLOCK_SIZE - 1) // HPC_FP8_BLOCK_SIZE) * HPC_FP8_BLOCK_SIZE
    if padded_L != L:
        q_scale = torch.nn.functional.pad(q_scale, (0, padded_L - L))
    return q_fp8, q_scale


def _pack_paged_cache(
    x_varlen: torch.Tensor,
    batch_size: int,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a varlen KV tensor into paged-cache layout.

    Parameters
    ----------
    x_varlen : torch.Tensor — ``(B*L, H, D)``
    batch_size, seq_len, block_size : int

    Returns
    -------
    cache      : torch.Tensor — ``(total_blocks, block_size, H, D)``
    kv_indices : torch.Tensor — ``(B, blocks_per_seq)``  block-table indices.
    """
    _, H, D = x_varlen.shape
    padded_L = ((seq_len + block_size - 1) // block_size) * block_size
    blocks_per_seq = padded_L // block_size

    x_seq = x_varlen.view(batch_size, seq_len, H, D)
    if padded_L > seq_len:
        pad = torch.zeros(
            (batch_size, padded_L - seq_len, H, D),
            dtype=x_varlen.dtype,
            device=x_varlen.device,
        )
        x_seq = torch.cat([x_seq, pad], dim=1)

    cache = (
        x_seq.view(batch_size, blocks_per_seq, block_size, H, D)
        .reshape(batch_size * blocks_per_seq, block_size, H, D)
        .contiguous()
    )
    kv_indices = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=x_varlen.device
    ).view(batch_size, blocks_per_seq)
    return cache, kv_indices


# ---------------------------------------------------------------------------
# Head-repeat helper (GQA → full Q heads)
# ---------------------------------------------------------------------------


def _repeat_to_q_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Repeat KV heads to match the query head count (GQA support).

    Parameters
    ----------
    x : torch.Tensor — ``(B, H_kv, L, D)``
    num_q_heads : int — target number of heads (``H_q``).

    Returns
    -------
    torch.Tensor — ``(B, H_q, L, D)``
    """
    B, H_kv, L, D = x.shape
    if num_q_heads == H_kv:
        return x
    if num_q_heads % H_kv != 0:
        raise ValueError(f"Cannot repeat kv heads from {H_kv} to {num_q_heads}.")
    n_rep = num_q_heads // H_kv
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, L, D)
    return x.reshape(B, num_q_heads, L, D)


# ---------------------------------------------------------------------------
# Fallback to the pure-torch backend
# ---------------------------------------------------------------------------


def _fallback_to_torch(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    prefill_kwargs: dict,
) -> torch.Tensor:
    """Fall back to :func:`stem_forward_torch` after repeating KV heads."""
    H_q = query_states.shape[1]
    if key_states.shape[1] != H_q:
        key_states = _repeat_to_q_heads(key_states, H_q)
    if value_states.shape[1] != H_q:
        value_states = _repeat_to_q_heads(value_states, H_q)
    return stem_forward_torch(query_states, key_states, value_states, prefill_kwargs)


# ---------------------------------------------------------------------------
# Runtime parameter extraction
# ---------------------------------------------------------------------------


def _stem_runtime_params(config: dict, layer_idx: int) -> dict:
    """Extract Stem runtime hyper-parameters from the user config dict."""
    alpha_cfg = config.get("stem_alpha", 1.0)
    alpha = alpha_cfg[layer_idx] if isinstance(alpha_cfg, (list, tuple)) else alpha_cfg
    return {
        "block_size": int(config.get("block_size", HPC_FP8_BLOCK_SIZE)),
        "stem_stride": int(config.get("hpc_stem_stride", config.get("stride", 16))),
        "chunk_size": int(config.get("chunk_size", 2048)),
        "norm": float(config.get("norm", 1.0)),
        "initial_blocks": int(config.get("initial_blocks", HPC_INITIAL_BLOCKS)),
        "window_size": int(config.get("window_size", HPC_WINDOW_SIZE)),
        "lambda_mag": float(config.get("lambda_mag", 0.3)),
        "alpha": float(alpha),
        "k_block_num_rate": float(config.get("k_block_num_rate", 0.1)),
        "k_block_num_bias": int(config.get("k_block_num_bias", 30)),
    }


# ===================================================================== #
#  Execution paths                                                       #
# ===================================================================== #

# --- Path 1: bf16 dense ------------------------------------------------


def _run_hpc_bf16_dense(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """BF16 dense prefill via ``hpc.attention_prefill_bf16``."""
    B, H, Lq, _ = query_states.shape
    D_v = value_states.shape[-1]

    q_varlen = _pack_bhld_to_varlen(query_states)
    k_varlen = _pack_bhld_to_varlen(key_states)
    v_varlen = _pack_bhld_to_varlen(value_states)
    seqlens_q, cu_seqlens_q = _uniform_seq_metadata(B, Lq, query_states.device)

    output = hpc.attention_prefill_bf16(q_varlen, k_varlen, v_varlen, seqlens_q, cu_seqlens_q, Lq)
    return output.view(B, Lq, H, D_v).transpose(1, 2).contiguous()


# --- Path 2: fp8 varlen ------------------------------------------------


def _build_varlen_stem_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Build the block-sparse mask using Triton scoring + ``hpc.stem_tpd``."""
    H_q = query_states.shape[1]
    if key_states.shape[1] != H_q:
        key_states = _repeat_to_q_heads(key_states, H_q)
    if value_states.shape[1] != H_q:
        value_states = _repeat_to_q_heads(value_states, H_q)

    block_logits = _compute_triton_block_logits(
        query_states,
        key_states,
        value_states,
        block_size=params["block_size"],
        stride=params["stem_stride"],
        chunk_size=params["chunk_size"],
        norm=params["norm"],
        causal=True,
    )

    B, _, Lq, _ = query_states.shape
    Lkv = key_states.shape[2]
    q_seq_lens = torch.full((B,), Lq, dtype=torch.int32, device=query_states.device)
    kv_seq_lens = torch.full((B,), Lkv, dtype=torch.int32, device=query_states.device)

    return hpc.stem_tpd(
        block_logits,
        q_seq_lens,
        kv_seq_lens,
        params["block_size"],
        params["alpha"],
        params["initial_blocks"],
        params["window_size"],
        params["k_block_num_rate"],
        params["k_block_num_bias"],
    )


def _run_hpc_varlen_stem(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """FP8 varlen sparse prefill via ``hpc.attention_blocksparse_prefill_fp8``."""
    B, H_q, Lq, _ = query_states.shape
    _, _, Lkv, D_v = value_states.shape

    q_varlen = _pack_bhld_to_varlen(query_states)
    k_varlen = _pack_bhld_to_varlen(key_states)
    v_varlen = _pack_bhld_to_varlen(value_states)

    q_fp8, q_scale = _quantize_per_tensor_fp8(q_varlen)
    k_fp8, k_scale = _quantize_per_tensor_fp8(k_varlen)
    v_fp8, v_scale = _quantize_per_tensor_fp8(v_varlen)

    _, cu_q = _uniform_seq_metadata(B, Lq, query_states.device)
    _, cu_kv = _uniform_seq_metadata(B, Lkv, query_states.device)

    mask = _build_varlen_stem_mask(query_states, key_states, value_states, params)

    output = hpc.attention_blocksparse_prefill_fp8(
        q_fp8,
        k_fp8,
        v_fp8,
        cu_q,
        cu_kv,
        Lq,
        Lkv,
        q_scale,
        k_scale,
        v_scale,
        block_mask=mask,
    )
    return output.view(B, Lq, H_q, D_v).transpose(1, 2).contiguous()


# --- Path 3: fp8 paged -------------------------------------------------


def _run_hpc_paged_stem(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Paged FP8 prefill path for chunked prefill with existing KV history.

    Key semantics
    -------------
    * ``hpc.stem_paged_kv`` expects ``kv_seq_lens`` = **total visible KV length**
      (history + current Q).
    * ``hpc.attention_with_kvcache_blocksparse_prefill_fp8`` expects
      ``seqlens_kvcache`` = **history length only** (tokens *before* the
      current Q chunk).
    """
    B, H_q, Lq, _ = query_states.shape
    _, _, Lkv, D_v = value_states.shape
    if Lkv < Lq:
        raise ValueError(
            f"Paged HPC prefill requires kv_len >= q_len, " f"got kv_len={Lkv}, q_len={Lq}."
        )

    history_len = Lkv - Lq

    # --- FP8 quantisation -------------------------------------------------
    q_fp8, q_scale = _quantize_query_for_paged_fp8(query_states)
    k_varlen = _pack_bhld_to_varlen(key_states)
    v_varlen = _pack_bhld_to_varlen(value_states)
    k_fp8, k_scale = _quantize_per_tensor_fp8(k_varlen)
    v_fp8, v_scale = _quantize_per_tensor_fp8(v_varlen)
    kcache, kv_indices = _pack_paged_cache(k_fp8, B, Lkv, params["block_size"])
    vcache, _ = _pack_paged_cache(v_fp8, B, Lkv, params["block_size"])

    visible_kv_lens = torch.full((B,), Lkv, dtype=torch.int32, device=query_states.device)
    history_kv_lens = torch.full((B,), history_len, dtype=torch.int32, device=query_states.device)
    _, cu_q = _uniform_seq_metadata(B, Lq, query_states.device)

    # --- Step 1: generate sparse mask -------------------------------------
    mask = hpc.stem_paged_kv(
        q_fp8,
        kcache,
        vcache,
        q_scale,
        k_scale,
        v_scale,
        kv_indices,
        cu_q,
        visible_kv_lens,
        lambda_mag=params["lambda_mag"],
        alpha=params["alpha"],
        stem_block_size=params["block_size"],
        stem_stride=params["stem_stride"],
        causal=True,
        initial_blocks=params["initial_blocks"],
        window_size=params["window_size"],
        k_block_num_rate=params["k_block_num_rate"],
        k_block_num_bias=params["k_block_num_bias"],
    )

    # --- Step 2: block-sparse attention -----------------------------------
    output = hpc.attention_with_kvcache_blocksparse_prefill_fp8(
        q_fp8,
        kcache,
        vcache,
        q_scale,
        k_scale,
        v_scale,
        cu_q,
        kv_indices,
        history_kv_lens,  # history length (not total visible!)
        Lq,  # max_seqlens_q
        mask,
    )
    return output.view(B, Lq, H_q, D_v).transpose(1, 2).contiguous()


# ===================================================================== #
#  Top-level HPC dispatcher                                              #
# ===================================================================== #


def stem_forward_hpc(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    prefill_kwargs: dict,
) -> torch.Tensor:
    """HPC backend entry point — selects among bf16 / fp8-varlen / fp8-paged.

    Falls back to :func:`stem_forward_torch` when the hardware or
    configuration is not compatible with the requested HPC path.
    """
    config = prefill_kwargs["attn_forward_config"]
    layer_idx = prefill_kwargs["layer_idx"]
    strict_hpc = config.get("hpc_strict", False)
    hpc_dtype = config.get("hpc_dtype", "bf16")
    fp8_path = config.get("hpc_fp8_path", "varlen")

    # --- Guard: CUDA required ---------------------------------------------
    if query_states.device.type != "cuda":
        if strict_hpc:
            raise RuntimeError("HPC stem backend requires CUDA tensors.")
        if layer_idx == 0:
            print("[Stem][HPC] CUDA tensors are required; falling back to torch backend.")
        return _fallback_to_torch(query_states, key_states, value_states, prefill_kwargs)

    params = _stem_runtime_params(config, layer_idx)

    # --- BF16 dense path --------------------------------------------------
    if hpc_dtype == "bf16":
        try:
            if layer_idx == 0:
                print("[Stem][HPC] using bf16 dense prefill path.")
            return _run_hpc_bf16_dense(query_states, key_states, value_states)
        except Exception as exc:
            if strict_hpc:
                raise RuntimeError(f"HPC bf16 backend failed: {exc}") from exc
            if layer_idx == 0:
                print(
                    f"[Stem][HPC] bf16 dense path failed ({exc}); falling back to torch backend."
                )
            return _fallback_to_torch(query_states, key_states, value_states, prefill_kwargs)

    # --- FP8 sparse path: validate dimensions -----------------------------
    dim_qk = query_states.shape[-1]
    dim_v = value_states.shape[-1]
    block_size = params["block_size"]

    if block_size != HPC_FP8_BLOCK_SIZE:
        if strict_hpc:
            raise RuntimeError(
                f"HPC fp8 sparse prefill only supports block_size=128, got {block_size}."
            )
        if layer_idx == 0:
            print(
                f"[Stem][HPC] fp8 sparse prefill only supports block_size=128, "
                f"got {block_size}; falling back to torch backend."
            )
        return _fallback_to_torch(query_states, key_states, value_states, prefill_kwargs)

    if dim_qk != 128 or dim_v != 128:
        if strict_hpc:
            raise RuntimeError(f"Unsupported HPC fp8 head dims: dim_qk={dim_qk}, dim_v={dim_v}.")
        if layer_idx == 0:
            print(
                f"[Stem][HPC] unsupported fp8 head dims dim_qk={dim_qk}, "
                f"dim_v={dim_v}; falling back to torch backend."
            )
        return _fallback_to_torch(query_states, key_states, value_states, prefill_kwargs)

    # --- Execute FP8 path -------------------------------------------------
    try:
        if fp8_path == "paged":
            if key_states.shape[2] == query_states.shape[2]:
                # First prefill chunk (no KV history). The paged attention
                # kernel needs seqlens_kvcache > 0; use varlen instead.
                if layer_idx == 0:
                    print(
                        "[Stem][HPC] first prefill chunk (q_len == kv_len, no history); "
                        "using varlen fp8 path."
                    )
                return _run_hpc_varlen_stem(query_states, key_states, value_states, params)
            if layer_idx == 0:
                print("[Stem][HPC] using paged fp8 prefill path with stem_paged_kv mask.")
            return _run_hpc_paged_stem(query_states, key_states, value_states, params)

        if fp8_path == "varlen":
            if layer_idx == 0:
                print("[Stem][HPC] using varlen fp8 prefill path with hpc tpd mask.")
            return _run_hpc_varlen_stem(query_states, key_states, value_states, params)

        raise ValueError(f"Unsupported hpc_fp8_path={fp8_path!r}; expected 'paged' or 'varlen'.")

    except Exception as exc:
        if strict_hpc:
            raise RuntimeError(f"HPC stem backend failed: {exc}") from exc
        if layer_idx == 0:
            print(
                f"[Stem][HPC] {fp8_path} fp8 path failed ({exc}); "
                "falling back to torch backend."
            )
        return _fallback_to_torch(query_states, key_states, value_states, prefill_kwargs)
