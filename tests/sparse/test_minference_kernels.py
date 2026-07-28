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

"""Real vendored MInference kernel tests (Triton + CUDA index extension).

These verify the *real* kernels (not the pure-torch reference) — the thing that
makes minference fast in production:
  * the CUDA index ext JIT-builds and is callable;
  * `kernels_available` reports True for all 3 variants here;
  * the real kernels run on REAL Qwen3-8B, are finite, and preserve semantics
    (top-1 agreement vs dense on a coherent prompt);
  * the real kernel and the pure-torch reference AGREE semantically (both
    high top-1 vs dense) — the numeric-alignment check;
  * hard-fail policy: when a kernel is forced-unavailable and
    allow_pseudo_sparse=false, prefill raises KernelMissingError; with
    allow_pseudo_sparse=true it routes to the reference.

Weights policy: correctness uses real Qwen3-8B. Runnable via __main__ (no pytest).
"""

from __future__ import annotations

import os
import sys
import warnings

import torch

# Shared scaffolding is the single source of truth for the drift-prone
# helpers (LIFO double, coverage-floored runner, real-model singleton). The
# previous local ``_Slim`` here had drifted — its ``pop_attn_forward`` ignored
# ``expected_label`` and ``attn_forward_labels`` returned ``set()``, leaving the
# LIFO contract untested in this suite.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA, HEAD_DIM  # noqa: E402
from _harness import FakeSlim as _Slim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import real_qwen3_8b, real_weights_available, run_all  # noqa: E402


def _real():
    """(model, tokenizer) on real Qwen3-8B, or (None, None) if unavailable."""
    return real_qwen3_8b(with_tokenizer=True)


def _coherent_ids(tok, n=1024):
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both the natural and social "
        "sciences. "
    ) * 50
    return tok(para, return_tensors="pt").to("cuda")["input_ids"][:, :n]


def _patch_real(model, variant, kw, allow_pseudo_sparse):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create(variant, attn_kwargs=dict(kw))
    algo.allow_pseudo_sparse = allow_pseudo_sparse
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


# ===========================================================================
# CUDA extension build
# ===========================================================================
def test_cuda_ext_builds_and_callable():
    if not CUDA:
        raise _Skip("CUDA unavailable")
    from angelslim.compressor.sparsity.algorithms.minference.kernels._cuda_ext import (
        cuda_ext_buildable,
        get_cuda_ext,
    )

    if not cuda_ext_buildable():
        raise _Skip("no nvcc / CUDA_HOME — cannot build the index extension")
    ext = get_cuda_ext()
    assert ext is not None, "CUDA index extension failed to build"
    assert hasattr(ext, "convert_vertical_slash_indexes")

    # Smoke call: B=1, H=2, 256-token context.
    seqlens = torch.tensor([256], dtype=torch.int32, device="cuda")
    vidx = torch.arange(0, 30, device="cuda", dtype=torch.int32).view(1, 1, 30).repeat(1, 2, 1)
    sidx = torch.arange(0, 50, device="cuda", dtype=torch.int32).view(1, 1, 50).repeat(1, 2, 1)
    out = ext.convert_vertical_slash_indexes(seqlens, vidx, sidx, 256, 64, 64)
    assert len(out) == 4  # block_count, block_offset, column_count, column_index


def test_kernels_available_reports_true_here():
    if not CUDA:
        raise _Skip("CUDA unavailable")
    from angelslim.compressor.sparsity.algorithms.minference import kernels as K

    # a_shape/tri_shape need only Triton+CUDA; minference also needs the ext.
    assert K.kernels_available("a_shape")
    assert K.kernels_available("tri_shape")
    # minference depends on the buildable CUDA ext.
    from angelslim.compressor.sparsity.algorithms.minference.kernels._cuda_ext import (
        cuda_ext_buildable,
    )

    assert K.kernels_available("minference") == cuda_ext_buildable()
    assert not K.kernels_available("not_a_variant")


# ===========================================================================
# Real kernels on real Qwen3-8B
# ===========================================================================
def test_real_kernels_preserve_semantics_on_qwen3_8b():
    """All 3 real kernels run on REAL Qwen3-8B with top-1 agreement ~1.0.

    allow_pseudo_sparse=False forces the real kernel path (no reference
    fallback). Measured ground truth (2026/06/02, coherent 1024-tok prompt):
    a_shape/tri_shape/minference all hit 1.000 top-1 agreement vs dense."""
    m, tok = _real()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    ids = _coherent_ids(tok, 1024)
    with torch.no_grad():
        dense = m(ids).logits.float()

    for variant, kw in [
        ("a_shape", {"n_init": 64, "n_local": 512}),
        ("tri_shape", {"n_init": 64, "n_local": 512, "n_last": 100}),
        ("minference", {}),
    ]:
        slim, patched = _patch_real(m, variant, kw, allow_pseudo_sparse=False)
        try:  # try/finally so an assertion failure can't leak the patch
            with torch.no_grad():
                sp = m(ids).logits.float()
        finally:
            unpatch_sparsity(slim, patched)
        assert torch.isfinite(sp).all(), variant
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"{variant} real-kernel top-1 agreement {agree:.3f}"


def test_real_kernel_and_reference_agree():
    """Numeric alignment: real kernel and pure-torch reference both preserve the
    prediction (high top-1 vs dense) for the same config.

    They are NOT bit-equal — the kernel is block-granular bf16 accumulation, the
    reference is exact per-position — but both must be semantically faithful, and
    they must agree with EACH OTHER at least as well as each agrees with dense."""
    m, tok = _real()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    ids = _coherent_ids(tok, 1024)
    with torch.no_grad():
        dense = m(ids).logits.float()

    variant, kw = "a_shape", {"n_init": 64, "n_local": 512}

    slim, patched = _patch_real(m, variant, kw, allow_pseudo_sparse=False)  # real kernel
    try:  # try/finally so an assertion failure can't leak the patch
        with torch.no_grad():
            kernel_out = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)

    # Force the reference path by making the kernel "unavailable" + opting in.
    from angelslim.compressor.sparsity.algorithms.minference import kernels as K

    orig = K.kernels_available
    K.kernels_available = lambda v, head_dim=None: False
    # prefill.py imported the name; patch there too.
    import angelslim.compressor.sparsity.algorithms.minference.prefill as P

    P._k.kernels_available = lambda v, head_dim=None: False
    try:
        slim, patched = _patch_real(m, variant, kw, allow_pseudo_sparse=True)  # reference
        with torch.no_grad():
            ref_out = m(ids).logits.float()
        unpatch_sparsity(slim, patched)
    finally:
        K.kernels_available = orig
        P._k.kernels_available = orig

    agree_kernel = (dense[0].argmax(-1) == kernel_out[0].argmax(-1)).float().mean().item()
    agree_ref = (dense[0].argmax(-1) == ref_out[0].argmax(-1)).float().mean().item()
    agree_each = (kernel_out[0].argmax(-1) == ref_out[0].argmax(-1)).float().mean().item()
    assert agree_kernel > 0.95, f"kernel vs dense {agree_kernel:.3f}"
    assert agree_ref > 0.95, f"reference vs dense {agree_ref:.3f}"
    assert agree_each > 0.95, f"kernel vs reference {agree_each:.3f}"


def test_real_kernel_decode_generates():
    """Real-kernel sparse prefill + decode generates the correct answer (REAL)."""
    m, tok = _real()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    slim, patched = _patch_real(
        m, "a_shape", {"n_init": 64, "n_local": 512}, allow_pseudo_sparse=False
    )
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    finally:
        unpatch_sparsity(slim, patched)
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"real-kernel decode lost the answer: {text!r}"


def test_vertical_slash_kernel_batch_gt_1_matches_per_element():
    """vertical_slash kernel with batch>1 must equal each element run alone.

    Regression: ``seqlens`` was a length-1 tensor while the Triton kernel reads
    ``seqlens[off_hz // H]`` for every (batch, head) program — a device-side
    out-of-bounds read for batch>1 that returned garbage (or an IMA) for every
    element past the first. With one seqlen per batch element, a B=2 run is
    bit-identical to running each row separately as B=1."""
    if not CUDA:
        raise _Skip("CUDA unavailable")
    from angelslim.compressor.sparsity.algorithms.minference.kernels._cuda_ext import (
        cuda_ext_buildable,
    )

    if not cuda_ext_buildable():
        raise _Skip("no nvcc / CUDA_HOME — cannot build the index extension")
    from angelslim.compressor.sparsity.algorithms.minference.kernels.pit_sparse_flash_attention import (  # noqa: E501
        vertical_slash_sparse_attention,
    )
    from angelslim.compressor.sparsity.algorithms.minference.prefill import (
        estimate_vertical_slash_indexes,
    )

    torch.manual_seed(0)
    B, H, L, D = 2, 4, 512, 128
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)

    v_idx, s_idx = estimate_vertical_slash_indexes(q, k, 1000, 6096, D)
    out_batched = vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)
    assert out_batched.shape == (B, H, L, D)
    assert torch.isfinite(out_batched).all()

    for b in range(B):
        vb_idx, sb_idx = estimate_vertical_slash_indexes(q[b : b + 1], k[b : b + 1], 1000, 6096, D)
        ob = vertical_slash_sparse_attention(
            q[b : b + 1], k[b : b + 1], v[b : b + 1], vb_idx, sb_idx
        )
        d = (out_batched[b : b + 1].float() - ob.float()).abs().max().item()
        assert d < 1e-2, f"batch element {b} differs from its B=1 run by {d}"


# ===========================================================================
# Decision-#5 policy when a kernel is unavailable
# ===========================================================================
def test_hardfail_when_kernel_unavailable_and_no_pseudo():
    """kernel forced-unavailable + allow_pseudo_sparse=false -> KernelMissingError."""
    if not CUDA:
        raise _Skip("CUDA unavailable")
    import angelslim.compressor.sparsity.algorithms.minference.prefill as P
    from angelslim.compressor.sparsity._kernel_check import KernelMissingError

    q = torch.randn(1, 2, 256, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 256, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, 256, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    orig = P._k.kernels_available
    P._k.kernels_available = lambda variant, head_dim=None: False
    try:
        try:
            P.minference_prefill(
                "minference",
                q,
                k,
                v,
                streaming_kwargs={},
                best_pattern=None,
                layer_idx=0,
                head_dim=HEAD_DIM,
                allow_pseudo_sparse=False,
            )
            raise AssertionError("expected KernelMissingError")
        except KernelMissingError as e:
            assert "allow_pseudo_sparse" in str(e)

        # With the opt-in, it routes to the reference (finite output).
        out = P.minference_prefill(
            "minference",
            q,
            k,
            v,
            streaming_kwargs={},
            best_pattern=None,
            layer_idx=0,
            head_dim=HEAD_DIM,
            allow_pseudo_sparse=True,
        )
        assert out.shape == q.shape and torch.isfinite(out).all()
    finally:
        P._k.kernels_available = orig


# ===========================================================================
# Runner
# ===========================================================================
if __name__ == "__main__":
    sys.exit(
        1
        if run_all(
            globals(),
            f"CUDA={CUDA}, real_weights={real_weights_available()}",
        )
        else 0
    )
