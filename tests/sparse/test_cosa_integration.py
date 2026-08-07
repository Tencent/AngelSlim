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

"""CoSA integration tests.

CoSA has exactly one code path — Triton proxy plus the internal ``hpc`` kernel,
both hard dependencies — so a forward can only be exercised where ``hpc`` is
installed. Everything else here (registration, traits, config validation, the
layout adapters, the patch lifecycle) is pure CPU and runs anywhere torch
imports; the layout-adapter tests matter most, since a silent shape mistake in
the varlen/paged conversion would surface as wrong numbers rather than a crash.

Run:  python tests/sparse/test_cosa_integration.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA  # noqa: E402
from _harness import HEAD_DIM  # noqa: E402
from _harness import FakeSlim  # noqa: E402
from _harness import Skip  # noqa: E402
from _harness import SkipReason  # noqa: E402
from _harness import patched  # noqa: E402
from _harness import real_qwen3_8b  # noqa: E402
from _harness import real_weights_available  # noqa: E402
from _harness import record_capability  # noqa: E402
from _harness import rel  # noqa: E402
from _harness import run_all  # noqa: E402
from _harness import tiny_qwen3  # noqa: E402


def _make_algo(attn_kwargs=None):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    return SparsityAlgorithmRegistry.create("cosa", attn_kwargs=dict(attn_kwargs or {}))


class _StubModel:
    """Minimal stand-in exposing only the config fields ``setup`` reads."""

    def __init__(self, num_attention_heads=4, num_hidden_layers=4, model_type="qwen3"):
        self.config = type(
            "_Cfg",
            (),
            {
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "model_type": model_type,
            },
        )()


def _hpc_available() -> bool:
    try:
        import hpc  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


# ===========================================================================
# Registration + traits (no weights)
# ===========================================================================
def test_cosa_registered_and_traits():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    assert "cosa" in SparsityAlgorithmRegistry._factories
    algo = _make_algo()
    assert algo.name == "cosa"
    t = algo.traits
    assert t.requires_unrepeated_kv is False
    assert t.needs_calibration is False
    assert t.supports_padding_mask is False
    assert t.model_modal == "any"
    for mt in ("qwen3", "qwen3_moe", "hy_v3"):
        assert mt in t.compatible_model_types, mt


def test_cosa_rejects_incompatible_model_type():
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import _check_model_type

    algo = _make_algo()
    _check_model_type(algo, _StubModel(model_type="qwen3"))  # supported: no raise
    try:
        _check_model_type(algo, _StubModel(model_type="qwen3_5_text"))
    except IncompatibleConfigError as e:
        assert "qwen3_5_text" in str(e)
    else:
        raise AssertionError("CoSA must reject a model_type outside its compatibility set")


# ===========================================================================
# Config validation
# ===========================================================================
def test_cosa_setup_defaults():
    algo = _make_algo()
    algo.setup(_StubModel())
    cfg = algo._cfg
    assert cfg["stride"] == 8
    assert cfg["topp"] == 0.8
    assert cfg["block_size"] == 128
    assert cfg["chunk_size"] is None  # derived per prefill from k_len
    assert cfg["skipsoftmax_threshold"] == 1000
    assert "allow_pseudo_sparse" not in cfg, "CoSA has no fallback path to configure"


def test_cosa_rejects_allow_pseudo_sparse():
    """The knob promises a torch fallback CoSA does not have; refuse, never ignore."""
    from angelslim.compressor.sparsity._base import IncompatibleConfigError

    algo = _make_algo({"allow_pseudo_sparse": True})
    try:
        algo.setup(_StubModel())
    except IncompatibleConfigError as e:
        assert "allow_pseudo_sparse" in str(e)
    else:
        raise AssertionError("allow_pseudo_sparse=true must be rejected, not silently ignored")


def test_cosa_setup_rejects_bad_stride():
    from angelslim.compressor.sparsity._base import IncompatibleConfigError

    algo = _make_algo({"stride": 0})
    try:
        algo.setup(_StubModel())
    except IncompatibleConfigError as e:
        assert "stride" in str(e)
    else:
        raise AssertionError("stride < 1 must be rejected")


def test_cosa_topp_must_be_a_scalar():
    """Only a scalar topp is accepted; per-head tables are rejected by design."""
    from angelslim.compressor.sparsity._base import IncompatibleConfigError

    algo = _make_algo({"topp": 1})  # an int is fine, and normalizes to float
    algo.setup(_StubModel())
    assert algo._cfg["topp"] == 1.0
    assert isinstance(algo._cfg["topp"], float)

    # A per-head list sized correctly for the model must STILL be refused —
    # silently accepting one would resurrect the table path.
    for bad in ([0.8, 0.9, 0.85, 0.95], (0.8, 0.9), "0.8", None, True):
        wrong = _make_algo({"topp": bad})
        try:
            wrong.setup(_StubModel(num_attention_heads=4))
        except IncompatibleConfigError as e:
            assert "scalar" in str(e), f"{bad!r} -> {e}"
        else:
            raise AssertionError(f"topp={bad!r} must be rejected as a non-scalar")


def test_cosa_topp_range_is_checked():
    from angelslim.compressor.sparsity._base import IncompatibleConfigError

    for bad in (0.0, -0.1, 1.5):
        algo = _make_algo({"topp": bad})
        try:
            algo.setup(_StubModel())
        except IncompatibleConfigError as e:
            assert "(0, 1]" in str(e), f"{bad} -> {e}"
        else:
            raise AssertionError(f"topp={bad} is outside (0, 1] and must be rejected")


def test_cosa_default_chunk_size_tiles_cleanly():
    from angelslim.compressor.sparsity.algorithms.cosa.prefill import default_chunk_size

    for k_len in (2048, 4096, 16384, 65536, 131072):
        cs = default_chunk_size(k_len)
        assert cs >= 2048, f"chunk_size floor violated at k_len={k_len}: {cs}"
        # The proxy GEMM tiles the reduced matrix at 128; at the documented
        # stride 8 the chunk must cover whole tiles or the kernel cannot launch.
        assert cs % (8 * 128) == 0, f"k_len={k_len} -> chunk_size={cs} untileable"


# ===========================================================================
# hpc layout adapters (CPU) — a silent shape bug here means wrong numbers
# ===========================================================================
def test_cosa_varlen_q_roundtrip():
    from angelslim.compressor.sparsity.algorithms.cosa.ops.hpc_adaption import (
        convert_to_varlen_q,
        convert_varlen_out_to_bshd,
    )

    b, h, s, d = 1, 4, 384, HEAD_DIM
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    q_thd, cu_seqlens, max_seq = convert_to_varlen_q(q)

    assert q_thd.shape == (b * s, h, d)
    assert max_seq == s
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, s]
    assert q_thd.is_contiguous()

    # The kernel returns (total_tokens, H, D); converting back must be exact.
    back = convert_varlen_out_to_bshd(q_thd, b, s)
    assert back.shape == q.shape
    assert torch.equal(back, q)


def test_cosa_paged_kv_layout():
    from angelslim.compressor.sparsity.algorithms.cosa.ops.hpc_adaption import (
        convert_to_paged_kv,
    )

    b, h, s, d = 1, 2, 384, HEAD_DIM
    page = 64
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    kcache, vcache, block_ids, seqlens = convert_to_paged_kv(k, v, page_size=page)

    pages = s // page
    assert kcache.shape == (b * pages, page, h, d)
    assert vcache.shape == kcache.shape
    assert block_ids.shape == (b, pages)
    assert block_ids.dtype == torch.int32
    assert seqlens.tolist() == [s]

    # Page p must hold tokens [p*page, (p+1)*page) for every head.
    for p in (0, pages // 2, pages - 1):
        expect = k[0, :, p * page : (p + 1) * page, :].permute(1, 0, 2)
        assert torch.equal(kcache[p], expect), f"page {p} holds the wrong tokens"


def test_cosa_paged_kv_pads_partial_block():
    """A k_len that is not a whole number of 128-blocks must be padded up."""
    from angelslim.compressor.sparsity.algorithms.cosa.ops.hpc_adaption import (
        convert_to_paged_kv,
    )

    b, h, d = 1, 2, HEAD_DIM
    s = 3 * 128 + 40  # 424 -> padded to 512
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    kcache, _, block_ids, seqlens = convert_to_paged_kv(k, k.clone(), page_size=64)

    assert kcache.shape[0] == b * (512 // 64)
    assert block_ids.shape == (b, 512 // 64)
    # seqlens still reports the REAL length; padding is addressable but not read.
    assert seqlens.tolist() == [s]


def test_cosa_adapters_reject_wrong_head_dim():
    from angelslim.compressor.sparsity.algorithms.cosa.ops.hpc_adaption import (
        convert_to_paged_kv,
        convert_to_varlen_q,
    )

    bad = torch.randn(1, 2, 256, 64, dtype=torch.bfloat16)
    for fn in (lambda: convert_to_varlen_q(bad), lambda: convert_to_paged_kv(bad, bad.clone())):
        try:
            fn()
        except ValueError as e:
            assert "128" in str(e)
        else:
            raise AssertionError("head_dim != 128 must be refused by the hpc adapters")


# ===========================================================================
# Patch lifecycle (structure only — no forward, so no kernel needed)
# ===========================================================================
def test_cosa_patches_every_layer_and_restores():
    model = tiny_qwen3(num_layers=4)
    slim = FakeSlim(model)
    algo = _make_algo({"topp": 0.8})

    layers = list(model.model.layers)
    originals = [layer.self_attn.forward for layer in layers]
    with patched(slim, algo) as handles:
        assert len(handles) == model.config.num_hidden_layers
        assert slim.attn_forward_labels() == {"sparse"}
        for layer in layers:
            assert layer.self_attn.attn_forward_config["topp"] == 0.8
    for i, layer in enumerate(layers):
        assert layer.self_attn.forward == originals[i], "unpatch did not restore the forward"
    assert slim.attn_forward_labels() == set()


# ===========================================================================
# Numerical correctness (needs real weights AND the internal hpc extension)
# ===========================================================================
def test_cosa_keep_all_matches_dense():
    if not real_weights_available():
        raise Skip("real Qwen3-8B weights unavailable", SkipReason.NO_QWEN3_8B)
    if not _hpc_available():
        raise Skip("internal hpc extension unavailable", SkipReason.NO_BLOCK_SPARSE_ATTN)

    m = real_qwen3_8b()
    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, 2048), device="cuda")
    with torch.no_grad():
        dense = m(ids).logits.float()

    # topp == 1.0 keeps every causal block and skipsoftmax_threshold < 0 disables
    # the runtime skip, so the two together must reproduce dense attention.
    slim = FakeSlim(m)
    algo = _make_algo(
        {"topp": 1.0, "stride": 8, "chunk_size": 2048, "skipsoftmax_threshold": -1.0}
    )
    with patched(slim, algo) as handles:
        assert len(handles) == m.config.num_hidden_layers
        with torch.no_grad():
            sp = m(ids).logits.float()
    with torch.no_grad():
        restored = m(ids).logits.float()

    assert not torch.isnan(sp).any()
    err = rel(sp, dense)
    assert err < 0.02, f"keep-all CoSA should match dense, rel={err}"
    assert dense[0, -1].argmax() == sp[0, -1].argmax(), "keep-all argmax drift"
    assert torch.equal(restored, dense), "unpatch did not restore the real model"
    record_capability("cosa_real_correctness")


if __name__ == "__main__":
    sys.exit(
        run_all(
            dict(globals()),
            summary_extra=f"cuda={CUDA} hpc={_hpc_available()}",
        )
    )
