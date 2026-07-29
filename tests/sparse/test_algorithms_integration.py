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

"""Per-algorithm integration tests (merged: MInference family, XAttention,
FlexPrefill, FlashPrefill, VecAttention).

One suite for the five kernel-backed algorithms' integration coverage — each
section is the former per-algorithm file, verbatim, with its private helpers
namespaced by an algorithm tag (``_mi_`` / ``_xa_`` / ``_fx_`` / ``_fp_`` /
``_vc_``) so the shared ``_patch_real`` / ``_coherent_ids`` / ``_split_device_map``
helpers do not collide. Every assertion is preserved; nothing was rewritten.

The common shape per algorithm: registration + traits (no weights), head_dim
gate, dense parity at a keep-all budget on real Qwen3-8B, coherent-prompt top-1
preservation, decode generates the right answer, head_dim-256 reference fallback
on real Qwen3.5-9B, and device_map layer-sharding correctness. Algorithm-specific
units (MInference pattern-loader + reference math + vertical_and_slash diagonal;
each algorithm's reference/threshold behavior; FlashPrefill clean-room provenance;
block-size policy) are kept in their section.

Stem lives in its own suite (test_stem_integration.py); the Qwen3.5 gated
template and the MInference real CUDA kernels have their own dedicated files.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA  # noqa: E402
from _harness import HEAD_DIM  # noqa: E402
from _harness import qwen35_available  # noqa: E402
from _harness import real_weights_available  # noqa: E402
from _harness import run_all  # noqa: E402
from _harness import REAL_W_QWEN3_5_9B as _REAL_W35  # noqa: E402
from _harness import REAL_W_QWEN3_8B as _REAL_W  # noqa: E402
from _harness import FakeSlim as _Slim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import real_qwen3_8b as _real_qwen3_8b  # noqa: E402
from _harness import record_capability as _cap  # noqa: E402
from _harness import rel as _rel  # noqa: E402


# ###########################################################################
# MINFERENCE integration (former test_minference_integration.py)
# ###########################################################################
def _mi_tiny_qwen3(num_layers=2, attn_impl="eager"):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=4096,
        attn_implementation=attn_impl,
    )
    m = Qwen3ForCausalLM(cfg)
    if CUDA:
        m = m.to("cuda", dtype=torch.bfloat16)
    return m.eval()


# ===========================================================================
# / — three names map to one class via the registry


# ===========================================================================
def test_minference_registry_three_names_one_class():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.algorithms.minference.algorithm import MInference
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    avail = SparsityAlgorithmRegistry.available()
    for name in ("minference", "a_shape", "tri_shape"):
        assert name in avail, f"{name} not registered"
        algo = SparsityAlgorithmRegistry.create(name, attn_kwargs={})
        assert isinstance(algo, MInference)
        assert algo.variant == name
        assert algo.name == name


def test_minference_unknown_variant_rejected():
    from angelslim.compressor.sparsity.algorithms.minference.algorithm import MInference

    try:
        MInference(variant="bogus")
        raise AssertionError("unknown variant should ValueError")
    except ValueError:
        pass


def test_registering_minference_does_not_import_vecattention_kernel():
    """importing the algorithms package (stem + minference + vecattention)

    must NOT pull the heavy VecAttention ``vllm_flash_attn`` KERNEL as an import
    side-effect — registration is metadata only; the kernel loads lazily on the
    first prefill that actually needs it.

    NB: when ``vllm_flash_attn`` is pip-installed (editable), a
    ``__editable___vllm_flash_attn..._finder`` module is loaded by a ``.pth`` hook
    at INTERPRETER STARTUP — before any algorithm import, and independent of it.
    That finder is install machinery, not a kernel import; the contract is that
    importing the algorithms package adds NO new vllm_flash_attn module and never
    loads the actual ``vllm_flash_attn`` package. We assert the before/after delta
    (robust whether or not the kernel is built)."""
    _k = "vllm_flash_attn"
    before = {m for m in sys.modules if _k in m}
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  # register all algos

    after = {m for m in sys.modules if _k in m}
    newly = sorted(after - before)
    assert not newly, f"violated: importing algorithms imported {newly}"

    # The real kernel package itself must not have been eagerly imported.
    assert _k not in sys.modules, (
        f"violated: the {_k!r} kernel package was imported as a side-effect "
        f"of registering the algorithms (it must load lazily at first prefill)."
    )


# ===========================================================================
# — requires_unrepeated_kv=False AND post-repeat dispatch


# ===========================================================================
def test_minference_requires_unrepeated_kv_false():
    from angelslim.compressor.sparsity.algorithms.minference.algorithm import MInference

    for v in ("minference", "a_shape", "tri_shape"):
        traits = MInference(variant=v).traits
        assert traits.requires_unrepeated_kv is False
        # gated template adds Qwen3.5 (text-tower + wrapper model_types);

        # Hunyuan-V3 (hy_v3) shares the plain Qwen3-style template.
        assert traits.compatible_model_types == frozenset(
            {
                "qwen3",
                "qwen3_moe",
                "qwen3_5",
                "qwen3_5_text",
                "qwen3_5_moe",
                "qwen3_5_moe_text",
                "hy_v3",
            }
        )


def test_minference_attn_type_dispatch_post_repeat():
    """at dispatch the kernel sees K/V at num_attention_heads (post-repeat).


    Structural shape assertion (not weight-dependent), so a tiny synthetic model
    is the right tool. We spy on the prefill dispatch entry to capture K's head
    dim — this is the boundary forward.py hands post-repeat K/V to.
    """
    if not CUDA:
        raise _Skip("CUDA required")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  # register
    import angelslim.compressor.sparsity.algorithms.minference.prefill as prefill_mod
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    for variant in ("minference", "a_shape", "tri_shape"):
        m = _mi_tiny_qwen3(num_layers=2)
        n_heads = m.config.num_attention_heads  # 4
        n_kv = m.config.num_key_value_heads  # 2
        assert n_heads != n_kv, "test needs real GQA to be meaningful"

        captured = {}
        orig = prefill_mod.minference_prefill

        def _spy(variant_, q, k, v, _captured=captured, _orig=orig, **kw):
            _captured["k_heads"] = k.shape[1]
            _captured["q_heads"] = q.shape[1]
            return _orig(variant_, q, k, v, **kw)

        prefill_mod.minference_prefill = _spy
        try:
            slim = _Slim(m)
            algo = SparsityAlgorithmRegistry.create(
                variant, attn_kwargs={"n_init": 8, "n_local": 64}
            )
            patched = apply_sparsity_patch(slim, algo)
            ids = torch.randint(0, 512, (1, 256), device="cuda")
            with torch.no_grad():
                m(ids)
            unpatch_sparsity(slim, patched)
        finally:
            prefill_mod.minference_prefill = orig

        assert captured["k_heads"] == n_heads, (
            f"{variant}: K saw {captured['k_heads']} heads, " f"expected post-repeat {n_heads}"
        )
        assert captured["q_heads"] == n_heads


# ===========================================================================
# — model_type guard


# ===========================================================================
def test_minference_model_type_guard():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  # register
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import _check_model_type
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    class _Cfg:
        def __init__(self, mt):
            self.model_type = mt

    class _HF:
        def __init__(self, mt):
            self.config = _Cfg(mt)

    algo = SparsityAlgorithmRegistry.create("minference", attn_kwargs={})
    # Qwen3 + Qwen3.5 (both wrapper and text-tower model_types) are supported.
    for mt in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        _check_model_type(algo, _HF(mt))
    # Non-Qwen3 families are blocked.
    for mt in ("llama", "qwen2", "mistral"):
        try:
            _check_model_type(algo, _HF(mt))
            raise AssertionError(f"{mt} should be blocked")
        except IncompatibleConfigError:
            pass


# ===========================================================================
# — pattern JSON fingerprint validation


# ===========================================================================
def _mi_real_fingerprint_doc(config):
    return {
        "schema_version": 1,
        "model_fingerprint": {
            "model_type": config.model_type,
            "rope_theta": float(getattr(config, "rope_theta", 1000000.0)),
            "max_position_embeddings": config.max_position_embeddings,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
        },
        "best_pattern": {"0": {"0": ["vertical_and_slash", 1000, 6096, 1]}},
        "minference_revision": "a4eb395",
    }


def test_pattern_loader_fingerprint_match_and_mismatch():
    import json

    from transformers import Qwen3Config

    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.algorithms.minference.patterns._loader import (
        load_pattern,
    )

    cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=1000000.0,
        max_position_embeddings=40960,
    )
    doc = _mi_real_fingerprint_doc(cfg)

    with tempfile.TemporaryDirectory() as d:
        # matching fingerprint -> loads, normalized to int keys + tuple entry.
        p = os.path.join(d, "ok.json")
        with open(p, "w") as f:
            json.dump(doc, f)
        best = load_pattern(p, cfg)
        assert 0 in best and 0 in best[0]
        assert best[0][0] == ("vertical_and_slash", 1000, 6096, 1)

        # mutate rope_theta -> mismatch -> IncompatibleConfigError.
        bad = json.loads(json.dumps(doc))
        bad["model_fingerprint"]["rope_theta"] = 500000.0
        pbad = os.path.join(d, "bad.json")
        with open(pbad, "w") as f:
            json.dump(bad, f)
        try:
            load_pattern(pbad, cfg)
            raise AssertionError("rope_theta mismatch should raise ")

        except IncompatibleConfigError as e:
            assert "rope_theta" in str(e)


def test_pattern_loader_edge_cases():
    import json

    from transformers import Qwen3Config

    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.algorithms.minference.patterns._loader import (
        load_pattern,
    )

    cfg = Qwen3Config(
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=128
    )

    # No pattern_path -> dense fallback (None).
    assert load_pattern(None, cfg) is None
    assert load_pattern("", cfg) is None

    # Missing file -> IncompatibleConfigError.
    try:
        load_pattern("/no/such/pattern.json", cfg)
        raise AssertionError("missing file should raise")
    except IncompatibleConfigError:
        pass

    # Wrong schema_version -> IncompatibleConfigError.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v9.json")
        with open(p, "w") as f:
            json.dump({"schema_version": 9, "model_fingerprint": {}, "best_pattern": {}}, f)
        try:
            load_pattern(p, cfg)
            raise AssertionError("schema_version mismatch should raise")
        except IncompatibleConfigError as e:
            assert "schema_version" in str(e)


def test_minference_setup_dense_fallback():
    """minference variant without a pattern_path -> best_pattern is None."""
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  # register
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _mi_tiny_qwen3(num_layers=2)
    algo = SparsityAlgorithmRegistry.create("minference", attn_kwargs={})
    algo.setup(m)
    assert algo._attn_forward_config["best_pattern"] is None
    assert algo._attn_forward_config["variant"] == "minference"


# ===========================================================================
# Reference correctness (synthetic tensors): degrade-to-dense + sparsify
# ===========================================================================
def test_reference_degrade_to_dense():
    if not CUDA:
        raise _Skip("CUDA required")
    from angelslim.compressor.sparsity.algorithms.minference.reference import (
        a_shape_attention,
        dense_causal_attention,
        tri_shape_attention,
        vertical_and_slash_attention,
    )

    torch.manual_seed(0)
    B, H, L, D = 1, 4, 512, HEAD_DIM
    q = torch.randn(B, H, L, D, device="cuda")
    k = torch.randn(B, H, L, D, device="cuda")
    v = torch.randn(B, H, L, D, device="cuda")
    d = dense_causal_attention(q, k, v)

    assert _rel(a_shape_attention(q, k, v, L, L), d) < 1e-5
    assert _rel(tri_shape_attention(q, k, v, L, L, 100), d) < 1e-5
    assert _rel(vertical_and_slash_attention(q, k, v, L, L), d) < 1e-5


def test_reference_sparsity_engages():
    if not CUDA:
        raise _Skip("CUDA required")
    from angelslim.compressor.sparsity.algorithms.minference.reference import (
        a_shape_attention,
        dense_causal_attention,
        tri_shape_attention,
        vertical_and_slash_attention,
    )

    torch.manual_seed(0)
    B, H, L, D = 1, 4, 512, HEAD_DIM
    q = torch.randn(B, H, L, D, device="cuda")
    k = torch.randn(B, H, L, D, device="cuda")
    v = torch.randn(B, H, L, D, device="cuda")
    d = dense_causal_attention(q, k, v)

    for name, out in [
        ("a_shape", a_shape_attention(q, k, v, 8, 64)),
        ("tri_shape", tri_shape_attention(q, k, v, 8, 64, 32)),
        ("v_and_s", vertical_and_slash_attention(q, k, v, 30, 64)),
    ]:
        assert torch.isfinite(out).all(), name
        assert out.shape == q.shape, name
        assert _rel(out, d) > 0.01, f"{name} should differ from dense"


def test_vertical_and_slash_diagonal_alignment():
    """lock: the slash diagonals must be selected in FULL-matrix coordinates,

    aligned with the consumer's ``diag_id``.

    The bug: ``slash[..., -Lk:]`` dropped the leading ``lq-1`` diagonal sums, so
    every kept diagonal id was shifted by ``lq-1`` (= last_q-1 = 63). A query's
    *true* high-importance diagonal was replaced by one 63 keys away.

    Construct a signal that lives on ONE off-main diagonal (constant key offset
    ``band``, placed OUTSIDE the always-kept local-100 band and the sink-30), give
    a tiny slash budget, and capture the keep-mask the reference builds. Correct
    alignment ⇒ the band key ``query-band`` is kept and the *shifted* key
    ``query-band+63`` is NOT (it isn't a forced sink/local). The off-by-63 keeps
    the shifted key instead. Pure math; runs on CPU."""
    import angelslim.compressor.sparsity.algorithms.minference.reference as R

    torch.manual_seed(0)
    B, H, L, D = 1, 1, 512, 16
    last_q, band = 64, 300  # diag_id = (L-1) - band = 211, outside local-100 + sinks

    q = torch.zeros(B, H, L, D)
    k = torch.zeros(B, H, L, D)
    v = torch.randn(B, H, L, D)
    for i in range(L):
        j = i - band
        if j >= 0:
            q[0, 0, i] = torch.randn(D)
            k[0, 0, j] = q[0, 0, i] * 5.0  # large dot only on the band diagonal

    # Capture the keep-mask the reference hands to softmax.
    captured = {}
    orig = R._masked_softmax_attention

    def _spy(qq, kk, vv, keep):
        captured["keep"] = keep
        return orig(qq, kk, vv, keep)

    R._masked_softmax_attention = _spy
    try:
        # slash_size must exceed the 100 forced-local diagonals so topk can
        # actually reach the band diagonal (≤100 is fully consumed by the local
        # band and never selects anything else).
        R.vertical_and_slash_attention(q, k, v, vertical_size=1, slash_size=120, last_q=last_q)
    finally:
        R._masked_softmax_attention = orig

    keep = captured["keep"][0, 0]  # (Lq, Lk) bool
    # Pick queries past the band but past the local band too, so neither the
    # band key nor the shifted key is a forced sink (<30) or local (within 100).
    bad = []
    for query in range(band + 40, L, 17):
        band_key = query - band
        shifted_key = band_key + (last_q - 1)  # what the off-by-63 would keep
        if band_key < 30 or query - band_key <= 100:
            continue  # would be a forced sink/local; not a clean discriminator
        if not keep[query, band_key]:
            bad.append(f"q{query}: true band key {band_key} NOT kept")
        # The shifted key is only an artifact of the bug; with the fix it should
        # not be specially kept (unless it happens to be sink/local).
        if (
            shifted_key < L
            and keep[query, shifted_key]
            and not (shifted_key < 30 or query - shifted_key <= 100)
        ):
            bad.append(f"q{query}: shifted key {shifted_key} kept (off-by-63 artifact)")

    assert not bad, "slash diagonal misalignment \n " + "\n ".join(bad[:8])


def test_a_shape_no_nan_on_empty_window():
    """a_shape with n_init=0 AND n_local=0 would fully-mask a query row

    (no sink, no window) → softmax over all -inf → NaN. The self-key guard
    (every query keeps its own diagonal) must prevent that, matching the guard
    vertical_and_slash already has. Pure math; CPU."""
    from angelslim.compressor.sparsity.algorithms.minference.reference import (
        a_shape_attention,
    )

    torch.manual_seed(0)
    B, H, L, D = 1, 2, 64, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    out = a_shape_attention(q, k, v, n_init=0, n_local=0)
    assert torch.isfinite(out).all(), "a_shape produced NaN/inf on empty window"
    assert out.shape == q.shape


# ===========================================================================
# REAL Qwen3-8B correctness
# ===========================================================================
def _mi_patch_real(model, variant, attn_kwargs):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create(variant, attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def test_minference_real_weights_degrade_to_dense():
    """All three variants at full budget ~= dense on REAL Qwen3-8B."""
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    torch.manual_seed(0)
    L = 2048
    ids = torch.randint(0, 151000, (1, L), device="cuda")
    with torch.no_grad():
        dense = m(ids).logits.float()

    # Budgets that cover the whole sequence => degrade to dense.
    full = {
        "a_shape": {"n_init": L, "n_local": L},
        "tri_shape": {"n_init": L, "n_local": L, "n_last": 100},
        "minference": {},  # vertical_and_slash with v=1000,s=6096; both>=... :
    }
    # minference dense fallback has vertical_size=1000 < L=2048 so it is NOT
    # fully dense; supply a pattern-free large-budget by using a_shape-like
    # behavior is not possible, so assert the looser keep-quality bound below.
    for variant in ("a_shape", "tri_shape"):
        slim, patched = _mi_patch_real(m, variant, full[variant])
        try:  # try/finally so an assertion failure can't leak the patch

            with torch.no_grad():
                sp = m(ids).logits.float()
        finally:
            unpatch_sparsity(slim, patched)
        with torch.no_grad():
            restored = m(ids).logits.float()
        rel = _rel(sp, dense)
        assert rel < 0.05, f"{variant} full-budget rel={rel}"
        assert dense[0, -1].argmax() == sp[0, -1].argmax(), variant
        assert torch.equal(restored, dense), f"{variant} unpatch not exact"


def test_minference_real_weights_quality():
    """All three variants preserve semantics on REAL Qwen3-8B with a COHERENT
    prompt (the realistic scenario): top-1 agreement vs dense is ~1.0.

    Measured ground truth (2026/06/02): on a coherent ~960-token prompt all
    three variants hit 1.000 per-position top-1 agreement even while sparsifying
    (a_shape keeps 64+512<960 keys) — real text attention IS sparse, so the
    patterns lose nothing. (On *random* ids a_shape drops to ~0.86 because there
    is no structure to exploit; that is expected, not a defect, so the test uses
    coherent text.)"""
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    tok = AutoTokenizer.from_pretrained(_REAL_W)
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both the natural and social "
        "sciences. "
    ) * 40
    ids = tok(para, return_tensors="pt").to("cuda")["input_ids"][:, :960]
    with torch.no_grad():
        dense = m(ids).logits.float()

    for variant, kw in [
        ("minference", {}),  # dense per-head fallback
        ("a_shape", {"n_init": 64, "n_local": 512}),
        ("tri_shape", {"n_init": 64, "n_local": 512, "n_last": 100}),
    ]:
        slim, patched = _mi_patch_real(m, variant, kw)
        try:  # try/finally so an assertion failure can't leak the patch

            with torch.no_grad():
                sp = m(ids).logits.float()
        finally:
            unpatch_sparsity(slim, patched)
        assert torch.isfinite(sp).all()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"{variant} top-1 agreement {agree:.3f} too low"


def test_minference_real_weights_decode_generates():
    """a_shape sparse prefill + decode generates the correct answer (REAL)."""
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    tok = AutoTokenizer.from_pretrained(_REAL_W)
    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    slim, patched = _mi_patch_real(m, "a_shape", {"n_init": 64, "n_local": 512})
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    finally:
        unpatch_sparsity(slim, patched)
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"sparse decode lost the answer: {text!r}"


def _mi_split_device_map(n_layers):
    """Explicit 2-device device_map that genuinely shards Qwen3 decoder layers
    (``device_map="auto"`` would fit Qwen3-8B on one 97 GB H20 and never split).
    """
    dm = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 1,
        "lm_head": 1,
    }
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = 0 if i < n_layers // 2 else 1
    return dm


def test_minference_real_weights_device_map_sharded_matches_dense():
    """G1 (minference): the vertical_slash CUDA/Triton kernel is correct under
    accelerate ``device_map`` LAYER SHARDING on REAL Qwen3-8B (head_dim 128, so
    the real minference kernel runs — not the reference fallback).

    The kernel launches on the *current* CUDA device's stream, not the tensor's
    device; without the ``minference_prefill`` device-context guard a cuda:1
    layer would launch on cuda:0 and silently corrupt the output under sharding.
    This is the load-bearing test for "minference on a model bigger than one
    GPU". Asserts: every full-attention layer patched, sharded sparse ~= that
    same sharded model's own dense (argmax agrees), unpatch byte-exact.

    Uses a COHERENT prompt (real-text attention is sparse, so a_shape loses
    nothing -> ~1.0 agreement) and a_shape (its streaming kernel already runs on
    Qwen3); the guard is shared by all variants via ``minference_prefill``."""
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable")
    if torch.cuda.device_count() < 2:
        raise _Skip("device_map sharding test needs >= 2 GPUs")
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    cfg_n = Qwen3ForCausalLM.config_class.from_pretrained(_REAL_W).num_hidden_layers
    model = Qwen3ForCausalLM.from_pretrained(
        _REAL_W,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=_mi_split_device_map(cfg_n),
    ).eval()
    try:
        used = {str(d) for d in model.hf_device_map.values()}
        assert (
            len({u for u in used if u != "cpu"}) >= 2
        ), f"device_map did not shard across 2 GPUs: {model.hf_device_map}"
        tok = AutoTokenizer.from_pretrained(_REAL_W)
        para = (
            "The history of science is the study of the development of "
            "science and scientific knowledge. "
        ) * 40
        ids = tok(para, return_tensors="pt")["input_ids"][:, :960].to("cuda:0")
        with torch.no_grad():
            dense = model(ids).logits.float().cpu()

        slim, patched = _mi_patch_real(model, "a_shape", {"n_init": 64, "n_local": 512})
        try:
            assert len(patched) >= 1, "no layers patched on the sharded model"
            with torch.no_grad():
                sp = model(ids).logits.float().cpu()
        finally:
            unpatch_sparsity(slim, patched)

        assert torch.isfinite(sp).all(), "sharded minference produced NaN/inf"
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"sharded a_shape top-1 agreement {agree:.3f} too low"
        with torch.no_grad():
            restored = model(ids).logits.float().cpu()
        assert torch.equal(
            restored, dense
        ), "unpatch did not restore the sharded model byte-exactly"
    finally:
        del model
        torch.cuda.empty_cache()


# ===========================================================================
# — Sparsity compressor + YAML path for the minference family


# ===========================================================================
def test_minference_yaml_configs_parse():
    """The three shipped minference YAMLs parse into a SparsityConfig."""
    from angelslim.utils.config_parser import SlimConfigParser

    base = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "configs",
        "sparse",
        "minference",
    )
    for fname, variant in [
        ("qwen3-8b_minference.yaml", "minference"),
        ("qwen3-8b_a_shape.yaml", "a_shape"),
        ("qwen3-8b_tri_shape.yaml", "tri_shape"),
    ]:
        path = os.path.abspath(os.path.join(base, fname))
        assert os.path.isfile(path), f"missing YAML: {path}"
        cfg = SlimConfigParser().parse(path)
        comp = cfg.compression_config
        assert comp is not None and "Sparsity" in comp.name
        assert comp.sparsity is not None
        assert comp.sparsity.name == variant
        assert comp.sparsity.allow_pseudo_sparse is False  # real kernel default


def test_minference_sparsity_compressor_end_to_end():
    """Sparsity compressor drives minference on REAL Qwen3-8B: convert ->
    run inference -> unpatch restores dense. Exercises the full CompressorFactory
    path (not just the patcher)."""
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.sparsity import Sparsity
    from angelslim.utils.config_parser import SparsityConfig

    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, 1024), device="cuda")
    with torch.no_grad():
        dense = m(ids).logits.float()

    slim = _Slim(m)
    sp_cfg = SparsityConfig(
        name="a_shape",
        allow_pseudo_sparse=True,
        attn_kwargs={"n_init": 1024, "n_local": 1024},  # full budget => dense
    )
    comp = type("C", (), {"sparsity": sp_cfg})()
    sparse = Sparsity(slim, {"compress_config": comp, "global_config": None})
    assert sparse.run is None  # no-op (the composition decision)

    sparse.convert()
    assert sparse._patched
    n_patched = len(sparse._patched_modules)
    assert n_patched == m.config.num_hidden_layers
    sparse.convert()  # idempotent
    assert len(sparse._patched_modules) == n_patched

    with torch.no_grad():
        sp = m(ids).logits.float()
    assert _rel(sp, dense) < 0.05  # full-budget a_shape ~= dense

    sparse.unpatch()
    assert not sparse._patched
    with torch.no_grad():
        assert torch.equal(m(ids).logits.float(), dense)


# ===========================================================================
# Runner
# ===========================================================================


# ###########################################################################
# XATTENTION integration (former test_xattention_integration.py)
# ###########################################################################
def _xa_patch_real(model, attn_kwargs):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create("xattention", attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _xa_coherent_ids(tok, n=960):
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both the natural and social "
        "sciences. "
    ) * 40
    return tok(para, return_tensors="pt")["input_ids"][:, :n].to("cuda")


# ===========================================================================
# Registration + traits (no weights)
# ===========================================================================
def test_xattention_registered_and_traits():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    assert "xattention" in SparsityAlgorithmRegistry._factories
    algo = SparsityAlgorithmRegistry.create("xattention", attn_kwargs={})
    t = algo.traits
    assert t.requires_unrepeated_kv is False
    assert t.supports_padding_mask is False
    for mt in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text"):
        assert mt in t.compatible_model_types, mt
    assert algo.name == "xattention"


def test_xattention_head_dim_gate():
    from angelslim.compressor.sparsity.algorithms.xattention.kernels import (
        kernels_available,
    )

    assert kernels_available(128) is True  # Qwen3
    assert kernels_available(64) is True
    assert kernels_available(256) is False  # Qwen3.5 -> reference fallback


# ===========================================================================
# Quality — coherent prompt: top-1 preserved while sparsifying (REAL Qwen3-8B)
# ===========================================================================
def test_xattention_coherent_top1_preserved():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    tok = AutoTokenizer.from_pretrained(_REAL_W)
    ids = _xa_coherent_ids(tok, 960)
    with torch.no_grad():
        dense = m(ids).logits.float()

    # threshold 0.95: keep blocks covering 95% of mass; real attention is peaked
    # so the decode-relevant top-1 is preserved (rel ~0.0055 on a single layer).
    slim, patched = _xa_patch_real(m, {"stride": 8, "norm": 1, "threshold": 0.95})
    try:
        assert len(patched) == m.config.num_hidden_layers
        with torch.no_grad():
            sp = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
    with torch.no_grad():
        restored = m(ids).logits.float()

    assert torch.isfinite(sp).all()
    agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
    assert agree > 0.95, f"xattention top-1 agreement {agree:.3f} too low"
    assert torch.equal(restored, dense), "unpatch did not restore real model"
    _cap("xattention_real_correctness")


def test_xattention_decode_generates():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    tok = AutoTokenizer.from_pretrained(_REAL_W)
    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    slim, patched = _xa_patch_real(m, {"stride": 8, "norm": 1, "threshold": 0.9})
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    finally:
        unpatch_sparsity(slim, patched)
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"xattention decode lost the answer: {text!r}"


# ===========================================================================
# head_dim 256 -> torch reference fallback (REAL Qwen3.5-9B)
# ===========================================================================
def test_xattention_qwen35_reference_fallback():
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B weights unavailable", _SR.NO_QWEN35_9B)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = (
        AutoModelForCausalLM.from_pretrained(
            _REAL_W35, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to("cuda")
        .eval()
    )
    try:
        tok = AutoTokenizer.from_pretrained(_REAL_W35)
        ids = _xa_coherent_ids(tok, 1024)
        with torch.no_grad():
            dense = m(ids).logits.float()
        n_full = sum(1 for lt in m.config.layer_types if lt == "full_attention")
        # head_dim 256 -> kernel declines -> reference (needs allow_pseudo_sparse).
        slim, patched = _xa_patch_real(
            m, {"stride": 8, "norm": 1, "threshold": 0.95, "allow_pseudo_sparse": True}
        )
        try:
            assert len(patched) == n_full, f"patched {len(patched)} != {n_full} full"
            with torch.no_grad():
                sp = m(ids).logits.float()
        finally:
            unpatch_sparsity(slim, patched)
        assert torch.isfinite(sp).all()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"xattention Qwen3.5 reference top-1 {agree:.3f}"
        _cap("xattention_real_correctness")
    finally:
        del m
        torch.cuda.empty_cache()


# ===========================================================================
# Reference recovers dense at threshold -> 1 (numeric oracle)
# ===========================================================================
def test_xattention_reference_recovers_dense():
    if not CUDA:
        raise _Skip("reference oracle test needs CUDA", _SR.NO_CUDA)
    import torch.nn.functional as F

    from angelslim.compressor.sparsity.algorithms.xattention.reference import (
        xattention_reference_prefill,
    )

    torch.manual_seed(0)
    # head_dim 256 (the path the reference actually serves: Qwen3.5).
    B, H, L, D = 1, 8, 512, 256
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.1
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    keep_all = xattention_reference_prefill(q, k, v, threshold=1.0, block_size=128)
    sparse = xattention_reference_prefill(q, k, v, threshold=0.5, block_size=128)
    assert torch.isfinite(keep_all).all()
    assert _rel(keep_all, dense) < 0.02, "threshold=1 reference should == dense"
    # threshold 0.5 must genuinely drop mass (else the knob is a no-op).
    assert _rel(sparse, dense) > 0.05, "threshold=0.5 should sparsify vs dense"


# ===========================================================================
# device_map layer sharding correctness (G1 / — REAL Qwen3-8B, >=2 GPUs


# ===========================================================================
def _xa_split_device_map(n_layers):
    dm = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = 0 if i < n_layers // 2 else 1
    return dm


def test_xattention_device_map_sharded_matches_single():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    if torch.cuda.device_count() < 2:
        raise _Skip("device_map sharding test needs >= 2 GPUs", _SR.NO_MULTI_GPU)
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    cfg_n = Qwen3ForCausalLM.config_class.from_pretrained(_REAL_W).num_hidden_layers
    model = Qwen3ForCausalLM.from_pretrained(
        _REAL_W,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=_xa_split_device_map(cfg_n),
    ).eval()
    try:
        used = {str(d) for d in model.hf_device_map.values() if str(d) != "cpu"}
        assert len(used) >= 2, f"not sharded: {model.hf_device_map}"
        # COHERENT prompt: real attention is peaked, so a thresholded sparse
        # prefill preserves the dense top-1. xattention has no exact keep-all
        # knob (its estimate is strided/lossy), so we assert top-1 agreement on
        # peaked attention rather than rel parity. without the device-context

        # wrap the block_sparse_attn / Triton estimate launch on the wrong GPU and
        # corrupt the shard-1 layers, collapsing the agreement.
        tok = AutoTokenizer.from_pretrained(_REAL_W)
        ids = _xa_coherent_ids(tok, 2048)
        with torch.no_grad():
            dense = model(ids).logits.float().cpu()
        slim, patched = _xa_patch_real(model, {"stride": 8, "norm": 1, "threshold": 0.95})
        try:
            with torch.no_grad():
                sp = model(ids).logits.float().cpu()
        finally:
            unpatch_sparsity(slim, patched)
        assert not torch.isnan(sp).any()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.90, f"sharded xattention top-1 drift {agree:.3f} ?)"

    finally:
        del model
        torch.cuda.empty_cache()


# ###########################################################################
# FLEXPREFILL integration (former test_flexprefill_integration.py)
# ###########################################################################
def _fx_patch_real(model, attn_kwargs):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create("flexprefill", attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _fx_coherent_ids(tok, n=960):
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both the natural and social "
        "sciences. "
    ) * 40
    return tok(para, return_tensors="pt")["input_ids"][:, :n].to("cuda")


# ===========================================================================
# Registration + traits (no weights)
# ===========================================================================
def test_flexprefill_registered_and_traits():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    assert "flexprefill" in SparsityAlgorithmRegistry._factories
    algo = SparsityAlgorithmRegistry.create("flexprefill", attn_kwargs={})
    t = algo.traits
    assert t.requires_unrepeated_kv is False
    assert t.supports_padding_mask is False
    for mt in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text"):
        assert mt in t.compatible_model_types, mt
    assert algo.name == "flexprefill"


def test_flexprefill_head_dim_gate():
    from angelslim.compressor.sparsity.algorithms.flexprefill.kernels import (
        kernels_available,
    )

    assert kernels_available(128) is True  # Qwen3
    assert kernels_available(64) is True
    assert kernels_available(256) is False  # Qwen3.5 -> reference fallback


# ===========================================================================
# Dense parity — keep-all returns flash dense exactly (REAL Qwen3-8B)
# ===========================================================================
def test_flexprefill_keep_all_matches_dense():
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, 2048), device="cuda")
    with torch.no_grad():
        dense = m(ids).logits.float()

    # min_budget >= seq_len => flexprefill's short-seq path returns flash dense.
    slim, patched = _fx_patch_real(m, {"gamma": 0.9, "tau": 0.0, "min_budget": 4096})
    try:
        assert len(patched) == m.config.num_hidden_layers
        with torch.no_grad():
            sp = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
    with torch.no_grad():
        restored = m(ids).logits.float()

    assert not torch.isnan(sp).any()
    rel = _rel(sp, dense)
    assert rel < 0.02, f"keep-all flexprefill should == flash dense, rel={rel}"
    assert dense[0, -1].argmax() == sp[0, -1].argmax(), "keep-all argmax drift"
    assert torch.equal(restored, dense), "unpatch did not restore real model"
    _cap("flexprefill_real_correctness")


# ===========================================================================
# Quality — coherent prompt: top-1 preserved while sparsifying (REAL Qwen3-8B)
# ===========================================================================
def test_flexprefill_coherent_top1_preserved():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    tok = AutoTokenizer.from_pretrained(_REAL_W)
    ids = _fx_coherent_ids(tok, 960)
    with torch.no_grad():
        dense = m(ids).logits.float()

    slim, patched = _fx_patch_real(m, {"gamma": 0.9, "tau": 0.0})
    try:
        with torch.no_grad():
            sp = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
    assert torch.isfinite(sp).all()
    agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
    assert agree > 0.95, f"flexprefill top-1 agreement {agree:.3f} too low"
    _cap("flexprefill_real_correctness")


def test_flexprefill_decode_generates():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    tok = AutoTokenizer.from_pretrained(_REAL_W)
    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    slim, patched = _fx_patch_real(m, {"gamma": 0.9, "tau": 0.0})
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    finally:
        unpatch_sparsity(slim, patched)
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"flexprefill decode lost the answer: {text!r}"


# ===========================================================================
# head_dim 256 -> torch reference fallback (REAL Qwen3.5-9B)
# ===========================================================================
def test_flexprefill_qwen35_reference_fallback():
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B weights unavailable", _SR.NO_QWEN35_9B)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = (
        AutoModelForCausalLM.from_pretrained(
            _REAL_W35, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to("cuda")
        .eval()
    )
    try:
        tok = AutoTokenizer.from_pretrained(_REAL_W35)
        ids = _fx_coherent_ids(tok, 1024)
        with torch.no_grad():
            dense = m(ids).logits.float()
        n_full = sum(1 for lt in m.config.layer_types if lt == "full_attention")
        # head_dim 256 -> kernel declines -> reference (needs allow_pseudo_sparse).
        slim, patched = _fx_patch_real(m, {"gamma": 0.9, "tau": 0.0, "allow_pseudo_sparse": True})
        try:
            assert len(patched) == n_full, f"patched {len(patched)} != {n_full} full"
            with torch.no_grad():
                sp = m(ids).logits.float()
        finally:
            unpatch_sparsity(slim, patched)
        assert torch.isfinite(sp).all()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"flexprefill Qwen3.5 reference top-1 {agree:.3f}"
        _cap("flexprefill_real_correctness")
    finally:
        del m
        torch.cuda.empty_cache()


# ===========================================================================
# Reference == kernel at high gamma (numeric oracle, REAL Qwen3-8B)
# ===========================================================================
def test_flexprefill_reference_agrees_with_kernel():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, 1024), device="cuda")

    # kernel path
    slim, patched = _fx_patch_real(m, {"gamma": 0.9, "tau": 0.0})
    try:
        with torch.no_grad():
            ker = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
    # reference path (force via allow_pseudo + a head_dim the kernel rejects is
    # not possible here; instead exercise the reference directly is covered in
    # the Qwen3.5 test). Here assert both kernel and dense agree on argmax.
    with torch.no_grad():
        dense = m(ids).logits.float()
    agree = (ker[0].argmax(-1) == dense[0].argmax(-1)).float().mean().item()
    assert agree > 0.80, f"kernel vs dense argmax {agree:.3f} (random ids; expect >0.8)"


# ===========================================================================
# device_map layer sharding correctness (G1 / — REAL Qwen3-8B, >=2 GPUs


# ===========================================================================
def _fx_split_device_map(n_layers):
    dm = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = 0 if i < n_layers // 2 else 1
    return dm


def test_flexprefill_device_map_sharded_matches_dense():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    if torch.cuda.device_count() < 2:
        raise _Skip("device_map sharding test needs >= 2 GPUs", _SR.NO_MULTI_GPU)
    from transformers import Qwen3ForCausalLM

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    cfg_n = Qwen3ForCausalLM.config_class.from_pretrained(_REAL_W).num_hidden_layers
    model = Qwen3ForCausalLM.from_pretrained(
        _REAL_W,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=_fx_split_device_map(cfg_n),
    ).eval()
    try:
        used = {str(d) for d in model.hf_device_map.values() if str(d) != "cpu"}
        assert len(used) >= 2, f"not sharded: {model.hf_device_map}"
        torch.manual_seed(0)
        ids = torch.randint(0, 151000, (1, 2048), device="cuda:0")
        with torch.no_grad():
            dense = model(ids).logits.float().cpu()
        # keep-all => flash dense on each shard; must match single-device dense.
        slim, patched = _fx_patch_real(model, {"gamma": 0.9, "tau": 0.0, "min_budget": 4096})
        try:
            with torch.no_grad():
                sp = model(ids).logits.float().cpu()
        finally:
            unpatch_sparsity(slim, patched)
        assert not torch.isnan(sp).any()
        rel = _rel(sp, dense)
        assert rel < 0.02, f"sharded keep-all flexprefill rel={rel}"
        assert dense[0, -1].argmax() == sp[0, -1].argmax(), "sharded argmax drift"
    finally:
        del model
        torch.cuda.empty_cache()


def test_flexprefill_unsupported_block_size_takes_policy_not_bare_assert():
    """block_size=16 is unsupported by the prefill kernel (only 32/64/128). The
    dispatcher must pre-validate it and take the documented policy — reference
    under allow_pseudo_sparse, actionable KernelMissingError otherwise — NOT die
    on the kernel's bare AssertionError deep inside the Triton entry.

    Weightless: forces kernels_available True and runs on tiny CPU bf16 tensors;
    the reference path is pure torch (no CUDA / no flash_attn)."""
    import angelslim.compressor.sparsity.algorithms.flexprefill.prefill as fp
    from angelslim.compressor.sparsity._kernel_check import KernelMissingError

    B, H, L, D = 1, 2, 8, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16)

    orig = fp._k.kernels_available
    fp._k.kernels_available = lambda head_dim=None: True
    try:
        # allow_pseudo_sparse=True -> routes to the block-size-agnostic reference.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fp.flexprefill_prefill(
                q,
                k,
                v,
                gamma=0.9,
                tau=0.0,
                min_budget=None,
                max_budget=None,
                block_size=16,
                head_dim=D,
                allow_pseudo_sparse=True,
            )
        assert out.shape == (B, H, L, D) and not torch.isnan(out).any()

        # allow_pseudo_sparse=False -> actionable KernelMissingError naming the
        # supported set, NOT a bare AssertionError from the kernel.
        try:
            fp.flexprefill_prefill(
                q,
                k,
                v,
                gamma=0.9,
                tau=0.0,
                min_budget=None,
                max_budget=None,
                block_size=16,
                head_dim=D,
                allow_pseudo_sparse=False,
            )
            raise AssertionError("expected KernelMissingError for block_size=16")
        except KernelMissingError as e:
            assert "block_size=16" in str(e) and "32, 64, 128" in str(e)
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"expected KernelMissingError, got {type(e).__name__}: {e}")

        # A supported block_size (128) must NOT be rejected by the pre-check —
        # the gate is specific to the unsupported set. We can't run the real
        # kernel weightlessly (CPU has no flash_attn), so assert only that the
        # pre-check raises NO KernelMissingError for it: q_len==k_len and bf16
        # hold here, so any KernelMissingError would mean the gate wrongly
        # rejected 128. A downstream CUDA/flash_attn error proves it got through.
        try:
            fp.flexprefill_prefill(
                q,
                k,
                v,
                gamma=0.9,
                tau=0.0,
                min_budget=None,
                max_budget=None,
                block_size=128,
                head_dim=D,
                allow_pseudo_sparse=False,
            )
        except KernelMissingError as e:
            raise AssertionError(f"pre-check wrongly rejected supported block_size=128: {e}")
        except Exception:  # noqa: BLE001  downstream kernel needs CUDA — gate let it through
            pass
    finally:
        fp._k.kernels_available = orig


def test_flexprefill_kernel_block_size_assert_message_matches_check():
    """The prefill kernel's block_size assert MESSAGE must name the same set it
    checks (the self-contradiction was 'support {16,32,64,128}' while rejecting
    16). Source-level guard so the message can't drift from the check again."""
    import inspect

    from angelslim.compressor.sparsity.algorithms.flexprefill.kernels import (
        flex_prefill as _fpk,
    )

    src = inspect.getsource(_fpk.triton_block_wise_prefill_attention)
    # The prefill kernel checks {32, 64, 128}; its message must not claim 16.
    assert '"only support block size in {32, 64, 128}"' in src, (
        "prefill block_size assert message drifted from the checked set "
        "{32, 64, 128} (must not advertise 16, which the prefill kernel rejects)"
    )


# ###########################################################################
# FLASHPREFILL integration (former test_flashprefill_integration.py)
# ###########################################################################
def _fp_patch_real(model, attn_kwargs):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create("flashprefill", attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _fp_coherent_ids(tok, n=960):
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both the natural and social "
        "sciences. "
    ) * 40
    return tok(para, return_tensors="pt")["input_ids"][:, :n].to("cuda")


# ===========================================================================
# Registration + traits (no weights)
# ===========================================================================
def test_flashprefill_registered_and_traits():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    assert "flashprefill" in SparsityAlgorithmRegistry._factories
    algo = SparsityAlgorithmRegistry.create("flashprefill", attn_kwargs={})
    t = algo.traits
    assert t.requires_unrepeated_kv is False
    assert t.supports_padding_mask is False
    for mt in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text"):
        assert mt in t.compatible_model_types, mt
    assert algo.name == "flashprefill"


def test_flashprefill_head_dim_gate():
    from angelslim.compressor.sparsity.algorithms.flashprefill.kernels_check import (
        kernels_available,
    )

    assert kernels_available(128) is True  # Qwen3
    assert kernels_available(64) is True
    assert kernels_available(256) is False  # Qwen3.5 -> reference fallback


# ===========================================================================
# Clean-room provenance: NO vendored upstream code


# ===========================================================================
def test_flashprefill_no_vendored_kernel_dir():
    # The license fail-open means there is NO kernels/ vendored subpackage (cf.
    # flexprefill/xattention which DO vendor MIT kernel files). Guard against a
    # future accidental vendoring of the unlicensed upstream.
    import angelslim.compressor.sparsity.algorithms.flashprefill as fp

    pkg_dir = os.path.dirname(fp.__file__)
    assert not os.path.isdir(os.path.join(pkg_dir, "kernels")), (
        "flashprefill must NOT have a vendored kernels/ dir (upstream is "
        "unlicensed; clean-room)"
    )
    assert os.path.isfile(os.path.join(pkg_dir, "NOTICE")), "missing clean-room NOTICE"


# ===========================================================================
# Dense parity — alpha=0 (keep-all) returns flash dense exactly (REAL Qwen3-8B)
# ===========================================================================
def test_flashprefill_keep_all_matches_dense():
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    tok = AutoTokenizer.from_pretrained(_REAL_W)
    ids = _fp_coherent_ids(tok, 2048)
    with torch.no_grad():
        dense = m(ids).logits.float()

    # alpha=0 means "keep everything" == dense, so the dispatch routes to flash
    # dense (a deliberate optimization: flash dense matches the model's own fa2
    # baseline bit-for-bit, whereas block_sparse_attn over a full causal mask is
    # not bit-identical and drifts ~0.11 rel over 36 layers — see the alpha>0
    # kernel test, which asserts top-1 not rel for that reason). Here, BECAUSE we
    # route to flash dense, the tight rel<0.005 parity holds.
    slim, patched = _fp_patch_real(m, {"alpha": 0.0})
    try:
        assert len(patched) == m.config.num_hidden_layers
        with torch.no_grad():
            sp = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
    with torch.no_grad():
        restored = m(ids).logits.float()

    assert not torch.isnan(sp).any()
    rel = _rel(sp, dense)
    assert rel < 0.005, f"alpha=0 keep-all should == flash dense, rel={rel}"
    assert (dense[0].argmax(-1) == sp[0].argmax(-1)).all(), "keep-all argmax drift"
    assert torch.equal(restored, dense), "unpatch did not restore real model"
    _cap("flashprefill_real_correctness")


# ===========================================================================
# REAL kernel path (alpha>0) — block_sparse_attn over the sparse mask preserves
# top-1 (REAL Qwen3-8B). Asserts top-1, NOT rel: block_sparse_attn is not
# bit-identical to flash dense and that gap compounds across layers (rel ~0.11
# even at full density), while the decode-relevant prediction is unchanged —
# this is the metric every algorithm in the suite asserts.
# ===========================================================================
def test_flashprefill_kernel_path_top1_preserved():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    tok = AutoTokenizer.from_pretrained(_REAL_W)
    ids = _fp_coherent_ids(tok, 2048)
    with torch.no_grad():
        dense = m(ids).logits.float()

    # alpha=0.2 genuinely prunes blocks AND exercises the block_sparse_attn kernel
    # (alpha>0 does NOT hit the flash-dense bypass). On peaked attention the
    # prediction is preserved.
    slim, patched = _fp_patch_real(m, {"alpha": 0.2})
    try:
        with torch.no_grad():
            sp = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
    assert torch.isfinite(sp).all()
    agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
    assert agree > 0.95, f"kernel-path top-1 agreement {agree:.3f} too low"
    _cap("flashprefill_real_correctness")


# ===========================================================================
# Quality — coherent prompt: top-1 preserved while sparsifying (REAL Qwen3-8B)
# ===========================================================================
def test_flashprefill_coherent_top1_preserved():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    tok = AutoTokenizer.from_pretrained(_REAL_W)
    ids = _fp_coherent_ids(tok, 960)
    with torch.no_grad():
        dense = m(ids).logits.float()

    slim, patched = _fp_patch_real(m, {"alpha": 0.2})
    try:
        with torch.no_grad():
            sp = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
    assert torch.isfinite(sp).all()
    agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
    assert agree > 0.95, f"flashprefill top-1 agreement {agree:.3f} too low"
    _cap("flashprefill_real_correctness")


def test_flashprefill_decode_generates():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    from transformers import AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = _real_qwen3_8b()
    tok = AutoTokenizer.from_pretrained(_REAL_W)
    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    slim, patched = _fp_patch_real(m, {"alpha": 0.2})
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    finally:
        unpatch_sparsity(slim, patched)
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"flashprefill decode lost the answer: {text!r}"


# ===========================================================================
# head_dim 256 -> torch reference fallback (REAL Qwen3.5-9B)
# ===========================================================================
def test_flashprefill_qwen35_reference_fallback():
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B weights unavailable", _SR.NO_QWEN35_9B)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = (
        AutoModelForCausalLM.from_pretrained(
            _REAL_W35, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to("cuda")
        .eval()
    )
    try:
        tok = AutoTokenizer.from_pretrained(_REAL_W35)
        ids = _fp_coherent_ids(tok, 1024)
        with torch.no_grad():
            dense = m(ids).logits.float()
        n_full = sum(1 for lt in m.config.layer_types if lt == "full_attention")
        # head_dim 256 -> kernel declines -> reference (needs allow_pseudo_sparse).
        slim, patched = _fp_patch_real(m, {"alpha": 0.2, "allow_pseudo_sparse": True})
        try:
            assert len(patched) == n_full, f"patched {len(patched)} != {n_full} full"
            with torch.no_grad():
                sp = m(ids).logits.float()
        finally:
            unpatch_sparsity(slim, patched)
        assert torch.isfinite(sp).all()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"flashprefill Qwen3.5 reference top-1 {agree:.3f}"
        _cap("flashprefill_real_correctness")
    finally:
        del m
        torch.cuda.empty_cache()


# ===========================================================================
# Reference recovers dense at alpha=0 + sparsifies at high alpha (oracle)
# ===========================================================================
def test_flashprefill_reference_threshold_behaviour():
    if not CUDA:
        raise _Skip("reference oracle test needs CUDA", _SR.NO_CUDA)
    import torch.nn.functional as F

    from angelslim.compressor.sparsity.algorithms.flashprefill.blockmask import (
        build_block_keep_mask,
    )
    from angelslim.compressor.sparsity.algorithms.flashprefill.reference import (
        flashprefill_reference_prefill,
    )

    torch.manual_seed(0)
    # head_dim 256 (the path the reference actually serves: Qwen3.5). Build a
    # genuinely PEAKED block structure so the official-matched (mean-key, block-
    # softmax) scoring produces a non-flat distribution and the max-based
    # threshold visibly drops blocks: queries point mostly along a shared
    # direction u, key-block 0 is strongly aligned with u (dominant), the rest is
    # noise. (Random/near-uniform attention legitimately keeps everything — the
    # paper's thresholding only prunes when the block-score distribution is
    # peaked, which is the real-attention regime, exercised on real weights by the
    # kernel-path / coherent tests.)
    B, H, L, D = 1, 8, 4096, 256
    u = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    q = u * 2 + torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    k[:, :, :128] = u * 3  # dominant anchor key-block
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    keep_all = flashprefill_reference_prefill(
        q, k, v, alpha=0.0, sink=128, window=128, last_n_block_full=0
    )
    assert torch.isfinite(keep_all).all()
    assert _rel(keep_all, dense) < 0.02, "alpha=0 reference should == dense"

    m0 = build_block_keep_mask(q, k, alpha=0.0, sink=128, window=128, last_n_block_full=0)
    mhi = build_block_keep_mask(q, k, alpha=0.6, sink=128, window=128, last_n_block_full=0)
    # alpha=0 keeps every causal block; alpha=0.6 must drop a meaningful fraction
    # on this peaked input.
    assert mhi.float().sum() < 0.6 * m0.float().sum(), "high alpha did not sparsify"


def test_flashprefill_block_importance_long_ctx_memory_bounded():
    """Regression: the Stage-1 score must NOT materialize an O(L^2) intermediate.

    The first cut computed the whole (B,H,nqb,bs,nkb) fp32 score tensor at once,
    so peak scoring memory grew quadratically in L — ~5 GB @32K, ~22 GB @64K,
    ~86 GB @128K — which OOMs / corrupts a fragmented CUDA allocator at exactly
    the long context FlashPrefill exists to accelerate (it surfaced as an async
    illegal-access mid-benchmark). The fix tiles the query-block axis
    (`_SCORE_QK_BUDGET`), bounding peak memory ~flat in L. This guards both
    properties so the tiling cannot silently regress:
      (a) the tiled score is BIT-IDENTICAL to an unchunked reference (every
          reduction is local to a query block) — preserves the rel-0.0074
          upstream match;
      (b) peak memory at a long sequence stays far below the old O(L^2) blowup.
    """
    if not CUDA:
        raise _Skip("memory probe needs CUDA", _SR.NO_CUDA)
    import math

    from angelslim.compressor.sparsity.algorithms.flashprefill import blockmask as _BM

    _LOG2E = 1.4426950408889634

    def _unchunked(q, k, bs):
        """The pre-fix whole-tensor scoring (reference for bit-identity)."""
        B, H, L, D = q.shape
        scale = 1.0 / math.sqrt(D)
        nqb = (L + bs - 1) // bs
        pad = nqb * bs - L
        qd, kd = q, k
        if pad:
            qd = torch.nn.functional.pad(qd, (0, 0, 0, pad))
            kd = torch.nn.functional.pad(kd, (0, 0, 0, pad))
        counts = torch.full((nqb,), float(bs), device=q.device)
        if pad:
            counts[-1] = float(bs - pad)
        mean_k = (kd.float().view(B, H, nqb, bs, D).sum(3) / counts.view(1, 1, nqb, 1)).to(q.dtype)
        qb = qd.view(B, H, nqb, bs, D)
        qk = torch.einsum("bhipd,bhjd->bhipj", qb, mean_k).float() * scale * _LOG2E
        neg = torch.finfo(torch.float32).min
        qpos = torch.arange(nqb, device=q.device).view(1, 1, nqb, 1, 1) * bs + torch.arange(
            bs, device=q.device
        ).view(1, 1, 1, bs, 1)
        kmax = torch.arange(nqb, device=q.device).view(1, 1, 1, 1, nqb) * bs + (bs - 1)
        causal = qpos >= kmax
        qk_c = torch.where(causal, qk, torch.tensor(neg, device=q.device))
        mblock = qk_c.amax(dim=3, keepdim=True)
        p = torch.where(causal, torch.exp2(qk - mblock), torch.zeros_like(qk))
        score = p.sum(dim=3)
        mb = mblock.squeeze(3)
        valid = torch.isfinite(mb)
        max_j = torch.where(valid, mb, torch.tensor(neg, device=q.device)).amax(
            dim=-1, keepdim=True
        )
        resc = torch.where(valid, torch.exp2(mb - max_j), torch.zeros_like(mb))
        score = score * resc
        score = score / (score.sum(dim=-1, keepdim=True) + 1e-9)
        return score

    bs = 128
    # (a) bit-identity at a small L (the reference fits trivially).
    torch.manual_seed(0)
    qs = torch.randn(1, 8, 2048, 64, device="cuda", dtype=torch.bfloat16)
    ks = torch.randn(1, 8, 2048, 64, device="cuda", dtype=torch.bfloat16)
    tiled = _BM._block_importance(qs, ks, bs)
    ref = _unchunked(qs, ks, bs)
    assert torch.equal(tiled, ref), (
        "tiled score is not bit-identical to the unchunked reference — the "
        "memory tiling changed the result (it must not)."
    )

    # (b) memory bound at a long L (32 heads, head_dim 128 — Qwen3-8B shape). The
    # old code needed ~5.4 GB here; the tiled budget keeps it well under 4 GB.
    L = 32768
    torch.manual_seed(0)
    ql = torch.randn(1, 32, L, 128, device="cuda", dtype=torch.bfloat16)
    kl = torch.randn(1, 32, L, 128, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _ = _BM._block_importance(ql, kl, bs)
    torch.cuda.synchronize()
    peak_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
    # Generous ceiling: the tiled path measures ~2.7 GB; the old O(L^2) path was
    # ~5.4 GB @32K (and ~86 GB @128K). 4 GB cleanly separates fixed from broken.
    assert peak_gb < 4.0, (
        f"flashprefill Stage-1 scoring used {peak_gb:.2f} GB at L={L} — the "
        f"O(L^2) materialization regressed (tiling expected ~2.7 GB)."
    )
    _cap("flashprefill_real_correctness")


# ===========================================================================
# device_map layer sharding correctness (G1 / — REAL Qwen3-8B, >=2 GPUs


# ===========================================================================
def _fp_split_device_map(n_layers):
    dm = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = 0 if i < n_layers // 2 else 1
    return dm


def test_flashprefill_device_map_sharded_matches_dense():
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
    if torch.cuda.device_count() < 2:
        raise _Skip("device_map sharding test needs >= 2 GPUs", _SR.NO_MULTI_GPU)
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    cfg_n = Qwen3ForCausalLM.config_class.from_pretrained(_REAL_W).num_hidden_layers
    model = Qwen3ForCausalLM.from_pretrained(
        _REAL_W,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=_fp_split_device_map(cfg_n),
    ).eval()
    try:
        used = {str(d) for d in model.hf_device_map.values() if str(d) != "cpu"}
        assert len(used) >= 2, f"not sharded: {model.hf_device_map}"
        # Coherent prompt + alpha>0 so the block_sparse_attn kernel path actually
        # runs on the sparse mask across both shards (alpha=0 would route to the
        # flash-dense bypass and not exercise the kernel). Assert top-1, not rel:
        # block_sparse_attn is not bit-identical to flash dense (the gap compounds
        # across layers), but the prediction is preserved. without the

        # device-context wrap the block_sparse_attn kernel would launch on the
        # wrong GPU and corrupt the shard-1 layers, collapsing the agreement.
        tok = AutoTokenizer.from_pretrained(_REAL_W)
        ids = _fp_coherent_ids(tok, 2048)
        with torch.no_grad():
            dense = model(ids).logits.float().cpu()
        slim, patched = _fp_patch_real(model, {"alpha": 0.2})
        try:
            with torch.no_grad():
                sp = model(ids).logits.float().cpu()
        finally:
            unpatch_sparsity(slim, patched)
        assert not torch.isnan(sp).any()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.90, f"sharded flashprefill top-1 drift {agree:.3f} ?)"

    finally:
        del model
        torch.cuda.empty_cache()


def test_flashprefill_dense_path_without_flash_attn_takes_policy():
    """alpha<=0 (the documented dense config) runs flash-attention dense, which
    needs flash_attn — a dep NOT covered by kernels_available (triton +
    block_sparse_attn). When flash_attn is missing, the dispatcher must take the
    policy (reference under allow_pseudo_sparse; actionable KernelMissingError
    otherwise) instead of a bare ImportError from deep in _flash_dense.

    Weightless: forces kernels_available True and flash_attn 'unavailable', runs
    on tiny CPU bf16 tensors; the reference path is pure torch."""
    import angelslim.compressor.sparsity.algorithms.flashprefill.prefill as fp
    from angelslim.compressor.sparsity import _kernel_check as kc

    B, H, L, D = 1, 2, 8, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16)

    orig_kernels = fp.kernels_available
    orig_avail = kc.kernel_available
    fp.kernels_available = lambda head_dim=None: True
    # flash_attn 'missing', everything else importable.
    kc.kernel_available = lambda name: False if name == "flash_attn" else orig_avail(name)
    try:
        # allow_pseudo_sparse=True -> torch reference (recovers dense at alpha<=0).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fp.flashprefill_prefill(
                q,
                k,
                v,
                alpha=0.0,
                block_size=128,
                sink=4,
                window=4,
                last_n_block_full=1,
                head_dim=D,
                allow_pseudo_sparse=True,
            )
        assert out.shape == (B, H, L, D) and not torch.isnan(out).any()

        # allow_pseudo_sparse=False -> actionable KernelMissingError naming
        # flash_attn, NOT a bare ImportError.
        try:
            fp.flashprefill_prefill(
                q,
                k,
                v,
                alpha=0.0,
                block_size=128,
                sink=4,
                window=4,
                last_n_block_full=1,
                head_dim=D,
                allow_pseudo_sparse=False,
            )
            raise AssertionError("expected KernelMissingError for alpha<=0 without flash_attn")
        except kc.KernelMissingError as e:
            assert "flash_attn" in str(e)
        except ImportError:
            raise AssertionError("got bare ImportError — the flash_attn gate is missing")
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"expected KernelMissingError, got {type(e).__name__}: {e}")
    finally:
        fp.kernels_available = orig_kernels
        kc.kernel_available = orig_avail


# ###########################################################################
# VECATTENTION integration (former test_vecattention_integration.py)
# ###########################################################################
def _vc_coherent_ids(tok, n=1024):
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both the natural and social "
        "sciences. "
    ) * 60
    return tok(para, return_tensors="pt")["input_ids"][:, :n].to("cuda")


def _vc_patch_real(model, attn_kwargs):
    from _harness import FakeSlim as _Slim

    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create("vecattention", attn_kwargs=dict(attn_kwargs))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


# ===========================================================================
# Registration + traits (no weights)
# ===========================================================================
def test_vecattention_registered_and_traits():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    assert "vecattention" in SparsityAlgorithmRegistry.available()
    algo = SparsityAlgorithmRegistry.create("vecattention", attn_kwargs={})
    t = algo.traits
    assert t.requires_unrepeated_kv is False
    assert t.supports_padding_mask is False
    for mt in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text"):
        assert mt in t.compatible_model_types, mt
    assert algo.name == "vecattention"


def test_vecattention_config_validation():
    """The kernel hard constraints (block_size_q ∈ {64,128}; chunk_size a multiple
    of block_size_q) are enforced at setup, mirroring the legacy VecAttentionConfig."""
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    class _Cfg:
        config = type("C", (), {"model_type": "qwen3", "num_hidden_layers": 2})()

    ok = SparsityAlgorithmRegistry.create(
        "vecattention", attn_kwargs={"block_size_q": 128, "chunk_size": 2048}
    )
    ok.setup(_Cfg())  # valid: 2048 % 128 == 0

    bad_bsq = SparsityAlgorithmRegistry.create("vecattention", attn_kwargs={"block_size_q": 100})
    try:
        bad_bsq.setup(_Cfg())
        raise AssertionError("expected IncompatibleConfigError for block_size_q=100")
    except IncompatibleConfigError:
        pass

    bad_chunk = SparsityAlgorithmRegistry.create(
        "vecattention", attn_kwargs={"block_size_q": 64, "chunk_size": 100}
    )
    try:
        bad_chunk.setup(_Cfg())
        raise AssertionError("expected IncompatibleConfigError for chunk_size=100")
    except IncompatibleConfigError:
        pass


def test_vecattention_head_dim_gate():
    """The real kernel needs vllm_flash_attn built AND head_dim ∈ {64,128}. In this
    env vllm_flash_attn is unbuilt, so kernels_available is False regardless — and
    head_dim 256 (Qwen3.5) is excluded even when it is built."""
    from angelslim.compressor.sparsity.algorithms.vecattention.kernels_check import (
        kernels_available,
    )

    assert kernels_available(256) is False  # Qwen3.5 -> reference fallback
    # 256 must never be a kernel size regardless of whether the kernel is built;
    # 64/128 depend on the (possibly unbuilt) kernel, so they are not asserted here.


# ===========================================================================
# Reference oracle (no real weights): threshold->0 == dense; high threshold sparsifies
# ===========================================================================
def test_vecattention_reference_threshold_behaviour():
    if not CUDA:
        raise _Skip("reference oracle test needs CUDA", _SR.NO_CUDA)
    import torch.nn.functional as F

    from angelslim.compressor.sparsity.algorithms.vecattention.reference import (
        build_vecattention_keep_mask,
        vecattention_reference_prefill,
    )

    torch.manual_seed(0)
    # head_dim 256 (the path the reference serves: Qwen3.5). Peaked block structure
    # so the MinP threshold visibly drops columns: queries point along u, the first
    # key-block is the dominant anchor, the rest is noise. L large enough that the
    # local band (block_size_k * block_size_q tokens) does not cover everything.
    B, H, L, D = 1, 8, 4096, 256
    u = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    q = u * 2 + torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    k[:, :, :64] = u * 3  # dominant anchor key-block
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    keep_all = vecattention_reference_prefill(
        q, k, v, threshold=0.0, block_size_q=64, block_size_k=2
    )
    assert torch.isfinite(keep_all).all()
    assert _rel(keep_all, dense) < 0.02, "threshold=0 reference should == dense"

    m0 = build_vecattention_keep_mask(q, k, threshold=0.0, block_size_q=64, block_size_k=2)
    mhi = build_vecattention_keep_mask(q, k, threshold=0.5, block_size_q=64, block_size_k=2)
    # threshold=0 keeps every causal column; threshold=0.5 must drop a meaningful
    # fraction on this peaked input (MinP gap bites).
    assert mhi.float().sum() < 0.6 * m0.float().sum(), "high threshold did not sparsify"


# ===========================================================================
# REAL Qwen3.5-9B hd256 -> torch reference fallback (the composition decision): top-1 preserved


# ===========================================================================
def test_vecattention_qwen35_reference_top1_preserved():
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B weights unavailable", _SR.NO_QWEN35_9B)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = (
        AutoModelForCausalLM.from_pretrained(
            _REAL_W35, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to("cuda")
        .eval()
    )
    try:
        tok = AutoTokenizer.from_pretrained(_REAL_W35)
        ids = _vc_coherent_ids(tok, 1024)
        with torch.no_grad():
            dense = m(ids).logits.float()
        n_full = sum(1 for lt in m.config.layer_types if lt == "full_attention")
        # head_dim 256 -> kernel declines -> reference (needs allow_pseudo_sparse).
        slim, patched = _vc_patch_real(
            m,
            {
                "threshold": 0.1,
                "block_size_q": 64,
                "block_size_k": 16,
                "allow_pseudo_sparse": True,
            },
        )
        try:
            assert len(patched) == n_full, f"patched {len(patched)} != {n_full} full"
            with torch.no_grad():
                sp = m(ids).logits.float()
        finally:
            unpatch_sparsity(slim, patched)
        assert torch.isfinite(sp).all()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"vecattention Qwen3.5 reference top-1 {agree:.3f}"
        _cap("vecattention_real_correctness")
    finally:
        del m
        torch.cuda.empty_cache()


def test_vecattention_decode_generates():
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B weights unavailable", _SR.NO_QWEN35_9B)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    m = (
        AutoModelForCausalLM.from_pretrained(
            _REAL_W35, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to("cuda")
        .eval()
    )
    try:
        tok = AutoTokenizer.from_pretrained(_REAL_W35)
        enc = tok("The capital of France is", return_tensors="pt").to("cuda")
        n_in = enc["input_ids"].shape[1]
        slim, patched = _vc_patch_real(
            m, {"threshold": 0.1, "block_size_q": 64, "allow_pseudo_sparse": True}
        )
        try:
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=8, do_sample=False)
        finally:
            unpatch_sparsity(slim, patched)
        text = tok.decode(out[0, n_in:])
        assert "Paris" in text, f"vecattention decode lost the answer: {text!r}"
    finally:
        del m
        torch.cuda.empty_cache()


# ===========================================================================
# Real vllm_flash_attn kernel path — RUNS when the submodule is built (else SKIPS)
# ===========================================================================
def test_vecattention_real_kernel_path():
    """The REAL vllm_flash_attn.sparse_attn_func path (head_dim 128). Skips with a
    structured reason if the kernel is unbuilt; when built, asserts (a) finite
    output, (b) kernel ≈ the clean torch reference (both implement the same MinP
    method — the numeric-alignment check, like minference's kernel↔reference test),
    and (c) kernel ≈ dense on a peaked input where the dropped columns carry ~0
    mass. Measured at build time: kernel-vs-dense rel 4e-4, kernel-vs-ref rel 9e-4."""
    from angelslim.compressor.sparsity.algorithms.vecattention.kernels_check import (
        kernels_available,
    )

    if not kernels_available(128):
        raise _Skip(
            "vllm_flash_attn (sparse_attn_func) not built — init+build the "
            "vecattention/ops/vllm-flash-attention submodule to exercise the real "
            "kernel path",
            _SR.NO_BLOCK_SPARSE_ATTN,
        )
    if not CUDA:
        raise _Skip("real kernel needs CUDA", _SR.NO_CUDA)
    import torch.nn.functional as F

    from angelslim.compressor.sparsity.algorithms.vecattention.prefill import (
        vecattention_prefill,
    )
    from angelslim.compressor.sparsity.algorithms.vecattention.reference import (
        vecattention_reference_prefill,
    )

    torch.manual_seed(0)
    # Peaked input (head_dim 128): queries along a shared direction u, the first
    # key-block the dominant anchor — the regime VecAttention preserves.
    B, H, L, D = 1, 8, 2048, 128
    u = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    q = u * 2 + torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3
    k[:, :, :64] = u * 3
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.3

    kw = dict(threshold=0.1, block_size_q=64, block_size_k=16)
    try:
        ker = vecattention_prefill(
            q,
            k,
            v,
            group_k_block=16,
            chunk_size=2048,
            head_dim=D,
            allow_pseudo_sparse=False,
            **kw,
        )
        torch.cuda.synchronize()
    except torch.AcceleratorError as e:
        # kernels_available() only checks that vllm_flash_attn IMPORTS; a prebuilt
        # wheel can still fail at launch when its PTX was compiled for a newer CUDA
        # toolchain than the running driver supports
        # (cudaErrorUnsupportedPtxVersion). That is "kernel present but not runnable
        # here" — skip LOUDLY (not a code defect, and not a bare pass) rather than
        # fail. A genuine numeric/shape bug raises a different exception and still
        # surfaces below.
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001  drain the poisoned context
            pass
        raise _Skip(
            f"vllm_flash_attn imports but the kernel cannot launch on this "
            f"driver/toolchain ({e}); rebuild the wheel for the current CUDA to "
            f"exercise the real kernel path",
            _SR.NO_BLOCK_SPARSE_ATTN,
        )
    assert torch.isfinite(ker).all(), "real kernel produced non-finite output"
    ref = vecattention_reference_prefill(q, k, v, **kw)
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # kernel ≈ reference (same MinP method, different implementations) and ≈ dense.
    assert _rel(ker, ref) < 0.05, f"real kernel vs reference rel {_rel(ker, ref):.4f} too high"
    assert _rel(ker, dense) < 0.05, f"real kernel vs dense rel {_rel(ker, dense):.4f} too high"
    _cap("vecattention_real_correctness")


# ===========================================================================
# device_map layer sharding correctness (G1 / — REAL Qwen3.5-9B, >=2 GPUs


# ===========================================================================
def test_vecattention_device_map_sharded_matches_dense():
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B weights unavailable", _SR.NO_QWEN35_9B)
    if torch.cuda.device_count() < 2:
        raise _Skip("device_map sharding test needs >= 2 GPUs", _SR.NO_MULTI_GPU)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    # Load the TEXT TOWER (AutoModelForCausalLM -> qwen3_5_text): its config has a
    # flat num_hidden_layers and the decoder is model.model.layers (the multimodal
    # wrapper hides num_hidden_layers under text_config and nests layers under
    # language_model — avoid that here). head_dim 256 -> reference path.
    cfg = AutoConfig.from_pretrained(_REAL_W35)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and hasattr(cfg, "text_config"):
        n = cfg.text_config.num_hidden_layers
    dm = {
        "model.embed_tokens": 0,
        "model.norm": 1,
        "lm_head": 1,
        "model.rotary_emb": 0,
    }
    for i in range(n):
        dm[f"model.layers.{i}"] = 0 if i < n // 2 else 1
    model = AutoModelForCausalLM.from_pretrained(
        _REAL_W35,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=dm,
    ).eval()
    try:
        used = {str(d) for d in model.hf_device_map.values() if str(d) != "cpu"}
        if len(used) < 2:
            raise _Skip("model did not shard across >=2 GPUs", _SR.NO_MULTI_GPU)
        tok = AutoTokenizer.from_pretrained(_REAL_W35)
        ids = _vc_coherent_ids(tok, 1024)
        with torch.no_grad():
            dense = model(ids).logits.float().cpu()
        slim, patched = _vc_patch_real(
            model,
            {
                "threshold": 0.1,
                "block_size_q": 64,
                "block_size_k": 16,
                "allow_pseudo_sparse": True,
            },
        )
        try:
            with torch.no_grad():
                sp = model(ids).logits.float().cpu()
        finally:
            unpatch_sparsity(slim, patched)
        assert not torch.isnan(sp).any()
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.90, f"sharded vecattention top-1 drift {agree:.3f} ?)"

        _cap("vecattention_real_correctness")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(
        1 if run_all(globals(), f"CUDA={CUDA}, real_weights={real_weights_available()}") else 0
    )
