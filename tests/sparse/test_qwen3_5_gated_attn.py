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

"""Qwen3.5 gated-attention forward template tests.

Verifies, on the REAL Qwen3.5-9B checkpoint (gated attention, head_dim 256,
interleaved linear/full layer_types):
  * resolve_sparsable_layers finds ONLY the full_attention
    layers (the gated-delta-net linear layers are skipped).
  * the gated template runs sparse prefill and restores the gate; all
    three minference-family variants preserve semantics vs dense (top-1 ~1.0),
    unpatch is byte-exact, decode generates the right answer.
  * modal namespace: qwen3_5_text -> llm, qwen3_5 -> vlm.

BOTH real load paths are covered (config.json declares model_type "qwen3_5" +
architectures ["Qwen3_5ForConditionalGeneration"]):
  * AutoModelForCausalLM -> Qwen3_5ForCausalLM, the text tower, config
    model_type "qwen3_5_text", decoder under model.layers. (AngelSlim
    base_model.py LLM path.)
  * Qwen3_5ForConditionalGeneration -> the multimodal wrapper, config
    model_type "qwen3_5", decoder under model.language_model.layers. (AngelSlim
    models/vlm/* path; the language-tower-only patcher.)
  * partial rotary — the template's RoPE handles head_dim 256 / rotary_dim 64.
  * head_dim-aware kernel gating — minference's vertical_slash kernel declines
    head_dim 256 and routes to the reference, while a_shape /
    tri_shape use the real Triton streaming kernel.
  * the registry has both Qwen3 and Qwen3.5 templates.

Weights policy: this whole file is about a real architecture surface; it loads
the real checkpoint or skips. Runnable via __main__ (no pytest).
"""

from __future__ import annotations

import os
import sys
import warnings

import torch

# Shared scaffolding (single source of truth). The local ``_Slim`` here had
# drifted — ``pop_attn_forward`` ignored ``expected_label`` and
# ``attn_forward_labels`` returned ``set()``, leaving the LIFO contract
# untested on the Qwen3.5 path. Qwen3.5-specific loaders stay local below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA  # noqa: E402
from _harness import qwen35_available  # noqa: E402
from _harness import run_all  # noqa: E402
from _harness import REAL_W_QWEN3_5_9B as _REAL_W  # noqa: E402
from _harness import FakeSlim as _Slim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import record_capability as _cap  # noqa: E402
from _harness import rel as _rel  # noqa: E402, F401

_MODEL = None
_TOK = None


def _qwen35():
    global _MODEL, _TOK
    if not qwen35_available():
        return None, None
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _MODEL = (
            AutoModelForCausalLM.from_pretrained(
                _REAL_W,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .to("cuda")
            .eval()
        )
        _TOK = AutoTokenizer.from_pretrained(_REAL_W)
    return _MODEL, _TOK


_VLM_MODEL = None


def _qwen35_vlm():
    """Load Qwen3.5-9B the way config.json declares it: the *multimodal*
    ``Qwen3_5ForConditionalGeneration`` (config model_type ``qwen3_5``, decoder
    layers under ``model.language_model.layers``).

    This is a DISTINCT production path from ``AutoModelForCausalLM`` (which gives
    the text tower, model_type ``qwen3_5_text``, layers under ``model.layers``).
    AngelSlim's VLM model classes (``models/vlm/*``) take this path, so both must
    work — config.json on disk says ``qwen3_5`` + ``Qwen3_5ForConditionalGeneration``.
    """
    global _VLM_MODEL, _TOK
    if not qwen35_available():
        return None, None
    if _VLM_MODEL is None:
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        _VLM_MODEL = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                _REAL_W,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .to("cuda")
            .eval()
        )
        _TOK = AutoTokenizer.from_pretrained(_REAL_W)
    return _VLM_MODEL, _TOK


def _coherent_ids(tok, n=1024):
    para = (
        "The history of science is the study of the development of science "
        "and scientific knowledge, including both natural and social "
        "sciences. "
    ) * 60
    return tok(para, return_tensors="pt").to("cuda")["input_ids"][:, :n]


# ===========================================================================
# Template registry (no weights needed)
# ===========================================================================
def test_template_registry_has_qwen3_and_qwen3_5():
    from angelslim.compressor.sparsity.algorithms._forward_templates import (  # noqa: F401
        TEMPLATE_REGISTRY,
        get_forward_template,
    )
    from angelslim.compressor.sparsity.algorithms._forward_templates.qwen3 import (
        build_qwen3_forward,
    )
    from angelslim.compressor.sparsity.algorithms._forward_templates.qwen3_5 import (
        build_qwen3_5_forward,
    )

    assert get_forward_template("qwen3") is build_qwen3_forward
    assert get_forward_template("qwen3_moe") is build_qwen3_forward
    # Both the wrapper and the text-tower model_types map to the gated template.
    for mt in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        assert get_forward_template(mt) is build_qwen3_5_forward, mt


def test_template_unknown_model_type_raises():
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.algorithms._forward_templates import (
        get_forward_template,
    )

    try:
        get_forward_template("llama")
        raise AssertionError("unknown model_type should raise")
    except IncompatibleConfigError:
        pass


def test_modal_namespace_qwen3_5_split():
    from angelslim.compressor.sparsity._modal import resolve_modal

    class _Cfg:
        def __init__(self, mt):
            self.model_type = mt

    class _M:
        def __init__(self, mt):
            self.config = _Cfg(mt)

    # The multimodal wrapper is vlm; the text tower (causal LM config) is llm.
    assert resolve_modal(_M("qwen3_5")) == "vlm"
    assert resolve_modal(_M("qwen3_5_text")) == "llm"
    assert resolve_modal(_M("qwen3_5_moe")) == "vlm"
    assert resolve_modal(_M("qwen3_5_moe_text")) == "llm"


# ===========================================================================
# REAL Qwen3.5-9B
# ===========================================================================
def test_qwen3_5_resolve_only_full_attention_layers():
    """Only the full_attention layers carry self_attn; the
    gated-delta-net linear layers are filtered out."""
    m, _ = _qwen35()
    if m is None:
        raise _Skip("real Qwen3.5-9B weights unavailable")
    from angelslim.compressor.sparsity._layers import (
        resolve_layers,
        resolve_sparsable_layers,
    )

    all_layers = resolve_layers(m)
    sparsable = resolve_sparsable_layers(m)
    n_full = sum(1 for lt in m.config.layer_types if lt == "full_attention")
    assert len(all_layers) == m.config.num_hidden_layers
    assert len(sparsable) == n_full, f"{len(sparsable)} != {n_full} full layers"
    assert all(hasattr(layer, "self_attn") for layer in sparsable)
    assert len(sparsable) < len(all_layers), "linear layers must be filtered out"


def test_qwen3_5_gated_template_preserves_semantics():
    """All 3 variants run through the gated template on REAL Qwen3.5-9B
    and preserve the prediction (top-1 ~1.0); unpatch is byte-exact.

    head_dim 256: a_shape/tri_shape use the real Triton streaming kernel;
    minference's vertical_slash kernel declines head_dim 256 and falls back to
    the pure-torch reference, so it needs allow_pseudo_sparse."""
    m, tok = _qwen35()
    if m is None:
        raise _Skip("real Qwen3.5-9B weights unavailable", _SR.NO_QWEN35_9B)
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    ids = _coherent_ids(tok, 1024)
    with torch.no_grad():
        dense = m(ids).logits.float()
    n_full = sum(1 for lt in m.config.layer_types if lt == "full_attention")

    for variant, kw, pseudo in [
        ("a_shape", {"n_init": 64, "n_local": 512}, False),
        ("tri_shape", {"n_init": 64, "n_local": 512, "n_last": 100}, False),
        ("minference", {}, True),  # head_dim 256 -> reference fallback
    ]:
        slim = _Slim(m)
        algo = SparsityAlgorithmRegistry.create(variant, attn_kwargs=dict(kw))
        algo.allow_pseudo_sparse = pseudo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            patched = apply_sparsity_patch(slim, algo)
        assert len(patched) == n_full, f"{variant}: patched {len(patched)} != {n_full}"
        with torch.no_grad():
            sp = m(ids).logits.float()
        unpatch_sparsity(slim, patched)
        with torch.no_grad():
            restored = m(ids).logits.float()

        assert torch.isfinite(sp).all(), variant
        agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
        assert agree > 0.95, f"{variant} top-1 agreement {agree:.3f}"
        assert torch.equal(restored, dense), f"{variant} unpatch not exact"
    _cap("qwen35_real_correctness")


def test_qwen3_5_decode_generates():
    """Gated sparse prefill + decode generates the correct answer (REAL)."""
    m, tok = _qwen35()
    if m is None:
        raise _Skip("real Qwen3.5-9B weights unavailable")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    slim = _Slim(m)
    algo = SparsityAlgorithmRegistry.create("a_shape", attn_kwargs={"n_init": 64, "n_local": 512})
    algo.allow_pseudo_sparse = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    finally:
        unpatch_sparsity(slim, patched)
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"gated-template decode lost the answer: {text!r}"


def test_qwen3_5_vlm_load_path_language_tower():
    """The OTHER production load path: config.json declares model_type=qwen3_5 +
    Qwen3_5ForConditionalGeneration (multimodal). Decoder layers live under
    model.language_model.layers, and the patcher must reach them (the
    language-tower-only patcher).

    This complements test_qwen3_5_gated_template_preserves_semantics, which uses
    the AutoModelForCausalLM text-tower path (model_type qwen3_5_text). Both are
    real AngelSlim paths (base_model.py LLM vs models/vlm/* VLM)."""
    m, tok = _qwen35_vlm()
    if m is None:
        raise _Skip("real Qwen3.5-9B weights unavailable")
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity._layers import (
        resolve_layers,
        resolve_sparsable_layers,
    )
    from angelslim.compressor.sparsity._modal import resolve_modal
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    # The multimodal wrapper reports model_type qwen3_5 (-> vlm) and nests the
    # decoder under language_model.
    assert m.config.model_type == "qwen3_5"
    assert resolve_modal(m) == "vlm"
    inner = m.model
    assert hasattr(inner, "language_model") and not hasattr(inner, "layers")

    n_full = sum(1 for lt in m.config.text_config.layer_types if lt == "full_attention")
    assert len(resolve_layers(m)) == m.config.text_config.num_hidden_layers
    assert len(resolve_sparsable_layers(m)) == n_full

    ids = _coherent_ids(tok, 1024)
    with torch.no_grad():
        dense = m(ids).logits.float()
    slim = _Slim(m)
    algo = SparsityAlgorithmRegistry.create("a_shape", attn_kwargs={"n_init": 64, "n_local": 512})
    algo.allow_pseudo_sparse = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    assert len(patched) == n_full
    with torch.no_grad():
        sp = m(ids).logits.float()
    unpatch_sparsity(slim, patched)
    with torch.no_grad():
        restored = m(ids).logits.float()
    agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
    assert agree > 0.95, f"VLM-path a_shape top-1 agreement {agree:.3f}"
    assert torch.equal(restored, dense), "VLM-path unpatch not exact"


def test_qwen3_5_head_dim_kernel_gating():
    """head_dim 256: a_shape/tri_shape real kernel available; minference's
    vertical_slash kernel is NOT (supports {16,32,64,128})."""
    from angelslim.compressor.sparsity.algorithms.minference import kernels as K

    if not CUDA:
        raise _Skip("CUDA unavailable")
    assert K.kernels_available("a_shape", head_dim=256)
    assert K.kernels_available("tri_shape", head_dim=256)
    assert not K.kernels_available("minference", head_dim=256)
    # head_dim 257..512 pads to the unsupported 512 — the gate must report False
    # so it falls back to the reference instead of asserting inside the kernel.
    for hd in (257, 320, 512):
        assert not K.kernels_available("a_shape", head_dim=hd), hd
        assert not K.kernels_available("tri_shape", head_dim=hd), hd
    # but supported head dims are fine for minference (when the ext builds)
    from angelslim.compressor.sparsity.algorithms.minference.kernels._cuda_ext import (
        cuda_ext_buildable,
    )

    assert K.kernels_available("minference", head_dim=128) == cuda_ext_buildable()


# ===========================================================================
# Runner
# ===========================================================================
def _run_all():
    return run_all(globals(), f"CUDA={CUDA}, qwen3_5={qwen35_available()}")


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
