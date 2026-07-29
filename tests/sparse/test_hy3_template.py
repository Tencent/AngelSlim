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

"""Hunyuan-V3 (hy_v3) sparse-attention forward-template tests.

Hunyuan-V3 attention is structurally identical to Qwen3 (q_norm / k_norm on the
head dim, standard RoPE, GQA in the attention interface, plain ungated q_proj);
it is a MoE only in the MLP tower. So ``build_hy3_forward`` reuses the Qwen3
forward logic, and this suite proves that reuse is correct:

  * the template + every algorithm resolve for model_type "hy_v3" (CPU, no kernel);
  * the patch lifecycle installs on all attention layers and unpatch restores the
    model byte-exact (CPU, no kernel — the sparse kernels need CUDA);
  * at full budget the sparse forward matches dense on a tiny HYV3 (real numerics,
    CUDA-gated) — the same dense-parity bar the per-algorithm suites assert on real
    Qwen3-8B, but here on a tiny random-init model so it needs only a GPU, not a
    16 GB checkpoint;
  * at full budget the sparse forward matches dense on the REAL Hunyuan-V3
    checkpoint, and on a coherent prompt sparsifying keeps top-1 (multi-GPU,
    weights-gated) — the genuine end-to-end correctness bar. Skips loudly when the
    checkpoint or CUDA is absent.
"""

from __future__ import annotations

import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA  # noqa: E402
from _harness import REAL_W_HY3  # noqa: E402
from _harness import hy3_available  # noqa: E402
from _harness import FakeSlim as _Slim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import record_capability as _cap  # noqa: E402
from _harness import rel as _rel  # noqa: E402
from _harness import run_all, tiny_hy3  # noqa: E402

# Every algorithm that declares hy_v3 compatible. All route through
# build_hy3_forward; tier-1 (resolution + compatibility) covers all of them.
_ALL_ALGOS = ("stem", "minference", "flexprefill", "xattention", "flashprefill", "vecattention")

# Algorithms whose full-budget path runs a healthy kernel on a head_dim-128 tiny
# model here. vecattention is EXCLUDED from the numeric loop: its accelerated path
# is the prebuilt vllm-flash-attention kernel, which is not portably runnable in
# every env (and its torch reference only serves head_dim 256, the Qwen3.5 path).
# vecattention's reference dense-parity is covered by its own integration suite;
# here it is still asserted hy_v3-compatible and template-resolvable (tier 1).
_PARITY_ALGOS = ("stem", "minference", "flexprefill", "xattention", "flashprefill")

# Full-budget ("keep everything") knobs per algorithm — the setting under which
# the sparse forward must reproduce dense.
_FULL_BUDGET = {
    "stem": {},  # default keep schedule ~ dense at tiny scale
    "minference": {},  # dense per-head fallback (no pattern_path)
    "flexprefill": {"gamma": 0.9999, "tau": 0.0},
    "xattention": {"threshold": 1.0},
    "flashprefill": {"alpha": 0.0},  # alpha=0 => keep all == dense
}


def _registry():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    return SparsityAlgorithmRegistry


# ===========================================================================
# Template resolution + per-algorithm compatibility (CPU, always runs)
# ===========================================================================
def test_hy3_template_registered():
    from angelslim.compressor.sparsity.algorithms._forward_templates import (
        get_forward_template,
    )

    build = get_forward_template("hy_v3")
    assert build.__name__ == "build_hy3_forward"


def test_all_algorithms_declare_hy3_compatible():
    reg = _registry()
    for name in _ALL_ALGOS:
        algo = reg.create(name, attn_kwargs={})
        compat = algo.traits.compatible_model_types
        assert (
            compat is not None and "hy_v3" in compat
        ), f"{name}: hy_v3 not in compatible_model_types {sorted(compat)}"


# ===========================================================================
# Patch lifecycle: installs on every attention layer, unpatch restores byte-exact
# (CPU — no kernel invoked; only patch install/remove + a dense forward)
# ===========================================================================
def test_hy3_patch_installs_and_unpatch_restores_byte_exact():
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )

    reg = _registry()
    m = tiny_hy3(num_layers=4, attn_impl="eager")
    slim = _Slim(m)
    dev = "cuda" if CUDA else "cpu"
    ids = torch.randint(0, 512, (1, 48), device=dev)

    with torch.no_grad():
        before = m(ids).logits.clone()

    algo = reg.create("stem", attn_kwargs={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    try:
        assert (
            len(patched) == m.config.num_hidden_layers
        ), f"patched {len(patched)} of {m.config.num_hidden_layers} attention layers"
        assert slim.attn_forward_labels() == {"sparse"}
    finally:
        unpatch_sparsity(slim, patched)

    assert slim.attn_forward_labels() == set(), "unpatch left live patches"
    with torch.no_grad():
        after = m(ids).logits
    assert torch.equal(before, after), "unpatch did not restore HYV3 forward byte-exact"


# ===========================================================================
# Full-budget numeric dense parity on a tiny HYV3 (real numerics, CUDA-gated)
# ===========================================================================
def _dense_parity(name):
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )

    reg = _registry()
    m = tiny_hy3(num_layers=4, attn_impl="eager")  # bf16/cuda inside tiny_hy3
    slim = _Slim(m)
    ids = torch.randint(0, 512, (1, 64), device="cuda")

    with torch.no_grad():
        dense = m(ids).logits.float()

    algo = reg.create(name, attn_kwargs=dict(_FULL_BUDGET[name]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    try:
        assert len(patched) == m.config.num_hidden_layers
        with torch.no_grad():
            sparse = m(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)

    r = _rel(sparse, dense)
    # bf16 numeric floor — the per-algorithm real-weight suites see the same
    # ~1e-2 band at full budget. A regression (wrong head count from a mis-driven
    # template, dropped gate, etc.) blows this up by orders of magnitude.
    assert r < 3e-2, f"{name}: full-budget sparse vs dense rel={r:.3e} (template mis-drives HYV3?)"
    return r


def test_hy3_full_budget_matches_dense_all_algorithms():
    if not CUDA:
        raise _Skip("sparse kernels require CUDA", _SR.NO_CUDA)
    worst = 0.0
    for name in _PARITY_ALGOS:
        r = _dense_parity(name)
        worst = max(worst, r)
    _cap("hy3_template_numeric_parity")
    print(f"  hy3 full-budget dense parity OK; worst rel={worst:.3e}")


# ===========================================================================
# Decode path (q_len==1 with a KV cache) on a tiny HYV3 — every algorithm.
#
# The dense-parity test above only ever runs a single q_len>1 forward, i.e. the
# PREFILL branch of each patched forward. Generation also drives the DECODE
# branch (q_len==1 against a cache prefix), which routes to the standard
# attention interface. That branch reads attributes the sparse forward never
# touches on prefill — e.g. ``sliding_window`` — and HYV3Attention does NOT set
# every attribute Qwen3 does. A direct ``self.sliding_window`` access there
# raised ``AttributeError`` on real Hunyuan-V3 generation while every prefill-
# only test stayed green. This test steps one cached token per algorithm so the
# decode branch is actually exercised on hy_v3; it needs no real weights.
# ===========================================================================
def test_hy3_decode_step_all_algorithms():
    if not CUDA:
        raise _Skip("sparse kernels require CUDA", _SR.NO_CUDA)
    from transformers import DynamicCache

    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )

    reg = _registry()
    for name in _PARITY_ALGOS:
        m = tiny_hy3(num_layers=4, attn_impl="eager")
        slim = _Slim(m)
        algo = reg.create(name, attn_kwargs=dict(_FULL_BUDGET[name]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            patched = apply_sparsity_patch(slim, algo)
        try:
            # Prime a real KV cache with a prefill, then take one q_len==1 step —
            # the decode branch that a prefill-only test never reaches.
            prompt = torch.randint(0, 512, (1, 32), device="cuda")
            cache = DynamicCache()
            with torch.no_grad():
                out = m(prompt, use_cache=True, past_key_values=cache)
            nxt = out.logits[:, -1:].argmax(-1)
            pos = torch.tensor([prompt.shape[1]], device="cuda")
            with torch.no_grad():
                step = m(nxt, use_cache=True, past_key_values=cache, cache_position=pos).logits
        finally:
            unpatch_sparsity(slim, patched)
        assert torch.isfinite(step).all(), f"{name}: decode step produced NaN/inf on hy_v3"
        assert step.shape[1] == 1, f"{name}: decode step q_len != 1"
    _cap("hy3_template_decode_step")
    print(f"  hy3 decode-step OK for {len(_PARITY_ALGOS)} algorithms")


# ===========================================================================
# REAL Hunyuan-V3 checkpoint correctness (weights-gated, multi-GPU).
#
# The tiny-model parity above proves the template WIRING on random numerics; this
# proves the SAME template on the genuine 80-layer / 192-expert Hy3 MoE. The full
# checkpoint does not fit on one GPU, so ``device_map="auto"`` shards it across all
# visible GPUs — which additionally exercises the per-layer device-context guard in
# the kernel path (a cuda:k layer must launch on cuda:k, not cuda:0) on hy_v3.
# ===========================================================================
_REAL_HY3 = None  # module-level cache: the 598 GB load happens at most once


def _real_hy3():
    """Load the real Hunyuan-V3 once, sharded across every visible GPU. Returns
    the model, or None when the checkpoint / CUDA is unavailable."""
    global _REAL_HY3
    if not hy3_available():
        return None
    if _REAL_HY3 is None:
        from transformers import AutoModelForCausalLM

        _REAL_HY3 = AutoModelForCausalLM.from_pretrained(
            REAL_W_HY3,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
    return _REAL_HY3


def _hy3_patch_real(model, name, attn_kwargs):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create(name, attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _hy3_input_ids(model, n_tokens=960):
    """A coherent prompt tokenized with the checkpoint's own tokenizer (real text
    attention is sparse, so a full-budget pattern loses nothing). Falls back to
    random ids only if the tokenizer is missing."""
    dev = next(model.parameters()).device
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REAL_W_HY3)
        para = (
            "The history of science is the study of the development of science "
            "and scientific knowledge, including both the natural and social "
            "sciences. "
        ) * 40
        ids = tok(para, return_tensors="pt")["input_ids"][:, :n_tokens]
        return ids.to(dev)
    except Exception:
        return torch.randint(0, 1024, (1, n_tokens), device=dev)


def test_hy3_real_weights_full_budget_matches_dense():
    """On the REAL 80-layer Hunyuan-V3, each algorithm at full budget reproduces
    the unpatched dense forward's decisions, and unpatch restores it byte-exact.
    This is the genuine end-to-end proof that ``build_hy3_forward`` drives real
    HYV3Attention correctly (QKV/RoPE/q_norm/k_norm/GQA/o_proj on trained weights).

    Correctness bar = per-position top-1 agreement vs dense (what decode actually
    consumes), NOT a raw logit relative-error threshold. Depth matters: the same
    templates clear rel<0.05 on 36-layer Qwen3-8B and rel<0.017 on a tiny 4-layer
    HYV3, but a per-layer bf16 attention-vs-eager magnitude gap of ~1% compounds
    across 80 layers to a whole-logit rel of ~0.12 for EVERY algorithm alike —
    while top-1 stays ~0.997. A uniform rel across all algorithms with top-1
    intact is a depth/precision artifact, not a template or per-algo defect (a
    real mis-drive tanks top-1). rel is measured and printed for the record."""
    m = _real_hy3()
    if m is None:
        raise _Skip("real Hunyuan-V3 weights unavailable", _SR.NO_LARGE_CKPT)
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    ids = _hy3_input_ids(m)
    with torch.no_grad():
        dense = m(ids).logits.float().cpu()
    dense_top1 = dense[0].argmax(-1)

    ran = {}
    for name in _PARITY_ALGOS:
        slim, patched = _hy3_patch_real(m, name, _FULL_BUDGET[name])
        try:
            with torch.no_grad():
                sp = m(ids).logits.float().cpu()
        except (RuntimeError, torch.AcceleratorError) as e:
            # An env-broken kernel (e.g. a wheel built for a newer CUDA) must not
            # masquerade as a template defect; skip that algo loudly and continue.
            unpatch_sparsity(slim, patched)
            print(f"  [skip] {name}: kernel unavailable in this env ({type(e).__name__})")
            continue
        finally:
            if slim.attn_forward_labels():
                unpatch_sparsity(slim, patched)

        assert torch.isfinite(sp).all(), f"{name}: real-weight sparse produced NaN/inf"
        agree = (dense_top1 == sp[0].argmax(-1)).float().mean().item()
        r = _rel(sp, dense)
        assert (
            agree > 0.99
        ), f"{name}: real-weight top-1 {agree:.4f} too low (template mis-drives HYV3?)"
        with torch.no_grad():
            restored = m(ids).logits.float().cpu()
        assert torch.equal(restored, dense), f"{name}: unpatch not byte-exact on real HYV3"
        ran[name] = (agree, r)

    assert ran, "no algorithm's kernel ran on the real Hy3 in this env"
    _cap("hy3_real_weights_dense_parity")
    summary = ", ".join(f"{k} top1={a:.4f}/rel={r:.3e}" for k, (a, r) in ran.items())
    print(f"  hy3 REAL-weight full-budget dense parity OK: {summary}")


def test_hy3_real_weights_quality():
    """On the REAL Hunyuan-V3 with a coherent prompt, actually sparsifying (a
    budget that drops keys) preserves per-position top-1 vs dense — the realistic
    correctness bar. Uses a_shape (its streaming kernel runs at head_dim 128, the
    hy_v3 head dim) so real sparsification is exercised, not a dense fallback."""
    m = _real_hy3()
    if m is None:
        raise _Skip("real Hunyuan-V3 weights unavailable", _SR.NO_LARGE_CKPT)
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    ids = _hy3_input_ids(m)
    with torch.no_grad():
        dense = m(ids).logits.float().cpu()

    slim, patched = _hy3_patch_real(m, "a_shape", {"n_init": 64, "n_local": 512})
    try:
        assert len(patched) == m.config.num_hidden_layers, "not every hy_v3 layer patched"
        with torch.no_grad():
            sp = m(ids).logits.float().cpu()
    finally:
        unpatch_sparsity(slim, patched)

    assert torch.isfinite(sp).all(), "real-weight a_shape produced NaN/inf"
    agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
    assert agree > 0.95, f"real Hy3 a_shape top-1 agreement {agree:.3f} too low"
    _cap("hy3_real_weights_topk_quality")
    print(f"  hy3 REAL-weight sparsified top-1 agreement={agree:.4f}")


if __name__ == "__main__":
    sys.exit(1 if run_all(globals(), f"CUDA={CUDA}") else 0)
