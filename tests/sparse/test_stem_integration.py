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

"""Comprehensive Stem integration tests (Stem-first slice).

Covers the core items Stem exercises, end to end through the framework:
  * sparsity subsystem imports without VecAttention kernel.
  * resolve_layers / resolve_sparsable_layers (incl. interleaved &
              VLM-tower nesting, and the error paths).
  * LIFO push/pop on BaseLLMModel (incl. cross-module interleave).
  * idempotent convert + unpatch round-trip.
  * FP8-attn collision guard (label + legacy-dict signals).
  * multi-node hard-fail (WORLD_SIZE / PMI_SIZE).
  * TP detector (device_map / vLLM-in-process).
  * CUDA-graph / compile runtime guard.
  * modal resolution + model_type guard (allow / block lists).
  * per-instance Stem layer-keep schedule (no hardcoded 38; OOB
              layer_idx no longer IndexErrors; short-seq vs long-seq regimes).
  * kernel hard-fail vs allow_pseudo_sparse opt-in.
  * KV-compression names blocked at the registry.
  * save/load round-trip via Engine + YAML *and* JSON.
  * Stem forward correctness vs dense (keep=1.0 ~= dense; sparsity genuinely
    engages at long sequence; unpatch is exact; pseudo-sparse ~= kernel).

Tests are pytest-style but also runnable via the bottom __main__ block with no
pytest installed (the angelslim env has no pytest). CPU-only logic tests run
everywhere; kernel/forward tests early-return when CUDA is unavailable.

Weights policy: any test whose assertion is
about model numerical / semantic correctness (sparse output ~= dense, argmax
agreement, quality preservation) loads the **real Qwen3-8B** checkpoint at
``_REAL_W``. Framework-mechanics tests (patch counting, LIFO, registry, guards,
config parsing, save-emits-files, kernel-gate policy, schedule math) use a tiny
synthetic model — they assert structure, never weight-dependent numerics, so a
random-init model is correct and fast there.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import warnings

import torch

# The LIFO double / real-model singleton / coverage-floored runner that
# this suite originated are now the SINGLE source of truth in _harness.py, so
# the other suites cannot ship a drifted copy. This file imports them back;
# the stem-specific ``_patched_real`` (which builds the stem algo) stays local
# but uses the shared ``FakeSlim``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import real_weights_available  # noqa: E402
from _harness import run_all  # noqa: E402
from _harness import CUDA, HEAD_DIM  # noqa: E402
from _harness import REAL_W_QWEN3_8B as _REAL_W  # noqa: E402
from _harness import SCRATCH_ROOT as _SCRATCH  # noqa: E402
from _harness import FakeSlim as _FakeSlim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import real_qwen3_8b as _real_qwen3_8b  # noqa: E402
from _harness import record_capability as _cap  # noqa: E402

# Stem's real-model LIFO double is identical to the shared one.
_RealSlim = _FakeSlim


@contextlib.contextmanager
def _patched_real(model, attn_kwargs):
    """Patch the real model with Stem, yield (slim, algo, patched), always unpatch.

    Wrapping in try/finally guarantees the *shared* singleton model is restored
    to its pristine forwards even if an assertion fails mid-test.
    """
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register stem
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _RealSlim(model)
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs=attn_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    try:
        yield slim, algo, patched
    finally:
        unpatch_sparsity(slim, patched)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _tiny_qwen3(num_layers=4, attn_impl="eager"):
    # The builder now lives in _harness (shared with the distributed /
    # runtime-guard suites). Kept as a thin local alias so the rest of this
    # suite is unchanged.
    from _harness import tiny_qwen3

    return tiny_qwen3(num_layers=num_layers, attn_impl=attn_impl)


def _tiny_qwen3_moe(num_layers=2):
    """A tiny Qwen3-MoE — the second model_type Stem declares compatible.

    NOTE: no real Qwen3-MoE checkpoint is available locally, so the MoE forward
    test asserts only the keep=1.0-equals-dense *algorithm property* (which is
    weight-independent) plus structural patching — not weight-dependent quality.
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=2048,
        attn_implementation="eager",
    )
    m = Qwen3MoeForCausalLM(cfg)
    if CUDA:
        m = m.to("cuda", dtype=torch.bfloat16)
    return m.eval()


# ===========================================================================
# Subsystem import does not depend on VecAttention kernel
# ===========================================================================
def test_sparsity_subsystem_imports_without_vecattention():
    import angelslim.compressor.sparsity as s

    # The public surface is the framework now (the legacy StemInference /
    # VecAttentionInference entry points were removed once every algorithm
    # became a registered SparsityAlgorithm).
    assert "Sparsity" in s.__dir__()
    # Registering Stem must work even though VecAttention's kernel may be
    # unbuilt — the lazy import policy keeps the registry usable.
    import angelslim.compressor.sparsity.algorithms.stem  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    assert "stem" in SparsityAlgorithmRegistry.available()


def test_lazy_getattr_unknown_name_raises():
    """The PEP-562 __getattr__ must reject unknown attributes (not import junk)."""
    import angelslim.compressor.sparsity as s

    try:
        s.NoSuchThing
        raise AssertionError("expected AttributeError for unknown lazy attr")
    except AttributeError:
        pass


# ===========================================================================
# Layer resolution (plain / interleaved / VLM tower / error paths)
# ===========================================================================
def test_resolve_sparsable_layers_qwen3():
    from angelslim.compressor.sparsity._layers import (
        resolve_layers,
        resolve_sparsable_layers,
    )

    m = _tiny_qwen3(num_layers=4)
    assert len(resolve_layers(m)) == 4
    # Plain Qwen3: every layer has self_attn.
    assert len(resolve_sparsable_layers(m)) == 4


def test_resolve_sparsable_layers_filters_linear_attention():
    """Interleaved linear/full attention (Qwen3.5 shape) — only full layers patch."""
    import torch.nn as nn

    from angelslim.compressor.sparsity._layers import (
        resolve_layers,
        resolve_sparsable_layers,
    )

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = nn.Linear(2, 2)  # gated delta-net, NOT self_attn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Full(), Linear(), Full(), Linear()])

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    o = Outer()
    assert len(resolve_layers(o)) == 4
    sparsable = resolve_sparsable_layers(o)
    assert len(sparsable) == 2  # only the two full-attention layers
    assert all(hasattr(layer, "self_attn") for layer in sparsable)


def test_resolve_layers_prefers_language_model_tower():
    """VLM nesting: model.model.language_model.layers wins over model.model.layers."""
    import torch.nn as nn

    from angelslim.compressor.sparsity._layers import resolve_layers

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(2, 2)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([Layer(), Layer()])
            # An outer `.layers` that must be *ignored* in favour of the tower.
            self.layers = nn.ModuleList([Layer()])

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    assert len(resolve_layers(Outer())) == 2  # the tower, not the 1-elem outer


def test_resolve_layers_error_paths():
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity._layers import resolve_layers

    class NoModel:
        pass

    # resolve_layers raises the subsystem's IncompatibleConfigError (was a
    # bare AttributeError) so callers can catch one consistent type.
    try:
        resolve_layers(NoModel())
        raise AssertionError("expected IncompatibleConfigError when .model is missing")
    except IncompatibleConfigError:
        pass

    class InnerNoLayers:
        pass

    class WrapNoLayers:
        def __init__(self):
            self.model = InnerNoLayers()

    try:
        resolve_layers(WrapNoLayers())
        raise AssertionError("expected IncompatibleConfigError when .layers is missing")
    except IncompatibleConfigError:
        pass


# ===========================================================================
# Per-module single-slot push/pop on the real BaseLLMModel
# (simplified from a LIFO stack: sparse never stacks, so a module
#  holds at most one patch; a second push to the same module hard-fails)
# ===========================================================================
def test_push_pop_on_base_model():
    from angelslim.models.base_model import BaseLLMModel  # noqa: F401

    slim = _make_base_llm_model()

    class _Attn:
        def forward(self):
            return "orig"

    attn = _Attn()
    slim.push_attn_forward("sparse", attn, lambda: "S")
    assert attn.forward() == "S"
    assert slim.attn_forward_labels() == {"sparse"}
    slim.pop_attn_forward(attn, expected_label="sparse")
    # Bound methods return a fresh object each access, so compare behavior,
    # not identity: after popping the patch the original forward runs again.
    assert attn.forward() == "orig"


def test_push_same_module_twice_raises():
    """Per-module single slot: a second push onto the same module is a
    hard error — sparse does not stack on any existing attention patch."""
    slim = _make_base_llm_model()

    class _Attn:
        def forward(self):
            return "orig"

    attn = _Attn()
    slim.push_attn_forward("a", attn, lambda: "A")
    try:
        slim.push_attn_forward("b", attn, lambda: "B")
        raise AssertionError("expected RuntimeError on double-push to same module")
    except RuntimeError:
        pass
    # First patch intact; refused push changed nothing.
    assert attn.forward() == "A"
    assert slim.attn_forward_labels() == {"a"}
    slim.pop_attn_forward(attn, expected_label="a")
    assert attn.forward() == "orig"


def test_pop_label_mismatch_raises():
    """Popping with the wrong expected_label must hard-fail (patch ownership)."""
    slim = _make_base_llm_model()

    class _Attn:
        def forward(self):
            return "orig"

    attn = _Attn()
    slim.push_attn_forward("sparse", attn, lambda: "S")
    try:
        slim.pop_attn_forward(attn, expected_label="qwen_fp8")
        raise AssertionError("expected RuntimeError on label mismatch")
    except RuntimeError:
        pass
    # The patch is untouched after a refused pop.
    assert attn.forward() == "S"
    slim.pop_attn_forward(attn, expected_label="sparse")
    assert attn.forward() == "orig"


def test_two_modules_independent_slots():
    """Two attention modules patched/unpatched independently restore correctly
    (each module has its own single slot)."""
    slim = _make_base_llm_model()

    class _Attn:
        def __init__(self, tag):
            self._tag = tag

        def forward(self):
            return self._tag

    a, b = _Attn("a0"), _Attn("b0")
    slim.push_attn_forward("sparse", a, lambda: "a1")
    slim.push_attn_forward("sparse", b, lambda: "b1")
    assert (a.forward(), b.forward()) == ("a1", "b1")
    # Pop a first, b must remain patched.
    slim.pop_attn_forward(a, expected_label="sparse")
    assert (a.forward(), b.forward()) == ("a0", "b1")
    slim.pop_attn_forward(b, expected_label="sparse")
    assert (a.forward(), b.forward()) == ("a0", "b0")


def _make_base_llm_model():
    from angelslim.models.base_model import BaseLLMModel

    class _M(BaseLLMModel):
        def get_observer_layers(self):
            return {}

        def get_save_func(self):
            return None

    slim = _M.__new__(_M)
    BaseLLMModel.__init__(slim)
    return slim


# ===========================================================================
# Per-instance Stem schedule (no hardcoded 38; OOB safe; regimes)
# ===========================================================================
def test_stem_interpolate_default_schedule():
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.algorithms.stem.algorithm import (
        interpolate_default_schedule,
    )

    for n in (28, 36, 40, 48, 64, 80):
        sched = interpolate_default_schedule(n)
        assert len(sched) == n
        assert sched[0] == 1.0  # warmup
        assert sched[-1] == 0.2  # steady

    # Edge counts smaller than the 2-layer warmup region.
    assert interpolate_default_schedule(1) == [1.0]
    assert interpolate_default_schedule(2) == [1.0, 1.0]
    assert interpolate_default_schedule(3) == [1.0, 1.0, 0.2]

    for bad in (0, -5):
        try:
            interpolate_default_schedule(bad)
            raise AssertionError(f"expected IncompatibleConfigError for n={bad}")
        except IncompatibleConfigError:
            pass


def test_stem_setup_mismatched_ratios_rejected():
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"layer_keep_ratios": [1.0, 0.2]})
    m = _tiny_qwen3(num_layers=4)
    try:
        algo.setup(m)
        raise AssertionError("expected IncompatibleConfigError for wrong-length ratios")
    except IncompatibleConfigError:
        pass


def test_stem_setup_derives_per_instance_schedule():
    """setup() with no explicit ratios derives a schedule matching layer count."""
    import warnings

    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    for n in (4, 6):
        algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
        m = _tiny_qwen3(num_layers=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            algo.setup(m)
        ratios = algo._attn_forward_config["layer_keep_ratios"]
        assert len(ratios) == n
        assert ratios[0] == 1.0 and ratios[-1] == 0.2


def test_generate_schedule_oob_layer_idx_no_indexerror():
    """Core regression: layer_idx >> hardcoded-38 table must not IndexError.

    The old hardcoded _DEFAULT_LAYER_KEEP_RATIOS had 38 entries; a 40+-layer
    model indexing layer 200 used to IndexError. It must now fall back to the
    steady ratio (== explicit keep_ratio=0.2).
    """
    from angelslim.compressor.sparsity.algorithms.stem.backends.torch_impl import (
        _DEFAULT_LAYER_KEEP_RATIOS,
        generate_exact_k_schedule,
    )

    dev = torch.device("cuda" if CUDA else "cpu")
    assert len(_DEFAULT_LAYER_KEEP_RATIOS) == 38  # the legacy table, kept as fallback
    s_oob = generate_exact_k_schedule(10, 1.0, 200, dev)  # idx 200 >> 38
    s_steady = generate_exact_k_schedule(10, 1.0, 0, dev, keep_ratio=0.2)
    assert torch.equal(s_oob, s_steady), "OOB layer_idx must reuse the steady ratio"


def test_generate_schedule_keep_ratio_regimes():
    """keep_ratio threads through; short-seq relaxation vs long-seq sparsity."""
    from angelslim.compressor.sparsity.algorithms.stem.backends.torch_impl import (
        generate_exact_k_schedule,
    )

    dev = torch.device("cuda" if CUDA else "cpu")

    # keep=1.0 keeps every block, regardless of length.
    assert generate_exact_k_schedule(8, 1.0, 0, dev, keep_ratio=1.0)[0].item() == 8
    assert generate_exact_k_schedule(64, 1.0, 0, dev, keep_ratio=1.0)[0].item() == 64

    # Short sequence (<56 blocks): relaxation keeps ALL blocks even at keep=0.2.
    assert generate_exact_k_schedule(40, 1.0, 0, dev, keep_ratio=0.2)[0].item() == 40

    # Long sequence (>=56 blocks): keep=0.2 genuinely drops blocks (< Qb).
    long_first = generate_exact_k_schedule(64, 1.0, 0, dev, keep_ratio=0.2)[0].item()
    assert long_first < 64, f"long-seq keep=0.2 should drop blocks, got {long_first}"

    # per-head shape vs single-head shape.
    ph = generate_exact_k_schedule(8, 1.0, 0, dev, num_heads=4, keep_ratio=0.2)
    assert tuple(ph.shape) == (4, 8) and bool((ph[0] == ph[1]).all())
    sh = generate_exact_k_schedule(8, 1.0, 0, dev, keep_ratio=0.2)
    assert tuple(sh.shape) == (8,)


def test_generate_schedule_alpha_decay_monotone():
    """alpha<1 produces a non-increasing budget that stays >=1."""
    from angelslim.compressor.sparsity.algorithms.stem.backends.torch_impl import (
        generate_exact_k_schedule,
    )

    dev = torch.device("cuda" if CUDA else "cpu")
    s = generate_exact_k_schedule(200, 0.5, 0, dev, keep_ratio=0.2)
    assert bool((s[1:] <= s[:-1]).all()), "decayed schedule must be non-increasing"
    assert bool((s >= 1).all()), "budget must stay >= 1"


# ===========================================================================
# Modal resolution + model_type guard
# ===========================================================================
def test_modal_resolution_table():
    from angelslim.compressor.sparsity._modal import resolve_modal

    class _Cfg:
        def __init__(self, mt):
            self.model_type = mt

    class _M:
        def __init__(self, mt):
            self.config = _Cfg(mt)

    cases = {
        "qwen3": "llm",
        "qwen3_moe": "llm",
        "qwen3_5": "vlm",
        "qwen3_5_moe": "vlm",
        "qwen3_omni_moe": "omni",
        "totally_unknown": "llm",  # default
    }
    for mt, expected in cases.items():
        assert resolve_modal(_M(mt)) == expected, mt


def test_model_type_guard_allow_and_block():
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import _check_model_type
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    class _Cfg:
        def __init__(self, mt):
            self.model_type = mt

    class _HF:
        def __init__(self, mt):
            self.config = _Cfg(mt)

    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    # Stem declares compatibility with qwen3 / qwen3_moe.
    for mt in ("qwen3", "qwen3_moe"):
        _check_model_type(algo, _HF(mt))  # must not raise
    for mt in ("qwen3_5", "llama", "qwen2"):
        try:
            _check_model_type(algo, _HF(mt))
            raise AssertionError(f"{mt} should be blocked by model_type guard")
        except IncompatibleConfigError:
            pass


# ===========================================================================
# KV-compression names blocked at the registry
# ===========================================================================
def test_kv_compression_blocked():
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    for kv in ("snapkv", "streamingllm", "pyramidkv", "quest"):
        try:
            SparsityAlgorithmRegistry.create(kv)
            raise AssertionError(f"{kv} should be blocked")
        except NotImplementedError:
            pass


def test_unknown_algorithm_name_raises_valueerror():
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    assert "stem" in SparsityAlgorithmRegistry.available()
    try:
        SparsityAlgorithmRegistry.create("definitely_not_an_algo")
        raise AssertionError("unknown algo name should ValueError")
    except ValueError:
        pass


# ===========================================================================
# Distributed & runtime hard-fails
# MOVED to dedicated files (no duplication — single source):
#   tests/sparse/test_distributed_hardfails.py  (multi-node, TP)
#   tests/sparse/test_runtime_guards.py          (CUDA-graph / vLLM)
# ===========================================================================


# ===========================================================================
# FP8-attn collision guard
# ===========================================================================
def test_fp8_attn_collision_guard():
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3(num_layers=2)
    slim = _FakeSlim(m)
    # Signal 1: an explicit fp8_attn_active flag.
    slim.fp8_attn_active = True
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected FP8-attn collision IncompatibleConfigError")
    except IncompatibleConfigError as e:
        assert "fp8_attn" in str(e)


def test_fp8_attn_collision_guard_via_label():
    """A 'qwen_fp8' label already on the LIFO stack must also block sparse."""
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3(num_layers=2)
    slim = _FakeSlim(m)
    # Simulate an FP8-attn patch already pushed onto the stack.
    fake_attn = m.model.layers[0].self_attn
    slim.push_attn_forward("qwen_fp8", fake_attn, lambda *a, **k: None)
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected FP8-attn label collision error")
    except IncompatibleConfigError as e:
        assert "fp8_attn" in str(e)


# ===========================================================================
# Kernel hard-fail vs pseudo-sparse opt-in
# ===========================================================================
def test_kernel_hardfail_and_pseudo_optin():
    # Simulate missing kernel by forcing HAS_BLOCK_SPARSE_KERNEL = False.
    import angelslim.compressor.sparsity.algorithms.stem.backends.torch_impl as ti
    from angelslim.compressor.sparsity._kernel_check import KernelMissingError

    saved = ti.HAS_BLOCK_SPARSE_KERNEL
    ti.HAS_BLOCK_SPARSE_KERNEL = False
    try:
        if not CUDA:
            raise _Skip("forward needs CUDA; skip the actual call off-GPU")
        q = torch.randn(1, 4, 300, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 4, 300, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 4, 300, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        # hard-fail by default
        try:
            ti.stem_forward_torch(
                q,
                k,
                v,
                {
                    "attn_forward_config": {"block_size": 128, "allow_pseudo_sparse": False},
                    "layer_idx": 0,
                },
            )
            raise AssertionError("expected KernelMissingError")
        except KernelMissingError as e:
            assert "allow_pseudo_sparse" in str(e)
        # pseudo-sparse opt-in succeeds
        out = ti.stem_forward_torch(
            q,
            k,
            v,
            {
                "attn_forward_config": {"block_size": 128, "allow_pseudo_sparse": True},
                "layer_idx": 0,
            },
        )
        assert out.shape == q.shape
        assert not torch.isnan(out).any()
    finally:
        ti.HAS_BLOCK_SPARSE_KERNEL = saved


def test_allow_pseudo_sparse_threads_from_framework():
    """The ``allow_pseudo_sparse``
    opt-in must thread from the algorithm's ``attn_kwargs`` through ``setup()`` to
    the per-module ``attn_forward_config`` the backend's kernel gate reads.
    This used to be surfaced as ``run_stem.py --allow-pseudo-sparse`` (CLI
    removed) and, before the framework migration, via the legacy ``StemInference``
    standalone class (removed once Stem became a registered SparsityAlgorithm).
    The unified ``tools/run.py -c <sparse.yaml>`` path carries the same key via
    YAML ``attn_kwargs.allow_pseudo_sparse``; the config plumbing this asserts is
    the load-bearing part and is unchanged. Pure config-plumbing; no CUDA needed."""
    import warnings

    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    # The flag set to True must land in the per-instance config setup() builds.
    on = SparsityAlgorithmRegistry.create(
        "stem", attn_kwargs={"backend": "torch", "allow_pseudo_sparse": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on.setup(_tiny_qwen3(num_layers=4))
    assert on._attn_forward_config.get("allow_pseudo_sparse") is True

    # Default (flag absent) means the missing-kernel hard-fail stays in force.
    off = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off.setup(_tiny_qwen3(num_layers=4))
    assert off._attn_forward_config.get("allow_pseudo_sparse", False) is False


def test_require_kernel_helper_policy():
    """The shared require_kernel gate: hard-fail vs warn-and-fallback."""
    import warnings

    from angelslim.compressor.sparsity._kernel_check import (
        KernelMissingError,
        kernel_available,
        require_kernel,
    )

    # A present module returns True regardless of the flag.
    assert kernel_available("torch") is True
    assert require_kernel("torch", "n/a", algo_name="stem", allow_pseudo_sparse=False)

    # A missing module hard-fails by default ...
    try:
        require_kernel(
            "no_such_kernel_xyz",
            "pip install x",
            algo_name="stem",
            allow_pseudo_sparse=False,
        )
        raise AssertionError("expected KernelMissingError")
    except KernelMissingError as e:
        assert "allow_pseudo_sparse" in str(e)

    # ... and returns False (with a warning) when pseudo-sparse is opted in.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = require_kernel(
            "no_such_kernel_xyz",
            "pip install x",
            algo_name="stem",
            allow_pseudo_sparse=True,
        )
    assert ok is False
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_stem_forward_padding_mask_rejected():
    """Stem prefill requires a no-padding mask. Stem now uses the SHARED
    assert_no_padding_mask (dtype-aware, handles 2D/4D-float/4D-bool) instead of
    its old weaker last-row-only 4D check."""
    from angelslim.compressor.sparsity.algorithms._forward_templates._common import (
        assert_no_padding_mask,
    )

    K = 4
    # Clean 4-D float mask (causal-only) passes.
    causal = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    clean = torch.zeros(1, 1, K, K).masked_fill(causal[None, None], float("-inf"))
    assert_no_padding_mask(clean, K)

    # A masked entry on an attendable key (padding) is rejected — both with -inf
    # and with the finite finfo.min transformers actually emits.
    for fill in (float("-inf"), torch.finfo(torch.float32).min):
        m = clean.clone()
        m[0, 0, -1, 0] = fill
        try:
            assert_no_padding_mask(m, K)
            raise AssertionError(f"padded mask ({fill}) should be rejected")
        except ValueError:
            pass

    # 2D padding mask is also caught (the old stem check ignored 2D entirely).
    pad2d = torch.ones(1, K, dtype=torch.long)
    pad2d[0, -1] = 0
    try:
        assert_no_padding_mask(pad2d, K)
        raise AssertionError("2D padded mask should be rejected")
    except ValueError:
        pass


# ===========================================================================
# Stem forward correctness vs dense — REAL Qwen3-8B weights (user constraint)
# ===========================================================================
def test_stem_real_weights_keep_one_matches_dense():
    """keep=1.0 sparse ~= dense on the REAL Qwen3-8B (GQA ratio 4, 36 layers).

    This is the load-bearing correctness test: random-init weights cannot
    surface GQA / repeat_kv / per-layer-schedule bugs, so it MUST run on real
    weights. Asserts low relative error AND last-token argmax agreement.
    """
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)

    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, 2048), device="cuda")
    with torch.no_grad():
        dense = m(ids).logits.float()

    n = m.config.num_hidden_layers  # 36
    with _patched_real(
        m,
        {
            "backend": "torch",
            "block_size": 128,
            "stride": 8,
            "chunk_size": 2048,
            "layer_keep_ratios": [1.0] * n,
            "initial_blocks": 0,
            "window_size": 0,
        },
    ) as (slim, algo, patched):
        assert len(patched) == n
        # idempotent re-apply on the real model
        from angelslim.compressor.sparsity.patcher import apply_sparsity_patch

        assert len(apply_sparsity_patch(slim, algo)) == 0
        with torch.no_grad():
            sp = m(ids).logits.float()

    assert not torch.isnan(sp).any()
    rel = ((dense - sp).abs().mean() / dense.abs().mean()).item()
    assert rel < 0.05, f"keep=1.0 sparse should ~= dense, rel={rel}"
    assert dense[0, -1].argmax() == sp[0, -1].argmax(), "last-token argmax drift"

    # unpatch restores dense exactly (the shared singleton must be pristine)
    with torch.no_grad():
        restored = m(ids).logits.float()
    assert torch.equal(restored, dense), "unpatch did not restore real model"
    _cap("qwen3_real_correctness")


def test_stem_real_weights_long_seq_quality():
    """Default interpolated schedule on REAL Qwen3-8B @ 8192 tokens preserves
    semantics: finite, sparsity engaged, high per-position top-1 agreement.

    Sparsity genuinely engages at this length (64 query blocks > the 56-block
    short-seq relaxation threshold), so this measures real block-selection
    quality — only meaningful on real weights.
    """
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")

    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, 8192), device="cuda")
    with torch.no_grad():
        dense = m(ids).logits.float()

    # No explicit ratios => interpolate_default_schedule onto 36 layers.
    with _patched_real(
        m,
        {
            "backend": "torch",
            "block_size": 128,
            "stride": 8,
            "chunk_size": 2048,
            "stem_alpha": 0.7,
            "initial_blocks": 4,
            "window_size": 4,
        },
    ):
        with torch.no_grad():
            sp = m(ids).logits.float()

    assert torch.isfinite(sp).all() and not torch.isnan(sp).any()
    rel = ((dense - sp).abs().mean() / dense.abs().mean()).item()
    # `rel > 0.0` was a near-tautology (any difference passed). Bound it on
    # BOTH sides: sparsity must measurably engage (lower bound), but a correct
    # interpolated schedule at keep~0.2 must NOT diverge wildly from dense (upper
    # bound) — a blown-up rel would mean the block selection is broken even if
    # argmax happened to survive. Measured ground truth here: rel ~0.05-0.15.
    assert 1e-3 < rel < 0.5, f"long-seq sparse rel out of expected band: {rel:.4f}"
    agree = (dense[0].argmax(-1) == sp[0].argmax(-1)).float().mean().item()
    # Measured ground truth on this checkpoint: ~0.977. Guard well below it.
    assert agree > 0.90, f"per-position top-1 agreement too low: {agree:.3f}"


def test_stem_real_weights_decode_generates():
    """After sparse prefill, the decode path (q_len==1) generates the correct
    answer on the REAL model with keep=1.0 — exercises the fa2 decode fallback
    AND asserts end-to-end semantic correctness (a factual completion).

    keep=1.0 makes sparse prefill ~= dense, so greedy decoding of a factual
    prompt must produce the known answer. A random-id prompt can't assert this;
    real weights + a real tokenizer can.
    """
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_REAL_W)
    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]

    n = m.config.num_hidden_layers
    with _patched_real(
        m,
        {
            "backend": "torch",
            "block_size": 128,
            "layer_keep_ratios": [1.0] * n,
            "initial_blocks": 0,
            "window_size": 0,
        },
    ):
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"sparse-prefill decode lost the answer: {text!r}"


def _build_split_device_map(n_layers):
    """An explicit 2-device ``device_map`` that genuinely shards Qwen3 layers.

    ``device_map="auto"`` would place all of a 16 GB model on one of our 97 GB
    H20s (no split), so it cannot exercise the cross-device path. An explicit
    map forces the first half of the decoder onto cuda:0 and the second half
    onto cuda:1 — reproducing, on small weights, exactly what accelerate does
    when a genuinely-large Qwen3 (e.g. 32B / 235B-MoE) is spread across GPUs.
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


def test_stem_real_weights_device_map_sharded_matches_dense():
    """Sparse patch is correct under accelerate ``device_map`` LAYER SHARDING.

    This is the load-bearing test for "support sparse on LARGER Qwen3/3.5": the
    only single-rank way to fit a model bigger than one GPU is accelerate's
    device_map, which places whole decoder layers on different devices and runs
    them sequentially in ONE process (NOT tensor parallelism — see
    ``test_detect_tp_device_map_is_not_tp``; the patcher must NOT refuse this).

    Verifies, on a genuinely 2-device-sharded REAL Qwen3-8B, that:
      * the patch applies to every layer regardless of which device it lives on,
      * keep=1.0 sparse ~= that same sharded model's own dense (argmax agrees) —
        i.e. the patched forward respects the per-layer device the accelerate
        hooks moved tensors to (reference.py derives everything from q.device),
      * unpatch restores the sharded model byte-exactly.

    Random-init weights cannot surface a cross-device bug that only bites when a
    real GQA/repeat_kv + block-selection forward runs on cuda:1's tensors, so
    this MUST use real weights — and it MUST be a separate, caller-owned
    instance (the shared singleton is single-cuda)."""
    if not real_weights_available():
        raise _Skip("real Qwen3-8B weights unavailable")
    if torch.cuda.device_count() < 2:
        raise _Skip("device_map sharding test needs >= 2 GPUs")
    from transformers import Qwen3ForCausalLM

    cfg_n = Qwen3ForCausalLM.config_class.from_pretrained(_REAL_W).num_hidden_layers
    model = Qwen3ForCausalLM.from_pretrained(
        _REAL_W,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=_build_split_device_map(cfg_n),
    ).eval()
    try:
        # Confirm the model really is sharded across >= 2 devices (else the test
        # would silently degrade to a single-device check and prove nothing).
        used = {str(d) for d in model.hf_device_map.values()}
        assert (
            len({u for u in used if u != "cpu"}) >= 2
        ), f"device_map did not shard across 2 GPUs: {model.hf_device_map}"

        torch.manual_seed(0)
        # inputs go to the embedding's device; accelerate hooks move activations
        # across the cuda:0 -> cuda:1 boundary between layers.
        ids = torch.randint(0, 151000, (1, 2048), device="cuda:0")
        with torch.no_grad():
            dense = model(ids).logits.float().cpu()

        n = model.config.num_hidden_layers
        with _patched_real_model(
            model,
            {
                "backend": "torch",
                "block_size": 128,
                "stride": 8,
                "chunk_size": 2048,
                "layer_keep_ratios": [1.0] * n,
                "initial_blocks": 0,
                "window_size": 0,
            },
        ) as (slim, algo, patched):
            assert len(patched) == n, "every layer must be patched across devices"
            with torch.no_grad():
                sp = model(ids).logits.float().cpu()

        assert not torch.isnan(sp).any(), "sharded sparse produced NaN"
        rel = ((dense - sp).abs().mean() / dense.abs().mean()).item()
        assert rel < 0.05, f"sharded keep=1.0 sparse should ~= dense, rel={rel}"
        assert dense[0, -1].argmax() == sp[0, -1].argmax(), "sharded last-token argmax drift"
        with torch.no_grad():
            restored = model(ids).logits.float().cpu()
        assert torch.equal(
            restored, dense
        ), "unpatch did not restore the sharded model byte-exactly"
    finally:
        del model
        torch.cuda.empty_cache()


def test_stem_forward_synthetic_structure_and_idempotent():
    """Synthetic-model structural check: patch count + idempotency + exact
    unpatch. Weight-independent (asserts structure, not numerics), so a tiny
    random-init model is the right tool — fast, no checkpoint needed."""
    if not CUDA:
        raise _Skip("CUDA required")  # kernel path needs CUDA
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    torch.manual_seed(0)
    m = _tiny_qwen3(num_layers=4)
    slim = _FakeSlim(m)
    ids = torch.randint(0, 512, (1, 600), device="cuda")
    with torch.no_grad():
        before = m(ids).logits.float()

    algo = SparsityAlgorithmRegistry.create(
        "stem",
        attn_kwargs={
            "backend": "torch",
            "block_size": 128,
            "stride": 8,
            "chunk_size": 2048,
            "layer_keep_ratios": [1.0] * 4,
            "initial_blocks": 0,
            "window_size": 0,
        },
    )
    patched = apply_sparsity_patch(slim, algo)
    assert len(patched) == 4
    # idempotent: second apply must not double-patch.
    assert len(apply_sparsity_patch(slim, algo)) == 0

    with torch.no_grad():
        sp = m(ids).logits.float()
    assert not torch.isnan(sp).any()  # structural: runs, finite

    # unpatch restores the original forward exactly.
    unpatch_sparsity(slim, patched)
    with torch.no_grad():
        restored = m(ids).logits.float()
    assert torch.equal(restored, before)


def test_stem_forward_on_qwen3_moe():
    """Stem patches Qwen3-MoE (the 2nd compatible model_type) and runs."""
    if not CUDA:
        raise _Skip("CUDA required")
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    torch.manual_seed(0)
    m = _tiny_qwen3_moe(num_layers=2)
    slim = _FakeSlim(m)
    ids = torch.randint(0, 256, (1, 400), device="cuda")
    with torch.no_grad():
        dense = m(ids).logits.float()
    algo = SparsityAlgorithmRegistry.create(
        "stem",
        attn_kwargs={
            "backend": "torch",
            "block_size": 128,
            "layer_keep_ratios": [1.0, 1.0],
            "initial_blocks": 0,
            "window_size": 0,
        },
    )
    patched = apply_sparsity_patch(slim, algo)
    assert len(patched) == 2
    with torch.no_grad():
        sp = m(ids).logits.float()
    assert not torch.isnan(sp).any()
    rel = ((dense - sp).abs().mean() / dense.abs().mean()).item()
    assert rel < 0.05, f"keep=1.0 MoE sparse should ~= dense, rel={rel}"
    unpatch_sparsity(slim, patched)
    with torch.no_grad():
        assert torch.equal(m(ids).logits.float(), dense)


def test_stem_backend_long_seq_sparsity_engages():
    """At Qb>=56 the keep<1.0 schedule genuinely drops blocks (vs dense)."""
    if not CUDA:
        raise _Skip("CUDA required")
    from angelslim.compressor.sparsity.algorithms.stem.backends.torch_impl import (
        stem_forward_torch,
    )

    torch.manual_seed(0)
    B, H, D = 1, 4, HEAD_DIM
    L = 64 * 128  # 8192 tokens -> 64 query blocks -> long-seq regime
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)

    def run(keep):
        cfg = {
            "attn_forward_config": {
                "block_size": 128,
                "stride": 8,
                "chunk_size": 2048,
                "stem_alpha": 1.0,
                "layer_keep_ratios": [keep] * 4,
                "initial_blocks": 0,
                "window_size": 0,
                "allow_pseudo_sparse": False,
            },
            "layer_idx": 0,
        }
        return stem_forward_torch(q, k, v, cfg)

    o_full = run(1.0)
    o_sparse = run(0.2)
    assert torch.isfinite(o_full).all() and torch.isfinite(o_sparse).all()
    rel = ((o_full - o_sparse).abs().mean() / o_full.abs().mean()).item()
    assert rel > 0.01, f"long-seq keep=0.2 should differ from dense, rel={rel}"


def test_stem_pseudo_sparse_matches_kernel_at_keep_one():
    """Pseudo-sparse fallback ~= the real kernel when keep=1.0 (numeric parity)."""
    if not CUDA:
        raise _Skip("CUDA required")
    import angelslim.compressor.sparsity.algorithms.stem.backends.torch_impl as ti

    torch.manual_seed(0)
    B, H, D = 1, 4, HEAD_DIM
    L = 1024
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    cfg = {
        "attn_forward_config": {
            "block_size": 128,
            "stride": 8,
            "chunk_size": 2048,
            "stem_alpha": 1.0,
            "layer_keep_ratios": [1.0] * 4,
            "initial_blocks": 0,
            "window_size": 0,
            "allow_pseudo_sparse": True,
        },
        "layer_idx": 0,
    }

    if not ti.HAS_BLOCK_SPARSE_KERNEL:
        raise _Skip("block_sparse_attn kernel absent — no kernel to compare against")
    o_kernel = ti.stem_forward_torch(q, k, v, cfg)
    saved = ti.HAS_BLOCK_SPARSE_KERNEL
    ti.HAS_BLOCK_SPARSE_KERNEL = False
    try:
        o_pseudo = ti.stem_forward_torch(q, k, v, cfg)
    finally:
        ti.HAS_BLOCK_SPARSE_KERNEL = saved
    rel = ((o_kernel - o_pseudo).abs().mean() / o_kernel.abs().mean()).item()
    assert rel < 0.05, f"pseudo-sparse should ~= kernel at keep=1.0, rel={rel}"


# ===========================================================================
# Sparsity compressor registration + lifecycle (no-op run/convert/unpatch)
# ===========================================================================
def test_sparsity_registered_in_compressor_factory():
    import angelslim.compressor  # noqa: F401  ensure import side-effects
    from angelslim.compressor.compressor_factory import CompressorFactory

    assert "Sparsity" in CompressorFactory.get_available_compressor()


def test_sparsity_requires_sparsity_subconfig():
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.sparsity import Sparsity

    class _Slim:
        def __init__(self):
            self.model = None

    bad = type("C", (), {"sparsity": None})()
    try:
        Sparsity(_Slim(), {"compress_config": bad})
        raise AssertionError("expected ValueError when sparsity sub-config missing")
    except ValueError:
        pass


def test_sparsity_convert_idempotent_and_unpatch():
    if not CUDA:
        raise _Skip("CUDA required")
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.sparsity import Sparsity
    from angelslim.utils.config_parser import SparsityConfig

    m = _tiny_qwen3(num_layers=4)
    slim = _FakeSlim(m)
    sp_cfg = SparsityConfig(
        name="stem",
        attn_kwargs={
            "backend": "torch",
            "block_size": 128,
            "layer_keep_ratios": [1.0, 1.0, 0.2, 0.2],
        },
    )
    comp = type("C", (), {"sparsity": sp_cfg})()
    sparse = Sparsity(slim, {"compress_config": comp, "global_config": None})

    assert sparse.run() is None  # no-op
    sparse.convert()
    assert sparse._patched
    n_patched = len(sparse._patched_modules)
    assert n_patched == 4

    sparse.convert()  # idempotent: no extra patches
    assert len(sparse._patched_modules) == n_patched

    sparse.unpatch()
    assert not sparse._patched
    # Labels stack drained.
    assert slim.attn_forward_labels() == set()


def test_apply_sparsity_patch_is_atomic_on_failure():
    """If build_attn_forward throws partway through, apply_sparsity_patch
    must pop every patch pushed so far and re-raise — leaving the model pristine.
    Otherwise the orphaned patches have no handle (unpatch only sees the returned
    list) and leak onto the LIFO stack."""
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch

    m = _tiny_qwen3(num_layers=4)
    slim = _FakeSlim(m)

    calls = {"n": 0}

    class _BoomAlgo:
        """Algo stub: patches succeed until the 3rd layer, then raises."""

        class traits:
            compatible_model_types = None
            model_modal = "any"

        def setup(self, hf_model):
            pass

        def build_attn_forward(self, attn, hf_model):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise RuntimeError("boom on the 3rd layer")
            return lambda *a, **k: None

    try:
        apply_sparsity_patch(slim, _BoomAlgo())
        raise AssertionError("expected the algo's RuntimeError to propagate")
    except RuntimeError as e:
        assert "boom" in str(e)

    # The two patches pushed before the failure must have been rolled back.
    assert slim.attn_forward_labels() == set(), "patch was not atomic — leaked"
    assert slim._attn_forward_patches == {}, "patch registry not drained after rollback"


def test_apply_sparsity_patch_preserves_global_layer_idx():
    """The loop counter is the FILTERED position; attn.layer_idx must keep
    the module's real global index (HF sets it at construction) so KV-cache
    writes / best_pattern reads use the right slot. The patcher must NOT
    overwrite an existing layer_idx with the filtered counter."""
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    if not CUDA:
        raise _Skip("Stem backend build needs CUDA; layer_idx logic is the point")

    m = _tiny_qwen3(num_layers=4)
    # Simulate an interleaved model: pretend only layers 0 and 2 are sparsable by
    # giving them distinctive global indices that differ from the filtered order.
    # (Real Qwen3 already carries 0..3; we re-stamp 10/11/12/13 to prove the
    # patcher keeps the module's own value rather than the 0..n loop counter.)
    for gi, layer in zip((10, 11, 12, 13), m.model.layers):
        layer.self_attn.layer_idx = gi

    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create(
        "stem",
        attn_kwargs={"backend": "torch", "block_size": 128, "layer_keep_ratios": [1.0] * 4},
    )
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    try:
        got = [a.layer_idx for a in patched]
        assert got == [10, 11, 12, 13], f"global layer_idx clobbered with filtered counter: {got}"
    finally:
        unpatch_sparsity(slim, patched)


# ===========================================================================
# Save/load round-trip via Engine + YAML & JSON
# ===========================================================================
def test_save_load_roundtrip_config_only():
    if not CUDA:
        raise _Skip("CUDA required")
    import angelslim.compressor.sparsity.algorithms  # register  # noqa: F401
    from angelslim.compressor.sparsity.sparsity import Sparsity
    from angelslim.utils.config_parser import SparsityConfig

    m = _tiny_qwen3(num_layers=4)

    class _Tok:
        def save_pretrained(self, p):
            with open(os.path.join(p, "tokenizer.json"), "w") as f:
                f.write("{}")

    slim = _FakeSlim(m, tokenizer=_Tok())
    sp_cfg = SparsityConfig(
        name="stem",
        attn_kwargs={
            "backend": "torch",
            "block_size": 128,
            "layer_keep_ratios": [1.0, 1.0, 0.2, 0.2],
        },
    )
    comp = type("C", (), {"sparsity": sp_cfg})()
    sparse = Sparsity(slim, {"compress_config": comp, "global_config": None})
    sparse.convert()
    assert sparse._patched

    with tempfile.TemporaryDirectory() as d:
        sparse.save(d)
        files = set(os.listdir(d))
        # Weights + config + tokenizer present => loadable directory
        has_weights = (
            any(f.endswith(".safetensors") for f in files)
            or "model.safetensors.index.json" in files
        )
        assert has_weights, f"no weights saved: {files}"
        assert "config.json" in files, f"no config.json: {files}"
        assert "tokenizer.json" in files, f"no tokenizer: {files}"


def test_save_reload_repatch_real_weights():
    """Save/load end-to-end on REAL weights: patch -> save -> reload from disk ->
    re-patch -> the reloaded sparse model still answers correctly.

    This is the only test that proves the *saved bytes* round-trip — the
    synthetic save test above only checks file presence. Heavy (~16GB write),
    so gated behind STEM_REAL_SAVE=1; runs in CI's nightly lane, skipped by
    default. When it runs it uses real weights (random init would make the
    factual-answer assertion meaningless)."""
    if os.environ.get("STEM_REAL_SAVE") != "1":
        raise _Skip("set STEM_REAL_SAVE=1 to run the heavy real save round-trip")
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B weights unavailable")
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.sparsity import Sparsity
    from angelslim.utils.config_parser import SparsityConfig

    tok = AutoTokenizer.from_pretrained(_REAL_W)
    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    n = m.config.num_hidden_layers
    kwargs = {
        "backend": "torch",
        "block_size": 128,
        "layer_keep_ratios": [1.0] * n,
        "initial_blocks": 0,
        "window_size": 0,
    }

    # Patch the shared model, save it, then unpatch to leave the singleton clean.
    slim = _RealSlim(m, tokenizer=tok)
    sp_cfg = SparsityConfig(name="stem", attn_kwargs=dict(kwargs))
    comp = type("C", (), {"sparsity": sp_cfg})()
    sparse = Sparsity(slim, {"compress_config": comp, "global_config": None})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sparse.convert()
        with tempfile.TemporaryDirectory(dir=_SCRATCH) as d:
            sparse.save(d)
            # Reload as a fresh model from the saved directory.
            reloaded = (
                Qwen3ForCausalLM.from_pretrained(
                    d,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                .to("cuda")
                .eval()
            )
    finally:
        sparse.unpatch()

    # Re-apply sparse to the reloaded model and confirm it still answers.
    with _patched_real_model(reloaded, kwargs):
        with torch.no_grad():
            out = reloaded.generate(**enc, max_new_tokens=8, do_sample=False)
    del reloaded
    torch.cuda.empty_cache()
    text = tok.decode(out[0, n_in:])
    assert "Paris" in text, f"reloaded sparse model lost the answer: {text!r}"


def _patched_real_model(model, attn_kwargs):
    """Like _patched_real but for a caller-owned (non-singleton) model."""
    return _patched_real(model, attn_kwargs)


def test_config_json_roundtrip_recovers_sparsity():
    import json

    from angelslim.utils.config_parser import parse_json_full_config

    doc = {
        "model_config": {"name": "Qwen", "model_path": "x"},
        "compression_config": {
            "name": ["Sparsity"],
            "sparsity": {"name": "stem", "attn_kwargs": {"backend": "torch"}},
        },
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "angelslim_config.json")
        with open(p, "w") as f:
            json.dump(doc, f)
        cfg = parse_json_full_config(p)
    assert cfg.compression_config is not None, "compression_config dropped"
    assert "Sparsity" in cfg.compression_config.name
    assert cfg.compression_config.sparsity is not None
    assert cfg.compression_config.sparsity.name == "stem"


def test_config_yaml_roundtrip_recovers_sparsity():
    """The shipped Stem YAML parses into a SparsityConfig with the right fields."""
    from angelslim.utils.config_parser import SlimConfigParser

    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "configs",
        "sparse",
        "stem",
        "qwen3-8b_stem_torch.yaml",
    )
    yaml_path = os.path.abspath(yaml_path)
    assert os.path.isfile(yaml_path), f"shipped YAML missing: {yaml_path}"

    cfg = SlimConfigParser().parse(yaml_path)
    comp = cfg.compression_config
    assert comp is not None
    assert "Sparsity" in comp.name
    assert comp.sparsity is not None
    assert comp.sparsity.name == "stem"
    assert comp.sparsity.allow_pseudo_sparse is False  # default
    assert comp.sparsity.attn_kwargs["backend"] == "torch"
    assert comp.sparsity.attn_kwargs["block_size"] == 128
    # model-level attn_implementation default (q3): flash_attention_2
    assert cfg.model_config.attn_implementation == "flash_attention_2"


def test_yaml_sparsity_requires_subsection():
    """compression.name=Sparsity without a compression.sparsity block must fail."""
    import yaml as _yaml

    from angelslim.utils.config_parser import SlimConfigParser

    doc = {
        "model": {"name": "Qwen", "model_path": "x"},
        "compression": {"name": "Sparsity"},  # missing the sparsity sub-section
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        with open(p, "w") as f:
            _yaml.safe_dump(doc, f)
        try:
            SlimConfigParser().parse(p)
            raise AssertionError("expected ValueError for missing sparsity section")
        except ValueError as e:
            assert "sparsity" in str(e).lower()


# ===========================================================================
# Lightweight runner (env has no pytest)
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
