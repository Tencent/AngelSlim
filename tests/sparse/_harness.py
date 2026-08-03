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

"""Shared scaffolding for the ``tests/sparse`` suites — the SINGLE source of
truth for the helpers that were previously copy-pasted (and silently drifted)
across the 7 test files.

The LIFO test-double ``_Slim`` had previously been duplicated into 6 files and
**drifted** — 2 of 6 copies dropped the ``expected_label`` mismatch check (so
the LIFO-label contract was untested there), silently returned on an empty stack
instead of raising, and stubbed ``attn_forward_labels()`` to ``set()``. A
related follow-on: 2 of 7 ``_run_all`` copies were missing the coverage floor,
so a weightless box could still false-green.

Centralizing the contract-bearing pieces here makes that drift *structurally
impossible*: there is now exactly one ``FakeSlim``, one coverage-floored
``run_all``, one ``Skip``, one ``patched`` context manager. A test file that
needs the BaseLLMModel-like LIFO double imports it; it cannot accidentally ship
a subtly-different copy.

These helpers are intentionally minimal and assertion-free wrappers around the
production API — they encode the *contract* the production ``BaseLLMModel``
must honor (push/pop is a strict per-module LIFO with label verification), so a
test using ``FakeSlim`` is testing the real contract, not a permissive stub.

Import idiom (suites run as ``$PY tests/sparse/test_x.py`` — sibling on path):

    from _harness import Skip, FakeSlim, run_all, patched, rel
"""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from unittest import SkipTest as _SkipBase

import torch

# ---------------------------------------------------------------------------
# Environment constants (single definition; suites used to each redefine these)
# ---------------------------------------------------------------------------
CUDA = torch.cuda.is_available()
HEAD_DIM = 128  # block_sparse_attn / minference kernels are happy at 128


# ---------------------------------------------------------------------------
# Root resolution (single source of truth for all suites).
#
# Real-weights tests need to locate the checkpoints directory. We DERIVE the
# base rather than hardcoding it, in priority order:
#   1. $ANGELSLIM_HOME if set (explicit override — wins).
#   2. five dirs up from THIS file (<base>/workspace/AngelSlim/tests/sparse/
#      _harness.py -> <base>): correct by construction wherever the repo is
#      checked out, so moving the checkout needs zero edits.
# The derived base's ``weights/Qwen3-8B/config.json`` is probed; if it is absent
# the absence surfaces as the normal "weights absent -> tests SKIP" path (never a
# false green).
# ---------------------------------------------------------------------------
def _resolve_home() -> str:
    env = os.environ.get("ANGELSLIM_HOME")
    if env:
        return env
    here = os.path.abspath(__file__)
    derived = here
    for _ in range(5):  # tests/sparse/_harness.py -> <base>
        derived = os.path.dirname(derived)
    return derived


_HOME = _resolve_home()
_WEIGHTS = os.path.join(_HOME, "weights")

# Real checkpoints. Correctness tests load these; structural tests do not.
REAL_W_QWEN3_8B = os.path.join(_WEIGHTS, "Qwen3-8B")
REAL_W_QWEN3_5_9B = os.path.join(_WEIGHTS, "Qwen3.5-9B")
REAL_W_HY3 = os.path.join(_WEIGHTS, "Hy3")

# Big-FS scratch root for save/load tempdirs (the home FS, not a small /tmp).
SCRATCH_ROOT = _HOME


# ---------------------------------------------------------------------------
# Capability accounting: verify CAPABILITIES, not script names.
# A test calls record_capability("qwen3_real_correctness") when it ACTUALLY
# exercises that capability (i.e. it got past its skip guard and ran the real
# work). run_all tallies counts and emits a machine-readable CAPABILITIES: line;
# a release lane asserts a capability FLOOR (e.g. qwen35_real_correctness >= 1)
# instead of trusting that a named file exists. This closes the "structure-green
# while every real test skipped" hole that file-name checks cannot catch.
# ---------------------------------------------------------------------------
_CAPABILITIES: dict[str, int] = {}


def record_capability(name: str) -> None:
    """Record that capability ``name`` was genuinely exercised (post-skip)."""
    _CAPABILITIES[name] = _CAPABILITIES.get(name, 0) + 1


def reset_capabilities() -> None:
    _CAPABILITIES.clear()


class SkipReason:
    """Structured skip reasons. Free-text skip messages made it
    impossible for a release gate to declare "NO_QWEN35_9B is fatal" — the reason
    was buried in prose. These are stable string codes a gate can key on.

    A test skips with e.g. ``raise Skip("real Qwen3-8B unavailable",
    SkipReason.NO_QWEN3_8B)``. ``run_all`` tallies skips by reason and emits a
    machine-readable ``SKIPS:`` line so a release lane can fail on a fatal code.
    """

    NO_CUDA = "NO_CUDA"
    NO_QWEN3_8B = "NO_QWEN3_8B"
    NO_QWEN35_9B = "NO_QWEN35_9B"
    NO_TOKENIZER = "NO_TOKENIZER"
    NO_FLASH_ATTN = "NO_FLASH_ATTN"
    NO_BLOCK_SPARSE_ATTN = "NO_BLOCK_SPARSE_ATTN"
    NO_MULTI_GPU = "NO_MULTI_GPU"
    NO_LARGE_CKPT = "NO_LARGE_CKPT"
    HEAVY_GATED = "HEAVY_GATED"  # opt-in heavy e2e not enabled (env flag off)
    OTHER = "OTHER"

    _ALL = frozenset(
        {
            NO_CUDA,
            NO_QWEN3_8B,
            NO_QWEN35_9B,
            NO_TOKENIZER,
            NO_FLASH_ATTN,
            NO_BLOCK_SPARSE_ATTN,
            NO_MULTI_GPU,
            NO_LARGE_CKPT,
            HEAVY_GATED,
            OTHER,
        }
    )


class Skip(_SkipBase):
    """Raised to mark a test skipped; the runner counts these separately.

    Subclasses ``unittest.SkipTest`` so a future pytest run treats it as a skip,
    not a failure. Tests gate on hardware/weights by *raising* this, never by
    a bare ``return`` (a bare return counts as a vacuous PASS and hides coverage
    loss behind a green run).

    Carries an optional structured ``reason`` (a ``SkipReason`` code) so a
    gate can decide a skip is fatal for a given lane (e.g. NO_QWEN35_9B is fatal
    at release). Defaults to ``SkipReason.OTHER`` so existing ``Skip("text")``
    callers keep working unchanged.
    """

    def __init__(self, message="", reason=SkipReason.OTHER):
        super().__init__(message)
        self.reason = reason if reason in SkipReason._ALL else SkipReason.OTHER


# ---------------------------------------------------------------------------
# Weights / hardware availability
# ---------------------------------------------------------------------------
def _ckpt_present(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def real_weights_available() -> bool:
    """Real Qwen3-8B present AND CUDA — the precondition for 8B correctness."""
    return CUDA and _ckpt_present(REAL_W_QWEN3_8B)


def qwen35_available() -> bool:
    """Real Qwen3.5-9B present AND CUDA — the precondition for Qwen3.5 tests."""
    return CUDA and _ckpt_present(REAL_W_QWEN3_5_9B)


def hy3_available() -> bool:
    """Real Hunyuan-V3 (hy_v3) checkpoint present AND CUDA — the precondition for
    the hy_v3 real-weights correctness surface. The full Hy3 is a large sharded
    MoE, so a partial download (config + some shards, no index yet) must NOT count
    as available: we require the safetensors index and every shard it names."""
    if not (CUDA and _ckpt_present(REAL_W_HY3)):
        return False
    index = os.path.join(REAL_W_HY3, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return False
    try:
        with open(index) as fh:
            shards = set(json.load(fh).get("weight_map", {}).values())
    except (OSError, ValueError):
        return False
    return bool(shards) and all(os.path.isfile(os.path.join(REAL_W_HY3, s)) for s in shards)


def tokenizer_available(path: str = REAL_W_QWEN3_8B) -> bool:
    """A checkpoint dir carries loadable tokenizer files (no CUDA needed).

    The synthetic save/load round-trip needs a real tokenizer to make a dir that
    ``from_pretrained`` will accept.
    """
    if not os.path.isdir(path):
        return False
    return any(
        os.path.isfile(os.path.join(path, f))
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    )


# ---------------------------------------------------------------------------
# Real-model singletons (load once — 8B is ~19 s / ~16 GB). Each suite runs in
# its OWN process (they are invoked separately), so the singleton is
# per-process; there is no cross-suite contamination, but WITHIN a process a
# leaked patch would corrupt later tests — hence ``patched`` below.
# ---------------------------------------------------------------------------
_REAL_8B = None
_REAL_8B_TOK = None
_REAL_35 = {}  # keyed by loader class name (LLM text-tower vs VLM wrapper)


def real_qwen3_8b(with_tokenizer: bool = False):
    """Load the real Qwen3-8B (fa2, bf16, cuda, eval) once and reuse.

    Returns the model, or (model, tokenizer) when ``with_tokenizer``. Returns
    ``None`` / ``(None, None)`` if CUDA or the checkpoint is unavailable — the
    caller must ``raise Skip(...)`` on that, never silently pass.
    """
    global _REAL_8B, _REAL_8B_TOK
    if not real_weights_available():
        return (None, None) if with_tokenizer else None
    if _REAL_8B is None:
        from transformers import Qwen3ForCausalLM

        _REAL_8B = (
            Qwen3ForCausalLM.from_pretrained(
                REAL_W_QWEN3_8B,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .to("cuda")
            .eval()
        )
    if with_tokenizer:
        if _REAL_8B_TOK is None:
            from transformers import AutoTokenizer

            _REAL_8B_TOK = AutoTokenizer.from_pretrained(REAL_W_QWEN3_8B)
        return _REAL_8B, _REAL_8B_TOK
    return _REAL_8B


# ---------------------------------------------------------------------------
# Tiny synthetic Qwen3 (structure-only tests — random init, fast, no checkpoint)
# Shared so the distributed/runtime-guard suites and the stem suite build
# the same fixture instead of drifting copies.
# ---------------------------------------------------------------------------
def tiny_qwen3(num_layers=4, attn_impl="eager"):
    """A tiny random-init Qwen3ForCausalLM for structural/guard tests.

    Asserts STRUCTURE (patch counts, guard hard-fails, schedule math), never
    weight-dependent numerics — so random init is correct and fast here. CUDA
    moves it to bf16/cuda when available.
    """
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


def tiny_hy3(num_layers=4, attn_impl="eager"):
    """A tiny random-init HYV3ForCausalLM (Hunyuan-V3, model_type "hy_v3").

    Same role as ``tiny_qwen3`` for the Hunyuan-V3 forward template: asserts the
    template drives HYV3Attention (q_norm/k_norm/plain q_proj, no gate) — patch
    lifecycle + full-budget dense parity — without a real 16 GB checkpoint (none
    exists in this env). A tiny MoE (4 experts, top-2, 1 shared) keeps the build
    fast; head_dim 128 matches the sparse kernels. CUDA moves it to bf16/cuda.
    """
    from transformers import HYV3Config, HYV3ForCausalLM

    cfg = HYV3Config(
        vocab_size=512,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=4096,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=256,
        attn_implementation=attn_impl,
    )
    m = HYV3ForCausalLM(cfg)
    if CUDA:
        m = m.to("cuda", dtype=torch.bfloat16)
    return m.eval()


# ---------------------------------------------------------------------------
# Canonical BaseLLMModel-like LIFO double (previously duplicated + drifted)
# ---------------------------------------------------------------------------
class FakeSlim:
    """Minimal ``BaseLLMModel``-like wrapper exposing the patch API,
    faithfully mirroring the production contract (per-module SINGLE SLOT —
    sparse never stacks, so a module holds at most one patch):

      * ``push_attn_forward(label, attn, fwd)`` saves the current forward and
        installs ``fwd``; pushing onto an already-patched module **raises**
        (the "no stacking" invariant, enforced at the API layer).
      * ``pop_attn_forward(attn, expected_label=None)`` restores the patch on
        that module; if ``expected_label`` is given and the recorded label
        mismatches, it **raises** — and it **raises** when no patch is present.
        (The drifted copies silently skipped both checks, leaving the contract
        untested.)
      * ``attn_forward_labels()`` reports the real set of live labels (a drifted
        copy returned ``set()``, making label assertions vacuous).
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._attn_forward_patches = {}  # id(attn) -> (label, original_forward)
        self._original_attn_forwards = {}

    def push_attn_forward(self, label, attn, fwd):
        key = id(attn)
        if key in self._attn_forward_patches:
            existing = self._attn_forward_patches[key][0]
            raise RuntimeError(
                f"push_attn_forward: module already patched with {existing!r}; "
                f"sparse does not stack attention patches."
            )
        self._attn_forward_patches[key] = (label, attn.forward)
        attn.forward = fwd

    def pop_attn_forward(self, attn, expected_label=None):
        key = id(attn)
        if key not in self._attn_forward_patches:
            raise RuntimeError("no patch on stack")
        label, original = self._attn_forward_patches[key]
        if expected_label is not None and label != expected_label:
            raise RuntimeError(f"label mismatch: recorded {label!r}, expected {expected_label!r}")
        attn.forward = original
        del self._attn_forward_patches[key]

    def attn_forward_labels(self):
        return {label for (label, _orig) in self._attn_forward_patches.values()}


# ---------------------------------------------------------------------------
# Patch lifecycle context manager (real-weights sites leaked patches into
# the shared singleton on assertion failure — order-dependent contamination)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def patched(slim, algo):
    """Apply the sparsity patch, yield the handle list, ALWAYS unpatch.

    ``try/finally`` guarantees a shared singleton model is restored to its
    pristine forwards even if an assertion fails mid-test, so a failure in one
    test cannot silently corrupt the next.
    """
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handles = apply_sparsity_patch(slim, algo)
    try:
        yield handles
    finally:
        unpatch_sparsity(slim, handles)


# ---------------------------------------------------------------------------
# Numeric helper
# ---------------------------------------------------------------------------
def rel(a, b):
    """Mean relative error ``mean(|a-b|) / mean(|b|)`` as a python float."""
    return ((a - b).abs().mean() / b.abs().mean()).item()


# ---------------------------------------------------------------------------
# Coverage-floored test runner (the floor used to be copied per-file and 2
# of 7 copies omitted it; centralizing makes the floor uniform and mandatory)
# ---------------------------------------------------------------------------
def run_all(test_globals, summary_extra: str = "") -> int:
    """Run every ``test_*`` callable in ``test_globals`` and return an exit code.

    Returns 0 only if at least one test actually ran AND none failed. A run where
    nothing asserted (everything skipped) returns 1 — the coverage floor — so
    a weightless / CPU box can never print a green that means "nothing ran".

    ``summary_extra`` is appended to the summary line (e.g. ``"real_weights=True"``)
    so each suite keeps its environment annotation.

    Emits two machine-readable lines a release gate can parse —
    ``CAPABILITIES: {json}`` (what was genuinely exercised, via record_capability)
    and ``SKIPS: {json}`` (skip counts keyed by SkipReason). This lets a lane
    assert capability/skip floors instead of trusting that a named file exists.
    """
    import json
    import traceback

    reset_capabilities()  # fresh per-suite tally
    tests = [v for k, v in sorted(test_globals.items()) if k.startswith("test_") and callable(v)]
    passed = failed = skipped = 0
    skip_reasons: dict[str, int] = {}
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS  {name}")
            passed += 1
        except Skip as e:
            reason = getattr(e, "reason", SkipReason.OTHER)
            print(f"SKIP  {name}: [{reason}] {e}")
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
        finally:
            # Inter-test CUDA hygiene: drain the stream and free cached blocks
            # between tests. General robustness for suites that load several large
            # models in one process (a cached single-device model + a separate
            # device_map-sharded model + a Qwen3.5 model) — keeps per-test memory
            # state predictable. NOTE: this is NOT a crash fix on its own — a CUDA
            # context already poisoned by an illegal-access cannot be recovered by
            # empty_cache; correctness bugs must be fixed in the tests/kernels
            # themselves. This is pure cleanup: it never changes a test's logic.
            if CUDA:
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001  - cleanup must never mask a result
                    pass
    tail = f" ({summary_extra})" if summary_extra else ""
    print(f"\n=== {passed} passed, {failed} failed, {skipped} skipped{tail} ===")
    # Structured accounting (stable, grep/JSON-parseable prefixes).
    print(f"CAPABILITIES: {json.dumps(dict(sorted(_CAPABILITIES.items())))}")
    print(f"SKIPS: {json.dumps(dict(sorted(skip_reasons.items())))}")
    if passed == 0:
        print("ERROR: 0 tests actually ran (all skipped) — treating as FAIL")
        return 1
    return failed
