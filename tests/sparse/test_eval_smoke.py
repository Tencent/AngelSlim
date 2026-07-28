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

"""Evaluation-driver + performance/correctness smoke (merged).

Folds three former files into one eval/regression suite:

  * longbench_lmeval driver helpers (structural, no weights/network): the default
    variant set covers dense + every registered algorithm; the score/collect
    helpers parse lm-eval's group/subgroup/leaf result shapes.
  * patch-overhead lower bound (real Qwen3-8B): streaming-mode

    prefill ratio floors + decode early-exit ceiling catch a per-forward rewalk
    regression.
  * long-context top-1 correctness (real Qwen3-8B @ 16K): one test per algorithm,
    materialized by name, asserting top-1 agreement vs dense stays above the
    per-algorithm floor.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA  # noqa: E402
from _harness import real_weights_available  # noqa: E402
from _harness import run_all  # noqa: E402
from _harness import REAL_W_QWEN3_8B as _REAL_W  # noqa: E402
from _harness import FakeSlim as _Slim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import real_qwen3_8b as _real_qwen3_8b  # noqa: E402
from _harness import record_capability as _cap  # noqa: E402

# ===========================================================================
# longbench_lmeval driver helpers (structural, no weights)
# ===========================================================================
# The driver + shared helpers live under evaluation/; import them directly.
_EVAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evaluation",
)
sys.path.insert(0, _EVAL)


def _drv():
    import longbench_lmeval as L  # noqa: F401

    return L


# ===========================================================================
# Default variant sweep == dense + the live registry
# ===========================================================================
def test_default_variants_cover_dense_plus_every_registered_algorithm():
    L = _drv()
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    registered = set(SparsityAlgorithmRegistry.available())
    assert "dense" not in registered
    assert "stem" in registered  # the algo the v2 run once dropped — must be in
    listed = {v for v in L.DEFAULT_VARIANTS.split(",") if v}
    assert listed == ({"dense"} | registered), (
        f"default sweep {sorted(listed)} != dense + registry " f"{sorted({'dense'} | registered)}"
    )


# ===========================================================================
# Result-collection helpers parse an lm-eval results dict
# ===========================================================================
def test_score_of_handles_key_forms_and_percent_scaling():
    L = _drv()
    # lm-eval stores fractions in [0,1]; we scale to 0-100.
    assert L._score_of({"score,none": 0.3079}) == 30.79
    assert L._score_of({"acc,none": 0.25}) == 25.0
    # already-percent values pass through (no double scaling)
    assert L._score_of({"score": 42.0}) == 42.0
    assert L._score_of({}) is None


def test_collect_reads_group_subgroup_and_leaf():
    L = _drv()
    results = {
        "results": {
            "longbench": {"score,none": 0.3079},
            "longbench_single": {"score,none": 0.14},
            "longbench_code": {"score,none": 0.122},
            "longbench_narrativeqa": {"score,none": 0.058},
            "longbench_lcc": {"score,none": 0.045},
        }
    }
    out = L._collect(results)
    assert out["overall"] == 30.79  # group rollup
    assert out["by_type"]["single"] == 14.0  # subgroup
    assert out["by_type"]["code"] == 12.2
    assert out["by_task"]["narrativeqa"] == 5.8  # leaf (×100, rounded)
    assert out["by_task"]["lcc"] == 4.5


def test_collect_falls_back_to_leaf_mean_without_group_rollup():
    """A subset run (named leaf tasks, no `longbench` group) still yields an
    overall = unweighted mean of the leaf scores."""
    L = _drv()
    results = {
        "results": {
            "longbench_narrativeqa": {"score,none": 0.10},
            "longbench_lcc": {"score,none": 0.20},
        }
    }
    out = L._collect(results)
    assert out["overall"] == 15.0  # mean(10.0, 20.0)
    assert out["by_type"] == {}  # no subgroup rollups present


# ===========================================================================
# Patch-overhead lower bound (/, real Qwen3-8B)

# ===========================================================================
# Light iteration counts — this is a gate, not a benchmark. Enough to median out
# a single hiccup; not enough to be slow inside sparse_verify.sh.
_PREFILL_WARMUP = 2
_PREFILL_ITERS = 5
_DECODE_WARMUP = 3
_DECODE_ITERS = 10

# Per-regime ratio floors/ceilings (see module docstring for derivation).
# a_shape/tri_shape floor raised 0.80 -> 0.95. After the no_grad fix

# (symmetric timing) their measured 4K ratio is ~1.5-1.6x even under keepalive
# contention, so 0.95 has ample margin yet catches a real >5% regression (the
# old 0.80 silently passed a 20% regression). minference stays exempt from a
# >=1.0 floor — it is long-context-only (measured ~0.36x @4K, the win is >=32K);
# 0.15 only asserts "not catastrophic" (a per-forward re-walk would collapse it).
_PREFILL_FLOOR = {"a_shape": 0.95, "tri_shape": 0.95, "minference": 0.15}
_DECODE_CEIL = 2.0


def _patch_real(model, variant, attn_kwargs):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    algo = SparsityAlgorithmRegistry.create(variant, attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _median_latency_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _variant_kwargs(variant, seq_len):
    if variant == "a_shape":
        return {"n_init": 64, "n_local": max(256, seq_len // 4)}
    if variant == "tri_shape":
        return {"n_init": 64, "n_local": max(256, seq_len // 4), "n_last": 100}
    if variant == "minference":
        return {}
    raise ValueError(variant)


def _measure_prefill_ratio(model, variant, seq_len, dense_ms):
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    ids = torch.randint(0, 151000, (1, seq_len), device="cuda")
    slim, patched = _patch_real(model, variant, _variant_kwargs(variant, seq_len))

    @torch.no_grad  # same protocol as the dense baseline (no autograd
    def _run():  # overhead asymmetry — both paths are inference-only).
        model(ids)

    try:
        sp_ms = _median_latency_ms(_run, _PREFILL_WARMUP, _PREFILL_ITERS)
    finally:
        unpatch_sparsity(slim, patched)
    return dense_ms / sp_ms  # speedup; higher = faster than dense


def _measure_decode_ratio(model, variant, dense_ms):
    """Decode = q_len==1 step. Build a real KV cache via a short prefill, then
    time a single-token forward (the path that MUST early-exit in sparse)."""
    from transformers import DynamicCache

    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    def _one_decode_step():
        prompt = torch.randint(0, 151000, (1, 64), device="cuda")
        cache = DynamicCache()
        with torch.no_grad():
            out = model(prompt, use_cache=True, past_key_values=cache)
        nxt = out.logits[:, -1:].argmax(-1)
        pos = torch.tensor([[prompt.shape[1]]], device="cuda")

        def _step():
            with torch.no_grad():
                model(nxt, use_cache=True, past_key_values=cache, cache_position=pos[0])

        return _step

    slim, patched = _patch_real(model, variant, _variant_kwargs(variant, 4096))
    try:
        step = _one_decode_step()
        sp_ms = _median_latency_ms(step, _DECODE_WARMUP, _DECODE_ITERS)
    finally:
        unpatch_sparsity(slim, patched)
    return sp_ms / dense_ms  # latency ratio; lower = closer to dense


def _dense_prefill_ms(model, seq_len):
    ids = torch.randint(0, 151000, (1, seq_len), device="cuda")
    with torch.no_grad():
        return _median_latency_ms(lambda: model(ids), _PREFILL_WARMUP, _PREFILL_ITERS)


def _dense_decode_ms(model):
    from transformers import DynamicCache

    prompt = torch.randint(0, 151000, (1, 64), device="cuda")
    cache = DynamicCache()
    with torch.no_grad():
        out = model(prompt, use_cache=True, past_key_values=cache)
    nxt = out.logits[:, -1:].argmax(-1)
    pos = torch.tensor([prompt.shape[1]], device="cuda")

    def _step():
        with torch.no_grad():
            model(nxt, use_cache=True, past_key_values=cache, cache_position=pos)

    return _median_latency_ms(_step, _DECODE_WARMUP, _DECODE_ITERS)


# ===========================================================================
# prefill lower-bound — a_shape / tri_shape must not regress at 4K


# ===========================================================================
def test_prefill_lower_bound_streaming_modes_4k():
    """a_shape / tri_shape @4096 prefill must be >= 0.95x dense (measured ~1.5x
    with symmetric no_grad timing).

    These cheap sink+window modes never regress; a per-forward re-walk
    regression would drop them well below 0.5x."""

    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B unavailable")
    torch.manual_seed(0)
    with torch.no_grad():
        dense_ms = _dense_prefill_ms(m, 4096)
    for variant in ("a_shape", "tri_shape"):
        ratio = _measure_prefill_ratio(m, variant, 4096, dense_ms)
        floor = _PREFILL_FLOOR[variant]
        assert ratio >= floor, (
            f"{variant} @4096 prefill speedup {ratio:.3f}x < floor {floor}x "
            f"(dense={dense_ms:.1f}ms) — suspect a per-forward re-walk regression"
        )


# ===========================================================================
# prefill non-catastrophic — minference is long-context-only


# ===========================================================================
def test_prefill_minference_not_catastrophic_4k():
    """minference @4096 is EXPECTED slower than dense (long-context-only mode;
    measured ~0.36x). This gate only asserts it is not CATASTROPHIC (>= 0.15x) —
    the real minference win is at long context (2.78x @128K, see the benchmark).
    A per-forward re-walk regression would push this below ~0.05x."""
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B unavailable")
    torch.manual_seed(0)
    with torch.no_grad():
        dense_ms = _dense_prefill_ms(m, 4096)
    ratio = _measure_prefill_ratio(m, "minference", 4096, dense_ms)
    floor = _PREFILL_FLOOR["minference"]
    assert ratio >= floor, (
        f"minference @4096 prefill speedup {ratio:.3f}x < floor {floor}x "
        f"(dense={dense_ms:.1f}ms) — below the long-context-mode tolerance; "
        f"suspect a per-forward re-walk regression"
    )


# ===========================================================================
# decode early-exit — the load-bearing gate (all modes)


# ===========================================================================
def test_decode_early_exit_all_modes():
    """Decode (q_len==1) latency must be <= 2.0x dense for EVERY variant.

    This is the core assertion: the patched forward must early-exit on

    decode and NOT re-resolve layers / rebuild config per token. A re-walk
    regression makes every 1-token step pay prefill-style setup, blowing this
    ratio far past 2x."""
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B unavailable")
    torch.manual_seed(0)
    dense_ms = _dense_decode_ms(m)
    for variant in ("a_shape", "tri_shape", "minference"):
        ratio = _measure_decode_ratio(m, variant, dense_ms)
        assert ratio <= _DECODE_CEIL, (
            f"{variant} decode latency {ratio:.3f}x dense > ceil {_DECODE_CEIL}x "
            f"(dense={dense_ms:.2f}ms) — decode is NOT early-exiting; suspect a "
            f"per-call re-resolve (regression)"
        )


# ===========================================================================
# Long-context top-1 correctness (real Qwen3-8B @ 16K)
# ===========================================================================
# Each algorithm at its shipped-YAML operating point (the realistic sparse
# setting), + a per-algorithm top-1 floor calibrated BELOW the measured 16K value
# with margin. allow_pseudo_sparse=False so head_dim-128 takes the REAL kernel.
_ALGOS = {
    "a_shape": ({}, 0.99),
    "tri_shape": ({}, 0.99),
    "minference": ({}, 0.99),
    "flexprefill": ({"gamma": 0.9, "tau": 0.1, "block_size": 128}, 0.99),
    "xattention": (
        {"stride": 8, "norm": 1, "threshold": 0.9, "block_size": 128, "chunk_size": 2048},
        0.99,
    ),
    "flashprefill": (
        {"alpha": 0.2, "block_size": 128, "sink": 256, "window": 512, "last_n_block_full": 2},
        0.99,
    ),
}
_SEQ_LEN = 16384

# Genuinely long, multi-topic coherent prose — NOT a short paragraph repeated,
# which would make attention artificially self-similar (sparse-friendly). Real
# long-document attention varies its content across the context.
_TOPICS = [
    "The history of science traces how empirical inquiry replaced superstition.",
    "Photosynthesis converts light energy into chemical energy within chloroplasts.",
    "The French Revolution reshaped European politics across the late eighteenth century.",
    "Quantum mechanics describes nature at the scale of atoms and subatomic particles.",
    "Plate tectonics explains the slow drift of continents over geological time.",
    "The printing press accelerated the spread of literacy throughout Renaissance Europe.",
    "Neural networks learn representations by adjusting weights through gradient descent.",
    "The water cycle moves moisture between oceans, atmosphere, and land surfaces.",
    "Roman engineering produced aqueducts, roads, and structures that endured for centuries.",
    "Genetic inheritance passes traits from one generation to the next through DNA.",
]


def _long_ids(tok, n):
    text = " ".join(_TOPICS[i % len(_TOPICS)] for i in range(n // 4 + 64))
    ids = tok(text, return_tensors="pt")["input_ids"]
    while ids.shape[1] < n:
        ids = torch.cat([ids, ids], dim=1)
    return ids[:, :n].to("cuda")


def _patch(model, name, attn_kwargs):
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _Slim(model)
    kw = dict(attn_kwargs)
    kw["allow_pseudo_sparse"] = False  # force the REAL kernel on head_dim 128
    algo = SparsityAlgorithmRegistry.create(name, attn_kwargs=kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


def _run_one(name, attn_kwargs, floor, dense_arg, ids, model):
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    slim, patched = _patch(model, name, attn_kwargs)
    try:
        with torch.no_grad():
            sp = model(ids).logits.float()
    finally:
        unpatch_sparsity(slim, patched)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    assert torch.isfinite(sp).all(), f"{name}: non-finite logits at L={_SEQ_LEN}"
    sp_arg = sp[0].argmax(-1)
    agree = (dense_arg == sp_arg).float().mean().item()
    last_ok = bool((dense_arg[-1] == sp_arg[-1]).item())
    assert agree >= floor, (
        f"{name}: long-context (L={_SEQ_LEN}) top-1 agreement {agree:.4f} "
        f"below floor {floor} — block-selection quality regressed at length."
    )
    assert last_ok, f"{name}: last-token argmax disagrees with dense at L={_SEQ_LEN}"
    return agree


def _make_test(name, attn_kwargs, floor):
    def _t():
        if not real_weights_available():
            raise _Skip("real Qwen3-8B weights unavailable", _SR.NO_QWEN3_8B)
        from transformers import AutoTokenizer

        model = _real_qwen3_8b()
        tok = AutoTokenizer.from_pretrained(_REAL_W)
        ids = _long_ids(tok, _SEQ_LEN)
        with torch.no_grad():
            dense_arg = model(ids).logits.float()[0].argmax(-1)
        _run_one(name, attn_kwargs, floor, dense_arg, ids, model)
        _cap(
            f"{'flashprefill' if name == 'flashprefill' else name}_real_correctness"
            if name in ("flexprefill", "xattention", "flashprefill")
            else "qwen3_real_correctness"
        )

    _t.__name__ = f"test_long_context_top1_{name}"
    _t.__qualname__ = _t.__name__
    return _t


# Materialize one test per algorithm so each shows up by name in the runner.
for _name, (_kw, _floor) in _ALGOS.items():
    globals()[f"test_long_context_top1_{_name}"] = _make_test(_name, _kw, _floor)


if __name__ == "__main__":
    sys.exit(
        1 if run_all(globals(), f"CUDA={CUDA}, real_weights={real_weights_available()}") else 0
    )
