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

"""Sparse hard-fail / runtime guards (merged).

One suite for the guards that must LOUDLY refuse an unsupported or unsafe
configuration rather than silently produce wrong results. Folds together four
former per-topic files:

  * distributed: multi-node + tensor-parallel detection.

  * runtime: CUDA-graph / torch.compile / vLLM-serving refusal.

  * / / — padding-mask rejection, cached-2nd-turn skip,

    output_attentions refusal, sliding-window refusal, strict unpatch ownership,
    Stem cache_position threading.
  * quant-collision: sparse refuses any quantization (parse-time AND

    convert-time), including reloaded quantized checkpoints and multi-method.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import run_all  # noqa: E402
from _harness import CUDA, HEAD_DIM  # noqa: E402
from _harness import FakeSlim as _FakeSlim  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import real_qwen3_8b as _real_qwen3_8b  # noqa: E402
from _harness import tiny_qwen3 as _tiny_qwen3  # noqa: E402

# ===========================================================================
# Distributed hard-fails (multi-node / TP)

# ===========================================================================


# ===========================================================================
# — multi-node / multi-rank hard-fail


# ===========================================================================
def test_multi_nodes_rejected():
    """WORLD_SIZE>1 → apply_sparsity_patch hard-fails (canonical name)."""

    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3(num_layers=2)
    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    old = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = "4"
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected multi-node IncompatibleConfigError")
    except IncompatibleConfigError as e:
        assert "single-rank" in str(e) or "WORLD_SIZE" in str(e)
    finally:
        if old is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = old


def test_detect_world_size_env_vars():
    """detect_world_size reads WORLD_SIZE / PMI_SIZE / OMPI_COMM_WORLD_SIZE."""
    from angelslim.compressor.sparsity._distributed import detect_world_size

    assert detect_world_size() == 1  # clean
    for var in ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE"):
        old = os.environ.get(var)
        os.environ[var] = "8"
        try:
            assert detect_world_size() == 8, var
        finally:
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old


# ===========================================================================
# — TP detector v2 (real TP trips; device_map sharding does NOT)


# ===========================================================================
def test_detect_tp_vectors():
    """detect_tp must flag REAL tensor parallelism (a layer's weights

    sharded across ranks → model._tp_size>1) but must NOT flag accelerate
    device_map layer sharding (whole layers placed sequentially in one process —
    fully compatible with sparse). The old code counted device_map devices and
    false-positived on every ≥2-GPU device_map=auto load (flagship Stem YAML
    hard-failed on any multi-GPU box). The dead DeepSpeed/`vllm-in-sys.modules`
    vectors were removed."""

    from angelslim.compressor.sparsity._distributed import detect_tp

    class _M:
        def __init__(self, dm=None, tp_size=None):
            self.hf_device_map = dm
            self._tp_size = tp_size

    # device_map layer sharding across 2 devices is NOT TP — must pass (return 1).
    assert detect_tp(_M(dm={"model.layers.0": 0, "model.layers.1": 1})) == 1
    assert detect_tp(_M(dm={"a": 0, "b": 0})) == 1
    assert detect_tp(_M(dm=None)) == 1

    # Real tensor parallelism (transformers sets model._tp_size) MUST trip.
    assert detect_tp(_M(tp_size=4)) == 4
    assert detect_tp(_M(dm={"model.layers.0": 0}, tp_size=2)) == 2  # tp wins

    # A bare 'vllm' import must NOT trip detect_tp anymore — it fired on a

    # mere import). The real vLLM-serving signal lives in
    # detect_cuda_graph_or_compile via VLLM_WORKER_MULTIPROC_METHOD.
    injected = "vllm" not in sys.modules
    if injected:
        sys.modules["vllm"] = type(sys)("vllm")
    try:
        assert detect_tp(_M(dm=None)) == 1
    finally:
        if injected:
            del sys.modules["vllm"]


# ===========================================================================
# Runtime guards (CUDA-graph / compile / vLLM serving)

# ===========================================================================


def test_detect_cuda_graph_or_compile_guard():
    """Detector: trips on config._compile_static_kv_cache, clean otherwise."""
    from angelslim.compressor.sparsity._distributed import detect_cuda_graph_or_compile

    class _Cfg:
        pass

    class _M:
        def __init__(self):
            self.hf_device_map = None
            self.config = _Cfg()

    clean = _M()
    assert detect_cuda_graph_or_compile(clean) == ""  # no guard tripped

    tripped = _M()
    tripped.config._compile_static_kv_cache = True
    reason = detect_cuda_graph_or_compile(tripped)
    assert reason and "CUDA graph" in reason


def test_vllm_worker_env_trips_runtime_guard():
    """VLLM_WORKER_MULTIPROC_METHOD set → detector reports the vLLM-serving
    reason (vLLM builds its own attention; the patched HF forward is never
    called, so sparse must refuse rather than become a silent no-op)."""
    from angelslim.compressor.sparsity._distributed import detect_cuda_graph_or_compile

    class _M:
        def __init__(self):
            self.hf_device_map = None
            self.config = type("C", (), {})()

    old = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        reason = detect_cuda_graph_or_compile(_M())
        assert reason and "vLLM" in reason
    finally:
        if old is None:
            os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        else:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = old


def test_apply_patch_runtime_guard_blocks_cuda_graph():
    """End-to-end: a CUDA-graph-flagged model is refused by apply_sparsity_patch."""
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3(num_layers=2)
    m.config._compile_static_kv_cache = True
    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected runtime-guard IncompatibleConfigError")
    except IncompatibleConfigError as e:
        assert "CUDA graph" in str(e) or "static-KV" in str(e)
    finally:
        m.config._compile_static_kv_cache = False


# ===========================================================================
# P1 guards (padding / cached-turn / output_attentions / unpatch)
# ===========================================================================


def _patch_real(model, variant, attn_kwargs):
    import warnings

    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    slim = _FakeSlim(model)
    algo = SparsityAlgorithmRegistry.create(variant, attn_kwargs=dict(attn_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    return slim, patched


# ===========================================================================
# — padded prefill batch is rejected (supports_padding_mask=False)


# ===========================================================================
def test_minference_rejects_padded_prefill_batch():
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B unavailable")
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    torch.manual_seed(0)
    L = 256
    ids = torch.randint(0, 151000, (1, L), device="cuda")
    # A 4-D causal mask whose last query row has -inf over some "padding" keys.
    mask = torch.zeros((1, 1, L, L), device="cuda", dtype=torch.float32)
    causal = torch.triu(torch.ones(L, L, device="cuda", dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(causal[None, None], float("-inf"))
    mask[:, :, :, : L // 4] = float("-inf")  # pretend the first quarter are PAD

    slim, patched = _patch_real(m, "a_shape", {"n_init": 16, "n_local": 64})
    try:
        raised = False
        try:
            with torch.no_grad():
                m(input_ids=ids, attention_mask=mask)
        except ValueError as e:
            raised = True
            assert "padding" in str(e).lower() or "unpadded" in str(e).lower()
        assert raised, (
            "padded prefill batch was NOT rejected — minference "
            "(supports_padding_mask=False) silently attended to padding "
        )
    finally:
        unpatch_sparsity(slim, patched)


def test_unpadded_prefill_still_runs():
    """Control: with NO padding (mask=None) the patched forward runs fine."""
    m = _real_qwen3_8b()
    if m is None:
        raise _Skip("real Qwen3-8B unavailable")
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, 256), device="cuda")
    slim, patched = _patch_real(m, "a_shape", {"n_init": 16, "n_local": 64})
    try:
        with torch.no_grad():
            out = m(ids).logits  # no attention_mask → no padding → OK
        assert torch.isfinite(out).all()
    finally:
        unpatch_sparsity(slim, patched)


# ===========================================================================
# — assert_no_padding_mask: reject genuine padding, ACCEPT no-padding


# masks (2D all-ones / 4D causal-only) so legitimate fa2/batched masks are not
# falsely rejected. Covers all dtype conventions transformers emits: 2D int/bool
# keep-masks, 4D float additive masks (-inf AND finfo.min) and 4D bool masks —
# a guard that only tested ~isfinite() would silently miss finfo.min/bool.
# ===========================================================================
def test_assert_no_padding_mask_mask_conventions():
    from angelslim.compressor.sparsity.algorithms._forward_templates._common import (
        assert_no_padding_mask,
    )

    K = 16
    # None → OK
    assert_no_padding_mask(None, K)

    # 2D all-ones keep-mask (the common unpadded fa2 case) → OK
    assert_no_padding_mask(torch.ones(2, K, dtype=torch.long), K)
    assert_no_padding_mask(torch.ones(2, K, dtype=torch.bool), K)

    # 2D with a real PAD (a 0 / False in the kept span) → hard-fail
    pad_int = torch.ones(2, K, dtype=torch.long)
    pad_int[1, -3:] = 0
    _expect_pad_error(lambda: assert_no_padding_mask(pad_int, K))
    pad_bool = torch.ones(2, K, dtype=torch.bool)
    pad_bool[0, :2] = False
    _expect_pad_error(lambda: assert_no_padding_mask(pad_bool, K))

    # 4D additive mask, no padding (only causal -inf above the diagonal in
    # earlier rows; the LAST query row is all-finite over valid keys) → OK
    add = torch.zeros(1, 1, K, K)
    causal = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    add = add.masked_fill(causal[None, None], float("-inf"))
    assert_no_padding_mask(add, K)

    # 4D additive mask WITH padding (-inf in the last query row over valid keys)
    add_pad = add.clone()
    add_pad[:, :, -1, :4] = float("-inf")
    _expect_pad_error(lambda: assert_no_padding_mask(add_pad, K))

    # 4D FLOAT additive mask using finfo.min (NOT -inf) for masked positions —
    # this is what transformers' eager/sdpa mask machinery actually emits (a
    # finite large-negative, so torch.isfinite() is True). A guard testing
    # ~isfinite() would silently MISS this and let padding through. Padding in
    # the kept span must still hard-fail.
    for dt in (torch.float32, torch.bfloat16):
        fmin = torch.finfo(dt).min
        add_min = torch.zeros(1, 1, K, K, dtype=dt)
        add_min = add_min.masked_fill(causal[None, None], fmin)
        assert_no_padding_mask(add_min, K)  # causal-only → OK
        add_min_pad = add_min.clone()
        add_min_pad[:, :, -1, :4] = fmin  # pad on attendable keys
        _expect_pad_error(lambda m=add_min_pad: assert_no_padding_mask(m, K))

    # 4D BOOL keep-mask (sdpa convention: True=keep, False=masked). isfinite()
    # is meaningless on bool, so the dtype branch must use ~mask.
    keep = ~causal  # lower-tri incl. diagonal
    assert_no_padding_mask(keep[None, None], K)  # causal-only → OK
    keep_pad = keep.clone()
    keep_pad[-1, :4] = False  # pad on attendable keys
    _expect_pad_error(lambda: assert_no_padding_mask(keep_pad[None, None], K))


def _expect_pad_error(fn):
    try:
        fn()
    except ValueError as e:
        assert "padding" in str(e).lower() or "unpadded" in str(e).lower()
        return
    raise AssertionError("expected a padding ValueError, none raised")


# ===========================================================================
# Cached / chunked prefill (k_len > q_len) routes to dense, NOT sparse
# ===========================================================================
def test_cached_second_turn_skips_sparse_prefill():
    """A multi-token 2nd turn (q_len>1 with a KV-cache prefix, so k_len>q_len)
    must NOT enter the sparse prefill path — the algorithms assume query i aligns
    to key i (no bottom-right offset), so a mis-anchored sparse run would silently
    corrupt the result. The forward template gates sparse on k_len==q_len; the
    cached case falls to the model's configured attention instead.

    We bind a SENTINEL prefill_fn that raises if invoked, build the qwen3 template
    directly, and feed it (a) a true first-fill (must call prefill) and (b) a
    cached turn (must NOT call prefill)."""
    if not CUDA:
        raise _Skip("CUDA unavailable")
    import torch
    from transformers import DynamicCache

    from angelslim.compressor.sparsity.algorithms._forward_templates.qwen3 import (
        build_qwen3_forward,
    )

    called = {"n": 0}

    def _sentinel_prefill(self, q, k, v):
        called["n"] += 1
        # return a correctly-shaped zero so the first-fill path completes
        return torch.zeros_like(q)

    m = _tiny_qwen3(num_layers=1)
    attn = m.model.layers[0].self_attn
    fwd = build_qwen3_forward(_sentinel_prefill, supports_padding_mask=False)
    bound = fwd.__get__(attn, type(attn))

    H = m.config.hidden_size
    dev = next(m.parameters()).device
    dt = next(m.parameters()).dtype
    rope = m.model.rotary_emb

    # (a) true first-fill: q_len == k_len == 8 → sparse path runs.
    hs = torch.randn(1, 8, H, device=dev, dtype=dt)
    pos = torch.arange(8, device=dev).unsqueeze(0)
    pe = rope(hs, pos)
    with torch.no_grad():
        bound(hs, pe, attention_mask=None, past_key_values=None)
    assert called["n"] == 1, "true first-fill should invoke sparse prefill"

    # (b) cached 2nd turn: prefill a cache of 8, then feed a 4-token chunk so
    # k_len=12 > q_len=4 → must NOT invoke sparse prefill.
    cache = DynamicCache()
    hs0 = torch.randn(1, 8, H, device=dev, dtype=dt)
    pe0 = rope(hs0, torch.arange(8, device=dev).unsqueeze(0))
    with torch.no_grad():
        bound(
            hs0,
            pe0,
            attention_mask=None,
            past_key_values=cache,
            cache_position=torch.arange(8, device=dev),
        )
    n_after_fill = called["n"]
    hs1 = torch.randn(1, 4, H, device=dev, dtype=dt)
    pe1 = rope(hs1, torch.arange(8, 12, device=dev).unsqueeze(0))
    with torch.no_grad():
        bound(
            hs1,
            pe1,
            attention_mask=None,
            past_key_values=cache,
            cache_position=torch.arange(8, 12, device=dev),
        )
    assert (
        called["n"] == n_after_fill
    ), "cached chunked prefill (k_len>q_len) wrongly entered the sparse path"


# ===========================================================================
# output_attentions=True is refused on the sparse prefill path
# ===========================================================================
def test_sparse_prefill_rejects_output_attentions():
    """A sparse prefill cannot return attention weights (its kernels never form
    the full attention matrix — it sets attn_weights=None). Requesting
    output_attentions=True must HARD-FAIL rather than silently return None.

    Weightless: build the qwen3 template with a sentinel prefill and feed a true
    first-fill with output_attentions=True → ValueError; without it → runs. Also
    checks the decode path (k_len>q_len) does NOT raise (it can return weights)."""
    if not CUDA:
        raise _Skip("CUDA unavailable")
    import torch

    from angelslim.compressor.sparsity.algorithms._forward_templates.qwen3 import (
        build_qwen3_forward,
    )

    def _sentinel_prefill(self, q, k, v):
        return torch.zeros_like(q)

    m = _tiny_qwen3(num_layers=1)
    attn = m.model.layers[0].self_attn
    fwd = build_qwen3_forward(_sentinel_prefill, supports_padding_mask=False)
    bound = fwd.__get__(attn, type(attn))

    H = m.config.hidden_size
    dev = next(m.parameters()).device
    dt = next(m.parameters()).dtype
    rope = m.model.rotary_emb

    hs = torch.randn(1, 8, H, device=dev, dtype=dt)
    pos = torch.arange(8, device=dev).unsqueeze(0)
    pe = rope(hs, pos)

    # (a) first-fill + output_attentions=True → hard-fail.
    raised = False
    try:
        with torch.no_grad():
            bound(hs, pe, attention_mask=None, past_key_values=None, output_attentions=True)
    except ValueError as e:
        raised = True
        assert "output_attentions" in str(e)
    assert raised, (
        "sparse prefill silently accepted output_attentions=True (returns None "
        "instead of the requested weights)"
    )

    # (b) control: same first-fill WITHOUT output_attentions runs fine.
    with torch.no_grad():
        bound(hs, pe, attention_mask=None, past_key_values=None)


# ===========================================================================
# Sliding-window layers are refused (sparse prefill ignores the window)
# ===========================================================================
def test_sliding_window_layers_rejected():
    """A model with sliding-window attention enabled must be refused: the sparse
    prefill carries no window and would attend outside it. The patcher's
    _guard_sliding_window raises IncompatibleConfigError."""
    import warnings

    from transformers import Qwen3Config, Qwen3ForCausalLM

    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        use_sliding_window=True,
        sliding_window=64,
        max_window_layers=2,
        attn_implementation="eager",
    )
    m = Qwen3ForCausalLM(cfg).eval()
    if CUDA:
        m = m.to("cuda", dtype=torch.bfloat16)

    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create("a_shape", attn_kwargs={"n_init": 16, "n_local": 64})
    raised = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            apply_sparsity_patch(slim, algo)
    except IncompatibleConfigError as e:
        raised = True
        assert "sliding" in str(e).lower()
    assert raised, "sliding-window model was NOT rejected by the patch guard"
    # the guard must run BEFORE any layer is patched — model left pristine.
    assert not slim.attn_forward_labels(), "patch leaked despite the guard raising"


# ===========================================================================
# — unpatch_sparsity RAISES on patch-ownership mismatch


# ===========================================================================
def test_unpatch_raises_on_ownership_mismatch():
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    class _Attn:
        def forward(self):
            return "orig"

    slim = _FakeSlim(model=None)
    attn = _Attn()
    # Sparse is NOT what's on the slot — simulate something else owning it.
    slim.push_attn_forward("qwen_fp8", attn, lambda: "fp8")
    try:
        unpatch_sparsity(slim, [attn])  # expects label "sparse" → mismatch
        raise AssertionError("unpatch_sparsity must RAISE on ownership mismatch ")

    except RuntimeError as e:
        assert "violation" in str(e).lower() or "mismatch" in str(e).lower()
    # The non-sparse patch is left intact (we did not steal it).
    assert slim.attn_forward_labels() == {"qwen_fp8"}


def test_unpatch_clean_and_benign_no_patch_succeed():
    """A matching unpatch succeeds; a second (no-patch) unpatch is benign."""
    from angelslim.compressor.sparsity.patcher import unpatch_sparsity

    class _Attn:
        def forward(self):
            return "orig"

    slim = _FakeSlim(model=None)
    attn = _Attn()
    slim.push_attn_forward("sparse", attn, lambda: "S")
    unpatch_sparsity(slim, [attn])  # matching → restores, no raise
    assert attn.forward() == "orig"
    # Second call: nothing on the slot → benign, must NOT raise.
    unpatch_sparsity(slim, [attn])


# ===========================================================================
# — Stem forward threads cache_position into past_key_values.update


# ===========================================================================
def test_stem_forward_passes_cache_position_to_update():
    """The Stem forward must pass cache_kwargs (incl. cache_position) to
    Cache.update — a bare update(k,v,layer_idx) writes wrong slots on
    StaticCache / quantized / sliding caches. We spy on a fake cache and assert
    cache_position arrives. Weight-independent (a tiny model is enough)."""
    if not CUDA:
        raise _Skip("CUDA required for the Stem kernel path")
    import warnings

    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register stem
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3(num_layers=2, attn_impl="eager")
    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create(
        "stem",
        attn_kwargs={
            "backend": "torch",
            "allow_pseudo_sparse": True,
            "layer_keep_ratios": [1.0, 1.0],
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)

    seen = {}

    class _SpyCache:
        def update(self, k, v, layer_idx, cache_kwargs=None):
            seen["called"] = True
            seen["cache_kwargs"] = cache_kwargs
            return k, v  # identity → forward proceeds

    try:
        attn = m.model.layers[0].self_attn
        B, L = 1, 32
        hidden = torch.randn(
            B, L, m.config.hidden_size, device="cuda", dtype=next(m.parameters()).dtype
        )
        cos = torch.ones(B, L, m.config.head_dim, device="cuda", dtype=hidden.dtype)
        sin = torch.zeros_like(cos)
        pos = torch.arange(L, device="cuda")
        with torch.no_grad():
            attn.forward(
                hidden_states=hidden,
                position_embeddings=(cos, sin),
                attention_mask=None,
                past_key_values=_SpyCache(),
                cache_position=pos,
            )
    finally:
        unpatch_sparsity(slim, patched)

    assert seen.get("called"), "Stem forward never called cache.update"
    ck = seen.get("cache_kwargs")
    assert isinstance(ck, dict), "Stem forward passed no cache_kwargs to update regression)"

    assert (
        "cache_position" in ck and ck["cache_position"] is not None
    ), f"Stem forward dropped cache_position: cache_kwargs={ck!r} "


# ===========================================================================
# Quant-collision (the no-quantization rule sparse refuses quantization)

# ===========================================================================


def _tiny_qwen3_2l(num_layers=2):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=2048,
        attn_implementation="eager",
    )
    m = Qwen3ForCausalLM(cfg)
    if CUDA:
        m = m.to("cuda", dtype=torch.bfloat16)
    return m.eval()


# A module whose *class name* matches a real AngelSlim quantized module, so the
# class-name detector treats it as quantized — without dragging in the real
# QDQModule constructor (which needs valid scales / a specific algo / GPU).
class QDQModule(nn.Module):  # noqa: N801  (deliberately mirrors the real name)
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


# ===========================================================================
# _is_quant_module — class-name + fragment detection
# ===========================================================================
def test_is_quant_module_detects_known_and_fragments():
    from angelslim.compressor.sparsity.patcher import _is_quant_module

    # Exact known names.
    assert _is_quant_module(QDQModule(nn.Identity()))  # matches frozenset + "QDQ"

    # Fragment matches (a future / renamed quantizer module).
    class FooQuantLinear(nn.Module):  # noqa: N801
        pass

    class BarWQLinearGEMM(nn.Module):  # noqa: N801
        pass

    assert _is_quant_module(FooQuantLinear())
    assert _is_quant_module(BarWQLinearGEMM())

    # Plain modules are NOT quant.
    assert not _is_quant_module(nn.Linear(4, 4))
    assert not _is_quant_module(nn.Identity())


# ===========================================================================
# convert-time guard — quantized module in the tree blocks sparse
# ===========================================================================
def test_apply_sparsity_refuses_quantized_module():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401  register
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3_2l(num_layers=2)
    # Swap one attention projection for a (name-matching) quantized module.
    attn0 = m.model.layers[0].self_attn
    attn0.q_proj = QDQModule(attn0.q_proj)

    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected quantization-collision IncompatibleConfigError")
    except IncompatibleConfigError as e:
        msg = str(e)
        assert "quantiz" in msg.lower(), msg
        assert "QDQModule" in msg, msg
    # Nothing should have been patched (guard runs before any push).
    assert not slim.attn_forward_labels()


# ===========================================================================
# convert-time guard — slim_model.quantized flag blocks sparse
# ===========================================================================
def test_apply_sparsity_refuses_quantized_flag():
    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3_2l(num_layers=2)
    slim = _FakeSlim(m)
    slim.quantized = True  # the wrapper's own "already quantized" flag

    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected quantized-flag IncompatibleConfigError")
    except IncompatibleConfigError as e:
        assert "quantiz" in str(e).lower(), str(e)


# ===========================================================================
# convert-time guard — a clean (unquantized) model still patches fine
# ===========================================================================
def test_apply_sparsity_allows_unquantized_model():
    import warnings

    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity.patcher import (
        apply_sparsity_patch,
        unpatch_sparsity,
    )
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    if not CUDA:
        raise _Skip("Stem backend needs CUDA; the guard itself is covered above")

    m = _tiny_qwen3_2l(num_layers=2)
    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create(
        "stem",
        attn_kwargs={"backend": "torch", "block_size": 128, "layer_keep_ratios": [1.0, 1.0]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patched = apply_sparsity_patch(slim, algo)
    try:
        assert len(patched) == 2, "clean model should patch both layers"
    finally:
        unpatch_sparsity(slim, patched)


# ===========================================================================
# parse-time guard — Sparsity + PTQ/QAT in one pipeline is rejected
# ===========================================================================
def test_parser_rejects_sparsity_plus_quantization():
    import yaml as _yaml

    from angelslim.utils.config_parser import SlimConfigParser

    for quant in ("PTQ", "QAT"):
        doc = {
            "model": {"name": "Qwen", "model_path": "x"},
            "compression": {
                "name": [quant, "Sparsity"],
                "sparsity": {"name": "stem", "attn_kwargs": {"backend": "torch"}},
                "quantization": {"name": "fp8_static"},
            },
        }
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.yaml")
            with open(p, "w") as f:
                _yaml.safe_dump(doc, f)
            try:
                SlimConfigParser().parse(p)
                raise AssertionError(f"expected ValueError for [{quant}, Sparsity] pipeline")
            except ValueError as e:
                msg = str(e)
                assert "Sparsity cannot be combined" in msg, msg
                assert quant in msg, msg


# ===========================================================================
# parse-time guard — Sparsity ALONE still parses fine (no false positive)
# ===========================================================================
def test_parser_allows_sparsity_alone():
    import yaml as _yaml

    from angelslim.utils.config_parser import SlimConfigParser

    doc = {
        "model": {"name": "Qwen", "model_path": "x"},
        "compression": {
            "name": "Sparsity",
            "sparsity": {"name": "a_shape", "attn_kwargs": {"n_init": 128}},
        },
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        with open(p, "w") as f:
            _yaml.safe_dump(doc, f)
        cfg = SlimConfigParser().parse(p)
    assert "Sparsity" in cfg.compression_config.name
    assert cfg.compression_config.sparsity.name == "a_shape"


# ===========================================================================
# — the fp8/legacy-attn guard is LIVE (was dead: keyed on a flag nothing set)


# ===========================================================================
def test_apply_sparsity_refuses_legacy_attn_patch():
    """A populated ``_original_attn_forwards`` (the real signal qwen.py /
    hunyuan_v3_moe leave after patching attn.forward for fp8-attn / KV
    observers) must block sparse. The old guard required a ``_fp8_attn_patched``
    flag that nothing ever set, so it could never fire — review finding."""

    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3_2l(num_layers=2)
    slim = _FakeSlim(m)
    # Mirror what qwen.py::patch_fp8_attention leaves behind: an original
    # forward captured in the legacy dict (no flag, no LIFO label).
    attn0 = m.model.layers[0].self_attn
    slim._original_attn_forwards["model.layers.0.self_attn"] = attn0.forward
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected legacy-attn-patch collision error")
    except IncompatibleConfigError as e:
        assert "_original_attn_forwards" in str(e) or "legacy" in str(e).lower()


# ===========================================================================
# — a reloaded quantized checkpoint (config.quantization_config) is refused


# ===========================================================================
def test_apply_sparsity_refuses_reloaded_quantized_checkpoint():
    """A checkpoint saved quantized (GPTQ/AWQ/compressed-tensors/fp8/bnb) and
    reloaded carries ``config.quantization_config`` but its modules may be plain
    nn.Linear look-alikes the name-walk can't catch — the config flag is the
    only reliable signal (review finding."""

    import angelslim.compressor.sparsity.algorithms  # noqa: F401
    from angelslim.compressor.sparsity._base import IncompatibleConfigError
    from angelslim.compressor.sparsity.patcher import apply_sparsity_patch
    from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry

    m = _tiny_qwen3_2l(num_layers=2)
    # Simulate a reloaded quantized checkpoint: HF stores the method here.
    m.config.quantization_config = {"quant_method": "gptq", "bits": 4}
    slim = _FakeSlim(m)
    algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={"backend": "torch"})
    try:
        apply_sparsity_patch(slim, algo)
        raise AssertionError("expected reloaded-quantized-checkpoint refusal")
    except IncompatibleConfigError as e:
        assert "quantization_config" in str(e) or "gptq" in str(e).lower()


# ===========================================================================
# — parse guard derives quant set from the enum (PTQWeightOnly) + refuses


#      Sparsity in ANY multi-method pipeline (incl. Distill)
# ===========================================================================
def test_parser_rejects_sparsity_plus_ptq_weight_only():
    """[Sparsity, PTQWeightOnly] slipped past the old hard-coded {PTQ,QAT,QAD}
    set — the runtime guard can't back it up (sparse converts before quant), so
    it must be caught at parse time (review finding."""

    import yaml as _yaml

    from angelslim.utils.config_parser import SlimConfigParser

    doc = {
        "model": {"name": "Qwen", "model_path": "x"},
        "compression": {
            "name": ["Sparsity", "PTQWeightOnly"],
            "sparsity": {"name": "stem", "attn_kwargs": {"backend": "torch"}},
            "quantization": {"name": "int4_gptq"},
        },
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        with open(p, "w") as f:
            _yaml.safe_dump(doc, f)
        try:
            SlimConfigParser().parse(p)
            raise AssertionError("expected ValueError for [Sparsity, PTQWeightOnly]")
        except ValueError as e:
            assert "PTQWeightOnly" in str(e), str(e)


def test_parser_rejects_sparsity_plus_distill():
    """Sparsity is standalone in P0 — it must not combine with ANY other method,
    including non-quant ones like Distill refuse multi-method pipelines)."""

    import yaml as _yaml

    from angelslim.utils.config_parser import SlimConfigParser

    doc = {
        "model": {"name": "Qwen", "model_path": "x"},
        "compression": {
            "name": ["Sparsity", "Distill"],
            "sparsity": {"name": "stem", "attn_kwargs": {"backend": "torch"}},
        },
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        with open(p, "w") as f:
            _yaml.safe_dump(doc, f)
        try:
            SlimConfigParser().parse(p)
            raise AssertionError("expected ValueError for [Sparsity, Distill]")
        except ValueError as e:
            assert "standalone" in str(e).lower() or "Distill" in str(e), str(e)


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(1 if run_all(globals(), f"CUDA={CUDA}") else 0)
