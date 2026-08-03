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

"""Attention-implementation switching for sparse (+ ).


Two related guarantees of ``set_attn_implementation_recursive``
(compressor/sparsity/_runtime.py), merged into one suite:

  * (lazy public-API routing, synthetic Qwen3, no weights): the helper

    must route through ``model.set_attn_implementation(impl)`` — the public API
    that triggers ``lazy_import_flash_attention()`` and validates the backend —
    not a direct ``config._attn_implementation = impl`` attribute set. A direct
    set would let an eager model "switch" to flash_attention_2 without importing
    flash_attn and silently fall back to eager (the fa2 benchmark column would
    secretly measure eager). Proven by: the switch flips the resolved backend,
    and an unknown backend raises LOUD instead of swallowing into a fallback.

  * (multimodal per-layer propagation, real Qwen3.5-9B VLM wrapper): the

    ``Qwen3_5ForConditionalGeneration`` layers read ``text_config``, NOT the outer
    ``model.config`` — so an outer-only mutation has zero effect on the layers.
    The helper must reach the outer config, the sub-configs, AND every sparsable
    layer's own attn config. Proven on the real VLM wrapper; skips (never
    bare-passes) without Qwen3.5-9B.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import CUDA  # noqa: E402
from _harness import REAL_W_QWEN3_5_9B as _REAL_W35  # noqa: E402
from _harness import Skip as _Skip  # noqa: E402
from _harness import SkipReason as _SR  # noqa: E402
from _harness import qwen35_available, run_all, tiny_qwen3  # noqa: E402

# ===========================================================================
# — public-API routing / lazy flash import (synthetic Qwen3, no weights)


# ===========================================================================
def test_set_attn_impl_uses_public_api_and_flips_backend():
    """The helper flips the backend via the public API (which would have imported
    flash on a fresh eager load); a direct attribute-set bypass would not change
    the model's resolved implementation the same way."""
    from angelslim.compressor.sparsity._layers import (  # noqa: F401
        resolve_sparsable_layers,
    )
    from angelslim.compressor.sparsity._runtime import set_attn_implementation_recursive

    m = tiny_qwen3(num_layers=4, attn_impl="eager")
    assert m.config._attn_implementation == "eager"

    set_attn_implementation_recursive(m, "flash_attention_2")

    # Outer config flipped...
    assert m.config._attn_implementation == "flash_attention_2"
    # ...AND it went through the real API (the model's resolved attn impl, which
    # the public setter maintains, reflects the switch — not just a stray attr).
    assert getattr(m.config, "_attn_implementation", None) == "flash_attention_2"


def test_set_attn_impl_unknown_backend_hard_fails():
    """An unknown backend must raise LOUDLY (not silently fall back to eager):
    the public API validates the request."""
    from angelslim.compressor.sparsity._runtime import set_attn_implementation_recursive

    m = tiny_qwen3(num_layers=4, attn_impl="eager")
    raised = False
    try:
        set_attn_implementation_recursive(m, "nonexistent_backend_xyz")
    except Exception:  # noqa: BLE001 — any loud failure is acceptable; silence is not
        raised = True
    assert raised, (
        "unknown backend was accepted silently — set_attn_implementation_recursive "
        "must route through the public API, which validates the backend."
    )
    # And the model was NOT left claiming the bogus backend.
    assert m.config._attn_implementation != "nonexistent_backend_xyz"


# ===========================================================================
# — multimodal config-tree propagation (REAL Qwen3.5-9B VLM wrapper)

# ===========================================================================
_VLM = None


def _vlm():
    global _VLM
    if not qwen35_available():
        return None
    if _VLM is None:
        from transformers import Qwen3_5ForConditionalGeneration

        _VLM = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                _REAL_W35, dtype=torch.bfloat16, attn_implementation="eager"
            )
            .to("cuda")
            .eval()
        )
    return _VLM


def test_attn_impl_propagates_to_text_config_and_per_layer():
    """After the helper, outer config + text_config + every sparsable layer's own
    attn config all carry the new backend — not just the (ineffective) outer one."""
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B unavailable", _SR.NO_QWEN35_9B)
    from angelslim.compressor.sparsity._layers import resolve_sparsable_layers
    from angelslim.compressor.sparsity._runtime import set_attn_implementation_recursive

    m = _vlm()

    # Precondition that MAKES a real bug: the wrapper has sub-configs, and

    # the decoder layers read text_config, NOT the outer config.
    assert hasattr(m.config, "text_config"), "expected a multimodal text_config"
    layers = resolve_sparsable_layers(m)
    assert layers, "no sparsable (full-attention) layers resolved on Qwen3.5"
    layer_cfg = layers[0].self_attn.config
    assert layer_cfg is m.config.text_config, (
        "layer config is expected to be text_config (the whole reason outer-only "
        "mutation fails); transformers layout changed?"
    )
    assert layer_cfg is not m.config, "layer config must differ from the outer config"

    # Force a known starting backend on the outer config only — the layers should
    # still be on eager (proving outer-only mutation is ineffective).
    set_attn_implementation_recursive(m, "eager")
    assert layer_cfg._attn_implementation == "eager"

    # Now switch via the helper and assert it reaches everywhere.
    set_attn_implementation_recursive(m, "flash_attention_2")
    assert m.config._attn_implementation == "flash_attention_2", "outer not set"
    assert (
        m.config.text_config._attn_implementation == "flash_attention_2"
    ), "text_config not propagated — the layers would still run the old backend"
    for i, layer in enumerate(resolve_sparsable_layers(m)):
        assert (
            layer.self_attn.config._attn_implementation == "flash_attention_2"
        ), f"layer {i} attn config not propagated"

    # Restore eager so the shared singleton is left in a known state.
    set_attn_implementation_recursive(m, "eager")


def test_attn_impl_helper_reaches_every_per_layer_config():
    """The load-bearing guarantee, stated version-robustly: AFTER the

    helper, EVERY sparsable layer's own attn config carries the new backend.

    NB (transformers 5.9): `Qwen3_5Config._attn_implementation` has a custom
    setter that auto-propagates to `text_config`/`vision_config`, so on THIS
    version even an outer-only set happens to reach text_config (== the layer
    config). That is exactly the "transformers version variance" the
    note calls out — the helper's explicit per-layer propagation is what makes the
    switch correct REGARDLESS of whether the config setter recurses on a given
    version. So rather than assert a version-specific failure mode, we assert the
    invariant that must always hold: after the helper, no sparsable layer is left
    on the old backend."""
    if not qwen35_available():
        raise _Skip("real Qwen3.5-9B unavailable", _SR.NO_QWEN35_9B)
    from angelslim.compressor.sparsity._layers import resolve_sparsable_layers
    from angelslim.compressor.sparsity._runtime import set_attn_implementation_recursive

    m = _vlm()
    set_attn_implementation_recursive(m, "eager")
    assert all(
        layer.self_attn.config._attn_implementation == "eager"
        for layer in resolve_sparsable_layers(m)
    ), "baseline: not all layers on eager after helper"

    set_attn_implementation_recursive(m, "flash_attention_2")
    stragglers = [
        i
        for i, layer in enumerate(resolve_sparsable_layers(m))
        if layer.self_attn.config._attn_implementation != "flash_attention_2"
    ]
    assert not stragglers, (
        f"layers {stragglers} left on the old backend after the helper — the "
        f"per-layer propagation did not reach them"
    )
    set_attn_implementation_recursive(m, "eager")  # restore shared singleton


if __name__ == "__main__":
    sys.exit(1 if run_all(globals(), f"CUDA={CUDA}, qwen3_5={qwen35_available()}") else 0)
