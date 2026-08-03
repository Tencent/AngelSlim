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

"""Runtime attention-implementation switching for sparse benchmarking.

A benchmark (or any tool) that wants to compare attention backends — e.g. eager
vs flash_attention_2 — switches the model's `attn_implementation`. Doing that
correctly on Qwen3.5 multimodal needs TWO things the naive
`model.config._attn_implementation = impl` attribute-set gets wrong:

  **Use the public API so flash is lazily imported.** A direct attribute
  set bypasses `lazy_import_flash_attention()`; if the model was loaded `eager`
  and `flash_attn` was never imported, the `ALL_ATTENTION_FUNCTIONS` registry
  lookup silently falls back to eager — the fa2 column then measures eager. The
  public `model.set_attn_implementation(impl)` triggers the lazy import (and
  hard-fails with an actionable error if `flash_attn` is genuinely absent),
  instead of failing open.

  **Propagate to the multimodal config tree.** For
  `Qwen3_5ForConditionalGeneration` the outer `model.config` carries sub-configs
  `text_config` / `vision_config`, and each `Qwen3_5Attention` layer reads
  `self.config` (== `text_config`), NOT the outer config. Mutating only the outer
  config has zero effect on the layers — the eager/fa2 columns measure the same
  backend. So after the public call we defensively propagate to the sub-configs
  and to each sparsable layer's own attn config (the public API recurses on some
  transformers versions but not all; the explicit propagate makes it
  version-robust).

This is a runtime/benchmark helper; it does not patch attention or touch the
sparse algorithms — it only selects which dense backend the unpatched layers use.
"""

from __future__ import annotations

from ._layers import resolve_sparsable_layers


def set_attn_implementation_recursive(model, impl: str) -> None:
    """Set ``attn_implementation`` on the outer config + sub-configs + per-layer.

    Args:
        model: a loaded HF model (plain Qwen3 or Qwen3.5 multimodal wrapper).
        impl: the backend name, e.g. ``"eager"`` or ``"flash_attention_2"``.

    Raises whatever ``model.set_attn_implementation`` raises when the requested
    backend's package is unavailable (e.g. flash_attn not installed) — a loud
    failure, never a silent eager fallback.
    """
    # Public API first — it triggers lazy_import_flash_attention and
    # hard-fails loudly if the backend is unavailable (no silent eager fallback).
    model.set_attn_implementation(impl)

    # Defensively propagate to multimodal sub-configs (Qwen3.5
    # ForConditionalGeneration: layers read text_config, not the outer config),
    # in case set_attn_implementation did not recurse on this transformers version.
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg.text_config._attn_implementation = impl
        if hasattr(cfg, "vision_config") and cfg.vision_config is not None:
            cfg.vision_config._attn_implementation = impl

    # Final defensive: the per-layer attn config the layer actually reads.
    for layer in resolve_sparsable_layers(model):
        attn = getattr(layer, "self_attn", None)
        attn_cfg = getattr(attn, "config", None) if attn is not None else None
        if attn_cfg is not None:
            attn_cfg._attn_implementation = impl
