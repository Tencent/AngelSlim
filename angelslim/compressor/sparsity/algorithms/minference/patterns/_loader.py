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

"""Loader + fingerprint validation for MInference per-head pattern JSON.

A pattern JSON searched against one model silently misapplies if
loaded onto a model with a different ``rope_theta`` / head layout. Every pattern
carries a ``model_fingerprint``; this module validates it against the live model
config and raises :class:`IncompatibleConfigError` on any mismatch.

Schema:
    {
      "schema_version": 1,
      "model_fingerprint": {
        "model_type": "qwen3", "rope_theta": 1000000.0,
        "max_position_embeddings": 131072, "num_attention_heads": 32,
        "num_key_value_heads": 8, "hidden_size": 4096, "num_hidden_layers": 36
      },
      "best_pattern": { "<layer_idx>": { "<head_idx>": ["vertical_and_slash", v, s, 1], ... }, ... },  # noqa: E501
      "minference_revision": "a4eb395"
    }

When no ``pattern_path`` is configured the ``minference`` variant runs with the
dense per-head fallback ``("vertical_and_slash", 1000, 6096)`` (dense defaults
are used until a searched pattern JSON is supplied).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ...._base import IncompatibleConfigError

SCHEMA_VERSION = 1

# Fingerprint fields checked against ``model.config``. ``rope_theta`` is special-
# cased because it can live under ``config.rope_scaling`` on some models.
_FINGERPRINT_FIELDS = (
    "model_type",
    "rope_theta",
    "max_position_embeddings",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "num_hidden_layers",
)


def _resolve_field(model_config, field):
    actual = getattr(model_config, field, None)
    if field == "rope_theta":
        rs = getattr(model_config, "rope_scaling", None)
        if isinstance(rs, dict) and "rope_theta" in rs:
            actual = rs.get("rope_theta", actual)
    return actual


def validate_fingerprint(model_config, fingerprint: dict) -> None:
    """Raise :class:`IncompatibleConfigError` if ``fingerprint`` mismatches.

    Only fields present in ``fingerprint`` are checked (forward-compatible with
    sparser fingerprints). Numbers compare by value (1000000 == 1000000.0).
    """
    diffs = []
    for field, expected in fingerprint.items():
        if field not in _FINGERPRINT_FIELDS:
            continue
        actual = _resolve_field(model_config, field)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            mismatch = float(expected) != float(actual)
        else:
            mismatch = actual != expected
        if mismatch:
            diffs.append(f"{field}: pattern_json={expected!r}, model={actual!r}")
    if diffs:
        raise IncompatibleConfigError(
            "Pattern JSON model_fingerprint does not match this model:\n  - "
            + "\n  - ".join(diffs)
            + "\nRegenerate the pattern via the offline pattern-search tooling."
        )


def _normalize_best_pattern(raw: dict) -> dict:
    """JSON keys are strings; convert layer/head keys to ints and tuple-ize the
    per-head entries so ``reference._minference_per_head`` can index them."""
    out = {}
    for layer_key, heads in raw.items():
        layer_idx = int(layer_key)
        head_map = {}
        for head_key, entry in heads.items():
            # entry: ["vertical_and_slash", vertical, slash, ...]
            head_map[int(head_key)] = tuple(entry)
        out[layer_idx] = head_map
    return out


def load_pattern(pattern_path: Optional[str], model_config) -> Optional[dict]:
    """Load + validate a pattern JSON. Returns the normalized ``best_pattern``
    dict, or ``None`` when ``pattern_path`` is falsy (dense fallback).

    ``pattern_path`` must already be resolved to an absolute path by the caller
    (anchored on the config dir / save dir, never CWD).
    """
    if not pattern_path:
        return None
    if not os.path.isfile(pattern_path):
        raise IncompatibleConfigError(
            f"minference pattern_path does not exist: {pattern_path!r}. "
            f"Either fix the path or omit it to use dense fallback patterns."
        )
    with open(pattern_path, "r") as f:
        doc = json.load(f)

    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise IncompatibleConfigError(
            f"Pattern JSON schema_version={version!r}, expected {SCHEMA_VERSION}. "
            f"Regenerate the pattern with the current tooling."
        )

    fingerprint = doc.get("model_fingerprint")
    if not isinstance(fingerprint, dict):
        raise IncompatibleConfigError("Pattern JSON is missing a 'model_fingerprint' block.")
    validate_fingerprint(model_config, fingerprint)

    best = doc.get("best_pattern")
    if not isinstance(best, dict):
        raise IncompatibleConfigError("Pattern JSON is missing a 'best_pattern' block.")
    return _normalize_best_pattern(best)
