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

"""GLM-5 layer-selection constants (torch-free).

Kept as a standalone module so unit tests can import it on any CPU-only
machine (no torch, no GPU, no model weights required).  ``glm5.py``
re-exports these names for convenience.

Any layer that is NOT in the leaf whitelist below, or whose fully-
qualified name contains one of the hard-skip substrings, stays bf16.
Users can extend the skip set at YAML level via
``compression.quantization.ignore_layers`` -- those entries are simply
appended by ``GLM5.get_observer_layers`` (see glm5.py).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Observer whitelist: ``leaf_name`` (last dotted segment of the FQN) must
# match one of these to be considered for INT8 quantization by
# ``GLM5.get_observer_layers``.
# ---------------------------------------------------------------------------
_QUANTIZABLE_LEAF_NAMES = (
    # MLA attention linears (kv_b_proj included by default; opt out via YAML)
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    # DSA indexer linears (opt out via YAML if a recipe forbids them)
    "wq_b",
    "wk",
    # Dense MLP / MoE experts / shared experts
    "gate_proj",
    "up_proj",
    "down_proj",
)

# ---------------------------------------------------------------------------
# Hard-skip substrings: ALWAYS excluded, regardless of YAML.  Quantizing
# any of these is either known to break the model outright (router /
# lm_head / embed) or is meaningless (tiny helper norms / MTP fuse).
#
# NOTE: MTP-block sub-modules are covered by the substring match too --
# their FQNs contain ``.indexer.k_norm`` / ``mlp.gate.`` / ``eh_proj`` etc.
# ---------------------------------------------------------------------------
_FORCED_SKIP_SUBSTRINGS = (
    ".indexer.k_norm",   # DSA per-head RMSNorm
    "mlp.gate.",          # MoE router (trailing dot avoids matching mlp.gate_proj)
    "lm_head",
    "embed_tokens",
    "eh_proj",            # GLM-5 MTP embedding-to-hidden fuse (model.layers.<N>.eh_proj)
)
