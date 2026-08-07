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

"""Sparse-attention algorithms.

Importing this package registers all available algorithms with
``SparsityAlgorithmRegistry`` (import side-effect). Imports are guarded so a
missing optional kernel for one algorithm does not break the others.
"""

from __future__ import annotations

import importlib
import logging

_logger = logging.getLogger(__name__)

# Register algorithms. Each import is the registration side-effect. Guarded so an
# algorithm whose optional kernel is unbuilt does not break the registry for the
# others — but the swallowed exception is LOGGED (WARNING) rather than silently
# dropped, so a genuine syntax/import bug surfaces its real cause here instead of
# a confusing "not registered" ValueError much later.
for _name in (
    "stem",
    "minference",
    "flexprefill",
    "xattention",
    "flashprefill",
    "vecattention",
    "cosa",
):
    try:
        importlib.import_module(f".{_name}", __name__)
    except Exception as _e:  # noqa: BLE001  pragma: no cover - defensive
        _logger.warning(
            "sparse algorithm %r failed to import and is NOT registered: %s: %s",
            _name,
            type(_e).__name__,
            _e,
        )
