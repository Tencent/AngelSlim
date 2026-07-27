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

"""Global quant on/off switch -- the trick that makes QAD need no separate teacher.

Because weights are frozen, the original (BF16) model == the model with all
quantizers bypassed. So the teacher is just a quant-off forward of the same model.
"""

from __future__ import annotations

from contextlib import contextmanager

from angelslim.compressor.mcore_qad.quant.grouped_quant import GROUPED_WEIGHT_QUANT
from angelslim.compressor.mcore_qad.quant.quantizer import Quantizer

#: per-linear/act Quantizer + every grouped-expert weight quantizer (auto-extends).
_QUANT_TYPES = (Quantizer, *GROUPED_WEIGHT_QUANT.values())


def iter_quantizers(model):
    for m in model.modules():
        if isinstance(m, _QUANT_TYPES):
            yield m


@contextmanager
def quant_disabled(model):
    """Temporarily disable every Quantizer in ``model`` (teacher path)."""
    saved = [(q, q.enabled) for q in iter_quantizers(model)]
    try:
        for q, _ in saved:
            q.enabled = False
        yield
    finally:
        for q, prev in saved:
            q.enabled = prev
