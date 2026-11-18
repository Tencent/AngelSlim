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

"""Calibration code that runs inside vLLM workers. Import submodules explicitly:
nothing in ``angelslim`` imports this package eagerly, so transformers-only use
needs no vLLM."""

import os

__all__ = ["worker_bootstrap"]


def worker_bootstrap():
    """``collective_rpc`` callable putting this repo on a Ray worker's ``sys.path``; run it
    before the first ``apply_model``, whose callables are pickled by reference.
    A closure, so cloudpickle ships it by value (workers cannot import us yet)."""
    root = os.path.abspath(__file__)
    for _ in range(5):  # vllm_calib/__init__.py -> quant -> compressor -> angelslim -> repo
        root = os.path.dirname(root)

    def _bootstrap(*_args):
        import sys

        if root not in sys.path:
            sys.path.insert(0, root)
        return f"sys.path += {root}"

    return _bootstrap
