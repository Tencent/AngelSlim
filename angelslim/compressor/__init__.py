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

from .compressor_factory import CompressorFactory  # noqa: F401

# The concrete compressors register themselves with CompressorFactory as an
# import side-effect. We do NOT eager-import them here, so that importing a
# lightweight submodule (e.g. ``angelslim.compressor.sparsity.registry``) does
# not pull in the quant/GPTQ stack (threadpoolctl etc.). Instead the engine
# calls ``ensure_compressors_registered()`` before it touches the factory; it is
# idempotent and cheap on repeat. ``from angelslim.compressor import PTQ`` (etc.)
# still works via the PEP-562 ``__getattr__`` below.

_registered = False


def ensure_compressors_registered():
    """Import every concrete compressor so it registers with CompressorFactory.

    Idempotent. The sparse subsystem is import-guarded so a missing optional
    kernel never breaks the core compressor set; the swallowed cause is logged.
    """
    global _registered
    if _registered:
        return
    from .distill import Distill  # noqa: F401
    from .mcore_qad.runner import MCoreQAD  # noqa: F401
    from .qad import QAD  # noqa: F401
    from .qat.qat import QAT  # noqa: F401
    from .quant import PTQ  # noqa: F401

    try:
        from .sparsity import algorithms  # noqa: F401  (registers stem, etc.)
        from .sparsity.sparsity import Sparsity  # noqa: F401
    except Exception as _e:  # noqa: BLE001  pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).warning(
            "sparse subsystem failed to import (Sparsity compressor NOT available): " "%s: %s",
            type(_e).__name__,
            _e,
        )
    _registered = True


def __getattr__(name):
    # Lazy access to the concrete compressors. Importing any of these names
    # triggers full registration (so the legacy ``from angelslim.compressor
    # import PTQ`` keeps working) without forcing it at package import time.
    if name in ("Distill", "QAD", "QAT", "PTQ", "Sparsity", "MCoreQAD"):
        ensure_compressors_registered()
        import importlib

        mod_map = {
            "Distill": (".distill", "Distill"),
            "QAD": (".qad", "QAD"),
            "QAT": (".qat.qat", "QAT"),
            "PTQ": (".quant", "PTQ"),
            "Sparsity": (".sparsity.sparsity", "Sparsity"),
            "MCoreQAD": (".mcore_qad.runner", "MCoreQAD"),
        }
        mod_name, attr = mod_map[name]
        return getattr(importlib.import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
