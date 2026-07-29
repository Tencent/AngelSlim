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

"""Sparsity compressor — the CompressorFactory entry point for sparse attention.

The CompressorFactory entry point with ``compress_type = "Sparsity"``. Mirrors PTQ/QAT/QAD/
Distill: constructed by ``CompressorFactory.create`` with ``(model, slim_config)``.

Lifecycle (driven by Engine):
  * ``run(dataloader=None)`` — no-op (no calibration needed).
  * ``convert()``           — applies the sparse patch (idempotent).
  * ``save(save_path)``     — re-emits HF weights + tokenizer, copies pattern
                              JSON. config-only on top of HF save.
"""

from __future__ import annotations

import os
import shutil

from ...utils import print_info
from ..compressor_factory import CompressorFactory
from .patcher import apply_sparsity_patch, unpatch_sparsity
from .registry import SparsityAlgorithmRegistry


@CompressorFactory.register("Sparsity")
class Sparsity:
    """Sparse-attention compressor."""

    def __init__(self, model, slim_config=None):
        # ``model`` is the AngelSlim BaseLLMModel wrapper; .model is the HF model.
        self.slim_model = model
        self.config = slim_config or {}
        compress_cfg = self.config.get("compress_config")
        # The sparsity sub-config: a SparsityConfig dataclass or a dict.
        sparsity_cfg = getattr(compress_cfg, "sparsity", None)
        if sparsity_cfg is None and isinstance(compress_cfg, dict):
            sparsity_cfg = compress_cfg.get("sparsity")
        if sparsity_cfg is None:
            raise ValueError("Sparsity compressor requires compression.sparsity in the config.")
        self.sparsity_cfg = sparsity_cfg

        name = self._cfg_get(sparsity_cfg, "name")
        attn_kwargs = dict(self._cfg_get(sparsity_cfg, "attn_kwargs", {}) or {})
        # allow_pseudo_sparse lives alongside attn_kwargs for the algorithm.
        allow_pseudo = self._cfg_get(sparsity_cfg, "allow_pseudo_sparse", False)
        attn_kwargs.setdefault("allow_pseudo_sparse", allow_pseudo)

        # Resolve pattern_path to an absolute path at load. A reloaded
        # checkpoint stores the pattern under <ckpt>/sparse_patterns/<name> and
        # the config carries a path relative to the checkpoint dir; anchor on
        # that dir (passed via global_config.model_path / save_path) when given,
        # else on CWD. An already-absolute path passes through unchanged. This
        # removes the off-CWD reload break (the path used to be read raw).
        raw_pp = attn_kwargs.get("pattern_path", None)
        self._pattern_path = self._resolve_pattern_path(raw_pp)
        if self._pattern_path is not None:
            attn_kwargs["pattern_path"] = self._pattern_path

        self.algo = SparsityAlgorithmRegistry.create(name, attn_kwargs=attn_kwargs)
        self._patched_modules = []
        self._patched = False

    def _resolve_pattern_path(self, raw):
        """Anchor a (possibly relative) ``pattern_path`` to an absolute path.

        Search order for a relative value: the checkpoint/model dir hint from
        ``global_config`` (where ``save()`` puts ``sparse_patterns/``), then the
        directory of the YAML config (``global_config.config_dir``, so a path
        written relative to the config file resolves regardless of the process
        CWD), then CWD. Returns ``None`` for a falsy input, the absolute path
        otherwise."""
        if not raw:
            return None
        if os.path.isabs(raw):
            return raw
        gc = self.config.get("global_config") if isinstance(self.config, dict) else None
        hint_dir = None
        for attr in ("model_path", "save_path"):
            v = getattr(gc, attr, None) if gc is not None else None
            if v and os.path.isdir(v):
                hint_dir = v
                break
        config_dir = getattr(gc, "config_dir", None) if gc is not None else None
        search = (
            ([hint_dir] if hint_dir else []) + ([config_dir] if config_dir else []) + [os.getcwd()]
        )
        for base in search:
            cand = os.path.join(base, raw)
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        # Not found anywhere yet; return the CWD-anchored absolute path so the
        # loader raises a clear "does not exist" with a concrete path.
        return os.path.abspath(raw)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    # -- Engine lifecycle ---------------------------------------------------

    def run(self, dataloader=None):
        """No-op: sparse needs no calibration."""
        return None

    def calibrate(self, dataloader=None):
        """Alias for run() so the Engine PTQ-style ladder also works."""
        return self.run(dataloader)

    def convert(self):
        """Apply the sparse attention patch (idempotent)."""
        if self._patched:
            return
        self._patched_modules = apply_sparsity_patch(self.slim_model, self.algo)
        self._patched = True
        print_info(
            f"[sparse] patched {len(self._patched_modules)} attention layers "
            f"with algorithm '{self.algo.name}'."
        )

    def unpatch(self):
        if not self._patched:
            return
        unpatch_sparsity(self.slim_model, self._patched_modules)
        self._patched = False
        self._patched_modules = []

    def save(self, save_path):
        """Config-only save *on top of* HF save_pretrained.

        Sparse adds no new safetensors, but it MUST re-emit the dense weights +
        tokenizer so the directory is loadable. ``Engine.save`` writes
        ``angelslim_config.json`` afterwards.
        """
        os.makedirs(save_path, exist_ok=True)
        hf_model = self.slim_model.model
        hf_model.save_pretrained(save_path)
        tok = getattr(self.slim_model, "tokenizer", None)
        if tok is not None:
            tok.save_pretrained(save_path)

        # Copy the pattern JSON (minference) into the standard
        # ``<save_path>/sparse_patterns/`` location and record a path RELATIVE to
        # save_path in the serialized config. On reload, _resolve_pattern_path
        # re-anchors that relative path against the checkpoint dir — so the
        # round-trip does not depend on CWD.
        if self._pattern_path:
            # If a pattern_path was configured but the file is gone at save
            # time, we must NOT silently skip the rewrite — doing so leaves the
            # ORIGINAL (often absolute, internal) path in the serialized config,
            # which both leaks an internal filesystem path into a publishable
            # checkpoint AND makes reload fail (the absolute path won't exist on
            # the loading box). Hard-fail with an actionable message instead.
            if not os.path.isfile(self._pattern_path):
                raise FileNotFoundError(
                    f"[sparse] configured pattern_path does not exist at save "
                    f"time: {self._pattern_path!r}. The sparse pattern JSON must be "
                    f"present so it can be copied into <save_path>/sparse_patterns/ "
                    f"and recorded as a relative path. Refusing to save: a missing "
                    f"pattern would otherwise serialize the original absolute path "
                    f"(leaking an internal path and breaking reload). Restore the "
                    f"pattern file or clear pattern_path from the sparsity config."
                )
            dst_dir = os.path.join(save_path, "sparse_patterns")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(self._pattern_path))
            shutil.copy(self._pattern_path, dst)
            rel = os.path.relpath(dst, save_path)
            # Do NOT mutate the shared live attn_kwargs dict in place (that
            # only worked by object-identity luck and lost the rewrite when
            # Engine.save was handed a different config object). Rebuild the
            # field with a fresh dict on the dataclass/dict that gets serialized.
            self._set_serialized_pattern_path(rel)
        print_info(f"[sparse] model + tokenizer saved to {save_path}")

    def _set_serialized_pattern_path(self, rel: str) -> None:
        """Record the save-relative pattern_path on ``self.sparsity_cfg`` without
        mutating any dict shared with the live algorithm. Replaces the whole
        ``attn_kwargs`` with a fresh copy so ``asdict(config)`` serializes the
        relative path while the in-memory algorithm keeps its absolute one."""
        existing = self._cfg_get(self.sparsity_cfg, "attn_kwargs", {}) or {}
        new_kwargs = dict(existing)
        new_kwargs["pattern_path"] = rel
        if isinstance(self.sparsity_cfg, dict):
            self.sparsity_cfg["attn_kwargs"] = new_kwargs
        else:
            setattr(self.sparsity_cfg, "attn_kwargs", new_kwargs)  # noqa: B010
