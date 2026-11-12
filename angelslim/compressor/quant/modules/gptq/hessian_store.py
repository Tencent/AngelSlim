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

import json
import os
import re
from collections import OrderedDict

import torch
from safetensors import safe_open

__all__ = ["HessianStore"]

# vLLM fuses these projections into one module that reads the same input.
_FUSED = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "q_a_proj": "fused_qkv_a_proj",
    "kv_a_proj_with_mqa": "fused_qkv_a_proj",
}
# vLLM mounts the VLM text stack under ``language_model.``; checkpoints do not.
_PREFIXES = ("", "language_model.")
# Some collectors key MTP modules as ``model.layers.<N + i>.mtp_block.<X>``.
_LAYER_RE = re.compile(r"^(model\.layers\.\d+\.)(.+)$")


class HessianStore:
    """Reads H = E[x x^T] dumped by ``vllm_calib.hessian`` (hessian_meta.json + h_*.safetensors)
    and serves the upper Cholesky factor of (H + damp * mean(diag) * I)^-1 that GPTQ uses."""

    def __init__(self, hessian_dir, damp=0.05, cache_slots=2):
        with open(os.path.join(hessian_dir, "hessian_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.dir = hessian_dir
        self.groups = meta["groups"]
        self.modules = meta["modules"]
        self.damp = damp
        self.cache_slots = cache_slots
        self._cache = OrderedDict()
        self._not_pd = set()

    def entry(self, name):
        """Meta entry of a vLLM module name, tolerating fused names and the VLM prefix."""
        parent, _, leaf = name.rpartition(".")
        names = [name]
        if leaf in _FUSED:
            names.append(f"{parent}.{_FUSED[leaf]}" if parent else _FUSED[leaf])
        names += [m.group(1) + "mtp_block." + m.group(2) for m in map(_LAYER_RE.match, names) if m]
        for n in names:
            for prefix in _PREFIXES:
                hit = self.modules.get(prefix + n)
                if hit is not None:
                    return hit
        return None

    def get_hinv(self, name, device, damp=None):
        """Upper Cholesky of the damped inverse Hessian, or None if H is missing or not PD."""
        entry = self.entry(name)
        if entry is None:
            return None
        damp = self.damp if damp is None else damp
        slot = (entry["group"], entry.get("expert"), damp)
        if slot in self._not_pd:
            return None
        hit = self._cache.get(slot)
        if hit is not None and hit.device == torch.device(device):
            return hit

        group = self.groups[entry["group"]]
        with safe_open(os.path.join(self.dir, group["file"]), framework="pt", device="cpu") as f:
            H = f.get_slice("H")[entry["expert"]] if group.get("per_expert") else f.get_tensor("H")
        H = H.to(device=device, dtype=torch.float64)
        H.diagonal().add_(damp * H.diagonal().mean())
        try:
            hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            hinv = torch.linalg.cholesky(hinv, upper=True)
        except torch.linalg.LinAlgError:
            self._not_pd.add(slot)
            return None
        hinv = hinv.float()
        self._cache[slot] = hinv
        while len(self._cache) > self.cache_slots:
            self._cache.popitem(last=False)
        return hinv
