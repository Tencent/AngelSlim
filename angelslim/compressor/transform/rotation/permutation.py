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

import concurrent
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from angelslim.utils.utils import print_info

from ..base import TransformBase
from .mapping import linear_mapping as default_linear_mapping

__all__ = ["Permutation", "PermutationConfig"]


@dataclass
class PermutationConfig:
    """Configuration for R4 channel permutation.
    Attributes:
        block_size: Quantization group size. The zig-zag interleave distributes
            large/small-norm channels evenly so each quant block of this size
            contains a balanced mix.
        mappings: Linear layer name mapping dict (same format as SpinConfig).
        ignore_layers: List of layer name patterns to skip.
        device: Device for temporary computation during permutation.
    """

    block_size: int = 128
    mappings: Optional[Dict] = field(default=None)
    ignore_layers: List[str] = field(default_factory=list)
    device: str = "cpu"


def _zigzag_permutation(order: torch.Tensor) -> torch.Tensor:
    """Build a zig-zag permutation from a descending-sorted index tensor.

    Uses a single global head/tail pointer to interleave the largest and
    smallest elements alternately, e.g.:

        order = [8, 7, 6, 5, 4, 3, 2, 1, 0]   (sorted largest → smallest)
        result = [8, 0, 7, 1, 6, 2, 5, 3, 4]  (0-indexed)
        # 1-indexed equivalent: [9, 1, 8, 2, 7, 3, 6, 4, 5]

    This ensures every contiguous block of output channels contains both
    high-norm and low-norm inputs, which is beneficial for per-group
    quantization (e.g. W4A8 with group_size=128).

    Args:
        order: 1-D index tensor sorted largest → smallest (length N).

    Returns:
        1-D permutation index tensor of length N.
    """
    n = order.numel()
    result = torch.empty(n, dtype=order.dtype)
    head, tail, pos = 0, n - 1, 0
    while head <= tail:
        result[pos] = order[head]
        pos += 1
        head += 1
        if head <= tail:
            result[pos] = order[tail]
            pos += 1
            tail -= 1
    return result


def get_r4_permutation(down_proj_weight: torch.Tensor) -> torch.Tensor:
    """Compute the zig-zag channel permutation for a down_proj weight.

    Computes the L1 norm of each input channel (dim 1), sorts descending,
    then applies the global head/tail zig-zag interleave.

    Args:
        down_proj_weight: Weight tensor of shape [out_features, in_features].

    Returns:
        1-D index tensor of shape [in_features] for channel reordering.
    """
    l1_norm = down_proj_weight.float().abs().sum(dim=0)  # [in_features]
    order = torch.argsort(l1_norm, descending=True)
    return _zigzag_permutation(order)


class Permutation(TransformBase):
    """R4 channel permutation for quantization-friendly weight layout.

    For each FFN layer, computes a zig-zag permutation of the intermediate
    (hidden) dimension based on the L1 norm of down_proj input channels, then
    applies the same permutation consistently to:
        - down_proj  : reorder input channels  (weight[:, perm])
        - up_proj    : reorder output channels (weight[perm, :])
        - gate_proj  : reorder output channels (weight[perm, :])

    This is a pure offline weight reordering — the forward pass remains
    numerically equivalent when the runtime activations are permuted the same
    way (or when the transform is composed end-to-end with e.g. SpinQuant R4).

    Args:
        quant_model: The wrapped quantization model.
        quant_config: Config dict containing ``"transform_config"`` with a
            ``PermutationConfig`` (or equivalent dict) under the key
            ``"permutation_config"``.
    """

    def __init__(self, quant_model, permutation_config=None, quant_config=None):
        super().__init__(quant_model, quant_config)

        self.perm_config = (
            PermutationConfig(**permutation_config)
            if isinstance(permutation_config, dict)
            else permutation_config
        )
        self.linear_mapping = (
            default_linear_mapping
            if self.perm_config.mappings is None
            else self.perm_config.mappings
        )
        self.ignore_layers = self.perm_config.ignore_layers

        # Stores per-layer permutation indices: {down_proj_name: torch.Tensor}
        self.permutations: Dict[str, torch.Tensor] = {}

    def run(self):
        """Compute and apply zig-zag R4 permutation to all FFN layers."""
        print_info("Applying R4 zig-zag channel permutation")

        down_proj_layers = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=(self.linear_mapping["mlp_out"], self.ignore_layers),
        )
        mlp_in_layers = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=(self.linear_mapping["mlp_in"], self.ignore_layers),
        )

        # multi-thread
        def process_down_proj(name, down_proj):
            # If weight is on meta device, materialise it via _hf_hook first.
            hook, original_exec_device = None, None
            if down_proj.weight.is_meta:
                hook = getattr(down_proj, "_hf_hook", None)
                if hook is None:
                    print_info(
                        f"[warning] '{name}' weight is on meta device and has no "
                        "_hf_hook; cannot compute R4 permutation, skipping"
                    )
                    return name, None
                original_exec_device, hook.execution_device = (
                    hook.execution_device,
                    self.perm_config.device,
                )
                hook.pre_forward(down_proj)
                if down_proj.weight.is_meta:
                    print_info(
                        f"[warning] '{name}' weight is still meta after hook.pre_forward; skipping"
                    )
                    hook.post_forward(down_proj, None)
                    hook.execution_device = original_exec_device
                    return name, None

            # Unified path: compute permutation and apply it.
            perm = get_r4_permutation(down_proj.weight.data.to(device=self.perm_config.device))
            self._permute_down_proj(name, down_proj, perm)

            if hook is not None:
                hook.post_forward(down_proj, None)
                hook.execution_device = original_exec_device
            return name, perm

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_down_proj, name, down_proj)
                for name, down_proj in down_proj_layers.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                name, perm = future.result()
                if perm is not None:
                    self.permutations[name] = perm
        print_info(f"Permutation down_proj done. {len(self.permutations)} layer(s) processed.")

        def process_mlp_in(name, linear):
            perm = self._find_matching_perm(name)
            if perm is None:
                print_info(f"[warning] No matching permutation for {name}, skipping")
                return
            self._permute_mlp_in(name, linear, perm)
            return name, perm

        mlp_in_sucess_count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_mlp_in, name, linear)
                for name, linear in mlp_in_layers.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                _, perm = future.result()
                if perm is not None:
                    mlp_in_sucess_count += 1

        print_info(
            f"Permutation up_proj&gate_proj done. {mlp_in_sucess_count} layer(s) processed."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_matching_perm(self, layer_name: str) -> Optional[torch.Tensor]:
        """Return the permutation for the down_proj in the same FFN block.

        Uses longest common prefix matching so MoE expert paths are handled
        correctly (same logic as SpinQuant linear_mapping longest-prefix rule).
        """
        best_key, best_len = None, -1
        for key in self.permutations:
            prefix = key.rsplit(".", 1)[0]
            if layer_name.startswith(prefix) and len(prefix) > best_len:
                best_key, best_len = key, len(prefix)
        return self.permutations[best_key] if best_key is not None else None

    @torch.no_grad()
    def _permute_down_proj(self, name: str, linear: torch.nn.Linear, perm: torch.Tensor):
        """Reorder input channels (dim 1) of down_proj in-place."""
        if linear.weight.is_meta:
            self._meta_permute(name, linear, perm, dim=1)
            return
        origin_device = linear.weight.device
        w = linear.weight.data.to(device=self.perm_config.device)
        linear.weight.data = w[:, perm].to(device=origin_device)

    @torch.no_grad()
    def _permute_mlp_in(self, name: str, linear: torch.nn.Linear, perm: torch.Tensor):
        """Reorder output channels (dim 0) of up_proj / gate_proj in-place."""
        if linear.weight.is_meta:
            self._meta_permute(name, linear, perm, dim=0)
            return
        origin_device = linear.weight.device
        w = linear.weight.data.to(device=self.perm_config.device)
        linear.weight.data = w[perm, :].to(device=origin_device)
        if hasattr(linear, "bias") and linear.bias is not None:
            b = linear.bias.data.to(device=self.perm_config.device)
            linear.bias.data = b[perm].to(device=origin_device)

    @torch.no_grad()
    def _meta_permute(self, name: str, linear: torch.nn.Linear, perm: torch.Tensor, dim: int):
        """Apply channel permutation to a linear layer whose weight lives on the meta device.

        When a model is loaded with accelerate offloading, parameters may reside
        on the ``meta`` device; actual data is managed by an ``_hf_hook``.
        This function materialises the weight by triggering the hook, applies
        the permutation (equivalent to the non-meta path in
        :meth:`_permute_down_proj` / :meth:`_permute_mlp_in`), and lets the
        hook offload the updated weight afterwards.

        Args:
            name: Fully-qualified layer name (used for logging).
            linear: The linear module whose weight is a meta tensor.
            perm: 1-D permutation index tensor.
            dim: Dimension along which to permute.
                ``0`` → reorder output channels (up_proj / gate_proj).
                ``1`` → reorder input  channels (down_proj).
        """
        hook = getattr(linear, "_hf_hook", None)
        if hook is None:
            print_info(
                f"[_meta_permute] '{name}' has no _hf_hook attached; "
                "cannot materialise weight, skipping"
            )
            return

        # Redirect execution_device to perm_config.device to avoid OOM when
        # materialising large weights that would otherwise land on GPU.
        original_exec_device = hook.execution_device
        hook.execution_device = self.perm_config.device
        hook.pre_forward(linear)

        if linear.weight.is_meta:
            print_info(
                f"[_meta_permute] '{name}' weight is still meta after hook.pre_forward; skipping"
            )
            hook.post_forward(linear, None)
            hook.execution_device = original_exec_device
            return

        origin_dtype = linear.weight.dtype
        w = linear.weight.data.to(device=self.perm_config.device)

        if dim == 1:
            # Reorder input channels: weight[:, perm]
            linear.weight.data = w[:, perm].to(dtype=origin_dtype, device=self.perm_config.device)
        else:
            # Reorder output channels: weight[perm, :]
            linear.weight.data = w[perm, :].to(dtype=origin_dtype, device=self.perm_config.device)
            if hasattr(linear, "bias") and linear.bias is not None:
                b = linear.bias.data.to(device=self.perm_config.device)
                linear.bias.data = b[perm].to(
                    dtype=linear.bias.dtype, device=self.perm_config.device
                )

        # Let the hook offload the updated weight back to its storage, then restore device.
        hook.post_forward(linear, None)
        hook.execution_device = original_exec_device

    def get_permutations(self) -> Dict[str, torch.Tensor]:
        """Return the per-layer permutation index tensors (keyed by down_proj name)."""
        return self.permutations
