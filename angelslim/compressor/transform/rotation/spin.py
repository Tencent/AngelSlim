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

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..base import TransformBase
from ..factory import TransformFactory
from .fuse_norm_utils import center_embeddings, fuse_ln_linear
from .hadamard_utils import hadamard_matrix, random_hadamard_matrix
from .mapping import linear_mapping as default_linear_mapping
from .mapping import norm_mapping as default_norm_mapping

__all__ = ["SpinQuant", "SpinConfig", "SpinquantRotation"]


class SpinquantRotation(str, Enum):
    """Enumeration for SpinQuant rotation types."""

    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


@dataclass
class SpinConfig:
    """Configuration for SpinQuant rotation.

    Attributes:
        had_dim: Hadamard block size for online (R3/R4) rotation. Must be a power of 2.
        rotation_mode: Rotation mode for R1/R2; R3/R4 are fixed to Hadamard.
        rotation: List of rotation types to apply. Defaults to [R1, R2]; R3 is not yet implemented.
        ignore_layers: List of layer names to ignore for rotation.
        mappings: Linear layer name mapping dict.
        norm_mappings: Norm-to-linear fuse mapping list.
    """

    had_dim: int = -1  # -1 for full size, support block online hadamard
    rotation_mode: str = "Hadamard"  # controls R1, R2; R3, R4 are fixed to Hadamard
    rotation: List[SpinquantRotation] = field(default_factory=lambda: [])
    ignore_layers: List[str] = field(default_factory=list)
    mappings: Optional[Dict] = field(default=None)
    norm_mappings: Optional[List] = field(default=None)
    device: str = "cpu"
    max_threads: int = 64


@TransformFactory.register("SpinQuant")
class SpinQuant(TransformBase):
    """SpinQuant: weight-space rotation for quantization-friendly distributions.

    Applies random orthogonal (Hadamard-based) rotations to model weights so that
    outlier channels are suppressed before quantization. The rotation is equivalent
    in the forward pass (W' = W @ R, x' = x @ R^T), leaving model output unchanged.

    Rotation types (following SpinQuant naming convention):
        R1 - fused offline rotation at embedding output / lm_head input.
        R2 - fused offline rotation absorbed into layer norms and adjacent linears.
        R3 - online Hadamard on Q/K projections, fused with RoPE.
        R4 - online Hadamard on the down-projection (FFN) input.

    Args:
        quant_model: The wrapped quantization model (provides layer accessors).
        spin_config: SpinConfig instance controlling per-rotation options.
    """

    R1: torch.Tensor = None
    R2: Dict[str, torch.Tensor] = None
    R3: torch.Tensor = None
    R4: torch.Tensor = None

    R1_embed_linears: List[torch.nn.Linear] = None  # embedding only
    R1_linears: List[torch.nn.Linear] = None  # attn_o, mlp_out (output-side, left-multiply)
    R1_inv_linears: List[torch.nn.Linear] = (
        None  # q/k/v, mlp_in, lm_head (input-side, right-multiply)
    )

    R2_linears: List[torch.nn.Linear] = None
    R2_inv_linears: List[torch.nn.Linear] = None

    R3_linears: List[torch.nn.Linear] = None
    R4_linears: List[torch.nn.Linear] = None

    def __init__(self, quant_model, quant_config=None, R1=None, R2=None):
        super().__init__(quant_model, quant_config)
        self.transform_config = quant_config.get("transform_config", None)
        assert self.transform_config is not None, "transform_config must be provided"
        self.deploy_vllm = quant_config.get("global_config").deploy_backend == "vllm"

        def _get(key, default):
            return getattr(self.transform_config, key, default)

        spin_config = _get("spin_config", SpinConfig())
        self.spin_config = (
            SpinConfig(**spin_config) if isinstance(spin_config, dict) else spin_config
        )
        self.norm_mapping = (
            default_norm_mapping
            if self.spin_config.norm_mappings is None
            else self.spin_config.norm_mappings
        )
        self.linear_mapping = (
            default_linear_mapping
            if self.spin_config.mappings is None
            else self.spin_config.mappings
        )

        self.ignore_layers = self.spin_config.ignore_layers

        if _get("output_log", False):
            logging.basicConfig(
                level=logging.INFO,
                filename=os.path.join(
                    self.config.global_config.absolute_model_path, "transform.log"
                ),
            )

        self.logger = logging.getLogger(__name__)

        self.R1 = R1
        self.R2 = R2 if R2 is not None else {}

        self.R4_hooks = []

    def silent_run(self):
        """only set linears and Rotations"""
        self._apply_fused_ln()
        if "R1" in self.spin_config.rotation:
            self._apply_r1(no_transform=True)
        if "R2" in self.spin_config.rotation:
            self._apply_r2(no_transform=True)
        if "R3" in self.spin_config.rotation:
            self._apply_r3(no_transform=True)
        if "R4" in self.spin_config.rotation:
            self._apply_r4(no_transform=True)

    def run(self):
        # add rotation
        if "R1" in self.spin_config.rotation:
            # fuse norm
            self._apply_fused_ln()
            self._apply_r1()
        if "R2" in self.spin_config.rotation:
            self._apply_r2()
        if "R3" in self.spin_config.rotation:
            self._apply_r3()
        if "R4" in self.spin_config.rotation:
            self._apply_r4()

    def _apply_linear_hook(self, linear, rotation, hook_input=True):
        """
        Apply a rotation hook to a linear layer.

        Args:
            linear: The linear layer to hook.
            rotation: The rotation matrix to apply.
            hook_input: If True, hook the input; if False, hook the output.
        """
        if hook_input:

            def pre_hook(module, inputs):
                x = inputs[0]
                rot = rotation.to(device=x.device)
                x_rot = (x.to(torch.float32) @ rot.to(torch.float32)).to(x.dtype)
                return (x_rot, *inputs[1:])

            return linear.register_forward_pre_hook(pre_hook)
        else:

            def out_hook(module, inputs, output):
                rot = rotation
                if isinstance(output, tuple):
                    y = output[0]
                    rot = rot.to(device=y.device, dtype=y.dtype)
                    y_rot = (y.to(torch.float32) @ rot.T.to(torch.float32)).to(y.dtype)
                    return (y_rot, *output[1:])
                else:
                    rot = rot.to(device=output.device, dtype=output.dtype)
                    return (output.to(torch.float32) @ rot.T.to(torch.float32)).to(output.dtype)

            return linear.register_forward_hook(out_hook)

    @torch.no_grad()
    def _meta_fuse(self, name, linear, rotation, fuse_input=False):
        """Fuse rotation into a linear layer whose weight lives on the meta device.

        When a model is loaded with accelerate offloading, parameters may reside
        on the ``meta`` device; actual data is managed by an ``_hf_hook``.
        This function materialises the weight by triggering the hook, applies
        the same rotation fuse logic as :meth:`_apply_linear_fuse` (non-meta path),
        and lets the hook offload the updated weight afterwards.

        Args:
            name: Fully-qualified layer name (used for logging).
            linear: The linear module whose weight is a meta tensor.
            rotation: The rotation matrix to fuse (float32, on DEVICE).
            fuse_input: Matches the ``fuse_input`` semantics of
                :meth:`_apply_linear_fuse`.
        """
        # self.logger.warning(
        #     f"[_meta_fuse] '{name}' weight is on meta device, attempting hook-based fuse"
        # )
        hook = getattr(linear, "_hf_hook", None)
        if hook is None:
            self.logger.warning(
                f"[_meta_fuse] '{name}' has no _hf_hook attached; "
                "cannot materialise weight, skipping"
            )
            return

        # Temporarily redirect execution_device to CPU to avoid OOM when materialising
        # large weights that would otherwise be sent to GPU.
        original_exec_device = hook.execution_device
        hook.execution_device = self.spin_config.device
        hook.pre_forward(linear)

        if linear.weight.is_meta:
            self.logger.warning(
                f"[_meta_fuse] '{name}' weight is still meta after hook.pre_forward; skipping"
            )
            hook.post_forward(linear, None)
            hook.execution_device = original_exec_device
            return

        # Apply the same fuse logic as _apply_linear_fuse (non-meta path)
        # Weight is already on DEVICE (cpu) thanks to the redirected execution_device above.
        weight = linear.weight.data.to(device=self.spin_config.device, dtype=torch.float32)
        origin_device = self.spin_config.device
        if fuse_input:
            new_weight = weight @ rotation
            linear.weight.data = new_weight.to(dtype=linear.weight.dtype, device=origin_device)
        else:
            new_weight = rotation.T @ weight
            linear.weight.data = new_weight.to(dtype=linear.weight.dtype, device=origin_device)
            if hasattr(linear, "bias") and linear.bias is not None:
                bias = linear.bias.data.to(device=self.spin_config.device, dtype=torch.float32)
                new_bias = rotation.T @ bias
                linear.bias.data = new_bias.to(dtype=linear.bias.dtype, device=origin_device)

        # Let the hook offload the updated weight back to its storage, then restore device
        hook.post_forward(linear, None)
        hook.execution_device = original_exec_device

    @torch.no_grad()
    def _apply_linear_fuse(self, linear, rotation, fuse_input=False, name=None):
        """Fuse a rotation matrix into a linear layer's weight in-place.

        Internally transposes `rotation` before use:
          fuse_input=False  ->  new_weight = rotation.T @ weight  (output-side rotation)
          fuse_input=True   ->  new_weight = weight @ rotation.T  (input-side de-rotation)

        To achieve W' = W @ R, pass rotation=R.T with fuse_input=True.
        To achieve W' = R.T @ W, pass rotation=R with fuse_input=False.

        Computation is performed on DEVICE; the result is moved back to the
        original device of the weight before writing.
        """
        is_meta = linear.weight.is_meta
        if is_meta:
            self._meta_fuse(name, linear, rotation, fuse_input=fuse_input)
            return True
        weight = linear.weight.data.to(device=self.spin_config.device, dtype=torch.float32)
        origin_device = linear.weight.device

        if fuse_input:
            new_weight = weight @ rotation  # W @ R, X W^T -> X R R^T W^T -> X R x (WR)^T
            linear.weight.data = new_weight.to(dtype=linear.weight.dtype, device=origin_device)
        else:
            new_weight = rotation.T @ weight
            linear.weight.data = new_weight.to(dtype=linear.weight.dtype, device=origin_device)
            if hasattr(linear, "bias") and linear.bias is not None:
                bias = linear.bias.data.to(device=self.spin_config.device, dtype=torch.float32)
                new_bias = rotation.T @ bias
                linear.bias.data = new_bias.to(dtype=linear.bias.dtype, device=origin_device)

        return False

    @torch.no_grad()
    def _apply_emb_fuse(self, embedding, rotation, fast_mode=False):
        """Fuse rotation into embedding weight.

        Computation is performed on DEVICE; the result is moved back to the
        original device of the weight before writing.
        """
        origin_device = embedding.weight.device
        weight = embedding.weight.data.to(device=self.spin_config.device, dtype=torch.float32)
        rotation = rotation.to(device=self.spin_config.device, dtype=torch.float32)
        embedding.weight.data = (weight @ rotation).to(
            dtype=embedding.weight.dtype, device=origin_device
        )

    def _parallel_apply(self, tasks, desc=None):
        """Concurrently execute a list of (fn, args, kwargs) tasks and wait for all to finish.

        Each task operates on an independent linear layer, so there are no write
        conflicts between workers.  The ThreadPoolExecutor context manager guarantees
        all submitted futures are done before __exit__ returns; calling f.result()
        additionally re-raises any exception thrown inside a worker thread.

        Args:
            tasks: list of (callable, args_tuple, kwargs_dict)
            desc: optional progress bar label shown next to the tqdm bar
        """

        pbar = tqdm(total=len(tasks), desc=desc, leave=False)

        def _wrap(fn, args, kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                pbar.update(1)

        with ThreadPoolExecutor(max_workers=self.spin_config.max_threads) as executor:
            futures = [executor.submit(_wrap, fn, args, kwargs) for fn, args, kwargs in tasks]
        # __exit__ above already joined all threads; iterate futures to surface exceptions
        pbar.close()
        for f in futures:
            f.result()

    def _apply_fused_ln(self):
        """Apply fused layer norm to a linear layer.
        1. centering embedding
        2. fuse layer norm with adjacent linear layers
        """
        self.logger.info("Applying fused layer norm to a linear layer")

        hf_model = self.quant_model.model
        if (
            hasattr(hf_model, "lm_head")
            and hasattr(hf_model, "model")
            and hasattr(hf_model.model, "embed_tokens")
        ):
            if hf_model.lm_head.weight.data_ptr() == hf_model.model.embed_tokens.weight.data_ptr():
                hf_model.lm_head.weight = torch.nn.Parameter(hf_model.lm_head.weight.data.clone())

        for _, embedding in self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=([self.linear_mapping["embedding"]], self.ignore_layers),
        ).items():
            center_embeddings(embedding)

        norm_layers = self.quant_model.get_rotation_mapping_layers(
            None,
            norm_mapping=self.norm_mapping,
        )

        for _, (norm_layer, linear_layers_list) in norm_layers.items():
            fuse_ln_linear(norm_layer, [layer for _, layer in linear_layers_list])

    @torch.no_grad()
    def _apply_r1(self, no_transform=False):
        """Apply R1 rotation to embedding and lm_head, R1^T to q/k/v, mlp_in, lm_head"""

        self.logger.info(
            "Applying R1 rotation to embedding and lm_head, R1^T to q/k/v, mlp_in, lm_head"
        )
        # generate R1
        if self.R1 is None:
            if self.spin_config.rotation_mode == "Hadamard":
                self.R1 = hadamard_matrix(
                    self.quant_model.model.config.hidden_size, self.spin_config.device
                )
            else:
                self.R1 = random_hadamard_matrix(
                    self.quant_model.model.config.hidden_size, self.spin_config.device
                )

        self.R1_embed_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=([self.linear_mapping["embedding"]], self.ignore_layers),
        )
        # attn_o, mlp_out: output in hidden_size dim → W' = R1.T @ W → fuse_input=False, pass R1
        self.R1_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=(
                [self.linear_mapping["attn_o"]] + self.linear_mapping["mlp_out"],
                [],  # if set R1, all linear layers will not be ignored
            ),
        )
        # q/k/v, mlp_in, lm_head:
        # input in hidden_size dim → W' = W @ R1 → fuse_input=True, pass R1.T
        self.R1_inv_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=(
                [
                    self.linear_mapping["attn_q"],
                    self.linear_mapping["attn_k"],
                    self.linear_mapping["attn_v"],
                ]
                + self.linear_mapping["mlp_in"]
                + [self.linear_mapping["lm_head"]],
                [],  # if set R1, all linear layers will not be ignored
            ),
        )

        if no_transform:
            return

        tasks = []
        # embedding: W' = W @ R1
        for linear in self.R1_embed_linears.values():
            tasks.append((self._apply_emb_fuse, (linear, self.R1), {}))
        # attn_o, mlp_out: W' = R1.T @ W
        for name, linear in self.R1_linears.items():
            tasks.append(
                (self._apply_linear_fuse, (linear, self.R1), {"fuse_input": False, "name": name})
            )
        # q/k/v, mlp_in, lm_head: W' = W @ R1
        for name, linear in self.R1_inv_linears.items():
            tasks.append(
                (self._apply_linear_fuse, (linear, self.R1), {"fuse_input": True, "name": name})
            )
        self._parallel_apply(tasks, desc="R1 fuse")

    @torch.no_grad()
    def _apply_r2(self, no_transform=False):
        """
        Absorb R2 into attn_v (output side) and attn_o (input side)
        """
        self.logger.info("Applying R2 rotation to attn_v and attn_o")

        # get linear layers
        self.R2_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=([self.linear_mapping["attn_v"]], self.ignore_layers),
        )
        self.R2_inv_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=([self.linear_mapping["attn_o"]], self.ignore_layers),
        )

        assert len(self.R2_linears) == len(
            self.R2_inv_linears
        ), "R2_linears and R2_inv_linears must have the same length"

        cfg = self.quant_model.model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        num_q_heads = cfg.num_attention_heads

        if no_transform:
            return

        def _fuse_one_r2_block(name, linear, inv_linear):
            # Generate a single head_dim Hadamard block H, then tile for kv/q heads
            if self.spin_config.rotation_mode == "Hadamard":
                H = hadamard_matrix(head_dim, self.spin_config.device)
            else:
                H = random_hadamard_matrix(head_dim, self.spin_config.device)
            # R2_v: block-diagonal with num_kv_heads copies of H  → [v_total, v_total]
            R2_v = torch.block_diag(*([H] * num_kv_heads))
            # R2_o: block-diagonal with num_q_heads copies of H   → [o_total, o_total]
            R2_o = torch.block_diag(*([H] * num_q_heads))
            self._apply_linear_fuse(linear, R2_v, fuse_input=False, name=name)
            self._apply_linear_fuse(inv_linear, R2_o, fuse_input=True, name=name)
            # record R2 per layer (store the shared head block H)
            self.R2[name] = H

        tasks = []
        for name, (linear, inv_linear) in zip(
            self.R2_linears.keys(), zip(self.R2_linears.values(), self.R2_inv_linears.values())
        ):
            tasks.append((_fuse_one_r2_block, (name, linear, inv_linear), {}))
        self._parallel_apply(tasks, desc="R2 fuse")

    @torch.no_grad()
    def _apply_r3(self, no_transform=False):
        """Insert online R3 Hadamard rotation for Q/K projections, fused with RoPE.

        TODO: Implement R3 online rotation module insertion.
        """
        raise NotImplementedError("SpinQuant._apply_r3 is not yet implemented.")

    @torch.no_grad()
    def _apply_r4(self, no_transform=False):
        """Insert online R4 Hadamard rotation for the down-projection input (FFN).

        Registers a forward pre-hook on down_proj that applies R4 to its input at runtime,
        and fuses R4.T into the down_proj weight so the combined forward is equivalent
        to the original: (x @ R4) @ (W @ R4).T = x @ W.T.
        """
        self.logger.info("Applying R4 rotation to down_proj")

        self.R4_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=(self.linear_mapping["mlp_out"], self.ignore_layers),
        )

        if self.spin_config.had_dim > 0:
            H = hadamard_matrix(self.spin_config.had_dim, self.spin_config.device)
        else:
            H = None
        R4_dict = {}

        if no_transform:
            return

        # Phase 1 (sequential): build R4_dict and handle one-time weight scaling per shape.
        # This must run serially because R4_dict is populated on first encounter of each shape
        # and the weight pre-scaling of that first linear must complete before fuse.
        weight_device = None
        for _, linear in self.R4_linears.items():
            if weight_device is None:
                weight_device = linear.weight.device
            if linear.weight.shape[-1] not in R4_dict:
                if self.spin_config.had_dim < 0:
                    rot = hadamard_matrix(linear.weight.shape[-1], self.spin_config.device)
                    # rot = rot * (linear.weight.shape[-1] ** 0.5)  # reverse to +1/-1 matrix
                    # linear.weight.data = linear.weight.data / (linear.weight.shape[-1] ** 0.5)
                    R4_dict[linear.weight.shape[-1]] = rot
                else:
                    rot = torch.block_diag(
                        *([H] * (linear.weight.shape[-1] // self.spin_config.had_dim))
                    )
                    # rot = rot * (self.spin_config.had_dim**0.5)  # reverse to +1/-1 matrix
                    # linear.weight.data = linear.weight.data / (self.spin_config.had_dim**0.5)
                    R4_dict[linear.weight.shape[-1]] = rot
            if H is None:
                H = R4_dict[linear.weight.shape[-1]]

        assert len(R4_dict) == 1, "R4_dict must have only one entry, please check your model"
        self.R4 = H
        rot = next(iter(R4_dict.values()))

        ORI_DEVICE_H = rot.to(weight_device)

        # Phase 2 (parallel): hook + fuse are independent per linear, safe to run concurrently.
        def _hook_and_fuse(name, linear, rotation):
            hook = self._apply_linear_hook(linear, ORI_DEVICE_H, hook_input=True)
            self._apply_linear_fuse(linear, rotation, fuse_input=True, name=name)
            self.R4_hooks.append(hook)

        tasks = [
            (_hook_and_fuse, (name, linear, rot), {}) for name, linear in self.R4_linears.items()
        ]
        self._parallel_apply(tasks, desc="R4 fuse")

        if self.deploy_vllm:
            transform_config = {
                "config_groups": {
                    "R4": {
                        "apply": [
                            {
                                "ignore": [
                                    f"re:.*{tgt}.*$" for tgt in self.spin_config.ignore_layers
                                ],
                                "inverse": False,
                                "location": "input",
                                "targets": [
                                    f"re:.*{tgt}$" for tgt in self.linear_mapping["mlp_out"]
                                ],
                            }
                        ],
                        "head_dim": self.spin_config.had_dim,
                        "randomize": False,
                        "requires_grad": False,
                        "type": "hadamard",
                    }
                }
            }
            self.quant_model.quant_config.transform_config = transform_config

    @torch.no_grad()
    def convert(self, R1=None, R2_list=None, R3_list=None, R4_list=None):
        """Fuse rotation matrices into weights after QAT training.

        Intended for use when hooks were registered during training (trainable mode).
        Call this after QAT training ends to fuse all online hooks into weights.
        """
        if R1 is not None:
            tasks = []
            # embedding: W' = W @ R1  (same as _apply_emb_fuse in PTQ)
            for linear in self.R1_embed_linears.values():
                tasks.append((self._apply_emb_fuse, (linear, self.R1), {}))
            # attn_o, mlp_out: W' = R1.T @ W
            for linear in self.R1_linears.values():
                tasks.append((self._apply_linear_fuse, (linear, self.R1), {"fuse_input": False}))
            # q/k/v, mlp_in, lm_head: W' = W @ R1
            for linear in self.R1_inv_linears.values():
                tasks.append((self._apply_linear_fuse, (linear, self.R1), {"fuse_input": True}))
            self._parallel_apply(tasks, desc="convert R1 fuse")

        if R2_list is not None:
            assert len(R2_list) == len(
                self.R2_linears
            ), "R2_list and R2_linears must have the same length"
            cfg = self.quant_model.model.config
            num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            num_q_heads = cfg.num_attention_heads

            def _fuse_one_r2_convert(linear, inv_linear, H):
                # Tile the per-head block H into full block-diagonal matrices, same as PTQ
                R2_v = torch.block_diag(*([H] * num_kv_heads))
                R2_o = torch.block_diag(*([H] * num_q_heads))
                self._apply_linear_fuse(linear, R2_v, fuse_input=False)
                self._apply_linear_fuse(inv_linear, R2_o, fuse_input=True)

            tasks = []
            for (name, linear), H in zip(self.R2_linears.items(), R2_list):
                inv_linear_name = name.replace(
                    self.linear_mapping["attn_v"], self.linear_mapping["attn_o"]
                )
                inv_linear = self.R2_inv_linears.get(inv_linear_name, None)
                assert inv_linear is not None, f"R2_inv_linear {inv_linear_name} not found"
                tasks.append((_fuse_one_r2_convert, (linear, inv_linear, H), {}))
            self._parallel_apply(tasks, desc="convert R2 fuse")

        if R3_list is not None:
            raise NotImplementedError("SpinQuant.convert R3 is not yet implemented.")

        if R4_list is not None:
            self.logger.warning(
                "Note: if R4 isn't hadamard matrix, there will be no acceleration in vllm"
            )
            self.logger.warning(
                "Note: this function will not hook for R4, only fuse hadamard into weight"
            )
            if len(R4_list) == 1:
                R4_list = R4_list * len(self.R4_linears)
            tasks = [
                (self._apply_linear_fuse, (linear, R4), {"fuse_input": True})
                for (_, linear), R4 in zip(self.R4_linears.items(), R4_list)
            ]
            self._parallel_apply(tasks, desc="convert R4 fuse")

    @torch.no_grad()
    def save(self):
        """Save model process."""
        # remove hooks
        for hook in tqdm(self.R4_hooks, desc="remove R4 hooks"):
            hook.remove()

    def get_rotation_mat(self):
        """Get the rotation matrices."""
        return dict(R1=self.R1, R2=self.R2, R3=self.R3, R4=self.R4)

    def get_linears(self):
        """Get the linear layers."""
        return dict(
            # R1 embed.token weight shape [input, output],
            # different from R1_linears and R1_inv_linears
            R1=[self.R1_embed_linears, self.R1_linears, self.R1_inv_linears],
            R2=[self.R2_linears, self.R2_inv_linears],
            R3=[self.R3_linears],
            R4=[self.R4_linears],
        )
