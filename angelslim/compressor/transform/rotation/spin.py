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
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch

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

    had_dim: int = -1  # -1 for full size, support block online hadamard [TODO]
    rotation_mode: str = "Hadamard"  # controls R1, R2; R3, R4 are fixed to Hadamard
    rotation: List[SpinquantRotation] = field(default_factory=lambda: [])
    ignore_layers: List[str] = field(default_factory=list)
    mappings: Optional[Dict] = field(default=None)
    norm_mappings: Optional[List] = field(default=None)


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
    R2_paired: Dict = None

    R3_linears: List[torch.nn.Linear] = None
    R4_linears: List[torch.nn.Linear] = None

    def __init__(self, quant_model, quant_config=None, R1=None, R2=None):
        super().__init__(quant_model, quant_config)
        self.transform_config = quant_config.get("transform_config", None)
        assert self.transform_config is not None, "transform_config must be provided"

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

        assert self.spin_config.had_dim == -1, "only support had_dim == -1 now"

    def slient_run(self):
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
                rot = rotation.to(device=x.device, dtype=x.dtype)
                x_rot = (x.to(torch.float32) @ rot.to(torch.float32)).to(x.dtype)
                return (x_rot, *inputs[1:])

            linear.register_forward_pre_hook(pre_hook)
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

            linear.register_forward_hook(out_hook)

    def _apply_fast_hadamard_input_hook(self, linear):

        raise NotImplementedError(
            "_apply_fast_hadamard_input_hook requires get_hadK and matmul_hadU_cuda "
            "which are not yet imported. Add them to hadamard_utils and import here."
        )

    @torch.no_grad()
    def _apply_linear_fuse(self, linear, rotation, fuse_input=False):
        """Fuse a rotation matrix into a linear layer's weight in-place.

        Internally transposes `rotation` before use:
          fuse_input=False  ->  new_weight = rotation.T @ weight  (output-side rotation)
          fuse_input=True   ->  new_weight = weight @ rotation.T  (input-side de-rotation)

        To achieve W' = W @ R, pass rotation=R.T with fuse_input=True.
        To achieve W' = R.T @ W, pass rotation=R with fuse_input=False.
        """

        weight = linear.weight.data.to(torch.float32)
        rotation = rotation.to(device=weight.device, dtype=torch.float32)
        if hasattr(linear, "bias") and linear.bias is not None:
            bias = linear.bias.data.to(torch.float32)

        if fuse_input:
            new_weight = weight @ rotation  # W @ R, X W^T -> X R R^T W^T -> X R x (WR)^T
            linear.weight.data = new_weight.to(linear.weight.dtype)
        else:
            new_weight = rotation.T @ weight
            linear.weight.data = new_weight.to(linear.weight.dtype)
            if hasattr(linear, "bias") and linear.bias is not None:
                new_bias = rotation.T @ bias
                linear.bias.data = new_bias.to(linear.bias.dtype)

    @torch.no_grad()
    def _apply_emb_fuse(self, embedding, rotation, fast_mode=False):

        weight = embedding.weight.data
        rotation = rotation.to(device=weight.device, dtype=weight.dtype)
        embedding.weight.data = weight @ rotation

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
                self.R1 = hadamard_matrix(self.quant_model.model.config.hidden_size, "cuda")
            else:
                self.R1 = random_hadamard_matrix(self.quant_model.model.config.hidden_size, "cuda")

        self.R1_embed_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=([self.linear_mapping["embedding"]], self.ignore_layers),
        )
        # attn_o, mlp_out: output in hidden_size dim → W' = R1.T @ W → fuse_input=False, pass R1
        self.R1_linears = self.quant_model.get_rotation_mapping_layers(
            None,
            linear_mapping=(
                [self.linear_mapping["attn_o"]] + self.linear_mapping["mlp_out"],
                self.ignore_layers,
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
                self.ignore_layers,
            ),
        )

        if no_transform:
            return

        # embedding: W' = W @ R1
        for linear in self.R1_embed_linears.values():
            self._apply_emb_fuse(linear, self.R1)
        # attn_o, mlp_out: W' = R1.T @ W
        for linear in self.R1_linears.values():
            self._apply_linear_fuse(linear, self.R1, fuse_input=False)
        # q/k/v, mlp_in, lm_head: W' = W @ R1
        for linear in self.R1_inv_linears.values():
            self._apply_linear_fuse(linear, self.R1, fuse_input=True)

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

        # Pair entries that share the same prefix (same transformer block)
        def get_prefix(name: str) -> str:
            return name.rsplit(".", 1)[0] if "." in name else name

        paired = defaultdict(lambda: [None, None])  # [attn_v, attn_o]

        for k, v in self.R2_linears.items():
            prefix = get_prefix(k)
            paired[prefix][0] = v

        for k, v in self.R2_inv_linears.items():
            prefix = get_prefix(k)
            paired[prefix][1] = v

        self.R2_paired = dict(paired)

        cfg = self.quant_model.model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        num_q_heads = cfg.num_attention_heads

        if no_transform:
            return

        for name, (linear, inv_linear) in self.R2_paired.items():
            # Generate a single head_dim Hadamard block H, then tile for kv/q heads
            if self.spin_config.rotation_mode == "Hadamard":
                H = hadamard_matrix(head_dim, "cuda")
            else:
                H = random_hadamard_matrix(head_dim, "cuda")

            # R2_v: block-diagonal with num_kv_heads copies of H  → [v_total, v_total]
            R2_v = torch.block_diag(*([H] * num_kv_heads))
            # R2_o: block-diagonal with num_q_heads copies of H   → [o_total, o_total]
            R2_o = torch.block_diag(*([H] * num_q_heads))

            self._apply_linear_fuse(linear, R2_v, fuse_input=False)
            self._apply_linear_fuse(inv_linear, R2_o, fuse_input=True)

            # record R2 per layer (store the shared head block H)
            self.R2[name] = H

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
            linear_mapping=([self.linear_mapping["mlp_out"]], self.ignore_layers),
        )

        # check one matrix size
        R4 = hadamard_matrix(self.quant_model.model.config.intermediate_size, "cuda")
        R4_dict = {}
        R4_dict[self.quant_model.model.config.intermediate_size] = R4
        self.R4 = R4

        if no_transform:
            return

        for _, linear in self.R4_linears.items():
            # check size
            if linear.weight.shape[-1] not in R4_dict:
                rot = hadamard_matrix(linear.weight.shape[-1], "cuda")
                R4_dict[linear.weight.shape[-1]] = rot
            else:
                rot = R4_dict[linear.weight.shape[-1]]
            self._apply_linear_hook(linear, rot, hook_input=True)
            self._apply_linear_fuse(linear, rot, fuse_input=True)

    @torch.no_grad()
    def convert(self, R1=None, R2_list=None, R3_list=None, R4_list=None):
        """Fuse rotation matrices into weights after QAT training.

        Intended for use when hooks were registered during training (trainable mode).
        Call this after QAT training ends to fuse all online hooks into weights.
        """
        if R1 is not None:
            # embedding: W' = W @ R1  (same as _apply_emb_fuse in PTQ)
            for linear in self.R1_embed_linears.values():
                self._apply_emb_fuse(linear, self.R1)
            # attn_o, mlp_out: W' = R1.T @ W
            for linear in self.R1_linears.values():
                self._apply_linear_fuse(linear, self.R1, fuse_input=False)
            # q/k/v, mlp_in, lm_head: W' = W @ R1
            for linear in self.R1_inv_linears.values():
                self._apply_linear_fuse(linear, self.R1, fuse_input=True)

        if R2_list is not None:
            assert len(R2_list) == len(
                self.R2_paired
            ), "R2_list and R2_paired must have the same length"
            cfg = self.quant_model.model.config
            num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            num_q_heads = cfg.num_attention_heads
            for (_, (linear, inv_linear)), H in zip(self.R2_paired.items(), R2_list):
                # Tile the per-head block H into full block-diagonal matrices, same as PTQ
                R2_v = torch.block_diag(*([H] * num_kv_heads))
                R2_o = torch.block_diag(*([H] * num_q_heads))
                self._apply_linear_fuse(linear, R2_v, fuse_input=False)
                self._apply_linear_fuse(inv_linear, R2_o, fuse_input=True)

        if R3_list is not None:
            raise NotImplementedError("SpinQuant.convert R3 is not yet implemented.")

        if R4_list is not None:
            if len(R4_list) == 1:
                R4_list = R4_list * len(self.R4_linears)
            for (_, linear), R4 in zip(self.R4_linears.items(), R4_list):
                # Remove existing forward pre-hooks registered by _apply_r4
                linear._forward_pre_hooks.clear()
                # Fuse R4 into weight: W' = W @ R4  (same as PTQ in _apply_r4)
                self._apply_linear_fuse(linear, R4, fuse_input=True)

    @torch.no_grad()
    def save(self):
        """Save the model with the applied rotations."""
        pass

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
