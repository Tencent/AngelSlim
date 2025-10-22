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

import copy
import re
from typing import List, Optional, Tuple, Union

import torch
import tqdm

from .modules import FP8DynamicLinear
from .quant_func import (
    fp8_per_block_quant,
    fp8_per_tensor_quant,
    fp8_per_token_group_quant,
)
from .utils import (
    QuantType,
    _ensure_deep_gemm,
    cleanup_memory,
    replace_module,
    should_quantize_layer,
)

__all__ = ["DynamicDiTQuantizer"]


class DynamicDiTQuantizer:
    """
    Quantizer for DiT that supports various FP8 quantization strategies.
    """

    def __init__(
        self,
        quant_type: str = QuantType.FP8_PER_TENSOR,
        layer_filter: Optional[callable] = None,
        include_patterns: Optional[List[Union[str, re.Pattern]]] = None,
        exclude_patterns: Optional[List[Union[str, re.Pattern]]] = None,
        native_fp8_support: Optional[bool] = None,
    ):
        """
        Args:
            quant_type: Choose from 'fp8-per-tensor', 'fp8-per-token', 'fp8-per-block'.
            layer_filter: Custom function to decide which layer names to quantize.
            include_patterns: List of keywords or regex to include layers.
            exclude_patterns: List of keywords or regex to exclude layers.
            native_fp8_support: Whether to use FP8-accelerated kernels if available.
        """
        QuantType.validate(quant_type)

        self.quant_type = quant_type
        self.include_patterns = (
            include_patterns
            if include_patterns is not None
            else ["wrapped_module", "block", "lin", "img", "txt"]
        )
        self.exclude_patterns = (
            exclude_patterns if exclude_patterns is not None else ["embed"]
        )

        # Layer filter: callable for layer name selection
        self.layer_filter = (
            layer_filter
            if layer_filter is not None
            else lambda name: should_quantize_layer(
                name, self.include_patterns, self.exclude_patterns
            )
        )

        # Auto-enable FP8 if native device supports it (Ampere+)
        if native_fp8_support is not None:
            self.native_fp8_support = native_fp8_support
        else:
            self.native_fp8_support = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability() >= (9, 0)
            )

        self.quantize_linear_module = self._set_quantize_linear_module()

    def _set_quantize_linear_module(self) -> torch.nn.Module:
        """
        Returns the quantized module type to replace nn.Linear.
        """
        if "fp8" in self.quant_type:
            return FP8DynamicLinear
        else:
            raise ValueError(f"Invalid quant_type: {self.quant_type}")

    def _quantize_linear_weight(
        self, linear: torch.nn.Linear
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the given Linear layer's weight according to the selected quant_type.

        Returns:
            quant_weight: Quantized weight (FP8 format)
            weight_scale: Scale used for quantization
        """
        if self.quant_type == QuantType.FP8_PER_TENSOR:
            quant_weight, weight_scale = fp8_per_tensor_quant(linear.weight)
        elif self.quant_type == QuantType.FP8_PER_TOKEN:
            quant_weight, weight_scale = fp8_per_token_group_quant(
                linear.weight, linear.weight.shape[-1]
            )
            weight_scale = weight_scale.t()  # match broadcast order for inference
        elif self.quant_type == QuantType.FP8_PER_BLOCK:
            if self.native_fp8_support:
                # Enable native accelerated FP8 kernels if available.
                _ensure_deep_gemm()
            quant_weight, weight_scale = fp8_per_block_quant(linear.weight)
        else:
            raise ValueError(f"Invalid quant_type: {self.quant_type}")
        return quant_weight, weight_scale

    def quantize(self, model: torch.nn.Module):
        """
        Quantize all eligible nn.Linear modules in the input model in-place.
        """
        # Use bfloat16 for better kernel compatibility and less memory pressure
        model.to(torch.bfloat16)

        named_modules = list(model.named_modules())

        # Quantize eligible Linear modules
        for name, module in tqdm.tqdm(named_modules, desc="Quantizing weights"):
            if isinstance(module, torch.nn.Linear) and self.layer_filter(name):
                quant_weight, weight_scale = self._quantize_linear_weight(module)
                # Deep copy bias to detach from original module
                bias = copy.deepcopy(module.bias) if module.bias is not None else None

                # Instantiate quantized FP8 linear layer
                quant_linear = self.quantize_linear_module(
                    weight=quant_weight,
                    weight_scale=weight_scale,
                    bias=bias,
                    native_fp8_support=self.native_fp8_support,
                    quant_type=self.quant_type,
                )

                # Replace Linear with Quantized Linear in the model
                replace_module(model, name, quant_linear)

                # Cleanup: Explicitly delete reference to old weights (accelerate GC)
                del module.weight
                del module.bias
                del module

        cleanup_memory()
