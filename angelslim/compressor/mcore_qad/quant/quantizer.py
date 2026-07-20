"""The single composition point: Quantizer = Format x ScaleScheme.

The ScaleScheme owns its scale storage (ScaleStores), so a Quantizer is just a
(format, scheme) pair plus the global on/off switch the distillation engine uses
to build the teacher (quant-off) path.
"""

from __future__ import annotations

import torch
from torch import Tensor

from angelslim.compressor.mcore_qad.quant.formats.base import QuantFormat
from angelslim.compressor.mcore_qad.quant.schemes.base import ScaleScheme


class Quantizer(torch.nn.Module):
    def __init__(self, fmt: QuantFormat, scheme: ScaleScheme) -> None:
        super().__init__()
        self.fmt = fmt
        self.scheme = scheme  # nn.Module -> learnable scales auto-registered
        self.enabled: bool = True  # toggled by distill.switch.quant_disabled()

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled or x.numel() == 0:
            return x  # teacher / bypass / empty-expert-input
        return self.scheme.quantize(x, self.fmt)


class IdentityQuantizer(torch.nn.Module):
    """No-op quantizer, used where a role is configured to stay high precision."""

    enabled = False

    def forward(self, x: Tensor) -> Tensor:
        return x
