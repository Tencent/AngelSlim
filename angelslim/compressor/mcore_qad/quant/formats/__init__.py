"""Format registry + concrete formats. Importing registers them."""

from angelslim.compressor.mcore_qad.quant.formats import (  # noqa: F401  (register side-effects)
    fp4,
    fp8,
    integer,
)
from angelslim.compressor.mcore_qad.quant.formats.base import (
    FORMAT_REGISTRY,
    QuantFormat,
)

__all__ = ["FORMAT_REGISTRY", "QuantFormat"]
