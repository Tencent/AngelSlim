"""build_quantizer factory: turn a declarative QuantSpec into a live Quantizer.

A QuantSpec names a Format x ScaleScheme x ScaleSource; this composes them (after
rejecting specs vLLM cannot load) into a Quantizer that fake-quantizes one tensor.
``weight_scale_shape`` reports the scale-store shape a learnable/calibrated weight needs.
"""

from __future__ import annotations

from angelslim.compressor.mcore_qad.quant.formats import FORMAT_REGISTRY
from angelslim.compressor.mcore_qad.quant.quantizer import IdentityQuantizer, Quantizer
from angelslim.compressor.mcore_qad.quant.schemes import SCHEME_REGISTRY, HostInfo
from angelslim.compressor.mcore_qad.quant.schemes.per_expert import PerExpertScheme
from angelslim.compressor.mcore_qad.quant.spec import QuantSpec


def weight_scale_shape(spec: QuantSpec, wshape):
    """Learnable/calibrated weight scale store shape for an N-D weight (None if dynamic)."""
    if spec.is_identity() or spec.source == "dynamic":
        return None
    wshape = tuple(wshape)
    g = spec.group_size or 16
    if spec.scheme in ("two_level_block", "per_group"):
        return wshape[:-1] + (wshape[-1] // g,)
    if spec.scheme == "per_channel":
        return wshape[:-1]
    if spec.scheme == "per_tensor":
        return ()
    return None


def _build_scheme(spec: QuantSpec, quant_shape):
    """Instantiate the (possibly per-expert-wrapped) ScaleScheme from a spec."""
    kwargs: dict = {"source": spec.source}
    if spec.axis is not None:
        kwargs["axis"] = spec.axis
    if spec.group_size is not None:
        kwargs["group_size"] = spec.group_size
    if spec.block_scale_fmt is not None:
        kwargs["block_scale_fmt"] = spec.block_scale_fmt
    if spec.scheme in ("two_level_block", "per_group"):
        kwargs["block_shape"] = quant_shape
    if spec.scheme == "per_channel":
        kwargs["channel_shape"] = quant_shape

    inner = SCHEME_REGISTRY.create(spec.scheme, **kwargs)
    if spec.per_expert:
        return PerExpertScheme(inner=inner)
    return inner


def build_quantizer(spec: QuantSpec, role: str, host: HostInfo, quant_shape=None):
    """Compose a Quantizer (or IdentityQuantizer) from a spec.

    ``quant_shape`` is the shape of the scale-bearing (non-token) dims, required
    for learnable/calibrated weight scales; dynamic activation scales omit it.
    """
    if spec.is_identity():
        return IdentityQuantizer()
    from angelslim.compressor.mcore_qad.quant.vllm_formats import validate_spec

    validate_spec(spec, role)
    fmt = FORMAT_REGISTRY.create(spec.fmt)
    scheme = _build_scheme(spec, quant_shape)
    q = Quantizer(fmt, scheme)
    # NOTE: parallel placement (configure_scale_parallelism with scheme.parallel_spec(host))
    # is attached by the caller AFTER moving the quantizer to the device
    # (see mcore.quantize.quantize_mcore_model and modules.quant._inject).
    return q
