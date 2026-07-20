"""Inject fake-quant into a built mcore GPTModel without reimplementing its forward.

Strategy (works for any mcore Column/RowParallelLinear, incl. per-expert experts):
  * WEIGHT: register a parametrization on `weight` so `module.weight` transparently
    returns fake-quant(weight). mcore's TP-aware forward then uses the quantized
    weight; gradients flow to the quantizer scale. The original (frozen) weight is
    kept by the parametrization.
  * ACTIVATION: a forward_pre_hook fake-quants the input before the matmul.

Router and output_layer are skipped (kept high precision). Because experts are
per-expert Column/RowParallelLinear modules, each expert gets its own scales.
"""

from __future__ import annotations

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from torch import nn
from torch.nn.utils import parametrize

from angelslim.compressor.mcore_qad.parallel.scale_parallel import (
    configure_scale_parallelism,
)
from angelslim.compressor.mcore_qad.quant.policy import (
    build_quantizer,
    weight_scale_shape,
)
from angelslim.compressor.mcore_qad.quant.quantizer import IdentityQuantizer
from angelslim.compressor.mcore_qad.quant.schemes.base import HostInfo
from angelslim.compressor.mcore_qad.quant.spec import QuantSpec

DEFAULT_SKIP = ("router", "output_layer")


class WeightFakeQuant(nn.Module):
    """Parametrization: maps the stored (frozen) weight -> fake-quantized weight."""

    def __init__(self, quantizer: nn.Module) -> None:
        super().__init__()
        self.quantizer = quantizer

    def forward(self, weight):
        return self.quantizer(weight.float()).to(weight.dtype)


def _act_pre_hook(module, args):
    aq = module._mcore_qad_act_q
    x = args[0]
    return (aq(x.float()).to(x.dtype),) + tuple(args[1:])


def inject_quant(
    module: nn.Module,
    parallel_mode: str,
    weight_spec: QuantSpec,
    act_spec: QuantSpec,
    is_expert: bool = False,
) -> None:
    """Attach fake-quant to ONE mcore Column/RowParallelLinear (the single injection point).

    Freezes the weight, adds a weight parametrization (fake-quant) + an activation
    forward-pre-hook, and tags the scales for parallel sync. Sole injection point used
    by the model-wide quantize_mcore_model pass.
    """
    for p in module.parameters(recurse=False):
        p.requires_grad_(False)
    device = module.weight.device
    sp = bool(getattr(getattr(module, "config", None), "sequence_parallel", False))
    host = HostInfo(
        parallel_mode=parallel_mode,
        shard_dim=1 if parallel_mode == "row" else 0,
        is_expert=is_expert,
        sequence_parallel=sp,
        context_parallel=False,
    )

    wq = build_quantizer(
        weight_spec,
        "weight",
        host,
        quant_shape=weight_scale_shape(weight_spec, module.weight.shape),
    ).to(device)
    if hasattr(wq, "scheme"):
        configure_scale_parallelism(wq, wq.scheme.parallel_spec(host))
    parametrize.register_parametrization(module, "weight", WeightFakeQuant(wq))

    aq = build_quantizer(act_spec, "act", host)
    if not isinstance(aq, IdentityQuantizer):
        aq = aq.to(device)
        configure_scale_parallelism(aq, aq.scheme.parallel_spec(host))
        module.add_module("_mcore_qad_act_q", aq)
        module.register_forward_pre_hook(_act_pre_hook)


def quantize_mcore_model(
    model: nn.Module, weight_spec: QuantSpec, act_spec: QuantSpec, skip_substr=DEFAULT_SKIP
) -> int:
    """In-place: freeze weights and inject fake-quant into every eligible linear.

    Router and output_layer are skipped (high precision). Returns the count.
    """
    model.requires_grad_(False)
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, (ColumnParallelLinear, RowParallelLinear)):
            continue
        if any(s in name for s in skip_substr):
            continue
        mode = "row" if isinstance(mod, RowParallelLinear) else "column"
        inject_quant(mod, mode, weight_spec, act_spec, is_expert=("experts" in name))
        n += 1
    return n


def collect_scale_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
