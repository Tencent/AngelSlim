import torch
import torch.nn as nn

from .....utils import find_layers
from ...core import DeepSeekV3W4A8Int8Save, weight_dequant
from ...modules.helper_layer import W4A8Int8QuantLinear
from ..gptq.gptq import GPTQ


class W4A8INT8(GPTQ):
    """GPTQ runner with FP8 -> BF16 pre-processing and W4A8 INT8 linear."""

    def __init__(
        self,
        model,
        seq_length: int = 2048,
        hidden_size: int = 2560,
        sym: bool = True,
        actorder: bool = True,
    ):
        super().__init__(model, seq_length, hidden_size, sym, actorder)
        self.dtype = torch.bfloat16
        self.quant_linear_cls = W4A8Int8QuantLinear

    @torch.no_grad()
    def run(self, dataloader):
        self._prepare_fp8_weights()
        torch.cuda.empty_cache()
        super().run(dataloader)

    @torch.no_grad()
    def _prepare_fp8_weights(self):
        for layer in self.layers:
            subset = find_layers(layer, layers=self.model.observer_layer_classes)
            for sub_layer in subset.values():
                self._maybe_dequant_fp8(sub_layer)
                sub_layer.to("cpu")

    @staticmethod
    def _maybe_dequant_fp8(module: nn.Module):
        if (
            hasattr(module, "weight")
            and module.weight.dtype == torch.float8_e4m3fn
            and hasattr(module, "weight_scale_inv")
        ):
            module.weight = nn.Parameter(
                weight_dequant(module.weight, module.weight_scale_inv),
                requires_grad=False,
            )
            module.weight_scale_inv = None

    def save(self, save_dir: str):
        saver = DeepSeekV3W4A8Int8Save(self.model)
        saver.save(save_dir)
