#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright 2024 the tencent authors.
#

from dataclasses import dataclass

import torch

__all__ = ["SmoothQuant"]


@dataclass
class SmoothConfig:
    alpha: float = 0.5
    smooth_first_linears: bool = True
    smooth_second_linears: bool = False


class SmoothQuant:
    def __init__(
        self,
        quant_model,
        ptq_hook,
        alpha=0.5,
        smooth_first_linears=True,
        smooth_second_linears=False,
    ):
        """
        SmoothConfig:
            alpha: smoothing parameter. Default: 0.5
            smooth_first_linears: whether to smooth qkv and ffn1 linears.
            smooth_second_linears: whether to smooth out and ffn2 linears.
        """
        self.quant_model = quant_model
        self.ptq_hook = ptq_hook
        self.smooth_config = SmoothConfig(alpha, smooth_first_linears, smooth_second_linears)
        self.apply_hook()

    def apply_hook(self):
        # smooth_mapping_layers: smooth_name: (smooth_layer, balance_layers)
        self.smooth_mapping_layers = self.quant_model.get_smooth_mapping_layers(self.smooth_config)
        smooth_observer = self.quant_model.quant_algo_dict["smooth_observer"]
        self.ptq_hook.apply_smooth_hook(self.smooth_mapping_layers, smooth_observer)

    @torch.no_grad()
    def convert(self):
        for smooth_name, (
            smooth_layer,
            balance_layers,
        ) in self.smooth_mapping_layers.items():
            assert hasattr(self.ptq_hook.observer_dict[smooth_layer], "smooth_act_observer")
            smooth_scale = self.ptq_hook.observer_dict[smooth_layer].smooth_act_observer.scales()

            balance_scales = []
            for _, balance_layer in balance_layers:
                balance_scale = balance_layer.weight.abs().max(dim=0, keepdim=True)[0]
                balance_scales.append(balance_scale)
            balance_scales = torch.cat(balance_scales, dim=0).max(dim=0)[0]

            # s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
            smooth_scale = smooth_scale.to(balance_scales.device)
            scales = smooth_scale.pow(self.smooth_config.alpha) / balance_scales.pow(
                1 - self.smooth_config.alpha
            )

            for _, balance_layer in balance_layers:
                balance_layer.weight.mul_(scales.view(1, -1))

            if smooth_layer.weight.ndim == 1:
                smooth_layer.weight.div_(scales)
            elif smooth_layer.weight.ndim == 2:
                smooth_layer.weight.div_(scales.view(-1, 1))
            else:
                raise ValueError(f"{smooth_name} {smooth_layer.weight.shape} not supported")
            if hasattr(smooth_layer, "bias") and smooth_layer.bias is not None:
                smooth_layer.bias.div_(scales)
