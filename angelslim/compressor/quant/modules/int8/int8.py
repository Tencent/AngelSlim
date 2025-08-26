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


import torch
import torch.nn as nn

from .....utils import get_best_device, print_info
from ...modules.catcher import Catcher

__all__ = ["INT8"]


class INT8:
    def __init__(
        self,
        model,
        seq_length=2048,
        hidden_size=2560,
        model_arch_type=None,
        low_memory=False,
    ):
        """
        Args:
            model(nn.Module, required): The model to be quanted.
            seq_length(int, optional): The length of the sequence. Default: 2048.
            hidden_size(int, optional): The size of the hidden layer. Default: 2560.
            model_arch_type(str, optional): model arch type.Default: None.
            low_memory(boll, optional): using low memory .Default: None.
        """
        super(INT8, self).__init__()
        self.model = model
        self.modal_type = self.model.modal_type
        self.layers = self.model.get_quant_module()
        self.quant_bits = self.model.quant_config.quant_bit
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.model_arch_type = model_arch_type
        self.low_memory = low_memory
        self.dtype = torch.bfloat16
        self.scales_dict = {}
        self.inps = None

    def move_embed(self, model, device: str):
        print_info(model)
        model.model.model.embed_tokens = model.model.model.embed_tokens.to(device)
        model.model.model.rotary_emb = model.model.model.rotary_emb.to(device)

    @torch.no_grad()
    def run(self, dataloader):
        if self.low_memory:
            print_info("Use INT8 low memory run")
            assert (
                str(next(self.model.model.parameters()).device) == "cpu"
            ), "[AngelSlim Error] INT8 low memory mode need model in cpu"
            self.low_memory_run(dataloader)
        else:
            print_info("[AngelSlim] Use INT8 fast forward")
            self.model.model_forward(dataloader)

    def low_memory_run(self, dataloader):
        for model_module in self.layers:
            model_module.eval()
        layers = self.layers
        dev = "cpu"
        nsamples = len(dataloader)
        print_info(f"nsamples:{nsamples}")
        self.inps = torch.zeros(
            (int(nsamples), self.seq_length, self.hidden_size),
            device=dev,
            dtype=self.dtype,
        )
        cache = {"i": 0}
        layers[0] = layers[0].to(dev)
        self.model.model.model.embed_tokens = self.model.model.model.embed_tokens.to(
            dev
        )
        layers[0] = Catcher(layers[0], self.inps, cache)
        self.model.model_forward(dataloader)
        layer_kwargs = layers[0].layer_kwargs
        dev = get_best_device()
        for k, v in layer_kwargs.items():
            # position embeddings
            if isinstance(v, tuple):
                layer_kwargs[k] = tuple(
                    (
                        item.to(dev)
                        if isinstance(item, (torch.Tensor, nn.Module))
                        else item
                    )
                    for item in v
                )

        print_info("cache['i']:{}".format(cache["i"]))
        print_info(len(layers))
        layers[0] = layers[0].module
        print_info(self.inps.shape)
        outs = torch.zeros_like(self.inps)
        # begin the INT8 process
        print_info("Ready.")
        layers = layers.cpu()
        torch.cuda.empty_cache()

        outs = outs.to("cpu")
        self.inps = self.inps.to("cpu")
        for i in range(len(layers)):
            if torch.cuda.is_available():
                print_info(
                    f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
                )

            layer = layers[i].to(dev)
            outs = outs.to(dev)
            self.inps = self.inps.to(dev)
            # being hook
            for j in range(min(self.inps.shape[0], nsamples)):
                with torch.no_grad():
                    outs[j, :, :] = layer(
                        hidden_states=self.inps[j, :, :].unsqueeze(0), **layer_kwargs
                    )[0].squeeze(1)

            print_info("HOOK Step{}".format(j))

            # Clear GPU memory
            torch.cuda.empty_cache()

            layers[i] = layers[i].cpu()
            layer = layer.cpu()
            torch.cuda.empty_cache()
            self.inps, outs = outs, self.inps
            print_info("INT8 end layer {}\n".format(i))
