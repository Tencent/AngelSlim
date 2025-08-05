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


import gc

import torch
import torch.nn as nn

from .....utils import get_best_device, print_info
from ...modules.catcher import Catcher

__all__ = ["FP8"]


class FP8:
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
            model(nn.Module, required): The model to be smoothed.
            seq_length(int, optional): The length of the sequence. Default: 2048.
            hidden_size(int, optional): The size of the hidden layer. Default: 2560.
            model_arch_type(str, optional): model arch type.Default: None.
            low_memory(boll, optional): using low memory .Default: None.
        """
        super(FP8, self).__init__()
        self.model = model
        self.modal_type = self.model.modal_type
        if self.modal_type == "VLM":
            self.layers = self.model.model.model.language_model.layers
        else:
            self.layers = self.model.model.model.layers
        self.quant_bits = self.model.quant_config.quant_bit
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.model_arch_type = model_arch_type
        self.low_memory = low_memory
        self.dtype = torch.bfloat16
        torch.set_default_dtype(self.dtype)
        self.scales_dict = {}
        self.inps = None

    def move_embed(self, model, device: str):
        print_info(model)
        model.model.model.embed_tokens = model.model.model.embed_tokens.to(device)
        model.model.model.rotary_emb = model.model.model.rotary_emb.to(device)

    @torch.no_grad()
    def run(self, dataloader):
        if self.low_memory:
            print_info("Use FP8 low memory run")
            assert (
                str(next(self.model.model.parameters()).device) == "cpu"
            ), "[AngelSlim Error] FP8 low memory mode need model in cpu"
            self.low_memory_run(dataloader)
        else:
            print_info("Use FP8 fast forward")
            self.model.model_forward(dataloader)

    def low_memory_run(self, dataloader):
        for model_module in self.layers:
            model_module.eval()
        layers = self.layers
        dev = "cpu"
        nsamples = len(dataloader) * dataloader.batch_size
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
            if isinstance(v, torch.Tensor):
                layer_kwargs[k] = v.to(dev)

        print_info("cache['i']:{}".format(cache["i"]))
        print_info(len(layers))
        layers[0] = layers[0].module
        print_info(self.inps.shape)
        # begin the FP8 process
        print_info("Ready.")
        layers = layers.cpu()
        self.inps = self.inps.to("cpu")
        bs = dataloader.batch_size

        for i in range(0, nsamples, bs):
            inps = self.inps[i : i + bs, :, :].to(dev)
            outs = torch.zeros_like(inps).to(dev)
            for j in range(len(layers)):
                layer = layers[j].to(dev)
                with torch.no_grad():
                    outs = layer(hidden_states=inps, **layer_kwargs)
                    if isinstance(outs, tuple):
                        outs = outs[0].squeeze(1)

                if torch.cuda.is_available():
                    print_info(
                        f"GPU Memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB"
                    )
                # Clear GPU memory
                torch.cuda.empty_cache()

                layers[j] = layers[j].cpu()
                inps, outs = outs, inps
                print_info("FP8 end layer {}\n".format(j))
            print_info("FP8 end batch {}\n".format(i // bs))
        del inps, outs
        self.inps = None
        gc.collect()
        torch.cuda.empty_cache()
