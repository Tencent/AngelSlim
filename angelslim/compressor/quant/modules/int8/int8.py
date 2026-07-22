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
        layers[0] = layers[0].to(dev)
        self.model.model.model.embed_tokens = self.model.model.model.embed_tokens.to(dev)
        layers[0] = Catcher(layers[0], max_seq_length=self.seq_length)
        self.model.model_forward(dataloader)
        # Catcher stores per-sample captures; rebuild the fixed-shape inps
        # tensor and a single layer_kwargs dict to match the previous API.
        captured_inputs = layers[0].captured_inputs
        captured_kwargs = layers[0].captured_kwargs
        for idx in range(min(len(captured_inputs), self.inps.shape[0])):
            inp = captured_inputs[idx]
            self.inps[idx, : inp.shape[1], :].copy_(inp[0])
        layer_kwargs = captured_kwargs[0] if captured_kwargs else {}
        # ``prev_topk_indices`` is a DSA-specific cross-layer state.  The HF
        # top-level forward passes ``prev_topk_indices=None`` into layer 0
        # via kwargs, so ``Catcher`` captures it into ``layer_kwargs``.
        # We re-inject a per-sample value below, so we must remove the
        # captured one to avoid "got multiple values for keyword argument
        # 'prev_topk_indices'".  Do the same for a few other DSA/MoE
        # per-layer scratch kwargs that must not be shared across layers.
        for _k in (
            "prev_topk_indices",
            "topk_indices",
            "router_logits",
            "output_router_logits",
        ):
            layer_kwargs.pop(_k, None)
        dev = get_best_device()
        for k, v in layer_kwargs.items():
            # Move every tensor/module kwarg to ``dev`` (cuda:0 in low_memory).
            # The previous branch only handled ``tuple``-valued kwargs, which
            # silently left plain ``torch.Tensor`` kwargs (e.g. position_ids)
            # on CPU and triggered a device-mismatch inside the GLM-5 DSA
            # indexer (key_positions on cuda:0 vs position_ids on cpu).
            if isinstance(v, (torch.Tensor, nn.Module)):
                layer_kwargs[k] = v.to(dev)
            elif isinstance(v, tuple):
                layer_kwargs[k] = tuple(
                    (item.to(dev) if isinstance(item, (torch.Tensor, nn.Module)) else item)
                    for item in v
                )

        print_info("captured samples: {}".format(len(captured_inputs)))
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

        # GLM-5 DSA (and other DeepSeek-Sparse-Attention models) route the
        # top-k selection through a cross-layer ``prev_topk_indices`` state:
        # a "full" indexer layer computes the top-k and the following
        # "shared" layers reuse it.  Because ``low_memory_run`` drives the
        # transformer blocks one-by-one (instead of through HF's top-level
        # forward), we must carry that state across layers (and across
        # samples) by hand — otherwise the first ``shared`` layer raises
        # "Shared DSA layers require top-k indices from a previous full
        # indexer layer.".  Detect the DSA layout once from the layer type.
        carry_topk = (
            len(layers) > 0
            and hasattr(layers[0], "self_attn")
            and hasattr(layers[0].self_attn, "indexer")
        )
        prev_topk_indices = [None] * nsamples if carry_topk else None

        for i in range(len(layers)):
            if torch.cuda.is_available():
                print_info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

            layer = layers[i].to(dev)
            outs = outs.to(dev)
            self.inps = self.inps.to(dev)
            # being hook
            for j in range(min(self.inps.shape[0], nsamples)):
                with torch.no_grad():
                    out = layer(
                        hidden_states=self.inps[j, :, :].unsqueeze(0),
                        **(
                            {"prev_topk_indices": prev_topk_indices[j]}
                            if carry_topk
                            else {}
                        ),
                        **layer_kwargs,
                    )
                    outs[j, :, :] = out[0].squeeze(1)
                    if carry_topk:
                        # [0]=hidden_states, [1]=topk_indices (to feed the next
                        # shared DSA layer).  Keep it on ``dev`` for the next iter.
                        prev_topk_indices[j] = out[1]

            print_info("HOOK Step{}".format(j))

            # Clear GPU memory
            torch.cuda.empty_cache()

            layers[i] = layers[i].cpu()
            layer = layer.cpu()
            torch.cuda.empty_cache()
            self.inps, outs = outs, self.inps
            print_info("INT8 end layer {}\n".format(i))
