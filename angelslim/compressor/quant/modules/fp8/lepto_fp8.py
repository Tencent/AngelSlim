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

import functools
from collections import defaultdict

import torch
import torch.nn as nn

from .....utils import find_parent_layer_and_sub_name, get_best_device, print_info
from ...core.quant_func import get_fp_maxval
from ...modules.catcher import Catcher
from .lepto_scale import AutoLayerScale

__all__ = ["LeptoFP8"]


class LeptoFP8:
    def __init__(
        self,
        ptq_hook,
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
        super(LeptoFP8, self).__init__()
        self.ptq_hook = ptq_hook
        self.quant_model = model  # self.quant_model
        self.modal_type = self.quant_model.modal_type
        self.layers = self.quant_model.get_quant_module()
        self.quant_bits = self.quant_model.quant_config.quant_bit
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.model_arch_type = model_arch_type
        self.low_memory = low_memory
        self.dtype = torch.bfloat16
        self.scales_dict = {}
        self.inps = None
        self.observer_layer_classes = [nn.Linear]
        self.scale_function = AutoLayerScale(
            model_type=self.model_arch_type,
            observer_layer_classes=self.observer_layer_classes,
        )

    def move_embed(self, model, device: str):
        print_info(model)
        model.model.model.embed_tokens = model.model.model.embed_tokens.to(device)
        model.model.model.rotary_emb = model.model.model.rotary_emb.to(device)

    @torch.no_grad()
    def run(self, dataloader):
        for model_module in self.layers:
            model_module.eval()
        layers = self.layers
        dev = get_best_device()
        nsamples = len(dataloader)
        self.inps = torch.zeros(
            (int(nsamples), self.seq_length, self.hidden_size),
            device=dev,
            dtype=self.dtype,
        )
        cache = {"i": 0}
        layers[0] = layers[0].to(dev)
        self.quant_model.model.model.embed_tokens = (
            self.quant_model.model.model.embed_tokens.to(dev)
        )
        layers[0] = Catcher(layers[0], self.inps, cache)
        self.quant_model.model_forward(dataloader)
        layer_kwargs = layers[0].layer_kwargs
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
        # begin the lepto process
        print_info("Ready.")
        layers = layers.cpu()
        torch.cuda.empty_cache()

        outs = outs.to("cpu")
        self.inps = self.inps.to("cpu")
        print_info(layer_kwargs)

        for i in range(len(layers)):
            if torch.cuda.is_available():
                print_info(
                    f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
                )

            layer = layers[i].to(dev)
            outs = outs.to(dev)
            self.inps = self.inps.to(dev)
            subset = self._find_layers(layer)

            if self.model_arch_type == "qwen3_moe":
                subset = {
                    **subset,
                    "mlp": layer.mlp,
                }

            # firstly, get input features of all linear layers
            def cache_input_hook(m, x, y, name, feat_dict, layer):
                x = x[0]
                x = x.detach().cpu()
                feat_dict[name].append(x)

            input_feat = defaultdict(list)
            handles = []
            for name in subset:
                handles.append(
                    subset[name].register_forward_hook(
                        functools.partial(
                            cache_input_hook,
                            name=name,
                            feat_dict=input_feat,
                            layer=subset[name],
                        )
                    )
                )
            # being hook
            for j in range(min(self.inps.shape[0], nsamples)):
                with torch.no_grad():
                    outs[j, :, :] = layer(
                        hidden_states=self.inps[j, :, :].unsqueeze(0), **layer_kwargs
                    )[0].squeeze(1)

            # remove duplicate
            def deduplicate_tensors(tensor_list):
                unique_tensors = []
                assert len(tensor_list) % 2 == 0
                for i in range(int(len(tensor_list) / 2)):
                    if torch.equal(tensor_list[i * 2], tensor_list[i * 2 + 1]):
                        unique_tensors.append(tensor_list[i * 2])
                    else:
                        raise ValueError
                for tensor in tensor_list:
                    if not any(torch.equal(tensor, t) for t in unique_tensors):
                        unique_tensors.append(tensor)
                return unique_tensors

            for k, v in input_feat.items():
                if len(v) > nsamples:
                    print_info(f"Warning: repetition hook {k}")
                    input_feat[k] = deduplicate_tensors(v)

            print_info("HOOK Step{}".format(j))
            for h in handles:
                h.remove()

            # now solve for scaling and clipping
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

            # Clear GPU memory
            torch.cuda.empty_cache()

            scales_list = self.scale_function.auto_scale(
                self.ptq_hook, layer, input_feat, layer_kwargs
            )

            for scales in scales_list:
                for kn in scales[0]:
                    name = "model.layers.{}.{}".format(i, kn)
                    self.scales_dict[name] = scales[1]

            layers[i] = layers[i].cpu()
            layer = layer.cpu()
            torch.cuda.empty_cache()
            self.inps, outs = outs, self.inps
            print_info("LEPTO FP8 end layer {}\n".format(i))

        print_info(self.scales_dict)

    def _find_layers(self, module, layers=None, name=""):
        if not layers:
            layers = self.observer_layer_classes
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(
                self._find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res

    def convert(self):
        # 1. get act, weight and kv-cache scale
        old_list = []
        new_list = []
        for name, sub_layer in self.ptq_hook.quant_layers_dict.items():
            weight_scales = self.quant_model.get_weight_scales(
                sub_layer, self.ptq_hook.observer_dict[sub_layer].weight_observer
            )

            self.quant_model.weight_scales_dict[name] = weight_scales / get_fp_maxval(
                bits=8
            ).type(weight_scales.dtype)
            old_scale = self.ptq_hook.observer_dict[sub_layer].act_observer.scales()
            lepto_scale = torch.clamp(
                self.scales_dict.pop(name).squeeze().detach().to(old_scale.device),
                min=0,
                max=99999,
            )

            self.quant_model.act_scales_dict[name] = lepto_scale
            print_info(
                f"{name} , {old_scale}, "
                f"{old_scale / get_fp_maxval(bits=8).type(weight_scales.dtype).item()} "
                f"{lepto_scale.item()}"
            )
            old_list.append(old_scale / get_fp_maxval(bits=8).type(weight_scales.dtype))
            new_list.append(self.quant_model.act_scales_dict[name])
        print_info(sum(old_list))
        print_info(sum(new_list))
        self.ptq_hook.remove_hook()
        torch.cuda.empty_cache()

        # 2. insert qdq module
        quant_convert_module = self.quant_model.get_quant_convert_module()
        for name, sub_layer in self.ptq_hook.quant_layers_dict.items():
            parent_layer, sub_name = find_parent_layer_and_sub_name(
                quant_convert_module, name
            )

            qdq_module = self.quant_model.get_qdq_module(sub_layer, name)
            setattr(parent_layer, sub_name, qdq_module)
        self.quant_model.quantized = True
