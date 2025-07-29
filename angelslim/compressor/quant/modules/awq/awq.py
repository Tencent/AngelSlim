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
import os
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from huggingface_hub import save_torch_state_dict
from tqdm import tqdm

from .....utils import get_best_device, print_info, set_op_by_name
from ...core import pseudo_quantize_tensor
from ...modules.catcher import Catcher
from ...modules.helper_layer import WQLinearGEMM
from .auto_clip import AutoLayerClip
from .auto_scale import AutoLayerScale

__all__ = ["AWQ"]


class AWQ:
    def __init__(
        self,
        model,
        seq_length=2048,
        hidden_size=2560,
        mse_range=False,
        model_arch_type=None,
        observer_layer_classes=None,
        smooth_all_linears=False,
        merge_samples=False,
    ):
        """
        Args:
            model(nn.Module, required): The model to be smoothed.
            nsamples(int, optional): The number of samples to be used for AWQ.
            seq_length(int, optional): The length of the sequence. Default: 2048.
            hidden_size(int, optional): The size of the hidden layer. Default: 2560.
            mse_range(bool, optional): Whether to use mse_range.
            model_arch_type(str, optional): model arch type.
            observer_layer_classes(list, optional): The layer need to observer.
            smooth_all_linears(bool, optional): Whether to smooth all linears.
            merge_samples(bool, optional): Whether to merge samples. Default: False.
        """
        super(AWQ, self).__init__()
        self.model = model
        self.modal_type = self.model.modal_type
        if self.modal_type == "VLM":
            self.layers = self.model.model.model.language_model.layers
        else:
            self.layers = self.model.model.model.layers
        self.quant_bits = self.model.quant_config.quant_bit
        self.group_size = self.model.quant_config.quant_algo_info["group_size"]
        self.zero_point = self.model.quant_config.quant_algo_info["zero_point"]
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.model_arch_type = model_arch_type
        self.mse_range = mse_range
        self.observer_layer_classes = observer_layer_classes
        self.smooth_all_linears = smooth_all_linears
        self.merge_samples = merge_samples
        self._init_linear_list_and_model_type()
        self.scale_function = AutoLayerScale(
            weight_bits=self.quant_bits,
            group_size=self.group_size,
            smooth_all_linears=smooth_all_linears,
            merge_samples=merge_samples,
            model_type=self.model_arch_type,
            observer_layer_classes=observer_layer_classes,
        )
        self.clip_function = AutoLayerClip(
            weight_bits=self.quant_bits,
            group_size=self.group_size,
            merge_samples=merge_samples,
        )
        self.dtype = torch.bfloat16
        self.scales_dict = {}
        self.inps = None

    def _init_linear_list_and_model_type(self):
        self.use_transformer_engine = False

        self.isinstance_list = self.observer_layer_classes

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
        pre_transformer_modules_dict = self.model.get_pre_transformer_modules()
        for _, module in pre_transformer_modules_dict.items():
            module.to(dev)
        layers[0] = Catcher(layers[0], self.inps, cache)
        self.model.model_forward(dataloader)
        for _, module in pre_transformer_modules_dict.items():
            module.cpu()
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
        # begin the awq process
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

            print_info("HOOK Step{}".format(j))
            for h in handles:
                h.remove()

            # now solve for scaling and clipping
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
            # Clear GPU memory
            torch.cuda.empty_cache()

            scales_list = self.scale_function.auto_scale(
                layer, input_feat, layer_kwargs
            )

            self.scale_function.apply_scale(layer, scales_list, input_feat)
            for scales in scales_list:
                name = "language_model.encoder.layers.{}.{}.scale".format(i, scales[0])
                self.scales_dict[name] = scales[2]

            if self.mse_range:
                clip_list = self.clip_function.auto_clip(layer, input_feat)
                self.clip_function.apply_clip(layer, clip_list)

                for j in range(min(self.inps.shape[1], nsamples)):
                    with torch.no_grad():
                        outs[j, :, :] = layer(
                            self.inps[j, :, :].unsqueeze(0), **layer_kwargs
                        )[0].squeeze(1)
            layers[i] = layers[i].cpu()
            layer = layer.cpu()
            torch.cuda.empty_cache()
            self.inps, outs = outs, self.inps
            print_info("AWQ end layer {}\n".format(i))

    def save(self, save_dir, shard_size="5GB", safetensors=True):
        save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # Save model and config files with empty state dict
        self.model.model.config.quantization_config = {
            "quant_method": "awq",
            "zero_point": self.zero_point,
            "group_size": self.group_size,
            "bits": self.quant_bits,
            "version": "gemm",
            "modules_to_not_convert": ["visual"] if self.modal_type == "VLM" else None,
        }
        self.model.model.config.save_pretrained(
            save_dir, state_dict=EmptyModule().state_dict()
        )

        # # Vision transformers have a processor
        # if self.processor is not None:
        #     self.processor.save_pretrained(save_dir)

        # Remove empty state dict
        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)

        save_torch_state_dict(
            state_dict=self.model.model.state_dict(),
            save_directory=save_dir,
            max_shard_size=shard_size,
            safe_serialization=safetensors,
            force_contiguous=True,
            shared_tensors_to_discard=self.model.model._tied_weights_keys,
        )
        self.model.model.config.torch_dtype = "float16"
        self.model.model.config.to_json_file(os.path.join(save_dir, "config.json"))

    def _find_layers(self, module, layers=None, name=""):
        if not layers:
            layers = self.isinstance_list
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

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            linear_layer.weight.data, scales, zeros = pseudo_quantize_tensor(
                linear_layer.weight.data,
                w_bit=self.quant_bits,
                zero_point=self.zero_point,
                q_group_size=self.group_size,
                get_scale_zp=True,
            )

            scales = scales.t().contiguous()
            if zeros is not None:
                zeros = zeros.t().contiguous()
            q_linear_module = WQLinearGEMM

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.quant_bits,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)

    def _convert_llm(self):
        for i in tqdm(range(len(self.layers)), desc="AWQ"):
            subset = self._find_layers(self.layers[i])
            self._apply_quant(self.layers[i], subset)

    def convert(self):
        """
        Saves scales and inserts QDQ modules.
        """
        print_info("Start convert model...")
        if self.modal_type in ["LLM", "VLM"]:
            self._convert_llm()
        elif self.modal_type == "AIGC":
            pass
        else:
            print_info("current {} modal type not support".format(self.modal_type))
            raise NotImplementedError
        print_info("convert model done.")
