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

from .....utils import get_op_by_name, print_info
from ...core import EMASampler, MultiStepSampler, fake_quant_dequant, mse_loss


class AutoClip:
    """
    AutoClip from AWQ[https://arxiv.org/abs/2306.00978]
    """

    def __init__(
        self,
        model,
        weight_bits=4,
        weight_quant_method="groupwise",
        loss_function=mse_loss,
        sample_function=None,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
        group_size=64,
    ):
        super(AutoClip, self).__init__()
        self.model = model
        self.weight_bits = weight_bits
        self.weight_method = weight_quant_method
        self.loss_function = loss_function
        self.n_grid = n_grid
        self.max_shrink = max_shrink
        self.n_sample_token = n_sample_token
        self.bnt = (1 << (self.weight_bits - 1)) - 1
        self.sampled_inputs = {}
        self.sample_function = sample_function
        self.group_size = group_size
        self.linear_name_dict = {}
        assert sample_function in [None, "cat", "ema"]
        if sample_function == "cat":
            self.sample_function = MultiStepSampler()
        elif sample_function == "ema":
            self.sample_function = EMASampler()
        else:
            self.sample_function = EMASampler()

        self._apply_hook()

    def _apply_hook(self):
        self._forward_hook_list = []
        for name, sub_layer in self.model.named_modules():
            if isinstance(sub_layer, (nn.Linear)):
                forward_pre_hook_handle = sub_layer.register_forward_pre_hook(
                    self._forward_pre_hook
                )
                self._forward_hook_list.append(forward_pre_hook_handle)
                self.linear_name_dict[sub_layer] = name

    def _forward_pre_hook(self, layer, input):
        self._sample_scale(input, self.linear_name_dict[layer])
        return input

    def _sample_scale(self, input, name):
        input = input[0] if isinstance(input, tuple) else input
        if name not in self.sampled_inputs:
            self.sampled_inputs[name] = input
        else:
            if self.sample_function is not None:
                self.sampled_inputs[name] = self.sample_function.sample(
                    input, self.sampled_inputs[name], name
                )
            else:
                self.sampled_inputs[name] = input

    def auto_clip(self, group_size=128, oc_batch_size=256):
        """
        search clip scale for each layer and update the layer's weight
        """
        for name, sub_layer in self.model.named_modules():
            # TODO: or 'out_linear' in name
            if name not in self.sampled_inputs:
                continue
            sub_layer.weight.all_gather()
            w = sub_layer.weight
            x = self.sampled_inputs[name]
            print_info(
                "AutoClipping {}, x shape: {}, weight shape: {}".format(name, x.shape, w.shape)
            )
            x = x.view(-1, x.shape[-1])
            x = x.reshape(1, x.shape[0], -1, group_size)
            x = x[:, 0 :: x.shape[1] // self.n_sample_token]
            w = w.reshape([w.shape[0], 1, -1, group_size])
            oc_batch_size = (
                oc_batch_size if w.shape[0] % oc_batch_size == 0 else 128
            )  # prevent OOM
            assert w.shape[0] % oc_batch_size == 0

            w_all = w
            best_max_val_all = []

            for i_b in range(w.shape[0] // oc_batch_size):
                w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

                org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
                best_max_val = org_max_val.clone()
                min_errs = torch.ones_like(org_max_val) * 1e9
                x = x.to(w.device)
                org_out = (x * w).sum(dim=-1)  # co, n_token, n_group
                for i_s in range(int(self.max_shrink * self.n_grid)):
                    max_val = org_max_val * (1 - i_s / self.n_grid)
                    min_val = -max_val
                    cur_w = torch.clamp(w, min_val, max_val)
                    org_w_shape = cur_w.shape
                    cur_w = cur_w.reshape([-1, self.group_size])
                    quant_dequant_weight = fake_quant_dequant(
                        cur_w, method="abs_max_channel_wise", bits=self.weight_bits
                    )
                    quant_dequant_weight = quant_dequant_weight.reshape(org_w_shape)
                    cur_out = (x * quant_dequant_weight).sum(dim=-1)
                    # co, 1, n_group, 1
                    err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                    print_info("block {} search s {} err {}".format(i_b, i_s, err.mean().item()))
                    del cur_w, cur_out, quant_dequant_weight
                    cur_best_idx = err < min_errs
                    min_errs[cur_best_idx] = err[cur_best_idx]
                    best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_max_val_all.append(best_max_val)

            best_max_val = torch.cat(best_max_val_all, dim=0).squeeze(1)
            del org_out, x, w
            gc.collect()
            torch.cuda.empty_cache()

            for name, param in sub_layer.named_parameters():
                if "weight" in name:
                    param.all_gather()
                    org_shape = param.shape
                    param_tmp = param.reshape(*best_max_val.shape[:2], -1)
                    param_tmp = torch.clamp(param_tmp, -max_val, max_val)
                    param_tmp = param_tmp.reshape(org_shape)
                    param.data.copy_(param_tmp)
                    param.partition(has_been_updated=True)
                    del param_tmp

            torch.cuda.empty_cache()


class AutoLayerClip:
    def __init__(
        self,
        weight_bits=4,
        weight_quant_method="groupwise",
        group_size=128,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=1024,
        loss_function=mse_loss,
        merge_samples=True,
    ):
        """
        The implementation from AWQ(https://arxiv.org/pdf/2306.00978.pdf).
        """
        self.weight_bits = weight_bits
        self.weight_quant_method = weight_quant_method
        self.bnt = (1 << (weight_bits - 1)) - 1
        self.group_size = group_size
        self.n_grid = n_grid
        self.max_shrink = max_shrink
        self.n_sample_token = n_sample_token
        self.loss_function = loss_function
        self.merge_samples = merge_samples

    def apply_clip(self, module, clip_list):
        print_info("[apply clip] start")
        for name, max_val in clip_list:
            layer = get_op_by_name(module, name)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)

        print_info("[apply clip] end")

    def auto_clip(self, module, input_feat):
        print_info("[auto clip] start")
        named_linears = {
            name: m for name, m in module.named_modules() if isinstance(m, (nn.Linear))
        }

        clip_list = []
        for name in named_linears:
            # # due to qk bmm, it is hard to clip precisely
            # if 'attention.query_key_value' in name:
            #     continue
            print_info(f"[auto clip] {name}")
            max_val = self._auto_clip_layer(named_linears[name].weight, input_feat[name])
            clip_list.append((name, max_val))

        print_info("[auto clip] end")
        return clip_list

    def _auto_clip_layer(self, w, inp, oc_batch_size=256):
        assert inp.dim() in [3, 4]
        inp = inp.permute(1, 0, 2).contiguous() if inp.dim() == 3 else inp.squeeze(0)
        assert w.dim() == 2
        torch.cuda.empty_cache()
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # inp  [n_token, ci] -> [1, n_token, n_group, group size]
        if self.merge_samples:
            inp = inp.to(w.device)
            inp = inp.view(-1, inp.shape[-1])
            inp = inp.reshape(1, inp.shape[0], -1, self.group_size)
            input_feat = inp[:, 0 :: inp.shape[1] // self.n_sample_token]
        else:
            inp = inp.to(w.device)
            inp = inp.view(-1, inp.shape[-1])
            inp = inp.reshape(1, inp.shape[0], -1, self.group_size)
            input_feat = inp[:, 0 :: inp.shape[1] // int(self.n_sample_token)].contiguous()
            torch.cuda.empty_cache()

        w = w.reshape([w.shape[0], 1, -1, self.group_size])
        oc_batch_size = oc_batch_size if w.shape[0] % oc_batch_size == 0 else 128  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0

        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(self.max_shrink * self.n_grid)):
                max_val = org_max_val * (1 - i_s / self.n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                org_w_shape = cur_w.shape
                cur_w = cur_w.reshape([-1, self.group_size])
                quant_dequant_weight = fake_quant_dequant(
                    cur_w, method="abs_max_channel_wise", bits=self.weight_bits
                )
                quant_dequant_weight = quant_dequant_weight.reshape(org_w_shape)
                cur_out = (input_feat * quant_dequant_weight).sum(dim=-1)
                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w, cur_out, quant_dequant_weight
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0).squeeze(1)  # co, n_group, 1
        # print_info(f"best_max_val: {best_max_val}")
        inp = inp.cpu()
        del org_out, input_feat, w
        gc.collect()
        torch.cuda.empty_cache()

        return best_max_val
