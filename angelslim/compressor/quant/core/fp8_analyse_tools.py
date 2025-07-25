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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors.torch import load_file


def draw_sub_lines(weight_dict, ax, opname):
    np_l = []
    for v in weight_dict[opname]:
        np_l.append(v)
    np_l.sort(key=lambda x: x[0])
    x_layer = [int(v[0]) for v in np_l]
    y_scale = [float(v[1]) for v in np_l]
    ax.plot(x_layer, y_scale, "b-")
    ax.set_title(opname)
    ax.set_xlabel("Layers")
    ax.set_ylabel("Scales")
    ax.grid(True)


def draw_fp8_scale_fig(model_path, save_path):
    g = os.walk(model_path)
    st_file_list = []
    for path, _, file_list in g:
        print(path)
        for file_name in file_list:
            if "safetensors" in file_name and "index" not in file_name:
                st_file_list.append(file_name)
    print(st_file_list)

    weight_dict = {}  # {"OP": (layer, data)}

    for file in st_file_list:
        model_weight = load_file(os.path.join(model_path, file), device="cpu")
        for k in model_weight.keys():
            if "layers" in k and "scale" in k:
                k_spllit = k.split("layers", 1)
                num_layer = int(k_spllit[-1].split(".", 2)[1])
                OP = k_spllit[-1].split(".", 2)[-1]
                if OP not in weight_dict.keys():
                    weight_dict[OP] = [(num_layer, model_weight[k].data.float())]
                else:
                    weight_dict[OP].append((num_layer, model_weight[k].data.float()))

    weight_op = weight_dict.keys()
    assert weight_op is not None, "fp8 weight does not exist."
    print(f"weight scale op {weight_op}")

    # all fp8 scale
    np_l = []
    for opname in weight_op:
        for v in weight_dict[opname]:
            np_l.append(v[-1])
            if v[-1].data > 1.5:
                print(
                    f"[AngelSlim Warning] "
                    f"layer_{v[0]}_{opname} The weight is too high:{v[-1]}. "
                    f"It is recommended to clip it to 1.5 "
                )
    a = np.array(np_l)
    if len(a.shape) == 2:
        a = a[:, 0]
    elif len(a.shape) <= 1:
        pass
    else:
        print("[AngelSlim Error] scale dim error ")
    s = pd.Series(a)
    plt.hist(s)
    plt.savefig(os.path.join(save_path, "all_quant_scale_histogram.jpg"))
    plt.cla()
    plt.clf()
    plt.close()

    # per layer scale line
    list_weight_op = list(weight_op)

    list_weight_op.sort()
    is_dynamic = True
    for k in list_weight_op:
        if "input_scale" in k:
            is_dynamic = False
            break
    group_weight_op = []
    if not is_dynamic:
        while len(list_weight_op) > 0:
            assert len(list_weight_op) % 2 == 0, "Some fp8 scale are missing. "
            group_weight_op.append((list_weight_op.pop(0), list_weight_op.pop(0)))
        print(f"all scale sum {sum(np_l)}")
        if sum(np_l) < 3:
            print(
                "[AngelSlim Warning] This model's scale is too small overall, "
                "which is not conducive to the expression of FP8 precision."
            )
        for g_opname in group_weight_op:
            plt.cla()
            plt.clf()
            plt.close()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            draw_sub_lines(weight_dict, ax1, g_opname[0])
            draw_sub_lines(weight_dict, ax2, g_opname[1])
            plt.savefig(
                os.path.join(save_path, f"./OP_{g_opname}_quant_scale_line.jpg")
            )
    else:
        print("dynamic fp8 analyse")
        for opname in list_weight_op:
            plt.cla()
            plt.clf()
            plt.close()
            fig, (ax1) = plt.subplots(1, 1, figsize=(6, 5))
            draw_sub_lines(weight_dict, ax1, opname)
            plt.savefig(os.path.join(save_path, f"./OP_{opname}_quant_scale_line.jpg"))


def get_weight_dict(model_path):
    g = os.walk(model_path)
    st_file_list = []

    for path, _, file_list in g:
        if model_path != path:
            break
        for file_name in file_list:
            if "safetensors" in file_name and "index" not in file_name:
                st_file_list.append(file_name)
    weight_dict = {}  # {"layer": {op: data}
    for file in st_file_list:
        model_weight = load_file(os.path.join(model_path, file), device="cpu")
        for k in model_weight.keys():
            if "layers" in k and ".weight" in k and "scale" not in k:
                k_spllit = k.split("layers", 1)
                num_layer = str(int(k_spllit[-1].split(".", 2)[1]))
                op = k_spllit[-1].split(".", 2)[-1]
                if num_layer not in weight_dict.keys():
                    weight_dict[num_layer] = {}
                    weight_dict[num_layer][op] = model_weight[k].data
                else:
                    weight_dict[num_layer][op] = model_weight[k].data

    return weight_dict


def draw_hist(uniform_data, ax, name):
    uniform_data.sort()
    s = pd.Series(uniform_data)
    ax.hist(s, bins=50, rwidth=1)
    ax.set_title(name + "_histgram")
    ax.grid(True)


def draw_bf16_fp8_weight_fig(bf16_path, fp8_path, save_path, layer_index):
    bf16_weight_dict = get_weight_dict(bf16_path)
    fp8_weight_dict = get_weight_dict(fp8_path)

    for op_name in bf16_weight_dict[str(layer_index)].keys():
        plt.cla()
        plt.clf()
        plt.close()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        tensor_data = bf16_weight_dict[str(layer_index)][op_name].float().view(-1)

        bf16w = np.array(tensor_data)

        draw_hist(bf16w, ax1, f"BF16_{op_name}")

        fp8w = fp8_weight_dict[str(layer_index)][op_name].float().view(-1)

        uniform_data = np.array(fp8w)
        draw_hist(uniform_data, ax2, f"FP8_{op_name}")

        plt.savefig(
            os.path.join(save_path, f"./layer_{layer_index}_op_{op_name}_histogram.jpg")
        )
