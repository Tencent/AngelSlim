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
from argparse import ArgumentParser
from safetensors.torch import load_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_weight_dict(model_path):
    g = os.walk(model_path)
    st_file_list = []

    for path, dir_list, file_list in g:
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

        draw_hist(bf16w, ax1, f'BF16_{op_name}')

        fp8w = fp8_weight_dict[str(layer_index)][op_name].float().view(-1)

        uniform_data = np.array(fp8w)
        draw_hist(uniform_data, ax2, f'FP8_{op_name}')

        plt.savefig(os.path.join(save_path, f"./layer_{layer_index}_op_{op_name}_histogram.jpg"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bf16-path", type=str, help="Bf16 path")
    parser.add_argument("--fp8-path", type=str, help="Fp8 path")
    parser.add_argument("--save-path", type=str, default="./Weight_analyse")  #/SAVE/PATH
    parser.add_argument("--layer-index", type=int, required=True)
    args = parser.parse_args()
    print(f"[AngelSlim] Save weight analyse graph to {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    assert os.path.exists(args.bf16_path), f"File {args.bf16_path} not exist"
    assert os.path.exists(args.fp8_path), f"File {args.fp8_path} not exist"
    bf16_path = args.bf16_path
    fp8_path = args.fp8_path
    layer_index = args.layer_index
    draw_bf16_fp8_weight_fig(bf16_path=bf16_path, fp8_path=fp8_path, save_path=args.save_path, layer_index=layer_index)

