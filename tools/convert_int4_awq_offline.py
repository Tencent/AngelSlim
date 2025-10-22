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

import json
import multiprocessing as mp
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

from angelslim.compressor.quant.core import weight_dequant

SUFFIX_TO_QUANT = [
    ".gate_and_up_proj.weight",
    ".gate_proj.weight",
    ".down_proj.weight",
    ".up_proj.weight",
    ".q_a_proj.weight",
    ".q_b_proj.weight",
    ".kv_b_proj.weight",
    ".kv_a_proj_with_mqa.weight",
    ".qkv_proj.weight",
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    "indexer.wq_b.weight",
    "indexer.wk.weight",
]


def post_process(weight, w_bit, group_size, scales=None, zeros=None):
    _, in_features = weight.shape
    assert scales is not None and zeros is not None
    scale_zeros = zeros * scales
    pack_num = 32 // w_bit
    intweight = []
    for idx in range(in_features):
        intweight.append(
            torch.round(
                (weight.data[:, idx] + scale_zeros[idx // group_size])
                / scales[idx // group_size]
            ).to(torch.int)[:, None]
        )
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.to(dtype=torch.int32)
    qweight = torch.zeros(
        (intweight.shape[0], intweight.shape[1] // 32 * w_bit),
        dtype=torch.int32,
        device=intweight.device,
    )
    for col in range(intweight.shape[1] // pack_num):
        if w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")
        for i in range(pack_num):
            qweight_col = intweight[:, col * pack_num + order_map[i]]
            qweight[:, col] |= qweight_col << (i * w_bit)
    zeros = zeros.to(dtype=torch.int32, device=intweight.device)
    qzeros = torch.zeros(
        (zeros.shape[0], zeros.shape[1] // 32 * w_bit),
        dtype=torch.int32,
        device=zeros.device,
    )
    for col in range(zeros.shape[1] // pack_num):
        if w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")
        for i in range(pack_num):
            qzero_col = zeros[:, col * pack_num + order_map[i]]
            qzeros[:, col] |= qzero_col << (i * w_bit)
    scales = scales.half()
    return qweight, qzeros, scales


def create_quantized_param(
    w,
    weight_scale_inv,
    w_bit=4,
    zero_point=True,
    q_group_size=-1,
    inplace=False,
    get_scale_zp=False,
):
    if weight_scale_inv is not None:
        weight_scale_inv = weight_scale_inv.to(w.device)
        w = weight_dequant(w, weight_scale_inv)
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (w_bit - 1) - 1
        min_int = -(2 ** (w_bit - 1))
        scales = max_val / max_int
        zeros = 0
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0
    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0
    w = w.reshape(org_w_shape)
    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


def process_safetensor(
    rank,
    file_name,
    input_path,
    weight_scale_invs,
    output_path,
    group_size,
    bit,
    zero_point,
    exclude_patterns=None,
):
    state_dict = {}
    index = {}
    with safe_open(
        os.path.join(input_path, file_name), framework="pt", device=f"cuda:{rank}"
    ) as f:
        print(f"Processing {file_name} with {len(f.keys())} weights")
        for weight_name in f.keys():
            weight = f.get_tensor(weight_name)
            if any(suffix in weight_name for suffix in SUFFIX_TO_QUANT) and not any(
                exclude_pattern in weight_name for exclude_pattern in exclude_patterns
            ):
                if weight_name.endswith("weight_scale_inv"):
                    continue
                scale_inv = weight_scale_invs.get(f"{weight_name}_scale_inv", None)
                quant_weight, scales, zeros = create_quantized_param(
                    weight,
                    scale_inv,
                    w_bit=bit,
                    zero_point=zero_point,
                    q_group_size=group_size,
                    get_scale_zp=True,
                )
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                qweight, qzeros, qscales = post_process(
                    quant_weight,
                    w_bit=bit,
                    group_size=group_size,
                    scales=scales,
                    zeros=zeros,
                )
                state_dict[weight_name.replace("weight", "qweight")] = qweight
                state_dict[weight_name.replace("weight", "scales")] = qscales
                state_dict[weight_name.replace("weight", "qzeros")] = qzeros
                index[weight_name.replace("weight", "qweight")] = file_name
                index[weight_name.replace("weight", "scales")] = file_name
                index[weight_name.replace("weight", "qzeros")] = file_name
            else:
                state_dict[weight_name] = weight
                index[weight_name] = file_name
    new_safetensor_file = os.path.join(output_path, file_name)
    save_file(state_dict, new_safetensor_file)
    return index


def worker(
    i,
    file_names,
    input_path,
    weight_scale_invs,
    output_path,
    group_size,
    bit,
    zero_point,
    return_dict,
    exclude_patterns=None,
):
    world_size = torch.cuda.device_count()
    for file_name in tqdm(file_names, desc=f"Worker {i}"):
        index = process_safetensor(
            i % world_size,
            file_name,
            input_path,
            weight_scale_invs,
            output_path,
            group_size,
            bit,
            zero_point,
            exclude_patterns,
        )
        return_dict[file_name] = index


def main(input_path, output_path, group_size, bit, zero_point, exclude_patterns=None):
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    safetensor_files = set(weight_map.values())
    safetensor_files = list(sorted(safetensor_files))
    print(f"Found {len(safetensor_files)} safetensor files")

    files = list(glob(os.path.join(input_path, "*.safetensors")))
    files.sort()
    weight_scale_invs = {}
    for file in tqdm(files):
        with safe_open(file, framework="pt") as f:
            target_tensors = {k: f.get_tensor(k) for k in f.keys()}
            for k, v in target_tensors.items():
                if k.endswith("weight_scale_inv"):
                    weight_scale_invs[k] = v

    file_subsets = [
        safetensor_files[i :: args.num_workers] for i in range(args.num_workers)
    ]
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=worker,
            args=(
                i,
                file_subsets[i],
                input_path,
                weight_scale_invs,
                output_path,
                group_size,
                bit,
                zero_point,
                return_dict,
                exclude_patterns,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    index = {}
    for result in return_dict.values():
        index.update(result)
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

    for file in os.listdir(input_path):
        src_path = os.path.join(input_path, file)
        dst_path = os.path.join(output_path, file)
        if os.path.exists(dst_path):
            continue
        if os.path.isdir(src_path):
            print(f"cp -r {src_path} {dst_path}")
            shutil.copytree(src_path, dst_path)
        else:
            print(f"cp {src_path} {dst_path}")
            shutil.copy2(src_path, dst_path)

    with open(os.path.join(output_path, "config.json"), "r") as f:
        config = json.load(f)
    config["quantization_config"] = {
        "bits": bit,
        "group_size": group_size,
        "modules_to_not_convert": exclude_patterns,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": zero_point,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bit", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--zero-point", type=bool)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--exclude-patterns", nargs="+", default=None)
    args = parser.parse_args()
    print(args)

    main(
        args.input_path,
        args.output_path,
        args.group_size,
        args.bit,
        args.zero_point,
        args.exclude_patterns,
    )
