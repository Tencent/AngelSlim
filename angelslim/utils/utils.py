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

import datetime
import importlib.metadata
import json
import os
import subprocess
from itertools import takewhile
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from transformers.utils.hub import cached_file


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def find_parent_layer_and_sub_name(model, name):
    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == ".":
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name


def find_layers(module, layers=None, name=""):
    if not layers:
        layers = [torch.nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def get_tensor_item(x):
    return x.item()


def print_info(info):
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "[AngelSlim]"
    try:
        _index = torch.distributed.get_rank()
    except ValueError:
        _index = 0
    if _index == 0:
        print("[{}] {} {}".format(time, prefix, info))


def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    elif torch.xpu.is_available():
        return "xpu:0"
    else:
        return "cpu"


def get_yaml_prefix_simple(file_path: str) -> Optional[str]:
    """
    Simplified version using os.path
    """
    if not file_path or not isinstance(file_path, str):
        return None

    filename = os.path.basename(file_path)

    # Handle hidden files
    if filename.startswith(".") and "." in filename[1:]:
        parts = filename.split(".")
        if parts[-1].lower() in ["yaml", "yml"]:
            return ".".join(parts[:-1])
        return filename

    # Process normal files
    name, ext = os.path.splitext(filename)
    if ext.lower() in [".yaml", ".yml"]:
        return name
    return filename


def get_hf_config(model_path) -> dict:
    "When model_path does not exist, fetch the model.config from cached_file."
    if os.path.isfile(model_path):
        config_path = os.path.join(model_path, "config.json")
    else:
        config_path = cached_file(model_path, "config.json")

    with open(config_path, "r", encoding="utf8") as fp:
        json_data = json.load(fp)
        return json_data


def get_hf_model_path(model_path) -> str:
    "When model_path does not exist, fetch the model.config from cached_file."
    if os.path.isfile(model_path):
        return model_path
    else:
        return os.path.dirname(cached_file(model_path, "config.json"))


def common_prefix(str1, str2):
    return "".join(
        x[0] for x in takewhile(lambda x: x[0] == x[1], zip(str1, str2))
    ).rpartition(".")[0]


def get_package_info(package_name: str) -> dict:
    info = {"name": package_name, "version": "N/A", "source": "Unknown"}
    try:
        version = importlib.metadata.version(package_name)
        info["version"] = version
        info["source"] = "pip"
    except Exception:
        try:
            package = __import__(package_name)
            path = Path(package.__path__[0]).parent
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=path, text=True
            ).strip()
            info["version"] = commit_hash
            info["source"] = "git"
        except Exception:
            pass
    return info


def rank0_print(*args, **kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank == 0:
        print(*args, **kwargs)


def _get_distributed_info():
    """
    Get distributed training information.

    Returns:
        Tuple of (rank, world_size, local_rank):
        - rank: Global rank in distributed training (-1 if not distributed)
        - world_size: Total number of processes (1 if not distributed)
        - local_rank: Local rank on current node (-1 if not set)
    """
    rank = -1
    world_size = 1
    local_rank = -1

    # Check for torchrun environment variable first
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    # Then check if distributed is initialized
    elif dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return rank, world_size, local_rank


def print_with_rank(*args, **kwargs):
    """
    Print function with rank information for distributed training.

    Automatically detects the current process rank and includes it in the output.
    Works with torchrun, torch.distributed, or single process environments.

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function

    Example:
        print_with_rank("Model loaded successfully")
        # Single node: [Rank 0/4] Model loaded successfully
        # Multi-node:  [Rank 0/8, Local 0] Model loaded successfully
    """
    rank, world_size, local_rank = _get_distributed_info()

    # Format rank information
    if rank >= 0:
        # Show local_rank only when it's different from rank (multi-node scenario)
        if local_rank >= 0 and local_rank != rank:
            prefix = f"[Rank {rank}/{world_size}, Local {local_rank}]"
        else:
            prefix = f"[Rank {rank}/{world_size}]"
    else:
        prefix = "[Single Process]"

    # Print with rank prefix
    print(prefix, *args, **kwargs)


def decide_device_for_distributed():
    """
    Decide the appropriate device for model in distributed training context (torchrun).

    Device selection priority:
    1. LOCAL_RANK environment variable (torchrun launcher)
    2. Distributed rank (if torch.distributed is initialized)
    3. cuda:0 or cpu (single process fallback)

    Returns:
        str: Device string like 'cuda:0' or 'cpu'

    Example:
        device = decide_device_for_distributed()
        model.to(device)
    """
    rank, _, local_rank = _get_distributed_info()

    # Determine device based on distributed info
    if local_rank >= 0:
        # torchrun with LOCAL_RANK
        device = f"cuda:{local_rank}"
    elif rank >= 0:
        # Distributed initialized without LOCAL_RANK
        device = f"cuda:{rank}"
    else:
        # Single process fallback
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return device
