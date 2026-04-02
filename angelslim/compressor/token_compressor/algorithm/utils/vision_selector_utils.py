# Copyright 2026 Tencent Inc. All Rights Reserved.
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

import glob
import importlib.util
import inspect
import json
import os
import time
from typing import Any, Dict, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Mapping conceptual roles to actual forward argument names in various
# model versions
CONCEPT_MAPPING = {
    "vision": [
        "vision_hidden",
        "x",
        "hidden_states",
        "visual_embeds",
        "xs",
        "vision_embeds",
    ],
    "text": [
        "text_hidden",
        "text_embeds",
        "context_hidden",
        "textual_hidden",
    ],
}

_SELECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}


@torch.no_grad()
def _find_ts(xs: torch.Tensor, k: float) -> torch.Tensor:
    """
    Binary search for offset 'ts' such that sum(sigmoid(xs + ts)) is approximately k.

    Args:
        xs (torch.Tensor): Raw importance scores [B, N].
        k (float): Target absolute number of tokens to retain.

    Returns:
        torch.Tensor: The calculated offsets for each batch.
    """
    xs = xs.float()
    # Define search bounds
    lo = -xs.max(dim=1, keepdims=True).values - 10.0
    hi = -xs.min(dim=1, keepdims=True).values + 10.0

    # 64 iterations provide sufficient precision for soft top-k
    for _ in range(64):
        mid = (hi + lo) / 2
        soft_count = torch.sigmoid(xs + mid).sum(dim=1)
        mask = soft_count < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]

    return (lo + hi) / 2


def compute_soft_topk_weight(scores: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    """
    Converts raw scores into soft Top-K weights in the range [0, 1].

    Args:
        scores (torch.Tensor): Model raw outputs.
        ratio (float): Retention ratio (default is 0.2 for training consistency).

    Returns:
        torch.Tensor: Normalized weights [B, N].
    """
    orig_dtype = scores.dtype
    xs = scores.float()
    b, n = xs.shape

    target_k = n * ratio
    # Numerical stability protection
    target_k = max(1e-4, min(target_k, n - 1e-4))

    ts = _find_ts(xs, target_k)
    return torch.sigmoid(xs + ts).to(dtype=orig_dtype)


def _init_selector_entry(selector_path: str, device: torch.device):
    """
    Performs one-time initialization for the Vision Selector model.
    Supports loading from a local directory or downloading from Hugging Face Hub.
    """
    # 1. Resolve model path: check local existence first, then try HF Hub
    if os.path.isdir(selector_path):
        resolved_path = selector_path
    else:
        try:
            print(
                "[TokenCompressor] "
                f"'{selector_path}' not found locally. Downloading from Hugging Face."
            )
            resolved_path = snapshot_download(repo_id=selector_path)
        except Exception as e:
            raise FileNotFoundError(
                "[TokenCompressor Error] "
                f"Failed to resolve selector path '{selector_path}'. "
                f"Details: {e}"
            )

    # 2. Locate essential files (config, weights, and python source)
    config_path = os.path.join(resolved_path, "config.json")
    weights_files = glob.glob(os.path.join(resolved_path, "*.safetensors")) + glob.glob(
        os.path.join(resolved_path, "*.bin")
    )
    py_files = glob.glob(os.path.join(resolved_path, "*.py"))

    if not weights_files or not py_files:
        raise FileNotFoundError(
            f"[TokenCompressor Error]" f"Incomplete selector components in: {resolved_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 3. Dynamically load the model class definition
    module_name = f"dynamic_selector_{int(time.time()*1000)}"
    spec = importlib.util.spec_from_file_location(module_name, py_files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Identify the class name from config or fallback to default
    arch_name = config.get("architecture", "TransformerScorer")
    ModelClass = getattr(mod, arch_name, None)
    if not ModelClass:
        raise AttributeError(
            f"[TokenCompressor Error] Class '{arch_name}' not found in {py_files[0]}"
        )

    # 4. Auto-align parameters for Initialization (__init__)
    init_params = inspect.signature(ModelClass.__init__).parameters
    init_kwargs = {p: config[p] for p in init_params if p in config}
    model = ModelClass(**init_kwargs)

    # 5. Auto-align parameters for Execution (forward)
    # Detect which arguments correspond to 'vision' and 'text' concepts
    forward_params = inspect.signature(model.forward).parameters
    arg_mapping = {}
    for p_name in forward_params:
        if p_name in CONCEPT_MAPPING["vision"]:
            arg_mapping[p_name] = "vision"
        elif p_name in CONCEPT_MAPPING["text"]:
            arg_mapping[p_name] = "text"

    if "vision" not in arg_mapping.values():
        raise TypeError(
            "[TokenCompressor Error]"
            f"Cannot identify vision input for forward function in {arch_name}."
        )

    # 6. Load weights and clean internal training prefixes
    weights_path = weights_files[0]
    if weights_path.endswith(".safetensors"):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    cleaned_sd = {}
    for k, v in state_dict.items():
        # Remove common wrapper prefixes inherited from training frameworks
        new_k = (
            k.replace("module.", "")
            .replace("visual.importance_scorer.", "")
            .replace("_orig_mod.", "")
        )
        cleaned_sd[new_k] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device).eval()

    # 7. Store in the global registry
    _SELECTOR_REGISTRY[selector_path] = {
        "model": model,
        "arg_mapping": arg_mapping,
        "dtype": next(model.parameters()).dtype,
    }


def get_universal_selector_scores(
    selector_path: str,
    vision_hidden: torch.Tensor,
    text_hidden: Optional[torch.Tensor] = None,
    mode: str = "soft_topk",
    **extra_kwargs,
) -> torch.Tensor:
    """
    Universal interface for Vision Selector score extraction.
    Handles caching, introspection, and post-processing.

    Args:
        selector_path (str, required): Path or HF Repo ID to the selector model.
        vision_hidden (torch.Tensor, required): Visual features [B, N, D].
        text_hidden (torch.Tensor, optional): Textual features [B, M, D].
        mode (str): Post-processing mode ('soft_topk', 'softmax', or 'raw').
        extra_kwargs: Additional arguments passed to the model's forward.

    Returns:
        torch.Tensor: Importance scores [B, N].
    """
    # Initialize and cache model if not present
    if selector_path not in _SELECTOR_REGISTRY:
        _init_selector_entry(selector_path, vision_hidden.device)

    entry = _SELECTOR_REGISTRY[selector_path]
    model = entry["model"]
    dtype = entry["dtype"]

    # Assemble arguments based on detected mapping
    call_params = {}
    for arg_name, role in entry["arg_mapping"].items():
        if role == "vision":
            call_params[arg_name] = vision_hidden.to(dtype=dtype)
        elif role == "text" and text_hidden is not None:
            call_params[arg_name] = text_hidden.to(dtype=dtype)

    # Include extra parameters like position_ids if provided
    call_params.update(extra_kwargs)

    # Inference
    with torch.no_grad():
        outputs = model(**call_params)
        # Ensure output is [B, N]
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)

    # Final post-processing
    if mode == "soft_topk":
        return compute_soft_topk_weight(outputs, ratio=0.2)
    elif mode == "softmax":
        return torch.softmax(outputs.float(), dim=-1).to(dtype=outputs.dtype)
    else:
        return outputs
