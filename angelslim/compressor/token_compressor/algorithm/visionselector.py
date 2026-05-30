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

"""
Vision Selector Pruning Strategy module.
"""


import torch

from ..base.context import PruningContext
from .utils.utils import _extract_and_validate_vision_token_info, get_valid_content_mask
from .utils.vision_selector_utils import get_universal_selector_scores


def vision_selector_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    Executes global token pruning using the pre-trained Vision Selector.

    Args:
        context (PruningContext):
        The execution context containing multimodal embeddings.
        **kwargs:
            selector_path (str):
            The absolute or relative path to the selector model folder.
            ratio (float):
            The percentage of vision tokens to be pruned.
            randomize (bool):
            If True, use multinomial sampling; otherwise, use deterministic Top-K.
            text_selection_mode (str):
            Specifies how to select text features for V2 selectors
            ('inverse' or 'valid_content').
            score_mode (str):
            The processing mode for importance weights (e.g., 'soft_topk', 'raw').

    Returns:
        torch.Tensor: A boolean keep_mask of shape [1, sequence_length].
    """
    # 1. Parse strategy hyperparameters
    try:
        selector_path = kwargs["selector_path"]
        pruning_ratio = kwargs["ratio"]
        randomize = kwargs.get("randomize", False)
        text_selection_mode = kwargs.get("text_selection_mode", "inverse")
        score_mode = kwargs.get("score_mode", "soft_topk")
    except KeyError as e:
        raise ValueError(
            f"[TokenCompressor Error] "
            f"'vision_selector_pruning' missing required parameter: {e}"
        )

    # 2. Access context attributes directly
    input_ids = context.input_ids
    inputs_embeds = context.inputs_embeds

    if input_ids is None or inputs_embeds is None:
        raise ValueError(
            "[TokenCompressor Error] "
            "Required tensors 'input_ids' or 'inputs_embeds' not found in context."
        )

    device = inputs_embeds.device
    batch_size = inputs_embeds.shape[0]
    if batch_size != 1:
        raise NotImplementedError(
            "[TokenCompressor Error] "
            "Vision Selector strategy currently only supports batch_size=1."
        )

    # 3. Locate modality-specific tokens
    vision_indices, non_vision_indices, vision_token_mask, _ = (
        _extract_and_validate_vision_token_info(context)
    )
    num_vision_tokens = len(vision_indices)

    # Return early if no vision tokens are present
    if num_vision_tokens == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # Calculate retention target
    num_to_keep = int(round(num_vision_tokens * (1.0 - pruning_ratio)))
    if pruning_ratio < 1.0 and num_to_keep == 0:
        num_to_keep = 1

    # If budget exceeds current token count, retain everything
    if num_to_keep >= num_vision_tokens:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 4. Prepare multimodal inputs for the selector
    vision_hidden = inputs_embeds[:, vision_indices, :]

    text_hidden = None
    if text_selection_mode == "valid_content":
        # Extract meaningful text content (e.g., instructions after the last
        # image)
        valid_mask = get_valid_content_mask(input_ids[0])
        text_indices = torch.where(valid_mask)[0]
        if len(text_indices) > 0:
            text_hidden = inputs_embeds[:, text_indices, :]

    # Fallback to all non-vision tokens if needed
    if text_hidden is None:
        text_indices = torch.where(~vision_token_mask)[0]
        text_hidden = inputs_embeds[:, text_indices, :]

    # 5. Inference via Universal Selector Scorer Utility
    # This call handles V1/V2 architecture detection and score normalization
    importance_weights = get_universal_selector_scores(
        selector_path=selector_path,
        vision_hidden=vision_hidden,
        text_hidden=text_hidden,
        mode=score_mode,
    )

    # Flatten scores for selection processing
    scores = importance_weights.squeeze(0).float()

    # 6. Discrete selection execution
    k = min(num_to_keep, num_vision_tokens)

    if randomize:
        # Probabilistic sampling based on importance distribution
        probs = torch.softmax(scores, dim=-1)
        try:
            top_k_local_indices = torch.multinomial(probs, num_samples=k, replacement=False)
        except RuntimeError as e:
            raise RuntimeError(
                "[TokenCompressor Error] "
                f"Multinomial sampling failed for Vision Selector. Details: {e}"
            )
    else:
        # Deterministic selection of highest scores
        _, top_k_local_indices = torch.topk(scores, k=k)

    # 7. Construct and return the global keep_mask
    kept_vision_global_indices = vision_indices[top_k_local_indices]
    final_kept_indices = torch.cat([non_vision_indices.to(device), kept_vision_global_indices])

    keep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    keep_mask[0, final_kept_indices] = True

    return keep_mask
