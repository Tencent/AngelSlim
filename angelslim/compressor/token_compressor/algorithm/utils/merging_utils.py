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

from typing import List, Optional, Tuple

import torch

from ...base.context import PruningContext


def get_dialogue_masks(
    context: PruningContext,
) -> Tuple[
    Optional[torch.Tensor],
    List[torch.Tensor],
    Optional[torch.Tensor],
    List[int],
    List[bool],
]:
    """
    Analyzes the dialogue structure to split the sequence into physical segments
    (e.g., individual frames/images).
    """
    input_ids = context.input_ids
    if input_ids is None:
        raise ValueError("[TokenCompressor Error] 'input_ids' missing in context.")

    device = input_ids.device
    input_ids_1d = input_ids.squeeze(0)
    seq_len = input_ids_1d.shape[0]

    from .utils import (
        _extract_and_validate_vision_token_info,
        identify_model_architecture,
    )

    model_type = identify_model_architecture(context)
    _, _, vision_mask, _ = _extract_and_validate_vision_token_info(context)

    # Calculate physical unit lengths based on cu_seqlens and spatial pooling
    # factor
    cu_seqlens = context.cu_seqlens_full
    merged_units = []
    if cu_seqlens is not None:
        m_unit = context.spatial_merge_size**2
        fine_lens = torch.diff(cu_seqlens).cpu().tolist()
        for flen in fine_lens:
            actual_len = flen - 1 if model_type == "llava_ov" else flen
            mlen = actual_len // m_unit
            if mlen > 0:
                merged_units.append(mlen)

    # Segment the sequence into text blocks and vision islands
    changes = torch.nonzero(vision_mask[1:] != vision_mask[:-1]).flatten() + 1
    bounds = torch.cat(
        [
            torch.tensor([0], device=device),
            changes,
            torch.tensor([seq_len], device=device),
        ]
    )

    sizes, is_vis_list, masks = [], [], []
    unit_ptr = 0

    for i in range(len(bounds) - 1):
        s, e = bounds[i].item(), bounds[i + 1].item()
        seg_size = e - s
        is_vis = bool(vision_mask[s].item())

        if is_vis and merged_units:
            consumed = 0
            while consumed < seg_size:
                u_len = merged_units[unit_ptr]
                sizes.append(u_len)
                is_vis_list.append(True)
                m = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
                m[0, s + consumed : s + consumed + u_len] = True
                masks.append(m)
                consumed += u_len
                unit_ptr += 1
        else:
            sizes.append(seg_size)
            is_vis_list.append(is_vis)
            if is_vis:
                m = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
                m[0, s:e] = True
                masks.append(m)

    return (None, masks, None, sizes, is_vis_list)
