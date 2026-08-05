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

"""NVFP4 scale handling for Linear layers fused by deployment runtimes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, MutableMapping

import torch

_FUSION_GROUP_MEMBERS = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "gate_up": ("gate_proj", "up_proj"),
}
_MEMBER_TO_GROUP = {
    member: group for group, members in _FUSION_GROUP_MEMBERS.items() for member in members
}
_MAX_E4M3_RELATIVE_ROUNDING_ERROR = 0.063


def nvfp4_scale_2_group_key(name: str) -> tuple[str, str] | None:
    """Return the deployment fusion group for a split Linear module name."""
    member = name.rsplit(".", 1)[-1]
    group = _MEMBER_TO_GROUP.get(member)
    if group is None:
        return None
    return name[: -len(member)], group


def nvfp4_fusion_groups(prefixes: Iterable[str]) -> list[tuple[str, str, list[str]]]:
    """Collect complete q/k/v and gate/up groups in deployment order."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for prefix in prefixes:
        key = nvfp4_scale_2_group_key(prefix)
        if key is not None:
            groups[key].append(prefix)

    result = []
    for (group_prefix, group), members in sorted(groups.items()):
        expected = _FUSION_GROUP_MEMBERS[group]
        by_member = {name.rsplit(".", 1)[-1]: name for name in members}
        if set(by_member) != set(expected):
            missing = sorted(set(expected) - set(by_member))
            unexpected = sorted(set(by_member) - set(expected))
            raise ValueError(
                f"Incomplete NVFP4 {group} fusion group '{group_prefix}*': "
                f"missing={missing}, unexpected={unexpected}"
            )
        result.append((group_prefix, group, [by_member[name] for name in expected]))
    return result


def harmonize_nvfp4_fused_scales(
    state_dict: MutableMapping[str, torch.Tensor],
) -> dict[str, float | int]:
    """Fold per-Linear global scales into FP8 block scales for fused GEMMs.

    compressed-tensors stores ``weight_global_scale = 1 / scale_2``. Deployment
    kernels use one scalar ``scale_2`` for each fused q/k/v or gate/up GEMM.
    For each member ``i`` this transformation keeps the packed FP4 codes fixed:

    ``local_i * scale_2_i == (local_i * scale_2_i / shared_scale_2) * shared_scale_2``.

    The adjusted local scale must be re-encoded as FP8 E4M3, so the equality is
    exact before that final FP8 rounding step.
    """
    packed_suffix = ".weight_packed"
    prefixes = [key[: -len(packed_suffix)] for key in state_dict if key.endswith(packed_suffix)]
    groups = nvfp4_fusion_groups(prefixes)
    finfo = torch.finfo(torch.float8_e4m3fn)

    rebased_local_scale_count = 0
    max_global_scale_ratio = 1.0
    max_local_scale_relative_error = 0.0

    for group_prefix, group, members in groups:
        weight_globals = []
        input_globals = []
        for prefix in members:
            local_key = f"{prefix}.weight_scale"
            weight_global_key = f"{prefix}.weight_global_scale"
            input_global_key = f"{prefix}.input_global_scale"
            missing = [
                key
                for key in (local_key, weight_global_key, input_global_key)
                if key not in state_dict
            ]
            if missing:
                raise KeyError(
                    f"Missing NVFP4 scale tensors for {group} group "
                    f"'{group_prefix}*': {missing}"
                )

            local_scale = state_dict[local_key]
            if local_scale.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{local_key} must be float8_e4m3fn, got {local_scale.dtype}")

            weight_global = state_dict[weight_global_key].detach().float().reshape(-1)
            input_global = state_dict[input_global_key].detach().float().reshape(-1)
            for name, value in (
                (weight_global_key, weight_global),
                (input_global_key, input_global),
            ):
                if value.numel() != 1 or not torch.isfinite(value).all() or not (value > 0).all():
                    raise ValueError(f"{name} must be one finite positive scalar")
            weight_globals.append(weight_global)
            input_globals.append(input_global)

        reference_input = input_globals[0]
        if any(not torch.equal(reference_input, value) for value in input_globals[1:]):
            values = [float(value.item()) for value in input_globals]
            raise ValueError(
                f"NVFP4 {group} fusion group '{group_prefix}*' has mismatched "
                f"input_global_scale values: {values}"
            )

        tensor_scales = [1.0 / value for value in weight_globals]
        shared_tensor_scale = torch.stack(tensor_scales).max()
        min_tensor_scale = torch.stack(tensor_scales).min()
        max_global_scale_ratio = max(
            max_global_scale_ratio,
            float((shared_tensor_scale / min_tensor_scale).item()),
        )
        shared_weight_global = (1.0 / shared_tensor_scale).reshape(1).to(torch.float32)

        for prefix, tensor_scale in zip(members, tensor_scales):
            local_key = f"{prefix}.weight_scale"
            weight_global_key = f"{prefix}.weight_global_scale"
            local_scale = state_dict[local_key]
            rebase_ratio = (tensor_scale / shared_tensor_scale).to(local_scale.device)
            target = local_scale.float() * rebase_ratio
            if not torch.isfinite(target).all() or not (target > 0).all():
                raise ValueError(f"Invalid rebased NVFP4 block scales for {prefix}")
            if (target < finfo.tiny).any() or (target > finfo.max).any():
                raise ValueError(
                    f"Rebased NVFP4 block scales for {prefix} exceed the supported "
                    f"E4M3 range [{finfo.tiny}, {finfo.max}]"
                )

            rebased = target.to(torch.float8_e4m3fn)
            relative_error = (rebased.float() - target).abs() / target
            layer_max_relative_error = float(relative_error.max().item())
            if layer_max_relative_error > _MAX_E4M3_RELATIVE_ROUNDING_ERROR:
                raise ValueError(
                    f"Rebased NVFP4 block scales for {prefix} exceed the E4M3 "
                    f"rounding error bound: {layer_max_relative_error:.6g}"
                )
            max_local_scale_relative_error = max(
                max_local_scale_relative_error,
                layer_max_relative_error,
            )
            state_dict[local_key] = rebased.contiguous()
            state_dict[weight_global_key] = shared_weight_global.clone()
            rebased_local_scale_count += local_scale.numel()

    return {
        "fused_group_count": len(groups),
        "fused_layer_count": sum(len(members) for _, _, members in groups),
        "rebased_local_scale_count": rebased_local_scale_count,
        "max_global_scale_ratio": max_global_scale_ratio,
        "max_local_scale_relative_error": max_local_scale_relative_error,
    }
