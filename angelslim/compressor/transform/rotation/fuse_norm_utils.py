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
# # Adapted from QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).

import typing

import torch


@torch.no_grad()
def center_embeddings(embedding: torch.nn.Module):
    """
    Shift each embedding to have a mean of zero

    :param embedding: embedding module containing embeddings to center
    """
    if not hasattr(embedding, "weight"):
        raise ValueError(f"Cannot fuse norm of type {type(embedding)}")

    weight_dtype = embedding.weight.dtype
    weight = embedding.weight.to(torch.float32)
    new_weight = weight - weight.mean(dim=-1, keepdim=True)
    new_weight = new_weight.to(weight_dtype)
    embedding.weight.data = new_weight


@torch.no_grad()
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.float()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.float()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


@torch.no_grad()
def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    ln_weight = layernorm.weight.float().cpu()
    ln_bias = layernorm.bias.float().cpu() if hasattr(layernorm, "bias") else None

    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        linear_device = linear.weight.device

        W_ = linear.weight.data.float().cpu()
        new_weight = (W_ * ln_weight).to(linear_dtype)
        linear.weight.data = new_weight.to(linear_device)

        if ln_bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float32)
                )
            new_bias = (linear.bias.data.float().cpu() + torch.matmul(W_, ln_bias)).to(
                linear_dtype
            )
            linear.bias.data = new_bias.to(linear_device)

    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)
    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
