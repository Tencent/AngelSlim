# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

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
    weight = embedding.weight.to(torch.float64)
    new_weight = weight - weight.mean(dim=-1, keepdim=True)
    new_weight = new_weight.to(weight_dtype)
    embedding.weight.data = new_weight


# [TODO] check this function correct or not
@torch.no_grad()
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


@torch.no_grad()
def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)
    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
