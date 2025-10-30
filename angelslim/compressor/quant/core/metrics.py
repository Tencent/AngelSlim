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

import torch


def mse_loss(
    y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute mse loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )

    mse = (y_pred - y_real) ** 2

    if reduction == "mean":
        return torch.mean(mse)
    elif reduction == "sum":
        return torch.sum(mse)
    elif reduction == "none":
        return mse
    else:
        raise ValueError("Unsupported reduction method.")


def snr_loss(
    y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)

    SNR can be calcualted as following equation:

        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2

    if x and y are matrixs, SNR error over matrix should be the
    mean value of SNR error over all elements.

        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)
    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.
    Raises:
        ValueError: _description_
        ValueError: _description_
    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute snr loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == "mean":
        return torch.mean(snr)
    elif reduction == "sum":
        return torch.sum(snr)
    elif reduction == "none":
        return snr
    else:
        raise ValueError("Unsupported reduction method.")


class LossFilter:
    def __init__(self, processor, model_config=None):
        self.processor = processor
        self.model_config = model_config or {}
        self.filter_tokens_map = {
            "QwenVL": [151643, 151654, 151655, 151656],
            "Qwen3VL": [151643, 151654, 151655, 151656],
            "default": [],  # default: do not filter any tokens
        }

    def get_filter_tokens(self, model_type=None):
        """Get the filter token for a specific model type"""
        if model_type and model_type in self.filter_tokens_map:
            return self.filter_tokens_map[model_type]
        return self.filter_tokens_map["default"]

    def filter_loss(self, loss, labels, model_type=None, custom_filter_tokens=None):
        """
        Filter loss corresponding to specific tokens

        Args:
            loss: precomputed loss, shape same as labels
            labels: target labels
            model_type: model type
            custom_filter_tokens: custom filter tokens, will override preset values

        Returns:
            filtered average loss
        """
        # Get the list of tokens to filter
        if custom_filter_tokens is not None:
            filter_tokens = custom_filter_tokens
        else:
            filter_tokens = self.get_filter_tokens(model_type)

        # Flatten loss and labels for processing
        loss_flat = loss.view(-1)
        labels_flat = labels.view(-1)

        # Create mask for valid tokens
        # First exclude padding tokens
        valid_mask = labels_flat != self.processor.tokenizer.pad_token_id

        # Exclude special tokens that need to be filtered
        for token_id in filter_tokens:
            valid_mask = valid_mask & (labels_flat != token_id)

        # Keep only loss for valid tokens
        filtered_loss = loss_flat[valid_mask]

        # Use assert to check if all tokens are filtered
        assert (
            len(filtered_loss) > 0
        ), "All tokens have been filtered, check your filter configuration"

        # Return average loss
        return filtered_loss
