#!/usr/bin/env python3
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

"""Test hidden-state dataloader: report per-batch statistics and calibrate noise std.

Loads one batch from each of the train and eval datasets (using the same
DatasetManager as training) and prints hidden_states shape, std, mean, and
abs-max. When --hidden-noise-std is set, also shows the noise/signal ratio
so you can calibrate the noise level before committing to a full run.

Loads the tokenizer and .ckpt hidden state files only.

Example:
    python3 tools/test_dataloader.py \\
        --train-hidden-path /path/to/hidden_states_train \\
        --eval-hidden-path  /path/to/hidden_states_eval \\
        --target-model      /path/to/model/snapshot \\
        --hidden-noise-std  0.05
"""

import argparse
import sys
import types

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from angelslim.compressor.speculative import (
    DatasetManager,
    GaussianNoise,
    TransformDataset,
    infer_model_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-hidden-path",
        required=True,
        help="Path to training hidden states directory (same as TRAIN_HIDDEN_PATH in training).",
    )
    parser.add_argument(
        "--eval-hidden-path",
        required=True,
        help="Path to eval hidden states directory (same as EVAL_HIDDEN_PATH in training).",
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help=(
            "Local path to the target model snapshot. Used only to load the tokenizer "
            "and infer chat_template_type — no model weights are loaded."
        ),
    )
    parser.add_argument(
        "--hidden-noise-std",
        type=float,
        default=0.0,
        help=(
            "Noise std to test (same value as --hidden_noise_std in training). "
            "0.0 = no augmentation (default). Reports noise/signal ratio when > 0."
        ),
    )
    parser.add_argument(
        "--chat-template-type",
        default=None,
        help=(
            "Chat template type (e.g. 'qwen3_vl'). Auto-detected from --target-model "
            "when not set. Use this to override if auto-detection fails on local paths."
        ),
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=16384,
        help="Maximum sequence length passed to DatasetManager (default: 16384).",
    )
    parser.add_argument(
        "--target-model-type",
        default="qwen3_vl",
        help=(
            "Target model type passed to DatasetManager for VLM collator selection "
            "(default: qwen3_vl). Override when testing against a different VLM "
            "(e.g. qwen2_5_vl, hunyuan_vl)."
        ),
    )
    return parser.parse_args()


def _stats(hs: torch.Tensor, label: str) -> float:
    f = hs.float()
    std = f.std().item()
    print(f"  {label}")
    print(f"    shape  : {tuple(f.shape)}")
    print(f"    dtype  : {hs.dtype}")
    print(f"    std    : {std:.6f}")
    print(f"    mean   : {f.mean().item():.6f}")
    print(f"    absmax : {f.abs().max().item():.6f}")
    return std


def _make_data_args(args, chat_template_type: str) -> types.SimpleNamespace:
    """Build a minimal data_args namespace for DatasetManager."""
    return types.SimpleNamespace(
        training_mode="offline",
        modal_type="VLM",
        train_hidden_path=args.train_hidden_path,
        eval_hidden_path=args.eval_hidden_path,
        target_model_name_or_path=args.target_model,
        train_data_path=None,
        sample_num=None,
        shuffle_seed=42,
        display=False,
        num_proc=4,
        chat_template_type=chat_template_type,
    )


def main():
    args = parse_args()

    # Resolve chat_template_type — use explicit override if provided, else infer.
    if args.chat_template_type:
        chat_template_type = args.chat_template_type
        print(f"chat_template_type : {chat_template_type} (from --chat-template-type)")
    else:
        print(f"Inferring model params from {args.target_model} ...")
        _, _, chat_template_type = infer_model_params(
            model_name_or_path=args.target_model, model_type=None
        )
        if chat_template_type is None:
            print(
                "ERROR: could not infer chat_template_type. "
                "Pass --chat-template-type explicitly (e.g. qwen3_vl).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  chat_template_type: {chat_template_type}")

    # Load tokenizer only (no model weights).
    print("Loading tokenizer ...")
    tokenizer = AutoProcessor.from_pretrained(args.target_model)

    # Build datasets via the same DatasetManager used in training.
    print("Building datasets ...")
    data_args = _make_data_args(args, chat_template_type)
    dataset_manager = DatasetManager(
        data_args=data_args,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=chat_template_type,
        target_model_type=args.target_model_type,
    )
    train_dataset, eval_dataset, data_collator = dataset_manager.create_offline_datasets()
    print(f"  train dataset size : {len(train_dataset)}")
    print(f"  eval  dataset size : {len(eval_dataset) if eval_dataset else 'N/A'}")

    # Eval batch — no noise, measures raw signal scale.
    raw_std = None
    if eval_dataset:
        print("\nEval batch (no noise):")
        eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=data_collator)
        eval_batch = next(iter(eval_loader))
        raw_std = _stats(eval_batch["hidden_states"], "hidden_states")

    # Train batch — with noise if requested.
    if args.hidden_noise_std > 0.0:
        aug_dataset = TransformDataset(train_dataset, GaussianNoise(std=args.hidden_noise_std))
        noise_label = f"noise std={args.hidden_noise_std}"
    else:
        aug_dataset = train_dataset
        noise_label = "no noise"

    print(f"\nTrain batch ({noise_label}):")
    train_loader = DataLoader(aug_dataset, batch_size=1, collate_fn=data_collator)
    train_batch = next(iter(train_loader))
    _stats(train_batch["hidden_states"], "hidden_states")

    # Noise/signal summary.
    if args.hidden_noise_std > 0.0 and raw_std is not None:
        ratio = args.hidden_noise_std / raw_std
        print(
            f"\nNoise/signal ratio  : {ratio:.4f}"
            f"  (noise_std={args.hidden_noise_std} / eval_hs_std={raw_std:.6f})"
        )
        if ratio > 0.2:
            print("  WARNING : noise_std >20% of signal — consider reducing --hidden-noise-std")
        elif ratio < 0.01:
            print("  NOTE    : noise_std <1% of signal — augmentation effect may be negligible")
        else:
            print("  OK      : noise level looks reasonable")

    print("\nDone.")


if __name__ == "__main__":
    main()
