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

import argparse
import os
from pathlib import Path

import transformers
from transformers import AutoProcessor, AutoTokenizer

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    Eagle3TrainerFactory,
    TargetHead,
    create_draft_model,
    get_supported_chat_template_type_strings,
)
from angelslim.utils import rank0_print


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EAGLE3 offline model")

    # Model arguments
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument(
        "--modal_type",
        type=str,
        default="LLM",
        choices=["LLM", "VLM"],
        help="Modal type: LLM for language models, VLM for vision-language models",
    )
    model_group.add_argument(
        "--training_mode",
        type=str,
        default="offline",
        choices=["online", "offline"],
        help="Training mode: online or offline",
    )
    model_group.add_argument(
        "--target_model_name_or_path",
        type=str,
        default=None,
        help="Path to target model, defaults to model_name_or_path if not specified",
    )
    model_group.add_argument(
        "--draft_model_config_path",
        type=str,
        default=None,
        help="Path to draft model config",
    )
    model_group.add_argument(
        "--target_backend",
        type=str,
        default="hf",
        choices=["hf"],
        help=("Target model backend: hf (HuggingFace Transformers)"),
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights: float16, bfloat16, float32",
    )
    model_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust remote code when loading models",
    )
    model_group.add_argument(
        "--lm_head_key",
        type=str,
        default="lm_head.weight",
        help="Key for lm head in model config",
    )
    model_group.add_argument(
        "--sub_config_name",
        type=str,
        default=None,
        help="Usually used for VLMs to specify sub-config name (e.g., 'text_config')",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data Arguments")
    data_group.add_argument(
        "--train_data_path",
        type=str,
        nargs="+",
        required=True,
        help="Path to training data file(s) (JSON format). Can specify multiple files.",
    )
    data_group.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to evaluation data file",
    )
    data_group.add_argument(
        "--train_hidden_path",
        type=str,
        required=True,
        help="Path to training hidden file",
    )
    data_group.add_argument(
        "--eval_hidden_path",
        type=str,
        default=None,
        help="Path to evaluation hidden file",
    )
    data_group.add_argument(
        "--chat_template_type",
        type=str,
        default="qwen3",
        help=(
            f"Chat template type for conversation formatting. "
            f"Supported types: {', '.join(get_supported_chat_template_type_strings())}"
        ),
    )
    data_group.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes for data preprocessing",
    )
    data_group.add_argument(
        "--sample_num",
        type=int,
        default=None,
        help="Number of max samples for data preprocessing",
    )
    data_group.add_argument(
        "--shuffle_seed", type=int, default=42, help="Random seed for shuffling dataset"
    )
    data_group.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display data samples during preprocessing (default: False)",
    )

    # Training arguments
    training_group = parser.add_argument_group("Training Arguments")
    training_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model checkpoints",
    )
    training_group.add_argument(
        "--optim", type=str, default="adamw_torch", help="Optimizer to use"
    )
    training_group.add_argument(
        "--training_time_test_length",
        type=int,
        default=7,
        help="Length of test data for training time",
    )
    training_group.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help=(
            "Maximum sequence length. "
            "Sequences will be right padded (and possibly truncated)."
        ),
    )
    training_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device during training",
    )
    training_group.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device during evaluation",
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Number of updates steps to accumulate before "
            "performing a backward/update pass"
        ),
    )
    training_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform",
    )
    training_group.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Initial learning rate"
    )
    training_group.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to apply"
    )
    training_group.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of steps for warmup"
    )
    training_group.add_argument(
        "--warmup_ratio", type=float, default=0.0, help="Ratio of warmup steps"
    )
    training_group.add_argument(
        "--logging_steps", type=int, default=10, help="Log every X updates steps"
    )
    training_group.add_argument(
        "--save_steps",
        type=float,
        default=500,
        help="Save checkpoint every X updates steps",
    )
    training_group.add_argument(
        "--eval_steps", type=int, default=500, help="Run evaluation every X steps"
    )
    training_group.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints",
    )
    training_group.add_argument(
        "--deepspeed", type=str, default=None, help="DeepSpeed config file"
    )
    training_group.add_argument(
        "--fp16", action="store_true", help="Whether to use fp16 training"
    )
    training_group.add_argument(
        "--bf16", action="store_true", help="Whether to use bf16 training"
    )
    training_group.add_argument(
        "--save_strategy", type=str, default="no", help="Save strategy for checkpoints"
    )
    training_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
        help=(
            "Learning rate scheduler type. "
            "Common options: 'linear', 'cosine', 'cosine_with_restarts', "
            "'polynomial', 'constant', 'constant_with_warmup'"
        ),
    )
    training_group.add_argument(
        "--run_name", type=str, default=None, help="Run name for tracking"
    )
    training_group.add_argument(
        "--report_to",
        type=str,
        default="none",
        help=(
            "The list of integrations to report the results and logs to. "
            "Supported platforms: 'tensorboard', 'wandb', 'mlflow', 'all', 'none'"
        ),
    )

    return parser.parse_args()


def train():
    args = parse_args()

    # Create draft model
    rank0_print("Loading draft model...")
    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
    draft_model = create_draft_model(draft_model_config)
    draft_model.load_embed_weights(args.target_model_name_or_path)
    draft_model.freeze_embed_weights()
    rank0_print("Draft model loaded successfully")

    # Load target head for computing logits from hidden states
    rank0_print("Loading target head...")
    target_head = TargetHead.from_pretrained(
        args.target_model_name_or_path,
        lm_head_key=args.lm_head_key,
        sub_config_name=args.sub_config_name,
    )
    rank0_print("Target head loaded successfully")

    # Load tokenizer
    rank0_print("Loading tokenizer...")
    if args.modal_type == "LLM":
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path)
    else:
        tokenizer = AutoProcessor.from_pretrained(args.target_model_name_or_path)

    # Create all datasets using unified DatasetManager
    rank0_print("Creating datasets...")
    rank0_print("- Offline mode: Loading pre-computed hidden states from .ckpt files")
    rank0_print(
        "- Online mode: Processing raw conversation data "
        f"(chat template: {args.chat_template_type})"
    )

    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
    )

    (
        offline_train_dataset,
        offline_eval_dataset,
        online_train_dataset,
        online_eval_dataset,
        data_collator,
    ) = dataset_manager.create_all_datasets()

    rank0_print(
        f"Offline train dataset size: {len(offline_train_dataset)}, "
        "Offline eval dataset size: "
        f"{len(offline_eval_dataset) if offline_eval_dataset else 0}"
    )

    # Build vocabulary mapping for draft model using online training dataset
    rank0_print("Building vocabulary mapping for draft model...")
    if online_train_dataset is not None:
        cache_path = os.path.join(args.output_dir, "vocab_mapping_cache.pt")
        draft_model.build_vocab_mapping(
            dataset=online_train_dataset,
            cache_path=cache_path,
        )
        rank0_print("Vocabulary mapping built successfully")
    else:
        rank0_print(
            "Warning: No online training dataset available, "
            "skipping vocab mapping build"
        )

    # Create a TrainingArguments object for the trainer
    # Organize training arguments by category
    basic_args = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
    }

    batch_args = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    optimizer_args = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
    }

    precision_args = {
        "fp16": args.fp16,
        "bf16": args.bf16,
    }

    checkpoint_args = {
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
    }

    logging_args = {
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "report_to": args.report_to,
        "run_name": args.run_name,
    }

    distributed_args = {
        "deepspeed": args.deepspeed,
    }

    training_args = transformers.TrainingArguments(
        **basic_args,
        **batch_args,
        **optimizer_args,
        **precision_args,
        **checkpoint_args,
        **logging_args,
        **distributed_args,
        remove_unused_columns=False,
    )

    # Initialize trainer with offline datasets
    rank0_print("Initializing trainer...")
    trainer = Eagle3TrainerFactory.create(
        training_mode=args.training_mode,
        modal_type=args.modal_type,
        draft_model=draft_model,
        target_head=target_head,
        length=args.training_time_test_length,
        args=training_args,
        train_dataset=offline_train_dataset,
        eval_dataset=offline_eval_dataset,
        data_collator=data_collator,
    )

    # Start training
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting training...")
        trainer.train()
    rank0_print("Training completed!")


if __name__ == "__main__":
    train()
