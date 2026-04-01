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
    infer_model_params,
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
        default=None,
        help=(
            "Key for lm head in model config. you can find it in model.safetensors.index.json. "
            "If not specified, will be auto deduced from target_model's model_type. "
            "Examples: lm_head.weight (default LLM), model.embed_tokens.weight (HunyuanOCR), "
            "model.language_model.embed_tokens.weight (Qwen3-VL)"
        ),
    )
    model_group.add_argument(
        "--embed_weight_key",
        type=str,
        default=None,
        help=(
            "Key for embedding weights in model config. "
            "If not specified, will be auto deduced from target_model's model_type. "
            "Examples: model.embed_tokens.weight (default LLM/HunyuanOCR), "
            "model.language_model.embed_tokens.weight (Qwen3-VL)"
        ),
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
        default=None,
        help=(
            "Chat template type for conversation formatting. "
            "If not specified, will be auto deduced from target_model's model_type. "
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
            "Maximum sequence length. " "Sequences will be right padded (and possibly truncated)."
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
        help=("Number of updates steps to accumulate before " "performing a backward/update pass"),
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
    training_group.add_argument("--fp16", action="store_true", help="Whether to use fp16 training")
    training_group.add_argument("--bf16", action="store_true", help="Whether to use bf16 training")
    training_group.add_argument(
        "--save_strategy", type=str, default="no", help="Save strategy for checkpoints"
    )
    training_group.add_argument(
        "--eval_strategy", type=str, default="no", help="Eval strategy for checkpoints"
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
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Whether to use gradient checkpointing",
    )
    training_group.add_argument("--run_name", type=str, default=None, help="Run name for tracking")
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

    rank0_print(f"Loading draft model: {args.draft_model_config_path}")
    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
    target_model_type = getattr(draft_model_config, "target_model_type", None)
    rank0_print(f"target_model_type from draft config: {target_model_type}")

    inferred_lm_head_key, inferred_embed_weight_key, inferred_chat_template_type = (
        infer_model_params(
            model_name_or_path=args.target_model_name_or_path,
            model_type=target_model_type,
        )
    )
    if args.lm_head_key is None:
        if inferred_lm_head_key is None:
            raise ValueError("lm_head_key not specified and cannot be auto deduced")
        else:
            args.lm_head_key = inferred_lm_head_key
            rank0_print(f"lm_head_key not specified, auto deduced: {args.lm_head_key}")

    if args.embed_weight_key is None:
        if inferred_embed_weight_key is None:
            raise ValueError("embed_weight_key not specified and cannot be auto deduced")
        else:
            args.embed_weight_key = inferred_embed_weight_key
            rank0_print(f"embed_weight_key not specified, auto deduced: {args.embed_weight_key}")

    if args.chat_template_type is None:
        if inferred_chat_template_type is None:
            raise ValueError("chat_template_type not specified and cannot be auto deduced")
        else:
            args.chat_template_type = inferred_chat_template_type
            rank0_print(
                f"chat_template_type not specified, auto deduced: {args.chat_template_type}"
            )

    # Create draft model
    draft_model = create_draft_model(draft_model_config)
    draft_model.load_embed_weights(args.target_model_name_or_path, args.embed_weight_key)
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
        target_model_type=None if args.modal_type in ("LLM", "TTS") else target_model_type,
    )

    (
        offline_train_dataset,
        offline_eval_dataset,
        data_collator,
    ) = dataset_manager.create_offline_datasets()

    rank0_print(
        f"Offline train dataset size: {len(offline_train_dataset)}, "
        "Offline eval dataset size: "
        f"{len(offline_eval_dataset) if offline_eval_dataset else 0}"
    )

    # Build vocabulary mapping for draft model from pre-computed vocab mapping
    rank0_print("Loading vocabulary mapping for draft model...")
    vocab_mapping_path = os.path.join(args.train_hidden_path, "vocab_mapping.pt")
    if os.path.exists(vocab_mapping_path):
        rank0_print(f"Loading vocab mapping from {vocab_mapping_path}...")
        draft_model.load_vocab_mapping(vocab_mapping_path=vocab_mapping_path)
        rank0_print("Vocabulary mapping loaded successfully")
    else:
        raise ValueError(f"vocab_mapping.pt not found at {vocab_mapping_path}")

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
        "eval_strategy": args.eval_strategy,
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
        "gradient_checkpointing": args.gradient_checkpointing,
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

    # Save final model to output_dir
    rank0_print(f"Saving final model to {training_args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    rank0_print("Final model saved successfully!")


if __name__ == "__main__":
    train()
