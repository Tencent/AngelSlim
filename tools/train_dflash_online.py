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

"""DFlash Online Training Script.

Based on train_eagle3_online.py but adapted for DFlash's block-parallel
cross-attention training approach.
"""

import argparse
import os
from pathlib import Path

import torch
import transformers

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    Eagle3TrainerFactory,
    create_draft_model,
    create_target_model,
    get_supported_chat_template_type_strings,
)
from angelslim.utils import rank0_print


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DFlash online model")

    # Model arguments
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument(
        "--modal_type",
        type=str,
        default="DFlash",
        help="Modal type, should be DFlash for DFlash training",
    )
    model_group.add_argument(
        "--training_mode",
        type=str,
        default="online",
        choices=["online"],
        help="Training mode (only online is supported for DFlash)",
    )
    model_group.add_argument(
        "--target_model_name_or_path",
        type=str,
        default=None,
        help="Path to target model",
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
        help="Target model backend: hf (HuggingFace Transformers)",
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    model_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust remote code when loading models",
    )
    model_group.add_argument(
        "--embed_weight_key",
        type=str,
        default="model.embed_tokens.weight",
        help="Key for embedding weights in model config",
    )

    # DFlash-specific arguments
    dflash_group = parser.add_argument_group("DFlash Arguments")
    dflash_group.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for DFlash parallel prediction",
    )
    dflash_group.add_argument(
        "--num_anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    dflash_group.add_argument(
        "--loss_decay_gamma",
        type=float,
        default=None,
        help=(
            "Gamma for exponential loss decay weighting. "
            "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. "
            "None disables decay."
        ),
    )
    dflash_group.add_argument(
        "--attention_backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model",
    )
    dflash_group.add_argument(
        "--mask_token_id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, uses config or auto-detect.",
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
        help="Path to evaluation data file (JSON format)",
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
        help="Display data samples during preprocessing",
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
        default=1,
        help="Not used for DFlash (kept for compatibility)",
    )
    training_group.add_argument(
        "--model_max_length",
        type=int,
        default=3072,
        help="Maximum sequence length",
    )
    training_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per device during training",
    )
    training_group.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size per device during evaluation",
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    training_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=6,
        help="Total number of training epochs to perform",
    )
    training_group.add_argument(
        "--learning_rate", type=float, default=6e-4, help="Initial learning rate"
    )
    training_group.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to apply"
    )
    training_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    training_group.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of steps for warmup"
    )
    training_group.add_argument(
        "--warmup_ratio", type=float, default=0.04, help="Ratio of warmup steps"
    )
    training_group.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps"
    )
    training_group.add_argument(
        "--save_steps",
        type=float,
        default=5000,
        help="Save checkpoint every X updates steps",
    )
    training_group.add_argument(
        "--eval_steps", type=int, default=1000, help="Run evaluation every X steps"
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
        "--eval_strategy", type=str, default="no", help="Evaluation strategy"
    )
    training_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type",
    )
    training_group.add_argument("--run_name", type=str, default=None, help="Run name for tracking")
    training_group.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="The list of integrations to report the results and logs to (e.g. 'wandb')",
    )

    # WandB arguments (mirrors SpecForge's --wandb-project / --wandb-name)
    wandb_group = parser.add_argument_group("WandB Arguments")
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name. Overrides WANDB_PROJECT env var if set.",
    )
    wandb_group.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name. Overrides --run_name if both are set.",
    )

    return parser.parse_args()


def _setup_wandb(args) -> None:
    """Set up WandB environment variables and initialize wandb run on rank 0.

    Mirrors the --wandb-project / --wandb-name pattern from SpecForge.
    Priority: CLI args > env vars > defaults.
    """
    if args.report_to not in ("wandb", "all"):
        return

    # CLI args take priority over env vars
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Resolve run name: --wandb_run_name > --run_name > env WANDB_RUN_NAME
    run_name = args.wandb_run_name or args.run_name or os.environ.get("WANDB_RUN_NAME")
    if run_name:
        os.environ["WANDB_RUN_NAME"] = run_name
        # Propagate back so TrainingArguments picks it up
        args.run_name = run_name

    # Explicit wandb.init() on rank 0 so project/name are registered immediately
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        try:
            import wandb

            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "angelslim-dflash"),
                name=run_name,
                resume="allow",
            )
        except ImportError:
            print("[WARNING] wandb not installed. " "Install via: pip install wandb")


def train():
    args = parse_args()
    _setup_wandb(args)

    # Parse torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping.get(args.torch_dtype, torch.bfloat16)

    rank0_print("Loading draft model config...")
    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
    target_model_type = getattr(draft_model_config, "target_model_type", None)

    # Inject DFlash-specific config into the draft model config
    # so the trainer can access them
    draft_model_config.target_model_name_or_path = args.target_model_name_or_path
    draft_model_config.embed_weight_key = args.embed_weight_key
    draft_model_config.trust_remote_code = args.trust_remote_code

    # Override DFlash params from CLI if specified
    if args.block_size is not None:
        draft_model_config.block_size = args.block_size
    if args.num_anchors is not None:
        draft_model_config.num_anchors = args.num_anchors
    if args.loss_decay_gamma is not None:
        draft_model_config.loss_decay_gamma = args.loss_decay_gamma
    if args.attention_backend is not None:
        draft_model_config.attention_backend = args.attention_backend
    if args.mask_token_id is not None:
        if (
            not hasattr(draft_model_config, "dflash_config")
            or draft_model_config.dflash_config is None
        ):
            draft_model_config.dflash_config = {}
        draft_model_config.dflash_config["mask_token_id"] = args.mask_token_id

    # Set attention implementation
    draft_model_config._attn_implementation = args.attention_backend

    # Create target model with specified backend
    rank0_print(f"Loading target model with {args.target_backend} backend...")
    target_model = create_target_model(
        backend=args.target_backend,
        model_path=args.target_model_name_or_path,
        modal_type="LLM",  # DFlash uses standard LLM target model
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        target_model_type=target_model_type,
    )
    rank0_print("Target model loaded successfully")

    # Configure target model to capture the right layers for DFlash
    dflash_config = getattr(draft_model_config, "dflash_config", {}) or {}
    target_layer_ids = dflash_config.get("target_layer_ids", None)
    if target_layer_ids is not None:
        # Set aux_hidden_states_layer_ids to match DFlash's target_layer_ids
        draft_model_config.aux_hidden_states_layer_ids = target_layer_ids
        rank0_print(f"DFlash target layer IDs: {target_layer_ids}")

    # Create draft model
    rank0_print("Loading draft model...")
    rank0_print(f"draft_model_config: {draft_model_config}")
    draft_model = create_draft_model(draft_model_config)
    rank0_print("Draft model loaded successfully")
    rank0_print(f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}")

    # Create datasets using DatasetManager
    rank0_print(
        "Creating training and evaluation datasets "
        f"with chat template type: {args.chat_template_type}..."
    )
    # DatasetBuilderFactory doesn't know "DFlash"; DFlash uses the same data
    # format as "LLM", so temporarily override modal_type for dataset creation.
    args.modal_type = "LLM"
    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        display=args.display,
        target_model_type=target_model_type,
    )
    args.modal_type = "DFlash"  # restore for trainer factory
    train_dataset, eval_dataset, data_collator = dataset_manager.create_online_datasets()
    rank0_print(
        f"Train dataset size: {len(train_dataset)}, "
        f"Eval dataset size: {len(eval_dataset) if eval_dataset else 0}"
    )

    # Create TrainingArguments
    basic_args = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
    }

    batch_args = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "remove_unused_columns": False,
    }

    optimizer_args = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
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
    }

    training_args = transformers.TrainingArguments(
        **basic_args,
        **batch_args,
        **optimizer_args,
        **precision_args,
        **checkpoint_args,
        **logging_args,
        **distributed_args,
    )

    # Initialize trainer
    rank0_print("Initializing DFlash trainer...")
    trainer = Eagle3TrainerFactory.create(
        training_mode=args.training_mode,
        modal_type=args.modal_type,
        draft_model=draft_model,
        target_model=target_model,
        length=args.training_time_test_length,
        draft_model_config=draft_model_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start training
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting DFlash training...")
        trainer.train()
    rank0_print("DFlash training completed!")


if __name__ == "__main__":
    train()
