#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# DFlash offline training script.
# Trains a DFlash draft model using pre-computed hidden states (.ckpt files).
#
# Workflow:
#   Step 1 (data generation):
#       bash scripts/speculative/generate_qwen3_dflash_data.sh
#   Step 2 (offline training, this script):
#       bash scripts/speculative/run_qwen3_dflash_offline.sh

import argparse
import os
from pathlib import Path

import torch
import transformers

from angelslim.compressor.speculative import (
    DraftModelConfig,
    Eagle3TrainerFactory,
    create_draft_model,
    get_supported_chat_template_type_strings,
)
from angelslim.compressor.speculative.train.data.data_utils import (
    DataCollatorWithPadding,
)
from angelslim.compressor.speculative.train.data.dataset_builder.offline_dataset_builder import (
    OfflineEagle3Dataset,
)
from angelslim.utils import rank0_print

# ---------------------------------------------------------------------------
# Offline DFlash Dataset
# ---------------------------------------------------------------------------


class OfflineDFlashDataset(OfflineEagle3Dataset):
    """
    DFlash variant of the offline dataset.

    Each .ckpt file must contain:
        - input_ids:      LongTensor  [1, S]
        - hidden_states:  BFloat16Tensor [1, S, D*num_target_layers]   ← multi-layer hidden states
        - loss_mask:      LongTensor  [1, S]
        - attention_mask: LongTensor  [1, S]   (auto-generated if missing)

    Note: DFlash does NOT need target_hiddens (only Eagle3 offline uses the
    single final-layer hidden state). The multi-layer hidden_states is the
    context feature passed directly to the DFlash cross-attention.
    """

    REQUIRED_KEYS = ["input_ids", "hidden_states", "loss_mask"]

    def _load_ckpt(self, idx: int):
        import warnings

        ckpt_path = self.ckpt_files[idx]
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            warnings.warn(
                f"Failed to load {ckpt_path}: {e}. Skipping.", RuntimeWarning, stacklevel=2
            )
            return None

        missing = [k for k in self.REQUIRED_KEYS if k not in data]
        if missing:
            warnings.warn(
                f"{ckpt_path} missing keys {missing}. Skipping.", RuntimeWarning, stacklevel=2
            )
            return None

        # Auto-generate attention_mask if absent
        if "attention_mask" not in data:
            data["attention_mask"] = torch.ones_like(data["input_ids"])

        return data


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash draft model (offline mode)")

    # Model
    m = parser.add_argument_group("Model Arguments")
    m.add_argument("--target_model_name_or_path", type=str, required=True)
    m.add_argument("--draft_model_config_path", type=str, required=True)
    m.add_argument(
        "--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    m.add_argument("--trust_remote_code", action="store_true", default=True)
    m.add_argument("--embed_weight_key", type=str, default="model.embed_tokens.weight")
    m.add_argument("--lm_head_key", type=str, default="lm_head.weight")

    # DFlash-specific (override values in config JSON)
    d = parser.add_argument_group("DFlash Arguments")
    d.add_argument("--block_size", type=int, default=None)
    d.add_argument("--num_anchors", type=int, default=None)
    d.add_argument("--loss_decay_gamma", type=float, default=None)
    d.add_argument("--mask_token_id", type=int, default=None)
    d.add_argument(
        "--attention_backend", type=str, default=None, choices=["flex_attention", "sdpa", "eager"]
    )

    # Data
    da = parser.add_argument_group("Data Arguments")
    da.add_argument(
        "--train_hidden_path",
        type=str,
        required=True,
        help="Directory of pre-computed training .ckpt files",
    )
    da.add_argument(
        "--eval_hidden_path",
        type=str,
        default=None,
        help="Directory of pre-computed eval .ckpt files (optional)",
    )
    da.add_argument(
        "--chat_template_type",
        type=str,
        default="qwen3",
        help=f"Supported: {', '.join(get_supported_chat_template_type_strings())}",
    )
    da.add_argument("--model_max_length", type=int, default=3072)
    da.add_argument("--num_proc", type=int, default=16)
    da.add_argument(
        "--cache_in_memory",
        action="store_true",
        default=False,
        help="Cache all .ckpt files in RAM (fast but memory-intensive)",
    )

    # Training
    t = parser.add_argument_group("Training Arguments")
    t.add_argument("--output_dir", type=str, required=True)
    t.add_argument("--optim", type=str, default="adamw_torch")
    t.add_argument("--num_train_epochs", type=int, default=6)
    t.add_argument("--per_device_train_batch_size", type=int, default=2)
    t.add_argument("--per_device_eval_batch_size", type=int, default=2)
    t.add_argument("--gradient_accumulation_steps", type=int, default=1)
    t.add_argument("--learning_rate", type=float, default=6e-4)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--warmup_steps", type=int, default=0)
    t.add_argument("--warmup_ratio", type=float, default=0.04)
    t.add_argument("--max_grad_norm", type=float, default=1.0)
    t.add_argument("--logging_steps", type=int, default=50)
    t.add_argument("--save_steps", type=float, default=2500)
    t.add_argument("--save_total_limit", type=int, default=None)
    t.add_argument("--eval_steps", type=int, default=500)
    t.add_argument("--save_strategy", type=str, default="steps")
    t.add_argument("--eval_strategy", type=str, default="no")
    t.add_argument("--lr_scheduler_type", type=str, default="cosine")
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--deepspeed", type=str, default=None)
    t.add_argument("--report_to", type=str, default="none")
    t.add_argument("--run_name", type=str, default=None)
    t.add_argument("--training_time_test_length", type=int, default=7)

    # WandB
    w = parser.add_argument_group("WandB Arguments")
    w.add_argument("--wandb_project", type=str, default=None)
    w.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def _setup_wandb(args):
    if args.report_to not in ("wandb", "all"):
        return
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = args.wandb_run_name or args.run_name or os.environ.get("WANDB_RUN_NAME")
    if run_name:
        os.environ["WANDB_RUN_NAME"] = run_name
        args.run_name = run_name
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
            print("[WARNING] wandb not installed. Install via: pip install wandb")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train():
    args = parse_args()
    _setup_wandb(args)

    # dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    # torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    # ------------------------------------------------------------------
    # 1. Draft model config
    # ------------------------------------------------------------------
    rank0_print("Loading draft model config...")
    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
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
        draft_model_config._attn_implementation = args.attention_backend
    if args.mask_token_id is not None:
        dfc = getattr(draft_model_config, "dflash_config", None) or {}
        dfc["mask_token_id"] = args.mask_token_id
        draft_model_config.dflash_config = dfc

    # ------------------------------------------------------------------
    # 2. Draft model
    # ------------------------------------------------------------------
    rank0_print("Loading draft model...")
    draft_model = create_draft_model(draft_model_config)
    rank0_print(f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}")

    # ------------------------------------------------------------------
    # 3. Offline datasets
    # ------------------------------------------------------------------
    rank0_print(f"Loading offline training data from: {args.train_hidden_path}")
    train_dataset = OfflineDFlashDataset(
        data_dir=args.train_hidden_path,
        file_pattern="*.ckpt",
        cache_in_memory=args.cache_in_memory,
    )
    rank0_print(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if args.eval_hidden_path:
        rank0_print(f"Loading offline eval data from: {args.eval_hidden_path}")
        eval_dataset = OfflineDFlashDataset(
            data_dir=args.eval_hidden_path,
            file_pattern="*.ckpt",
            cache_in_memory=args.cache_in_memory,
        )
        rank0_print(f"Eval samples: {len(eval_dataset)}")

    data_collator = DataCollatorWithPadding()

    # ------------------------------------------------------------------
    # 4. TrainingArguments
    # ------------------------------------------------------------------
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.run_name,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # 5. Trainer -- use Eagle3TrainerFactory
    # ------------------------------------------------------------------
    rank0_print("Initializing trainer...")
    trainer = Eagle3TrainerFactory.create(
        training_mode="offline",
        modal_type="DFlash",
        draft_model=draft_model,
        target_model=None,  # Not needed — hidden states are pre-computed
        length=args.training_time_test_length,
        draft_model_config=draft_model_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    output_dir = Path(training_args.output_dir)
    if list(output_dir.glob("checkpoint-*")):
        rank0_print("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting fresh training run...")
        trainer.train()

    rank0_print("Training completed!")


if __name__ == "__main__":
    train()
