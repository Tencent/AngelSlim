#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# DFlash offline data pre-generation script.
#
# Usage:
#   torchrun --nproc_per_node=8 tools/generate_dflash_data.py \
#       --target_model_name_or_path /path/to/Qwen3-4B \
#       --draft_model_config_path configs/qwen3_dflash.json \
#       --train_data_path /path/to/data.jsonl \
#       --output_dir /path/to/output/ckpts \
#       --model_max_length 3072 \
#       --chat_template_type qwen3
#
# Each output .ckpt file contains:
#   - input_ids:      LongTensor  [1, S]
#   - hidden_states:  BFloat16Tensor [1, S, D*num_target_layers]  (multi-layer concat)
#   - loss_mask:      LongTensor  [1, S]
#   - attention_mask: LongTensor  [1, S]

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    create_target_model,
    get_supported_chat_template_type_strings,
)
from angelslim.utils import rank0_print


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-generate DFlash training data (hidden states) from target model"
    )

    # Model
    parser.add_argument("--target_model_name_or_path", type=str, required=True)
    parser.add_argument("--draft_model_config_path", type=str, required=True)
    parser.add_argument(
        "--target_backend",
        type=str,
        default="hf",
        choices=["hf"],
        help="Target model backend",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    # Data
    parser.add_argument(
        "--train_data_path", type=str, nargs="+", required=True, help="Input JSONL file(s)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save .ckpt files"
    )
    parser.add_argument(
        "--chat_template_type",
        type=str,
        default="qwen3",
        help=f"Supported: {', '.join(get_supported_chat_template_type_strings())}",
    )
    parser.add_argument("--model_max_length", type=int, default=3072)
    parser.add_argument(
        "--block_size", type=int, default=16, help="Block size for DFlash parallel prediction"
    )
    parser.add_argument(
        "--num_proc", type=int, default=16, help="Workers for tokenization (dataset.map)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Samples per forward pass (keep at 1 for variable-length seqs)",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument(
        "--sample_num", type=int, default=None, help="Limit number of samples (for debugging)"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="Save a new sub-directory every N files (0 = no sharding)",
    )

    return parser.parse_args()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank():
    return int(os.environ.get("RANK", 0))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def init_distributed():
    if get_world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)


def main():
    args = parse_args()
    init_distributed()

    rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    # --------------------------------------------------------------------------
    # 1. Load draft-model config (to get target_layer_ids)
    # --------------------------------------------------------------------------
    rank0_print("Loading draft model config...")
    draft_model_config = DraftModelConfig.from_file(args.draft_model_config_path)
    dflash_config = getattr(draft_model_config, "dflash_config", {}) or {}
    target_layer_ids = dflash_config.get("target_layer_ids", None)
    if target_layer_ids is None:
        raise ValueError(
            "dflash_config.target_layer_ids not found in draft_model_config. "
            f"Please set it in {args.draft_model_config_path}"
        )
    rank0_print(f"DFlash target layer IDs: {target_layer_ids}")

    # --------------------------------------------------------------------------
    # 2. Load target model (on this rank's GPU)
    # --------------------------------------------------------------------------
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    rank0_print("Loading target model...")
    target_model = create_target_model(
        backend=args.target_backend,
        model_path=args.target_model_name_or_path,
        modal_type="LLM",
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    rank0_print("Target model loaded successfully")

    # --------------------------------------------------------------------------
    # 3. Tokenize dataset (using DatasetManager, same as online training)
    # --------------------------------------------------------------------------
    rank0_print("Building dataset...")
    # Temporarily patch args so DatasetManager picks the correct builder
    args.modal_type = "LLM"  # DFlash uses the LLM tokenisation path
    args.training_mode = "online"  # We want the text→token builder, not offline .ckpt loader

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_name_or_path, trust_remote_code=True
    )

    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
    )

    # Restore modal_type to DFlash so DFlash-specific filtering (min_loss_tokens) applies
    args.modal_type = "DFlash"

    (
        _,  # offline_train_dataset  (unused here)
        _,  # offline_eval_dataset
        online_train_dataset,
        _,  # online_eval_dataset
        _,  # data_collator
    ) = dataset_manager.create_all_datasets()

    if online_train_dataset is None:
        raise RuntimeError("No training dataset was created. Check --train_data_path.")

    rank0_print(f"Dataset size: {len(online_train_dataset)}")

    # --------------------------------------------------------------------------
    # 4. Distributed sampler: each rank processes its own shard
    # --------------------------------------------------------------------------
    sampler = DistributedSampler(
        online_train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    def collate_fn(batch):
        """Simple collate: each element is already a dict of 1-D tensors."""
        result = {}
        for key in batch[0]:
            tensors = [item[key] for item in batch]
            # Tensors may be 1-D or 2-D ([1, S]) — keep original shape
            try:
                result[key] = torch.stack(tensors)
            except Exception:
                result[key] = tensors
        return result

    dataloader = DataLoader(
        online_train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------------------------
    # 5. Output directory setup
    # --------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank0_print(f"Saving .ckpt files to: {output_dir}")
    rank0_print(f"World size={world_size}, this rank={rank}")

    # --------------------------------------------------------------------------
    # 6. Main loop: forward target model, save hidden states
    # --------------------------------------------------------------------------
    global_idx = 0  # index within this rank's portion
    total = len(dataloader)
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        # Shape may be [B, 1, S] or [B, S] depending on how dataset stores it
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)  # → [B, S]

        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(1)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(1)

        input_ids = input_ids.to(f"cuda:{local_rank}")
        attention_mask = attention_mask.to(f"cuda:{local_rank}")

        # Run target model
        hidden_states, _ = target_model.get_hidden_states_and_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=target_layer_ids,
        )
        # hidden_states: [B, S, D*len(target_layer_ids)]

        # Save one .ckpt per sample in the batch
        for i in range(input_ids.size(0)):
            # Use a globally unique name: rank × position within rank
            sample_idx = rank * len(dataloader) + global_idx
            global_idx += 1

            if args.shard_size > 0:
                shard_id = sample_idx // args.shard_size
                save_dir = output_dir / f"shard_{shard_id:05d}"
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = output_dir

            ckpt_path = save_dir / f"sample_{sample_idx:08d}_rank{rank}.ckpt"

            ckpt = {
                "input_ids": input_ids[i : i + 1].cpu(),  # [1, S]
                "hidden_states": hidden_states[i : i + 1].cpu().to(torch.bfloat16),  # [1, S, D*L]
                "loss_mask": loss_mask[i : i + 1].cpu(),  # [1, S]
                "attention_mask": attention_mask[i : i + 1].cpu(),  # [1, S]
            }
            torch.save(ckpt, ckpt_path)

        # Progress log
        if batch_idx % 100 == 0:
            elapsed = time.time() - t0
            samples_done = (batch_idx + 1) * args.batch_size
            speed = samples_done / elapsed if elapsed > 0 else 0
            rank0_print(
                f"[rank {rank}] {batch_idx + 1}/{total} batches | "
                f"{speed:.1f} samples/s | elapsed {elapsed:.0f}s"
            )

    if world_size > 1:
        dist.barrier()

    rank0_print(f"Data generation complete. " f"Saved files to {output_dir}")


if __name__ == "__main__":
    main()
