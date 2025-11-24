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
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from angelslim.compressor.speculative import DatasetManager, create_target_model
from angelslim.utils import decide_device_for_distributed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Rank %(rank)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """
    Setup distributed training environment.

    Returns:
        Tuple of (rank, world_size, local_rank) or (0, 1, 0) if not distributed
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize process group
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single process mode
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class HiddenStateGenerator:
    """Generator for creating hidden states from target model."""

    def __init__(
        self, target_model, output_dir: str, group_size: int = 5000, rank: int = 0
    ):
        """
        Initialize the hidden state generator.

        Args:
            target_model: The target model for generating hidden states
            output_dir: Directory to save generated hidden states
            group_size: Number of samples per subdirectory group
            rank: Process rank for distributed training
        """
        self.target_model = target_model
        self.output_dir = Path(output_dir)
        self.group_size = group_size
        self.rank = rank
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, idx: int) -> Path:
        """
        Get the output file path for a given sample index.

        Args:
            idx: Sample index

        Returns:
            Path object for the output file
        """
        start = (idx // self.group_size) * self.group_size
        end = start + self.group_size
        grouped_subdir = f"rows_{start}-{end}"
        grouped_path = self.output_dir / grouped_subdir
        grouped_path.mkdir(parents=True, exist_ok=True)

        return grouped_path / f"data_{idx}.ckpt"

    def _process_single_sample(self, idx: int, row: Dict[str, Any]) -> bool:
        """
        Process a single sample and save its hidden states.

        Args:
            idx: Sample index
            row: Sample data containing input_ids and loss_mask

        Returns:
            True if processing succeeded, False otherwise
        """
        output_file = self._get_output_path(idx)

        # Skip if file already exists
        if output_file.exists():
            logger.debug(
                f"Skipping existing file: {output_file}", extra={"rank": self.rank}
            )
            return True

        try:
            # Generate aux and target hiddens
            device = decide_device_for_distributed()
            results = self.target_model.get_aux_and_target_hiddens(
                input_ids=row["input_ids"].to(device),
            )
            # hidden_states: B, N, 3*D
            # target_hiddens: B, N, D
            for k, v in results.items():
                results[k] = v.cpu()

            # Prepare data point
            data_point = {
                "input_ids": row["input_ids"].cpu(),  # B, N
                "loss_mask": row["loss_mask"].cpu(),  # B, N
                **results,
            }

            # Save to disk
            torch.save(data_point, output_file)
            return True

        except Exception as e:
            logger.error(
                f"Error processing sample {idx}: {str(e)}", extra={"rank": self.rank}
            )
            return False

    def generate(self, dataset) -> Tuple[int, int]:
        """
        Generate hidden states for all samples in the dataset.

        Args:
            dataset: Dataset to process

        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0

        # Only show progress bar on rank 0
        iterator = (
            tqdm(
                enumerate(dataset),
                total=len(dataset),
                desc=f"Rank {self.rank} processing",
            )
            if self.rank == 0
            else enumerate(dataset)
        )

        for idx, row in iterator:
            if self._process_single_sample(idx, row):
                successful += 1
            else:
                failed += 1

        logger.info(
            f"Processing complete. Success: {successful}, Failed: {failed}",
            extra={"rank": self.rank},
        )
        logger.info(f"Results saved to {self.output_dir}", extra={"rank": self.rank})

        return successful, failed


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate hidden states for draft model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset range arguments
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Global start index of dataset (applies before distribution to GPUs)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Global end index of dataset (None means use full dataset). "
        "The range [start, end) will be automatically distributed across all GPUs.",
    )

    # Output configuration
    parser.add_argument(
        "--outdir",
        type=str,
        default="outdir0",
        help="Output directory for generated hidden states",
    )

    # Model configuration
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-4B", help="Model name or path"
    )
    parser.add_argument(
        "--target_model_name_or_path",
        type=str,
        help="Target model name or path (if different from model_name)",
    )
    parser.add_argument(
        "--target_backend",
        type=str,
        default="hf",
        choices=["hf"],
        help="Backend for target model",
    )
    parser.add_argument(
        "--modal_type",
        type=str,
        default="LLM",
        choices=["LLM", "VLM"],
        help="Modal type: LLM for language models, VLM for vision-language models",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="offline",
        choices=["online", "offline"],
        help="Training mode: online or offline",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path", type=str, nargs="+", required=True, help="Dataset to use"
    )
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="Maximum token length"
    )
    parser.add_argument(
        "--chat_template_type", type=str, default="default", help="Chat template type"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display dataset samples (only on rank 0)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes for data preprocessing",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=None,
        help="Number of max samples for data preprocessing",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, default=42, help="Random seed for shuffling dataset"
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: String representation of dtype

    Returns:
        Corresponding torch dtype
    """
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_mapping.get(dtype_str, torch.bfloat16)


def load_dataset(args: argparse.Namespace, tokenizer, rank: int):
    """
    Load and prepare dataset.

    Args:
        args: Parsed command line arguments
        tokenizer: Tokenizer from target model
        rank: Process rank

    Returns:
        Prepared dataset
    """
    logger.info(f"Loading dataset: {args.dataset_path}", extra={"rank": rank})

    # Only display on rank 0
    display = args.display and rank == 0

    args.train_data_path = None
    args.eval_data_path = args.dataset_path
    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        display=display,
    )

    _, dataset, _ = dataset_manager.create_online_datasets()
    logger.info(f"Dataset loaded: {len(dataset)} samples", extra={"rank": rank})

    return dataset


def split_dataset_for_rank(
    dataset, rank: int, world_size: int, start: int = 0, end: int = None
):
    """
    Split dataset for distributed processing.

    The dataset is first sliced to [start:end] range (global range),
    then evenly distributed across all ranks.

    Args:
        dataset: Full dataset
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes
        start: Global start index (default: 0)
        end: Global end index (default: None, means len(dataset))

    Returns:
        Dataset slice for current rank

    Example:
        Dataset has 10000 samples, world_size=4, start=1000, end=5000
        - Global range: [1000, 5000) = 4000 samples
        - Rank 0: [1000, 2000) = 1000 samples
        - Rank 1: [2000, 3000) = 1000 samples
        - Rank 2: [3000, 4000) = 1000 samples
        - Rank 3: [4000, 5000) = 1000 samples
    """
    # Determine the global range to process
    if end is None:
        end = len(dataset)

    # Validate range
    if start < 0 or end > len(dataset) or start >= end:
        raise ValueError(
            f"Invalid range: start={start}, end={end}, dataset_size={len(dataset)}"
        )

    total_samples = end - start
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size

    # Calculate start and end for this rank
    rank_start = start + rank * samples_per_rank + min(rank, remainder)
    rank_end = rank_start + samples_per_rank + (1 if rank < remainder else 0)

    logger.info(
        f"Rank {rank}/{world_size}: Processing global range [{start}, {end}) -> "
        f"assigned range [{rank_start}, {rank_end}) ({rank_end - rank_start} samples)",
        extra={"rank": rank},
    )

    return dataset.select(range(rank_start, rank_end))


def main():
    """Main execution function."""
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()

    # Parse arguments
    args = parse_arguments()
    args.train_data_path = None
    args.eval_data_path = args.dataset_path

    try:
        # Load target model
        torch_dtype = get_torch_dtype(args.torch_dtype)
        target_model = create_target_model(
            backend=args.target_backend,
            modal_type=args.modal_type,
            model_path=args.target_model_name_or_path or args.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )

        # Load dataset
        dataset = load_dataset(args, target_model.tokenizer, rank)

        # Split dataset for this rank
        dataset_slice = split_dataset_for_rank(
            dataset, rank, world_size, args.start, args.end
        )

        # Generate hidden states
        output_dir = f"{args.outdir}/rank_{rank}"
        generator = HiddenStateGenerator(target_model, output_dir, rank=rank)
        successful, failed = generator.generate(dataset_slice)

        # Synchronize all processes
        if world_size > 1:
            dist.barrier()

        # Log final statistics (only on rank 0)
        if rank == 0:
            logger.info("=" * 50, extra={"rank": rank})
            logger.info("Generation Complete!", extra={"rank": rank})
            logger.info(
                f"Total samples processed across all ranks: {len(dataset)}",
                extra={"rank": rank},
            )
            logger.info("=" * 50, extra={"rank": rank})

        logger.info(
            f"Rank {rank} - Successful: {successful}, Failed: {failed}",
            extra={"rank": rank},
        )

    finally:
        # Cleanup distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()
