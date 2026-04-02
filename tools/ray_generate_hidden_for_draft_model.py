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

"""
Ray-based distributed hidden state generation script.

For VLMVLLMBackend (vLLM backend), replaces torchrun-based launching,
completely eliminating distributed environment conflicts between torchrun and vLLM.

VLMTransformersBackend (HF backend) still uses torchrun:
    torchrun tools/generate_hidden_for_draft_model.py --target_backend hf ...

VLMVLLMBackend (vLLM backend) uses Ray:
    python tools/ray_generate_hidden_for_draft_model.py --target_backend vllm ...
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import ray
import torch
from generate_hidden_for_draft_model import HiddenStateGenerator
from transformers import AutoProcessor

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    create_target_model,
    infer_model_params,
)
from angelslim.compressor.speculative.train.data.data_utils import (
    process_token_dict_to_mappings,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments (consistent with generate_hidden_for_draft_model.py,
    with additional Ray-specific parameters).
    """
    parser = argparse.ArgumentParser(
        description="Generate hidden states using Ray distributed execution (for vLLM backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset range arguments
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Global start index",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Global end index (None means use full dataset)",
    )

    # Output configuration
    parser.add_argument(
        "--outdir",
        type=str,
        default="outdir0",
        help="Output directory",
    )

    # Model configuration
    parser.add_argument(
        "--target_model_name_or_path",
        type=str,
        required=True,
        help="Target model path",
    )
    parser.add_argument(
        "--target_backend",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="Target model backend (this script is dedicated to vllm)",
    )
    parser.add_argument(
        "--modal_type",
        type=str,
        default="VLM",
        choices=["LLM", "VLM"],
        help="Modal type",
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
        help="Torch dtype",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        required=True,
        help="Dataset path",
    )
    parser.add_argument("--model_max_length", type=int, default=2048, help="Maximum token length")
    parser.add_argument(
        "--chat_template_type",
        type=str,
        default=None,
        help="Chat template type (auto-inferred if not specified)",
    )
    parser.add_argument("--display", action="store_true", help="Display dataset samples")
    parser.add_argument(
        "--num_proc", type=int, default=16, help="Number of data preprocessing processes"
    )
    parser.add_argument("--sample_num", type=int, default=None, help="Maximum number of samples")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed")

    # Draft model config
    parser.add_argument(
        "--draft_model_config_path",
        type=str,
        default=None,
        help="Path to draft model config file, used to compute vocab mapping",
    )

    # Ray / vLLM specific parameters
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size (number of GPUs per worker)",
    )
    parser.add_argument(
        "--total_gpus",
        type=int,
        default=None,
        help="Total number of GPUs (default: auto-detected)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="vLLM max_model_len",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=8,
        help="vLLM max_num_seqs",
    )
    parser.add_argument(
        "--limit_mm_per_prompt",
        type=str,
        default=None,
        help='vLLM limit_mm_per_prompt, JSON format, e.g. \'{"image": 10, "video": 10}\'',
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_mapping.get(dtype_str, torch.bfloat16)


def split_dataset_indices(total_len, num_workers, start=0, end=None):
    """
    Evenly distribute dataset index range [start, end) among num_workers workers.

    Returns: List[Tuple[int, int]], each element is (rank_start, rank_end)
    """
    if end is None:
        end = total_len
    if start < 0 or end > total_len or start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, dataset_size={total_len}")

    total_samples = end - start
    samples_per_worker = total_samples // num_workers
    remainder = total_samples % num_workers

    slices = []
    for i in range(num_workers):
        worker_start = start + i * samples_per_worker + min(i, remainder)
        worker_end = worker_start + samples_per_worker + (1 if i < remainder else 0)
        slices.append((worker_start, worker_end))
    return slices


def main():
    args = parse_arguments()

    draft_vocab_size = None
    target_vocab_size = None
    logger.info(f"args.draft_model_config_path: {args.draft_model_config_path}")
    if args.draft_model_config_path is not None:
        draft_config = DraftModelConfig.from_file(args.draft_model_config_path)
        draft_vocab_size = getattr(draft_config, "draft_vocab_size", None)
        target_vocab_size = getattr(draft_config, "vocab_size", None)
        args.target_model_type = getattr(draft_config, "target_model_type", None)
        logger.info(
            f"Read from draft model config: draft_vocab_size={draft_vocab_size}, "
            f"target_vocab_size={target_vocab_size}, "
            f"target_model_type={args.target_model_type}"
        )
    else:
        raise ValueError("--draft_model_config_path must be specified")

    # Auto-infer chat_template_type
    if args.chat_template_type is None:
        _, _, inferred_chat_template_type = infer_model_params(
            model_name_or_path=args.target_model_name_or_path,
            model_type=args.target_model_type,
        )
        args.chat_template_type = (
            inferred_chat_template_type if inferred_chat_template_type is not None else "default"
        )
        logger.info(f"chat_template_type auto-inferred as: {args.chat_template_type}")
    else:
        logger.info(f"Using specified chat_template_type: {args.chat_template_type}")

    # Parse limit_mm_per_prompt
    limit_mm_per_prompt = None
    if args.limit_mm_per_prompt is not None:
        import json

        try:
            limit_mm_per_prompt = json.loads(args.limit_mm_per_prompt)
            logger.info(f"limit_mm_per_prompt: {limit_mm_per_prompt}")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON for --limit_mm_per_prompt: {args.limit_mm_per_prompt}. "
                f"Error: {e}"
            )

    # Calculate number of workers
    total_gpus = args.total_gpus
    tp_size = args.tensor_parallel_size
    num_workers = total_gpus // tp_size

    if num_workers <= 0:
        raise ValueError(
            f"Cannot create workers: total_gpus={total_gpus}, " f"tensor_parallel_size={tp_size}"
        )
    logger.info(
        f"Total GPUs: {total_gpus}, TP size: {tp_size}, " f"Number of workers: {num_workers}"
    )

    # ==========================================
    # Load dataset in Driver process BEFORE Ray init
    # (Dataset preprocessing uses multiprocessing via datasets.map(num_proc=N),
    #  which forks child processes. If Ray is already initialized, the forked
    #  children inherit Ray's internal threads/gRPC connections in an inconsistent
    #  state, causing deadlocks. Therefore we must load data BEFORE ray.init().)
    # ==========================================
    # Disable tokenizers parallelism BEFORE loading tokenizer.
    # HuggingFace tokenizers (Rust) spawns internal threads for parallelism.
    # When datasets.map(num_proc=N) forks child processes, these Rust threads'
    # locks are copied in an inconsistent state (locked but no owning thread),
    # causing the forked children to deadlock immediately.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Force tqdm to print each progress update on a new line instead of using
    # carriage return (\r) to overwrite in place.
    import tqdm as _tqdm_mod

    # _orig_display = _tqdm_mod.tqdm.display
    def _newline_display(self, msg=None, pos=None):
        # Call original display to get the formatted message
        if msg is None:
            msg = self.__str__()
        # Write to tqdm's file object, forcing a newline each time
        self.fp.write(msg.rstrip() + "\n")
        self.fp.flush()

    _tqdm_mod.tqdm.display = _newline_display

    # Need a tokenizer for data preprocessing.
    # Load tokenizer via transformers first (without loading model weights).
    tokenizer = AutoProcessor.from_pretrained(
        args.target_model_name_or_path,
        trust_remote_code=True,
    )
    logger.info(f"Tokenizer loaded: {type(tokenizer)}")
    # Load and preprocess dataset
    args.train_data_path = None
    args.eval_data_path = args.dataset_path
    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=tokenizer,
        target_model_type=None if args.modal_type in ("LLM", "TTS") else args.target_model_type,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        display=args.display,
    )
    _, dataset, _ = dataset_manager.create_online_datasets()
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    if len(dataset) == 0:
        logger.warning("Dataset is empty, exiting")
        return

    # Calculate data slices for each worker
    slices = split_dataset_indices(len(dataset), num_workers, args.start, args.end)
    for i, (s, e) in enumerate(slices):
        logger.info(f"Worker {i}: processing samples [{s}, {e}) ({e - s} samples)")

    # ==========================================
    # Initialize Ray (AFTER dataset preprocessing to avoid fork deadlocks)
    # ==========================================
    # Pass the tools directory path to Ray actors via environment variable,
    # so they can import HiddenStateGenerator from generate_hidden_for_draft_model.
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["_RAY_TOOLS_DIR"] = tools_dir

    ray.init(ignore_reinit_error=True)
    logger.info(f"Ray initialized, available resources: {ray.available_resources()}")

    # Put dataset into Ray object store
    # (Dataset was already loaded and preprocessed above)
    dataset_ref = ray.put(dataset)

    # ==========================================
    # Define Ray Worker Actor
    # ==========================================
    @ray.remote
    class HiddenStateWorker:
        """Ray Actor: each worker holds an independent vLLM instance."""

        def __init__(
            self,
            worker_id,
            model_path,
            target_model_type,
            modal_type,
            tp_size,
            max_model_len,
            gpu_memory_utilization,
            max_num_seqs,
            trust_remote_code,
            output_dir,
            draft_vocab_size,
            target_vocab_size,
            limit_mm_per_prompt=None,
        ):
            self.worker_id = worker_id
            self.model_path = model_path
            self.target_model_type = target_model_type
            self.output_dir = output_dir
            self.draft_vocab_size = draft_vocab_size
            self.target_vocab_size = target_vocab_size

            # Inside Ray actor:
            #   - No RANK/WORLD_SIZE or other torchrun environment variables
            #   - CUDA_VISIBLE_DEVICES is automatically set by Ray
            #   - vLLM can freely use NCCL without any conflicts
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
            # Log GPU memory status via nvidia-smi subprocess.
            # IMPORTANT: Do NOT call torch.cuda.mem_get_info() or any
            # PyTorch CUDA API here!  Doing so initializes a CUDA context
            # in the actor process, which:
            #   1. Occupies ~1-2 GiB GPU VRAM in the actor (parent) process
            #   2. Forces vLLM's EngineCore to use 'spawn' instead of 'fork'
            #   3. The EngineCore subprocess then sees reduced free memory
            # Using nvidia-smi avoids CUDA initialization entirely.
            gpu_mem_info = ""
            try:
                import subprocess as _sp

                _result = _sp.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free,memory.total",
                        "--format=csv,noheader,nounits",
                        f"--id={cuda_visible}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if _result.returncode == 0 and _result.stdout.strip():
                    _parts = _result.stdout.strip().split(",")
                    if len(_parts) == 2:
                        gpu_mem_info = (
                            f", GPU free={int(_parts[0].strip())} MiB"
                            f"/{int(_parts[1].strip())} MiB"
                        )
            except Exception:
                pass
            logger.info(
                f"[Worker {worker_id}] Initializing... "
                f"CUDA_VISIBLE_DEVICES={cuda_visible}{gpu_mem_info}"
            )

            create_kwargs = dict(
                backend="vllm",
                modal_type=modal_type,
                model_path=model_path,
                trust_remote_code=trust_remote_code,
                target_model_type=target_model_type,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=max_num_seqs,
            )
            if limit_mm_per_prompt is not None:
                create_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
            self.target_model = create_target_model(**create_kwargs)
            logger.info(f"[Worker {worker_id}] Model loaded")

        def process_slice(self, dataset, slice_start, slice_end):
            """Process a dataset slice, generate hidden states and save."""
            # Note: Ray automatically dereferences ObjectRef arguments passed
            # to remote methods, so `dataset` is already the actual Dataset
            # object (not an ObjectRef). No ray.get() needed here.
            dataset_slice = dataset.select(range(slice_start, slice_end))

            # Reuse HiddenStateGenerator
            # Ray actor's sys.path may not include the tools/ directory, need to add manually.
            # Cannot get script path via ray.get_runtime_context,
            # so the tools directory path is passed via _RAY_TOOLS_DIR environment variable.
            tools_dir = os.environ.get("_RAY_TOOLS_DIR", "")
            if tools_dir and tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)

            output_dir = f"{self.output_dir}/rank_{self.worker_id}"
            generator = HiddenStateGenerator(
                self.target_model,
                output_dir,
                rank=self.worker_id,
                draft_vocab_size=self.draft_vocab_size,
                target_vocab_size=self.target_vocab_size,
            )
            successful, failed = generator.generate(dataset_slice)

            # Return statistics and token_dict
            logger.info(
                f"[Worker {self.worker_id}] Processing complete: "
                f"successful={successful}, failed={failed}"
            )
            return {
                "successful": successful,
                "failed": failed,
                "token_dict": dict(generator.token_dict),
            }

    # ==========================================
    # Create Workers and dispatch tasks
    # ==========================================
    logger.info("Creating Ray workers...")
    workers = []
    for i in range(num_workers):
        worker = HiddenStateWorker.options(
            num_gpus=tp_size,
            max_concurrency=1,
        ).remote(
            worker_id=i,
            model_path=args.target_model_name_or_path,
            target_model_type=args.target_model_type,
            modal_type=args.modal_type,
            tp_size=tp_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=args.trust_remote_code,
            output_dir=args.outdir,
            draft_vocab_size=draft_vocab_size,
            target_vocab_size=target_vocab_size,
            limit_mm_per_prompt=limit_mm_per_prompt,
        )
        workers.append(worker)

    logger.info(f"Created {num_workers} workers, starting processing...")

    # Dispatch tasks (each worker processes its own data slice)
    futures = []
    for i, (slice_start, slice_end) in enumerate(slices):
        future = workers[i].process_slice.remote(dataset_ref, slice_start, slice_end)
        futures.append(future)

    # Wait for all workers to complete
    results = ray.get(futures)

    # ==========================================
    # Aggregate results
    # ==========================================
    total_successful = 0
    total_failed = 0
    merged_token_dict = Counter()

    for i, result in enumerate(results):
        total_successful += result["successful"]
        total_failed += result["failed"]
        merged_token_dict.update(result["token_dict"])
        logger.info(f"Worker {i}: successful={result['successful']}, failed={result['failed']}")

    logger.info("=" * 50)
    logger.info("All workers finished!")
    logger.info(f"Total successful: {total_successful}, Total failed: {total_failed}")
    logger.info("=" * 50)

    # ==========================================
    # Compute and save vocab mapping (done in driver process)
    # ==========================================
    if draft_vocab_size is not None and target_vocab_size is not None:
        vocab_mapping_path = Path(args.outdir) / "vocab_mapping.pt"
        logger.info(
            f"Computing vocab mapping (draft_vocab_size={draft_vocab_size}, "
            f"target_vocab_size={target_vocab_size})..."
        )

        d2t, t2d = process_token_dict_to_mappings(
            merged_token_dict,
            draft_vocab_size,
            target_vocab_size,
        )

        vocab_mapping = {"d2t": d2t, "t2d": t2d}
        torch.save(vocab_mapping, vocab_mapping_path)
        logger.info(f"Vocab mapping saved to {vocab_mapping_path}")

    # Cleanup
    ray.shutdown()
    logger.info("Ray shut down, task complete")


if __name__ == "__main__":
    main()
