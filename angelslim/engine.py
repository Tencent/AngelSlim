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

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from .compressor import CompressorFactory
from .compressor.speculative.benchmark import pytorch as pytorch_benchmark
from .compressor.speculative.benchmark import vllm as vllm_benchmark
from .data.dataloader import DataLoaderFactory
from .models import SlimModelFactory
from .utils import (
    default_compress_config,
    get_loaders,
    get_package_info,
    parse_json_full_config,
    print_info,
)

DEFAULT_COMPRESSION_CONFIG = {
    "fp8_static": default_compress_config.default_fp8_static_config(),
    "fp8_dynamic": default_compress_config.default_fp8_dynamic_config(),
    "int8_dynamic": default_compress_config.default_int8_dynamic_config(),
    "int4_awq": default_compress_config.default_int4_awq_config(),
    "int4_gptq": default_compress_config.default_int4_gptq_config(),
    "w4a8_fp8": default_compress_config.default_w4a8_fp8_static_config(),
}


def get_supported_compress_method():
    return DEFAULT_COMPRESSION_CONFIG.keys()


class Engine:
    def __init__(self):
        """
        Initialize engine configuration
        """
        self.slim_model = None
        self.tokenizer = None
        self.dataloader = None
        self.compressor = None
        self.compress_type = None
        self.only_inference = False
        self.model_path = None
        self.max_seq_length = None

    def prepare_model(
        self,
        model_name="Qwen",
        model=None,
        tokenizer=None,
        model_path=None,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        cache_dir=None,
        deploy_backend="vllm",
        using_multi_nodes=False,
        use_audio_in_video=False,
        attn_implementation="default",
    ) -> Any:
        """Load pretrained model and tokenizer
        Args:
            model_name (str): Name of the model to load.
            model (Any, optional): Preloaded model instance.
                If provided, `model_path` is ignored.
            tokenizer (Any, optional): Preloaded tokenizer instance.
                If model is set, tokenizer must be also set in LLM and VLM.
            model_path (str, optional): Path to the pretrained model.
            torch_dtype (str): Data type for the model weights.
            device_map (str): Device map for the model.
            trust_remote_code (bool): Whether to trust remote code.
            low_cpu_mem_usage (bool): Whether to use low CPU memory usage mode.
            use_cache (bool): Whether to use cache during loading.
            cache_dir (str, optional): Directory to cache the model.
            deploy_backend (str): Backend for deployment, e.g., "torch", "vllm".
            using_multi_nodes (bool): Whether to use multi-nodes for calibration.
            use_audio_in_video (bool): Whether to add audio track to a video file.
            attn_implementation (str): The attention implementation to use in the model.
        """
        assert model_name, "model_name must be specified."
        assert model_path, "model_path must be specified."

        # Initialize slim model by ModelFactory
        self.slim_model = SlimModelFactory.create(
            model_name, model=model, deploy_backend=deploy_backend
        )

        self.series = SlimModelFactory.get_series_by_models(model_name)

        if self.series in ["LLM", "VLM", "Audio"]:
            if model:
                assert tokenizer, " If model is set, tokenizer must be also set."
                self.slim_model.tokenizer = tokenizer
            else:
                self.slim_model.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    use_cache=use_cache,
                    using_multi_nodes=using_multi_nodes,
                )
                self.model_path = model_path
        elif self.series in ["Omni"]:
            if not model:
                self.slim_model.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    use_audio_in_video=use_audio_in_video,
                    attn_implementation=attn_implementation,
                )
                self.model_path = model_path
        else:
            raise ValueError(f"Unsupported series: {self.series}")

        return self.slim_model

    def prepare_data(
        self,
        data_path=None,
        data_type="TextDataset",
        custom_dataloader=None,
        max_length=2048,
        batch_size=1,
        num_samples=128,
        shuffle=True,
        inference_settings=None,
        use_audio_in_video=False,
        model_name=None,
        quantization_config=None,
    ) -> Optional[Any]:
        """Prepare compression dataset"""
        if custom_dataloader is not None:
            print_info("Using custom provided dataloader...")
            self.dataloader = custom_dataloader
            return self.dataloader

        assert data_path, "data_path must be specified."
        # Dynamically create dataloader by DataLoaderFactory
        self.dataloader = DataLoaderFactory.create_data_loader(
            data_type=data_type,
            processor=(
                self.slim_model.processor
                if self.series in ["VLM", "Omni", "Audio"]
                else self.slim_model.tokenizer
            ),
            device=self.slim_model.model.device,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=shuffle,
            num_samples=num_samples,
            data_source=data_path,
            inference_settings=inference_settings,
            use_audio_in_video=use_audio_in_video,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.max_seq_length = max_length

        return self.dataloader

    def prepare_compressor(
        self,
        compress_name="PTQ",
        global_config=None,
        compress_config=None,
        transform_config=None,
        default_method=None,
    ) -> Any:
        """
        Initialize compression components.
        Args:
            compress_name (str): Name of the compression method to use.
            global_config (dict, optional): Global configuration for the model.
            compress_config (dict, optional): Configuration for the compression method.
            default_method (str, optional): Default compression method if not specified.
               If set default_method, compress_config and global_config will be ignored.
        """
        if isinstance(compress_name, str):
            compress_names = [compress_name]
        elif isinstance(compress_name, list):
            compress_names = compress_name
        for method_name in compress_names:
            if method_name not in CompressorFactory.get_available_compressor():
                raise ValueError(
                    f"Compression method '{method_name}' not registered. "
                    f"Available methods: {CompressorFactory.get_available_compressor()}"
                )
        if self.series in ["LLM", "VLM", "Omni", "Audio"]:
            global_config.update(self.model_path, self.max_seq_length)

        if default_method:
            assert (
                default_method in DEFAULT_COMPRESSION_CONFIG
            ), f"`default_method` not found in : {DEFAULT_COMPRESSION_CONFIG.keys()}."
            slim_config = DEFAULT_COMPRESSION_CONFIG[default_method]
        else:
            slim_config = {
                "global_config": global_config,
                "compress_config": compress_config,
                "transform_config": transform_config,
            }
        self.compress_type = compress_names
        self.only_inference = compress_config.only_inference if compress_config else False
        # Create compressor by CompressorFactory
        self.compressor = CompressorFactory.create(
            compress_names, self.slim_model, slim_config=slim_config
        )
        return self.compressor

    def run(self) -> Any:
        """Execute compression pipeline"""
        if not self.compressor:
            raise RuntimeError("Compressor not initialized. Call prepare_compressor() first")
        if isinstance(self.compressor, str):
            compressors = [self.compressor]
        elif isinstance(self.compressor, list):
            compressors = self.compressor
        for idx, compress_type in enumerate(self.compress_type):
            if self.only_inference[idx]:
                continue
            if compress_type == "PTQ":
                compressors[idx].calibrate(self.dataloader)
            elif compress_type == "QAT":
                compressors[idx].run(self.dataloader)
            else:
                raise NotImplementedError(
                    f"Compression type {self.compress_type} is not implemented"
                )

    def save(self, save_path: Optional[str] = None, config: Optional[dataclass] = None) -> None:
        """Save compressed model and tokenizer
        Args:
            save_path (str, optional): Path to save the compressed model and tokenizer.
        """
        assert save_path, "Save path must be provided in model_config or as an argument"
        if isinstance(self.compressor, str):
            compressors = [self.compressor]
        elif isinstance(self.compressor, list):
            compressors = self.compressor
        for idx, compress_type in enumerate(self.compress_type):
            if self.only_inference[idx]:
                continue
            if compress_type in ["PTQ", "QAT"]:
                # Execute model conversion
                compressors[idx].convert()

            # Save quantized model
            compressors[idx].save(save_path)

        # Save all config
        if config is not None and compress_type != "QAT":
            config_dict = asdict(config)
            config_dict["debug_info"] = {
                "python": sys.version,
                "angelslim": get_package_info("angelslim"),
                "torch": get_package_info("torch"),
                "transformers": get_package_info("transformers"),
                "torch_cuda_version": (torch.version.cuda if torch.cuda.is_available() else None),
            }
            config_dict["model_config"]["model_path"] = "Base Model Path"
            config_dict["global_config"]["save_path"] = "Save Model Path"
            if "dataset_config" in config_dict and isinstance(config_dict["dataset_config"], dict):
                config_dict["dataset_config"]["data_path"] = "Data Path"
            with open(os.path.join(save_path, "angelslim_config.json"), "w") as f:
                json.dump(config_dict, f, indent=4)

        print_info(f"Compressed model saved to {save_path}")

    @torch.no_grad()
    def ppl_eval(self, tasks, seqlen=2048, cache_dir=None):
        results = {}
        model = self.slim_model.model
        task_names = tasks.split(",")

        for dataset in task_names:
            testloader = get_loaders(
                self.slim_model.tokenizer, dataset, seqlen=seqlen, cache_dir=cache_dir
            )
            testenc = testloader if "c4" in dataset else testloader.input_ids
            nsamples = testenc.numel() // seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()

            if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
                classifier = model.lm_head
            elif hasattr(model.model, "lm_head"):
                classifier = None
            elif hasattr(model, "output"):
                classifier = model.output
            else:
                raise NotImplementedError

            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
                outputs = model.model(batch)
                if classifier is not None:
                    hidden_states = outputs[0]
                    logits = classifier(hidden_states.to(classifier.weight.dtype))
                else:
                    logits = outputs[0]
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:].to(
                    shift_logits.device
                )
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

            results[dataset] = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
        model.config.use_cache = use_cache

        print_info(results)
        return results

    def lm_eval(
        self,
        tasks: str,
        batch_size: int = 1,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        output_path: Optional[str] = None,
    ) -> dict:
        """Evaluate the (compressed) model with lm-evaluation-harness.

        Args:
            tasks: Comma-separated list of lm-eval task names.
            batch_size: Batch size for evaluation.
            num_fewshot: Number of few-shot examples.
            limit: Maximum number of samples per task (None = all).
            apply_chat_template: Apply the tokenizer chat template to prompts.
            fewshot_as_multiturn: Treat few-shot examples as multi-turn conversation.
            output_path: If provided, save raw results to this path with torch.save().

        Returns:
            The results dict returned by lm_eval.evaluator.simple_evaluate().
        """
        try:
            from lm_eval import evaluator as lm_evaluator
            from lm_eval.models.huggingface import HFLM
            from lm_eval.utils import make_table
        except ImportError as e:
            raise ImportError(
                "lm-evaluation-harness is required for test_lm_eval. "
                "Install it with: pip install lm-eval"
            ) from e

        if self.slim_model is None or self.slim_model.model is None:
            raise RuntimeError("Model not initialized. Call prepare_model() first.")

        model = self.slim_model.model
        tokenizer = self.slim_model.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        lm_eval_model = HFLM(model, tokenizer=tokenizer, batch_size=batch_size)

        task_names = tasks.split(",")
        results = lm_evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            limit=limit,
            num_fewshot=num_fewshot,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

        print_info(make_table(results))

        if output_path is not None:
            torch.save(results, output_path)
            print_info(f"lm_eval results saved to {output_path}")

        return results


class InferEngine(Engine):
    def __init__(self):
        """
        Initialize engine configuration
        """
        super().__init__()
        self.slim_model = None
        self.tokenizer = None
        self.dataloader = None
        self.compressor = None
        self.compress_type = None
        self.model_path = None
        self.max_seq_length = None

    def from_pretrained(
        self,
        model_path,
        torch_dtype=None,
        device_map=None,
        trust_remote_code=None,
        low_cpu_mem_usage=None,
        use_cache=None,
    ) -> Any:
        """Load pretrained model and tokenizer
        Args:
            model_path (str): Path to the pretrained model.
            torch_dtype (str): Data type for the model weights.
            device_map (str): Device map for the model.
            trust_remote_code (bool): Whether to trust remote code.
            low_cpu_mem_usage (bool): Whether to use low CPU memory usage mode.
            use_cache (bool): Whether to use cache during loading.
            cache_dir (str, optional): Directory to cache the model.
        """
        assert model_path, "model_path must be specified."
        # load slim config
        slim_config_path = os.path.join(model_path, "angelslim_config.json")
        if not os.path.exists(slim_config_path):
            raise FileNotFoundError(
                f"angelslim_config.json not found in {model_path}. "
                "Please ensure the model is compressed with Angelslim."
            )
        slim_config = parse_json_full_config(slim_config_path)
        if torch_dtype:
            slim_config.model_config.torch_dtype = torch_dtype
        if device_map:
            slim_config.model_config.device_map = device_map
        if trust_remote_code is not None:
            slim_config.model_config.trust_remote_code = trust_remote_code
        if low_cpu_mem_usage is not None:
            slim_config.model_config.low_cpu_mem_usage = low_cpu_mem_usage
        if use_cache is not None:
            slim_config.model_config.use_cache = use_cache

        self.slim_model = SlimModelFactory.create(
            slim_config.model_config.name, deploy_backend="huggingface"
        )

        self.slim_model.from_pretrained(
            model_path=model_path,
            torch_dtype=slim_config.model_config.torch_dtype,
            device_map=slim_config.model_config.device_map,
            trust_remote_code=slim_config.model_config.trust_remote_code,
            low_cpu_mem_usage=slim_config.model_config.low_cpu_mem_usage,
            use_cache=slim_config.model_config.use_cache,
            compress_config=slim_config.compression_config,
        )

        self.series = SlimModelFactory.get_series_by_models(slim_config.model_config.name)

    def generate(self, input_prompt: str, **kwargs) -> Any:
        """Run inference with the compressed model
        Args:
            input_prompt (str): Input prompt for the model.
        """
        if not self.slim_model or not self.slim_model.model:
            raise RuntimeError("Model not initialized. Call from_pretrained() first")

        if self.series in ["LLM", "VLM"]:
            return self.slim_model.generate(
                input_ids=self.slim_model.tokenizer(input_prompt, return_tensors="pt").input_ids,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Series {self.series} is not implemented for inference")


class VLLMCalibrateEngine:
    """
    Engine for vLLM-based calibration to collect activation and MoE expert statistics.
    Wraps vLLM's LLM instance and provides a unified interface for calibration workflow.
    """

    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.prompts = None
        self.output_dir = None
        self.verbose = False

    def prepare_model(
        self,
        model_path: str,
        tp_size: int = 1,
        max_num_seqs: int = 128,
        max_length: int = 16384,
        distributed_executor_backend: str = "ray",
        skip_weight_loading: bool = False,
    ) -> Any:
        """Create vLLM LLM instance and setup activation hooks.

        Args:
            model_path: Path to the model directory.
            tp_size: Tensor parallel size.
            max_num_seqs: Maximum number of sequences per batch.
            max_length: Maximum sequence length for tokenization.
            distributed_executor_backend: Distributed executor backend ('ray' or 'mp').
            skip_weight_loading: Use dummy weights for fast debug mode.
        """
        from vllm import LLM

        from .compressor.quant import setup_activation_hooks

        print_info(f"VLLM_MOE_COLLECT_STATS: {os.environ.get('VLLM_MOE_COLLECT_STATS')}")
        print_info("\nConfiguration:")
        print_info(f"  Model: {model_path}")
        print_info(f"  TP Size: {tp_size}")
        print_info(f"  Max Num Seqs: {max_num_seqs}")
        print_info(f"  Skip Weight Loading: {skip_weight_loading}")

        self.llm = LLM(
            model=model_path,
            load_format="dummy" if skip_weight_loading else "auto",
            disable_log_stats=False,
            enforce_eager=True,
            enable_chunked_prefill=False,
            tensor_parallel_size=tp_size,
            distributed_executor_backend=distributed_executor_backend,
            enable_expert_parallel=False,
            max_num_seqs=max_num_seqs,
            max_model_len=max_length + 16,
        )

        if skip_weight_loading:
            print_info("\n" + "!" * 80)
            print_info("WARNING: Running with dummy weights (random values)!")
            print_info("Outputs will NOT make sense. This is for debugging only.")
            print_info("!" * 80 + "\n")

        # Setup activation hooks on all workers
        print_info("\n" + "=" * 80)
        print_info("Setting up activation hooks...")
        print_info("=" * 80)
        hook_results = self.llm.apply_model(setup_activation_hooks)
        for i, result in enumerate(hook_results):
            print_info(f"Worker {i}: {result}")

        self.tokenizer = self.llm.get_tokenizer()
        return self.llm

    def prepare_data(
        self,
        ptq_data_path: str,
        max_length: int = 16384,
        num_samples: int = 512,
    ) -> list:
        """Load calibration dataset and prepare prompts.

        Args:
            ptq_data_path: Path to the PTQ calibration data (JSONL format).
            max_length: Maximum sequence length for tokenization.
            num_samples: Number of samples to process from dataset.

        Returns:
            List of prompt strings ready for inference.
        """
        if self.llm is None:
            raise RuntimeError("Model not initialized. Call prepare_model() first.")

        print_info("\n" + "=" * 80)
        print_info("Loading dataset and preparing prompts...")
        print_info("=" * 80)

        dataloader = DataLoaderFactory.create_data_loader(
            data_type="TextDataset",
            processor=self.tokenizer,
            device="cpu",
            max_length=max_length,
            batch_size=1,
            shuffle=False,
            num_samples=num_samples,
            data_source=ptq_data_path,
        )

        self.prompts = [self.tokenizer.decode(data["input_ids"][0]) for data in dataloader]
        print_info(f"Loaded {len(self.prompts)} prompts from dataset")
        return self.prompts

    def run(
        self,
        output_dir: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Execute calibration: generate outputs and collect statistics.

        Args:
            output_dir: Directory to save output statistics.
            verbose: Enable verbose output for debugging.

        Returns:
            Dictionary with 'activation_stats' and 'moe_stats' results.
        """
        from vllm import SamplingParams

        from .compressor.quant import (
            get_activation_stats,
            get_moe_stats,
            print_activation_stats,
            print_moe_stats,
        )

        if self.llm is None:
            raise RuntimeError("Model not initialized. Call prepare_model() first.")
        if self.prompts is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        self.output_dir = output_dir
        self.verbose = verbose

        # Generate outputs
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1,
        )

        print_info("\n" + "=" * 80)
        print_info("Generating outputs...")
        print_info("=" * 80)
        outputs = self.llm.generate(self.prompts, sampling_params)

        # Print sample outputs
        print_info("\n" + "=" * 80)
        print_info("Sample Generated Outputs (first 5):")
        print_info("=" * 80)
        for i, output in enumerate(outputs[:5]):
            generated_text = output.outputs[0].text
            print_info(f"[{i + 1}] Output: {generated_text!r}")
        print_info(f"\nTotal outputs generated: {len(outputs)}")

        # Collect and save statistics
        print_info("\n" + "=" * 80)
        print_info("Collecting Statistics...")
        print_info("=" * 80)

        print_info("\nActivation Statistics:")
        self.llm.apply_model(print_activation_stats)

        print_info("\nMoE Expert Statistics:")
        self.llm.apply_model(lambda model: print_moe_stats(model, verbose=verbose))

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save activation statistics
        stats_list = self.llm.apply_model(get_activation_stats)
        activation_stats = self._extract_stats(stats_list)
        self._save_stats_to_json(
            activation_stats, "activation_stats.json", stats_type="activation statistics"
        )

        # Save MoE expert statistics
        moe_stats_list = self.llm.apply_model(get_moe_stats)
        moe_stats = self._extract_stats(moe_stats_list)
        self._save_stats_to_json(
            moe_stats, "moe_expert_stats.json", stats_type="MoE expert statistics"
        )

        print_info("\n" + "=" * 80)
        print_info("Calibration completed successfully!")
        print_info(f"Results saved to: {output_dir}")
        print_info("=" * 80)

        return {"activation_stats": activation_stats, "moe_stats": moe_stats}

    def quantize(
        self,
        model_path: str,
        output_dir: str,
        quant_name: str = "fp8_static",
        group_size: int = None,
        zero_point: bool = True,
        ignore_layers: list = None,
        num_workers: int = 32,
    ) -> None:
        """Quantize model weights based on compression config.

        Supports FP8 blockwise quantization and W4A8 mixed quantization.
        For w4a8_fp8, performs two-step quantization:
          1. FP8 blockwise weight quantization for ALL layers first (block_size=128x128)
          2. W4 per-group quantization on top (for layers NOT in ignore_layers)
        Also converts activation stats JSON to input_scale tensors.

        Args:
            model_path: Path to the original model directory (with safetensors).
            output_dir: Directory to save the quantized model.
            quant_name: Quantization method name from compression config.
                Supported: 'fp8_static', 'w4a8_fp8', 'int4_awq'.
            group_size: Group size for INT4 per-group quantization.
            zero_point: Whether to use zero point for INT4 quantization.
            ignore_layers: List of layer name patterns to skip W4 quantization
                (these layers keep FP8 quantization only).
            num_workers: Number of parallel workers for processing safetensors files.
        """
        # Default FP8 block size for w4a8_fp8 (non-W4 layers)
        _FP8_BLOCK_SIZE = (128, 128)
        # Determine moe_expert_stats.json path from previous calibration run
        moe_expert_stats_path = None
        if self.output_dir:
            candidate = os.path.join(self.output_dir, "moe_expert_stats.json")
            if os.path.exists(candidate):
                moe_expert_stats_path = candidate

        print_info("\n" + "=" * 80)
        print_info(f"Quantizing model weights (method: {quant_name})...")
        print_info(f"  Input: {model_path}")
        print_info(f"  Output: {output_dir}")
        if moe_expert_stats_path:
            print_info(f"  MoE expert stats: {moe_expert_stats_path}")
        print_info(f"  Num workers: {num_workers}")

        if quant_name == "w4a8_fp8":
            # Mixed precision weight quantization in a single pass:
            # - fp8_only_layers → FP8 blockwise quantization
            # - no_quant_layers (e.g. lm_head) → copy as-is
            # - other quantizable layers → INT4 symmetric per-group + pack
            from .compressor.quant.core.weight_quantize import mixed_weight_quantize

            if group_size is None:
                raise ValueError(
                    "group_size is required for w4a8_fp8 quantization. "
                    "Please set "
                    "compression.quantization.quant_method.group_size "
                    "in your YAML config."
                )
            print_info(f"  FP8 block size: {_FP8_BLOCK_SIZE}")
            print_info(f"  W4 group size: {group_size}")

            # Separate ignore_layers into:
            #   - fp8_only_layers: layers that should be FP8 quantized (skip W4)
            #   - no_quant_layers: output head layers (e.g. lm_head) that skip both FP8 and W4
            _NO_QUANT_PATTERNS = ["lm_head"]
            fp8_only_layers = []
            no_quant_layers = []
            if ignore_layers:
                for pattern in ignore_layers:
                    if any(nq in pattern for nq in _NO_QUANT_PATTERNS):
                        no_quant_layers.append(pattern)
                    else:
                        fp8_only_layers.append(pattern)
            print_info(f"  FP8-only layers (skip W4): {fp8_only_layers}")
            print_info(f"  No-quant layers (skip both FP8 & W4): {no_quant_layers}")
            print_info("=" * 80)

            # Step 1: W4A8 mixed quantization (single pass)
            print_info("\n[Step 1/2] Mixed precision weight quantization (INT4 + FP8)...")
            mixed_weight_quantize(
                input_path=model_path,
                output_path=output_dir,
                fp8_block_size=_FP8_BLOCK_SIZE,
                w4_group_size=group_size,
                num_workers=num_workers,
                fp8_only_layers=fp8_only_layers if fp8_only_layers else None,
                no_quant_layers=no_quant_layers if no_quant_layers else None,
                modules_to_not_convert=ignore_layers,
            )

            # Step 2: Convert MoE expert stats to input_scale
            if moe_expert_stats_path:
                from .compressor.quant.core.weight_quantize import (
                    merge_moe_input_scales,
                )

                print_info(
                    "\n[Step 2/2] Converting MoE expert stats "
                    "to input_scale tensors (static activation)..."
                )
                merge_moe_input_scales(
                    moe_expert_stats_path=moe_expert_stats_path,
                    output_dir=output_dir,
                )
        else:
            raise ValueError(
                f"Unsupported quantization method: {quant_name}. "
                f"Supported: fp8_static, w4a8_fp8"
            )

        print_info("\n" + "=" * 80)
        print_info(f"Quantized model saved to: {output_dir}")
        print_info("=" * 80)

    @staticmethod
    def _extract_stats(stats_data):
        """Extract stats from worker results (take first worker's data)."""
        if isinstance(stats_data, list):
            if not stats_data or stats_data[0] is None:
                return None
            return stats_data[0]
        return stats_data

    def _save_stats_to_json(
        self, stats_data, filename: str, stats_type: str = "statistics"
    ) -> None:
        """Save statistics to JSON file.

        Args:
            stats_data: Statistics data dict.
            filename: Output filename.
            stats_type: Type of statistics for log messages.
        """
        if stats_data is None:
            print_info(f"\nNo {stats_type} available.")
            if "moe" in stats_type.lower():
                print_info(
                    "Make sure VLLM_MOE_COLLECT_STATS=1 is set " "and the model has MoE layers."
                )
            return

        output_file = os.path.join(self.output_dir, filename)
        with open(output_file, "w") as f:
            json.dump(stats_data, f, indent=2)
        print_info(f"\n{stats_type.capitalize()} saved to: {output_file}")


class SpecEngine:
    """
    High-level interface for speculative decoding benchmarks
    Integrates BenchmarkEngine with additional workflow management
    """

    def __init__(self, config=None, deploy_backend: str = "pytorch"):
        """
        Initialize SpecEngine

        Args:
            config: BenchmarkConfig instance (optional)
            deploy_backend: Backend to use ('pytorch' or 'vllm')
        """
        self.config = config
        self.benchmark_engine = None
        self.results = {}
        self.deploy_backend = deploy_backend.lower()

        if self.deploy_backend == "pytorch":
            self.BenchmarkConfig = pytorch_benchmark.BenchmarkConfig
            self.BenchmarkEngine = pytorch_benchmark.BenchmarkEngine
            self.BenchmarkMode = pytorch_benchmark.BenchmarkMode
        elif self.deploy_backend == "vllm":
            self.BenchmarkConfig = vllm_benchmark.BenchmarkConfig
            self.BenchmarkEngine = vllm_benchmark.BenchmarkEngine
            self.BenchmarkMode = vllm_benchmark.BenchmarkMode
        else:
            raise ValueError(f"Unsupported deploy_backend: {deploy_backend}")

    def setup_benchmark(
        self,
        base_model_path: str,
        eagle_model_path: str,
        model_id: str,
        bench_name: str = "mt_bench",
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Setup benchmark configuration

        Args:
            base_model_path: Path to base model
            eagle_model_path: Path to Eagle model
            model_id: Model identifier
            bench_name: Benchmark dataset name
            output_dir: Output directory for results
            **kwargs: Additional configuration parameters

        Returns:
            BenchmarkConfig instance
        """
        config_dict = {
            "base_model_path": base_model_path,
            "eagle_model_path": eagle_model_path,
            "model_id": model_id,
            "bench_name": bench_name,
            "output_dir": output_dir,
        }
        config_dict.update(kwargs)

        self.config = self.BenchmarkConfig(**config_dict)
        if self.config.is_tts:
            self.BenchmarkEngine = pytorch_benchmark.TTSBenchmarkEngine
        self.benchmark_engine = self.BenchmarkEngine(self.config)

        return self.config

    def run_eagle_benchmark(self) -> Dict[str, Any]:
        """Run Eagle speculative decoding benchmark only"""
        if not self.benchmark_engine:
            raise RuntimeError("Benchmark not configured. Call setup_benchmark() first.")

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.EAGLE)
        return self.results

    def run_baseline_benchmark(self) -> Dict[str, Any]:
        """Run baseline benchmark only"""
        if not self.benchmark_engine:
            raise RuntimeError("Benchmark not configured. Call setup_benchmark() first.")

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.BASELINE)
        return self.results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark (both Eagle and baseline) with automatic analysis

        Returns:
            Dictionary containing all results and metrics
        """
        if not self.benchmark_engine:
            raise RuntimeError("Benchmark not configured. Call setup_benchmark() first.")

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.BOTH)
        return self.results

    def calculate_acceptance_length(self, eagle_file: Optional[str] = None) -> float:
        """
        Calculate acceptance length from Eagle benchmark results

        Args:
            eagle_file: Path to Eagle results file
                (optional, uses default if not provided)

        Returns:
            Average acceptance length
        """
        if not self.benchmark_engine:
            raise RuntimeError("Benchmark not configured. Call setup_benchmark() first.")

        if eagle_file is None:
            eagle_file = self.benchmark_engine.eagle_file

        return self.benchmark_engine._calculate_acceptance_length(eagle_file)

    def calculate_speedup_ratio(
        self,
        baseline_file: Optional[str] = None,
        eagle_file: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> float:
        """
        Calculate speedup ratio between baseline and Eagle

        Args:
            baseline_file: Path to baseline results file
            eagle_file: Path to Eagle results file
            model_path: Path to model for tokenization

        Returns:
            Speedup ratio
        """
        if not self.benchmark_engine:
            raise RuntimeError("Benchmark not configured. Call setup_benchmark() first.")

        if baseline_file is None:
            baseline_file = self.benchmark_engine.baseline_file
        if eagle_file is None:
            eagle_file = self.benchmark_engine.eagle_file
        if model_path is None:
            model_path = self.config.base_model_path

        return self.benchmark_engine._calculate_speedup_ratio(
            model_path, baseline_file, eagle_file
        )

    def get_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.benchmark_engine:
            return "Benchmark not configured."

        return self.benchmark_engine.get_performance_summary()

    def cleanup_results(self):
        """Clean up temporary result files"""
        if self.benchmark_engine:
            for file_path in [
                self.benchmark_engine.eagle_file,
                self.benchmark_engine.baseline_file,
                self.benchmark_engine.analysis_file,
            ]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
