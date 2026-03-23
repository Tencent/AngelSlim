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
from datetime import timedelta

import torch
import torch.distributed as dist

from angelslim.engine import Engine, VLLMCalibrateEngine
from angelslim.utils import get_yaml_prefix_simple, print_info
from angelslim.utils.config_parser import SlimConfigParser, print_config


def get_args():
    parser = argparse.ArgumentParser(description="AngelSlim")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--multi-nodes", action="store_true")
    parser.add_argument("--lm-eval", action="store_true")
    args = parser.parse_args()
    return args


def merge_config(config, args):
    """
    Merge command line arguments into the configuration dictionary.

    Args:
        config (dict): Configuration dictionary to be updated.
        args (argparse.Namespace): Parsed command line arguments.
    """
    if args.save_path is not None:
        config.global_config.save_path = args.save_path
    if args.model_path is not None:
        config.model_config.model_path = args.model_path
    config.global_config.save_path = os.path.join(
        config.global_config.save_path,
        get_yaml_prefix_simple(args.config),
    )


def multi_nodes_run(config):
    """
    Run the LLM compression process based on the provided configuration
    using multiple nodes.

    Args:
        config (dict): Configuration dictionary containing
                       parameters for LLM compression.
    """
    # Step 1: Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    # Step 2: Initialize configurations
    model_config = config.model_config
    dataset_config = config.dataset_config
    compress_config = config.compression_config
    global_config = config.global_config
    transform_config = config.transform_config

    # Step 3: Execute complete pipeline
    slim_engine = Engine()

    # Step 4: Prepare model
    slim_engine.prepare_model(
        model_name=model_config.name,
        model_path=model_config.model_path,
        torch_dtype=model_config.torch_dtype,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        use_cache=model_config.use_cache,
        cache_dir=model_config.cache_dir,
        use_audio_in_video=model_config.use_audio_in_video,
        attn_implementation=model_config.attn_implementation,
        deploy_backend=global_config.deploy_backend,
        using_multi_nodes=True,
    )

    # Step 5: Prepare data (optional custom dataloader)
    if compress_config.need_dataset:
        slim_engine.prepare_data(
            data_path=dataset_config.data_path,
            data_type=dataset_config.name,
            custom_dataloader=None,
            max_length=dataset_config.max_seq_length,
            batch_size=dataset_config.batch_size,
            num_samples=dataset_config.num_samples,
            shuffle=dataset_config.shuffle,
            inference_settings=dataset_config.inference_settings,
            use_audio_in_video=model_config.use_audio_in_video,
        )

    # Step 6: Initialize compressor
    slim_engine.prepare_compressor(
        compress_name=compress_config.name,
        compress_config=compress_config,
        global_config=global_config,
        transform_config=transform_config,
    )

    # Step 7: Compress model
    slim_engine.run()

    # Step 8: Save compressed model
    slim_engine.save(global_config.save_path, config)


def vllm_calibrate_run(config):
    """
    Run vLLM-based calibration to collect activation and MoE expert statistics,
    then quantize model weights based on compression config.

    If both activation_stats.json and moe_expert_stats.json already exist in the
    output directory, the calibration step is skipped and only weight quantization
    and input_scale merging are performed.

    Args:
        config (dict): Configuration dictionary containing
                       parameters for vLLM calibration and compression.
    """
    model_config = config.model_config
    dataset_config = config.dataset_config
    global_config = config.global_config
    calibrate_config = config.compression_config.calibrate
    compress_config = config.compression_config

    # Check if calibration stats already exist — skip vLLM calibration if so
    activation_stats_file = os.path.join(global_config.save_path, "activation_stats.json")
    moe_stats_file = os.path.join(global_config.save_path, "moe_expert_stats.json")
    skip_calibration = os.path.exists(activation_stats_file) and os.path.exists(moe_stats_file)

    engine = VLLMCalibrateEngine()

    if skip_calibration:
        print_info("\n" + "=" * 80)
        print_info("Calibration stats already exist, skipping vLLM calibration:")
        print_info(f"  - {activation_stats_file}")
        print_info(f"  - {moe_stats_file}")
        print_info("Proceeding directly to weight quantization and input_scale merging.")
        print_info("=" * 80)
        # Set output_dir so that engine.quantize() can find activation_stats.json
        engine.output_dir = global_config.save_path
    else:
        print_info("\n" + "=" * 80)
        print_info("Starting vLLM calibration:")
        engine.prepare_model(
            model_path=model_config.model_path,
            tp_size=calibrate_config.tp_size,
            max_num_seqs=calibrate_config.max_num_seqs,
            max_length=dataset_config.max_seq_length,
            distributed_executor_backend=calibrate_config.distributed_executor_backend,
            skip_weight_loading=calibrate_config.skip_weight_loading,
        )

        engine.prepare_data(
            ptq_data_path=dataset_config.data_path,
            max_length=dataset_config.max_seq_length,
            num_samples=dataset_config.num_samples,
        )

        engine.run(
            output_dir=global_config.save_path,
            verbose=calibrate_config.verbose,
        )

    # Extract quantization parameters from compression config
    quant_config = compress_config.quantization if compress_config else None
    quant_name = quant_config.name if quant_config else "fp8_static"
    ignore_layers = getattr(quant_config, "ignore_layers", None) if quant_config else None

    # quant_method is a dict with weight/activation/group_size keys
    quant_method = getattr(quant_config, "quant_method", {}) if quant_config else {}
    group_size = quant_method.get("group_size", None) if isinstance(quant_method, dict) else None
    num_workers = quant_method.get("num_workers", 32) if isinstance(quant_method, dict) else 32

    # Quantize model weights based on compression config
    engine.quantize(
        model_path=model_config.model_path,
        output_dir=global_config.save_path,
        quant_name=quant_name,
        group_size=group_size,
        ignore_layers=ignore_layers,
        num_workers=num_workers,
    )


def run(config):
    """
    Run the LLM compression process based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing
                       parameters for LLM compression.
    """
    # Step 1: Initialize configurations
    model_config = config.model_config
    dataset_config = config.dataset_config
    compress_config = config.compression_config
    global_config = config.global_config
    transform_config = config.transform_config

    # Dispatch to vLLM calibration if calibrate config specifies vllm backend
    if (
        config.compression_config.calibrate
        and config.compression_config.calibrate.backend == "vllm"
    ):
        vllm_calibrate_run(config)
        return

    # Step 2: Execute complete pipeline
    slim_engine = Engine()

    # Step 3: Prepare model
    slim_engine.prepare_model(
        model_name=model_config.name,
        model_path=model_config.model_path,
        torch_dtype=model_config.torch_dtype,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        use_cache=model_config.use_cache,
        cache_dir=model_config.cache_dir,
        use_audio_in_video=model_config.use_audio_in_video,
        attn_implementation=model_config.attn_implementation,
        deploy_backend=global_config.deploy_backend,
        build_model=compress_config.need_build_model,
    )

    # Step 4: Prepare data (optional custom dataloader)
    if compress_config.need_dataset:
        slim_engine.prepare_data(
            data_path=dataset_config.data_path,
            data_type=dataset_config.name,
            custom_dataloader=None,
            max_length=dataset_config.max_seq_length,
            batch_size=dataset_config.batch_size,
            num_samples=dataset_config.num_samples,
            shuffle=dataset_config.shuffle,
            inference_settings=dataset_config.inference_settings,
            use_audio_in_video=model_config.use_audio_in_video,
            model_name=model_config.name,
            quantization_config=compress_config.quantization,
        )

    # Step 5: Initialize compressor
    slim_engine.prepare_compressor(
        compress_name=compress_config.name,
        compress_config=compress_config,
        global_config=global_config,
        transform_config=transform_config,
    )

    # Step 6: Compress model
    slim_engine.run()

    # Step 7: Eval
    if args.lm_eval:
        slim_engine.lm_eval(
            tasks="arc_easy,hellaswag",
            batch_size=4,
            num_fewshot=0,
        )

    # Step 8: Save compressed model
    slim_engine.save(global_config.save_path, config)


if __name__ == "__main__":
    args = get_args()
    parser = SlimConfigParser()
    config = parser.parse(args.config)
    merge_config(config, args)
    print_config(config)
    if args.multi_nodes:
        multi_nodes_run(config)
    else:
        run(config)
