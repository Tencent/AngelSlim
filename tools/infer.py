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

from angelslim.engine import InferEngine
from angelslim.utils import get_yaml_prefix_simple
from angelslim.utils.config_parser import SlimConfigParser, print_config


def get_args():
    parser = argparse.ArgumentParser(description="AngelSlim")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--input-prompt", type=str, default=None)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="./output/")

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


def infer(config, args):
    """
    Evaluate the compression process.
    This function is a placeholder for future evaluation logic.
    """
    assert (
        config or args.model_path
    ), "Please provide a model path or a configuration file."
    slim_engine = InferEngine()

    if config:
        # Step 1: Initialize configurations
        model_config = config.model_config
        compress_config = config.compression_config
        global_config = config.global_config
        infer_config = config.infer_config

        # Step 2: Prepare model
        slim_engine.prepare_model(
            model_name=model_config.name,
            model_path=model_config.model_path,
            torch_dtype=model_config.torch_dtype,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            use_cache=model_config.use_cache,
            cache_dir=model_config.cache_dir,
            deploy_backend=global_config.deploy_backend,
        )

        # Step 4: Initialize compressor
        slim_engine.prepare_compressor(
            compress_name=compress_config.name,
            compress_config=compress_config,
            global_config=global_config,
        )
    else:
        slim_engine.from_pretrained(model_path=args.model_path)

    if config and infer_config:
        _ = slim_engine.generate(args.input_prompt, **infer_config.__dict__)
    else:
        _ = slim_engine.generate(args.input_prompt)


if __name__ == "__main__":
    args = get_args()
    config = None
    if args.config:
        parser = SlimConfigParser()
        config = parser.parse(args.config)
        merge_config(config, args)
        print_config(config)
    assert args.input_prompt, "Please provide an input prompt for inference."
    infer(config, args)
