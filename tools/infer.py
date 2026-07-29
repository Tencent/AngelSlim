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
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="If set, also write the generated text to this file.",
    )
    parser.add_argument(
        "--print-full",
        action="store_true",
        help="Print prompt + generated text instead of only the new tokens.",
    )

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


def _report_generation(slim_engine, prompt, output_ids, args):
    """Decode ``output_ids`` and print (and optionally save) the result.

    ``InferEngine.generate`` returns HF ``generate`` token ids, which INCLUDE the
    prompt tokens. We decode the full sequence, then derive the newly generated
    text by re-tokenizing the prompt and dropping that many leading tokens (robust
    to batch dim). Falls back to printing the full decoded text if anything about
    the slicing is unexpected.
    """
    tokenizer = getattr(slim_engine.slim_model, "tokenizer", None)
    if tokenizer is None or output_ids is None:
        print("[infer] no tokenizer / no generation output to decode.")
        return None

    seq = output_ids[0] if hasattr(output_ids, "ndim") and output_ids.ndim > 1 else output_ids
    full_text = tokenizer.decode(seq, skip_special_tokens=True)

    # New tokens = generated sequence minus the prompt prefix.
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
    try:
        new_ids = seq[prompt_len:]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    except Exception:  # noqa: BLE001 — decoding must never crash the CLI
        new_text = full_text

    to_print = full_text if args.print_full else new_text
    print("=== generated ===")
    print(to_print)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(to_print)
        print(f"[infer] wrote generation to {args.output_file}")
    return to_print


def infer(config, args):
    """
    Evaluate the compression process.
    This function is a placeholder for future evaluation logic.
    """
    assert config or args.model_path, "Please provide a model path or a configuration file."
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
            attn_implementation=getattr(model_config, "attn_implementation", "default"),
        )

        # Step 3: Initialize compressor
        slim_engine.prepare_compressor(
            compress_name=compress_config.name,
            compress_config=compress_config,
            global_config=global_config,
        )

        # Step 4: For Sparsity, apply the attention patch here — convert() ->
        # patcher is what actually swaps the attention forward; without it
        # `compression.name: Sparsity` would load a DENSE model and silently
        # ignore the sparse config. This is gated on Sparsity ONLY: PTQ / QAT /
        # QAD / Distill must NOT auto-run calibration/convert from an inference
        # entrypoint (that would need a dataset and mutate the model), matching
        # the pre-sparse infer.py behavior for those methods. The compress name
        # may be a str or a list (the parser yields e.g. ['Sparsity']).
        _cname = getattr(compress_config, "name", None)
        _cnames = _cname if isinstance(_cname, (list, tuple)) else [_cname]
        if "Sparsity" in _cnames:
            slim_engine.run()  # no-op for sparse (no calibration), kept for parity
            slim_engine.convert()  # patches the attention forward
    else:
        slim_engine.from_pretrained(model_path=args.model_path)

    if config and infer_config:
        output_ids = slim_engine.generate(args.input_prompt, **infer_config.__dict__)
    else:
        output_ids = slim_engine.generate(args.input_prompt)

    # Decode + surface the result: a bare `_ = generate(...)` made the CLI look
    # like it "ran but produced nothing" for community users.
    return _report_generation(slim_engine, args.input_prompt, output_ids, args)


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
