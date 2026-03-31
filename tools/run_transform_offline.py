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
"""Offline transform tool: apply weight-space transformations (e.g. SpinQuant)
to a pretrained model and save the result.

Usage
-----
    python tools/run_transform_offline.py -c configs/qwen3/spin/qwen3_spinquant.yaml

Optional overrides
------------------
    --model-path /path/to/model   override model.model_path in the YAML
    --save-path  /path/to/output  override global.save_path in the YAML
    --test-output-diff            verify logits are numerically unchanged after transform

Supported transforms
--------------------
    SpinQuant - Applies offline Hadamard rotations (R1/R2/R4) to suppress activation
                outliers and improve post-training quantization accuracy.
                See: https://arxiv.org/abs/2405.16406
"""

import argparse
import json
import os

import torch

from angelslim.compressor.transform import TransformFactory
from angelslim.models import SlimModelFactory
from angelslim.utils import get_yaml_prefix_simple
from angelslim.utils.config_parser import (
    SlimConfigParser,
    TransformConfig,
    print_config,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(description="AngelSlim offline transform (SpinQuant etc.)")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Override model.model_path from the YAML"
    )
    parser.add_argument(
        "--save-path", type=str, default=None, help="Override global.save_path from the YAML"
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument(
        "--test-output-diff",
        action="store_true",
        help="Verify logits are numerically unchanged after transform (atol=1e-2)",
    )
    return args


def merge_config(config, args):
    if args.save_path is not None:
        config.global_config.save_path = args.save_path
    if args.model_path is not None:
        config.model_config.model_path = args.model_path
    config.global_config.save_path = os.path.join(
        config.global_config.save_path,
        get_yaml_prefix_simple(args.config),
    )


# ---------------------------------------------------------------------------
# slim_config builder for TransformFactory
# ---------------------------------------------------------------------------


def build_slim_config(transform_config: TransformConfig, global_config, compress_config):
    """Build a slim_config dict for TransformFactory.create().

    SpinQuant.__init__ calls quant_config.get('transform_config'), so the top-level
    config must support .get().  TransformConfig is a dataclass so getattr() works
    directly for the _get() helper inside SpinQuant.
    """
    return {
        "transform_config": transform_config,
        "global_config": global_config,
        "compress_config": compress_config,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_transform(config, test_diff=False):
    """Load model, apply transform, and save the result.

    Args:
        config: FullConfig parsed from YAML (must contain transform_config).
        test_diff: If True, verify logits are numerically unchanged after transform.
    """
    model_config = config.model_config
    transform_config = config.transform_config
    global_config = config.global_config
    compress_config = config.compression_config

    if transform_config is None:
        raise ValueError(
            "No 'transform' section found in the YAML config. "
            "Please add a 'transform:' block specifying the method to use."
        )

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"[run_transform] Loading model '{model_config.name}' from {model_config.model_path}")
    slim_model = SlimModelFactory.create(
        model_config.name,
        deploy_backend=global_config.deploy_backend,
    )
    slim_model.from_pretrained(
        model_config.model_path,
        torch_dtype=(
            getattr(torch, model_config.torch_dtype, "auto")
            if model_config.torch_dtype not in ("auto", None)
            else model_config.torch_dtype
        ),
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        use_cache=model_config.use_cache,
    )

    # Populate global_config fields expected by SpinQuant logging
    global_config.update(model_config.model_path)

    # ------------------------------------------------------------------
    # 2. Apply transform (with optional output diff test)
    # ------------------------------------------------------------------
    print(f"[run_transform] Applying transform: {transform_config.name}")
    slim_config = build_slim_config(transform_config, global_config, compress_config)
    slim_model.init_ptq(slim_config)

    transform = TransformFactory.create(slim_model, slim_config)

    if test_diff:
        hf_model = slim_model.model
        tokenizer = slim_model.tokenizer
        device = next(hf_model.parameters()).device
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 32), device=device)
        print(f"[test] Input shape: {input_ids.shape}, device: {device}")
        hf_model.eval()
        with torch.no_grad():
            logits_before = hf_model(input_ids).logits.float().cpu()
        print("[test] Forward BEFORE done.")

        with torch.no_grad():
            output_ids = hf_model.generate(
                **tokenizer("你好", return_tensors="pt").to(device), max_new_tokens=32
            )
        print(
            f"[test] Generate BEFORE: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}"
        )

    transform.run()

    if test_diff:
        hf_model.eval()
        with torch.no_grad():
            logits_after = hf_model(input_ids).logits.float().cpu()
        print("[test] Forward AFTER done.")

        max_diff = (logits_before - logits_after).abs().max().item()
        mean_diff = (logits_before - logits_after).abs().mean().item()
        # diff = (logits_before - logits_after).abs().cpu().numpy()
        print(f"[test] Max  diff = {max_diff:.6e}")
        print(f"[test] Mean diff = {mean_diff:.6e}")

        with torch.no_grad():
            output_ids = hf_model.generate(
                **tokenizer("你好", return_tensors="pt").to(device), max_new_tokens=32
            )
        print(
            f"[test] Generate AFTER:  {tokenizer.decode(output_ids[0], skip_special_tokens=True)}"
        )

    # ------------------------------------------------------------------
    # 3. Save transformed model
    # ------------------------------------------------------------------
    save_path = global_config.save_path
    os.makedirs(save_path, exist_ok=True)
    print(f"[run_transform] Saving transformed model to {save_path}")
    slim_model.model.save_pretrained(save_path)
    if slim_model.tokenizer is not None:
        slim_model.tokenizer.save_pretrained(save_path)
    print(f"[run_transform] Done. Model saved to {save_path}")

    if (
        hasattr(slim_model.quant_config, "transform_config")
        and slim_model.quant_config.transform_config is not None
    ):
        quantization_config = {}
        quantization_config["transform_config"] = slim_model.quant_config.transform_config
        print(f"[run_transform] Saving transform config to {save_path}")
        with open(os.path.join(save_path, "transform_config.json"), "w") as f:
            json.dump(quantization_config, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    parser = SlimConfigParser()
    config = parser.parse(args.config)
    merge_config(config, args)
    print_config(config)
    run_transform(config, test_diff=args.test_output_diff)
