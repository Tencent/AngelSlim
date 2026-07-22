"""Convert a supported HF checkpoint to a reshardable mcore checkpoint.

Run this once before MCoreQAD training:

    python -m angelslim.compressor.mcore_qad.tools.hf_to_megatron \
        --hf <hf_dir> --out <checkpoint_dir>
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os

import torch

from ..checkpoint import save_dist_checkpoint
from ..mcore.dist import init_model_parallel, teardown
from ..mcore.model import build_gpt_model
from ..models.base import auto_config, get_adapter, load_hf_config


def _load_hf_state_dict(hf_dir: str):
    from safetensors.torch import load_file

    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as index_file:
            files = sorted(set(json.load(index_file)["weight_map"].values()))
    else:
        files = [
            os.path.basename(path) for path in glob.glob(os.path.join(hf_dir, "*.safetensors"))
        ]
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_dir}.")

    state_dict = {}
    for filename in files:
        state_dict.update(load_file(os.path.join(hf_dir, filename), device="cpu"))
    return state_dict


def convert(hf_path: str, output_path: str, use_cpu: bool = False) -> None:
    """Convert one registered model from HF tensors to mcore tensors."""
    init_model_parallel()
    try:
        hf_config = load_hf_config(hf_path)
        model_type = hf_config["model_type"]
        config, metadata = auto_config(hf_config, params_dtype=torch.bfloat16)
        if use_cpu:
            config.use_cpu_initialization = True

        print(
            f"loading HF safetensors (cpu), model_type={model_type} ...",
            flush=True,
        )
        hf_state_dict = _load_hf_state_dict(hf_path)
        state_dict = get_adapter(model_type).convert_fn(
            hf_state_dict,
            config,
            metadata,
        )
        del hf_state_dict
        gc.collect()

        device = "cpu" if use_cpu else "cuda"
        model = build_gpt_model(config, metadata).to(device)
        target = model.state_dict()
        missing_keys = sorted(target.keys() - state_dict.keys())
        unexpected_keys = sorted(state_dict.keys() - target.keys())
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "HF->MCore conversion was not exact: "
                f"missing={missing_keys[:20]} unexpected={unexpected_keys[:20]}"
            )
        state_dict = {key: value.to(target[key].dtype) for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        print("loaded HF->mcore: missing=0 unexpected=0", flush=True)
        save_dist_checkpoint(model, output_path)
        print(f"saved dist-checkpoint -> {output_path}", flush=True)
    finally:
        teardown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf",
        required=True,
        help="HF model directory (config.json + safetensors)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output mcore distributed-checkpoint directory",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Build on CPU for models that do not fit on one GPU",
    )
    args = parser.parse_args()
    convert(args.hf, args.out, use_cpu=args.cpu)


if __name__ == "__main__":
    main()
