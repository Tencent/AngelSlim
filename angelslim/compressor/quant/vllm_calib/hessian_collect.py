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


"""CLI for one vLLM forward that collects GPTQ Hessians: build the engine, arm the hooks from
``hessian``, prefill, dump. Prompts: an angelslim dataset (``--dataset``), VLM rollout rows, or
chunked text JSONL.
CLI: ``python -m angelslim.compressor.quant.vllm_calib.hessian_collect -c <yaml> [flags]``."""

import argparse
import json
import math
import os
import re
import sys

from .hessian import (
    dump_hessian_stats,
    setup_hessian_hooks,
    setup_hessian_hooks_on_draft,
)
from .llm import attention_backend
from .llm import hf_config as load_hf_config
from .llm import (
    parse_with_yaml,
    patch_hpc_version,
    remap_mtp_stats_to_checkpoint,
    store_true,
)

DEFAULTS = {
    "tp_size": 1,
    "batch_size": 4,
    "max_length": 16384,
    "vision_budget": 8448,
    "max_images_per_prompt": 4,
    "num_samples": 0,
    # H is allocated during the forward, after the KV cache took this fraction: keep it low.
    "gpu_memory_utilization": 0.72,
    "max_tokens_per_module": 400_000,
    "rows_per_expert": 20_000,
    "expert_quota": 0.9,
    "row_stride": 1,
    "num_speculative_tokens": 1,
    "max_num_batched_tokens": 16384,
}

_COLLECTOR_ENV = {
    "hessian_dir": "VLLM_HESSIAN_DIR",
    "max_tokens_per_module": "VLLM_HESSIAN_MAX_TOKENS",
    "rows_per_expert": "VLLM_HESSIAN_ROWS_PER_EXPERT",
    "expert_quota": "VLLM_HESSIAN_EXPERT_QUOTA",
    "row_stride": "VLLM_HESSIAN_ROW_STRIDE",
    "modules": "VLLM_HESSIAN_MODULES",
    "online_smooth_path": "VLLM_MOE_ONLINE_SMOOTH_PATH",
    "image_embeds_dir": "VLLM_IMAGE_EMBEDS_DIR",
}


def parse_args(argv=None, overrides=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", help="flat YAML whose keys are option names")
    p.add_argument("--model-path")
    p.add_argument("--data-path", dest="ptq_data_path")
    p.add_argument("--dataset", help="angelslim DataLoaderFactory type, e.g. TextDataset")
    p.add_argument("--image-dir", help="VLM images (default: <data dir>/images)")
    p.add_argument("--image-embeds-dir", help="offline ViT cache; skips the vision tower")
    p.add_argument("--hessian-dir", help="shared-FS output for h_*.safetensors + meta")
    p.add_argument("--tp-size", type=int)
    p.add_argument("--batch-size", type=int, help="max_num_seqs")
    p.add_argument("--max-length", type=int, help="per-prompt token budget")
    p.add_argument("--vision-budget", type=int, help="max_model_len headroom for image tokens")
    p.add_argument("--max-images-per-prompt", type=int)
    p.add_argument("--num-samples", type=int, help="0 = all rows")
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--distributed-executor-backend", choices=["ray", "mp"])
    p.add_argument("--attention-backend", help="override the backend derived from config.json")
    p.add_argument("--max-num-batched-tokens", type=int)
    store_true(p, "--skip-weight-loading", help="dummy weights")
    p.add_argument("--max-tokens-per-module", type=int, help="rows per shared H; 0 = no cap")
    p.add_argument("--rows-per-expert", type=int, help="row budget per expert of down_proj")
    p.add_argument("--expert-quota", type=float, help="fraction of experts that must fill")
    p.add_argument("--row-stride", type=int, help="keep every N-th row")
    p.add_argument("--modules", help="regex over module names (re.search)")
    p.add_argument("--online-smooth-path", help="online smooth scales for down_proj inputs")
    store_true(p, "--enable-mtp", help="also collect the MTP draft")
    p.add_argument("--num-speculative-tokens", type=int)
    p.add_argument("--moe-stats-out", help="trunk moe_expert_stats.json")
    p.add_argument("--mtp-moe-stats-out", help="MTP stats, in checkpoint key form")
    args = parse_with_yaml(
        p, DEFAULTS, argv, overrides, required=("model_path", "ptq_data_path", "hessian_dir")
    )
    if args.mtp_moe_stats_out and not args.moe_stats_out:
        p.error("--mtp-moe-stats-out needs --moe-stats-out")
    return args


def load_chunks(path, tokenizer, max_length, num_samples):
    """Chunked text JSONL as token ids: re-encoding keeps the chunk seams exact."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line) if line.strip() else {}
            text = rec.get("applied_message") or rec.get("text")
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_length] if text else []
            if len(ids) >= 16:  # llama-imatrix skips chunks this short as well
                prompts.append({"prompt_token_ids": ids})
            if num_samples and len(prompts) >= num_samples:
                break
    return prompts


def build_inputs(args, tokenizer, hf_config):
    """``(prompts, max_images)``; ``max_images`` is 0 for text-only input."""
    if args.dataset:
        from ....data.dataloader import DataLoaderFactory

        loader = DataLoaderFactory.create_data_loader(
            data_type=args.dataset, processor=tokenizer, device="cpu",
            max_length=args.max_length, batch_size=1, shuffle=False,
            num_samples=args.num_samples, data_source=args.ptq_data_path,
        )  # fmt: skip
        ids = [b["input_ids"][0][b["attention_mask"][0].bool()].tolist() for b in loader]
        return [{"prompt_token_ids": x} for x in ids], 0
    pad_id = getattr(hf_config, "image_token_id", None)
    if pad_id is None:
        return load_chunks(args.ptq_data_path, tokenizer, args.max_length, args.num_samples), 0

    from .llm import build_prompts, load_records

    image_dir = args.image_dir or os.path.join(os.path.dirname(args.ptq_data_path), "images")
    prompts, n_mm, max_images = build_prompts(
        load_records(args.ptq_data_path, args.num_samples), tokenizer, image_dir,
        int(pad_id), args.max_length, args.max_images_per_prompt,
    )  # fmt: skip
    print(f"[hessian] {len(prompts)} prompts, {n_mm} multimodal")
    return prompts, max(1, max_images)


def trunk_depth(hf_config):
    """``num_hidden_layers`` of the text stack: draft layers are indexed from it."""
    return getattr(getattr(hf_config, "text_config", hf_config), "num_hidden_layers", 0)


_LAYER_RE = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.")


def write_moe_stats(metas, out_path, mtp_out_path, depth):
    """Reduce per-rank MoE min/max; MTP groups (layer >= depth) go to ``mtp_out_path``."""
    merged = {}
    for meta in metas:
        for group, st in (meta.get("moe_stats") or {}).items():
            cur = merged.setdefault(group, dict(st))
            cur["min"], cur["max"] = min(cur["min"], st["min"]), max(cur["max"], st["max"])
    trunk, mtp = {}, {}
    for group, st in merged.items():
        if not (math.isfinite(st["min"]) and math.isfinite(st["max"])):
            continue  # matched but never reached by a token
        m = _LAYER_RE.match(group)
        is_mtp = (
            ".mtp_block." in group
            or ".mtp_layers." in group
            or (bool(depth) and m is not None and int(m.group(1)) >= depth)
        )
        (mtp if is_mtp else trunk)[group] = {"min": st["min"], "max": st["max"]}
    outputs = [(out_path, trunk)]
    if mtp_out_path:
        outputs.append((mtp_out_path, remap_mtp_stats_to_checkpoint(mtp, depth) if depth else mtp))
    for path, stats in outputs:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=1)
        print(f"[hessian] {len(stats)} MoE stats groups -> {path}")
    return len(trunk), len(mtp)


def main(argv=None, overrides=None):
    args = parse_args(argv, overrides)
    os.makedirs(args.hessian_dir, exist_ok=True)
    os.environ["VLLM_COLLECT_HESSIAN"] = "1"
    for key, env in _COLLECTOR_ENV.items():
        if getattr(args, key, None) is not None:
            os.environ[env] = str(getattr(args, key))
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc callables
    # H lands outside vLLM's budget, so the allocator ends up holding GiBs of
    # reserved-but-unallocated blocks; expandable segments give those back.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    patch_hpc_version()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from . import image_embeds, worker_bootstrap

    hf_config = load_hf_config(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts, max_images = build_inputs(args, tokenizer, hf_config)
    if not prompts:
        raise ValueError(f"no usable prompts in {args.ptq_data_path}")

    kwargs = dict(
        model=args.model_path, load_format="dummy" if args.skip_weight_loading else "auto",
        enforce_eager=True, enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization, tensor_parallel_size=args.tp_size,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=False, max_num_seqs=args.batch_size,
        max_model_len=args.max_length + (args.vision_budget if max_images else 0) + 16,
        # vllm_calib.hooks reads the routed-expert inputs from TritonExperts.apply.
        moe_backend="triton",
        attention_backend=attention_backend(args.model_path, args.attention_backend),
    )  # fmt: skip
    if args.enable_mtp:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if max_images:
        # "data" replicates the 16-head vision tower; "weights" cannot split it over TP=32.
        kwargs.update(
            limit_mm_per_prompt={"image": max_images},
            mm_encoder_tp_mode="data",
            mm_encoder_attn_backend="TORCH_SDPA",
            **image_embeds.engine_kwargs(),
        )
    llm = LLM(**kwargs)

    rpc = llm.llm_engine.collective_rpc
    rpc(worker_bootstrap())
    print("[hessian] " + str(llm.apply_model(setup_hessian_hooks)[0]))
    if args.enable_mtp:
        print("[hessian] draft: " + str(rpc(setup_hessian_hooks_on_draft)[0]))

    # The drafter only runs in a decode step's propose phase, so MTP needs max_tokens > 1.
    max_tokens = max(8, 4 * args.num_speculative_tokens) if args.enable_mtp else 1
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, detokenize=False))

    metas = [m for m in llm.apply_model(dump_hessian_stats) if m]
    groups, modules = {}, {}
    for m in metas:
        groups.update(m["groups"])
        modules.update(m["modules"])
    if not groups:
        print("[hessian] nothing collected: check --modules and the [moe_hook] log lines")
        return 1
    path = os.path.join(args.hessian_dir, "hessian_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"groups": groups, "modules": modules}, f, indent=1)
    print(f"[hessian] {len(groups)} groups, {len(modules)} modules -> {path}")

    if args.moe_stats_out:
        n_trunk, n_mtp = write_moe_stats(
            metas, args.moe_stats_out, args.mtp_moe_stats_out, trunk_depth(hf_config)
        )
        if not n_trunk or (args.mtp_moe_stats_out and not n_mtp):
            print("[hessian] ERROR: empty MoE stats would quantize with input_scale=1.0")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
