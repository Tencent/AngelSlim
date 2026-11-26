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


"""vLLM engine helpers shared by the calibration and Hessian entry points: option parsing,
MTP draft-model access, and the Hy4-VL JSONL -> prompts path.

MTP stat keys are remapped from the draft's own ``layers.{num_hidden_layers+i}`` to
checkpoint form (``mtp_layers.{i}``). VLM images must be opened exactly as in
``image_embeds.precompute``: the embedding cache is keyed by a ``pixel_values`` hash, so any
decode difference misses and re-enables the live ViT."""

import argparse
import json
import os
import platform
import re

__all__ = [
    "apply_on_draft_model",
    "attention_backend",
    "build_prompts",
    "flatten_angelslim_yaml",
    "get_draft_model_from_worker",
    "hf_config",
    "load_records",
    "parse_with_yaml",
    "patch_hpc_version",
    "remap_mtp_stats_to_checkpoint",
    "store_true",
]


def patch_hpc_version():
    """The HPC vLLM build's version string ends in '+', which vLLM's own check rejects."""
    version = platform.python_version
    platform.python_version = lambda: version().rstrip("+")


def hf_config(model_path):
    """HF config via vLLM, whose registry knows model types (e.g. ``hy_v4_vl``) that the
    installed transformers may not."""
    from vllm.transformers_utils.config import get_config

    return get_config(model_path, trust_remote_code=True)


def attention_backend(model_path, override=None):
    """Explicit override, else HPC_MLA_SPARSE for DSA, HPC_MLA for MLA, else vLLM's default.
    Community vLLM has no HPC_* backends and keeps its own choice."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if override:
        return AttentionBackendEnum[override]
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    text = cfg.get("text_config") or {}
    if cfg.get("index_topk") is not None or text.get("index_topk") is not None:
        return getattr(AttentionBackendEnum, "HPC_MLA_SPARSE", None)
    if cfg.get("kv_lora_rank") or text.get("kv_lora_rank"):
        return getattr(AttentionBackendEnum, "HPC_MLA", None)
    return None


def flatten_angelslim_yaml(cfg):
    """Nested AngelSlim YAML (model/dataset/compression.calibrate) -> flat option names.
    Unset calibration dirs fall back to ``global.save_path``, as ckpt_convert does."""
    from ....utils.config_parser import default_run_dirs

    model = cfg.get("model") or {}
    data = cfg.get("dataset") or {}
    compression = cfg.get("compression") or {}
    calib = compression.get("calibrate") or {}
    args = dict(calib.get("args") or {})
    save_path = (cfg.get("global") or {}).get("save_path")
    dirs = default_run_dirs(save_path) if save_path else {}
    hessian_dir = calib.get("hessian_dir", None) or dirs.get("hessian_dir", None)
    stats_dir = calib.get("stats_dir", None) or dirs.get("stats_dir", None)
    args.setdefault("image_embeds_dir", dirs.get("image_embeds_dir"))
    flat = {
        "model_path": model.get("model_path"),
        "ptq_data_path": data.get("data_path"),
        "max_length": data.get("max_seq_length"),
        "num_samples": data.get("num_samples"),
        "tp_size": calib.get("tp_size"),
        "batch_size": calib.get("max_num_seqs"),
        "distributed_executor_backend": calib.get("distributed_executor_backend"),
        "hessian_dir": hessian_dir,
    }
    if stats_dir:
        flat["output_dir"] = stats_dir
        flat["moe_stats_out"] = os.path.join(stats_dir, "moe_expert_stats.json")
        if args.get("enable_mtp"):
            flat["mtp_moe_stats_out"] = os.path.join(stats_dir, "mtp_moe_expert_stats.json")
    flat.update(args)
    return {k: v for k, v in flat.items() if v is not None}


def parse_with_yaml(parser, defaults, argv=None, overrides=None, required=()):
    """Precedence: CLI flag > ``overrides`` > YAML (``-c``, keys = dest names) > ``defaults``.
    Every option in ``parser`` must default to None so an unset flag is detectable."""
    args = parser.parse_args(argv)
    for key, value in (overrides or {}).items():
        if not hasattr(args, key):
            parser.error(f"unknown option {key}")
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "compression" in cfg:
            cfg = flatten_angelslim_yaml(cfg)
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    missing = [k for k in required if not getattr(args, k)]
    if missing:
        parser.error("missing: " + ", ".join(missing))
    return args


def store_true(parser, flag, **kwargs):
    """``store_true`` whose unset value is None, so YAML can still set it."""
    parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None, **kwargs)


# model.layers.<idx>.<rest>  (the draft model's own naming)
_DRAFT_LAYER_RE = re.compile(r"^(?P<pre>.*?)\blayers\.(?P<idx>\d+)\.(?P<rest>.*)$")


def get_draft_model_from_worker(worker):
    """``worker -> model_runner -> drafter|speculator -> model``, or None (works for mp/ray and
    uniproc). The V2 model runner names it ``speculator``."""
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        return None
    drafter = getattr(model_runner, "drafter", None) or getattr(model_runner, "speculator", None)
    if drafter is None:
        return None
    return getattr(drafter, "model", None)


def apply_on_draft_model(worker, fn):
    """``collective_rpc``-compatible: run ``fn`` on this worker's draft model, or return None."""
    draft_model = get_draft_model_from_worker(worker)
    if draft_model is None:
        return None
    return fn(draft_model)


def remap_mtp_stats_to_checkpoint(stats: dict, num_hidden_layers: int) -> dict:
    """``layers.{num_hidden_layers + i}.X`` -> ``mtp_layers.{i}.X``; other keys pass through."""
    out = {}
    for key, value in stats.items():
        m = _DRAFT_LAYER_RE.match(key)
        if m is None:
            out[key] = value
            continue
        idx = int(m.group("idx"))
        if idx < num_hidden_layers:
            # Not a draft layer; avoid a negative mtp_layers index.
            out[key] = value
            continue
        out[f"{m.group('pre')}mtp_layers.{idx - num_hidden_layers}.{m.group('rest')}"] = value
    return out


def load_records(data_path: str, num_samples: int) -> list:
    """Read the ``rollout_sample_vlm.py`` JSONL; ``num_samples=0`` reads the whole file."""
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("applied_message") or rec.get("text") or rec.get("input")
            if not isinstance(text, str) or not text:
                continue
            records.append({"text": text, "images": list(rec.get("images") or [])})
            if num_samples and len(records) >= num_samples:
                break
    if not records:
        raise ValueError(f"No usable records in {data_path}")
    return records


def build_prompts(records, tokenizer, image_dir, image_pad_id, max_length, image_cap):
    """Build vLLM prompts truncated to ``max_length`` tokens, keeping image/pad counts equal
    (a mismatch crashes vLLM). Returns ``(prompts, n_multimodal, max_images_in_any_prompt)``."""
    from PIL import Image

    prompts = []
    n_mm = 0
    max_images = 0
    n_trunc_img = 0
    n_dropped = 0

    for rec in records:
        ids = tokenizer.encode(rec["text"], add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[:max_length]
        names = rec["images"]
        if names:
            # Keep only images whose placeholder survived the cut, capped per request.
            n_pads = sum(1 for i in ids if i == image_pad_id)
            if n_pads < len(names):
                n_trunc_img += 1
            keep = min(n_pads, len(names), image_cap)
            if keep < n_pads:
                # More pads than images we can supply: drop trailing pads to match.
                pad_positions = [p for p, i in enumerate(ids) if i == image_pad_id]
                drop = set(pad_positions[keep:])
                ids = [i for p, i in enumerate(ids) if p not in drop]
            names = names[:keep]

        text = tokenizer.decode(ids)
        if not names:
            prompts.append(text)
            continue

        images = []
        for name in names:
            path = name if os.path.isabs(name) else os.path.join(image_dir, name)
            try:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            except Exception as e:
                print(f"  [warn] unreadable image {name}: {e}")
                images = None
                break
        if images is None or len(images) != sum(1 for i in ids if i == image_pad_id):
            n_dropped += 1
            continue

        prompts.append({"prompt": text, "multi_modal_data": {"image": images}})
        n_mm += 1
        max_images = max(max_images, len(images))

    if n_trunc_img:
        print(
            f"  {n_trunc_img} row(s) lost image placeholders to the max_length "
            f"cut; their trailing images were dropped to keep the "
            f"placeholder/image counts consistent."
        )
    if n_dropped:
        print(f"  {n_dropped} multimodal row(s) dropped (unreadable images)")
    return prompts, n_mm, max_images
