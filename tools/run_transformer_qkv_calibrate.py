"""
Transformer-based QKV Scale Calibration Tool

This script calibrates q_scale / k_scale / v_scale for HunyuanV3 MoE models
using HuggingFace Transformers inference (no vLLM required).

The output format is identical to the vLLM-based run_vllm_calibrate_and_quant.py:
  - kv_scales.safetensors  (per-layer q_scale / k_scale / v_scale tensors)
  - kv_scales.json         (same content as scalar values, sorted by layer)

Key difference from vLLM version:
  - Observers are applied via monkey-patching (apply_kvcache_observers) on the
    HuggingFace model's attention forward, capturing Q/K/V AFTER RoPE is applied.
  - No Ray / tensor-parallel required; runs on a single node with device_map="auto".
"""

import argparse
import ast
import glob
import json
import os
import shutil
from typing import List

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from angelslim.engine import Engine
from angelslim.utils import print_info


# =============================================================================
# Custom Parquet data loader for hyeval3 format
# =============================================================================

class HyevalParquetDataset(Dataset):
    """
    Dataset for hyeval3-style Parquet files.

    Expected schema:
        prompt       - stringified Python list of chat messages,
                       e.g. "[{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]"
        reward_model - (unused for calibration)
        data_source  - (unused for calibration)
        extra_info   - (unused for calibration)
    """

    def __init__(self, data_path: str, tokenizer, max_length: int, num_samples: int, device: str = "cpu"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.data: List[dict] = []
        self._load(data_path, num_samples)

    def _load(self, data_path: str, num_samples: int):
        table = pq.read_table(data_path)
        df = table.to_pandas()

        total = min(num_samples, len(df)) if num_samples > 0 else len(df)

        for i in range(total):
            prompt_raw = df["prompt"].iloc[i]

            # Parse stringified list of message dicts
            try:
                messages = ast.literal_eval(prompt_raw)
            except Exception:
                # Fallback: treat as plain text user message
                messages = [{"role": "user", "content": str(prompt_raw)}]

            # Normalize numpy str / object types in message content
            for msg in messages:
                if "content" in msg and not isinstance(msg["content"], str):
                    msg["content"] = str(msg["content"])

            # Apply chat template (no add_generation_prompt, consistent with calibrate_fp8_qkv_scales.py)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            model_inputs = self.tokenizer(
                text=[text],
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=False,
            )

            labels = model_inputs["input_ids"].squeeze(0).roll(shifts=-1, dims=-1)
            labels[-1] = -100

            self.data.append({
                "input_ids": model_inputs["input_ids"].squeeze(0).to(self.device),
                "attention_mask": model_inputs["attention_mask"].squeeze(0).to(self.device),
                "labels": labels.squeeze(0).to(self.device),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_hyeval_parquet_dataloader(
    data_path: str,
    tokenizer,
    max_length: int,
    num_samples: int,
    batch_size: int,
    device: str = "cpu",
) -> DataLoader:
    """Build a DataLoader from a hyeval3-style Parquet file."""
    dataset = HyevalParquetDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_samples,
        device=device,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# =============================================================================
# Scale saving helpers (same logic as run_vllm_calibrate_and_quant.py)
# =============================================================================

def save_qkv_scales(
    kv_scales_raw: dict,
    output_dir: str,
    qk_headroom: float = 4.0,
    v_headroom: float = 4.0,
) -> None:
    """
    Convert raw observer scales (absmax tensors) to FP8 scales with headroom,
    then save as kv_scales.safetensors and kv_scales.json.

    Args:
        kv_scales_raw: dict returned by model.get_kvcache_scales(), keys like:
            "model.layers.0.self_attn.q_cache.scale"
            "model.layers.0.self_attn.k_cache.scale"
            "model.layers.0.self_attn.v_cache.scale"
        output_dir: directory to save output files
        qk_headroom: headroom multiplier for q_scale and k_scale
        v_headroom: headroom multiplier for v_scale
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    scales = {}

    for raw_key, raw_scale_tensor in kv_scales_raw.items():
        # raw_scale_tensor is the absmax value from AbsmaxPertensorObserver
        absmax = raw_scale_tensor.float().abs().item()
        fp8_scale = absmax / fp8_max

        # "model.layers.0.self_attn.q_cache.scale" ->
        #   layer_prefix = "model.layers.0.self_attn"
        #   cache_type   = "q"
        if raw_key.endswith(".q_cache.scale"):
            layer_prefix = raw_key[: -len(".q_cache.scale")]
            out_key = f"{layer_prefix}.q_scale"
            scales[out_key] = torch.tensor([fp8_scale * qk_headroom])
        elif raw_key.endswith(".k_cache.scale"):
            layer_prefix = raw_key[: -len(".k_cache.scale")]
            out_key = f"{layer_prefix}.k_scale"
            scales[out_key] = torch.tensor([fp8_scale * qk_headroom])
        elif raw_key.endswith(".v_cache.scale"):
            layer_prefix = raw_key[: -len(".v_cache.scale")]
            out_key = f"{layer_prefix}.v_scale"
            scales[out_key] = torch.tensor([fp8_scale * v_headroom])
        else:
            print_info(f"[WARNING] Unexpected key in kv_scales_raw, skipping: {raw_key}")
            continue

    if not scales:
        print_info("[WARNING] No QKV scales collected. Check that observers were applied.")
        return scales

    os.makedirs(output_dir, exist_ok=True)

    # Save safetensors
    scales_path = os.path.join(output_dir, "kv_scales.safetensors")
    save_file(scales, scales_path)
    print_info(f"\nQKV scales (safetensors) saved to: {scales_path}")

    # Sort by layer index then q/k/v order, then save JSON
    _scale_order = {"q_scale": 0, "k_scale": 1, "v_scale": 2}

    def _sort_key(key: str):
        parts = key.split(".")
        layer_idx = 0
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        return (layer_idx, _scale_order.get(parts[-1], 99))

    sorted_scales = dict(sorted(scales.items(), key=lambda kv: _sort_key(kv[0])))
    json_path = os.path.join(output_dir, "kv_scales.json")
    with open(json_path, "w") as f:
        json.dump({k: v.item() for k, v in sorted_scales.items()}, f, indent=2)
    print_info(f"QKV scales (JSON)        saved to: {json_path}")

    # Print summary table
    num_layers = len(set(k.rsplit(".", 1)[0] for k in scales))
    has_q = any(k.endswith(".q_scale") for k in scales)
    print_info(f"\n  {num_layers} attention layers, scales: {'q/k/v' if has_q else 'k/v'}")
    print_info(
        f"  Headroom: q_scale*{qk_headroom}, k_scale*{qk_headroom}, v_scale*{v_headroom}"
    )

    header = f"  {'Layer':<55}"
    if has_q:
        header += f" {'q_scale':>10}"
    header += f" {'k_scale':>10} {'v_scale':>10}"
    print_info(f"\n{header}")
    print_info("  " + "-" * (len(header) - 2))

    layer_names = sorted(
        set(k.rsplit(".", 1)[0] for k in scales),
        key=lambda x: int(x.split(".")[-2]) if x.split(".")[-2].isdigit() else 0,
    )
    for layer_name in layer_names:
        short = layer_name.replace("model.layers.", "L").replace(".self_attn", "")
        line = f"  {short:<55}"
        if has_q:
            q_val = scales.get(f"{layer_name}.q_scale")
            line += f" {q_val.item():>10.6f}" if q_val is not None else f" {'N/A':>10}"
        k_val = scales.get(f"{layer_name}.k_scale")
        v_val = scales.get(f"{layer_name}.v_scale")
        line += f" {k_val.item():>10.6f}" if k_val is not None else f" {'N/A':>10}"
        line += f" {v_val.item():>10.6f}" if v_val is not None else f" {'N/A':>10}"
        print_info(line)

    return scales


def save_model_with_scales(
    model_dir: str,
    output_dir: str,
    scales: dict,
) -> None:
    """
    Copy original BF16 model and inject QKV scales into the safetensors shards.

    Scales are stored with their original HF names (e.g. self_attn.q_scale).
    vLLM's maybe_remap_kv_scale_name() handles the final remap to
    self_attn.attn.q_scale at load time.

    Output is compatible with: vllm serve <output_dir> --kv-cache-dtype fp8
    """
    from safetensors import safe_open

    os.makedirs(output_dir, exist_ok=True)

    # -- Copy all non-safetensors files (config, tokenizer, etc.) --
    for fname in os.listdir(model_dir):
        src = os.path.join(model_dir, fname)
        if fname.endswith(".safetensors"):
            continue
        dst = os.path.join(output_dir, fname)
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)

    # -- Process each shard: copy weights + inject scales --
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        fname = os.path.basename(path)
        with safe_open(path, framework="pt") as f:
            tensors = {}
            metadata = f.metadata()
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        # Inject scales that belong to layers in this shard
        for scale_name, scale_val in scales.items():
            layer_prefix = ".".join(scale_name.split(".")[:-1])
            if any(k.startswith(layer_prefix) for k in tensors.keys()):
                tensors[scale_name] = scale_val

        save_file(tensors, os.path.join(output_dir, fname), metadata=metadata)

    # -- Update safetensors index (sharded models) --
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

        for scale_name in scales.keys():
            layer_prefix = ".".join(scale_name.split(".")[:-1])
            for existing_key, shard in index["weight_map"].items():
                if existing_key.startswith(layer_prefix):
                    index["weight_map"][scale_name] = shard
                    break

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    # -- Add quantization_config to config.json --
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "config_groups": {},
        "kv_cache_scheme": {
            "type": "float",
            "num_bits": 8,
            "strategy": "tensor",
            "symmetric": True,
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print_info(f"BF16 model + FP8 QKV scales saved to: {output_dir}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer-based QKV Scale Calibration for HunyuanV3 MoE"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the HunyuanV3 MoE model directory",
    )
    parser.add_argument(
        "--ptq-data-path", type=str, required=True,
        help="Path to calibration data (JSONL / JSON / Parquet)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save kv_scales.safetensors and kv_scales.json",
    )
    parser.add_argument(
        "--num-samples", type=int, default=32,
        help="Number of calibration samples (default: 32)",
    )
    parser.add_argument(
        "--max-length", type=int, default=8192,
        help="Maximum sequence length (default: 8192)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for calibration inference (default: 1)",
    )
    parser.add_argument(
        "--qk-headroom", type=float, default=4.0,
        help="Headroom factor for q_scale and k_scale (default: 4.0)",
    )
    parser.add_argument(
        "--v-headroom", type=float, default=4.0,
        help="Headroom factor for v_scale (default: 4.0)",
    )
    parser.add_argument(
        "--quant-bits", type=int, default=8,
        help="Quantization bits for the observer (default: 8)",
    )
    parser.add_argument(
        "--model-output-dir", type=str, default=None,
        help="Directory to save BF16 model with FP8 QKV scales injected. "
             "If set, outputs a full model directory compatible with "
             "vLLM --kv-cache-dtype fp8. If not set, only scale files are saved.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print_info("=" * 70)
    print_info("Transformer-based QKV Scale Calibration")
    print_info("=" * 70)
    print_info(f"  Model:       {args.model_path}")
    print_info(f"  Data:        {args.ptq_data_path}")
    print_info(f"  Output:      {args.output_dir}")
    print_info(f"  Samples:     {args.num_samples}")
    print_info(f"  Max length:  {args.max_length}")
    print_info(f"  Batch size:  {args.batch_size}")
    print_info(f"  QK headroom: {args.qk_headroom}")
    print_info(f"  V  headroom: {args.v_headroom}")

    # -------------------------------------------------------------------------
    # Step 1: Load model via Engine (HuggingFace Transformers)
    # -------------------------------------------------------------------------
    print_info("\n" + "=" * 70)
    print_info("Step 1: Loading model...")
    engine = Engine()
    engine.prepare_model(
        model_name="HYV3MoE",
        model_path=args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        deploy_backend="huggingface",
    )
    slim_model = engine.slim_model
    print_info("Model loaded successfully.")

    # -------------------------------------------------------------------------
    # Step 2: Apply QKV observers (monkey-patch attention forward)
    # -------------------------------------------------------------------------
    print_info("\n" + "=" * 70)
    print_info("Step 2: Applying QKV observers (after RoPE)...")
    from angelslim.compressor.quant.observers import AbsmaxPertensorObserver

    slim_model.apply_kvcache_observers(
        kv_cache_observer_class=AbsmaxPertensorObserver,
        quant_bits=args.quant_bits,
    )
    attn_count = len(slim_model.kv_cache_observers)
    print_info(f"Registered observers on {attn_count} attention layers.")

    # -------------------------------------------------------------------------
    # Step 3: Prepare calibration data
    # -------------------------------------------------------------------------
    print_info("\n" + "=" * 70)
    print_info("Step 3: Preparing calibration data...")

    if args.ptq_data_path.lower().endswith(".parquet"):
        # Use custom loader for hyeval3-style Parquet (prompt column = stringified messages)
        dataloader = load_hyeval_parquet_dataloader(
            data_path=args.ptq_data_path,
            tokenizer=slim_model.tokenizer,
            max_length=args.max_length,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
    else:
        dataloader = engine.prepare_data(
            data_path=args.ptq_data_path,
            data_type="TextDataset",
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            shuffle=False,
        )
    print_info(f"Loaded {len(dataloader)} batches.")

    # -------------------------------------------------------------------------
    # Step 4: Run calibration inference
    # -------------------------------------------------------------------------
    print_info("\n" + "=" * 70)
    print_info("Step 4: Running calibration inference...")
    slim_model.model.eval()
    device = next(slim_model.model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calibrating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask_1d = batch["attention_mask"].to(device)  # [B, L]     
            bsz, seq_len = input_ids.shape
            # Build a 4D additive causal mask so create_causal_mask early-exits
            # (it returns as-is when mask is already 4D), bypassing the 5D path.
            # causal: [1, 1, L, L], True = attend
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
            # pad_mask: [B, 1, 1, L] — mask out padding positions in kv dim
            pad_mask = attention_mask_1d.bool().unsqueeze(1).unsqueeze(2)
            # combined: [B, 1, L, L]
            mask_4d = causal.unsqueeze(0).unsqueeze(0) & pad_mask
            # Convert to float additive mask (0.0 = attend, -inf = masked)
            additive_mask = torch.zeros(
                bsz, 1, seq_len, seq_len,
                dtype=torch.bfloat16, device=device
            ).masked_fill(~mask_4d, torch.finfo(torch.bfloat16).min)

            slim_model.model(
                input_ids=input_ids,
                attention_mask=additive_mask,
                use_cache=False,
            )

    print_info("Calibration inference done.")

    # -------------------------------------------------------------------------
    # Step 5: Restore original attention forward methods
    # -------------------------------------------------------------------------
    slim_model.remove_kvcache_observers()
    print_info("Attention forward methods restored.")

    # -------------------------------------------------------------------------
    # Step 6: Collect scales and save
    # -------------------------------------------------------------------------
    print_info("\n" + "=" * 70)
    print_info("Step 5: Collecting and saving QKV scales...")
    kv_scales_raw = slim_model.get_kvcache_scales()
    print_info(f"Collected {len(kv_scales_raw)} raw scale entries.")

    scales = save_qkv_scales(
        kv_scales_raw=kv_scales_raw,
        output_dir=args.output_dir,
        qk_headroom=args.qk_headroom,
        v_headroom=args.v_headroom,
    )

    # -------------------------------------------------------------------------
    # Step 6: Save full model with injected scales (optional)
    # -------------------------------------------------------------------------
    if args.model_output_dir is not None:
        print_info("\n" + "=" * 70)
        print_info("Step 6: Saving BF16 model + FP8 QKV scales...")
        save_model_with_scales(
            model_dir=args.model_path,
            output_dir=args.model_output_dir,
            scales=scales,
        )
        slim_model.tokenizer.save_pretrained(args.model_output_dir)

    print_info("\n" + "=" * 70)
    print_info("All done!")
    print_info(f"  QKV scales saved to: {args.output_dir}")
    if args.model_output_dir is not None:
        print_info(f"  Full model saved to: {args.model_output_dir}")
        print_info(f"  Use with vLLM: --model {args.model_output_dir} --kv-cache-dtype fp8")
    print_info("=" * 70)


if __name__ == "__main__":
    main()
