# AngelSlim mcore QAD backend

This package provides AngelSlim's isolated mcore backend for distributed,
scale-only quantization-aware training and distillation.

It supports Qwen3-MoE and Hy3 (HunYuan-3), with six formats: `nvfp4`, `nvfp4a16`, `w4a16`, `w8a8`, `fp8`, and `w4afp8`.

AngelSlim's YAML runner automatically creates a missing reshardable mcore checkpoint
from the configured Hugging Face model before scale-only training starts.

This package is distributed under AngelSlim's Apache-2.0 license.
