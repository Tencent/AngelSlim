# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AngelSlim is a large model compression toolkit by Tencent that supports quantization (FP8/INT8/INT4), speculative decoding (Eagle3), and diffusion model optimization. It targets LLMs, VLMs, Diffusion models, and Omni models.

## Common Commands

### Installation
```bash
# Standard install
pip install angelslim

# Install with extras
pip install angelslim[speculative]   # Eagle3 speculative decoding
pip install angelslim[multimodal]    # VLM support
pip install angelslim[diffusion]     # Diffusion models
pip install angelslim[all]           # Everything

# Development install
python setup.py install

# Or set PYTHONPATH for source changes
export PYTHONPATH=/path/to/AngelSlim/:$PYTHONPATH
```

### Running Quantization
```bash
# Run quantization via YAML config
python3 tools/run.py -c configs/<model>/<method>/<config>.yaml

# Example: Qwen3-1.7B FP8 static quantization
python3 tools/run.py -c configs/qwen3/fp8_static/qwen3-1_7b_fp8_static.yaml

# Override paths via CLI
python3 tools/run.py -c <config.yaml> --model-path <path> --save-path <output>

# Multi-node quantization
python3 tools/run.py -c <config.yaml> --multi-nodes
```

### Speculative Decoding (Eagle3)
```bash
# Start vLLM server
bash scripts/speculative/run_vllm_server.sh

# Generate training data
bash scripts/speculative/generate_data_for_target_model.sh

# Train Eagle3 model
bash scripts/speculative/train_eagle3_online.sh
bash scripts/speculative/train_eagle3_offline.sh
```

### Diffusion Model Quantization
```bash
python scripts/diffusion/run_diffusion.py \
  --model-name-or-path <model> \
  --quant-type fp8-per-tensor \
  --prompt "..." --height 1024 --width 1024
```

### Deployment
```bash
# vLLM server (requires vllm>=0.8.5.post1)
bash scripts/deploy/run_vllm.sh --model-path $MODEL_PATH --port 8080 -d 0,1,2,3 -t 4

# SGLang server (requires sglang>=0.4.6.post1)
bash scripts/deploy/run_sglang.sh --model-path $MODEL_PATH --port 8080 -d 0,1,2,3 -t 4

# Offline inference test
python scripts/deploy/offline.py $MODEL_PATH "Hello, my name is"

# OpenAI-compatible API call
bash scripts/deploy/openai.sh -m $MODEL_PATH -p "prompt" --port 8080
```

### Evaluation
```bash
# lm-evaluation-harness (requires lm-eval>=0.4.8)
bash scripts/deploy/lm_eval.sh -d 0,1 -t 2 -g 0.8 -r $RESULT_PATH \
  --tasks ceval-valid,mmlu,gsm8k,humaneval $MODEL_PATH
```

## Architecture

### Core Package Structure (`angelslim/`)

```
angelslim/
├── engine.py           # Main Engine, InferEngine, SpecEngine classes
├── models/             # Model definitions (LLM, VLM, Diffusion, Omni)
│   └── model_factory.py  # SlimModelFactory with @register decorator
├── compressor/         # Compression algorithms
│   ├── compressor_factory.py  # CompressorFactory with @register decorator
│   ├── quant/          # Quantization modules (PTQ, FP8, INT8, AWQ, GPTQ, NVFP4)
│   ├── speculative/    # Eagle3 training and benchmarking
│   └── diffusion/      # Diffusion model quantization and caching
├── data/               # DataLoaderFactory and dataset implementations
└── utils/              # Config parsing, helpers
```

### Factory Pattern

Models and compressors use factory registration patterns:

```python
# Register a model
@SlimModelFactory.register
class Qwen(BaseModel):
    ...

# Register a compressor
@CompressorFactory.register
class PTQ:
    ...
```

### Engine Workflow

The main quantization pipeline follows:
1. `Engine.prepare_model()` - Load model via SlimModelFactory
2. `Engine.prepare_data()` - Create calibration dataloader via DataLoaderFactory
3. `Engine.prepare_compressor()` - Initialize compressor via CompressorFactory
4. `Engine.run()` - Execute calibration
5. `Engine.save()` - Save quantized model

### Configuration System

YAML configs in `configs/` define:
- `global`: save_path, deploy_backend
- `model`: name, model_path, torch_dtype, device_map
- `compression`: name (PTQ), quantization method/bits/ignore_layers
- `dataset`: name, data_path, max_seq_length, num_samples

### Supported Quantization Methods

Available via `engine.get_supported_compress_method()`:
- `fp8_static`, `fp8_dynamic` - FP8 quantization
- `int8_dynamic` - INT8 dynamic quantization
- `int4_awq`, `int4_gptq` - INT4 weight-only quantization
- `w4a8_fp8` - Mixed precision (W4A8)

### Model Series

Models are categorized into series (LLM, VLM, Diffusion, Omni) based on module path during registration. Series affects dataloader selection and inference behavior.

## Key Files

- `tools/run.py` - Main CLI entry point for quantization
- `tools/train_eagle3_*.py` - Eagle3 speculative decoding training
- `tools/spec_benchmark.py`, `tools/vllm_spec_benchmark.py` - Benchmarking
- `scripts/deploy/*.sh` - Deployment scripts for vLLM/SGLang
- `configs/` - Pre-configured YAML files organized by model and quantization method
