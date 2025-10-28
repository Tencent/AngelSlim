# AngelSlim Diffusion Model Compression

AngelSlim offers flexible and efficient tools for compressing Diffusion Transformer (DiT) diffusion models. The quantization utilities are modular and easy to integrate into custom inference pipelines.

## Quick Start: FP8 Quantization for Diffusion Models

### Method 1: Quantize with Pre-computed Scales

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from angelslim.compressor.diffusion import DynamicDiTQuantizer
from safetensors.torch import load_file

# Load pre-quantized transformer and scales
dit = FluxTransformer2DModel.from_pretrained("/path/to/quantized_model/")
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", transformer=dit, torch_dtype=torch.bfloat16)

# Load pre-computed scales
scale = load_file("/path/to/quantized_model/fp8_scales.safetensors")

# Apply quantization with scales
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")
quantizer.convert_linear(pipe.transformer, scale=scale)

pipe.to("cuda")

# Run pipeline with FP8-quantized transformer
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_tensor.png")
```

### Method 2: Quantize from Scratch

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import DynamicDiTQuantizer

# Load DiT pipeline with bfloat16 to reduce memory usage
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

# Supported quantization types: "fp8-per-tensor", "fp8-per-block", "fp8-per-token", "fp8-per-tensor-weight-only"
# If you want to use "fp8-per-block" + DeepGEMM on NVIDIA Hopper (SM90+) devices,
# please refer to https://github.com/deepseek-ai/DeepGEMM for installation instructions.
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")
quantizer.convert_linear(pipe.transformer)

pipe.to("cuda")

# Run pipeline with FP8-quantized transformer
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_tensor.png")
```

### Method 3: Export Quantized Model

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import DynamicDiTQuantizer

# Load and quantize model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")

# Export quantized weights and scales
quantizer.export_quantized_weight(pipe.transformer, save_path="/path/to/save/quantized_model/")
```

## Supported Quantization Types

AngelSlim supports four FP8 quantization strategies:

- **`fp8-per-tensor`**: Per-tensor quantization for both weights and activations (recommended for most use cases)
- **`fp8-per-tensor-weight-only`**: Weight-only quantization with per-tensor scaling (weights: FP8, activations: BF16/FP16)
- **`fp8-per-block`**: Per-block quantization with DeepGEMM support for NVIDIA Hopper (SM90+) devices
- **`fp8-per-token`**: Per-token quantization for fine-grained control

## Customizable Quantization Layer Selection

AngelSlim provides fine-grained control over which layers are quantized. You can specify inclusion and exclusion patterns as substrings or regular expressions.

```python
from angelslim.compressor.diffusion import DynamicDiTQuantizer

# Option 1: Default filtering (quantizes common linear layers)
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")

# Option 2: String-based include/exclude patterns
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=["linear", "attention"],
    exclude_patterns=["embed", "norm"]
)

# Option 3: Regex pattern matching (auto-detected)
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=[r".*\.linear\d+", r".*\.attn.*"],
    exclude_patterns=[r".*embed.*"]
)

# Option 4: Mix of strings and regex for flexible rules
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=["linear", r".*\.attn.*"],
    exclude_patterns=["embed", r".*norm.*"]
)
```

## API Reference

### DynamicDiTQuantizer

The main quantizer class for DiT models.

#### Constructor Parameters

- `quant_type` (str): Quantization type - "fp8-per-tensor", "fp8-per-tensor-weight-only", "fp8-per-block", or "fp8-per-token"
- `layer_filter` (Callable, optional): Custom function to determine which layers to quantize
- `include_patterns` (List[str|re.Pattern], optional): Patterns for layers to include
- `exclude_patterns` (List[str|re.Pattern], optional): Patterns for layers to exclude
- `native_fp8_support` (bool, optional): Whether to use native FP8 support (auto-detected if None)

#### Methods

- `convert_linear(model, scale=None)`: Convert linear layers to quantized versions
  - `model`: The DiT model to quantize
  - `scale`: Optional pre-computed scales (dict or safetensors file)
- `export_quantized_weight(model, save_path)`: Export quantized model and scales
  - `model`: The quantized model
  - `save_path`: Directory to save the model and fp8_scales.safetensors

For more details on customizing quantization behavior, see the API documentation.
