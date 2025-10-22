# AngelSlim Diffusion Model Compression

AngelSlim offers flexible and efficient tools for compressing Diffusion Transformer (DiT) diffusion models. The quantization utilities are modular and easy to integrate into custom inference pipelines.

## Quick Start: FP8 Quantization for Diffusion Models

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import DynamicDiTQuantizer

# Load DiT pipeline with bfloat16 to reduce memory usage
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

# Supported quantization types: "fp8-per-tensor", "fp8-per-block", "fp8-per-token"
# If you want to use "fp8-per-block" + DeepGEMM on NVIDIA Hopper (SM90+) devices,
# please refer to https://github.com/deepseek-ai/DeepGEMM for installation instructions.
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")
quantizer.quantize(pipe.transformer)

pipe.to("cuda")

# Run pipeline with FP8-quantized transformer
image = pipe(
    "A cat holding a sign that says hello world",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_tensor.png")
```

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

For more details on customizing quantization behavior, see the API documentation.
