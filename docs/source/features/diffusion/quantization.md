# AngelSlim Diffusion模型量化

AngelSlim支持高效灵活的Diffusion Transformer (DiT) 模型FP8量化。接口简单易用，可直接集成到推理流程。

## 快速开始：DiT模型FP8量化

### 方式1：加载已量化权重+缩放因子

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from angelslim.compressor.diffusion import DynamicDiTQuantizer
from safetensors.torch import load_file

# 加载量化后的transformer和缩放因子
dit = FluxTransformer2DModel.from_pretrained("/path/to/quantized_model/")
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", transformer=dit, torch_dtype=torch.bfloat16)
scale = load_file("/path/to/quantized_model/fp8_scales.safetensors")

# 使用缩放因子进行量化
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")
quantizer.convert_linear(pipe.transformer, scale=scale)
pipe.to("cuda")

# 推理示例
image = pipe("A cat holding a sign that says hello world",
    height=1024, width=1024,
    guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_tensor.png")
```

### 方式2：模型动态量化

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import DynamicDiTQuantizer

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")
quantizer.convert_linear(pipe.transformer)
pipe.to("cuda")

image = pipe("A cat holding a sign that says hello world",
    height=1024, width=1024,
    guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_tensor.png")
```

### 方式3：导出量化权重

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import DynamicDiTQuantizer

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")
quantizer.export_quantized_weight(pipe.transformer, save_path="/path/to/save/quantized_model/")
```

## 支持的FP8量化类型

- **fp8-per-tensor**：全局per-tensor量化（推荐）
- **fp8-per-block**：per-block量化，支持NVIDIA Hopper (SM90+) DeepGEMM
- **fp8-per-token**：per-token量化，粒度更细

## 灵活选择量化层

支持include/exclude字符串或正则指定量化哪些层：

```python
from angelslim.compressor.diffusion import DynamicDiTQuantizer

# 默认：常见线性层量化
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")

# 指定包含/排除某些层（字符串匹配）
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=["linear", "attention"],   # 只量化包含"linear"/"attention"的层
    exclude_patterns=["embed", "norm"]          # 跳过"embed"或"norm"层
)

# 指定正则表达式
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=[r".*\.linear\d+", r".*\.attn.*"],
    exclude_patterns=[r".*embed.*"]
)

# 混合字符串+正则
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=["linear", r".*\.attn.*"],
    exclude_patterns=["embed", r".*norm.*"]
)
```

## 主要API

### DynamicDiTQuantizer

- `quant_type`：量化类型（"fp8-per-tensor"/"fp8-per-block"/"fp8-per-token"）
- `layer_filter`：自定义筛选函数（可选）
- `include_patterns`/`exclude_patterns`：包含/排除哪些层（字符串或正则，支持混用）
- `native_fp8_support`：是否使用原生FP8支持（如支持自动检测）

#### 主要方法

- `convert_linear(model, scale=None)`：对模型进行量化，scale可选（dict/safetensors文件）
- `export_quantized_weight(model, save_path)`：导出模型及缩放因子文件（fp8_scales.safetensors）

