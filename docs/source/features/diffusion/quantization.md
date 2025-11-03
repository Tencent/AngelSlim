# Diffusion模型量化

AngelSlim 提供高效灵活的 Diffusion Transformer (DiT) 模型 FP8 量化能力。通过少量校准数据（无需训练）实现模型压缩，提升推理效率并降低部署门槛。

## 支持的量化类型

AngelSlim 支持以下四种 FP8 量化策略：

- **fp8-per-tensor**：全局 per-tensor 量化（推荐，速度/精度平衡最佳）
- **fp8-per-tensor-weight-only**：仅对权重量化（权重：FP8，激活仍为 BF16/FP16），适合对精度有更高要求的场景
- **fp8-per-block**：支持 per-block 量化，适用于 NVIDIA Hopper (SM90+) 架构，block_size目前只支持128
- **fp8-per-token**：精细的 per-token 量化，对多样输入有更强适应性

## 配置

DynamicDiTQuantizer 类提供灵活的配置选项，您可以通过以下参数自定义量化行为：

### 构造函数参数

- `quant_type`（str）：量化类型，可选值 "fp8-per-tensor"、"fp8-per-tensor-weight-only"、"fp8-per-block"、"fp8-per-token"
- `include_patterns`（List[str|re.Pattern], 可选）：指定需要量化的层名称模式，支持字符串或正则表达式
- `exclude_patterns`（List[str|re.Pattern], 可选）：指定需要排除的层名称模式，支持字符串或正则表达式
- `layer_filter`（Callable, 可选）：自定义层筛选函数（高级自定义场景专用）
- `native_fp8_support`（bool, 可选）：是否启用原生FP8硬件加速（自动检测，默认None）

### 主要方法

- `convert_linear(model, scale=None)`：对模型指定层进行量化
  - `model`：需要量化的 DiT 模型
  - `scale`：可选的预计算缩放因子（dict 或 safetensors 文件路径）
- `export_quantized_weight(model, save_path)`：导出量化权重及缩放因子
  - `model`：已量化的模型
  - `save_path`：保存目录，将导出模型权重和 `fp8_scales.safetensors` 文件

## 启动量化流程

### 方式1：使用命令行工具

使用 `scripts/diffusion/run_diffusion.py` 脚本进行量化与推理：

```shell
# 在线量化并运行推理
python scripts/diffusion/run_diffusion.py \
  --model-name-or-path black-forest-labs/FLUX.1-schnell \
  --quant-type fp8-per-tensor \
  --prompt "A cat holding a sign that says hello world" \
  --height 1024 --width 1024 --steps 4 --guidance 0.0 --seed 0
```

```shell
# 量化模型并导出量化权重
python scripts/diffusion/run_diffusion.py \
  --model-name-or-path black-forest-labs/FLUX.1-schnell \
  --fp8-model-save-path /path/to/save/quantized_model \
  --quant-type fp8-per-tensor
```

```shell
# 加载已量化模型并运行推理
python scripts/diffusion/run_diffusion.py \
  --model-name-or-path black-forest-labs/FLUX.1-schnell \
  --fp8-model-load-path /path/to/quantized_model \
  --quant-type fp8-per-tensor \
  --prompt "A cat holding a sign that says hello world" \
  --height 1024 --width 1024 --steps 4 --guidance 0.0 --seed 0
```

### 方式2：使用 Python API

#### 从零开始量化

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import DynamicDiTQuantizer

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")
quantizer.convert_linear(pipe.transformer)
pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    height=1024, width=1024,
    guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_tensor.png")
```

#### 加载预量化模型和缩放因子

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
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024, width=1024,
    guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell_fp8_per_tensor.png")
```

#### 灵活的层选择

支持通过 include/exclude 模式灵活筛选需要量化的层：

```python
from angelslim.compressor.diffusion import DynamicDiTQuantizer

# 字符串匹配
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=["linear", "attention"],
    exclude_patterns=["embed", "norm"]
)

# 正则表达式匹配
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=[r".*\.linear\d+", r".*\.attn.*"],
    exclude_patterns=[r".*embed.*"]
)

# 字符串和正则混合使用
quantizer = DynamicDiTQuantizer(
    quant_type="fp8-per-tensor",
    include_patterns=["linear", r".*\.attn.*"],
    exclude_patterns=["embed", r".*norm.*"]
)
```

## 部署

量化后的模型可直接用于推理。如需导出量化权重以便后续复用，可使用以下方法：

```python
import torch
from diffusers import FluxPipeline
from angelslim.compressor.diffusion import DynamicDiTQuantizer

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
quantizer = DynamicDiTQuantizer(quant_type="fp8-per-tensor")

# 导出量化权重和缩放因子
quantizer.export_quantized_weight(pipe.transformer, save_path="/path/to/save/quantized_model/")
```

导出的模型目录将包含：
- 量化后的模型权重文件
- `fp8_scales.safetensors`：FP8 缩放因子文件

导出后可通过上述"加载预量化模型和缩放因子"的方式加载使用。



