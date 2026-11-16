(nvfp4)=

# NVFP4量化

## 概述

NVFP4（NVIDIA FP4）是一种超低比特浮点量化格式，使用 E2M1（2位指数 + 1位尾数）表示，每个元素仅占4位。通过分组量化（block_size=16）配合 FP8 E4M3 缩放因子，在大幅压缩模型体积的同时保持较好的精度。

AngelSlim 支持两种 NVFP4 量化模式：

- **NVFP4 Weight-Only**：仅对权重进行 NVFP4 量化，无需校准数据，适合快速量化部署。
- **NVFP4 W4A4**：权重和激活均进行 NVFP4 量化，需要校准数据收集激活统计信息。

## 数据格式

NVFP4 使用 E2M1 格式（4-bit 浮点），可表示的值范围为 {0, 0.5, 1, 1.5, 2, 3, 4, 6}（及其负值）。量化过程中采用两级缩放：

- **weight_scale_2**（per-tensor FP32 标量）：全局缩放因子，由权重的 absmax 计算得到。
- **weight_scale**（per-group FP8 E4M3）：每16个元素一个缩放因子，精细调节各组的量化范围。

最终权重的反量化公式为：

```
weight_dequant = weight_fp4 × weight_scale(per-group, FP8) × weight_scale_2(per-tensor, FP32)
```

## NVFP4 Weight-Only 量化

Weight-Only 模式仅量化权重，不需要校准数据集，量化速度快，适合 MoE 模型的 Expert 层量化。

### 运行示例

```shell
python3 tools/run.py -c configs/glm5/ptq/nvfp4/glm5_1_nvfp4_weight_only.yaml
```

### 配置文件说明

```yaml
# Global configuration of pipeline
global:
  save_path: ./output

# Simplified Configuration for LLM compression
model:
  name: GLM5_1
  model_path: zai-org/GLM-5.1
  trust_remote_code: true
  low_cpu_mem_usage: true
  use_cache: false
  torch_dtype: auto
  device_map: cpu

# Compression configuration
compression:
  name: PTQ
  quantization:
    name: nvfp4_weight_only   # data-free: scales computed directly from weights
    bits: 4
    cpu_convert: false
    quant_method:
      weight: "per-group"
      group_size: 16
      dequant_to_bf16: false
    ignore_layers:            # Quantize routed experts only; skip everything else.
      - "lm_head"
      - "self_attn"             # all MLA projections + DSA indexer
      - "mlp.gate"              # MoE router
      - "mlp.shared_experts"    # shared expert MLP
      - "model.layers.0.mlp"    # leading dense MLP layers (first_k_dense_replace = 3)
      - "model.layers.1.mlp"
      - "model.layers.2.mlp"
```

### 配置参数说明

| 参数 | 值 | 含义 |
|------|------|------|
| `quantization.name` | `nvfp4_weight_only` | 指定使用 NVFP4 仅权重量化算法 |
| `quantization.bits` | `4` | 量化位宽为4位 |
| `quantization.cpu_convert` | `false` | 是否在 CPU 上执行量化转换（设为 `true` 可减少 GPU 显存占用） |
| `quant_method.weight` | `per-group` | 权重按组量化 |
| `quant_method.group_size` | `16` | 每16个元素为一组（NVFP4 block_size） |
| `quant_method.dequant_to_bf16` | `false` | 不将量化权重反量化回 BF16 |
| `ignore_layers` | 列表 | 跳过量化的层（attention、router、shared experts、dense MLP 等） |

### 设计要点

- **无需校准数据**：缩放因子直接从权重张量的 absmax 计算，无需前向推理。
- **Expert-Only 量化**：通过 `ignore_layers` 配置，仅对 MoE Expert 层的权重做量化，保持 Attention、Router、Shared Experts 等关键路径为原始精度。
- **CPU 转换**：设置 `cpu_convert: true` 可在 CPU 上完成量化转换，适合大模型场景避免 GPU OOM。

## 端到端流水线

如需将 NVFP4 Weight-Only 量化与 vLLM 激活校准、权重合并串联为可部署的完整模型，请参考 [scripts/ptq/README.md](../../../../scripts/ptq/README.md) 中的 GLM-5 系列脚本（`scripts/ptq/run_vllm_quant_for_glm5.sh`）。
