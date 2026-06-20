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
python3 tools/run.py -c configs/hunyuan/nvfp4_weight_only/hunyuan_a13b_nvfp4_weight_only.yaml
```

### 配置文件说明

```yaml
# Global configuration of pipeline
global:
  save_path: ./output

# Model configuration
model:
  name: HYV3MoE
  model_path: <your_model_path>
  trust_remote_code: true
  low_cpu_mem_usage: true
  use_cache: false
  torch_dtype: auto
  device_map: auto

# Compression configuration
compression:
  name: PTQ
  quantization:
    name: nvfp4_weight_only
    bits: 4
    cpu_convert: true
    quant_method:
      weight: "per-group"
      group_size: 16
      dequant_to_bf16: false
    ignore_layers:         # Skip quantization for these layers
      - "lm_head"
      - "self_attn.q_proj"
      - "self_attn.k_proj"
      - "self_attn.v_proj"
      - "self_attn.o_proj"
      - "mlp.router.gate"
      - "mlp.gate_proj"
      - "mlp.up_proj"
      - "mlp.down_proj"
      - "mlp.shared_mlp.gate_proj"
      - "mlp.shared_mlp.up_proj"
      - "mlp.shared_mlp.down_proj"
```

### 配置参数说明

| 参数 | 值 | 含义 |
|------|------|------|
| `quantization.name` | `nvfp4_weight_only` | 指定使用 NVFP4 仅权重量化算法 |
| `quantization.bits` | `4` | 量化位宽为4位 |
| `quantization.cpu_convert` | `true` | 在 CPU 上执行量化转换，减少 GPU 显存占用 |
| `quant_method.weight` | `per-group` | 权重按组量化 |
| `quant_method.group_size` | `16` | 每16个元素为一组（NVFP4 block_size） |
| `quant_method.dequant_to_bf16` | `false` | 不将量化权重反量化回 BF16 |
| `ignore_layers` | 列表 | 跳过量化的层（attention、router、shared MLP 等） |

### 设计要点

- **无需校准数据**：缩放因子直接从权重张量的 absmax 计算，无需前向推理。
- **Expert-Only 量化**：通过 `ignore_layers` 配置，仅对 MoE Expert 层的权重做量化，保持 Attention、Router、Shared MLP 等关键路径为原始精度。
- **CPU 转换**：设置 `cpu_convert: true` 可在 CPU 上完成量化转换，适合大模型场景避免 GPU OOM。

## 校准统计收集

合并工具所需的 activation scales 和 KV cache scales 通过运行 vLLM 校准脚本获得。该脚本基于 vLLM 推理引擎，在真实数据上做前向推理，收集各层激活和 KV cache 的统计信息（min/max）。

### 运行示例

```shell
bash scripts/ptq/run_vllm_calibrate_for_Hy3.sh
```

### 流程说明

1. **设置环境变量**：脚本通过环境变量开启 MoE Expert 统计收集功能：
   - `VLLM_MOE_COLLECT_STATS=1`：启用 MoE 统计收集
   - `VLLM_MOE_COLLECT_PER_EXPERT_STATS=1`：启用逐 Expert 统计
   - 其他推理加速配置（chunked prefill、FlashInfer attention backend 等）

2. **指定配置文件**：通过 `CONFIG` 变量指定校准用的 YAML 配置文件（需替换为您自己的配置路径）。

3. **执行校准推理**：调用 `tools/run_vllm_calibrate.py`，利用 vLLM 引擎在校准数据集上做前向推理，收集：
   - **activation_stats.json**：各层激活的 min/max 统计（用于计算 KV cache 的 FP8 缩放因子）
   - **moe_expert_stats.json**：每个 Expert 的 gate_up_proj / down_proj 激活 min/max（用于计算 Expert input_scale）

4. **输出产物**：统计文件保存在配置中指定的输出目录，后续作为合并工具的 `--statistics_path` 输入。

```{note}
运行前需将脚本中的 `CONFIG` 替换为您自己的配置文件路径，并确保 `PYTHONPATH` 包含项目根目录。
```

## 合并工具：生成推理用模型

NVFP4 Weight-Only 量化后的模型需要与上述校准统计信息（KV cache scales、激活 input_scale）合并，生成可供 vLLM 推理的完整模型。使用 `tools/merge_hy3_nvfp4_c8.py` 完成此步骤。

### 运行示例

```shell
python3 tools/merge_hy3_nvfp4_c8.py \
    --statistics_path <calibration_statistics_dir> \
    --nvfp4_w_path <nvfp4_weight_only_model_dir> \
    --output_path <merged_output_dir> \
    --bf16_model_path <original_bf16_model_dir>
```

### 参数说明

| 参数 | 含义 |
|------|------|
| `--statistics_path` | 校准统计目录，包含 `activation_stats.json` 和 `moe_expert_stats.json` |
| `--nvfp4_w_path` | NVFP4 Weight-Only 量化后的模型目录 |
| `--output_path` | 合并后模型的输出目录 |
| `--bf16_model_path` | 原始 BF16 模型路径（用于获取 shared_mlp 的 BF16 权重、config 和 tokenizer） |
| `--kv_statistics_path` | （可选）单独的 KV cache 统计文件路径 |

### 合并后模型包含

- Expert 权重：NVFP4 格式（`.weight`、`.weight_scale`、`.weight_scale_2`）
- Expert 激活缩放：`input_scale`（FP32 标量，从 `moe_expert_stats` 计算）
- KV Cache 缩放：`k_proj.k_scale`、`v_proj.v_scale`（FP8 per-tensor 缩放因子）
- 非 Expert 权重：BF16（attention、shared_mlp、layernorm、embedding、lm_head）
- `config.json`：包含 `quantization_config`（quant_method=modelopt, NVFP4, kv_cache_scheme）

## 使用前的准备工作

在运行量化前，需要将脚本中的以下内容替换为您自己的配置：

1. **配置文件路径**：将 `CONFIG` 变量指向您自己的 YAML 配置文件路径。
2. **PYTHONPATH**：确保 `PYTHONPATH` 包含 AngelSlim 项目根目录（`export PYTHONPATH=.`）。
3. **模型路径**：在 YAML 配置中将 `model_path` 修改为您的模型实际存放路径。
4. **输出路径**：在 YAML 配置中将 `save_path` 修改为您期望的输出目录。

## 完整流程总结

```{mermaid}
flowchart TD
    A[原始 BF16 模型] --> B[NVFP4 Weight-Only 量化]
    A --> C[校准数据收集<br/>activation_stats / moe_expert_stats]
    B --> D[merge_hy3_nvfp4_c8.py]
    C --> D
    D --> E[完整推理模型<br/>NVFP4 Expert + FP8 KV + input_scale]
    E --> F[vLLM 部署推理]
```

**步骤一**：运行 NVFP4 Weight-Only 量化，生成 Expert 层的量化权重。

**步骤二**：运行 `scripts/ptq/run_vllm_calibrate_for_Hy3.sh` 校准脚本，通过 vLLM 推理收集激活统计信息（KV cache 的 min/max、Expert 激活的 min/max）。

**步骤三**：使用合并工具将量化权重与校准统计合并，生成最终可部署的推理模型。
