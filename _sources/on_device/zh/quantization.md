# 端侧量化（PTQ）

[← 返回总览](./index.md) ｜ [低比特训练 →](./low_bit.md)

本页介绍用于端侧模型的训练后量化（PTQ，Post-Training Quantization）算法。PTQ 不需要训练，只需要少量校准数据即可完成量化，是端侧压缩的第一步。

## 目录

- [1. 算法总览](#1-算法总览)
- [2. 通用配置结构](#2-通用配置结构)
- [3. INT8：动态量化与 SmoothQuant](#3-int8动态量化与-smoothquant)
- [4. FP8：动态、静态与 LeptoQuant](#4-fp8动态静态与-leptoquant)
- [5. INT4：GPTQ、AWQ 与 GPTAQ](#5-int4gptqawq-与-gptaq)
- [6. NVFP4 Weight-Only](#6-nvfp4-weight-only)
- [7. KV Cache 量化](#7-kv-cache-量化)
- [8. DAQ：面向后训练增量的量化](#8-daq面向后训练增量的量化)
- [9. 端侧多模态模型（Qwen3-VL）](#9-端侧多模态模型qwen3-vl)
- [10. 产出格式与 config.json](#10-产出格式与-configjson)

---

## 1. 算法总览

| 算法（`quantization.name`） | 精度组合 | 是否需要校准数据 | 典型端侧场景 |
| --- | --- | --- | --- |
| `int8_dynamic` | W8A8：权重 per-channel 离线，激活 per-token 动态 | 否（权重）/ 是（若做静态校准） | 通用 INT8 推理引擎 |
| `int8_dynamic` + `quant_helpers: [smooth]` | SmoothQuant 平滑后的 W8A8 | 是 | 激活离群值较大的模型 |
| `fp8_dynamic` | W8A8-FP8：per-tensor，激活在线动态 | 否 | 支持 FP8 的 GPU / NPU |
| `fp8_static` | W8A8-FP8：per-tensor，激活离线校准 | 是 | 同上，追求更低的激活开销 |
| `fp8_lepto` | FP8 静态，LeptoQuant 搜索 scale | 是 | 数学、严格格式任务上 FP8 精度损失较大时 |
| `int4_gptq` | W4A16：per-group，GPTQ 逐层重构 | 是 | 端侧 INT4 权重，通用性最好 |
| `int4_awq` | W4A16：per-group，AWQ 保护重要通道 | 是 | 端侧 INT4 权重，无需训练 |
| `int4_gptaq` | W4A16：GPTQ 的激活感知变体 | 是 | INT4 精度要求更高的场景 |
| `nvfp4` / `nvfp4_weight_only` | NVFP4（E2M1 + FP8 E4M3 scale，block 16） | 否（Weight-Only） | NVFP4 硬件加速平台 |

> 更多算法（如 MXFP4、W4A8-FP8、QAT、QAD）见 [低比特训练](./low_bit.md)。

---

## 2. 通用配置结构

所有 PTQ 配置都遵循同样的结构。以下是一个 INT4 GPTQ 的完整配置，并附字段说明：

```yaml
global:
  save_path: ./output                 # 产出目录

model:
  name: Qwen                          # 模型类型，与 configs 目录下的模型一致
  model_path: Qwen/Qwen3-0.6B         # Hugging Face 模型名或本地路径
  trust_remote_code: true
  low_cpu_mem_usage: true
  use_cache: false
  torch_dtype: auto
  device_map: cpu                     # 量化时使用的设备，cpu 或 auto/cuda

compression:
  name: PTQ                           # 压缩类型
  quantization:
    name: int4_gptq                   # 量化算法
    bits: 4                           # 量化位宽
    quant_method:
      weight: "per-group"             # 权重量化粒度
      group_size: 128                 # 分组大小
    ignore_layers:                    # 不量化的层
      - "lm_head"

dataset:                              # 校准数据（需要校准的算法才需要）
  name: TextDataset
  data_path: ./dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
  max_seq_length: 4096                # 校准序列最大长度
  num_samples: 128                    # 校准样本数
  batch_size: 1
```

| 字段 | 说明 |
| --- | --- |
| `global.save_path` | 产出模型保存目录，可用 `--save-path` 覆盖 |
| `model.model_path` | 源模型，可用 `--model-path` 覆盖 |
| `compression.quantization.name` | 量化算法，见上表 |
| `quantization.bits` | 位宽，INT4 填 `4`，INT8 / FP8 填 `8` |
| `quantization.quant_method` | 权重与激活的量化粒度 |
| `quantization.ignore_layers` | 跳过量化的层，`lm_head` 与 `embed_tokens` 通常保持高精度 |
| `quantization.quant_helpers` | 额外的预处理，例如 `smooth` |
| `dataset.*` | 校准数据，格式见[数据准备文档](../../design/prepare_dataset.md) |

运行方式：

```shell
python3 tools/run.py -c configs/qwen3/ptq/int4_gptq/qwen3-0_6b_int4_gptq.yaml
```

如需指向本地模型并指定产出目录：

```shell
python3 tools/run.py \
    -c configs/qwen3/ptq/int4_gptq/qwen3-0_6b_int4_gptq.yaml \
    --model-path /path/to/Qwen3-0.6B \
    --save-path ./output_qwen3_0_6b_int4_gptq
```

---

## 3. INT8：动态量化与 SmoothQuant

### 3.1 W8A8-INT8 动态量化

权重离线做 per-channel 量化，激活在推理时按 token 动态量化。

```shell
python3 tools/run.py -c configs/qwen3/ptq/int8_dynamic/qwen3-0_6b_int8_dynamic.yaml
```

关键配置：

```yaml
compression:
  name: PTQ
  quantization:
    name: int8_dynamic
    bits: 8
    quant_method:
      weight: "per-channel"
      activation: "per-token"
    ignore_layers:
      - "lm_head"
```

其它可选规模：`configs/qwen3/ptq/int8_dynamic/qwen3-1_7b_int8_dynamic.yaml`、`configs/qwen3/ptq/int8_dynamic/qwen3-4b_int8_dynamic.yaml`。

### 3.2 SmoothQuant + INT8

SmoothQuant 通过把激活中的离群值迁移到权重上来平滑分布，再做 INT8 量化。在 `quant_helpers` 中加入 `smooth` 即可启用：

```shell
python3 tools/run.py -c configs/qwen3/ptq/smooth_int8/qwen3-0_6b_int8_dynamic_smooth.yaml
```

```yaml
compression:
  name: PTQ
  quantization:
    name: int8_dynamic
    bits: 8
    quant_method:
      weight: "per-channel"
      activation: "per-token"
    quant_helpers:
      - "smooth"
    ignore_layers:
      - "lm_head"
```

SmoothQuant 的实现位于 `angelslim/compressor/quant/modules/smooth/`，更详细的说明见 [SmoothQuant 文档](../../features/quantization/smooth.md)。

---

## 4. FP8：动态、静态与 LeptoQuant

### 4.1 W8A8-FP8 动态量化

权重和激活都采用 per-tensor 粒度，权重离线量化，激活在线动态量化。

```shell
python3 tools/run.py -c configs/qwen3/ptq/fp8_dynamic/qwen3-0_6b_fp8_dynamic.yaml
```

```yaml
compression:
  name: PTQ
  quantization:
    name: fp8_dynamic
    bits: 8
    quant_method:
      weight: "per-tensor"
      activation: "per-tensor"
    ignore_layers:
      - "lm_head"
```

### 4.2 W8A8-FP8 静态量化

激活的 scale 通过校准数据离线确定，需要配置数据集：

```shell
python3 tools/run.py -c configs/qwen3/ptq/fp8_static/qwen3-0_6b_fp8_static.yaml
```

小模型示例（Hunyuan 0.5B）：

```shell
python3 tools/run.py -c configs/hunyuan/fp8_static/hunyuan_0_5b_dense_fp8_static.yaml
```

### 4.3 LeptoQuant（FP8 静态的搜索式 scale）

当普通 FP8 量化后模型在数学或严格格式任务上明显掉点时，可以尝试 LeptoQuant。它隔离权重与激活中的离群值，把数值映射集中到 FP8 的高精度区间。

```shell
python3 tools/run.py -c configs/qwen3/ptq/fp8_static/qwen3-0_6b_lepto_fp8_static.yaml
```

```yaml
compression:
  name: PTQ
  quantization:
    name: fp8_lepto
    bits: 8
    quant_method:
      weight: "per-tensor"
      activation: "per-tensor"
    ignore_layers:
      - "lm_head"
```

Hunyuan 4B 的对应配置为 `configs/hunyuan/fp8_static/hunyuan_4b_dense_lepto_fp8_static.yaml`。详细原理见 [FP8_LEPTO 文档](../../features/quantization/fp8_lepto.md)。

### 4.4 FP8 低显存模式

对于较大的模型，标准 FP8 静态量化的显存占用较高，可以开启 `low_memory` 模式。该模式在 `quantization` 中设置 `low_memory: True`，其余配置与 FP8 静态量化相同。详见 [FP8 文档](../../features/quantization/fp8.md)。

---

## 5. INT4：GPTQ、AWQ 与 GPTAQ

INT4 量化只量化权重（W4A16），是端侧 LLM 最常用的压缩方式。三种算法的区别如下：

| 算法 | 思路 | 适用场景 |
| --- | --- | --- |
| GPTQ | 逐层利用 Hessian 信息最小化重构误差 | 通用，精度与速度平衡 |
| AWQ | 统计激活幅值，为重要通道计算缩放系数，无需训练 | 快速量化，兼顾精度 |
| GPTAQ | 在 GPTQ 基础上引入激活感知的优化目标 | 对 INT4 精度要求更高的场景 |

### 5.1 INT4-GPTQ

```shell
python3 tools/run.py -c configs/qwen3/ptq/int4_gptq/qwen3-0_6b_int4_gptq.yaml
```

```yaml
compression:
  name: PTQ
  quantization:
    name: int4_gptq
    bits: 4
    quant_method:
      weight: "per-group"
      group_size: 128
    ignore_layers:
      - "lm_head"
```

Hunyuan 系列端侧规模的示例：

```shell
python3 tools/run.py -c configs/hunyuan/int4_gptq/hunyuan_0_5b_dense_int4_gptq.yaml
python3 tools/run.py -c configs/hunyuan/int4_gptq/hunyuan_1_8b_dense_int4_gptq.yaml
python3 tools/run.py -c configs/hunyuan/int4_gptq/hunyuan_4b_dense_int4_gptq.yaml
```

Qwen2.5 端侧规模的示例：`configs/qwen2_5/int4_gptq/qwen2_5-1_5b_int4_gptq.yaml`。

### 5.2 INT4-AWQ

AWQ 常用配置项：`zero_point: true`（非对称量化，推荐开启）与 `mse_range: false`（自动裁剪，推荐关闭）。

```shell
python3 tools/run.py -c configs/qwen3/ptq/int4_awq/qwen3-0_6b_int4_awq.yaml
```

```yaml
compression:
  name: PTQ
  quantization:
    name: int4_awq
    bits: 4
    quant_method:
      weight: "per-group"
      group_size: 128
      zero_point: true
      mse_range: false
    ignore_layers:
      - "lm_head"
```

内存受限时可使用低显存版本，例如 `configs/qwen3/ptq/int4_awq/qwen3-1_7b_int4_awq_low_memory.yaml`。

Hunyuan 端侧规模示例：`configs/hunyuan/int4_awq/hunyuan_0_5b_dense_int4_awq.yaml`、`configs/hunyuan/int4_awq/hunyuan_4b_dense_int4_awq.yaml`。

### 5.3 INT4-GPTAQ

```shell
python3 tools/run.py -c configs/qwen3/ptq/int4_gptaq/qwen3-4b_int4_gptaq.yaml
```

### 5.4 进一步提升 INT4 精度（可选）

如果 PTQ 后的 INT4 模型精度仍不满足要求，可以进入[低比特训练](./low_bit.md)中的 INT4 QAT 流程，通过训练让模型适应量化误差。

---

## 6. NVFP4 Weight-Only

NVFP4 使用 E2M1 的 4 位浮点元素，每 16 个元素共享一个 FP8 E4M3 的 scale，并附加一个 per-tensor 的 FP32 全局 scale。Weight-Only 模式不需要校准数据，缩放因子直接由权重的 absmax 计算得到。

```shell
python3 tools/run.py -c configs/qwen3/ptq/nvfp4/qwen3-0_6b_nvfp4.yaml
```

```yaml
compression:
  name: PTQ
  quantization:
    name: nvfp4
    bits: 4
    quant_method:
      weight: "per-block"
      activation: "per-block"
      group_size: 16
    ignore_layers:
      - "lm_head"
```

NVFP4 的 Weight-Only 与 MoE 专家层的专用配置见 [NVFP4 文档](../../features/quantization/nvfp4.md)，可参考 `configs/glm5/ptq/mixed/glm5_nvfp4_rtn.yaml`。

---

## 7. KV Cache 量化

KV Cache 量化可以减少长上下文下的内存占用。AngelSlim 将校准与 scale 替换分为两个独立入口，vLLM 无需打补丁。

### 7.1 KV-only 校准

只挂载 Attention 的 KV hook 进行 vLLM 校准，不统计 Linear / MoE 层：

```shell
python3 -m angelslim.compressor.quant.vllm_calib.calibrate --kv-only \
    -c <kv_calibrate_yaml>
```

YAML 中的键名与命令行参数的 `dest` 一致，例如 `kv_granularity: per-head` 与 `search_kv_scale: true`。配置示例可参考 `configs/Hy3/ptq/fp8/Hy3_kvcache_calibrate.yaml`，将其中的模型路径替换为你的模型即可。

校准产出包括 `activation_stats*.json`，以及在开启 `search_kv_scale` 时生成的 `kv_cache_tuned_scales*.json`。

### 7.2 替换已量化模型的 KV scale

用校准结果改写模型目录中的 `kv_cache_scales.safetensors`，并更新 `config.json` 中的 `attn_quant_config`：

```shell
# per-head（默认）
python3 -m angelslim.compressor.quant.ckpt_convert.kv_scales \
    --granularity per-head \
    --json   /path/to/kv_cache_tuned_scales_per_head.json \
    --src    /path/to/model_dir/kv_cache_scales.safetensors \
    --output /path/to/output_dir/kv_cache_scales.safetensors

# per-tensor
python3 -m angelslim.compressor.quant.ckpt_convert.kv_scales \
    --granularity per-tensor \
    --json   /path/to/kv_cache_tuned_scales.json \
    --src    /path/to/model_dir/kv_cache_scales.safetensors \
    --output /path/to/output_dir/kv_cache_scales.safetensors
```

省略 `--output` 时会原地覆盖 `--src` 并自动生成 `.bak` 备份。完整说明见 [KV Cache 文档](../../features/quantization/kvcache.md)。

---

## 8. DAQ：面向后训练增量的量化

如果你的模型是在基座模型上做了 SFT / RLHF / DPO 等后训练，标准的重建误差优化会破坏后训练带来的小幅权重增量（ΔW）。DAQ（Delta-Aware Quantization）只需要**基座模型与后训练模型的权重**，不需要校准数据，以符号保持率和余弦相似度来优化 ΔW 的方向保真度。

```shell
python3 tools/run.py -c configs/deepseek_r1/fp8_daq/deepseek_r1_daq_fp8_w8a8_block.yaml
```

该示例为大模型配置，小模型可参照其结构修改 `model.model_path`。DAQ 的分析工具为 `tools/daq_analyze.py`，方法与参数详见 [DAQ 文档](../../features/quantization/daq.md)。

---

## 9. 端侧多模态模型（Qwen3-VL）

视觉语言模型在端侧的压缩同样使用 PTQ。Qwen3-VL 目前提供 FP8 静态与 INT8 动态两类配置，并可通过 `quant_vit` 控制是否量化视觉编码器。

```shell
# Qwen3-VL-2B：FP8 静态
python3 tools/run.py -c configs/qwen3_vl/fp8_static/qwen3_vl-2b_fp8_static.yaml

# Qwen3-VL-4B：FP8 静态
python3 tools/run.py -c configs/qwen3_vl/fp8_static/qwen3_vl-4b_fp8_static.yaml

# Qwen3-VL-2B：INT8 动态
python3 tools/run.py -c configs/qwen3_vl/int8_dynamic/qwen3_vl-2b_int8_dynamic.yaml
```

关键配置：

```yaml
model:
  name: Qwen3VL
  model_path: Qwen/Qwen3-VL-2B-Instruct
compression:
  name: PTQ
  quantization:
    name: fp8_static
    bits: 8
    quant_method:
      weight: "per-tensor"
      activation: "per-tensor"
    ignore_layers:
      - "model.visual.patch_embed.proj"
      - "model.lm_head"
      - "model.language_model.embed_tokens"
    quant_vit: false                  # 是否量化视觉编码器
dataset:
  name: MultiModalDataset
  data_path: HuggingFaceM4/ChartQA
  max_seq_length: 4096
  num_samples: 256
  batch_size: 1
```

---

## 10. 产出格式与 config.json

量化产出模型会在保存目录中生成权重与 `config.json`。部署引擎需要依赖 `config.json` 中的 `quantization_config` 正确加载量化模型，例如 INT8 动态量化的记录如下：

```json
"quantization_config": {
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "input_activations": {
        "dynamic": true,
        "num_bits": 8,
        "strategy": "token",
        "type": "int"
      },
      "output_activations": null,
      "weights": {
        "dynamic": false,
        "num_bits": 8,
        "strategy": "channel",
        "type": "int"
      }
    }
  },
  "format": "int-quantized",
  "ignore": ["lm_head"],
  "kv_cache_scheme": null,
  "quant_method": "compressed-tensors",
  "quantization_status": "compressed"
}
```

- 每个被量化的线性层保存 `weight`（整数或 FP8）与 `weight_scale`，激活 scale 在静态量化时额外保存 `input_scale`。
- 格式字段的兼容性以所用推理引擎的文档为准：vLLM 的 [INT8](https://docs.vllm.ai/en/stable/features/quantization/int8.html)、[FP8](https://docs.vllm.ai/en/stable/features/quantization/fp8.html) 页面给出了加载要求。
- 如果需要通过 Transformers 直接加载，参见 [部署总览的 Transformers 校验](./deployment/index.md)。

下一步：[低比特训练 →](./low_bit.md)，或直接前往 [端侧部署 →](./deployment/index.md)。
