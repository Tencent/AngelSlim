(focus-fp4)=

# FOCUS：面向 FP4 的端到端 Scale 优化

FOCUS（**FP4 Optimization via Coupled-Relaxation and Dual-Granularity
Scaling**）是一种面向 MXFP4 和 NVFP4 W4A4 的后训练量化方法。它冻结原始
模型权重，仅通过端到端优化量化 scale 来恢复 FP4 精度，部署时仍输出标准
硬件格式，不引入额外推理开销。

:::{tip}
论文：[FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity
Scaling](https://arxiv.org/abs/2608.01847)
:::

当前实现提供：

- Qwen3-4B 的 MXFP4 W4A4 与 NVFP4 W4A4 配置；
- Coupled-Relaxation Scaling（CRS）与 Dual-Granularity Scaling（DGS）；
- DeepSpeed ZeRO-3 多卡训练；
- fake-quant checkpoint、compressed-tensors packed checkpoint 与校验工具；
- 可由 vLLM 加载的标准部署产物。

## 方法概述

标准 FP4 量化通常让量化 scale 与反量化 scale 完全相同：

```{math}
\bar{\mathbf{W}}_i =
\mathcal{Q}_{\mathrm{E2M1}}\left(\mathbf{W}_i / S_i\right),
\qquad
\hat{\mathbf{W}}_i = \bar{\mathbf{W}}_i \cdot S_i .
```

但量化 scale 只在离线生成 FP4 code 时使用，并不会保存到部署模型中；真正受
硬件格式约束的只有反量化 scale。FOCUS 利用这一差异，从精度和粒度两个维度
扩大 scale 的优化空间。

### Coupled-Relaxation Scaling（CRS）

CRS 为每个 block 引入可学习的全精度系数 \(c_i\)，放松量化与反量化 scale
之间的耦合：

```{math}
S_i^{q}=S_i^{dq}\cdot c_i,
\qquad
\bar{\mathbf{W}}_i =
\mathcal{Q}_{\mathrm{E2M1}}
\left(\mathbf{W}_i / S_i^{q}\right),
\qquad
\hat{\mathbf{W}}_i = \bar{\mathbf{W}}_i \cdot S_i^{dq}.
```

其中 \(S_i^{dq}\) 始终满足 E8M0 或 E4M3 等硬件约束，而 \(c_i\) 只参与离线
优化并在导出时丢弃。

### Dual-Granularity Scaling（DGS）

DGS 进一步把每个硬件 block 划分为多个 8 元素 sub-block，并为每个
sub-block 分配独立系数 \(c_i^k\)。反量化 scale 仍保持原硬件粒度，因此不会
改变部署格式：

```{math}
\bar{\mathbf{W}}_i^k =
\mathcal{Q}_{\mathrm{E2M1}}
\left(\mathbf{W}_i^k / (S_i^{dq}\cdot c_i^k)\right),
\qquad
\hat{\mathbf{W}}_i^k = \bar{\mathbf{W}}_i^k \cdot S_i^{dq}.
```

| 格式 | 硬件 block | Scale 格式 | Sub-block | `num_sub` |
|------|------------|------------|-----------|-----------|
| MXFP4 | 32 | E8M0 | 8 | 4 |
| NVFP4 | 16 | FP8 E4M3 + FP32 global scale | 8 | 2 |

训练完成后，仅 FP4 code 与硬件兼容的反量化 scale 被保留；CRS/DGS 系数不会
进入最终 checkpoint。

## 工作流程

```text
BF16 基础模型 + WikiText2 校准数据
                │
                ▼
         FOCUS scale 优化
          ┌─────┴─────┐
          ▼           ▼
   fake checkpoint   direct-real packed checkpoint
          │           │
          ├─ validator│
          ▼           │
     offline pack ────┘
                │
                ▼
       packed validator → vLLM 部署
```

## 环境准备

请先按照 [AngelSlim 安装文档](../../getting_started/installation.md) 从源码安装
当前版本，并额外安装 DeepSpeed：

```shell
pip install -e .
pip install deepspeed
```

默认配置会从 Hugging Face 加载 `Qwen/Qwen3-4B` 和
`Salesforce/wikitext`。运行环境需要能够访问 Hugging Face，或者已经在本地
缓存模型和数据集；也可以通过 `MODEL_PATH` 指定本地模型目录。

从仓库根目录运行以下命令。示例默认使用 2 张 GPU，`NPROC` 应与
`CUDA_VISIBLE_DEVICES` 中的 GPU 数量一致。

## 快速开始

### MXFP4

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=mxfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh
```

### NVFP4

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=nvfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh
```

脚本默认生成 fake checkpoint。可通过以下环境变量调整运行参数：

- `MODEL_PATH`：基础模型路径，默认使用 `Qwen/Qwen3-4B`。
- `SAVE_PATH`：输出路径；未设置时使用配置文件中的 `global.save_path`。
- `NPROC`：训练进程数，默认值为 `2`。
- `FORMAT`：量化格式，可选 `mxfp4` 或 `nvfp4`。
- `SAVE_FORMAT`：保存格式，可选 `fake` 或 `real`，默认值为 `fake`。
- `CONFIG`：自定义 YAML 路径；未设置时由 `FORMAT` 与 `SAVE_FORMAT`
  自动选择仓库内配置。

### 调整 GPU 数量

正式配置使用 global batch size 32：

```text
global batch size = per-device batch size × gradient accumulation steps × NPROC
```

保持配置中的 per-device batch size 不变时，建议按以下方式设置：

- MXFP4（per-device batch size 1）：2/4/8 张 GPU 分别使用
  gradient accumulation steps 16/8/4。
- NVFP4（per-device batch size 2）：2/4/8 张 GPU 分别使用
  gradient accumulation steps 8/4/2。

只修改 `NPROC` 会改变 global batch size。使用非 2-GPU 配置时，请复制对应
YAML、修改 `compression.QAT.hf_args.gradient_accumulation_steps`，再通过
`CONFIG=/path/to/config.yaml` 启动。

## 关键训练配置

仓库配置与论文的 Qwen3-4B 设置保持一致：

| 配置 | MXFP4 | NVFP4 |
|------|-------|-------|
| Weight / activation | W4A4 | W4A4 |
| Block size | 32 | 16 |
| DGS sub-block | 4 × 8 | 2 × 8 |
| Scale learning rate | `2e-2` | `5e-3` |
| Relaxation coefficient learning rate | `5e-2` | `1e-3` |
| Loss | KL-Top，`k=1000` | KL-Top，`k=1000` |
| Epoch / global batch size | 1 / 32 | 1 / 32 |
| Sequence length | 2048 | 2048 |

训练期间原始权重保持冻结，仅 `max_scale` 与 DGS relaxation coefficient
参与优化。激活量化保持动态，不学习 activation scale。

## 输出与部署

默认的 `SAVE_FORMAT=fake` 会保存包含量化后权重和已优化 scale 的 fake
checkpoint。

如需直接生成 packed compressed-tensors checkpoint，可设置
`SAVE_FORMAT=real`：

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=mxfp4 \
SAVE_FORMAT=real \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
SAVE_PATH=./output/focus-mxfp4-real \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh
```

将 `FORMAT` 改为 `nvfp4` 即可导出 NVFP4。`SAVE_PATH` 和
`global.save_path` 表示输出根目录，运行时还会追加配置文件名（不含
`.yaml`）：

- fake checkpoint：
  `<output_root>/<config_stem>_fake_quant_model.pt`
- real checkpoint：
  `<output_root>/<config_stem>/final_quant_checkpoint/`

例如，上面的 real 命令会保存到
`./output/focus-mxfp4-real/qwen3-4b_focus_mxfp4_w4a4_real_zero3/final_quant_checkpoint/`。

### 从 fake checkpoint 离线导出

如果训练阶段保存的是 fake checkpoint，可以结合冻结的 BF16 基础模型离线生成
packed checkpoint：

```shell
# MXFP4
python tools/focus_fp4/export_focus_mxfp4.py \
  --checkpoint /path/to/mxfp4_fake_quant_model.pt \
  --model-path /path/to/Qwen3-4B \
  --output-path ./output/focus-mxfp4-packed

# NVFP4
python tools/focus_fp4/export_focus_nvfp4.py \
  --checkpoint /path/to/nvfp4_fake_quant_model.pt \
  --model-path /path/to/Qwen3-4B \
  --output-path ./output/focus-nvfp4-packed
```

:::{note}
Fake checkpoint 中的权重已经完成 fake quant。离线导出必须使用冻结的 BF16
基础权重与 checkpoint 中学习到的 scale 重新打包，不能直接对 fake 权重进行
第二次量化。
:::

对于 NVFP4，导出器会在保持 FP4 code 不变的前提下，将 QKV 与 gate/up
各分支的 global-scale 比例折入 FP8 block scale，并为融合 GEMM 写入共享的
global scale，以满足 compressed-tensors 与 vLLM 的部署契约。

## Checkpoint 校验

验证 MXFP4 fake checkpoint 中的权重和 subgroup scale：

```shell
python tools/focus_fp4/validate_focus_fp4_checkpoint.py \
  --checkpoint /path/to/mxfp4_fake_quant_model.pt \
  --model-path /path/to/Qwen3-4B \
  --qtype mxfp4 \
  --group-size 32 \
  --num-sub 4
```

验证 NVFP4 fake checkpoint：

```shell
python tools/focus_fp4/validate_focus_fp4_checkpoint.py \
  --checkpoint /path/to/nvfp4_fake_quant_model.pt \
  --model-path /path/to/Qwen3-4B \
  --qtype nvfp4 \
  --group-size 16 \
  --num-sub 2
```

direct-real checkpoint 没有独立的 fake checkpoint，可执行结构校验：

```shell
python tools/focus_fp4/validate_focus_mxfp4_export.py \
  --export-path /path/to/final_quant_checkpoint \
  --model-path /path/to/Qwen3-4B
```

NVFP4 使用 `tools/focus_fp4/validate_focus_nvfp4_export.py`。对于由 fake
checkpoint 离线导出的 packed checkpoint，可额外传入 `--checkpoint`，验证
FP4 code、block scale 与 global scale：

```shell
python tools/focus_fp4/validate_focus_nvfp4_export.py \
  --export-path ./output/focus-nvfp4-packed \
  --checkpoint /path/to/nvfp4_fake_quant_model.pt \
  --model-path /path/to/Qwen3-4B
```

## 评测

ZeRO-3 多进程命令仅用于训练和保存。请勿向上述 `torchrun` 启动流程追加
`--ppl-eval` 或 `--lm-eval`；保存完成后，请在独立的单 GPU 进程中加载
checkpoint 并运行 PPL 或 lm-evaluation-harness 评测。

论文使用以下协议：

- WikiText2 与 C4 perplexity，sequence length 为 2048；
- ARC-Challenge、ARC-Easy、HellaSwag、PIQA 与 WinoGrande 五项
  zero-shot accuracy；
- `Avg.` 为五项 zero-shot accuracy 的平均值。

## 主要结果

| 格式 | 方法 | WikiText2 ↓ | C4 ↓ | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg. ↑ |
|------|------|------------:|-----:|------:|------:|----------:|-----:|-----------:|-------:|
| FP16 | FP16 | 13.66 | 16.63 | 54.18 | 78.07 | 68.50 | 74.81 | 65.67 | 68.25 |
| MXFP4 | RTN | 18.60 | 20.62 | 46.08 | 70.41 | 62.91 | 71.60 | 60.77 | 62.35 |
| MXFP4 | GPTQ | 16.85 | 19.37 | 47.95 | 72.81 | 63.09 | 72.91 | 61.48 | 63.65 |
| MXFP4 | MR-GPTQ | 15.48 | 18.38 | 48.89 | 74.37 | 63.39 | 73.23 | 64.09 | 64.79 |
| MXFP4 | **FOCUS** | **12.85** | **17.77** | **49.15** | **74.83** | **66.42** | **73.61** | **64.25** | **65.65** |
| NVFP4 | RTN | 13.88 | 17.29 | 51.54 | 74.33 | 65.72 | 74.27 | 62.04 | 65.58 |
| NVFP4 | GPTQ | 13.91 | 17.30 | 51.28 | 75.46 | 66.59 | 74.59 | 62.83 | 66.15 |
| NVFP4 | MR-GPTQ | 14.64 | 17.60 | 49.57 | 75.34 | 65.86 | 72.91 | 63.61 | 65.46 |
| NVFP4 | 4over6 | 14.17 | 17.28 | 48.63 | 74.58 | 66.39 | 71.93 | 62.19 | 64.74 |
| NVFP4 | RaZeR | 14.11 | 17.26 | 50.34 | 74.75 | 67.10 | 73.29 | 63.06 | 65.71 |
| NVFP4 | **FOCUS** | **12.57** | **16.97** | **52.56** | **76.01** | **67.26** | **75.03** | **64.40** | **67.05** |

在 Qwen3-4B 上，NVFP4 与 MXFP4 FOCUS 分别恢复 FP16 平均 zero-shot
accuracy 的 **98.2%** 与 **96.2%**。

## 配置文件

仓库提供以下 Qwen3-4B 配置：

| 格式 | 保存类型 | 配置文件 |
|------|----------|----------|
| MXFP4 | fake | `configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_zero3.yaml` |
| MXFP4 | real | `configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_real_zero3.yaml` |
| NVFP4 | fake | `configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_zero3.yaml` |
| NVFP4 | real | `configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_real_zero3.yaml` |

如需调整训练数据、batch size、训练轮数或输出路径，请复制对应配置后修改，
并通过 `CONFIG=/path/to/config.yaml` 传给启动脚本。
