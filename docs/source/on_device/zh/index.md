# 端侧部署教程（总览）

本教程介绍如何使用 AngelSlim 对面向端侧（手机、PC、嵌入式 CPU/NPU 等设备）部署的大模型进行压缩，并将压缩后的模型部署到端侧推理引擎中。内容涵盖 AngelSlim 中与端侧相关的模型、量化与低比特训练算法、以及部署方式，每个步骤均给出可直接执行的命令与对应的配置文件路径。

> 本教程只介绍**方法、命令与配置**，不包含任何性能或精度结果。如需评测数据，请参考各算法页面中的链接与对应技术报告。

## 目录

- [1. 端侧压缩流程概览](#1-端侧压缩流程概览)
- [2. 支持的端侧相关模型](#2-支持的端侧相关模型)
- [3. 环境准备](#3-环境准备)
- [4. 教程导航](#4-教程导航)
- [5. 端侧相关的论文与资源](#5-端侧相关的论文与资源)
- [6. 范围说明](#6-范围说明)

---

## 1. 端侧压缩流程概览

端侧部署通常同时受限于显存/内存、存储空间、带宽与功耗，因此需要把模型权重压缩到 INT8、INT4 甚至 1~2 bit 的水平。AngelSlim 提供的端到端流程如下：

```text
源模型（BF16 / FP8 Hugging Face 权重）
   │
   ├─ PTQ 训练后量化：FP8 / INT8 / INT4(GPTQ、AWQ) / NVFP4 / MXFP4 ...
   │        └─ 无需训练，只需少量校准数据
   │
   ├─ QAT / QAD 量化感知训练：INT4 / W4A8-FP8 / 1.25-bit(Sherry) / 2-bit(SEQ) ...
   │        └─ 需要训练数据与训练资源，精度更高
   │
   ▼
压缩后的模型（compressed-tensors / fake / real 格式）
   │
   ├─ Hugging Face Transformers（调试、校验）
   ├─ vLLM / SGLang（GPU 服务端 / 工作站）
   └─ llama.cpp GGUF（PC、手机、CPU 端侧推理）
```

选择建议：

| 目标 | 推荐路径 | 对应页面 |
| --- | --- | --- |
| 快速得到端侧可用的小模型，不想训练 | PTQ：INT8 / FP8 / INT4-GPTQ / INT4-AWQ | [量化](./quantization.md) |
| 在较低比特下尽量保持精度 | QAT / QAD：INT4、W4A8-FP8 | [低比特训练](./low_bit.md) |
| 极致压缩（1.25 bit / 2 bit） | Sherry、SEQ 的 QAD 训练 + GGUF/STQ 推理 | [低比特训练](./low_bit.md)、[llama.cpp 部署](./deployment/llamacpp.md) |
| 在手机或 PC 上直接跑 GGUF 模型 | llama.cpp 推理 | [llama.cpp 部署](./deployment/llamacpp.md) |
| Hy4-flash 本地高性能推理 | Strata | [Strata 部署](./deployment/strata.md) |
| 超大模型的多机本地运行 | Hy4 preview GGUF + prima.cpp | [多机部署](./deployment/prima_multi_machine.md) |

---

## 2. 支持的端侧相关模型

下表列出 AngelSlim 中与端侧直接相关的模型、可用压缩方式与对应的配置目录。配置文件中的 `model_path` 默认指向 Hugging Face 上的基础模型，可替换为本地路径。

### 2.1 小参数量语言模型（适合手机与边缘设备）

| 模型 | 参数规模 | 可用压缩方式 | 配置目录 | Hugging Face 模型 |
| --- | --- | --- | --- | --- |
| Hunyuan Dense | 0.5B / 1.8B / 4B | FP8-Static、LeptoQuant、INT4-GPTQ、INT4-AWQ | `configs/hunyuan/` | [tencent/Hunyuan-0.5B-Instruct](https://huggingface.co/tencent/Hunyuan-0.5B-Instruct)、[tencent/Hunyuan-1.8B-Instruct](https://huggingface.co/tencent/Hunyuan-1.8B-Instruct)、[tencent/Hunyuan-4B-Instruct](https://huggingface.co/tencent/Hunyuan-4B-Instruct)（[合集](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)） |
| Qwen3 | 0.6B / 1.7B / 4B | FP8-Static、FP8-Dynamic、LeptoQuant、INT8-Dynamic、SmoothQuant-INT8、INT4-GPTQ、INT4-AWQ、INT4-GPTAQ、NVFP4、QAT（INT4、W4A8-FP8）、QAD（Sherry、SEQ、LSQ、TWN、AbsMean、DLT、W4A8-FP8） | `configs/qwen3/` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)、[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen2.5-Instruct | 0.5B / 1.5B / 3B | FP8-Static、FP8-Dynamic、INT4-GPTQ、INT4-AWQ | `configs/qwen2_5/` | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)、[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |

### 2.2 端侧多模态模型（视觉语言）

| 模型 | 参数规模 | 可用压缩方式 | 配置目录 | Hugging Face 模型 |
| --- | --- | --- | --- | --- |
| Qwen3-VL-Instruct | 2B / 4B | FP8-Static、INT8-Dynamic | `configs/qwen3_vl/` | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)（[合集](https://huggingface.co/collections/Qwen/qwen3-vl)） |
| Qwen2.5-VL | 视觉 Token 压缩（IDPruner 等） | IDPruner 等视觉 Token 剪枝 | `configs/qwen2_5_vl/pruning/` | 见 [Token 压缩文档](../../features/token_compressor/index.md) |

### 2.3 Hy-MT 模型与其它端侧模型（已发布端侧权重）

| 模型 | 压缩方式 | 端侧格式 | 相关链接 |
| --- | --- | --- | --- |
| HY-MT2-1.8B-1.25Bit | 1.25-bit 低比特量化 | GGUF（STQ） | [GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) |
| HY-MT2-1.8B-2Bit | 2-bit 低比特量化 | GGUF（STQ） | [GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF) |
| HY-MT2-1.8B / 7B | 标准量化（Q4_K_M、Q6_K、Q8_0）与 FP8 | GGUF、FP8 | [1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)、[7B-GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF)、[1.8B-FP8](https://huggingface.co/tencent/Hy-MT2-1.8B-FP8)、[7B-FP8](https://huggingface.co/tencent/Hy-MT2-7B-FP8) |
| HY-MT2-30B-A3B | FP8 与 GGUF | FP8、GGUF | [30B-A3B-GGUF](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF)、[30B-A3B-FP8](https://huggingface.co/tencent/Hy-MT2-30B-A3B-FP8) |
| Hy-MT1.5-1.8B-1.25bit | Sherry 1.25-bit 三值量化（3:4 细粒度稀疏） | 权重 + GGUF（STQ1_0） | [权重](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit)、[GGUF](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF) |
| Hy-MT1.5-1.8B-2bit | SEQ（Stretched Elastic Quantization）2-bit + 量化感知蒸馏 | 权重 + GGUF | [权重](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-2bit)、[GGUF](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-2bit-GGUF) |
| Hy-MT3 | 云端版本与端侧 GGUF 量化版本 | GGUF | [合集](https://huggingface.co/collections/tencent/hy-mt3) |

以下为非翻译类的端侧模型：

| 模型 | 压缩方式 | 端侧格式 | 相关链接 |
| --- | --- | --- | --- |
| HY-1.8B-2Bit | 2-bit 端侧大语言模型 | 权重 | [权重](https://huggingface.co/AngelSlim/HY-1.8B-2Bit) |

Hy-MT1.5 的完整说明见 [Hy-MT1.5 量化文档](../../models/Hy-MT1.5/hy-mt1.5.md)。

### 2.4 大参数量模型的端侧 GGUF 版本（PC 本地运行）

| 模型 | 端侧格式 | 推理方式 | 相关链接 |
| --- | --- | --- | --- |
| Hy4（preview） | MIX-STQ1_0 GGUF | prima.cpp，多机多卡分片 | [GGUF 合集](https://huggingface.co/AngelSlim/Hy4-preview-GGUF)、[部署指南](https://github.com/Tencent/AngelSlim/blob/main/docs/source/_extra/hy4_preview_gguf_guideline.md) |
| Hy4 | FP8 权重与低 bit GGUF | 推理方式见对应模型卡 | [Hy4-FP8](https://huggingface.co/tencent/Hy4-FP8)、[Hy4-GGUF](https://huggingface.co/AngelSlim/Hy4-GGUF) |
| Hy4-flash | 多种量化 GGUF（Q4_K_M、Q3_K、IQ2_AH、SPARK-4Bit 等） | llama.cpp（PR #32517）；Strata（PR #7120）高性能推理 | [Hy4-flash-FP8](https://huggingface.co/tencent/Hy4-flash-FP8)、[Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)、[Strata PR](https://github.com/Niko1221/Strata/pull/7120) |
| Hy-SimulMT（同传） | Q4_K_M / Q8_0 GGUF | llama.cpp | [translation-only](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-GGUF/tree/main)、[interleave](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF/tree/main)、[llama.cpp PR #31198](https://github.com/ggml-org/llama.cpp/pull/31198) |

---

## 3. 环境准备

### 3.1 安装 AngelSlim

```shell
# 默认安装（LLM 量化）
pip install angelslim

# 低比特 QAT / QAD 训练所需的依赖（需要 DeepSpeed 时）
pip install deepspeed

# 从源码安装（修改了源码时推荐）
cd AngelSlim && python setup.py install
```

更多安装选项参见 [安装教程](../../getting_started/installation.md)。

### 3.2 运行入口

所有压缩任务统一通过 `tools/run.py` 执行，由 YAML 配置决定压缩算法、模型与数据：

```shell
python3 tools/run.py -c <配置文件路径>
```

常用覆盖参数：

| 参数 | 说明 |
| --- | --- |
| `-c, --config` | 必填，YAML 配置文件路径 |
| `--model-path` | 覆盖配置中的 `model.model_path`（例如替换为本地模型目录） |
| `--save-path` | 覆盖配置中的 `global.save_path`（压缩产出目录） |
| `--lm-eval` / `--ppl-eval` | 压缩后运行精度评测（可选） |

查看 AngelSlim 支持的压缩策略列表：

```python
from angelslim import engine
engine.get_supported_compress_method()
```

### 3.3 Hugging Face 访问

默认配置会从 Hugging Face 下载模型与数据集。若访问受限，可先从 ModelScope 下载，再使用 `--model-path` 指向本地目录。

---

## 4. 教程导航

| 页面 | 内容 |
| --- | --- |
| [量化](./quantization.md) | PTQ 算法：INT8、SmoothQuant、FP8（含 LeptoQuant）、INT4（GPTQ / AWQ / GPTAQ）、NVFP4、KV Cache 量化、DAQ |
| [低比特训练](./low_bit.md) | QAT（INT4、W4A8-FP8）、QAD（Sherry / SEQ / LSQ / TWN / AbsMean / DLT）、2-bit 翻译模型训练、FP4 W4A4（FOCUS） |
| [端侧部署](./deployment/index.md) | 部署总览与推理校验、vLLM / SGLang、llama.cpp、Strata、多机 prima.cpp、移动端、视觉 Token 压缩、检查清单 |

建议阅读顺序：**总览 → 量化（先跑通 PTQ）→ 部署（确认端侧可用）→ 低比特训练（追求更高压缩率）**。

---

## 5. 端侧相关的论文与资源

| 主题 | 链接 |
| --- | --- |
| AngelSlim 技术报告 | [arXiv:2602.21233](https://arxiv.org/abs/2602.21233) |
| HY-MT1.5 技术报告 | [arXiv:2512.24092](https://arxiv.org/abs/2512.24092) |
| Sherry：3:4 细粒度稀疏的 1.25-bit 三值量化（ACL 2026） | [arXiv:2601.07892](https://arxiv.org/abs/2601.07892) |
| FOCUS：面向 FP4 的端到端 scale 优化 | [arXiv:2608.01847](https://arxiv.org/abs/2608.01847) |
| DAQ：免标定数据的 Delta-Aware 量化 | [arXiv:2603.22324](https://arxiv.org/abs/2603.22324) |
| GPTQ | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| AWQ | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| llama.cpp | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| llama.cpp STQ_0 支持 PR | [PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836) |
| prima.cpp（多机分片推理） | [github.com/OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp) |
| vLLM | [docs.vllm.ai](https://docs.vllm.ai/) |
| lm-evaluation-harness | [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |

---

## 6. 范围说明

- **纳入本教程**：面向端侧模型的量化（PTQ）、低比特量化感知训练（QAT / QAD）、视觉 Token 压缩，以及对应的端侧推理部署。
- **未纳入主线**：投机采样（Eagle3、DFly 等）、稀疏注意力、扩散模型压缩，这些主要面向服务端与训练加速，可分别参考对应的特性文档。
- 本教程中的 GGUF 转换与推理步骤依赖 llama.cpp 等第三方引擎，其参数与支持的算法以对应引擎的版本为准。

有问题可以在 [GitHub Issues](https://github.com/Tencent/AngelSlim/issues) 提出，也可以加入[微信交流群](../../assets/angel_slim_wechat.png)讨论。
