# 打包 GGUF + 量化

[← 返回总览](./index.md) ｜ [llama.cpp 部署 →](./llamacpp.md)

本章介绍如何自行打包 GGUF 并量化。如果不想自己转换和量化，可以直接使用 AngelSlim 已发布的[量化 GGUF](./index.md#已发布的端侧-gguf-模型)。

```text
Hugging Face 模型
   │  convert_hf_to_gguf.py --outtype bf16
   ▼
bf16 GGUF
   │  llama-quantize <类型>
   ▼
端侧 GGUF（Q4_K_M / Q8_0 / STQ1_0 ...）
```

## 目录

- [1. 准备 llama.cpp](#1-准备-llamacpp)
- [2. 转换与量化](#2-转换与量化)

---

## 1. 准备 llama.cpp

### 1.1 下载和编译

```shell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
cmake -B build                  # NVIDIA GPU 加 -DGGML_CUDA=ON；macOS 默认启用 Metal
cmake --build build --config Release -j
```

`Q4_K_M`、`Q8_0` 等标准类型可以直接进行打包gguf 和量化。以下模型或量化格式需要先切换到对应的 PR 分支，或应用补丁，才能完成转换和量化：

| 模型 / 量化格式 | 所需内容 | 说明 |
| --- | --- | --- |
| `STQ1_0` 量化格式 | [PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836) | STQ1_0 内核 |
| `Q2_0c` 量化格式 | [PR #19357](https://github.com/ggml-org/llama.cpp/pull/19357) | Q2_0c 内核 |
| Hy-SimulMT | [PR #31198](https://github.com/ggml-org/llama.cpp/pull/31198) | 架构支持与 ASR 推理 |
| Hy4-flash | [PR #32517](https://github.com/ggml-org/llama.cpp/pull/32517)，或使用高性能[补丁](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/tree/main/patch) | 架构支持和算子优化 |
| Hy4 | [PR #32517](https://github.com/ggml-org/llama.cpp/pull/32517) | 提供 VLM 支持 |

### 1.2 切换 PR 分支（可选）

以 STQ1_0 为例，切换到 PR #22836 后重新编译：

```shell
git fetch origin pull/22836/head:pr-22836-stq_0
git checkout pr-22836-stq_0
cmake --build build --config Release -j
```

### 1.3 应用补丁（可选）

以 Hy4-flash 的 macOS 补丁 `hy4-flash-mac-minimal.patch` 为例，在 llama.cpp 根目录执行：

```shell
curl -LO https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/main/patch/hy4-flash-mac-minimal.patch
git apply --check hy4-flash-mac-minimal.patch && git apply hy4-flash-mac-minimal.patch
cmake --build build --config Release -j
```

---

## 2. 转换与量化

### 2.1 转换为 bf16 GGUF

```shell
python convert_hf_to_gguf.py /path/to/hf_model \
    --outfile ./model_zoo/model-bf16.gguf \
    --outtype bf16
```

| 参数 | 说明 |
| --- | --- |
| `/path/to/hf_model` | Hugging Face 模型目录 |
| `--outfile` | 输出的 GGUF 文件 |
| `--outtype bf16` | 保留 bf16 精度，量化在下一步完成 |

### 2.2 量化

```shell
./build/bin/llama-quantize ./model_zoo/model-bf16.gguf ./model_zoo/model-Q4_K_M.gguf Q4_K_M
```

| 类型 | 说明 |
| --- | --- |
| `Q8_0` | 8-bit，精度最接近原模型，体积最大 |
| `Q6_K` | 6-bit，精度与体积折中 |
| `Q4_K_M` | 4-bit，端侧最常用 |
| `STQ1_0` | 1.25-bit 三值量化，需要 PR #22836 分支 |

### 2.3 示例：Hy-MT1.5 1.25bit 打包为 STQ1_0

Hy-MT1.5-1.8B 的 1.25-bit 权重发布在 [AngelSlim/Hy-MT1.5-1.8B-1.25bit](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit)，打包为 STQ1_0 的流程如下：

```shell
huggingface-cli download AngelSlim/Hy-MT1.5-1.8B-1.25bit --local-dir model_zoo/Hy-MT1.5-1.8B-1.25bit

python convert_hf_to_gguf.py model_zoo/Hy-MT1.5-1.8B-1.25bit \
    --outfile model_zoo/Hy-MT1.5-1.8B-bf16.gguf --outtype bf16

./build/bin/llama-quantize model_zoo/Hy-MT1.5-1.8B-bf16.gguf \
    model_zoo/Hy-MT1.5-1.8B-STQ1_0.gguf STQ1_0
```

下一步：[llama.cpp 部署 →](./llamacpp.md)
