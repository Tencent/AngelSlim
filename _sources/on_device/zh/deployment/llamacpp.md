# llama.cpp 部署

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)

llama.cpp 使用 GGUF 格式，适用于 PC、手机与纯 CPU 设备。本页说明 llama.cpp 的安装，以及 HY-MT2 与 Hy4-flash GGUF 的运行方式。

## 目录

- [1. 安装 llama.cpp](#1-安装-llamacpp)
- [2. 运行 HY-MT2](#2-运行-hy-mt2)
- [3. 运行 Hy4-flash](#3-运行-hy4-flash)
- [4. 其它已发布的 GGUF 模型](#4-其它已发布的-gguf-模型)

---

## 1. 安装 llama.cpp

**方式一：模型页面提供的安装脚本（macOS / Linux）**

```shell
curl -LsSf https://llama.app/install.sh | sh
```

**方式二：Windows**

```shell
winget install llama.cpp
```

**方式三：从源码编译**

```shell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build                  # NVIDIA GPU 加 -DGGML_CUDA=ON；macOS 默认开启 Metal
cmake --build build --config Release -j
```

> **注意**：部分 GGUF 依赖尚未合入主线的 llama.cpp 支持分支。STQ 低比特内核见 [PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836)，Hy4-flash 见 [PR #32517](https://github.com/ggml-org/llama.cpp/pull/32517)。请按各节说明切换分支后再编译。

---

## 2. 运行 HY-MT2

HY-MT2 提供标准量化与低比特量化两类 GGUF。

**（1）标准量化（`Q4_K_M`、`Q6_K`、`Q8_0`）**

以 [tencent/Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) 为例，可以直接用 Hugging Face 标签启动：

```shell
# 兼容 OpenAI 的服务（带 Web UI）
llama serve -hf tencent/Hy-MT2-1.8B-GGUF:Q4_K_M

# 终端交互
llama cli -hf tencent/Hy-MT2-1.8B-GGUF:Q4_K_M
```

也可以使用 `ollama run hf.co/tencent/Hy-MT2-1.8B-GGUF:Q4_K_M` 或 `docker model run hf.co/tencent/Hy-MT2-1.8B-GGUF:Q4_K_M`。

**（2）低比特量化（1.25-bit、2-bit）**

低比特版本依赖 STQ 内核，需要先切换到 PR #22836 分支并编译（见[第 1 节](#1-安装-llamacpp)）。以 1.25-bit 版本 [tencent/Hy-MT2-1.8B-1.25Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) 为例：

```shell
# 下载模型（请先在模型页确认 GGUF 文件名）
huggingface-cli download tencent/Hy-MT2-1.8B-1.25Bit-GGUF --local-dir model_zoo/Hy-MT2-1.8B-1.25Bit-GGUF

# 运行一次翻译（CPU）
./build/bin/llama-completion \
  --model model_zoo/Hy-MT2-1.8B-1.25Bit-GGUF/model.gguf \
  -p "Translate the following segment into Chinese, without additional explanation：Hello" \
  --jinja \
  -ngl 0 \
  -n 64 -st

# 吞吐测量
./build/bin/llama-bench -m model_zoo/Hy-MT2-1.8B-1.25Bit-GGUF/model.gguf -ngl 0
```

`model.gguf` 是模型卡中的占位文件名，请替换为实际文件名。2-bit 版本（[tencent/Hy-MT2-1.8B-2Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF)）的运行方式相同，只需替换仓库名。注意该卡片引用的 STQ 支持 PR 为 #19357，与其它卡片的 #22836 不一致，请以模型卡最新说明为准。

**参数说明**

| 参数 | 说明 |
| --- | --- |
| `--model` | GGUF 文件路径 |
| `-p` | 提示词。翻译提示词需写明目标语言，并要求只输出译文 |
| `--jinja` | 使用模型自带的 Jinja 聊天模板 |
| `-ngl 0` | GPU 层数为 0，即全部在 CPU 上运行 |
| `-n 64` | 最多生成 64 个 Token |
| `-st` | 模型卡原样保留的参数 |

**推荐采样参数**（1.8B 与 7B，来自模型卡）：`temperature=0.7`、`top_p=0.6`、`top_k=20`、`repetition_penalty=1.05`、`max_tokens=4096`。

---

## 3. 运行 Hy4-flash

Hy4-flash 的 llama.cpp 支持位于 [PR #32517](https://github.com/ggml-org/llama.cpp/pull/32517)，尚未合入主线，需要切换到对应分支：

```shell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin pull/32517/head:hy4-flash
git checkout hy4-flash

cmake -B build                  # NVIDIA GPU 加 -DGGML_CUDA=ON；macOS 默认开启 Metal
cmake --build build --config Release -j
```

Hy4-flash 的 GGUF 为多分片文件，请保持所有分片在同一目录，并指定第一个分片。文件列表见 [Strata 部署的文件说明](./strata.md#2-选择-hy4-flash-模型文件)。

**终端对话**（上下文 8192）：

```shell
./build/bin/llama-cli -m Hy4-flash-Q4_K_M-00001-of-00002.gguf -c 8192 -ngl 99 --jinja
```

**OpenAI 兼容服务**（端口 8080）：

```shell
./build/bin/llama-server -m Hy4-flash-Q4_K_M-00001-of-00002.gguf -c 8192 -ngl 99 --jinja --port 8080
```

**图像输入**（需要 `mmproj-Hy4-flash-bf16.gguf`）：

```shell
./build/bin/llama-mtmd-cli -m Hy4-flash-Q4_K_M-00001-of-00002.gguf --mmproj mmproj-Hy4-flash-bf16.gguf \
    --image photo.png -p "Describe this image." -c 8192 -ngl 99 --jinja
```

参数说明：`-c` 为上下文长度，`-ngl 99` 尽可能把层放到 GPU 上，`--jinja` 使用模型聊天模板。平台专用说明见模型仓库中的 `README_MAC-Metal.md` 与 `README_SPARK-4Bit.md`。

模型仓库的 [`patch/` 目录](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/tree/main/patch) 提供了 llama.cpp 高性能版本的两个补丁：`hy4-flash-mac-minimal.patch`（macOS）与 `hy4-flash-spark.patch`（NVIDIA Spark）。使用前请参考模型仓库中的平台说明。


## 4. 其它已发布的 GGUF 模型

| 模型 | 格式 | 链接 |
| --- | --- | --- |
| HY-MT2-30B-A3B-GGUF | GGUF | [tencent/Hy-MT2-30B-A3B-GGUF](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF) |
| HY-MT2-7B-GGUF | GGUF | [tencent/Hy-MT2-7B-GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) |
| Hy-MT1.5-1.8B-2bit | GGUF（2-bit） | [AngelSlim/Hy-MT1.5-1.8B-2bit-GGUF](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-2bit-GGUF) |
| Hy-MT3 端侧版本 | GGUF | [tencent/hy-mt3 合集](https://huggingface.co/collections/tencent/hy-mt3) |
| Hy-SimulMT（translation-only） | Q4_K_M、Q8_0 | [tencent/Hy-SimulMT-30B-A3B-GGUF](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-GGUF/tree/main) |
| Hy-SimulMT（interleave） | Q4_K_M、Q8_0 | [tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF/tree/main) |
| Hy4-flash | 多种量化 GGUF | [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) |
| Hy4 | 低 bit GGUF | [AngelSlim/Hy4-GGUF](https://huggingface.co/AngelSlim/Hy4-GGUF) |

Hy-SimulMT 的 llama.cpp 支持 PR 见 [PR #31198](https://github.com/ggml-org/llama.cpp/pull/31198)。

[← 部署总览](./index.md)
