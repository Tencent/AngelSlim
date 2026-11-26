# Strata 部署（Hy4-flash）

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)

Strata 是面向 Hy4-flash GGUF 的高性能推理运行时，支持 NVIDIA CUDA 与 Apple Silicon（Metal）。它提供启动脚本，可配置 CPU 专家卸载、MTP 与 DFly 投机解码等，适合在单张 GPU 或 Apple 设备上追求更高吞吐。

## 目录

- [1. 获取 Strata 源码](#1-获取-strata-源码)
- [2. 选择 Hy4-flash 模型文件](#2-选择-hy4-flash-模型文件)
- [3. NVIDIA CUDA 启动](#3-nvidia-cuda-启动)
- [4. Apple Silicon（Metal）启动](#3-nvidia-cuda-启动)
- [5. 调用 API 与图像输入](#5-调用-api-与图像输入)

---

Strata 的支持分支见 [Strata PR #7120](https://github.com/Niko1221/Strata/pull/7120)。本页命令沿用模型仓库中的写法，本地分支名为 `hy4-flash-gguf`。

## 1. 获取 Strata 源码

```shell
git clone https://github.com/Niko1221/Strata
cd Strata
git fetch origin pull/7120/head:hy4-flash-gguf
git checkout hy4-flash-gguf
```

编译步骤请参考仓库中的 CUDA 与 Metal 构建文档。

## 2. 选择 Hy4-flash 模型文件

模型仓库：[AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)。主要文件如下（大小为仓库页面所示的分片合计）：

| 文件 | 约合大小 | 说明 |
| --- | --- | --- |
| `Hy4-flash-Q4_K_M-*-of-00002.gguf` | 168 GB | 标准 4-bit，模型卡的默认推荐，内存充足时使用 |
| `Hy4-flash-Q3_K-*-of-00002.gguf` | 155 GB | 路由专家为 Q3_K / Q4_K 混合，其余与 Q4_K_M 相同 |
| `Hy4-flash-IQ2_AH-*-of-00002.gguf` | 135 GB | 最小的版本，路由 gate/up 为 IQ2_XXS，非专家权重多为 BF16 |
| `Hy4-flash-SPARK-4Bit.gguf` | 163 GB | NVIDIA Spark 设备的 4-bit 浮点版本 |
| `Hy4-flash-SPARK-4Bit-Fast.gguf` | 160 GB | SPARK-4Bit 的加速版本 |
| `Hy4-flash-Q4_K_M_MTP_Q8_0-*-of-00002.gguf` | 164 GB | 带 MTP 层，用于 Strata 的 CUDA 示例 |
| `Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25-*-of-00002.gguf` | 141 GB | MTP 版本的剪枝 25% 变体 |
| `hy4-flash_dfly_Q8_0.gguf` | 约 0.9 GB | DFly 草稿模型，用于投机解码 |
| `mmproj-Hy4-flash-bf16.gguf` | 约 1 GB | 视觉投影文件，图像输入时需要 |

多分片文件必须放在同一目录，将 `MODEL` 指向第一个分片即可。

## 3. NVIDIA CUDA 启动

以下示例使用 `Q4_K_M_MTP_Q8_0` 版本，开启 CPU 专家卸载以及 MTP / DFly 投机解码，上下文 32768：

```shell
cd Strata
MODEL=/path/to/Hy4-flash-Q4_K_M_MTP_Q8_0-00001-of-00002.gguf CTX=32768 \
  ./tools/hy4_flash/launch-hy4-flash.sh --no-tui
```

## 4. Apple Silicon（Metal）启动

以下示例使用 IQ2 版本，启用 SSD 专家流式加载与 MTP，上下文 32768：

```shell
cd Strata
MODEL=/path/to/Hy4-flash-IQ2_AH-00001-of-00002.gguf \
  STRATA_BIN="$PWD/build-metal/strata" KV=fp16 CTX=32768 MTP=on DRAFT_TOKENS=2 \
  ./tools/hy4_flash/launch-hy4-flash.sh --no-tui
```

`STRATA_BIN` 需指向 Metal 构建输出的可执行文件。macOS 的专用说明见模型仓库的 `README_strata_MAC-Metal.md`。

## 5. 调用 API 与图像输入

启动成功后，API 地址为：

```text
http://localhost:8080/v1
```

使用 OpenAI 兼容客户端即可调用。如需图像输入，将 `mmproj-Hy4-flash-bf16.gguf` 放在模型文件同一目录，并在启动时增加 `--vision on`。

Strata 的完整配置与其它启动参数见模型仓库中的 `README_strata.md` 与 `README_strata.zh-CN.md`。

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)
