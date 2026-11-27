# llama.cpp 部署

[← 打包 GGUF + 量化](./gguf_quant.md) ｜ [← 返回总览](./index.md) ｜ [Strata 部署 →](./strata.md)

llama.cpp 加载 GGUF 模型。本章介绍 HY-MT2 与 STQ 模型的运行方法，以及 Hy4-flash 在 Mac 和 DGX Spark 上的高性能 patch。llama.cpp 的编译方法见[打包 GGUF + 量化](./gguf_quant.md#1-准备-llamacpp)。

## 目录

- [1. 运行 HY-MT2（标准量化）](#1-运行-hy-mt2标准量化)
- [2. 运行 STQ 低 bit 模型](#2-运行-stq-低-bit-模型)
- [3. Hy4-flash 高性能 patch](#3-hy4-flash-高性能-patch)
- 4. 运行 Hy4-flash（PR 分支）
- [5. 可用的 GGUF 模型](#5-可用的-gguf-模型)

---

## 1. 运行 HY-MT2（标准量化）

以 Q4_K_M 为例，可直接用 Hugging Face 标签启动：

```shell
# OpenAI 兼容服务
./build/bin/llama-server -hf tencent/Hy-MT2-1.8B-GGUF:Q4_K_M --port 8080

# 终端交互
./build/bin/llama-cli -hf tencent/Hy-MT2-1.8B-GGUF:Q4_K_M
```

推荐采样参数（来自模型卡）：`temperature=0.7`、`top_p=0.6`、`top_k=20`、`repetition_penalty=1.05`。

---

## 2. 运行 STQ 低 bit 模型

需要 PR #22836 分支。以 Hy-MT2 1.25-bit 为例：

```shell
huggingface-cli download tencent/Hy-MT2-1.8B-1.25Bit-GGUF --local-dir model_zoo/Hy-MT2-1.8B-1.25Bit-GGUF

./build/bin/llama-completion \
    --model model_zoo/Hy-MT2-1.8B-1.25Bit-GGUF/<模型文件名>.gguf \
    -p "Translate the following segment into Chinese, without additional explanation：Hello" \
    --jinja -ngl 0 -n 64
```

将 `<模型文件名>.gguf` 替换为模型仓库中的实际文件名。Hy-MT1.5 的 1.25-bit 与 2-bit GGUF 运行方式相同，参见 [Hy-MT1.5 文档](../../models/Hy-MT1.5/hy-mt1.5.md)。

---

## 3. Hy4-flash 高性能 patch

Hy4-flash 提供了针对 Apple Silicon（Metal）和 NVIDIA DGX Spark（CUDA）的高性能 patch。两份 patch 都基于 llama.cpp 提交 `2cbca389f`，应用前请务必切换到该提交。补丁文件位于模型仓库的 [`patch/`](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/tree/main/patch) 目录。

| 平台 | 补丁 | 基线提交 |
| --- | --- | --- |
| macOS（Metal） | `hy4-flash-mac-minimal.patch` | `2cbca389f` |
| NVIDIA DGX Spark（CUDA，GB10） | `hy4-flash-spark.patch` | `2cbca389f515df455412dafa07e2b1359cc72a6d` |

### 3.1 macOS（Metal）

**准备源码并应用补丁**

```shell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 2cbca389f
curl -LO https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/main/patch/hy4-flash-mac-minimal.patch
git apply --check hy4-flash-mac-minimal.patch && git apply hy4-flash-mac-minimal.patch
```

**编译**

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j --target llama-server llama-cli llama-bench
```

**系统设置**

- 提高 GPU 可用内存上限：`sudo sysctl iogpu.wired_limit_mb=114688`，重启后失效。
- GGUF 放在本地 SSD 上，使用交流电源，运行基准测试前先停止服务。

**内存预算**

- 256k 共享池：`-ub 512` 约占 105.5 GiB，`-ub 1024` 约占 107.6 GiB，可以运行。
- 512k 池、4 个 slot、`-ub 1024`：内存不足（约 113.3 GiB）。
- 出现 OOM 时，先缩小上下文池，再降低 `-ub`，然后重启服务。

**启动服务（MTP，推荐）**

```shell
build/bin/llama-server \
  --model /path/to/A12B-Q4_K_M-00001-of-00002.gguf \
  --mmproj /path/to/mmproj-A12B-bf16.gguf \
  --alias hy4-flash-a12b --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 --kv-unified --kv-unified-per-slot 262144 --parallel 2 \
  --n-gpu-layers all --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --batch-size 2048 --ubatch-size 1024 \
  --cache-ram 8192 --ctx-checkpoints 32 \
  --slot-save-path /path/to/slots \
  --reasoning-format auto --jinja --metrics \
  --n-predict 8192 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0 --spec-draft-n-ubatch 256
```

去掉最后一行即为普通自回归解码。

**切换为 DFly 草稿模型**

- 将最后一行替换为 `--fit off --dfly /path/to/hy4-flash_dfly_Q8_0.gguf`。
- 删除 `--mmproj` 与 `--slot-save-path`，DFly 不支持这两个参数。
- `--ubatch-size` 至少为 `1 + K`。

**关键参数说明**

| 参数 | 说明 |
| --- | --- |
| `--cache-type-k f16 --cache-type-v f16` | KV cache 使用 f16。该模型上 q8_0 更慢 |
| `--ubatch-size 1024` | 比 512 的 prefill 更快 |
| `--spec-draft-n-max 2` | MTP 的草稿 token 数。值越大，验证开销越高，建议与 K=0 对比测试 |
| `--n-gpu-layers all` | 尽可能把所有层放到 GPU |

MTP 的验证是精确的拒绝采样，输出与普通解码一致。

**参考性能**（128 GB Mac，Metal）：自回归约 36 tok/s；MTP（K=2）44–48 tok/s，Q4_K_M 在 4K 上下文为 48.13 tok/s；DFly（K=4）在约 81k 上下文为 51.95 tok/s。

运行时可选的环境变量：`HY4_MLA_DISABLE` 回退到通用内核；`GGML_METAL_OP_PROFILE=1` 与 `GGML_METAL_CB_TIMING=1` 开启性能分析。

### 3.2 NVIDIA DGX Spark（CUDA）

适用于 DGX Spark / GB10（计算能力 12.1，架构 `121a`），支持 Linux 与 Windows。模型使用 `Hy4-flash-SPARK-4Bit.gguf`。

**准备源码并应用补丁**

```shell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 2cbca389f515df455412dafa07e2b1359cc72a6d
cd ..
curl -LO https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/main/patch/hy4-flash-spark.patch
cd llama.cpp
git apply --check ../hy4-flash-spark.patch && git apply ../hy4-flash-spark.patch
```

**编译**

CUDA 版本为 13.0，请按实际路径调整。

```shell
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121a -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build -j --target llama-server llama-bench
```

**环境变量**

```shell
export GGML_CUDA_FUSE_HC_GATE=1
export GGML_CUDA_MOE_CONCURRENT=1
export GGML_CUDA_MMQ_PREC=q4
export GGML_CUDA_FA_576_WHOLE_TILES=0
export GGML_SCHED_BATCH_HOST_INPUTS=0   # Linux；Windows 设为 2
```

README 中还列出了其他 `GGML_*` 变量，完整列表见 [README_SPARK-4Bit.md](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/blob/main/README_SPARK-4Bit.md)。

**启动服务（MTP，推荐）**

```shell
build/bin/llama-server \
  --model /path/to/Hy4-flash-SPARK-4Bit.gguf \
  --mmproj /path/to/mmproj-Hy4-flash-bf16.gguf --mmproj-offload \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 196608 --parallel 1 \
  --n-gpu-layers 999 --split-mode none \
  --batch-size 2048 --ubatch-size 1024 \
  --threads 4 --threads-batch 4 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --fit off \
  --spec-type draft-mtp --metrics --jinja
```

- 纯文本服务：删除 `--mmproj` 与 `--mmproj-offload`。
- 内存不足时，降低 `--ctx-size`（推荐值 196608 可能需要调小）。

**DFly 草稿模型（可选）**

纯文本场景下，删除 mmproj 相关参数，加入 `--no-mmproj --no-context-shift --cache-reuse 0`，并把 MTP 行替换为：

```shell
--spec-type draft-dfly --spec-draft-model /path/to/hy4-flash_dfly_Q8_0.gguf \
--spec-draft-ngl 999 --spec-draft-n-max 4 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy
```

---

## 4. 运行 Hy4-flash（PR 分支）

只使用 PR 分支、不应用补丁时，可以运行普通的 CLI 与服务：

```shell
./build/bin/llama-cli -m Hy4-flash-Q4_K_M-00001-of-00002.gguf -c 8192 -ngl 99 --jinja
./build/bin/llama-server -m Hy4-flash-Q4_K_M-00001-of-00002.gguf -c 8192 -ngl 99 --jinja --port 8080
```

模型为多分片，所有分片放在同一目录，指定第一个分片即可。PR 分支的切换方式见[打包 GGUF + 量化](./gguf_quant.md)。

---

## 5. 可用的 GGUF 模型

完整文件列表见[总览的已发布模型](./index.md#已发布的端侧-gguf-模型)。

下一步：[Strata 部署 →](./strata.md)
