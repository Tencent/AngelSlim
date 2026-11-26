# 多机部署（prima.cpp）

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)

Hy4 preview 参数规模较大，AngelSlim 发布了低 bit 的 GGUF 版本，可以在多台 PC 组成的集群上运行。本页基于 prima.cpp 说明多机部署流程。完整说明也可参考 [Hy4 Preview GGUF 部署指南](https://github.com/Tencent/AngelSlim/blob/main/docs/source/_extra/hy4_preview_gguf_guideline.md)。

## 目录

- [1. 发布资源](#1-发布资源)
- [2. 前置条件](#2-前置条件)
- [3. 解压与检查依赖](#3-解压与检查依赖)
- [4. 生成本机的 receipt 与拓扑文件](#4-生成本机的-receipt-与拓扑文件)
- [5. 生成共享的执行计划](#5-生成共享的执行计划)
- [6. 启动两个 rank](#6-启动两个-rank)
- [7. 调用 API](#7-调用-api)

---

## 1. 发布资源

| 资源 | 用途 |
| --- | --- |
| [Hy4-preview-STQ1_0.gguf](https://huggingface.co/AngelSlim/Hy4-preview-GGUF/tree/main) | 模型文件（需在每台机器上放置） |
| `prima-server-preview-a4000-rank0-linux-x86_64-cuda-runtime.tar.gz` | 4 卡 A4000 机器的 rank-0 运行包（包含 API 服务） |
| `prima-server-preview-4090-rank1-linux-x86_64-cuda-runtime.tar.gz` | 单卡 4090 Laptop 机器的 rank-1 运行包 |

运行包位于 [prima-runtime 目录](https://huggingface.co/AngelSlim/Hy4-preview-GGUF/tree/main/prima-runtime)。

## 2. 前置条件

- 两台 x86-64 Linux 机器，已安装 NVIDIA 驱动；
- 4 卡机器需要 CUDA 12 与 cuBLAS，单卡机器需要 CUDA 13 与 cuBLAS；
- 每台机器约 31 GB 系统内存，并建议使用高速本地存储；
- 两台机器之间开放 4 个 Ring 端口（每个 rank 的数据端口与信号端口）。

## 3. 解压与检查依赖

```shell
# A4000 机器
tar -xzf prima-server-preview-a4000-rank0-linux-x86_64-cuda-runtime.tar.gz
cd prima-server-preview-a4000-rank0-linux-x86_64-cuda-runtime
ldd ./bin/prima-server | sed -n '/not found/p'

# 4090 机器
tar -xzf prima-server-preview-4090-rank1-linux-x86_64-cuda-runtime.tar.gz
cd prima-server-preview-4090-rank1-linux-x86_64-cuda-runtime
ldd ./bin/prima-server | sed -n '/not found/p'
```

`ldd` 没有输出即表示依赖齐全；若出现 `not found`，请安装对应的系统或 CUDA 库。

## 4. 生成本机的 receipt 与拓扑文件

receipt 与拓扑文件与用户自己的模型目录和 GPU 拓扑绑定，需要在本地生成。

```shell
# 两台机器上分别校验 6 个模型文件并生成 receipt
./bin/prima-moe-artifact create-v1 \
  --adapter-id hyv4-middle-iq1-m-r8-v1 \
  --repository HongHuang/middle \
  --revision fc5f23d19a0d6552da1f1f3e39725d068c08444e \
  --manifest /path/to/model-manifest.json \
  --model-root /path/to/model-directory \
  --output rank0-receipt.prar        # 4090 机器上使用 rank1-receipt.prar

# A4000 机器：采集本机 GPU 拓扑
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./bin/prima-server --moe-device-group CUDA0,CUDA1,CUDA2,CUDA3 \
  --moe-topology-probe > rank0-topology.bin

# 4090 机器：采集本机 GPU 拓扑
CUDA_VISIBLE_DEVICES=0 \
  ./bin/prima-server --moe-device-group CUDA0 \
  --moe-topology-probe > rank1-topology.bin
```

manifest 必须列出相同的仓库版本号，以及 6 个文件的精确文件名、字节数与 SHA-256。

## 5. 生成共享的执行计划

把 4090 机器的 receipt 与拓扑文件复制到 A4000 机器，在该机器上生成一份共享计划，然后把计划复制回 4090 机器：

```shell
./bin/prima-moe-plan create-v1 \
  --adapter hyv4-middle-iq1-m-r8-v1 \
  --rank0-receipt rank0-receipt.prar \
  --rank1-receipt rank1-receipt.prar \
  --rank0-topology rank0-topology.bin \
  --rank1-topology rank1-topology.bin \
  --cut 61 \
  --rank0-device-layers 15,15,15,16 \
  --rank1-device-layers 17 \
  --output manual-plan.bin
```

## 6. 启动两个 rank

两台机器都需要设置公共运行环境变量并创建可写目录：

```shell
mkdir -p run/tmp run/logs
unset GGML_CUDA_P2P
export TMPDIR="$PWD/run/tmp"
export PRIMA_LOG_ROOT="$PWD/run/logs"
export CUDA_CACHE_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=0
export GGML_CUDA_DISABLE_GRAPHS=1
export PRIMA_ACTIVATION_WIRE_DTYPE=fp32
export PRIMA_MOE_REQUEST_PLANNER_V1=1
export PRIMA_MOE_BG_PROMOTION_V1=0
export PRIMA_MOE_HOTCACHE_POLICY_V1=recent
export PRIMA_MOE_DECODE_TIMING_V1=1
export PRIMA_MOE_BACKGROUND_STAGING_V1=1
export PRIMA_MOE_PUBLISH_READY_BEFORE_DECODE_V1=1
export PRIMA_MOE_PRECISE_PREFETCH_V1=1
export PRIMA_MOE_ROUTE_PAGE_PREFETCH_V1=1
export PRIMA_MOE_ROUTE_PAGECACHE_POPULATE_V1=1
export PRIMA_MOE_STARTUP_COLD_PREFETCH_V1=1
export PRIMA_MOE_COPY_STREAM_V1=1
export PRIMA_MOE_COPY_STREAM_READY_SYNC_V1=1
export PRIMA_MOE_DRAIN_PROMOTION_COPIES_WITHIN_ALLOWANCE_V1=1
```

**先在 4090 机器（rank 1）启动：**

```shell
export CUDA_VISIBLE_DEVICES=0

./bin/prima-server \
  --model /path/to/model-directory/Hy4-preview-STQ1_0.gguf \
  --load-mode mmap --log-profile production \
  --topology static --world-timeout 7200 \
  --ctx-size 8192 --batch-size 2 --ubatch-size 1 --parallel 1 --n-predict -1 \
  --moe-hybrid manual \
  --moe-provider tiered-local-cold-hot-v1 \
  --moe-model-adapter hyv4-middle-iq1-m-r8-v1 \
  --moe-routing-policy local-cold-no-remote-v1 \
  --moe-min-request-hits 2 --moe-ttft-guard-percent 3 \
  --moe-artifact-receipt /path/to/rank1-receipt.prar \
  --moe-profile-cache /path/to/rank1-profile-cache.bin \
  --moe-manual-plan /path/to/manual-plan.bin \
  --jinja --no-context-shift --no-warmup --no-repack \
  --cache-prompt --cache-reuse 0 --no-slots --no-webui \
  --seed 0 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
  --next A4000_HOST:19867:20894 \
  --data-port 19868 --signal-port 20895 --worker-ready-listener \
  --host 127.0.0.1 --port 18728 \
  --device CUDA0 --split-mode layer --moe-device-group CUDA0 \
  --chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

**再在 A4000 机器（rank 0）启动，它也是唯一对外提供 API 的进程：**

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3

./bin/prima-server \
  --model /path/to/model-directory/Hy4-preview-STQ1_0.gguf \
  --load-mode mmap --log-profile production \
  --topology static --head --world 2 --world-timeout 7200 \
  --ctx-size 8192 --batch-size 2 --ubatch-size 1 --parallel 1 --n-predict -1 \
  --moe-hybrid manual \
  --moe-provider tiered-local-cold-hot-v1 \
  --moe-model-adapter hyv4-middle-iq1-m-r8-v1 \
  --moe-routing-policy local-cold-no-remote-v1 \
  --moe-min-request-hits 2 --moe-ttft-guard-percent 3 \
  --moe-artifact-receipt /path/to/rank0-receipt.prar \
  --moe-profile-cache /path/to/rank0-profile-cache.bin \
  --moe-manual-plan /path/to/manual-plan.bin \
  --jinja --no-context-shift --no-warmup --no-repack \
  --cache-prompt --cache-reuse 0 --no-slots --no-webui \
  --seed 0 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
  --next RANK1_HOST:19868:20895 \
  --data-port 19867 --signal-port 20894 \
  --host 0.0.0.0 --port 18727 \
  --device CUDA0 --split-mode layer \
  --moe-device-group CUDA0,CUDA1,CUDA2,CUDA3 \
  --chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

请将 `A4000_HOST`、`RANK1_HOST` 与所有 `/path/to/...` 替换为实际值。

## 7. 调用 API

rank 0 健康检查通过后，发送一次预热请求，再通过 OpenAI 兼容接口调用：

```text
http://A4000_HOST:18727/v1/chat/completions
```

---

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)
