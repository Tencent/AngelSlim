# Strata 部署

[← llama.cpp 部署](./llamacpp.md) ｜ [← 返回总览](./index.md) ｜ [其他部署体验方案 →](./other.md)

Strata 支持高性能 offload 推理。本章以 AngelSlim 发布的 [Hy4-flash GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) 为例，执行Strata推理。

## 目录

- 1. 获取 Strata 源码
- 2. NVIDIA CUDA 部署
- 3. Apple Silicon（Metal）部署
- 4. 调用与图像输入
- 5. 可用的模型文件

---

## 1. 获取 Strata 源码

```shell
git clone https://github.com/Niko1221/Strata
cd Strata
git fetch origin pull/7120/head:hy4-flash-gguf
git checkout hy4-flash-gguf
```

支持分支见 [Strata PR #7120](https://github.com/Niko1221/Strata/pull/7120)。

---

## 2. NVIDIA CUDA 部署

### 2.1 硬件与软件要求

- **显存**：建议至少 16 GiB。16 GiB 显卡从 `CTX=32768` 开始尝试。
- **内存**：建议至少 128 GiB 可用。默认会在内存中保存完整的 trunk 专家权重（完整模型约 83.4 GiB）。内存不足时使用 `--resident-experts`，在显存与内存之间拆分专家权重。
- **软件**：Linux x86_64、CUDA Toolkit 12.8、CMake 3.24+、Ninja、支持 C++20 的编译器、Python 3.10+。

### 2.2 编译

```shell
cmake -S engine -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -G Ninja
cmake --build build --target strata --parallel 16

python3 -m venv .venv
.venv/bin/python -m pip install -r terminal/requirements.txt -r terminal/requirements-monitor.txt
```

- `CMAKE_CUDA_ARCHITECTURES` 按显卡的计算能力设置，例如 RTX 4090D 使用 `89`，也可以设为 `native`。
- K-quant 的 prefill 需要 `libstrata_kq.so` 与 `strata` 放在同一目录。
- 无界面的服务只需安装 `terminal/requirements.txt`，启动时加 `--no-tui`。

### 2.3 启动

```shell
export MODEL=/path/to/Hy4-flash-Q4_K_M_MTP_Q8_0
./tools/hy4_flash/launch-hy4-flash.sh --plan
./tools/hy4_flash/launch-hy4-flash.sh
```

- `MODEL` 可以指向包含两个分片的目录，也可以指向第一个分片。
- 首次启动会在模型目录下生成 `pack/`。如需指定其他位置，设置 `PACK`。
- 停止服务：`.venv/bin/python terminal/strata.py launch --stop --preset hy4-flash`。

### 2.4 投机解码

- **MTP**：模型和显存条件满足时自动启用。
- **DFly**：将 `hy4-flash_dfly_Q8_0.gguf` 下载到单独目录，然后启动：

  ```shell
  ./tools/hy4_flash/launch-hy4-flash.sh --dfly-model /path/to/drafts/hy4-flash_dfly_Q8_0.gguf
  ```

- 使用 `--draft-tokens 4` 尝试 4 个草稿 token（DFly 默认 3）。`--plan` 可以预览实际配置。

### 2.5 主要启动参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL` | 必填 | 模型目录或第一个分片 |
| `PACK` | `<模型目录>/pack` | 预处理后的 pack 目录 |
| `GPU` / `PORT` / `PRESET` | `0` / `8080` / `hy4-flash` | GPU 编号、HTTP 端口、保存的预设名 |
| `CTX` | `196608` | 最大上下文长度（token） |
| `EXPERT_VRAM` | 自动 | trunk 专家在显存中的预算（GiB） |
| `--mtp on/off` | 自动 | 开启或关闭内置 MTP |
| `--drafting` | `auto` | `auto`、`mtp`、`dfly` 或 `off` |
| `--dfly-model PATH` | 未设置 | 外部 Q8_0 DFly 模型 |
| `--draft-tokens N` | MTP 自动，DFly 为 `3` | MTP 取值 1–7，DFly 取值 1–5 |
| `KV` | `int8` | 潜在 KV 精度，`int8` 或 `fp16` |
| `CPU_MISS_MAX` | `16` | 每层在 CPU 上计算的最大 cache miss 数；设为 `0` 则关闭 CPU 专家计算 |
| `--resident-experts` | 关闭 | 在显存与内存之间拆分专家存储 |
| `--vision on/off` | `off` | 接受聊天请求中的 `image_url` |
| `--set KEY=VAL` | 未设置 | 覆盖自动生成的设置或引擎参数 |

**引擎设置**（通过 `--set KEY=VAL` 传入）：

| 设置 | 默认值 | 说明 |
| --- | --- | --- |
| `STRATA_HY4_FLASH_CPU_WORKERS` | `32` | CPU 专家计算线程数 |
| `STRATA_HY4_FLASH_PREFILL_CHUNK` | 随硬件而定 | 每个 prefill 分块的 token 数 |
| `STRATA_HY4_FLASH_PREFILL_BORROW` | 由启动脚本设置 | `1` 允许 prefill 借用显存，`0` 关闭 |
| `STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR` | `1` | 长 prompt 逐层处理 |
| `STRATA_HY4_FLASH_PREFILL_GROUP_ROWS` | `1024` | 每个专家的 prefill 行容量；出现 "exceeds row capacity" 时可改为 `2048` |
| `STRATA_HY4_FLASH_MTP_HOST_EXPERTS` | 自动 | 将 MTP 专家放在内存中（约节省 3.6 GiB） |
| `STRATA_HY4_FLASH_ENGRAM_IO` | `mmap` | Engram 读取方式，`mmap` 或 `direct`（直接 I/O） |

---

## 3. Apple Silicon（Metal）部署

Metal 版本是为 Apple Silicon 单独实现的原生引擎，支持 MTP 投机解码与图像输入。README 中的测试环境为 M4 Pro、64 GB 统一内存。运行时使用交流电源，模型放在本地 SSD 上。

### 3.1 必需文件

- **模型**：第一个 GGUF 分片，对应 `--model-path`。
- **pack 目录**：包含 tokenizer、聊天模板和稠密权重，对应 `--pack-path`。编译不会生成它，需要先用以下命令构建并校验：

  ```shell
  python terminal/tools/strata_pack.py build  --gguf <第一个分片> --out <pack 目录>
  python terminal/tools/strata_pack.py verify --gguf <第一个分片> --out <pack 目录>
  ```

- **视觉投影（可选）**：`mmproj-Hy4-flash-bf16.gguf`，与模型放在同一目录，或通过 `STRATA_HY4_FLASH_VISION_SHARD` 指定。

未通过 `verify` 的 pack 不要使用。

### 3.2 内存占用

| 组成部分 | 约占用 |
| --- | --- |
| 稠密权重 | ~10.8 GB（常驻） |
| Embedding | ~0.7 GB（常驻） |
| 专家权重 | ~53.4 GB（磁盘，按需加载） |

专家缓存区会自动设置大小。不要设置 `STRATA_METAL_ARENA_GB`，它会关闭引擎的内存保护，可能导致进程被系统因内存不足而终止。内存不足时，让引擎先缩小缓存区，再考虑缩短上下文。

### 3.3 编译

下载对应 macOS 版本的 metal-cpp，解压后的目录中需要包含 `Metal/Metal.hpp`。

```shell
cmake -S engine/metal -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_METAL_ENGINE=ON \
  -DMETAL_CPP_DIR=$PWD/metal-cpp
cmake --build build-metal -j
```

生成两个可执行文件：`strata`（服务端，由 `strata launch` 调用）和 `strata_hy4_flash_metal`（离线驱动与基准测试）。首次编译前需要接受 Xcode 许可协议。

### 3.4 启动

```shell
strata launch --no-tui --yes \
  --model-path <第一个分片> \
  --pack-path <pack 目录> \
  --engine-path build-metal/strata \
  --port 8099
```

服务启动后，`/v1/models` 会将模型列为 `loaded`，`/v1/chat/completions` 接受标准请求。

> **注意**：模型默认开启思考。输出会进入 `reasoning_content`，思考结束前 `content` 为 `null`。只读取 `content` 的客户端会看到没有响应。

### 3.5 启动参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--vision` | 关闭 | `on` 开启图像输入 |
| `--prefill-chunk` | Mac 上为 4096 | 分块 prefill 大小 |
| `--draft-tokens` | 2 | 每轮 MTP 草拟的 token 数 |
| `--context` | 模型默认值 | 处理大图前需要调大 |

**环境变量**：

| 变量 | 说明 |
| --- | --- |
| `STRATA_METAL_ARENA_GB` | 固定缓存区大小并关闭保护（不推荐） |
| `STRATA_HY4_FLASH_VISION_SHARD` | mmproj GGUF 的路径 |
| `HY4_FLASH_MTP_EXPERT_GB` | 草拟阶段的专家预算 |

### 3.6 参考性能

测试环境为 M4 Pro、64 GB，IQ2 模型，8K 上下文：

- prefill：约 125–127 tok/s（原为 35.8）
- decode（256 token）：17.3 tok/s（原为 7.31）
- decode（1024 token）：18.98 tok/s

decode 速度受内存带宽限制：每个 token 约读取 11.7 GB 权重，主要是稠密部分。MTP 每次稠密读取可以验证多个 token，从而提升速度。

---

## 4. 调用与图像输入

启动后接口地址为 `http://localhost:8080/v1`（CUDA 默认端口），可使用任意 OpenAI 兼容客户端。

```shell
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"What is two plus two?"}],"temperature":0,"reasoning_effort":"none","max_tokens":20}'
```

图像输入：

- CUDA：启动时加 `--vision on`，`mmproj-Hy4-flash-bf16.gguf` 放在模型同一目录。
- Metal：加 `--vision on`，以 `data:` URL 形式的 `image_url` 发送图片。图片尺寸为 `h × w` 像素（取 32 的倍数）时，占用 `(h/32) · (w/32 + 1)` 个上下文位置，因此处理大图时需要调大 `--context`。开启视觉不影响纯文本解码。

---

## 5. 可用的模型文件

| 文件 | 说明 |
| --- | --- |
| `Hy4-flash-Q4_K_M*.gguf` | 标准 4-bit，显存与内存充足时推荐 |
| `Hy4-flash-Q3_K*.gguf` | 路由专家为 Q3_K / Q4_K 混合 |
| `Hy4-flash-IQ2_AH*.gguf` | 体积最小，Metal 示例使用该版本 |
| `Hy4-flash-SPARK-4Bit*.gguf` | NVIDIA Spark 设备使用 |
| `hy4-flash_dfly_Q8_0.gguf` | DFly 草稿模型，用于投机解码 |
| `mmproj-Hy4-flash-bf16.gguf` | 视觉投影，图像输入时需要 |

完整文件列表见[总览的已发布模型](./index.md#已发布的端侧-gguf-模型)。
