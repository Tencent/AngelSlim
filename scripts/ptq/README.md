# PTQ 校准 / 量化脚本说明

本目录包含基于 [vLLM](https://github.com/vllm-project/vllm) 的 **PTQ（Post-Training Quantization）** 校准和量化脚本。

> ⚠️ **重要**：所有脚本必须从 `AngelSlim` 仓库根目录执行（脚本内部使用 `tools/...` 形式的相对路径）。

---

## 一、环境准备（运行校准脚本前必须完成）

> 📌 **硬性要求**（当前 Hy3 校准脚本经过验证的配置）：
> - **算力**：**16 卡**（两个节点 × 每节点 8 卡），用于 TP/PP 跨节点切分
> - **vLLM 版本**：**v0.20.0**（补丁文件按此版本对齐，其它版本需要重新生成补丁）
> - **Python 环境**：所有节点保持一致（建议使用同一个 conda / venv）
>
> 本节包含两步：
> 1. 搭建 **Ray 集群**（跨节点拉起 16 卡）
> 2. 在 **每个 vLLM 运行节点** 上打 AngelSlim 补丁
>
> 两步完成后才能运行后续的校准 / 量化脚本。

### 1. 准备 Ray 集群（2 节点 × 8 卡 = 16 卡）

Hy3 等大模型需要跨节点 TP/PP，校准脚本默认走 vLLM 的 Ray distributed executor，必须先在 **两台 8 卡节点** 上分别拉起 Ray，组成一个 16 卡集群。

下面给出的环境变量按 **RDMA / 多网卡** 集群的常见配置示例，请按实际网络拓扑调整（特别是 `*_SOCKET_IFNAME`、`NCCL_IB_GID_INDEX`）。

#### 主节点（head）

```bash
# —— NCCL / GLOO 通信网卡 —— 
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
# —— vLLM 相关 —— 
export VLLM_USE_DEEP_GEMM=0
# —— 提高文件句柄上限，避免 Ray 大量连接报 EMFILE —— 
ulimit -n 65536

ray start --head \
    --port 6700 \
    --num-gpus=8 \
    --disable-usage-stats \
    --metrics-export-port=8080
```

#### 从节点（worker）

```bash
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export VLLM_USE_DEEP_GEMM=0
ulimit -n 65536

ray start \
    --address <head_ip>:6700 \
    --num-gpus=8 \
    --disable-usage-stats
```

> ⚠️ 注意：
> - 两个节点的 **环境变量**、**Python 环境** 与 **vLLM 版本（v0.20.0）** 必须完全一致，否则会出现 NCCL 通信失败或 worker 崩溃。
> - 集群拉起后用 `ray status` 确认 `Total GPUs = 16`（两节点各 8 卡）。

### 2. 给 vLLM 打 AngelSlim 补丁

校准流程依赖 **打过 AngelSlim patch 的 vLLM**——`FusedMoE` 的 expert 统计钩子、`VLLM_MOE_COLLECT_STATS*` 环境变量都来自这套补丁。**未打补丁时，MoE expert 统计将全部缺失，最终 FP8 模型不可用。**

补丁源码位于 [`tools/vllm_patch/`](../../tools/vllm_patch/)，详见该目录下的 [`README.md`](../../tools/vllm_patch/README.md)。

#### 2.1 一键安装（推荐）

在 **每一台**会运行 vLLM 的机器上（含 Ray 多节点集群的 head + 所有 worker），从 `AngelSlim` 仓库根目录执行：

```bash
bash tools/vllm_patch/install.sh install
```

`install.sh` 会自动：

1. 通过 `python3 -c 'import vllm'` 定位当前 Python 环境下的 vLLM 安装目录。
2. **首次安装时** 把 `envs.py` 与 `model_executor/layers/fused_moe/fused_moe.py` 备份为 `*.bak`（重复执行不会覆盖已有备份）。
3. 用 `tools/vllm_patch/{envs.py, fused_moe.py}` 替换原文件。
4. 把 `angelslim/compressor/quant/core/vllm_calibrate_utils/` 拷贝到 `<vllm_install_dir>/tools/vllm_calibrate_utils/`（`fused_moe.py` 运行时会从这里 import `collect_fused_moe_internal_stats`，拆分后仍以包名 `vllm_calibrate_utils` 暴露该符号）。
5. 自动跑一次 `check`，校验补丁标记是否生效。

#### 2.2 验证 / 还原

```bash
bash tools/vllm_patch/install.sh check       # 校验补丁是否处于激活状态
bash tools/vllm_patch/install.sh uninstall   # 用 *.bak 还原原始 vLLM 文件
bash tools/vllm_patch/install.sh --help      # 查看完整用法
```

`check` 通过的标志：

- `envs.py` 包含 `VLLM_MOE_COLLECT_PER_EXPERT_STATS`
- `fused_moe.py` 包含 `collect_fused_moe_internal_stats`
- `<vllm_install_dir>/tools/vllm_calibrate_utils/__init__.py` 存在

#### 2.3 多节点 / 多环境注意事项

- **Ray 集群**：`install.sh` 只会修改本机的 vLLM，head 与每个 remote worker 都需独立执行一次。
- **vLLM 版本耦合**：补丁文件对齐当前校准环境内的 vLLM 版本，升降级 vLLM 后请重新生成补丁或回退。
- **切换 Python 环境**：补丁会装到 `python3` 默认指向的 vLLM；用 conda / venv 切环境后需在新环境里重跑 `install`。

补丁就绪后，再继续下面的脚本使用说明。

---

## 二、Hy3.0 系列脚本（Hunyuan-A20B 等 Hy3 模型）

下面 7 个脚本共享同一套 vLLM 运行时环境（chunked prefill / FlashInfer attention / mp distributed executor / fused MoE 等），区别在于产出物不同。

| 脚本 | 用途 | 入口 |
| --- | --- | --- |
| [`run_vllm_quant_for_Hy3.sh`](./run_vllm_quant_for_Hy3.sh) | ★ FP8 一键流水线：校准 + 量化 | `tools/run_vllm_calibrate.py` + `tools/fp8_quant_with_vllm_activation.py` |
| [`run_nvfp4_quant_for_Hy3.sh`](./run_nvfp4_quant_for_Hy3.sh) | ★ NVFP4 一键流水线：校准 + weight-only 量化 + 合并 | `tools/run_vllm_calibrate.py` + `tools/run.py` + `tools/merge_hy3_nvfp4_c8.py` |
| [`run_vllm_calibrate_for_Hy3.sh`](./run_vllm_calibrate_for_Hy3.sh) | 仅 W8A8C8 联合校准 | `tools/run_vllm_calibrate.py` |
| [`run_kvcache_calibrate_for_Hy3.sh`](./run_kvcache_calibrate_for_Hy3.sh) | 仅 KV-cache 校准（轻量） | `tools/kvcache/run_kvcache_calibrate.py` |
| [`run_smooth_for_HY3.sh`](./run_smooth_for_HY3.sh) | SmoothQuant 一键流水线：统计收集 + 权重变换 | `tools/smooth/run_vllm_smooth.py` + `tools/smooth/convert_smooth_weights.py` |
| [`run_smooth_calibrate_for_HY3.sh`](./run_smooth_calibrate_for_HY3.sh) | 仅 Smooth 统计收集（+ 可选 Alpha 搜索） | `tools/smooth/run_vllm_smooth.py` |
| [`run_smooth_convert_for_HY3.sh`](./run_smooth_convert_for_HY3.sh) | 仅 Smooth 离线权重变换 | `tools/smooth/convert_smooth_weights.py` |

> 📖 SmoothQuant 完整文档（核心概念、配置详解、Alpha 搜索原理、故障排查）见 [tools/smooth/README.md](../../tools/smooth/README.md)。

---

### 0. `run_smooth_for_HY3.sh` — 可选的模型 Smooth 转换

**功能**：Smooth 预处理（生成平滑后的模型），可作为后续 FP8 量化的前置步骤，提升低比特量化精度。

```bash
bash run_smooth_for_HY3.sh                    # 两阶段都跑
bash run_smooth_for_HY3.sh --skip-calibrate   # 仅 Phase 2（复用已有统计）
bash run_smooth_for_HY3.sh --skip-convert     # 仅 Phase 1
```

#### Phase 1：调用 `tools/smooth/run_vllm_smooth.py`

- 用 vLLM 加载模型，在校准数据集上跑前向，收集 Attention / MLP / MoE 各层的 per-channel 激活统计（absmax + EMA）。
- 可选执行 per-layer Alpha 网格搜索，自动寻找最优平滑参数。
- 输出到 `${output_dir}`：
  - `smooth_stats.json` — 各层 per-channel absmax / EMA 统计
  - `smooth_alpha_search.json`（若 `enable_alpha_search: true`）— 每层最优 alpha 及对应的 smooth_weight

#### Phase 2：调用 `tools/smooth/convert_smooth_weights.py`

- 读取 Phase 1 产出的统计文件，对 QK / VO / Down 投影层权重做离线缩放变换。
- 输出到 `${save_path}`：平滑后的 HuggingFace safetensors 模型（可直接用于后续量化或推理）。

#### 配置

默认读取 `configs/Hy3/ptq/fp8/Hy3_smooth.yaml`（Phase 1 和 Phase 2 共享同一份 YAML）。详见 [tools/smooth/README.md](../../tools/smooth/README.md)。

---

### 1. `run_vllm_quant_for_Hy3.sh` ★推荐的"一键流水线"

**功能**：bf16 模型 → vLLM 激活校准 → FP8 HF safetensors，全流程一次完成。

#### 阶段 1：调用 `tools/run_vllm_calibrate.py`

- 用 vLLM 加载 bf16 模型，在 PTQ 数据集上跑前向，注册 weight / activation / MoE / KV-cache 钩子。
- 输出到 `${stats_dir}`：
  - `activation_stats.json` — per-tensor min/max（含合并后的 per-head 项）
  - `moe_expert_stats.json` — 每个 MoE expert 的输入激活统计
  - `kv_scale_multipliers*.json` — 若开启 `--search-kv-scale`
  - `kv_cache_tuned_scales*.json` — 搜索后的最终 KV scale

#### 阶段 2：调用 `tools/fp8_quant_with_vllm_activation.py`

- 读取 `${stats_dir}` 下的 `activation_stats.json` / `moe_expert_stats.json`，结合原 bf16 权重，做 per-tensor FP8 量化（含 weight + input scale），写出到 `${fp8_path}`。
- 校准（stage-1）与量化（stage-2）共享 **同一份 YAML**：[`configs/Hy3/ptq/fp8/Hy3_vllm_ptq_per_tensor.yaml`](../../configs/Hy3/ptq/fp8/Hy3_vllm_ptq_per_tensor.yaml)。
  - 路径只配一次：stage-2 的 `input_bf16_hf_path` 默认回退到 stage-1 的 `model_path`，`input_vllm_ac_json_path` 默认回退到 stage-1 的 `output_dir`。
  - 每个阶段只读取自己关心的字段，不认识的字段会打一行 `[yaml-config] WARNING: unknown keys` 然后忽略，属于正常现象。
- KV-cache 的"校准粒度"与"量化粒度"分开控制：
  - 校准阶段（stage-1）由 `kv_granularity`（`none` | `per-tensor` | `per-head`）决定 KV scale 的收集粒度。
  - 量化阶段（stage-2）由 `k_scheme` / `v_scheme`（`dynamic` | `static`）决定是否把 scale 写进 safetensor；当 scheme=`static` 时，再由 `quant_k_granularity` / `quant_v_granularity`（`none` | `per-tensor` | `per-head`）决定写入粒度。
- KV-cache scale 的写入行为由量化阶段的 `k_scheme` / `v_scheme` 控制：
  - `static`：将校准得到的 scale 写入 `kv_cache_scales.safetensors`，粒度由 `quant_k_granularity` / `quant_v_granularity` 决定（`none` | `per-tensor` | `per-head`）。
  - `dynamic`：不写入对应的 scale（`model.safetensors.index.json` 中也不包含对应 key），`config.json` 中标记为 `"scheme": "dynamic", "granularity": "per_token_per_head"`（与 `q_quant` 一致）。
- 产出的 `config.json` 中 `attn_quant_config.kv_cache_quant` 的 `k_quant` 和 `v_quant` 独立配置，支持 K/V 使用不同的 scheme。

#### CLI 开关

```bash
bash run_vllm_quant_for_Hy3.sh                    # 两阶段都跑
bash run_vllm_quant_for_Hy3.sh --skip-calibrate   # 仅量化（复用已有 stats_dir）
bash run_vllm_quant_for_Hy3.sh --skip-quantize    # 仅校准
bash run_vllm_quant_for_Hy3.sh --help             # 打印用法
```

> 脚本开启 `set -euo pipefail`，任一阶段失败将立即中断。

---

### 2. `run_nvfp4_quant_for_Hy3.sh` ★推荐的 NVFP4 "一键流水线"

**功能**：bf16 模型 → vLLM 激活校准 → NVFP4 weight-only 量化 → NVFP4 + FP8 scale HF safetensors，全流程一次完成。

#### 阶段 1：调用 `tools/run_vllm_calibrate.py`

- 用 vLLM 加载 bf16 模型，在 PTQ 数据集上跑前向，注册 activation / MoE / KV-cache 钩子。
- 脚本传入 `--auto-detect-mtp`：若 checkpoint 在 `num_hidden_layers` 之后还包含追加的 MTP 层，则自动开启 MTP draft model 校准；无 MTP 时保持原校准流程。
- 输出到 `${STATISTICS_PATH}`：
  - `activation_stats.json` — activation 与 KV-cache 统计
  - `moe_expert_stats.json` — 每个 MoE expert 的输入激活统计
  - `mtp_activation_stats.json` — 检测到 MTP 时生成的 MTP Linear / KV-cache 统计
  - `mtp_moe_expert_stats.json` — 检测到 MTP MoE 时生成的 per-expert 统计

#### 阶段 2：调用 `tools/run.py`

- 读取 bf16 权重，对 MoE expert 权重执行 NVFP4 weight-only 量化。
- 默认配置为 [`configs/Hy3/ptq/nvfp4_weight_only/hunyuan_a20b_nvfp4_weight_only.yaml`](../../configs/Hy3/ptq/nvfp4_weight_only/hunyuan_a20b_nvfp4_weight_only.yaml)。
- 量化模型目录需与 `${NVFP4_W_PATH}` 保持一致，供阶段 3 读取。

#### 阶段 3：调用 `tools/merge_hy3_nvfp4_c8.py`

- 读取 `${STATISTICS_PATH}` 下的 `activation_stats.json` / `moe_expert_stats.json`、`${NVFP4_W_PATH}` 下的 NVFP4 权重及 `${BF16_MODEL_PATH}` 下的原始模型文件。
- 把 NVFP4 expert 权重、bf16 非 expert 权重、expert input scale 和 KV-cache scale 合并到 `${OUTPUT_PATH}`。
- 默认使用 `--mtp-fp8-mode auto`：检测到 MTP 时，从原始 bf16 checkpoint 重新读取 MTP 权重，并使用 `mtp_activation_stats.json` / `mtp_moe_expert_stats.json` 将支持的 MTP Linear 与 MoE GEMM 权重量化为静态 per-tensor FP8；norm、router 等非量化参数保持 bf16。
- 含 MTP 的最终 `config.json` / `hf_quant_config.json` 使用 ModelOpt `MIXED_PRECISION`，通过 `quantized_layers` 标记主模型 expert 为 NVFP4、MTP 量化模块为 FP8；无 MTP 时保持原 NVFP4 配置。
- 产出的 HuggingFace 模型包含 NVFP4 权重、对应 scale、KV-cache scale、`config.json` 和 tokenizer 文件。

校准阶段默认使用 [`configs/Hy3/ptq/fp8/Hy3_vllm_ptq_per_tensor.yaml`](../../configs/Hy3/ptq/fp8/Hy3_vllm_ptq_per_tensor.yaml)，weight-only 阶段默认使用 [`configs/Hy3/ptq/nvfp4_weight_only/hunyuan_a20b_nvfp4_weight_only.yaml`](../../configs/Hy3/ptq/nvfp4_weight_only/hunyuan_a20b_nvfp4_weight_only.yaml)。运行前需要确认：

- 校准 YAML 的 `model_path` / `output_dir` 与实际 bf16 模型、`${STATISTICS_PATH}` 一致。
- NVFP4 YAML 的 `model.model_path` / `global.save_path` 与实际 bf16 模型、`${NVFP4_W_PATH}` 一致。
- `${BF16_MODEL_PATH}` 指向原始 bf16 HuggingFace 模型；默认占位值 `/path/to/bf16_model` 必须替换。
- `${BF16_MODEL_PATH}`、校准 YAML 的 `model_path` 和 NVFP4 YAML 的 `model.model_path` 必须指向同一个模型版本，否则 MTP 层检测、校准统计和权重名称可能无法对应。
- `PTQ_CONFIG`、`NVFP4_CONFIG`、`WORK_DIR`、`STATISTICS_PATH`、`NVFP4_W_PATH`、`BF16_MODEL_PATH`、`OUTPUT_PATH` 和 `LOG_DIR` 均可通过环境变量覆盖。

#### CLI 开关

```bash
bash run_nvfp4_quant_for_Hy3.sh                       # 三阶段都跑
bash run_nvfp4_quant_for_Hy3.sh --skip-calibrate      # 复用已有校准统计
bash run_nvfp4_quant_for_Hy3.sh --skip-weight-only    # 复用已有 NVFP4 权重
bash run_nvfp4_quant_for_Hy3.sh --skip-merge          # 只运行校准和 weight-only 量化
bash run_nvfp4_quant_for_Hy3.sh --help                # 打印用法
```

> 多个 `--skip-*` 开关可以组合。脚本开启 `set -euo pipefail`，任一阶段失败将立即中断。

---

### 3. `run_vllm_calibrate_for_Hy3.sh` — 一键脚本里的"阶段 1"独立版

**功能**：只跑 W8A8C8 联合校准，不做量化。

- **入口**：`tools/run_vllm_calibrate.py`
- **开启的环境变量**：
  ```bash
  VLLM_MOE_COLLECT_STATS=1
  VLLM_MOE_COLLECT_PER_EXPERT_STATS=1
  VLLM_MOE_COLLECT_STATS_VERBOSE=0
  ```
- **默认配置**：`--kv-granularity per-head`，并开启 `--search-kv-scale`。
- **注意**：校准阶段无论后续 scheme 是 dynamic 还是 static，都会正常收集 KV 统计数据。scheme 的判断仅在阶段 2（量化）时生效。
- **产物**（写入 `${output_dir}`）：
  - `activation_stats.json`
  - `moe_expert_stats.json`
  - `kv_scale_multipliers.json`
  - `kv_cache_tuned_scales*.json`

#### 适用场景

- 想自己接后续量化工具，不走 `fp8_quant_with_vllm_activation.py`。
- 想单独调校 PTQ 数据集 / `num_samples` / `max_length`，再用 `run_vllm_quant_for_Hy3.sh --skip-calibrate` 复用结果。
- Debug 用 `--skip-weight-loading` 跑 dummy 权重，快速验证 hook 注册流程。

---

### 4. `run_kvcache_calibrate_for_Hy3.sh` — 仅校准 KV-cache（轻量）

**功能**：只校准 KV-cache（K/V min/max），不做 weight / activation / MoE 统计。

- **入口**：`tools/kvcache/run_kvcache_calibrate.py`

#### 关键差异（与 `run_vllm_calibrate_for_Hy3.sh` 对比）

| 维度 | `run_kvcache_calibrate_for_Hy3.sh` | `run_vllm_calibrate_for_Hy3.sh` |
| --- | --- | --- |
| MoE / Linear 钩子 | 故意 **NOT** 设置 `VLLM_MOE_COLLECT_STATS`，完全跳过，启动更快、CPU 内存占用更低 | 全开 |
| KV 搜索范围 | `[0.4, 8.0]`，`num_steps=50`（更窄、更聚焦） | `[0.8, 16.0]` |
| 默认开关 | `--per-head` + `--search-kv-scale` | `--kv-granularity per-head` + `--search-kv-scale` |
| 产物文件名 | 单独写 `activation_stats_per_head.json`（不再合并到 `activation_stats.json`），便于做 KV-only 实验对比 | 合并写入 `activation_stats.json` |

#### 适用场景

- 已有 W8A8 量化模型，想单独研究 / 调优 KV scale。
- 多组 KV 搜索参数对比实验，节省"无关"前向计算。

---

### 5. `tools/kvcache/replace_kv_scales.py` — KV scale 离线替换器

**功能**：把上述任一校准脚本产出的 `kv_cache_tuned_scales*.json` 写回到已量化 FP8 模型的 `kv_cache_scales.safetensors`，并同步更新该模型 `config.json` 中的 `attn_quant_config`。

- **入口**：`tools/kvcache/replace_kv_scales.py`（详见 [`tools/kvcache/README.md`](../../tools/kvcache/README.md)）
- **支持粒度**：`per-tensor`、`per-head`（默认 `per-head`），由 `--granularity` 切换；JSON 布局自动匹配。
- **典型用法**：
  ```bash
  # per-head：把 kvcache 校准产物写回到现有 FP8 模型
  python3 tools/kvcache/replace_kv_scales.py \
      --granularity per-head \
      --json   ${stats_dir}/kv_cache_tuned_scales_per_head.json \
      --src    ${fp8_path}/kv_cache_scales.safetensors \
      --output ${fp8_path}/kv_cache_scales.safetensors \
      --num-kv-heads 8
  ```
  省略 `--output` 时会原地覆盖 `--src`，并自动保留 `*.bak` 备份。

#### 适用场景

- 已经量化好的 FP8 模型，只想刷一组新的 KV scale，不愿重跑 `fp8_quant_with_vllm_activation.py`。
- A/B 对比不同搜索范围（multiplier / num_steps）下的 KV scale，对**同一个**底层 FP8 模型快速热替换。

---

## 三、GLM-5 系列脚本（GLM-5.1-MoE 等 GLM-5 模型）

GLM-5 统一入口同时支持 **FP8-blockwise** 和 **BF16** 两种源模型。通过环境变量 `GLM5_SOURCE_FORMAT` 选择分支：

- `fp8`（默认）：vLLM MoE 激活校准 → routed experts NVFP4 + dense FP8 ue8m0。
- `bf16`：先执行 vLLM activation/MoE 校准，再通过 `BF16_QUANT_METHOD=weight_only|gptq` 生成一种 NVFP4 MoE 权重，最后与原 BF16 checkpoint 中的其他权重合并。

| 脚本 | 用途 | 入口 |
| --- | --- | --- |
| [`run_vllm_quant_for_glm5.sh`](./run_vllm_quant_for_glm5.sh) | ★ GLM-5 统一量化流水线，支持 FP8 和 BF16 源模型 | `tools/run_vllm_calibrate.py` + `tools/glm5_nvfp4_weight_only_blockwise.py` + `tools/run.py` + `tools/merge_nvfp4.py` |

---

### 1. `run_vllm_quant_for_glm5.sh` ★推荐的"一键流水线"

#### 选择源模型格式

`GLM5_SOURCE_FORMAT` 是流水线的主分支变量，允许值为 `fp8` 或 `bf16`，默认保持原行为使用 `fp8`：

```bash
# FP8-blockwise 源模型，默认分支
GLM5_SOURCE_FORMAT=fp8 \
SOURCE_MODEL_PATH=/path/to/GLM-5.1-FP8 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh

# BF16 源模型，选择 NVFP4 weight-only
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=weight_only \
SOURCE_MODEL_PATH=/path/to/GLM-5.1-BF16 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh

# BF16 源模型，选择 NVFP4-GPTQ
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=gptq \
SOURCE_MODEL_PATH=/path/to/GLM-5.1-BF16 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh
```

`SOURCE_MODEL_PATH` 是可选覆盖项；未设置时，每个分支从对应 YAML 的 `model_path` 读取源模型。FP8 流式转换要求源 checkpoint 是本地目录。BF16 的校准/量化阶段可以使用 Hugging Face model ID，但 Stage 3 合并必须通过 `BF16_MODEL_PATH` 提供本地 BF16 checkpoint；当 `SOURCE_MODEL_PATH` 本身是本地目录时，脚本会默认将它同时用作 `BF16_MODEL_PATH`。

---

### 2. FP8 源模型分支

#### 前置条件

- 输入必须是 FP8-blockwise GLM-5 checkpoint，`config.json` 中应满足 `quantization_config.quant_method == "fp8"`。
- FP8 权重必须以 `weight` + `weight_scale_inv` 的 128×128 blockwise scale 形式存储。
- 校准阶段依赖打过 AngelSlim patch 的 vLLM；按第一节完成 Ray 集群和补丁安装。

#### FP8 Stage 1：MoE 激活校准

`tools/run_vllm_calibrate.py` 通过 vLLM 采集：

- `model.layers.{L}.mlp.experts.gate_up_proj.scale_amax`
- `model.layers.{L}.mlp.experts.down_proj.scale_amax`

FP8 scale-only 模式不会注册普通 Linear/KV hooks，因而只生成流水线需要的 `moe_expert_stats.json`，不会生成 `activation_stats.json`。MTP 校准默认关闭。

#### FP8 Stage 2：NVFP4 + FP8 ue8m0 转换

`tools/glm5_nvfp4_weight_only_blockwise.py` 逐 shard 处理 checkpoint：

| 层类型 | 输出格式 |
| --- | --- |
| routed experts 的 `gate_proj` / `up_proj` / `down_proj` | packed NVFP4 `weight` + `weight_scale` + `weight_scale_2` + 校准得到的 `input_scale` |
| Attention、dense MLP、shared experts、indexer、MTP | FP8 E4M3 `weight` + ue8m0 block `scale` |
| Router、LayerNorm、embedding、lm_head 等 | 原 BF16/FP32 权重 |

默认统一配置：

```bash
configs/glm5/ptq/nvfp4/glm5_vllm_ptq_moe_fp8.yaml
```

可通过以下环境变量覆盖：

| 变量 | 作用 |
| --- | --- |
| `FP8_PTQ_CONFIG` | 同时驱动校准和转换的 YAML |
| `FP8_STATISTICS_PATH` | 覆盖校准统计目录，同时作为 Stage 2 的统计输入目录 |
| `FP8_OUTPUT_PATH` | 覆盖最终 NVFP4 + FP8 ue8m0 输出目录 |

常用命令：

```bash
# 两阶段都执行
GLM5_SOURCE_FORMAT=fp8 bash scripts/ptq/run_vllm_quant_for_glm5.sh

# 复用已有 moe_expert_stats.json，只执行转换
GLM5_SOURCE_FORMAT=fp8 bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-calibrate

# 只执行校准
GLM5_SOURCE_FORMAT=fp8 bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-quantize

# 只跳过 FP8 转换
GLM5_SOURCE_FORMAT=fp8 bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-fp8-quantize
```

---

### 3. BF16 源模型分支

BF16 分支包含三个阶段：

1. 使用 vLLM 收集 BF16 主模型的 Linear activation 和 routed-MoE expert 统计。
2. 通过 `BF16_QUANT_METHOD` 在两种量化方法中选择一种，生成一份量化模型。
3. 使用 `tools/merge_nvfp4.py` 把 NVFP4 MoE 权重、expert input scale 和 BF16 其他权重合并成最终 Hugging Face checkpoint。

```bash
BF16_QUANT_METHOD=weight_only  # 默认
BF16_QUANT_METHOD=gptq
```

#### BF16 Stage 1：vLLM activation/MoE 校准

入口和默认配置：

```text
tools/run_vllm_calibrate.py
configs/glm5/ptq/nvfp4/glm5_vllm_calibrate_bf16.yaml
```

默认配置会：

- 用 vLLM 加载 BF16 GLM-5.1。
- 注册普通 Linear activation hooks。
- 注册 routed-MoE 的 gate/up/down 输入统计 hooks。
- 设置 `kv_granularity: none`，默认不采集 KV-cache。
- 设置 `enable_mtp: false`，默认不执行 MTP draft model 校准。
- 使用 `num_samples=64`、`max_length=2048` 进行校准。

默认统计目录为：

```text
output/glm5_bf16/statistics
```

主要产物：

```text
activation_stats.json
moe_expert_stats.json
```

Stage 2 的 NVFP4 weight-only 和 NVFP4-GPTQ 不直接读取这两个 vLLM JSON：weight-only 的 scale 从权重计算，GPTQ 使用自己的校准数据构建 Hessian。Stage 3 会读取 `moe_expert_stats.json`，计算并注入每个 expert projection 的 `input_scale`。

可通过以下变量覆盖：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `BF16_CALIB_CONFIG` | `configs/glm5/ptq/nvfp4/glm5_vllm_calibrate_bf16.yaml` | BF16 vLLM 校准配置 |
| `BF16_STATISTICS_PATH` | `${BF16_WORK_DIR}/statistics` | BF16 校准统计输出目录 |

跳过 BF16 vLLM 校准：

```bash
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=weight_only \
bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-calibrate
```

只执行 BF16 vLLM 校准、不执行量化和合并：

```bash
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=weight_only \
bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-quantize --skip-merge
```

#### BF16 Stage 2 方法一：NVFP4 weight-only

入口和默认配置：

```text
tools/run.py
configs/glm5/ptq/nvfp4/glm5_1_nvfp4_weight_only.yaml
```

- 数据无关，scale 直接从权重计算。
- 只量化 routed experts。
- Attention、router、shared experts、前置 dense MLP 和 `lm_head` 由 `ignore_layers` 排除。
- 默认 group size 为 16。

#### BF16 Stage 2 方法二：NVFP4-GPTQ

入口和默认配置：

```text
tools/run.py
configs/glm5/ptq/nvfp4/glm5_1_nvfp4_gptq.yaml
```

- 使用 GPTQ 配置中的校准数据集重新执行前向、构建 Hessian，并对 routed experts 执行 GPTQ；它不使用 Stage 1 的 vLLM JSON。
- 默认校准参数为 `max_seq_length=2048`、`num_samples=128`、`batch_size=1`。
- 配置启用了 expert parallel；`tools/run.py` 会根据可见 GPU 自动通过 torchrun 启动。
- GPTQ 直接读取原始 BF16 checkpoint，不会读取或叠加在 weight-only checkpoint 上。

#### BF16 Stage 3：合并 NVFP4 MoE 和 BF16 权重

入口：

```text
tools/merge_nvfp4.py
```

流水线等价调用：

```bash
python3 tools/merge_nvfp4.py \
    --statistics_path "${BF16_STATISTICS_PATH}" \
    --nvfp4_modelpath "${BF16_NVFP4_MODEL_PATH}" \
    --bf16_modelpath "${BF16_MODEL_PATH}" \
    --output_path "${BF16_MERGED_OUTPUT_PATH}" \
    --num_workers "${BF16_MERGE_NUM_WORKERS}"
```

Stage 3 会：

- 从 `${BF16_STATISTICS_PATH}/moe_expert_stats.json` 计算 expert `input_scale`。
- 读取 Stage 2 生成的 NVFP4 expert 权重和 scale。
- 使用原 BF16 模型配置作为最终模型配置基础。
- 从 BF16 checkpoint 补齐 Stage 2 中缺失的 MTP 层。
- 复制 tokenizer、generation config 和建模代码等辅助文件。
- 写出最终 `model.safetensors.index.json`、`config.json` 和 `hf_quant_config.json`。

默认 Stage 2 输入路径由所选量化配置和 save root 自动推导：

```text
weight_only:
  output/glm5_bf16/nvfp4_weight_only/glm5_1_nvfp4_weight_only

gptq:
  output/glm5_bf16/nvfp4_gptq/glm5_1_nvfp4_gptq
```

默认最终输出目录：

```text
weight_only:
  output/glm5_bf16/merged_weight_only

gptq:
  output/glm5_bf16/merged_gptq
```

Stage 3 要求：

- `BF16_MODEL_PATH` 必须是本地 BF16 checkpoint 目录，不能是 Hugging Face model ID。
- `${BF16_STATISTICS_PATH}/moe_expert_stats.json` 必须存在。
- `BF16_NVFP4_MODEL_PATH` 必须指向已经生成的 Stage 2 checkpoint。

常用命令：

```bash
# 生成 NVFP4 weight-only（BF16_QUANT_METHOD 默认值）
GLM5_SOURCE_FORMAT=bf16 \
SOURCE_MODEL_PATH=/path/to/GLM-5.1-BF16 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh

# 显式选择 NVFP4 weight-only
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=weight_only \
SOURCE_MODEL_PATH=/path/to/GLM-5.1-BF16 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh

# 选择 NVFP4-GPTQ
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=gptq \
SOURCE_MODEL_PATH=/path/to/GLM-5.1-BF16 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh

# 复用已有 Stage 2 checkpoint，只执行校准和合并
GLM5_SOURCE_FORMAT=bf16 \
BF16_NVFP4_MODEL_PATH=/path/to/existing/nvfp4_model \
BF16_MODEL_PATH=/path/to/GLM-5.1-BF16 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-quantize

# 复用已有统计，只执行所选量化和合并
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=gptq \
BF16_STATISTICS_PATH=/path/to/existing/statistics \
BF16_MODEL_PATH=/path/to/GLM-5.1-BF16 \
bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-calibrate

# 只执行校准和量化，不合并
GLM5_SOURCE_FORMAT=bf16 \
BF16_QUANT_METHOD=weight_only \
bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-merge
```

BF16 分支可配置变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `BF16_QUANT_METHOD` | `weight_only` | BF16 量化方法：`weight_only` 或 `gptq` |
| `BF16_CALIB_CONFIG` | `configs/glm5/ptq/nvfp4/glm5_vllm_calibrate_bf16.yaml` | BF16 vLLM 校准配置 |
| `BF16_STATISTICS_PATH` | `${BF16_WORK_DIR}/statistics` | BF16 vLLM 校准统计目录 |
| `BF16_MODEL_PATH` | `${SOURCE_MODEL_PATH}` | Stage 3 使用的本地 BF16 checkpoint |
| `NVFP4_WEIGHT_ONLY_CONFIG` | `configs/glm5/ptq/nvfp4/glm5_1_nvfp4_weight_only.yaml` | weight-only 配置 |
| `NVFP4_GPTQ_CONFIG` | `configs/glm5/ptq/nvfp4/glm5_1_nvfp4_gptq.yaml` | GPTQ 配置，包括校准数据集 |
| `BF16_WORK_DIR` | `output/glm5_bf16` | BF16 输出工作根目录 |
| `BF16_WEIGHT_ONLY_SAVE_ROOT` | `${BF16_WORK_DIR}/nvfp4_weight_only` | 传给 weight-only 的 `--save-path` |
| `BF16_GPTQ_SAVE_ROOT` | `${BF16_WORK_DIR}/nvfp4_gptq` | 传给 GPTQ 的 `--save-path` |
| `BF16_NVFP4_MODEL_PATH` | 根据所选配置和 save root 自动推导 | Stage 3 读取的 NVFP4 checkpoint |
| `BF16_MERGED_OUTPUT_PATH` | `${BF16_WORK_DIR}/merged_${BF16_QUANT_METHOD}` | 最终合并模型目录 |
| `BF16_MERGE_NUM_WORKERS` | `8` | Stage 3 shard 合并进程数 |

`tools/run.py` 会在所选方法的 `--save-path` 后追加 YAML 文件名，因此默认实际输出目录分别为：

```text
output/glm5_bf16/nvfp4_weight_only/glm5_1_nvfp4_weight_only
output/glm5_bf16/nvfp4_gptq/glm5_1_nvfp4_gptq
```

运行前务必检查 GPTQ YAML 中的：

```yaml
dataset:
  data_path: ./dataset/sharegpt_gpt4/sharegpt_gpt4_256.jsonl
```

确保校准数据文件存在，或通过自定义 `NVFP4_GPTQ_CONFIG` 指向另一份配置。

---

### 4. 通用变量和开关

| 变量/开关 | 适用分支 | 说明 |
| --- | --- | --- |
| `GLM5_SOURCE_FORMAT=fp8\|bf16` | 全部 | 选择源模型格式，默认 `fp8` |
| `BF16_QUANT_METHOD=weight_only\|gptq` | BF16 | 选择唯一执行的 BF16 量化方法，默认 `weight_only` |
| `SOURCE_MODEL_PATH` | 全部 | 覆盖所选 YAML 中的源模型路径 |
| `LOG_DIR` | 全部 | 日志目录，默认 `logs` |
| `--skip-calibrate` | 全部 | 跳过当前源模型分支的 vLLM 校准 |
| `--skip-fp8-quantize` | FP8 | 跳过 FP8 checkpoint 转换 |
| `--skip-quantize` | 全部 | 跳过当前分支已选择的量化阶段；BF16 可继续复用已有 Stage 2 checkpoint 做合并 |
| `--skip-merge` | BF16 | 跳过 NVFP4/BF16 最终合并 |

```bash
bash scripts/ptq/run_vllm_quant_for_glm5.sh --help
```

脚本开启 `set -euo pipefail`，任一实际执行的阶段失败都会立即中断。
