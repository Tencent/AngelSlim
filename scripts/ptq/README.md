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

下面 6 个脚本共享同一套 vLLM 运行时环境（chunked prefill / FlashInfer attention / mp distributed executor / fused MoE 等），区别在于产出物不同。

| 脚本 | 用途 | 入口 |
| --- | --- | --- |
| [`run_vllm_quant_for_HY3.sh`](./run_vllm_quant_for_HY3.sh) | ★ 推荐的"一键流水线"：校准 + 量化 | `tools/run_vllm_calibrate.py` + `tools/fp8_quant_with_vllm_activation.py` |
| [`run_vllm_calibrate_for_HY3.sh`](./run_vllm_calibrate_for_HY3.sh) | 仅 W8A8C8 联合校准 | `tools/run_vllm_calibrate.py` |
| [`run_kvcache_calibrate_for_HY3.sh`](./run_kvcache_calibrate_for_HY3.sh) | 仅 KV-cache 校准（轻量） | `tools/kvcache/run_kvcache_calibrate.py` |
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

### 2. `run_vllm_calibrate_for_Hy3.sh` — 一键脚本里的"阶段 1"独立版

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

### 3. `run_kvcache_calibrate_for_Hy3.sh` — 仅校准 KV-cache（轻量）

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

### 4. `tools/kvcache/replace_kv_scales.py` — KV scale 离线替换器

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

GLM-5 走的是与 Hy3 不同的一条量化链路：**输入是 FP8-blockwise 的 GLM-5 checkpoint**，产出是 **MoE 专家 NVFP4 + Dense FP8 ue8m0** 的混合精度模型（对齐官方参考实现 `GLM-5.1-moe_nvfp4-dense_fp8_ue8m0`）。校准阶段只对 MoE 的**路由专家**采集激活统计，其他模块（attention / dense MLP / shared_experts / router gate / lm_head / MTP 层）不做激活校准。

| 脚本 | 用途 | 入口 |
| --- | --- | --- |
| [`run_vllm_quant_for_glm5.sh`](./run_vllm_quant_for_glm5.sh) | ★ 一键流水线：MoE 专家激活校准 + NVFP4 权重量化 | `tools/run_vllm_calibrate.py` + `tools/glm5_nvfp4_weight_only_blockwise.py` |

---

### 1. `run_vllm_quant_for_glm5.sh` ★推荐的"一键流水线"

**功能**：FP8-blockwise 的 GLM-5 模型 → vLLM MoE 专家激活校准 → NVFP4（experts）+ FP8 ue8m0（dense）HF safetensors，全流程一次完成。

#### 硬性前置条件

- **输入模型必须是 FP8-blockwise 的 GLM-5 checkpoint**（例如官方 GLM-5.1-FP8 发布版），即 `config.json` 中 `quantization_config.quant_method == "fp8"`，且权重以 `weight` + `weight_scale_inv`（128×128 blockwise scale）配对存储。
  - 阶段 2 会显式 assert 这一点，传入 BF16/FP16 或其它 quant_method 的模型会直接报错。
  - 阶段 2 的 dense 层 ue8m0 重量化是从 "FP8 dequant 回 BF16 → 再量化到 FP8 ue8m0" 走一次的，输入不是 blockwise FP8 时无法复用现有 128×128 scale 布局。
- **vLLM 版本 + 补丁**：与 Hy3 完全一致，先按第一节 [环境准备](#一环境准备运行校准脚本前必须完成) 拉起 Ray 集群并给每个节点的 vLLM 打上 AngelSlim patch（`bash tools/vllm_patch/install.sh install`），补丁提供的 `VLLM_MOE_COLLECT_PER_EXPERT_STATS` 钩子即使权重是 FP8 也能正常触发（vLLM 内部仍走 `FusedMoE` 前向）。

#### 阶段 1：MoE 路由专家激活校准（`tools/run_vllm_calibrate.py`）

- 用 vLLM 加载 **FP8 GLM-5** 模型，在 PTQ 数据集上跑前向，**仅注册 MoE 路由专家的激活采集钩子**：
  - 采集 `model.layers.{L}.mlp.experts.gate_up_proj.scale_amax`（gate_proj / up_proj 共享此 scale）
  - 采集 `model.layers.{L}.mlp.experts.down_proj.scale_amax`
  - **不**采集 attention / dense MLP / shared_experts / router gate 的激活；**不**采集 KV-cache 统计（YAML 里未开启 `--search-kv-scale`，也未指定 `kv_granularity`，保持轻量）。
- 相关环境变量（脚本已内置）：
  ```bash
  VLLM_MOE_COLLECT_STATS=1
  VLLM_MOE_COLLECT_PER_EXPERT_STATS=1
  VLLM_MOE_COLLECT_STATS_VERBOSE=0
  ```
- 产物（写入 YAML 中 `output_dir`）：
  - `moe_expert_stats.json` — 每个 MoE 层路由专家的 `scale_amax`（NVFP4 全局 input_scale 就是 `scale_amax_max / 6`，阶段 2 自动换算）
  - `activation_stats.json` — 会同步生成，但 GLM-5 流水线**不消费**这个文件（阶段 2 只读 `moe_expert_stats.json`）

> ⚠️ GLM-5.1 的 MTP（Multi-Token Prediction）层在 YAML 里通过 `enable_mtp: false` 关闭：MoE-only 校准不需要走 draft path，MTP 层最终在阶段 2 会保持在 dense FP8 分支。

#### 阶段 2：NVFP4 权重-only 重量化（`tools/glm5_nvfp4_weight_only_blockwise.py`）

流式（shard-by-shard, `num_workers` 并行）读取源 FP8 safetensors，分类处理：

| 层类型 | 处理方式 | 输出字段 |
| --- | --- | --- |
| MoE 路由专家（layer ≥ `first_k_dense_replace` 且 `< num_hidden_layers`）的 `gate_proj` / `up_proj` / `down_proj` | FP8 dequant → BF16 → **NVFP4 weight-only 量化**（block_size=16），再注入阶段 1 得到的 per-tensor `input_scale` | `weight`（uint8 packed FP4）+ `weight_scale`（fp8 e4m3, per-block）+ `weight_scale_2`（fp32 scalar）+ `input_scale`（fp32 scalar） |
| Dense FP8 层（attention / 前 `first_k_dense_replace` 层的 dense MLP / `shared_experts` / indexer / MTP 层） | FP8 dequant → BF16 → **重量化为 FP8 E4M3 + ue8m0 per-block scale**（block_size=128） | `weight`（fp8 e4m3）+ `scale`（uint8 ue8m0, per-block） |
| 其它（router gate / layernorms / embed / lm_head / e_score_correction_bias 等） | 原样透传 | 保持 BF16 / FP32 |

产出目录（YAML `output_nvfp4_hf_path`）额外会写入：
- 重写后的 `config.json` — 保留源 FP8 的 `quantization_config`，追加 `scale_fmt: "ue8m0"`。
- `hf_quant_config.json.nvfp4` — modelopt 风格的 NVFP4 sidecar，标注 `quant_algo=NVFP4`、`group_size`、`exclude_modules`（dense / attention / shared_experts / MTP 层的 glob 列表）。切换到 NVFP4 推理后端时，将该文件重命名为 `hf_quant_config.json` 即可生效。
- `model.safetensors.index.json` — 合并所有 shard 的 weight_map。
- tokenizer / generation_config / 建模代码 (`*.py`, `*.json`, `*.md`, `*.txt`, `*.jinja`) 从源目录 copy 过来。

#### 配置

校准和量化共享同一份 YAML：[`configs/glm5/ptq/fp8/glm5_vllm_ptq_moe.yaml`](../../configs/glm5/ptq/fp8/glm5_vllm_ptq_moe.yaml)。关键字段：

```yaml
model_path: /path/to/GLM-5.1-FP8            # 必须是 FP8-blockwise 的 GLM-5 源模型
ptq_data_path: /path/to/calibration.jsonl   # 校准数据集
output_dir: /path/to/statistics             # 阶段 1 产物（moe_expert_stats.json）
output_nvfp4_hf_path: /path/to/nvfp4_model  # 阶段 2 产物：NVFP4+FP8 ue8m0 混合模型

tp_size: 16                                  # 单机 8 卡填 8；跨节点 Ray 16 卡填 16
dp_size: 1
batch_size: 4
num_samples: 64
max_length: 2048
distributed_executor_backend: ray            # ray | mp
enable_mtp: false                            # MoE-only 校准，不启用 MTP

nvfp4_block_size: 16                         # NVFP4 group size
fp8_block_size: 128                          # Dense FP8 ue8m0 block size
num_workers: 8                               # Stage 2 shard 并行度
use_gpu: true                                # true => triton on GPU；false => CPU-only
no_quantize_dense_fp8: false                 # true => 退化到 legacy：dense 层 dequant 回 BF16
```

路径回退规则（阶段 2 只需要在 YAML 里配一次）：
- `input_path` ← `model_path`（FP8 源 checkpoint）
- `output_path` ← `output_nvfp4_hf_path`
- `moe_stats_json` ← `${output_dir}/moe_expert_stats.json`

#### CLI 开关

```bash
bash scripts/ptq/run_vllm_quant_for_glm5.sh                    # 两阶段都跑
bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-calibrate   # 仅量化（复用已有 moe_expert_stats.json）
bash scripts/ptq/run_vllm_quant_for_glm5.sh --skip-quantize    # 仅校准
bash scripts/ptq/run_vllm_quant_for_glm5.sh --help             # 打印用法
```

> 脚本开启 `set -euo pipefail`，任一阶段失败将立即中断。日志会 tee 到 `logs/run_vllm_quant_glm5-{calibrate,quantize}.log`。

#### 与 Hy3 流水线的关键差异

| 维度 | GLM-5 | Hy3 |
| --- | --- | --- |
| 输入模型精度 | **必须** FP8-blockwise（`quant_method=fp8` + `weight_scale_inv`） | BF16 |
| 校准范围 | **仅 MoE 路由专家**（gate_up_proj / down_proj） | 全模型 W8A8C8（attention + MoE + KV-cache） |
| 校准产物消费 | 只用 `moe_expert_stats.json` | `activation_stats.json` + `moe_expert_stats.json` + KV scale |
| 输出格式 | MoE 专家 NVFP4 + Dense FP8 ue8m0（混合精度） | 全模型 per-tensor FP8 |
| KV-cache 处理 | 校准 + 量化均不处理 | 校准 + 量化均可配置（per-tensor / per-head） |
| 量化脚本内存 | 流式 shard-by-shard，**不需要**把整模型载入显存 | 依赖 `AutoModelForCausalLM.from_pretrained`，需要能装下模型 |
