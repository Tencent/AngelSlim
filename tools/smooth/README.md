# SmoothQuant PTQ 量化脚本说明

本文档介绍 **SmoothQuant** 两阶段离线量化流水线的使用、配置和原理。SmoothQuant 是一种后训练量化（PTQ）技术，通过平滑激活和权重分布，提升低比特量化的精度。

> ⚠️ **前置条件**：本节所有脚本均依赖 [一、环境准备](#环境准备必要条件) 中的 Ray 集群和 vLLM 补丁。**请先完成环境准备再执行 Smooth 脚本。**

---

## 目录

1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [两阶段流水线](#两阶段流水线)
4. [环境准备（必要条件）](#环境准备必要条件)
5. [脚本使用](#脚本使用)
6. [配置管理](#配置管理)
7. [Alpha 网格搜索](#alpha-网格搜索)
8. [工作流示例](#工作流示例)
9. [数据格式](#数据格式)
10. [集成与扩展](#集成与扩展)
11. [故障排查](#故障排查)
12. [参考资源](#参考资源)

---

## 快速开始

### 一键执行完整流水线

```bash
cd /path/to/AngelSlim  # 必须在仓库根目录
bash scripts/ptq/run_smooth_for_HY3.sh
```

**分步执行**（调试或复用统计时）：

```bash
# 仅运行 Phase 1（收集统计，需时较长）
bash scripts/ptq/run_smooth_calibrate_for_HY3.sh

# 仅运行 Phase 2（应用变换，较快）
bash scripts/ptq/run_smooth_convert_for_HY3.sh

# Phase 1 完成后，复用其输出的统计再执行 Phase 2
bash scripts/ptq/run_smooth_for_HY3.sh --skip-calibrate
```

---

## 核心概念

### SmoothQuant 算法

**目标**：解决 PTQ 中"离群值"导致的量化误差。

**原理**：通过对激活和权重分布进行缩放变换，使其更"均匀"（smooth），从而提高低比特量化精度。

**三类变换**：

| 变换类型 | 应用层 | 作用 |
| --- | --- | --- |
| **QK Smooth** | Attention Q/K 投影层 | 平滑注意力查询和键的激活分布 |
| **VO Smooth** | Attention V/O 投影层 | 平滑 Value 输入和 Output 投影的激活 |
| **Down Smooth** | MLP Down 投影层 | 平滑前馈网络激活 |

### 关键术语

- **Alpha (α)**：控制平滑强度的参数，范围通常 `[0.3, 1.0]`
  - α = 0：无平滑（完全依赖权重）
  - α = 1：完全平滑（完全依赖激活）
  - 最优 α 因层而异，通过网格搜索获得

- **EMA（指数移动平均）**：在线追踪激活统计的技术，避免全缓存
  - `ema_momentum`：动量系数，通常 0.9
  - 更新公式：`new_stat = momentum * old_stat + (1 - momentum) * current_stat`

- **Per-channel 统计**：为每个输出通道独立计算 abs_max 和 EMA 值

---

## 两阶段流水线

### 阶段 1：统计收集与 Alpha 搜索

**任务**：
1. 在 PTQ 数据集上前向推理，收集激活值统计
2. 计算每层每通道的 abs_max 和 EMA
3. 执行可选的 Alpha 网格搜索，找到最优平滑参数

**入口**：`tools/smooth/run_vllm_smooth.py`

**输出**：
- `smooth_stats.json` — 所有层的 abs_max / EMA 统计
- `smooth_alpha_search.json`（可选）— 每层搜索得到的最优 alpha

**关键文件**：
- `angelslim/compressor/quant/core/vllm_calibrate_utils/hooks.py` — 统计钩子
- `angelslim/compressor/quant/core/vllm_calibrate_utils/search.py` — Alpha 搜索实现

### 阶段 2：离线权重变换

**任务**：
1. 读取阶段 1 产出的统计文件
2. 根据 alpha 值和统计，对权重进行缩放变换
3. 输出平滑后的权重到 HuggingFace safetensors 格式

**入口**：`tools/smooth/convert_smooth_weights.py`

**输出**：
- `{save_path}/` — 平滑后的模型（safetensors 格式）

**关键文件**：
- `angelslim/compressor/quant/modules/smooth/smooth.py` — 核心变换实现
- `tools/smooth/convert_smooth_weights.py` — 权重处理和转换逻辑

---

## 环境准备（必要条件）

### 1. Ray 集群与 vLLM 补丁

**必须先完成** `scripts/ptq/README.md` 中"一、环境准备"的两步：

1. **搭建 Ray 集群**（2 节点 × 8 卡 = 16 卡）
2. **给 vLLM 打 AngelSlim 补丁**

```bash
# 检查补丁是否已安装
bash tools/vllm_patch/install.sh check

# 若未安装，执行一键安装（在每个 Ray 节点上运行）
bash tools/vllm_patch/install.sh install
```

### 2. 环境变量

Smooth 脚本会自动设置以下环境变量，无需手动配置：

```bash
export VLLM_MOE_COLLECT_SMOOTH_STATS=1          # 启用 Smooth 统计收集
export VLLM_MOE_COLLECT_ALPHA_SEARCH=1          # 启用 Alpha 搜索
export VLLM_ALLOW_INSECURE_SERIALIZATION=1      # 允许不安全序列化
export PYTHONPATH=<AngelSlim_root>              # 确保 angelslim 包可导入
```

---

## 脚本使用

### 1. `run_smooth_for_HY3.sh` ★ 推荐的一键流水线

**功能**：自动执行 Phase 1 + Phase 2，完成完整的 Smooth 量化流程。

**用法**：

```bash
# 标准用法：执行 Phase 1 + Phase 2
bash scripts/ptq/run_smooth_for_HY3.sh

# 仅执行 Phase 2（复用已有的 smooth_stats.json）
bash scripts/ptq/run_smooth_for_HY3.sh --skip-calibrate

# 仅执行 Phase 1（调试或仅收集统计）
bash scripts/ptq/run_smooth_for_HY3.sh --skip-convert

# 查看帮助
bash scripts/ptq/run_smooth_for_HY3.sh --help
```

**运行日志示例**：

```
========================================
[Phase 1] Smooth Stats Calibration
========================================
[2025-05-26 10:15:23] Loading model: /path/to/model ...
[2025-05-26 10:15:45] Setting up hooks for smooth stats collection ...
[2025-05-26 10:20:18] Processing 512 samples, max_length=16384 ...
[2025-05-26 10:35:42] Phase 1 complete. Output: /path/to/output_dir/smooth_stats.json
[2025-05-26 10:35:43] Starting alpha search (enable_alpha_search=true) ...
[2025-05-26 10:45:20] Alpha search complete: /path/to/output_dir/smooth_alpha_search.json
========================================
[Phase 2] Offline Weight Conversion
========================================
[2025-05-26 10:45:21] Reading stats from: /path/to/output_dir/smooth_stats.json
[2025-05-26 10:45:22] Applying QK smooth ...
[2025-05-26 10:50:15] Applying VO smooth ...
[2025-05-26 10:55:30] Applying down_proj smooth ...
[2025-05-26 11:05:45] Saving smoothed model to: /path/to/save_path/
========================================
Done.
========================================
```

### 2. `run_smooth_calibrate_for_HY3.sh` — 仅 Phase 1

**功能**：独立运行 Phase 1（统计收集 + 可选 Alpha 搜索）。

**用法**：

```bash
bash scripts/ptq/run_smooth_calibrate_for_HY3.sh
```

**何时使用**：

- 需要多次调整 `alpha_min`、`alpha_max`、`alpha_steps` 等搜索参数，复用统计结果
- 需要手工检查或修改 `smooth_stats.json` 或 `smooth_alpha_search.json`
- 需要与其他工具链集成（如自定义量化后处理）

### 3. `run_smooth_convert_for_HY3.sh` — 仅 Phase 2

**功能**：独立运行 Phase 2（离线权重变换）。

**用法**：

```bash
bash scripts/ptq/run_smooth_convert_for_HY3.sh
```

**前置条件**：必须已有 `output_dir` 下的 `smooth_stats.json` 文件（通常来自阶段 1）。

**何时使用**：

- 已有来自阶段 1 的 `smooth_stats.json`，想测试不同的 `alpha_qk` / `alpha_vo` / `alpha_down` 值
- 阶段 1 成功完成，阶段 2 因故障中断，需要单独重跑
- 线下编辑 `smooth_stats.json` 进行手工调试

---

## 配置管理

### 配置文件位置

所有 Smooth 脚本共享同一套 YAML 配置，默认为：

```
configs/Hy3/ptq/fp8/Hy3_smooth.yaml
```

### 关键配置项

#### 路径配置

```yaml
# Phase 1 读取；Phase 2 读取和输出 stats 的位置
model_path: /path/to/hunyuan/model

# Phase 1 读取的 PTQ 校准数据集
ptq_data_path: /path/to/calibration_data.jsonl

# Phase 1 输出 smooth_stats.json 和 smooth_alpha_search.json 的目录
output_dir: /path/to/smooth_output

# Phase 2 输出平滑权重的目录
save_path: /path/to/smoothed_model
```

#### Phase 1 运行时配置

```yaml
# vLLM 分布式执行参数
tp_size: 16                                    # Tensor Parallel 卡数
batch_size: 4                                  # 前向 batch 大小
num_samples: 512                               # PTQ 数据集采样数
max_length: 16384                              # 最大序列长度
distributed_executor_backend: ray              # 使用 Ray 分布式执行
skip_weight_loading: false                     # 是否跳过权重加载（调试用）

# 统计收集范围
collect_attn: true                             # 收集 Q/K/V/O 激活统计
collect_down_proj: true                        # 收集 MLP 激活统计
collect_moe: true                              # 收集 MoE expert 统计
ema_momentum: 0.9                              # EMA 指数移动平均动量

verbose: false                                 # 是否输出详细日志
```

#### Phase 2 变换配置

```yaml
# 是否启用各类变换
smooth_qk: true                                # 启用 Q/K 平滑
smooth_vo: true                                # 启用 V/O 平滑
smooth_down: true                              # 启用 Down 平滑

# 手工指定的 alpha 值（未进行网格搜索时的默认值）
alpha_qk: 0.6                                  # Q/K 平滑 alpha
alpha_vo: 0.5                                  # V/O 平滑 alpha
alpha_down: 0.6                                # Down 平滑 alpha（无搜索结果时使用）

use_ema: false                                 # 是否使用 EMA 统计（通常 false）
device: cpu                                    # 权重处理设备
dtype: auto                                    # 权重数据类型（auto 自动推断）
```

#### Alpha 网格搜索配置（Phase 1 执行，Phase 2 应用）

```yaml
# 启用 Alpha 网格搜索
enable_alpha_search: true

# 搜索参数
num_search_samples: 16                         # 搜索用的 calibration 样本数
search_max_length: 8192                        # 搜索用的最大序列长度
alpha_min: 0.3                                 # 搜索下界
alpha_max: 1.0                                 # 搜索上界
alpha_steps: 8                                 # 搜索步数（8 步 => 采样 9 个 alpha 值）

# 量化配置（用于评估搜索得到的 alpha）
alpha_act_quant_method: per_token              # 激活量化粒度
alpha_act_quant_type: int8                     # 激活量化类型
alpha_weight_quant_method: per_channel         # 权重量化粒度
alpha_weight_quant_type: int8                  # 权重量化类型
alpha_weight_quant_bits: 8                     # 权重位宽
alpha_weight_group_size: 128                   # 权重分组大小
alpha_max_tokens: 4096                         # 搜索过程中的最大 token 数
alpha_use_ema: false                           # 搜索中是否使用 EMA
alpha_smooth_search_mode: default              # 搜索模式
```

### 配置示例

#### 典型 HY3 配置

```yaml
# configs/Hy3/ptq/fp8/Hy3_smooth.yaml
model_path: /apdcephfs_zwfy14/share_300532381/gavinlee/share_model/one-agent/hunyuan/yonewu/.../ckpt/global_step_hf
ptq_data_path: /cfs_cloud_code/gavinlee/work/code-RL/data/0521-oneagent/sampled_3000_for_quant_shuf.jsonl
output_dir: /apdcephfs_zwfy14/share_300532381/.../ckpt/stat_debug
save_path: /apdcephfs_zwfy14/share_300532381/.../ckpt/smooth_model_debug2

tp_size: 16
batch_size: 4
num_samples: 512
max_length: 16384
distributed_executor_backend: ray

collect_attn: true
collect_down_proj: true
collect_moe: true
ema_momentum: 0.9

smooth_qk: true
smooth_vo: true
smooth_down: true
alpha_qk: 0.6
alpha_vo: 0.5
alpha_down: 0.6

enable_alpha_search: true
num_search_samples: 16
alpha_min: 0.3
alpha_max: 1.0
alpha_steps: 8
```

#### 自定义配置（Qwen3 示例）

```yaml
# configs/qwen3/ptq/smooth_int8/qwen3-8b_int8_dynamic_smooth.yaml
model_path: /path/to/qwen3-8b

# 数据集
ptq_data_path: /path/to/calibration.jsonl

# 统计输出目录
output_dir: /path/to/smooth_stats
save_path: /path/to/qwen3_smoothed

# 运行时
tp_size: 8                                     # Qwen3-8B 用较少卡数
batch_size: 8
num_samples: 256                               # 更少样本，更快迭代
max_length: 4096

# 平滑配置
smooth_qk: true
smooth_vo: true
smooth_down: true
alpha_qk: 0.5
alpha_vo: 0.4
alpha_down: 0.5

# Alpha 搜索
enable_alpha_search: true
alpha_steps: 6                                 # 更少步数加快搜索
```

---

## Alpha 网格搜索

### 原理

Alpha 网格搜索的目标是为每一层自动找到最优的 α 值，使得经过 Smooth 变换后的权重，在量化时的误差最小。

### 搜索流程（Phase 1 中执行）

1. **采样配置**：从 PTQ 数据集中采样 `num_search_samples` 个样本
2. **Alpha 网格**：在 `[alpha_min, alpha_max]` 范围内均匀采样 `alpha_steps + 1` 个 alpha 值
3. **逐层评估**：对每一层，尝试所有 alpha 值
4. **量化评估**：对每个 alpha，进行模拟量化，计算量化后的 MSE / 相对误差
5. **记录最优值**：选择误差最小的 alpha，写入 `smooth_alpha_search.json`

### 输出格式

```json
{
  "layer_0": {
    "attn.q_proj": { "best_alpha": 0.6, "mse": 0.00123 },
    "attn.k_proj": { "best_alpha": 0.65, "mse": 0.00118 },
    "attn.v_proj": { "best_alpha": 0.55, "mse": 0.00145 },
    "attn.o_proj": { "best_alpha": 0.5, "mse": 0.00162 }
  },
  "layer_1": {
    ...
  }
}
```

### 快速迭代建议

**若搜索过程过长**（通常 5-10 小时），可尝试：

```yaml
alpha_min: 0.4              # 缩小搜索范围（从 0.3 改为 0.4）
alpha_max: 0.9              # （从 1.0 改为 0.9）
alpha_steps: 4              # 减少搜索步数（从 8 改为 4）
num_search_samples: 8       # 减少评估用的样本数（从 16 改为 8）
search_max_length: 4096     # 减少最大序列长度（从 8192 改为 4096）
```

---

## 工作流示例

### 示例 1：完整流程（一键执行）

```bash
cd /path/to/AngelSlim

# 编辑配置文件，指定模型、数据、输出路径
vim configs/Hy3/ptq/fp8/Hy3_smooth.yaml

# 一键执行 Phase 1 + Phase 2（约 4-6 小时）
bash scripts/ptq/run_smooth_for_HY3.sh

# 查看输出
ls -la $(grep "save_path:" configs/Hy3/ptq/fp8/Hy3_smooth.yaml | cut -d' ' -f2)
```

### 示例 2：调试 Alpha 搜索

```bash
# 1. 先跑一次快速搜索（较小搜索范围）
cat > /tmp/smooth_debug.yaml << 'YAML'
# 复制 hy3_smooth.yaml，修改以下参数
alpha_steps: 4
num_search_samples: 8
search_max_length: 4096
YAML

# 2. 运行 Phase 1（仅统计收集 + 搜索）
cp configs/Hy3/ptq/fp8/Hy3_smooth.yaml configs/Hy3/ptq/fp8/Hy3_smooth_debug.yaml
vim configs/Hy3/ptq/fp8/Hy3_smooth_debug.yaml  # 编辑搜索参数

python3 tools/smooth/run_vllm_smooth.py -c configs/Hy3/ptq/fp8/Hy3_smooth_debug.yaml

# 3. 检查搜索结果
cat output_dir/smooth_alpha_search.json | python3 -m json.tool | head -50

# 4. 若满意，再用更精细参数重新搜索
vim configs/Hy3/ptq/fp8/Hy3_smooth_debug.yaml
python3 tools/smooth/run_vllm_smooth.py -c configs/Hy3/ptq/fp8/Hy3_smooth_debug.yaml

# 5. 查看最终搜索结果
cat output_dir/smooth_alpha_search.json | python3 -m json.tool
```

### 示例 3：复用统计进行多次 Phase 2

```bash
# Phase 1 已完成，假设输出在 /data/smooth_stats

# 方案 A：测试不同的 alpha_qk 值
for alpha in 0.4 0.5 0.6 0.7; do
    sed -i "s/alpha_qk: .*/alpha_qk: $alpha/" configs/Hy3/ptq/fp8/Hy3_smooth.yaml
    sed -i "s|save_path: .*|save_path: /data/smooth_model_alpha_qk_${alpha}|" configs/Hy3/ptq/fp8/Hy3_smooth.yaml
    bash scripts/ptq/run_smooth_convert_for_HY3.sh
done

# 方案 B：使用网格搜索结果（自动应用最优 alpha）
# 若 smooth_alpha_search.json 存在，Phase 2 会自动使用其中的最优 alpha，
# 忽略手工指定的 alpha_qk / alpha_vo / alpha_down
```

### 示例 4：集成到现有 PTQ 流水线

```bash
# 1. 若已有 W8A8C8 量化的模型，先跑 Smooth 统计
bash scripts/ptq/run_smooth_calibrate_for_HY3.sh

# 2. 应用 Smooth 变换后再量化
python3 tools/fp8_quant_with_vllm_activation.py \
    --model-path /data/smooth_model \
    --stats-dir /data/smooth_output \
    --fp8-path /data/fp8_smoothed_model
```

---

## 数据格式

### Phase 1 输出：`smooth_stats.json`

```json
{
  "model.layers.0.self_attn.q_proj": {
    "absmax": [0.234, 0.456, 0.789, ...],      // per-channel absmax，长度 = 输出维度
    "ema": [0.230, 0.450, 0.780, ...]          // per-channel EMA
  },
  "model.layers.0.self_attn.k_proj": {
    "absmax": [...],
    "ema": [...]
  },
  "model.layers.0.self_attn.v_proj": {
    "absmax": [...],
    "ema": [...]
  },
  "model.layers.0.mlp.down_proj": {
    "absmax": [...],
    "ema": [...]
  },
  ...
}
```

### Phase 1 输出：`smooth_alpha_search.json`（若启用）

```json
{
  "model.layers.0.self_attn.q_proj": {
    "best_alpha": 0.65,
    "mse": 0.001234
  },
  "model.layers.0.self_attn.k_proj": {
    "best_alpha": 0.60,
    "mse": 0.001210
  },
  ...
}
```

### 变换公式（Phase 2）

#### QK Smooth

```
smooth_weight = k_absmax^alpha
q_scaled = q_proj / smooth_weight
k_scaled = k_proj * smooth_weight
```

#### VO Smooth

```
smooth_weight = attn_out_absmax^alpha / o_proj_max^(1-alpha)
v_scaled = v_proj * smooth_weight
attn_out_scaled = attn_output / smooth_weight
o_proj_scaled = o_proj / smooth_weight
```

#### Down Smooth

```
smooth_weight = down_proj_absmax^alpha / up_proj_max^(1-alpha)
mlp_act_scaled = mlp_activation / smooth_weight
down_scaled = down_proj * smooth_weight
```

---

## 集成与扩展

### 添加新模型配置

1. **创建模型特定配置**：

```bash
cp configs/Hy3/ptq/fp8/Hy3_smooth.yaml configs/my_model/ptq/my_model_smooth.yaml
```

2. **编辑配置文件**：

```yaml
# configs/my_model/ptq/my_model_smooth.yaml
model_path: /path/to/my_model
ptq_data_path: /path/to/my_calibration_data.jsonl
output_dir: /path/to/my_smooth_output
save_path: /path/to/my_smooth_model

tp_size: 8                    # 根据模型大小调整
batch_size: 8
num_samples: 256
max_length: 4096

# 其他配置项...
```

3. **创建快捷脚本**（可选）：

```bash
# scripts/ptq/run_smooth_for_my_model.sh
#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/my_model/ptq/my_model_smooth.yaml"

export VLLM_MOE_COLLECT_SMOOTH_STATS=1
export VLLM_MOE_COLLECT_ALPHA_SEARCH=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

python3 tools/smooth/run_vllm_smooth.py -c "$CONFIG" "$@" && \
python3 tools/smooth/convert_smooth_weights.py -c "$CONFIG" "$@"
```

### 自定义 Phase 2 变换

若需自定义 Phase 2 逻辑（如自己的量化后处理），可直接调用核心 API：

```python
from angelslim.compressor.quant.modules.smooth import SmoothQuant, SmoothConfig
import json

# 1. 加载统计
with open("/path/to/smooth_stats.json") as f:
    smooth_stats = json.load(f)

# 2. 创建配置
smooth_config = SmoothConfig(
    alpha=0.6,
    smooth_first_linears=True,   # Q/K
    smooth_second_linears=True   # V/O
)

# 3. 应用变换
smoother = SmoothQuant(smooth_config)
model = smoother.convert(model, smooth_stats)

# 4. 后续量化或保存...
```

### 与其他工具集成

**与 vLLM 量化集成**：

```bash
# 1. Smooth 处理
bash scripts/ptq/run_smooth_calibrate_for_HY3.sh
bash scripts/ptq/run_smooth_convert_for_HY3.sh

# 2. 后续 FP8 量化
python3 tools/fp8_quant_with_vllm_activation.py \
    --model-path /data/smooth_model \
    --ptq-data-path /path/to/calibration_data.jsonl \
    --fp8-path /data/fp8_smoothed_model
```

---

## 故障排查

### 问题 1：Ray 集群连接失败

**症状**：`RuntimeError: Job submission client cannot connect to Ray head.`

**解决方案**：

```bash
# 1. 检查 Ray 集群状态
ray status

# 2. 若集群未启动，重新启动（参考 README.md 的"环境准备"）
ray stop
ray start --head --port 6700 --num-gpus=8

# 3. 确保 PYTHONPATH 包含 AngelSlim 根目录
export PYTHONPATH=/path/to/AngelSlim:$PYTHONPATH
```

### 问题 2：vLLM 补丁未应用

**症状**：`AttributeError: collect_fused_moe_internal_stats not found` 或 `VLLM_MOE_COLLECT_STATS environment variable not recognized`

**解决方案**：

```bash
# 1. 检查补丁状态
bash tools/vllm_patch/install.sh check

# 2. 若检查失败，重新安装补丁
bash tools/vllm_patch/install.sh uninstall
bash tools/vllm_patch/install.sh install

# 3. 若 Ray 集群已启动，需要在 worker 节点上也安装补丁
# （在每个 worker 节点上运行）
bash tools/vllm_patch/install.sh install
```

### 问题 3：OOM（内存溢出）

**症状**：`CUDA out of memory` 或 `RuntimeError: CUDA memory`

**解决方案**：

```yaml
# 在配置文件中减少以下参数
batch_size: 2              # 从 4 减为 2
max_length: 8192           # 从 16384 减为 8192
num_samples: 256           # 从 512 减为 256

# 对 Alpha 搜索也减少
num_search_samples: 8      # 从 16 减为 8
search_max_length: 4096    # 从 8192 减为 4096
```

### 问题 4：Phase 2 转换失败

**症状**：`KeyError: 'model.layers.0.self_attn.q_proj' not found in smooth_stats.json`

**解决方案**：

```bash
# 1. 检查 smooth_stats.json 文件是否存在且非空
ls -la /path/to/output_dir/smooth_stats.json
wc -l /path/to/output_dir/smooth_stats.json

# 2. 查看 JSON 结构
python3 -c "import json; f=json.load(open('/path/to/smooth_stats.json')); print(list(f.keys())[:5])"

# 3. 检查配置中的 output_dir 是否正确
grep "output_dir:" configs/Hy3/ptq/fp8/Hy3_smooth.yaml
```

### 问题 5：Alpha 搜索耗时过长

**症状**：Phase 1 运行超过 10 小时仍未完成

**解决方案**：

```yaml
# 减少搜索复杂度
enable_alpha_search: false         # 暂时禁用搜索，仅收集统计
# 或改用手工指定的 alpha 值

# 若确实需要搜索，减少搜索参数
alpha_steps: 4                     # 从 8 改为 4
num_search_samples: 8              # 从 16 改为 8
alpha_min: 0.4                     # 缩小搜索范围
alpha_max: 0.8                     # 缩小搜索范围
```

---

## 参考资源

### 论文和文献

- **SmoothQuant 原论文**：https://arxiv.org/abs/2211.10438
  - 提出 Smooth 的核心思想和数学框架
  
- **GPTQ（含与 Smooth 的对比）**：https://arxiv.org/abs/2210.17323

### 相关文档

- [vLLM Smooth 补丁说明](../vllm_patch/README.md)
- [KV-Cache 校准说明](../kvcache/README.md)
- [主 PTQ 校准说明](../../scripts/ptq/README.md)
- [AngelSlim 量化模块文档](../../angelslim/compressor/quant/)

### 核心代码位置

| 文件 | 功能 |
| --- | --- |
| `tools/smooth/run_vllm_smooth.py` | Phase 1 主程序（统计收集 + Alpha 搜索） |
| `tools/smooth/convert_smooth_weights.py` | Phase 2 主程序（权重变换） |
| `angelslim/compressor/quant/modules/smooth/smooth.py` | 核心 Smooth 算法实现 |
| `angelslim/compressor/quant/core/vllm_calibrate_utils/` | 统计钩子 / Alpha 搜索实现 |
| `configs/Hy3/ptq/fp8/Hy3_smooth.yaml` | HY3 默认配置 |

### 快速命令参考

```bash
# 快速开始
bash scripts/ptq/run_smooth_for_HY3.sh

# 分步执行
bash scripts/ptq/run_smooth_calibrate_for_HY3.sh   # Phase 1
bash scripts/ptq/run_smooth_convert_for_HY3.sh     # Phase 2

# 查看日志
tail -f /path/to/output_dir/smooth_stats.json

# 验证输出
python3 -c "import json; print(json.load(open('/path/to/smooth_model/model.safetensors.metadata.json'))['_file_type'])"

# 集成到量化流水线
python3 tools/fp8_quant_with_vllm_activation.py --model-path /data/smooth_model ...
```

---

## 常见问题（FAQ）

**Q: Smooth 和 GPTQ 有什么区别？**

A: Smooth 是一种"前处理"方法，在量化前对激活和权重分布进行缩放变换，提升量化友好性；而 GPTQ 是一种量化算法本身，通过海森矩阵进行逐层优化。两者可以互补：先用 Smooth 预处理，再用 GPTQ 量化。

**Q: Alpha 应该如何选择？**

A: 若启用 `enable_alpha_search: true`，系统会自动为每层搜索最优 alpha；否则使用手工指定的 `alpha_qk` / `alpha_vo` / `alpha_down`。通常推荐启用搜索以获得更好的精度。

**Q: smooth_stats.json 可以手工编辑吗？**

A: 可以。文件格式为标准 JSON，每层包含 `absmax` 和 `ema` 两个数组。若某层的统计不合理，可以手工修改后重跑 Phase 2。

**Q: 如何应用 Smooth 到自己的模型？**

A: 在 `configs/` 下创建模型特定配置文件，修改 `model_path`、`ptq_data_path`、`output_dir`、`save_path` 等路径，然后执行相应的脚本。详见"集成与扩展"章节。

---

**最后更新**：2026-05-26

**维护者**：AngelSlim 开发团队

**反馈**：若发现问题或有改进建议，欢迎提交 Issue 或 Pull Request。
