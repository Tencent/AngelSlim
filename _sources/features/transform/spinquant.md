# Spinquant Transform

## 1. 概述

SpinQuant 是 AngelSlim 中的权重空间旋转变换模块，通过对模型权重施加正交旋转（Hadamard 或随机正交矩阵），抑制激活值中的异常通道，从而提升后训练量化（PTQ）的精度。旋转变换在数学上等价——不改变模型输出，但能使权重分布更加均匀，对量化更友好。

**核心特性：**

- 支持 R1/R2/R4 三种旋转（R3 尚未实现），可自由组合
- 离线旋转（R1/R2）融合进权重，推理零开销
- 在线旋转（R4）通过 forward pre-hook 实现，可导出至 vLLM
- 工厂注册机制 `@TransformFactory.register("SpinQuant")`，支持扩展
- 两种运行模式：仅旋转保存 / 旋转+量化联合压缩

**论文参考：** [SpinQuant (arXiv:2405.16406)](https://arxiv.org/abs/2405.16406)

---

## 2. 旋转类型

| 旋转 | 类型 | 作用位置 | 说明 |
|------|------|----------|------|
| **R1** | 离线 fused | embedding 输出、每层 norm 相邻线性层、lm_head | 全局正交旋转，融合进权重后推理零开销 |
| **R2** | 离线 fused | attn_v 输出侧 / attn_o 输入侧 | 按 head 维度做块对角旋转，正确处理 GQA（kv_heads / q_heads 分别构建 R2_v / R2_o） |
| **R3** | 在线（未实现） | Q/K 旋转 | 需推理端与 RoPE 融合，暂未实现 |
| **R4** | 在线 hook | down_proj 输入 | 注册 forward pre-hook 在线旋转，同时将 R4.T fuse 进权重，数值等价 |

### 旋转执行顺序

```
SpinQuant.run()
  ├── _apply_fused_ln()   # 1. center embeddings + fuse norm → linear
  ├── _apply_r1()         # 2. R1 fuse 进 embed/qkvo/mlp/lm_head
  ├── _apply_r2()         # 3. R2 按 head 块对角 fuse 进 v/o_proj
  └── _apply_r4()         # 4. R4 hook + fuse 进 down_proj
```

### 旋转融合数据流

```
原始权重                           旋转后权重
───────────────────────────────────────────────────────────────────
embed_tokens.weight  @ R1  →  embed_tokens.weight'
q/k/v_proj.weight    @ R1  →  q/k/v_proj.weight'   (R1 input-side)
attn_o.weight   R1.T @     →  attn_o.weight'        (R1 output-side)
up/gate_proj.weight  @ R1  →  up/gate_proj.weight'  (R1 input-side)
down_proj.weight     @ R1  →  down_proj.weight'     (R1 output-side)
lm_head.weight       @ R1  →  lm_head.weight'       (R1 input-side)

v_proj.weight   R2_v.T @   →  v_proj.weight'        (R2 output-side, kv_heads 块对角)
attn_o.weight    @ R2_o    →  attn_o.weight'         (R2 input-side,  q_heads  块对角)

down_proj  (hook: x @ R4 at runtime)
down_proj.weight     @ R4  →  down_proj.weight'     (R4 fuse，与 hook 等价抵消)
```

---

## 3. 两种运行模式

### 模式一：仅 Transform（离线旋转保存）

只对模型做旋转变换并保存，不做量化。适合预先旋转权重后再用其他工具量化。

**入口脚本：** `tools/run_transform_offline.py`

**流程：**

```
YAML 配置 → SlimConfigParser.parse() → FullConfig
  → SlimModelFactory.create() + from_pretrained() → 加载模型
    → TransformFactory.create() → SpinQuant.run() → 旋转融合
      → model.save_pretrained() → 保存旋转后权重
```

**YAML 配置示例（仅 transform，无 compression/dataset）：**

```yaml
global:
  save_path: ./output

model:
  name: Qwen
  model_path: Qwen/Qwen3-8B
  trust_remote_code: true
  low_cpu_mem_usage: true
  use_cache: false
  torch_dtype: bfloat16
  device_map: auto

transform:
  name: SpinQuant
  spin_config:
    had_dim: -1                 # -1 表示全尺寸 Hadamard
    rotation_mode: Hadamard     # "Hadamard"（确定性）或 "Random"（随机正交）
    rotation:
      - R1
      - R2
      # - R4                   # 按需启用
    ignore_layers: []
  output_log: false
```

**运行命令：**

```bash
# 基本用法
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/path/to/AngelSlim \
python tools/run_transform_offline.py \
    -c configs/qwen3/spin/qwen3_spinquant.yaml \
    --model-path /path/to/model \
    --save-path ./output

# 同时验证 transform 前后 logits 一致（atol=1e-2）
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/path/to/AngelSlim \
python tools/run_transform_offline.py \
    -c configs/qwen3/spin/qwen3_spinquant.yaml \
    --model-path /path/to/model \
    --test-output-diff
```

**CLI 参数：**

| 参数 | 说明 |
|------|------|
| `-c / --config` | YAML 配置文件路径（必需） |
| `--model-path` | 覆盖 YAML 中的 `model.model_path` |
| `--save-path` | 覆盖 YAML 中的 `global.save_path` |
| `--test-output-diff` | 验证旋转前后 logits 数值一致性 |

---

### 模式二：Transform + PTQ（一键旋转+量化）

在同一流程中先做旋转再做量化，适合端到端压缩。

**入口脚本：** `tools/run.py`

**流程：**

```
YAML 配置 → SlimConfigParser.parse() → FullConfig
  → Engine.prepare_model() → 加载模型
    → Engine.prepare_data() → 准备校准数据
      → Engine.prepare_compressor() → PTQ.__init__()
        ├── TransformFactory.create() → SpinQuant（先做旋转）
        └── PTQHook / GPTQ / AWQ（再做量化）
          → Engine.run() → compressor.calibrate(dataloader)
            → Engine.save() → 保存旋转+量化后权重
```

**YAML 配置示例（transform + compression + dataset）：**

```yaml
global:
  save_path: ./output

model:
  name: Qwen
  model_path: Qwen/Qwen3-8B
  trust_remote_code: true
  low_cpu_mem_usage: true
  use_cache: false
  torch_dtype: bfloat16
  device_map: auto

# Transform 配置
transform:
  name: SpinQuant
  spin_config:
    had_dim: -1
    rotation_mode: Hadamard
    rotation:
      - R1
      - R2
      - R4
    ignore_layers: []
  output_log: false

# 量化配置（在 transform 基础上追加）
compression:
  name: PTQ
  quantization:
    name: fp8_static          # 支持: fp8_static, fp8_dynamic, int4_awq, int4_gptq
    bits: 8
    quant_method:
      weight: "per-tensor"
      activation: "per-tensor"
    ignore_layers:
      - "lm_head"

# 校准数据集
dataset:
  name: TextDataset
  data_path: ./dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
  max_seq_length: 2048
  num_samples: 256
  batch_size: 1
```

**运行命令：**

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/path/to/AngelSlim \
python tools/run.py \
    -c configs/qwen3/spinquant/qwen3_spinquant_fp8_static.yaml \
    --model-path /path/to/model \
    --save-path ./output
```

---

## 4. 配置参数详解

### SpinConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `had_dim` | int | -1 | Hadamard 块大小。`-1` 表示全尺寸（即 hidden_size / intermediate_size），也可设为 64、128 等 2 的幂次 |
| `rotation_mode` | str | `"Hadamard"` | 旋转模式。`"Hadamard"` 为确定性 Hadamard 矩阵；`"Random"` 为随机正交矩阵 |
| `rotation` | List[str] | `[]` | 启用的旋转列表，可选值：`R1`、`R2`、`R4`（R3 未实现） |
| `ignore_layers` | List[str] | `[]` | 跳过旋转的层名列表（精确匹配 named_modules key） |

### TransformConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | 必填 | Transform 名称，固定为 `"SpinQuant"` |
| `spin_config` | dict / SpinConfig | `{}` | SpinQuant 旋转参数，见上表 |
| `output_log` | bool | `false` | 是否写入 transform.log 日志文件 |

### 常见配置组合

```yaml
# 仅 R1+R2（纯离线，推理零开销）
spin_config:
  rotation: [R1, R2]
  rotation_mode: Hadamard

# R1+R2+R4（离线+在线，精度最高）
spin_config:
  rotation: [R1, R2, R4]
  rotation_mode: Hadamard
  had_dim: -1              # 全尺寸 R4

# 块对角 R4（推理更快，精度稍低）
spin_config:
  rotation: [R1, R2, R4]
  rotation_mode: Hadamard
  had_dim: 64              # 64x64 块 Hadamard
```

---

## 5. 已有 YAML 配置文件

仓库中已提供以下示例配置：

| 配置文件 | 模型 | 量化方式 | 旋转 |
|----------|------|----------|------|
| `configs/qwen3/spinquant/qwen3_spinquant_fp8_static.yaml` | Qwen3-8B | FP8 Static (per-tensor) | R1+R2+R4 |
| `configs/qwen3/spinquant/qwen3-8b_spinquant_int4_awq.yaml` | Qwen3-8B | INT4 AWQ (per-group) | R1+R2+R4 |
| `configs/qwen3/spinquant/qwen3-a3b_spinquant_fp8_static.yaml` | Qwen3-30B-A3B (MoE) | FP8 Static (per-tensor) | R1+R2（无 R4） |

---

## 6. Python API

### 通过工厂模式创建（推荐）

```python
from angelslim.compressor.transform import TransformFactory

# slim_config 是包含 transform_config 和 global_config 的 dict
slim_config = {
    "transform_config": TransformConfig(
        name="SpinQuant",
        spin_config={
            "rotation_mode": "Hadamard",
            "rotation": ["R1", "R2", "R4"],
            "had_dim": -1,
            "ignore_layers": [],
        },
        output_log=False,
    ),
    "global_config": global_config,
}

transform = TransformFactory.create(slim_model, slim_config)
transform.run()   # 执行旋转
```

### 直接构造

```python
from angelslim.compressor.transform import SpinQuant

spin = SpinQuant(slim_model, quant_config=slim_config)
spin.run()       # 执行旋转
```

> **注意：** `__init__` 只做初始化，不执行旋转；调用 `run()` 才触发完整的 norm fuse + 旋转融合。

### 静默模式（仅收集 linears 和旋转矩阵，不修改权重）

```python
spin.slient_run()  # QAT 准备阶段使用
```

### 获取旋转矩阵

```python
rot = spin.get_rotation_mat()
# 返回 dict：
# rot["R1"]: Tensor [hidden_size, hidden_size]
# rot["R2"]: Dict[layer_prefix -> Tensor [head_dim, head_dim]]
# rot["R3"]: None（未实现）
# rot["R4"]: Tensor [intermediate_size, intermediate_size]
```

### 获取受影响的线性层

```python
linears = spin.get_linears()
# 返回 dict：
# linears["R1"]: [R1_embed_linears, R1_linears, R1_inv_linears]
# linears["R2"]: [R2_linears, R2_inv_linears]
# linears["R3"]: [R3_linears]
# linears["R4"]: [R4_linears]
```

### QAT 后融合旋转

```python
spin.convert(R1=R1_mat, R2_list=[H1, H2, ...], R4_list=[R4_mat])
```

---

## 7. 层名映射

SpinQuant 通过 `mapping.py` 中的映射字典定位模型中的各类层。默认映射兼容 LLaMA / Qwen 架构：

```python
linear_mapping = dict(
    embedding="embed_tokens",
    attn_q="q_proj",
    attn_k="k_proj",
    attn_v="v_proj",
    attn_o="o_proj",
    mlp_in=["up_proj", "gate_proj"],
    mlp_out=["down_proj"],
    lm_head="lm_head",
)

norm_mapping = [
    (["q_proj", "k_proj", "v_proj"], "input_layernorm"),
    (["up_proj", "gate_proj"], "post_attention_layernorm"),
    (["lm_head"], "norm"),
]
```

如果你的模型架构层名不同，可通过 `spin_config.mappings` 和 `spin_config.norm_mappings` 自定义。

---

## 8. vLLM适配导出

当配置 `global.deploy_backend: vllm` 且启用 R4 旋转时，SpinQuant 会自动生成 `TransformConfig` 并写入模型的 `config.json`，vLLM 通过 `compressed_tensors` 标准流程重建在线旋转。

生成的 transform_config 格式：

```json
{
  "transform_config": {
    "config_groups": {
      "R4": {
        "apply": [{
          "ignore": [],
          "inverse": false,
          "location": "input",
          "targets": ["re:.*down_proj$"]
        }],
        "head_dim": -1,
        "randomize": false,
        "requires_grad": false,
        "type": "hadamard"
      }
    }
  }
}
```

---

## 9. 文件结构

### 新增文件

| 文件 | 作用 |
|------|------|
| `angelslim/compressor/transform/__init__.py` | 导出 `TransformBase`、`TransformFactory`、`SpinQuant` |
| `angelslim/compressor/transform/base.py` | `TransformBase` 抽象基类（run/convert/save 生命周期） |
| `angelslim/compressor/transform/factory.py` | `TransformFactory` 工厂类，`@register` 装饰器 |
| `angelslim/compressor/transform/rotation/__init__.py` | 导出 `SpinQuant` |
| `angelslim/compressor/transform/rotation/spin.py` | `SpinQuant` 主实现：R1/R2/R4 旋转融合 |
| `angelslim/compressor/transform/rotation/mapping.py` | 层名映射 `linear_mapping` / `norm_mapping` |
| `angelslim/compressor/transform/rotation/fuse_norm_utils.py` | `fuse_ln_linear`、`center_embeddings` 工具函数 |
| `angelslim/compressor/transform/rotation/hadamard_utils.py` | Hadamard 矩阵生成工具 |
| `tools/run_transform_offline.py` | 离线 transform 入口脚本 |
| `tools/test_spinquant_equivalence.py` | 等价性测试脚本 |
| `configs/qwen3/spinquant/*.yaml` | Qwen3 SpinQuant 配置示例 |

---

## 10. 快速上手示例

### 示例 1：仅旋转（R1+R2），保存旋转后模型

```bash
# 1. 准备配置文件 my_spin.yaml
cat > my_spin.yaml << 'EOF'
global:
  save_path: ./output/spin_only

model:
  name: Qwen
  model_path: /path/to/Qwen3-8B
  trust_remote_code: true
  torch_dtype: bfloat16
  device_map: auto

transform:
  name: SpinQuant
  spin_config:
    rotation_mode: Hadamard
    rotation: [R1, R2]
  output_log: false
EOF

# 2. 运行
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/AngelSlim \
python tools/run_transform_offline.py -c my_spin.yaml
```

### 示例 2：SpinQuant + FP8 量化

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/path/to/AngelSlim \
python tools/run.py \
    -c configs/qwen3/spinquant/qwen3_spinquant_fp8_static.yaml \
    --model-path /path/to/Qwen3-8B \
    --save-path ./output/spin_fp8
```

### 示例 3：SpinQuant + INT4 AWQ 量化

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/path/to/AngelSlim \
python tools/run.py \
    -c configs/qwen3/spinquant/qwen3-8b_spinquant_int4_awq.yaml \
    --model-path /path/to/Qwen3-8B \
    --save-path ./output/spin_int4_awq
```

### 示例 4：验证旋转等价性

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/AngelSlim \
python tools/run_transform_offline.py \
    -c my_spin.yaml \
    --test-output-diff
# 输出示例：
# [test] Max  diff = 1.234567e-03
# [test] Mean diff = 5.678901e-05
```

## 实现特性

以下是 AngelSlim 中 SpinQuant 实现（`angelslim/compressor/transform/rotation/spin.py`）的关键设计特性。

### 1. CPU 计算 + 设备回写

旋转矩阵的生成和权重融合计算统一在 CPU 上进行（`DEVICE='cpu'`），计算完成后将结果写回权重原始所在设备（GPU）。这一设计避免了大尺寸旋转矩阵（如 `[hidden_size, hidden_size]`）占用宝贵的 GPU 显存，对多卡 `device_map=auto` 场景尤其重要。

```python
DEVICE = 'cpu'

def _apply_linear_fuse(self, linear, rotation, fuse_input=False):
    origin_device = linear.weight.device                         # 记住原始设备
    weight = linear.weight.data.to(device=DEVICE, dtype=torch.float32)  # 搬到 CPU
    rotation = rotation.to(device=DEVICE, dtype=torch.float32)
    # ... 在 CPU 上做矩阵乘 ...
    linear.weight.data = new_weight.to(dtype=linear.weight.dtype, device=origin_device)  # 写回原设备
```

`_apply_emb_fuse()` 采用相同策略。R4 的在线 hook 中则预先将旋转矩阵缓存到权重所在设备（`ORI_DEVICE_H = rot.to(linear.weight.device)`），避免每次推理 forward 时重复做 CPU→GPU 迁移。

### 2. 多线程并行 Fuse

引入 `_parallel_apply()` 方法，使用 `ThreadPoolExecutor`（最大 64 线程）并行执行独立的权重融合任务，并附带 tqdm 进度条。R1/R2/R4 的 fuse 操作均改为任务列表后批量提交，相比逐层串行循环可显著缩短大模型的旋转时间。

```python
MAX_THREADS = 64

def _parallel_apply(self, tasks, desc=None):
    """并行执行 (fn, args, kwargs) 任务列表，每个任务操作独立的 linear 层，无写冲突。"""
    from tqdm import tqdm
    pbar = tqdm(total=len(tasks), desc=desc, leave=False)
    def _wrap(fn, args, kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            pbar.update(1)
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(_wrap, fn, args, kwargs) for fn, args, kwargs in tasks]
    pbar.close()
    for f in futures:
        f.result()   # 重新抛出 worker 中的异常
```
