# QAT 量化感知训练

## 概述

QAT（Quantization Aware Training，量化感知训练）是一种在训练过程中模拟量化效果的压缩方法，通过在前向传播中插入伪量化（Fake Quantization）操作，让模型在训练阶段感知量化带来的精度损失，从而学习到更适合量化部署的参数分布。相比 PTQ（Post-Training Quantization，训练后量化），QAT 能够在低比特量化场景（如 W4A8、INT4）下获得更优的精度表现。

AngelSlim 的 QAT 模块提供了完整的量化感知训练流程，核心特性包括：

- **多种量化格式**：支持 INT（任意位宽）和 FP8（E4M3）两种数据类型，以及 W4A8-FP8 混合精度量化
- **灵活的量化粒度**：per-tensor、per-channel、per-group、per-token 四种粒度可选
- **对称 / 非对称量化**：同时支持对称（symmetric）和非对称（asymmetric）量化模式
- **两种训练模式**：端到端（End-to-End）训练和逐块（Blockwise）训练
- **插件化架构**：可学习 Scale（Learnable Scale）等功能以插件形式集成，易于扩展
- **完善的评测**：内置 PPL（Perplexity）和 ACC（Accuracy）评测，支持 wikitext2、c4、piqa、arc 等多种基准

## 架构设计

### 模块目录结构

```
angelslim/compressor/qat/
├── qat.py                          # QAT 主入口类
├── modules/
│   ├── __init__.py
│   ├── quantizer.py                # 量化核心：Quantizer + QuantLinear
│   └── scaler.py                   # AMP 梯度缩放器
├── plugins/
│   ├── __init__.py
│   ├── base_plugin.py              # 插件基类
│   ├── learnable_scale.py          # 可学习 Scale 插件
│   └── plugin_manager.py           # 插件注册与管理
└── trainers/
    ├── __init__.py
    ├── blockwise_trainer.py        # 逐块训练器
    ├── end2end_trainer.py          # 端到端训练器
    └── trainer_factory.py          # 训练器工厂

angelslim/data/
└── qat_dataset.py                  # 数据集封装（QATDataset / BlockTrainDataset）
```

### 核心架构图

```
                    ┌──────────────────────────────────┐
                    │         Engine (engine.py)       │
                    │  prepare_model → prepare_data    │
                    │  prepare_compressor → run → save │
                    └───────────────┬──────────────────┘
                                    │
                     CompressorFactory.create(["QAT"])
                                    │
                    ┌───────────────▼──────────────────┐
                    │           QAT (qat.py)           │
                    │  init_ptq → _init_plugins        │
                    │  _init_trainer → run → convert   │
                    └───┬───────────┬──────────┬───────┘
                        │           │          │
         ┌──────────────▼──-┐  ┌────▼─────┐  ┌─▼──────────────-┐
         │  PluginManager   │  │ Trainer  │  │modules/quantizer│
         │ ┌──────────────┐ │  │ Factory  │  │ ┌────────────┐  │
         │ │ Learnable    │ │  │ ┌──────┐ │  │ │ Quantizer  │  │
         │ │ ScalePlugin  │ │  │ │E2E   │ │  │ │ QuantLinear│  │
         │ └──────────────┘ │  │ │Block │ │  │ └────────────┘  │
         └──────────────────┘  │ └──────┘ │  └────────────────-┘
                               └──────────┘
```

### 执行流程

1. **初始化**：`QAT.__init__()` 加载 PTQ 配置、初始化插件系统和训练器
2. **数据准备**：准备训练数据集（支持 HuggingFace 开源数据集或内部数据格式）
3. **插件 `before_train`**：将 `nn.Linear` 替换为 `QuantLinear`，执行 activation 校准初始化
4. **训练**：根据配置选择端到端训练或逐块训练模式
5. **插件 `after_train`**：执行训练后处理
6. **转换**：`convert()` 将 `QuantLinear` 替换为推理用的 `QDQModule`
7. **保存**：支持以下保存格式：

| `save_format` | 描述 | 备注 |
|---------------|------|------|
| `fake` | 保存伪量化的 state_dict（`.pt` 文件） | **仅支持非分布式 / 单卡** |
| `real` | 保存真实量化模型（通过模型子类的 `get_save_func` 导出） | |
| `save_kvcache_only` | 仅导出 KV Cache 的 scale 参数到 `kv_cache_scales.safetensors` | |
| `real_and_kvcache` | 同时保存真实量化模型 **和** KV Cache scales | |
| 不设置 | 跳过保存 | |

---

## 参数配置说明

QAT 的配置文件遵循 AngelSlim 标准 YAML 格式，分为 `model`、`compression`、`dataset`、`global` 四个部分。其中 QAT 的训练相关配置嵌套在 `compression.QAT` 下。以下详细说明 QAT 相关的配置参数。

### model — 模型配置

```yaml
model:
  name: Qwen                        # 模型架构名称
  model_path: /path/to/model        # HuggingFace 模型路径
  trust_remote_code: true           # 是否信任远程代码
  torch_dtype: auto                 # 数据类型（auto / fp16 / bf16）
  device_map: auto                  # 设备映射（auto / cuda / cpu）
  low_cpu_mem_usage: true           # 是否开启低内存模式
  use_cache: false                  # 训练时建议关闭 KV Cache
```

### compression — 压缩配置

```yaml
compression:
  name: QAT                         # 压缩方法，QAT 固定填写 "QAT"
  quantization:
    name: "w4a8_fp8"                # 量化算法名称,可选
    bits: 8                         # 量化位宽,可选
    quant_method:                   # 可选
      weight: "per-group"           # 权重量化粒度
      activation: "per-tensor"      # 激活量化粒度
      group_size: 128               # 分组量化的组大小（per-group 时需要）
    ignore_layers:                  # 忽略量化的层列表
      - "lm_head"
      - "embed_tokens"
```

**量化算法名称说明**：

| `quantization.name` | 描述 | 权重量化 | 激活量化 |
|---------------------|------|---------|---------|
| `w4a8_fp8` | W4A8 混合精度 | INT4 per-group | FP8 per-tensor |
| `fp8_dynamic` | FP8 动态量化 | FP8 per-tensor | FP8 per-tensor（动态） |
| `fp8_static` | FP8 静态量化 | FP8 per-tensor | FP8 per-tensor（静态） |


**量化粒度说明**：

| 粒度 | 描述 | 适用场景 |
|------|------|---------|
| `per-tensor` | 整个张量共享一组 scale/zero_point | 激活量化，FP8 权重量化 |
| `per-channel` | 每个输出通道一组参数 | 通用权重量化 |
| `per-group` | 每 `group_size` 个元素一组参数 | INT4 权重量化，精度更优 |
| `per-token` | 每个 token 一组参数（始终为动态） | 激活动态量化 |
| `per-head` | 每个 KV head 一组参数 | QKV 投影输出量化（KV Cache 压缩） |

### QKV / Attention 量化配置

QAT 支持对 Q/K/V 投影层的**输出**进行独立量化配置，用于模拟 KV Cache 量化压缩的精度损失，使模型在训练阶段感知 KV Cache 量化带来的影响。同时支持 FP8 Attention 模拟，对 softmax 后的 attention weights 执行 FP8 cast。

QKV 量化配置位于 `compression.QAT.plugin_config.quant_config` 下，与 `weight` / `activation` 平级：

| 配置项 | 类型 | 描述 |
|--------|------|------|
| `fp8_attn` | bool | 是否启用 FP8 Attention 模拟。启用后会 monkey-patch attention forward，对 attn_weights 执行 FP8 cast（需模型子类实现 `patch_fp8_attention`） |
| `q` | dict | Query 投影输出的量化配置 |
| `k` | dict | Key 投影输出的量化配置（KV Cache 中的 Key） |
| `v` | dict | Value 投影输出的量化配置（KV Cache 中的 Value） |

每个 `q` / `k` / `v` 配置字典支持以下字段：

| 字段 | 类型 | 描述 |
|------|------|------|
| `qtype` | str | 量化类型，如 `fp8`、`int8` |
| `granularity` | str | 量化粒度：`per-token`、`per-head`、`per-tensor`、`per-channel` |
| `group_size` | int | 分组大小，`-1` 表示不分组 |
| `is_sym` | bool | 是否对称量化 |

### dataset — 校准数据集配置

```yaml
dataset:
  name: TextDataset                 # 数据集类型
  data_path: ./dataset/data.jsonl   # 数据路径，支持 jsonl、parquet 格式以及 HF 数据集
  max_seq_length: 2048              # 最大序列长度
  num_samples: 256                  # 样本数量
  batch_size: 1                     # 批量大小
```

### QAT 训练配置

这是 QAT 的核心配置部分，统一嵌套在 `compression.QAT` 下：

```yaml
# 以下配置统一位于 compression.QAT 下
  # ========== 基础配置 ==========
  training_mode: "end2end"          # 训练模式："end2end" 或 "blockwise"
  dist_mode: "hf"                   # 分布式模式："hf"（HuggingFace Trainer）
  do_train: true                    # 是否执行训练
  save_format: "real"               # 保存格式（见下表），不设置表示跳过保存
  resume_ckpt_dir: ""               # checkpoint 恢复路径，用于加载 save_format 为 fake 的权重。不设置表示不使用 resume

  # ========== 数据集配置 ==========
  hf_dataset: "Salesforce/wikitext,wikitext-2-raw-v1"  # HuggingFace 数据集（可选，逗号分隔路径和子集名）
  hf_cache_dir: /path/to/cache      # HuggingFace 数据集缓存目录

  # ========== 插件配置 ==========
  plugin_config:
    enable_scale: true              # 是否启用可学习 Scale 插件
    enable_rotation: false          # 是否启用可学习 Rotation 插件（实验性）
    quant_config:
      use_weight_quant: true        # 是否量化权重
      use_activation_quant: true    # 是否量化激活
      lazy_init_samples: 60         # activation 校准所需的样本数（默认 10）
      # ========== 可学习参数控制（可选） ==========
      learnable:
        act_scale: true             # 激活量化器的 scale/zero_point（默认：true）
        weight_scale: false         # 权重量化器的 scale/zero_point（默认：false）
        kv_scale: false             # KV Cache 量化器的 scale（默认：false）
        norm: false                 # Norm 层权重（默认：false）
      # 可选：覆盖 compression.quantization 中的默认量化配置
      weight:
        qtype: int8                 # 权重量化类型（如 int4, int8, fp8）
        granularity: per-tensor     # 权重量化粒度
        group_size: 128             # 分组大小（per-group 时需要）
        is_sym: true                # 是否对称量化
      activation:
        qtype: int8                 # 激活量化类型（如 int8, fp8）
        granularity: per-tensor     # 激活量化粒度
        is_sym: true                # 是否对称量化
        dynamic: false              # 是否动态量化

      # ========== QKV / Attention 量化配置（可选） ==========
      fp8_attn: true                # 是否启用 FP8 Attention 模拟（对 attn_weights 做 FP8 cast）
                                      # 注意：开启 fp8_attn 时，需要在 model 配置中设置 attn_implementation: eager
      q:                            # Query 投影输出量化
        qtype: fp8                  # 量化类型（fp8 / int8 等）
        granularity: per-token      # 量化粒度（per-token / per-tensor）
        group_size: -1              # 分组大小（-1 表示不分组）
        is_sym: true                # 是否对称量化
      k:                            # Key 投影输出量化（通常用于 KV Cache 压缩）
        qtype: fp8
        granularity: per-head       # per-head 粒度，每个 KV head 独立 scale
        group_size: -1
        is_sym: true
      v:                            # Value 投影输出量化
        qtype: fp8
        granularity: per-head
        group_size: -1
        is_sym: true

```

#### End-to-End 训练专属配置

端到端训练使用 HuggingFace `Seq2SeqTrainer`，QAT 相关配置嵌套在 `compression.QAT` 下，需要额外配置 `hf_args`，详情可参考 [TrainingArguments](https://huggingface.co/docs/transformers/v5.1.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments)：

```yaml
compression:
  name: QAT
  quantization:
    # ...量化配置...
  QAT:
    training_mode: "end2end"
    dist_mode: hf
    hf_args:
      # output_dir: /path/to/output   训练输出目录，不需要再指定，同 global.save_path
      # 其余参数同 HF 的 TrainingArguments
```

#### Blockwise 训练专属配置

逐块训练将模型按 Transformer Block 逐层训练，显存占用更低。逐块模式下 QAT 配置同样位于 `compression.QAT` 下：

```yaml
compression:
  name: QAT
  quantization:
    # ...量化配置...
  QAT:
    training_mode: "blockwise"
    block_wise_config:
      epochs: 10                      # 每个 block 的训练轮数（默认 20）
      batch_size: 2                   # 批量大小（默认 1）
      train_size: 4096                # 训练样本数量（默认 128）
      val_size: 64                    # 验证样本数量（默认 64）
      training_seqlen: 2048           # 训练序列长度（默认 2048）
      quant_lr: 1e-4                  # scale/zero_point 参数学习率
      weight_lr: 1e-4                 # 权重参数学习率（0 则冻结权重，默认 1e-3）
      min_lr_factor: 20               # CosineAnnealing 最小学习率因子（lr / factor）
      wd: 0                           # 权重衰减

```

### global — 全局配置

```yaml
global:
  save_path: /path/to/save/model    # 模型保存路径
```

---

## 核心模块详解

### Quantizer — 量化器

`Quantizer` 位于 `modules/quantizer.py`，负责执行伪量化（Fake Quantization）操作。

**关键特性**：

- 使用 **STE（Straight-Through Estimator）** 使 `round` 和 `clamp` 操作可微分，允许梯度通过量化操作反向传播
- 支持 **延迟初始化（Lazy Initialization）**：对于静态 activation 量化，通过前 N 个样本校准确定 scale
- `scale` 和 `zero_point` 注册为 `nn.Parameter`，在训练过程中可学习优化

**伪量化过程**（INT 类型为例）：

```
x → x / scale → round_ste → clamp(qmin, qmax) → × scale → x_quant
```

### QuantLinear — 量化线性层

`QuantLinear` 是 `nn.Linear` 的量化替换层，在前向传播中分别对权重和激活执行伪量化：

```python
def forward(self, input: torch.Tensor):
    if input.shape[0] == 0:
        return self.fwd_func(input, self.weight, self.bias)

    weight = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight  # 权重伪量化
    if self.use_act_quant:
        input = self.act_quantizer(input)  # 激活伪量化
    return self.fwd_func(input, weight, self.bias)
```

### LearnableScalePlugin — 可学习 Scale 插件

该插件在训练前（`before_train`）执行以下操作：

1. 遍历模型所有 `nn.Linear` 层（跳过 `ignore_layers` 指定的层），替换为 `QuantLinear`
2. 对静态 activation 量化执行 lazy 校准初始化
3. 根据 `learnable` 配置选择性地设置哪些参数可训练

#### Learnable 参数控制

通过 `learnable` 配置字典，可以独立控制每种参数组是否参与训练。模型权重本身始终保持冻结。

```yaml
plugin_config:
  enable_scale: true
  quant_config:
    learnable:
      act_scale: true           # 激活量化器的 scale/zero_point（默认：true）
      weight_scale: false       # 权重量化器的 scale/zero_point（默认：false）
      kv_scale: false           # KV Cache 量化器的 scale（k_proj/v_proj 中的 qkv_quantizer）（默认：false）
      norm: false               # Norm 层（RMSNorm / LayerNorm）的权重（默认：false）
```

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `act_scale` | bool | `true` | 是否学习激活量化器（`act_quantizer`）的 scale / zero_point |
| `weight_scale` | bool | `false` | 是否学习权重量化器（`weight_quantizer`）的 scale / zero_point |
| `kv_scale` | bool | `false` | 是否学习 KV Cache 量化器（`qkv_quantizer`）的 scale（仅 k_proj / v_proj） |
| `norm` | bool | `false` | 是否学习 Norm 层（如 `input_layernorm`、`post_attention_layernorm`）的 weight 参数 |

各开关可以自由组合，例如同时开启 `weight_scale` 和 `kv_scale` 来联合优化权重量化和 KV Cache 量化的 scale 参数。

> **注意**：如果未提供 `learnable` 配置，默认行为等价于 `act_scale: true`（其余为 `false`），即仅学习激活量化器的 scale 参数。

### TrainerFactory — 训练器工厂

通过装饰器模式自动注册训练器，使用 `training_mode` 字段选择：

| `training_mode` | 训练器类 | 描述 |
|-----------------|---------|------|
| `end2end` | `End2EndTrainer` | 端到端训练，使用 HuggingFace Trainer |
| `blockwise` | `BlockwiseTrainer` | 逐 Transformer Block 训练，继承自 `End2EndTrainer` |

### End-to-End 训练器

- 使用 HuggingFace `Seq2SeqTrainer` 进行训练
- 使用 `AdamW` 优化器，仅针对 `scale` 和 `zero_point` 参数（默认学习率 1e-5）
- 支持 HuggingFace 生态的各种训练参数（学习率调度、梯度累积等）
- 完整执行流程：`prepare_dataset` → `prepare_trainer` → `call_before_train` → 可选 `resume` → `train` → `call_after_train`

### Blockwise 训练器

`BlockwiseTrainer` 继承自 `End2EndTrainer`，其工作原理：

1. **数据集准备**：创建 `BlockTrainDataset` 存储每层的输入激活（`fp_train_inps` 和 `quant_train_inps`）
2. **捕获输入**：使用 `_Catcher` 截获第一个 Transformer Block 的输入激活及 `layer_kwargs`（如 position embeddings）
3. **逐层训练**：对每个 Block 依次执行：
   - 先用 FP（全精度）前向得到该层的原始输出，更新 `fp_train_inps`
   - 开启量化状态，用 MSE Loss 对齐量化层输出与 FP 层输出
   - 使用 `AdamW` 优化器，支持量化参数（`quant_lr`）和权重参数（`weight_lr`）的分离学习率
   - 使用 `CosineAnnealingLR` 调度器，最小学习率为 `lr / min_lr_factor`
   - 使用 `NativeScalerWithGradNormCount` 进行混合精度梯度缩放
4. **量化固化**：每个 Block 训练完成后转为半精度并执行就地量化（`quant_inplace`）
5. **传播更新**：将量化后的输出更新到 `quant_train_inps`，作为下一层的输入



---

## 评测功能

AngelSlim 内置了模型评测器，支持以下评测任务：

| 评测指标 | 数据集 |
|--------|------|
| PPL | `wikitext2`、`c4` |
| ACC | `piqa`、`arc_easy`、`arc_challenge`、`hellaswag`、`winogrande`等 |

详情参见 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 框架。

运行示例：

```python
python3 tools/run.py -c "configs/qwen3/qat/int4_weight_only/learn_scale/qwen3-4b_int4_weight_only_end2end_learn_scale.yaml" --lm-eval --ppl-eval
```

---

## 与现有模块的集成

### 扩展新插件

QAT 的插件系统基于 `PluginManager` 提供了便捷的扩展机制。要添加新插件，只需：

1. 继承 `BasePlugin` 基类
2. 使用 `@PluginManager.plugin("plugin_name")` 装饰器注册
3. 实现 `before_train()` 和/或 `after_train()` 方法

```python
from angelslim.compressor.qat.plugins.base_plugin import BasePlugin
from angelslim.compressor.qat.plugins.plugin_manager import PluginManager

@PluginManager.plugin("my_custom_plugin")
class MyPlugin(BasePlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_train(self, **kwargs):
        # 在训练前执行自定义逻辑
        pass

    def after_train(self, **kwargs):
        # 在训练后执行自定义逻辑
        pass
```

然后在配置中启用：

```yaml
compression:
  QAT:
    plugin_config:
      enable_my_custom_plugin: true
```

### 扩展新训练器

类似地，可通过 `TrainerFactory` 注册新的训练模式：

```python
from angelslim.compressor.qat.trainers.trainer_factory import TrainerFactory

@TrainerFactory.register("my_custom_trainer")
class MyTrainer:
    def __init__(self, quant_model, config, plugin_manager):
        pass

    def run(self, dataloader):
        # 自定义训练逻辑
        pass

```
