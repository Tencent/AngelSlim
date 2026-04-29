# QAT + DeepSpeed ZeRO-3

## 概述

本文档描述 AngelSlim QAT 模块的 **DeepSpeed ZeRO-3** 支持。核心动机：当基础模型体量超过单卡显存（例如 Qwen3-30B-A3B 这类 MoE 模型）时，需要把模型参数、梯度、优化器状态都切片到多张 GPU 上，HuggingFace + DeepSpeed 的 ZeRO-3 是一套成熟的方案，但直接用在 QAT 流程里会遇到以下问题：

1. HuggingFace `from_pretrained` 在 ZeRO-3 下会让每个 rank 先在 CPU 上加载**完整** state_dict，再 partition，峰值内存约 `world_size × model_size`，大模型必然 OOM。
2. QAT 会把原始 `nn.Linear` 替换成 `QuantLinear`，又会把 fused MoE expert 拆成 per-expert `nn.Linear`。这些替换操作如果在 ZeRO-3 sharded 参数上执行，会同时触发切片 / 合并 / 重分发，逻辑极易出错、峰值不可控。
3. QAT 的 activation scale 默认通过 forward 校准（lazy init），在 ZeRO-3 下无法运行；weight scale 的初始化依赖读取完整权重，也不可行。
4. `convert()` 把 `QuantLinear` 转为 `QDQModule`，以及 `save_format=real` 导出压缩权重时，需要在多卡之间合并参数并保持 CPU 内存在 **rank0 一份**。

针对上述问题，本实现的关键设计如下：

- **每个 rank 独立构造一个空模型**（不读磁盘权重），通过 `deepspeed.zero.Init` 立即 partition，峰值内存仅 `model_size / world_size`。
- **fused MoE 拆解发生在空模型阶段**（duck-typing 识别 `gate_up_proj / down_proj / num_experts / hidden_dim / intermediate_dim / act_fn`），新建的 per-expert `nn.Linear` 被 `deepspeed.zero.Init` 立即切片，**不需要从旧的 fused tensor 拷贝数据**。
- **权重和 scale 都通过 safetensors 流式加载**：rank0 按文件顺序读取，其它 rank 通过 `GatheredParameters(modifier_rank=0)` 接收数据。fused MoE 的 `.experts.gate_up_proj` / `.experts.down_proj` key 会被自动切片写入每个 per-expert target。
- **ZeRO-3 下 scale 强制从外部 PTQ checkpoint 加载**（`from_ptq_ckpt`），跳过 forward 校准；未命中的 scale 按 `weight_scale_init_value` / `activation_scale_init_value` 填充。
- **`convert / save` 只在 rank0 构造 `QDQModule` 并直接写入合并 state_dict**，其它 rank 只参与 NCCL gather 的 collective，不持有 CPU 数据。

整体流程下，非 ZeRO-3 路径的行为与 main 分支完全保持一致——所有 ZeRO-3 逻辑都收敛在一个新文件 `angelslim/utils/zero3_io.py` 中，其他文件只做少量薄调用。

## 架构设计

### 模块目录结构

```
angelslim/
├── utils/
│   └── zero3_io.py                 # ALL ZeRO-3 helpers: detection,
│                                   # gather/scatter, empty-model build,
│                                   # streaming weight/scale loaders,
│                                   # consolidated save, optimizer patch.
├── compressor/qat/
│   ├── qat.py                      # ZeRO-3 branch for convert() / save()
│   ├── modules/quantizer.py        # init scales from init_value when
│                                   # weight data is not accessible; dtype
│                                   # alignment for DeepSpeed autocast.
│   ├── plugins/learnable_scale.py  # stream_load_scales + skip lazy_init
│   │                               # + gathered quant_inplace
│   └── trainers/end2end_trainer.py # lm_loss + kd_loss composition with
│                                   # cakld support + per-component logging
├── data/text_dataset.py            # supervise ONLY the last assistant turn
├── models/base_model.py            # from_pretrained → ZeRO-3 path
├── engine.py                       # normalise ``device_map`` string
└── utils/
    ├── config_parser.py            # QATTrainingConfig new fields
    ├── utils.py                    # set_op_by_name handles string-indexed
    └── __init__.py                 # re-export zero3_io helpers
```

### 执行流程（ZeRO-3 路径）

```
tools/run.py
  └── _prewarm_hf_deepspeed_config()             # register HfTrainerDeepSpeedConfig
  └── Engine.prepare_model()
      └── BaseLLMModel.from_pretrained()         # ZeRO-3 branch
          ├── zero3_empty_model_from_pretrained()
          │   ├── AutoModelForCausalLM.from_config(...)
          │   │   # triggers deepspeed.zero.Init for every Parameter
          │   └── linearize_moe_experts_empty()
          │       # fused Qwen3MoeExperts → empty LinearizedMoeExperts
          └── stream_load_weights()              # rank0 reads safetensors
  └── QAT.__init__()
      ├── init_ptq() → Qwen.replace_moe()        # no-op: already linearised
      └── register LearnableScalePlugin(from_ptq_ckpt_dir=...)
  └── LearnableScalePlugin.before_train()
      ├── replace nn.Linear with QuantLinear
      │   └── Quantizer allocates scale Parameters using init_value
      │       (no dependency on weight data)
      └── stream_load_scales(from_ptq_ckpt)      # fill weight/act/kv scales
  └── End2EndTrainer.prepare_trainer()
      ├── _init_optimizer() with id-deduped scale/LWC params
      ├── patch_deepspeed_duplicate_check()      # scales may be tied
      └── HF Trainer.train()
          # student + teacher forward → lm_loss + kd_loss composition
  └── QAT.convert() + QAT.save()
      # rank0-only QDQModule + consolidated state_dict → single rank write
```

## 使用方法

### 前置条件

1. 安装依赖：`deepspeed`、`safetensors`、`compressed-tensors`（可选，用于读取导出的 fp8 checkpoint）。
2. 硬件：支持 NCCL 的多 GPU 节点；ZeRO-3 路径要求 `torchrun --nproc_per_node=N`。

### 完整两阶段流程

#### 阶段 1：PTQ 校准生成初始 scale

ZeRO-3 QAT 启动时**不再**跑 forward 校准，因此必须先产出一个带 scale 的 PTQ checkpoint。使用现有 PTQ 配置（单卡即可）：

```bash
python tools/run.py \
    -c configs/qwen3/ptq/fp8_static/qwen3-a3b_fp8_static.yaml \
    --model-path /path/to/Qwen3-30B-A3B \
    --save-path ./output_ptq_30b
```

产出的 `./output_ptq_30b/qwen3-a3b_fp8_static/model-*.safetensors` 中包含 `<layer>.weight_scale` 和 `<layer>.input_scale`。

#### 阶段 2：ZeRO-3 QAT

```bash
bash scripts/qat/run_qat_for_qwen_30b_a3b_zero3.sh
```

或直接：

```bash
torchrun --nproc_per_node=8 tools/run.py \
    -c configs/qwen3/qat/fp8_static/learn_scale/qwen3-30b-a3b_fp8_static_end2end_learn_scale_zero3.yaml
```

训练完成后，压缩权重会保存到 `./output/qwen3-30b-a3b_fp8_static_end2end_learn_scale_zero3/final_quant_checkpoint/`。

### 最小 4B 烟囱测试

快速验证流程（2 张卡、5 步训练）：

```bash
# Step 1: PTQ
python tools/run.py \
    -c configs/qwen3/ptq/fp8_static/qwen3-4b_fp8_static.yaml \
    --model-path Qwen/Qwen3-4B \
    --save-path ./output_ptq

# Step 2: QAT
bash scripts/qat/run_qat_for_qwen_4b_zero3.sh
```

## 配置说明

### ZeRO-3 QAT 新增 / 修改字段（`compression.QAT`）

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `from_ptq_ckpt` | str / null | `null` | PTQ `save_format="real"` 产出目录。ZeRO-3 下**必填**，否则 `before_train` 会报错。目录可指向最顶层（自动识别嵌套的 `final_quant_checkpoint/`）。 |
| `lm_loss_weight` | float | `1.0` | HF CausalLM CE loss 的权重。为 0 则不计算 lm loss，也不出现在日志中。 |
| `kd_loss_weight` | float | `0.0` | KD loss 的权重（使用 `loss_type` 选定的 KD 变体）。为 0 则不计算 KD，也不启动 teacher forward。 |

注：`lm_loss_weight` / `kd_loss_weight` 任意一个 > 0 即可参与训练；两者都为 0 会在 `compute_loss` 中抛错。

### KD 变体（`loss_type`）

| `loss_type` | 描述 |
|-------------|------|
| `kl` | `KL(teacher || student)`（forward KL），per-valid-token 平均 |
| `rkl` | `KL(student || teacher)`（reverse / backward KL），per-valid-token 平均 |
| `mse` | student 与 teacher logits 的 MSE（per-valid-token 平均） |
| `cakld` | Confidence-Aware KL Distillation：按 teacher 在 label 上的概率做 token-wise 的 `fkl` / `rkl` 混合，`conf * rkl + (1 - conf) * fkl` |
| `kd` | 经典 temperature KD：`T² * KL(soft_student || soft_teacher)`，保留兼容 |
| `kl_top_K` / `r_kl_top_K` | top-K token 上的 forward / reverse KL。`K` 可写在 `loss_type` 字符串里（例如 `kl_top_1000`），或通过 `loss_topk` 字段指定 |
| `origin` | 纯 HF CE loss（等价于 `kd_loss_weight = 0`） |

选择 `loss_type = cakld` 时的核心公式（参考 `_compute_kd_components`）：

```python
# 仅在 labels != -100 的 token 上计算
forward_kl  = KL(log_softmax(student), softmax(teacher))
backward_kl = KL(log_softmax(teacher), softmax(student))
conf        = softmax(teacher).gather(-1, label)      # teacher 对目标 token 的置信度
cakld       = (conf * backward_kl + (1 - conf) * forward_kl).mean()
```

### 训练日志

`QATSeq2SeqTrainer.log()` 会自动把下列指标注入 HF Trainer 的标准日志字典（所以 wandb / console / tqdm 都能看到），仅当对应权重 > 0 时才会出现：

| 指标 | 含义 |
|------|------|
| `lm_loss` | HF CausalLM CE loss（仅对 assistant 回复位置计算，见下） |
| `kd/<loss_type>` | 当前选定的 KD 主 loss（`cakld` / `kl` / ...） |
| `kd/forward_kl` | 诊断用：`KL(teacher || student)`，始终在 `kd_loss_weight > 0` 时打印 |
| `kd/backward_kl` | 诊断用：`KL(student || teacher)` |
| `total_loss` | `lm_loss_weight * lm_loss + kd_loss_weight * kd/<loss_type>` |

示例输出（30B-A3B，3 步训练）：

```
{'loss': 2.08, 'grad_norm': 43.1, 'learning_rate': 1e-6,
 'lm_loss': 1.23, 'kd/cakld': 0.0075, 'kd/forward_kl': 0.0075, 'kd/backward_kl': 0.0075,
 'total_loss': 1.24, 'epoch': 0.03}
```

### Dataset：仅监督最后一个 assistant 回复

对于 JSONL 格式的 SFT 数据（`messages` / `conversations` / `input+output` 三种 schema），`TextDataset._load_jsonl_data` 现在只对**最后一个 assistant 回复**位置计算 loss：

- 拼接 prompt（对话中最后一个 assistant 之前的所有 turn）并通过 `apply_chat_template(..., add_generation_prompt=True)` tokenize，得到 `prompt_len`。
- 拼接完整对话（含最后 assistant）tokenize 得到 `input_ids`，`labels = input_ids.clone()`。
- 把 `labels[:, :prompt_len]` 和 padding 位置都置为 `-100`，HF CausalLM loss 会自动忽略。

这与 HF CausalLM 内部的 shift 行为一致（`shift_logits[..., :-1]` 对齐 `shift_labels[..., 1:]`），**不需要**手动 `roll`。

### DeepSpeed 配置

参考 `configs/qwen3/qat/fp8_static/learn_scale/ds_config_zero3.json`。关键项：

```json
{
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

`hf_args` 中需要设 `bf16: true`（或 `fp16: true`）显式开启混合精度，否则 DeepSpeed 的 `zero3_linear_wrap` 会默认用 fp16 autocast，而模型权重是 bf16，导致 dtype 失配。

### Quantizer 初始值

当 ZeRO-3 启用（或 weight 是 ZeRO-3 sharded / meta / 0 numel 的任一情况）时，`Quantizer._init_quant_params` 不再依赖权重数值，而是直接按 shape 创建 `nn.Parameter`：

| 配置项 | 默认值 | 描述 |
|------|--------|------|
| `weight_scale_init_value` | `1.0` | Weight quantizer scale 的初始值，在 `from_ptq_ckpt` 未命中时保留 |
| `activation_scale_init_value` | `1.0` | Activation quantizer scale 的初始值 |

典型场景下 PTQ 产出的 weight scale 约为 `max(|W|) / 448 ≈ 1e-3`，建议在 yaml 里把 init value 设为 `0.1` 左右作为保底（`from_ptq_ckpt` 命中时这些值会被覆盖）。

### 示例 yaml 关键片段

```yaml
compression:
  name: QAT
  quantization:
    name: fp8_static
    quant_method:
      weight: per-tensor
      activation: per-tensor
    ignore_layers: ["lm_head", "embed_tokens", "gate.weight"]
  QAT:
    hf_dataset: null
    from_ptq_ckpt: ./output_ptq_30b/qwen3-a3b_fp8_static
    training_mode: end2end
    dist_mode: hf
    save_format: real
    loss_type: cakld
    lm_loss_weight: 1.0
    kd_loss_weight: 1.0
    plugin_config:
      enable_scale: true
      quant_config:
        use_weight_quant: true
        use_activation_quant: true
        weight_scale_init_value: 0.1
        activation_scale_init_value: 0.1
        learnable:
          act_scale: false
          weight_scale: true
          kv_scale: false
          norm: false
          lwc: false
    hf_args:
      bf16: true
      per_device_train_batch_size: 1
      learning_rate: 1.0e-6
      gradient_checkpointing: true
      deepspeed: configs/qwen3/qat/fp8_static/learn_scale/ds_config_zero3.json
```

完整示例：
- `configs/qwen3/qat/fp8_static/learn_scale/qwen3-4b_fp8_static_end2end_learn_scale_zero3.yaml`
- `configs/qwen3/qat/fp8_static/learn_scale/qwen3-30b-a3b_fp8_static_end2end_learn_scale_zero3.yaml`

## 核心实现要点

### `zero3_io.py` 中的主要 API

| 函数 / 类 | 作用 |
|-----------|------|
| `is_deepspeed_zero3_enabled()` | 通过 HF 的 `HfTrainerDeepSpeedConfig` 弱引用判断是否已注册 ZeRO-3 配置 |
| `is_zero3_param(p)` | 判断 Parameter 是否带有 `ds_id / ds_status / ds_numel / ds_tensor` 元数据 |
| `gathered_param_if_zero3(p, modifier_rank=None)` | 上下文管理器；非 ZeRO-3 参数为 no-op |
| `LinearizedMoeExperts` | 通用空 per-expert `nn.Linear` 容器，`forward` 与 HF fused 等价 |
| `linearize_moe_experts_empty(model)` | duck-typing 扫描并原地替换，在 `deepspeed.zero.Init` 内构造以直接 partition |
| `zero3_empty_model_from_pretrained(model_path, ...)` | `no_init_weights` + `from_config` 构造空模型 + 自动拆 MoE |
| `stream_load_weights(model, model_path)` | 流式灌权，支持 fused MoE key 的 per-expert 切片分发 |
| `stream_load_scales(model, ckpt_dir)` | 流式灌 scale，支持 `.weight_scale / .input_scale / .k_cache.scale / .v_cache.scale` |
| `save_via_model_save_func(quant_model, save_func, path, prebuilt_state_dict)` | 只在 rank0 调用原 save_func，`state_dict()` 被 patch 返回合并后的字典 |
| `patch_deepspeed_duplicate_check()` | 置空 `DeepSpeedEngine._check_for_duplicates`，允许 tied scale 参数 |

### `Quantizer` 的变更

- 新增 `weight_shape` 构造参数：`QuantLinear.__init__` 传入 `(out_features, in_features)`，使得 Quantizer 在不访问权重数据的情况下也能计算 scale 的形状。
- 新增 `weight_scale_init_value` / `activation_scale_init_value` 配置项；`_init_quant_params` / `_init_lwc_params` 在 `_needs_external_weight_init(x)` 为真时使用。
- `QuantLinear.forward` 末尾的 `F.linear` 现在包在 `torch.amp.autocast(device_type="cuda", enabled=False)` 中，并在调用前把 `input.dtype` 对齐到 `weight.dtype`，以避免 DeepSpeed `zero3_linear_wrap` 的 autocast 把 bf16 input 回转成 fp16。
- `fake_quant` 末尾把 `out` cast 回 `x.dtype`，防止 `bf16 * fp32 = fp32` 的 dtype 泄漏。

### 优化器去重

`End2EndTrainer._init_optimizer` 使用 `_unique_named_params(...)` 根据 `id()` 去重收集 trainable 的 scale / LWC 参数，避免同一 Parameter 被多次加入 param group（在 MoE expert 共享 tensor 的场景下会触发）。配合 `patch_deepspeed_duplicate_check()` 即可通过 DeepSpeed 的安全检查。

### `convert` + `save` 的内存控制

`QAT.convert` 在 ZeRO-3 路径下：

1. 对每个 `QuantLinear`：**所有 rank** 都进入 `gathered_param_if_zero3` 拿到完整 weight（NCCL collective 保持对称），但**只 rank0** 保留 CPU clone。
2. rank0 在临时 `QDQModule` 内部跑一次 fp8/int 量化，把 `weight / weight_scale / input_scale / bias` 取出塞进 `self._rank0_state_dict`，随后丢弃临时模块。
3. **不修改模型结构**：保持 `QuantLinear`，使得第二轮扫描（收集非 QuantLinear 参数，如 embed、lm_head、layernorm、MoE router gate）在所有 rank 上 `named_parameters` 顺序一致，collective gather 不会死锁。
4. `QAT.save` 把 `_rank0_state_dict` 透传给 `save_via_model_save_func`，后者 patch `hf_model.state_dict`，**只 rank0** 调原 `save_func.save(...)`。

rank>0 convert 阶段 CPU 峰值 ≈ 一层的完整 weight（几十 MB 到 GB 量级）；rank0 峰值 ≈ 累积的合并 state_dict（完整模型大小）。

## 已验证场景

| 场景 | 模型 | 硬件 | 结果 |
|------|------|------|------|
| Dense ZeRO-3 QAT | Qwen3-4B | 2×H20 | ✓ PTQ→QAT→save 打通，产物能被 transformers 加载 |
| MoE ZeRO-3 QAT | Qwen3-30B-A3B（48 层 × 128 experts）| 8×H20 | ✓ `stream_load_scales` 命中 37248 个 scale，训练 loss 稳定，输出 31 GB fp8 checkpoint |
| KD + LM loss 组合 | 同上 | 同上 | ✓ `lm_loss / kd/cakld / kd/forward_kl / kd/backward_kl / total_loss` 按权重打印 |
| 最后 assistant 仅监督 | TextDataset(jsonl) | - | ✓ 首个 valid label idx 落在 `<\|im_start\|>assistant\n` 之后 |
| 非 ZeRO-3 回归 | Qwen3-4B 单卡 PTQ | 1×H20 | ✓ 行为与 main 一致，无回归 |

