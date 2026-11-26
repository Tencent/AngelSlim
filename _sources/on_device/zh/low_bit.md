# 端侧低比特训练（QAT / QAD）

[← 量化（PTQ）](./quantization.md) ｜ [端侧部署 →](./deployment/index.md)

当 PTQ 无法满足端侧的精度或压缩率要求时，可以使用量化感知训练（QAT，Quantization-Aware Training）与量化感知蒸馏（QAD，Quantization-Aware Distillation）。它们在训练过程中插入伪量化算子，让模型学习适应量化误差。本页涵盖 1.25-bit、2-bit 与 FP4 等低比特方案的训练入口。

## 目录

- [1. 方法选择](#1-方法选择)
- [2. QAT：INT4 权重量化训练](#2-qatint4-权重量化训练)
- [3. QAT：W4A8-FP8 训练](#3-qatw4a8-fp8-训练)
- [4. QAD：带教师模型的量化蒸馏](#4-qad带教师模型的量化蒸馏)
- [5. 特殊权重量化器：Sherry / SEQ / LSQ / TWN / AbsMean / DLT](#5-特殊权重量化器sherry--seq--lsq--twn--absmean--dlt)
- [6. 1.25-bit：Sherry QAD](#6-125-bitsherry-qad)
- [7. 2-bit：翻译模型的 SEQ QAD](#7-2-bit翻译模型的-seq-qad)
- [8. 大模型的 ZeRO-3 QAT](#8-大模型的-zero-3-qat)
- [9. FP4 W4A4：FOCUS](#9-fp4-w4a4focus)
- [10. 训练后的评测（可选）](#10-训练后的评测可选)
- [11. 注意事项](#11-注意事项)

---

## 1. 方法选择

| 方法 | 教师模型 | 训练内容 | 典型用途 | 配置目录 |
| --- | --- | --- | --- | --- |
| QAT | 无 | 量化后的学生模型，可学习 scale | INT4 / W4A8-FP8 的精度恢复 | `configs/qwen3/qat/` |
| QAD | 有（全精度教师） | 量化后的学生模型，蒸馏损失 + LM 损失 | 更低比特（1.25 / 2 bit）或精度要求高的场景 | `configs/qwen3/qad/`、`configs/hunyuan/qad/` |
| Distill | 有 | 全精度学生模型（不量化） | 训练更小的端侧模型 | `configs/qwen3/distill/` |

简单判断：

- 已有 PTQ 精度可接受 → 直接部署，无需训练；
- INT4 / W4A8 精度不足 → QAT；
- 1.25-bit、2-bit 等极低比特 → QAD，并使用特殊权重量化器。

---

## 2. QAT：INT4 权重量化训练

以 Qwen3-4B 的 INT4 权重量化为例，使用端到端（end2end）训练，并启用可学习 scale。

```shell
python3 tools/run.py -c configs/qwen3/qat/int4_weight_only/learn_scale/qwen3-4b_int4_weight_only_end2end_learn_scale.yaml
```

关键配置：

```yaml
compression:
  name: QAT
  quantization:
    ignore_layers: ["lm_head", "embed_tokens"]
  QAT:
    hf_dataset: Salesforce/wikitext,wikitext-2-raw-v1   # 训练数据，会覆盖 dataset.data_path
    training_mode: end2end                              # end2end 或 blockwise
    dist_mode: hf
    save_format: fake                                   # fake / real，或不设置（跳过保存）
    do_train: true
    plugin_config:
      enable_scale: true                                # 启用可学习 scale
      quant_config:
        use_weight_quant: true
        use_activation_quant: false
        weight:
          qtype: int4
          granularity: per-group
          group_size: 128
          is_sym: true
    hf_args:                                            # 透传给 HuggingFace Trainer 的参数，如 learning_rate、num_train_epochs
      max_steps: 1000
```

若使用逐块训练（blockwise），将配置替换为：

```shell
python3 tools/run.py -c configs/qwen3/qat/int4_weight_only/learn_scale/qwen3-4b_int4_weight_only_blockwise_learn_scale.yaml
```

训练流程：插件在 `before_train` 阶段把 `nn.Linear` 替换为 `QuantLinear` 并做激活校准；训练结束后 `convert()` 把 `QuantLinear` 替换为推理用的 QDQ 模块并保存。完整架构见 [QAT 文档](../../features/quantization/qat.md)。

---

## 3. QAT：W4A8-FP8 训练

W4A8-FP8 表示权重为 4 bit、激活为 FP8。以 Qwen3-4B 为例：

```shell
python3 tools/run.py -c configs/qwen3/qat/w4a8_fp8/learn_scale/qwen3-4b_w4a8_fp8_end2end_learn_scale.yaml
```

该目录下还提供了若干变体：

| 配置 | 说明 |
| --- | --- |
| `qwen3-4b_w4a8_fp8_end2end_learn_scale.yaml` | 基础端到端版本 |
| `qwen3-4b_w4a8_fp8_end2end_learn_scale_lwc.yaml` | 增加 LWC（learnable weight clipping） |
| `qwen3-4b_w4a8_fp8_end2end_learn_scale_qkv_fp8attn.yaml` | 注意力 QKV 使用 FP8 |
| `qwen3-4b_w4a8_fp8_end2end_learn_scale_lwc_qkv_fp8attn.yaml` | 上述两项同时开启 |

---

## 4. QAD：带教师模型的量化蒸馏

QAD 使用独立的全精度教师模型指导量化学生模型训练，学生模型的所有量化参数可以按需训练。W4A8-FP8 的 QAD 示例：

```shell
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/qad/w4a8_fp8/qwen3-4b_w4a8_fp8_qad_zero2.yaml
```

关键字段：

```yaml
compression:
  name: QAD
  quantization:
    name: w4a8_fp8
  QAD:
    teacher_model_path: Qwen/Qwen3-4B      # 全精度教师
    student_type: quantized                # 学生为量化模型
    trainable_parameters: quant            # 只训练量化相关参数；all 为全部参数
    save_format: real
    plugin_config:
      enable_scale: true
```

QAD 的损失组合与 Distill 相同（`loss_type`、`kd_temperature`、`lm_loss_weight`、`kd_loss_weight` 等）。更多说明见 [QAD 文档](../../features/qad/index.md)。

---

## 5. 特殊权重量化器：Sherry / SEQ / LSQ / TWN / AbsMean / DLT

特殊权重量化器保留标准的 `QuantLinear` 封装，仅通过配置切换权重量化器实现。在 QAD 配置中，将 `weight_quantizer` 设为 `special`，并在 `special` 下指定 `quant_method`：

```yaml
plugin_config:
  enable_scale: true
  quant_config:
    use_weight_quant: true
    use_activation_quant: false
    weight_quantizer: special
    special:
      quant_method: sherry         # sherry / seq / lsq / twn / absmean / dlt
      granularity: per_group
      group_size: 128
      w_bits: 1
      N: 3
      M: 4
```

| `quant_method` | 含义 | 对应配置（`configs/qwen3/qad/special/`） |
| --- | --- | --- |
| `sherry` | 3:4 细粒度稀疏的 1.25-bit 三值量化，每 4 个权重中 3 个为 1-bit（{-1, +1}），1 个置零 | `qwen3-1_7b_sherry_qad_from_qwen3-4b_zero2.yaml` |
| `seq` | Stretched Elastic Quantization，2-bit 权重取值 {-1.5, -0.5, 0.5, 1.5} | `qwen3-1_7b_seq_qad_from_qwen3-4b_zero2.yaml` |
| `lsq` | Learned Step-size Quantization，可学习步长 | `qwen3-1_7b_lsq_qad_from_qwen3-4b_zero2.yaml` |
| `twn` | Ternary Weight Network 风格的三值量化 | `qwen3-1_7b_twn_qad_from_qwen3-4b_zero2.yaml` |
| `absmean` | 基于绝对均值的三值量化 | `qwen3-1_7b_absmean_qad_from_qwen3-4b_zero2.yaml` |
| `dlt` | DLT 量化器 | `qwen3-1_7b_dlt_qad_from_qwen3-4b_zero2.yaml` |

运行任意一种方法，只需替换配置路径：

```shell
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/qad/special/qwen3-1_7b_seq_qad_from_qwen3-4b_zero2.yaml
```

这些量化器的实现位于 `angelslim/compressor/qat/modules/special_quantizer.py`。

---

## 6. 1.25-bit：Sherry QAD

Sherry 是本项目 Hy-MT1.5 1.25-bit 模型所采用的方法（论文 [arXiv:2601.07892](https://arxiv.org/abs/2601.07892)，ACL 2026）。以 Qwen3-1.7B 为学生、Qwen3-4B 为教师的示例：

```shell
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/qad/special/qwen3-1_7b_sherry_qad_from_qwen3-4b_zero2.yaml
```

关键字段解释：

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `quant_method` | `sherry` | 使用 Sherry 量化器 |
| `w_bits` | `1` | 每个非零权重为 1 bit |
| `N` / `M` | `3` / `4` | 每 4 个权重中保留 3 个非零值 |
| `granularity` / `group_size` | `per_group` / `128` | 分组量化粒度 |

该示例配置中 `max_steps: 5`，用于验证流程。正式训练时请增大步数与数据量。

---

## 7. 2-bit：翻译模型的 SEQ QAD

Hy-MT1.5-1.8B-2bit 使用 SEQ 量化加量化感知蒸馏。仓库提供了对应的训练模板：

```text
configs/hunyuan/qad/special/hunyuan_seq_2bit_qad_zero2.yaml
```

运行前，需要替换配置中的三个占位符：

| 字段 | 替换为 |
| --- | --- |
| `model.model_path` | 你的翻译模型路径（学生模型） |
| `compression.QAD.teacher_model_path` | 全精度教师模型路径 |
| `dataset.data_path` | 翻译训练数据（JSONL） |

该配置的核心设置如下：

```yaml
compression:
  name: QAD
  quantization:
    name: w4a8_fp8
  QAD:
    student_type: quantized
    plugin_config:
      enable_scale: true
      quant_config:
        weight_quantizer: special
        special:
          quant_method: seq
          granularity: per_channel
          w_bits: 2
          N: 3
          M: 4
          max_eps: 0.0
dataset:
  name: TextDataset
  is_sft_data: true
```

启动命令：

```shell
torchrun --nproc_per_node=8 tools/run.py -c configs/hunyuan/qad/special/hunyuan_seq_2bit_qad_zero2.yaml
```

2-bit 量化的方法论与技术报告见 [AngelSlim 技术报告](https://arxiv.org/abs/2602.21233)。训练产物的端侧推理方法见 [llama.cpp 部署](./deployment/llamacpp.md)。

---

## 8. 大模型的 ZeRO-3 QAT

对于单卡无法容纳的大模型，可以使用 ZeRO-3 的 QAT 流程。其流程分为两步：先用 PTQ 生成带 scale 的初始 checkpoint，再进行 ZeRO-3 QAT。

```shell
# 阶段 1：PTQ 生成带 scale 的初始 checkpoint（单卡即可）
python tools/run.py \
    -c configs/qwen3/ptq/fp8_static/qwen3-4b_fp8_static.yaml \
    --model-path Qwen/Qwen3-4B \
    --save-path ./output_ptq

# 阶段 2：ZeRO-3 QAT（多卡）
torchrun --nproc_per_node=8 tools/run.py \
    -c configs/qwen3/qat/fp8_static/learn_scale/qwen3-4b_fp8_static_end2end_learn_scale_zero3.yaml
```

更大规模的 Qwen3-30B-A3B 对应配置为 `configs/qwen3/qat/fp8_static/learn_scale/qwen3-30b-a3b_fp8_static_end2end_learn_scale_zero3.yaml`，流程相同。

详细流程与配置字段见 [ZeRO-3 QAT 文档](../../features/quantization/qat_zero3.md)。

---

## 9. FP4 W4A4：FOCUS

FOCUS 是面向 MXFP4 与 NVFP4 W4A4 的后训练 scale 优化方法（论文 [arXiv:2608.01847](https://arxiv.org/abs/2608.01847)）。它冻结原始权重，仅通过端到端优化量化 scale 恢复 FP4 精度，产出仍为标准硬件格式。

环境准备：

```shell
pip install -e .
pip install deepspeed
```

运行入口为 `scripts/fp4/qwen-3-4b/run_focus_fp4.sh`，通过环境变量选择格式：

```shell
# MXFP4
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=mxfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh

# NVFP4
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=nvfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh
```

对应配置：

- `configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_zero3.yaml`
- `configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_zero3.yaml`

注意：`NPROC` 应与 `CUDA_VISIBLE_DEVICES` 中的 GPU 数量一致。FP4 W4A4 需要支持对应数据格式的硬件或推理引擎。

---

## 10. 训练后的评测（可选）

QAT 运行时可附加评测参数，包括 PPL 与 lm-eval 任务：

```shell
python3 tools/run.py \
    -c "configs/qwen3/qat/int4_weight_only/learn_scale/qwen3-4b_int4_weight_only_end2end_learn_scale.yaml" \
    --lm-eval --ppl-eval
```

- `--ppl-eval`：在 wikitext2、c4 等数据集上计算困惑度；
- `--lm-eval`：运行 lm-evaluation-harness 任务，任务由 `--lm-eval-task` 指定，默认为 `ceval-valid`。

---

## 11. 注意事项

- **显存**：QAD 需要同时加载教师与学生模型，请优先使用 ZeRO-2/ZeRO-3 配置（如 `configs/qwen3/qad/w4a8_fp8/ds_config_zero2.json`）。
- **数据**：QAT 默认使用配置中的 `hf_dataset`（如 wikitext），QAD 与 SEQ 训练推荐使用与目标任务一致的数据。
- **保存格式**：`save_format` 可设为 `fake` 或 `real`，不同格式的说明见 [QAT 文档的保存格式表](../../features/quantization/qat.md)。
- **样例规模**：仓库中的 QAD 样例（如 Sherry 的 `max_steps: 5`）用于验证流程，正式训练需按需调整步数、学习率与数据量。

下一步：[端侧部署 →](./deployment/index.md)
