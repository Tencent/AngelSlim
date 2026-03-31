(daq)=

# DAQ量化

## DAQ Dynamic量化简介

**DAQ (Delta-Aware Quantization)** 是一个面向大语言模型（LLM）的免标定数据的后训练量化方法。标准量化方法以最小化重建误差为目标，但对基座模型无感知，导致量化噪声会不成比例地破坏编码后训练行为（如 SFT、RLHF、DPO）的小幅度参数增量（ΔW）。DAQ 通过将基于重建的优化目标替换为两个 delta-aware 指标——**Sign Preservation Rate（符号保持率）** 和 **Cosine Similarity（余弦相似度）**——来直接优化 ΔW 的方向保真度，仅需**基座模型**和**后训练模型**的权重矩阵，无需校准数据、激活统计量或 Hessian 估计。完整的方法论和实验细节请参阅[技术报告](https://arxiv.org/abs/2603.22324)。

## 核心特性

- **Delta-Aware 指标**：使用 Sign Preservation Rate 和 Cosine Similarity 优化 ΔW 的方向保真度，而非传统的重建误差
- **免数据**：仅需基座模型和后训练模型的权重矩阵——无需校准数据、激活统计量和 Hessian 估计
- **粗-精两阶段搜索**：先进行大范围粗搜索，再在最优候选附近精细搜索，实现高效的 scale 优化
- **多 GPU 支持**：支持多 GPU 并行处理
- **灵活的量化方式**：支持 block-wise 和 per-channel 两种量化方式

## 技术报告

关于 DAQ 方法论的完整描述——包括问题定义、指标定义、粗-精两阶段 scale 搜索算法以及详细的实验结果——请参阅[技术报告](https://arxiv.org/abs/2603.22324)。报告中还分析了为什么标准的基于重建的优化目标（如 MSE）会主动破坏后训练知识，并将 DAQ 与 AbsMax、SmoothQuant、AWQ 等基线方法进行了对比。


## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+
- safetensors
- huggingface_hub
- tqdm

## 使用方法

运行示例如下

```bash
python3 tools/run.py -c configs/deepseek_r1/fp8_daq/deepseek_r1_daq_fp8_w8a8_block.yaml
```

该配置文件中，量化相关参数如下：

该配置文件中，量化相关参数如下：
- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`daq`。
- `quantization.bits`：目标量化比特数，如fp8量化对应填写8bit。
- `quantization.base_model_path`：基座模型路径。
- `quantization.base_is_fp8`：基座模型是否是FP8格式。
- `quantization.metric`：优化指标，选填`sign`、`cosine`、`mse`。详细说明可参见[指标说明](#指标说明)或[技术报告](https://arxiv.org/abs/2603.22324)
- `quantization.quantization_method`：量化方式，选填`blockwise`、`per_channel`。详细说明可参见[量化方式](#量化方式)
- `quantization.scale_search.min_range`：scale 搜索最小倍率。
- `quantization.scale_search.max_range`：scale 搜索最大倍率。
- `quantization.scale_search.coarse_intervals`：粗搜索区间数。
- `quantization.scale_search.fine_intervals`：精搜索区间数。
- `quantization.scale_search.delta_threshold`：优化指标为`sign`时，判断Δweight是否有意义的阈值。小于该阈值的值不参与指标计算
- `quantization.num_workers`：多进程并行处理数。
- `quantization.gpus`：GPU ID 列表。
- `quantization.ignore_layers`：忽略量化层列表。

```yaml
compression:
  name: PTQ
  quantization:
    name: daq
    bits: 8
    base_model_path: deepseek-ai/DeepSeek-R1-Base # DAQ-specific: path to the base model
    base_is_fp8: true # Set to true if the base model is FP8 format
    metric: cosine    # Optimization metric: "sign"，"cosine"，or "mse"
    quantization_method: blockwise # Quantization method: "blockwise" or "per_channel"
    # Scale search hyper-parameters (optional, shown with defaults)
    scale_search:
      min_range: 0.8
      max_range: 1.5
      coarse_intervals: 5
      fine_intervals: 10
      delta_threshold: 1.0e-5
    # Multi-process parallelism
    num_workers: 8
    gpus: "0,1,2,3"
    # Skip quantization for these layers
    ignore_layers:
      - "lm_head"

```

## 指标说明

### 1. Sign Preservation Rate（符号保持率）

- 衡量量化后 `sign(ΔW_quant) == sign(ΔW_post)` 的元素比例
- 简单、可解释、对幅度差异鲁棒；值越高表示对微调方向的保持越好

### 2. Cosine Similarity（余弦相似度）

- 衡量原始 delta 向量与量化 delta 向量之间的方向对齐程度，归一化到 [-1, 1]
- 同时捕捉方向和相对幅度信息；值越高表示对微调方向的保持越好

### 3. Mean Squared Error（均方误差，MSE）

- 标准重建损失，衡量量化权重与原始权重之间的平方距离
- 注意：MSE **不是** delta-aware 的——它对基座模型无感知，并且可能主动破坏后训练知识（详见[技术报告](tech_report.pdf)）

## 量化方式

### 1. Block-wise 量化

- 将权重划分为 128×128 的块
- 为每个块计算最优 scale


### 2. Per-channel 量化

- 为每个通道（行）计算最优 scale
- 更加内存高效

## 使用场景

- **模型压缩**：缩减模型体积以便部署
- **边缘部署**：在资源受限的设备上实现 LLM 推理
- **微调模型优化**：在量化过程中保留微调带来的能力提升

## 量化后模型权重分析

DAQ 内置了分析工具（`tools/`），用于评估量化模型对微调信号的保持程度。

### 计算指标

- **Sign Preservation** — 逐元素检查 `sign(ΔW_quant) == sign(ΔW_sft)` 是否成立
- **Delta Cosine** — `ΔW_quant` 与 `ΔW_sft` 向量之间的余弦相似度
- **Distance Metrics** — 模型对之间的 L2、MSE、MAE、余弦相似度
- **Magnitude Breakdown** — 按 `|ΔW_sft|` 幅度分段统计符号保持率

### 用法

```bash
python tools/daq_analyze.py \
    --quantized-model /path/to/quantized/model \
    --sft-path /path/to/sft/model \
    --base-model /path/to/base/model \
    --quant-type block_fp8 \
    --output-report /path/to/report.json \
    --gpus 0,1,2,3 \
    --num-workers 4
```

### 分析选项

- `--quant-type`：量化格式（`block_fp8`、`channel_fp8`、`channel_int8`、`bf16`）
- `--output-report`：JSON 报告输出路径（可选）
- `--num-workers`：并行工作进程数
- `--gpus`：GPU ID 列表，逗号分隔（如 `0,1,2,3`）
- `--verbose`：在报告中包含逐权重详细信息
- `--block-size`：量化格式为 block_fp8时，block量化的块大小（默认：128）
