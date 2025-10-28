(fp8_lepto)=

# FP8_LEPTO量化

通常情况下PTQ统计Activation和Weight的abs Max值作为量化缩放系数。通过观察FP8 PTQ量化后有损的模型的数值分布发现，相较于量化无损模型会出现激活值分布方差过大的情况，这种数值分布会使得量化数值落在FP8难以表达的量化范围，导致模型在一些数学难题或文本格式要求严格的任务上损失过大。


通过观察原始精度数值分布发现，该权重整体数值集中分布为尖峰分布，存在明显的outlier且大部分数据集中在0点附近，数值之间的相对距离较小导致计算过程中对于数值的精度要求更高。FP8 QDQ量化后的权重分布如右图所示，可以发现量化后的分布对比原始精度较为平滑。由于FP8-E4M3的数值表达在越靠近零点可表示的数值越多，趋近于正态分布的下的原始精度权重Worigin，通过传统FP8量化会导致原本数值较密集的数被平滑到了表达能力较差的FP8精度范围，导致精度表达能力下降带来效果损失。

针对上述FP8量化问题，我们推出了Leptokurtic Quant（LeptoQuant），一种通过隔离outlier将FP8权重映射范围集中至高精度区域的搜索策略。通常情况下对于权重和激活的量化难易度上激活值难度更高，因此我们着重优化激活的FP8。LEPTO将原始FP8的outlier值作为FP8精度表达上限，从而计算出新的Scale将数值分布压缩至高精度分布范围，使得量化后的激活值具有更好的精度表达。


运行示例如下：

```shell
python3 tools/run.py -c configs/qwen3/fp8_static/qwen3-0_6b_lepto_fp8_static.yaml
```

该配置文件中，量化相关参数如下：
- `name`：压缩策略。
- `quantization.name`：压缩算法填`fp8_lepto`。
- `quantization.bits`：fp8量化对应填写8bit。
- `quantization.quant_method`：主要指定权重和激活的量化粒度为`per-tensor`。
- `quantization.ignore_layers`：需要忽略不进行量化的线性层。

```yaml
compression:
  name: PTQ
  quantization:
    name: fp8_lepto
    bits: 8
    quant_method:
      weight: "per-tensor"
      activation: "per-tensor"
    ignore_layers:         # Skip quantization for these layers
      - "lm_head"
```

激活静态量化需要指定校准数据集，例如使用`sharegpt`数据集：

```yaml
dataset:
  name: TextDataset
  data_path: ./dataset/sharegpt_gpt4/sharegpt_gpt4_256.jsonl
  max_seq_length: 4096
  num_samples: 256
  batch_size: 1
```

数据集相关参数如下：
- `name`：校准数据集类型，文生文任务选择`TextDataset`。
- `data_path`：校准数据JSONL文件位置。
- `max_seq_length`：校准截断最大上下文长度。
- `num_samples`：校准最大样本个数。
- `batch_size`：校准批量大小。

支持数据格式详见[数据准备文档](../design/prepare_dataset.md)。


## 产出模型&部署

同FP8
