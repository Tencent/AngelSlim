# 如何准备config

AngelSlim中运行模型压缩均采用解析yaml文件的

```yaml
# 全局配置
global:
  save_path: ./output
# 模型相关配置
model:
  name: Qwen
  model_path: Qwen/Qwen3-1.7B
  trust_remote_code: true
  low_cpu_mem_usage: true
  use_cache: false
  torch_dtype: auto
  device_map: auto

# 压缩算法相关配置
compression:
  name: 
  quantization:
    name: fp8_static
    bits: 8
    quant_method:
      weight: "per-tensor"
      activation: "per-tensor"
    ignore_layers:
      - "lm_head"

# 数据集相关配置
dataset:
  name: TextDataset
  data_path: ./dataset/sharegpt_gpt4/sharegpt_gpt4_256.jsonl
  max_seq_length: 4096
  num_samples: 256
  batch_size: 1

```

### 全局配置

- `save_path`：模型压缩后存储到本地的路径，默认存储到当前路径的`./output`文件夹下。

### model模型配置

- `name`：必须是工具中注册过的模型名，目前支持`Qwen`、`HunyuanDense`、`HunyuanMoE`、`Llama`、`DeepSeek`、`QwenVL`
- `model_path`：可以指定Hugging face网站的Model ID，如果本地没有缓存，可以直接在运行时从网络下载；同时`model_path`可以指定本地的具体绝对模型路径，只要符合Hugging face模型格式即可。
- 其他模型的超参对齐`transformers`中`AutoModelForCausalLM.from_pretrained`的超参。

### 压缩算法配置

- `name`：压缩策略，目前支持[`PTQ`, `speculative_decoding`]
- `quantization.name`：目前支持: `fp8_static`, `fp8_dynamic`, `int4_awq`, `int4_gpt`、`int8_dynamic`五种算法，后续会补充更多算法。
- `quantization.bits`：量化比特数，目前支持4bit和8bit。
- `quantization.quant_method`：主要指定权重和激活的量化粒度，分为`per-token`、`per-tensor`、`per-channel`、`per-group`等。
- `quantization.ignore_layers`：跳过量化的层的name。

## 数据集配置

- `name`：必须是工具中注册过的数据集名称，目前支持`TextDataset`、`MultiModalDataset`。
- `data_path`：数据集本地路径。
- `max_seq_length`：数据集的最大seq长度，query长度不足的会pad到`max_seq_length`。
- `num_samples`：压缩算法使用的样本数量，比如量化校准使用的校准集条数。
- `batch_size`：压缩算法执行时的batch_size大小。