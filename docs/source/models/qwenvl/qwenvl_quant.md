# Qwen2.5VL量化指南

Qwen2.5VL模型主要采用**FP8（static、dynamic）** 和**W4A16（AWQ、GPTQ）** 两种方式进行模型压缩，以下是详细的量化配置与操作说明。


## FP8 量化（W8A8）

Qwen2.5VL的FP8量化采用**per-tensor粒度**，支持动态量化（dynamic）和静态量化（static）两种模式。

### 配置参数说明

FP8量化的配置文件可参考路径：`config/qwen2_5_vl/fp8_dynamic` 和 `config/qwen2_5_vl/fp8_static`，核心参数如下：

#### model配置
- `name`：模型名称，固定填写`QwenVL`。
- `low_cpu_mem_usage`：启用单卡运行模式（权重存于CPU）时设为`True`，同时需将`model.device_map`设置为`cpu`。

#### compression配置
- `name`：压缩策略类型，固定选择量化模式`PTQ`。
- `quantization.name`：量化算法类型，根据需求选择`fp8_static`（静态量化）或`fp8_dynamic`（动态量化）。
- `quantization.bits`：量化比特数，FP8量化固定填写`8`。
- `quantization.quant_method`：权重量化粒度，FP8量化固定为`per-tensor`。
- `quantization.ignore_layers`：指定模型中无需量化的层（可选）。
- `quantization.quant_vit`：是否量化QwenVL模型中的ViT结构，可选`true`或`false`。

#### dataset配置
- `name`：数据集类型，固定选择`MultiModalDataset`。
- `data_path`：数据集路径，支持HuggingFace数据集（默认使用`HuggingFaceM4/ChartQA`）或jsonl文件路径。自定义数据集需参考`dataset/multimodal_fake_data/fake_data.json`格式。

### 启动量化流程

通过以下命令启动FP8量化校准：

```shell
# 动态FP8量化
python3 tools/run.py -c configs/qwen2_5_vl/fp8_dynamic/qwen2_5_vl-7b_fp8_dynamic.yaml
```

```shell
# 静态FP8量化
python3 tools/run.py -c configs/qwen2_5_vl/fp8_static/qwen2_5_vl-7b_fp8_static.yaml
```


## INT4 量化（W4A16）

QwenVL的W4A16量化中，权重采用**per-group粒度**（分组大小为128），激活不进行量化，支持AWQ和GPTQ两种算法。

### 配置参数说明

INT4量化的配置文件可参考路径：`config/qwen2_5_vl/int4_awq` 和 `config/qwen2_5_vl/int4_gptq`，核心参数如下：

#### model配置
- `name`：模型名称，固定填写`QwenVL`。
- `low_cpu_mem_usage`：启用单卡运行模式（权重存于CPU）时设为`True`，同时需将`model.device_map`设置为`cpu`。

#### compression配置
- `name`：压缩策略类型，固定选择`quantization`。
- `quantization.name`：量化算法类型，根据需求选择`int4_awq`或`int4_gptq`。
- `quantization.bits`：量化比特数，INT4量化固定填写`4`。
- `quantization.low_memory`：仅用于`int4_awq`量化，设为`true`可减少量化过程中的显存占用，默认`false`。
- `quantization.quant_method`：权重量化粒度，INT4量化固定为`per-group`。
- `quantization.group_size`：权重量化分组大小，默认值为`128`。
- `quantization.ignore_layers`：指定模型中无需量化的层（可选）。

### 启动量化流程

通过以下命令启动INT4量化校准：

```shell
# AWQ算法INT4量化
python3 tools/run.py -c configs/qwen2_5_vl/int4_awq/qwen2_5_vl-32b_int4_awq.yaml
```

```shell
# GPTQ算法INT4量化
python3 tools/run.py -c configs/qwen2_5_vl/int4_gptq/qwen2_5_vl-7b_int4_gptq.yaml
```


## 模型部署

vLLM框架支持QwenVL2.5的FP8（per-tensor）量化和INT4（AWQ、GPTQ）量化模型部署，建议使用`vllm==0.10.0`版本。

部署步骤：
1. 修改`AngelSlim/scripts/deploy/run_vllm.sh`中的`MODEL_PATH`字段，指定量化后的模型路径。
2. 执行以下命令启动部署：

```shell
cd AngelSlim/scripts/deploy
sh run_vllm.sh 
```