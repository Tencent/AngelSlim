# Qwen3-Omni量化指南

Qwen3-Omni模型可采用**FP8（static、dynamic）** 方式进行模型压缩，以下是详细的量化配置与操作说明。


## FP8 量化（W8A8）

Qwen3-Omni的FP8量化采用**per-tensor粒度**，支持thinker与talker的llm模块的动态量化（dynamic）和静态量化（static）两种模式。

### 配置参数说明

FP8量化的配置文件可参考路径：`configs/qwen3_omni/fp8_static` 和 `configs/qwen3_omni/fp8_dynamic`，核心参数如下：

#### model配置
- `name`：模型名称，固定填写`Qwen_Omni`。
- `model_path`：可填写hugging face模型卡片名称或者本地路径。
- `use_audio_in_video`: 用于控制是否使用源视频的音频轨道
- `attn_implementation`: 模型中要使用的注意力实现，默认值为`default`，设为`flash_attention_2`可以降低GPU显存占用

#### compression配置
- `name`：压缩策略类型，固定选择量化模式`PTQ`。
- `quantization.name`：量化算法类型，根据需求选择`fp8_static`（静态量化）或`fp8_dynamic`（动态量化）。
- `quantization.bits`：量化比特数，FP8量化固定填写`8`。
- `quantization.quant_method`：权重量化粒度，FP8量化固定为`per-tensor`。

#### dataset配置
- `name`：数据集类型，固定选择`OmniDataset`。
- `data_path`：数据集路径，支持jsonl文件路径。自定义数据集需参考`dataset/omni_fake_data/fake_data.json`格式。

### 启动量化流程

若在`model`配置中设置了`attn_implementation`为`flash_attention_2`，需要另外安装`FlashAttention 2`：
```shell
pip install -U flash-attn --no-build-isolation

# ldd --version 如果 < 2.32，可降到 2.7.4.post1 以下版本
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

通过以下命令启动FP8量化校准：

```shell
# 动态FP8量化
python3 tools/run.py -c configs/qwen3_omni/fp8_dynamic/qwen3_omni_fp8_dynamic.yaml
```

```shell
# 静态FP8量化
python3 tools/run.py -c configs/qwen3_omni/fp8_static/qwen3_omni_fp8_static.yaml
```

## 模型部署

vLLM框架支持Qwen3-Omni的FP8（per-tensor）量化模型部署，建议使用官方部署方式：
### vllm
参考[https://github.com/QwenLM/Qwen3-Omni?tab=readme-ov-file#vllm-usage](URL)

### transformers
参考[https://github.com/QwenLM/Qwen3-Omni?tab=readme-ov-file#transformers-usage](URL)