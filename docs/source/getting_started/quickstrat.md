# 快速开始

## 环境准备

推荐使用pip直接安装最新版本的`AngelSlim`：
```shell
pip install angelslim
```

如果需要编译或者自定义安装，具体可参考[安装文档](./installation.md)

## 运行

### 量化

量化我们目前支持`FP8`、`INT8`和`INT4`的量化方式，执行流程也非常简单，`tools/run.py`是执行脚本，通过`--config`或者`-c`指定模型压缩的yaml配置文件，即可一键式运行程序，例如执行`Qwen3-1.7B`模型的`fp8_static`量化流程，可以直接运行：

```shell
python3 tools/run.py -c configs/qwen3/fp8_static/qwen3-1_7b_fp8_static.yaml       
```

- 更多的模型、压缩策略具体请在`configs`文件夹中查看，工具中每类模型，不同压缩策略都有一个单独的yaml文件，开发者可以直接找到对应的yaml文件进行执行。

- 如果在量化时显存资源不足，可采取`low_memory`的模式，可以使用最少的显存进行模型量化过程，请参考[](../features/quantization/fp8.html#fp8-low-memory)

- 如果想要修改yaml配置文件中的内容，请参考[准备配置文件](../design/prepare_config)的详细教程文档。

:::{note}
如果Hugging Face模型下载地址访问不通，为了解决下载问题，建议您尝试从 ModelScope 进行模型下载。
:::

- 如果想要获取AngelSlim中支持的压缩策略列表，可以执行下面代码：

```python
from angelslim import engine
engine.get_supported_compress_method()
```
- fp8 block-wise量化可以使用并行GPU转化脚本，其中block_size是权重量化scale对应分块形状，`num_workers`是并行数

```shell
python3 tools/fp8_quant_blockwise.py \
    --block_size 128 128 \
    --num_workers 32 \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH}
```

### 投机采样

投机采样（Speculative Decoding）是一种加速大语言模型推理的技术，通过使用较小的辅助模型来预测后续token，然后由主模型进行验证，从而提高生成效率。AngelSlim提供了完整的Eagle3训练和基准测试工具。

#### 1. 训练Eagle3模型

Eagle3模型的训练分为两个步骤：数据准备和在线训练。

##### 1.1 准备训练数据

训练数据的准备分为两个步骤：启动vLLM server和生成采样数据。

**步骤1：启动vLLM server**

首先需要启动vLLM server来提供模型推理服务：

```shell
bash scripts/speculative/run_vllm_server.sh
```

**server配置说明：**
- 该脚本会启动目标基础模型的vLLM推理服务
- 确保服务器成功启动后再进行下一步数据生成
- 可以通过修改脚本中的参数来调整vLLM server配置（如vLLM启动参数、GPU数量等），来适应不同的目标模型

**步骤2：生成采样数据**

vLLM server启动后，使用 `scripts/speculative/generate_data_for_target_model.sh` 脚本生成训练数据：

```shell
bash scripts/speculative/generate_data_for_target_model.sh
```

**脚本功能说明：**
- 通过vLLM server调用目标基础模型对输入数据进行采样
- 生成 `.jsonl` 格式的训练数据集
- 数据将用于后续Eagle模型的在线训练

**脚本参数说明：**

在使用前，需要在脚本中配置以下参数：

- `DATA_NAME_OR_PATH`: 输入数据集的HF名称或本地路径
- `OUTPUT_DIR`: 生成的数据集输出路径
- `DATA_FORMAT`: 输入数据集的格式（sharegpt|ultrachat）
- `DATA_SHARD_SIZE`: 生成数据集的切分子集大小
- `BASE_PORT`: vLLM server的端口号

**注意事项：**
- 确保vLLM服务器已成功启动并正常运行
- 数据生成过程可能需要较长时间，取决于样本数量和模型规模

##### 1.2 在线训练

使用 `scripts/speculative/train_eagle3_online.sh` 脚本进行Eagle3模型的在线训练：

```shell
bash scripts/speculative/train_eagle3_online.sh
```

**脚本参数说明：**

在使用前，需要在脚本中配置以下参数：

- `TARGET_MODEL_NAME_OR_PATH`: 目标模型的HF名称或本地名称
- `DRAFT_MODEL_CONFIG_PATH`: 草稿模型的config路径
- `TRAIN_DATA_PATH`: 训练数据路径
- `EVAL_DATA_PATH`: 验证数据路径
- `OUTPUT_DIR`: Eagle3模型输出路径
- `MODEL_MAX_LENGTH`: 训练数据的最大长度
- `CHAT_TEMPLATE_TYPE`: 目标模型的数据模板类型

#### 2. 基准测试

AngelSlim提供了完整的Eagle3基准测试工具，用于评估投机采样的性能提升。

##### 2.1 基本用法

使用 `tools/spec_benchmark.py` 脚本进行投机采样基准测试：

```shell
python3 tools/spec_benchmark.py \
    --base-model-path ${BASE_MODEL_PATH} \
    --eagle-model-path ${EAGLE_MODEL_PATH} \
    --model-id ${MODEL_ID} \
    --mode both
```

##### 2.2 参数说明

**模型配置参数：**
- `--base-model-path`: 基础模型路径（必需）
- `--eagle-model-path`: Eagle辅助模型路径（必需）
- `--model-id`: 模型标识符（必需）

**基准测试配置：**
- `--bench-name`: 基准数据集名称，默认为 `mt_bench`, 可选【`alpaca`,`gsm8k`,`humaneval`,`mt_bench`】
- `--mode`: 执行模式，可选 `eagle`（仅投机采样）、`baseline`（仅基线）、`both`（两者都执行），默认为 `both`
- `--output-dir`: 结果输出目录

**生成参数：**
- `--temperature`: 采样温度，默认为 1.0
- `--max-new-token`: 最大生成token数，默认为 1024
- `--total-token`: 草稿树中的总节点数，默认为 60
- `--depth`: 树深度，默认为 5
- `--top-k`: Top-k采样，默认为 10

**硬件配置：**
- `--num-gpus-per-model`: 每个模型使用的GPU数量，默认为 1
- `--num-gpus-total`: 总GPU数量，默认为 1
- `--max-gpu-memory`: 每个GPU的最大内存限制

**其他设置：**
- `--seed`: 随机种子，默认为 42
- `--question-begin`: 问题起始索引（用于调试）
- `--question-end`: 问题结束索引（用于调试）
- `--no-metrics`: 跳过自动指标计算

##### 2.3 使用示例

**完整基准测试（推荐）：**
```shell
python3 tools/spec_benchmark.py \
    --base-model-path /path/to/base/model \
    --eagle-model-path /path/to/eagle/model \
    --model-id qwen3-8b \
    --mode both \
    --output-dir ./results \
    --max-new-token 512 \
    --temperature 0.0
```

**仅运行投机采样：**
```shell
python3 tools/spec_benchmark.py \
    --base-model-path /path/to/base/model \
    --eagle-model-path /path/to/eagle/model \
    --model-id qwen3-8b \
    --mode eagle
```

**多GPU配置：**
```shell
python3 tools/spec_benchmark.py \
    --base-model-path /path/to/base/model \
    --eagle-model-path /path/to/eagle/model \
    --model-id qwen3-8b \
    --num-gpus-per-model 1 \
    --num-gpus-total 8
```

##### 2.4 性能报告

运行完成后，工具会自动生成性能报告，包括：
- 投机采样与基线模型的性能对比
- 加速比统计
- 生成质量指标（如果启用）

结果将保存在指定的输出目录中，便于后续分析和比较。

## 部署

以`vLLM`部署为例，如果已经产出压缩好的模型，可以按照下面的方式快速部署，如果想要查看能多backend和更多部署测试方式，请查看[部署文档](../deployment/deploy.md)

- 环境配置：

  为保证vLLM环境的正常运行，请使用以下命令安装所需依赖：

    ```shell
    pip install "vllm>=0.8.5.post1"
    ```

- 启动兼容OpenAI格式的API服务
    
    以下指令将启动兼容OpenAI API格式的服务，默认在 http://0.0.0.0:8080 地址进行访问：

    ```shell
    python3 -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port 8080 \
        --model ${MODEL_PATH} \
        --tensor-parallel-size 4 \
        --pipeline_parallel_size 1 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096 \
        --trust-remote-code
    ```
    其中:
    - `MODEL_PATH`模型路径可以为本地路径或 huggingface 路径;
    - `--tensor-parallel-size`设置张量并行数;
    - `--gpu-memory-utilization`设置显存占用比例;
    - `--max-model-len`指定模型最大上下文长度。