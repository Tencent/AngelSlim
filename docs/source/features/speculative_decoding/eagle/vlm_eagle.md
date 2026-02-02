# 视觉理解模型EAGLE3

[Eagle3](https://arxiv.org/pdf/2503.01840)是目前最常用、加速效果最好的投机采样算法。
本项目包括Eagle3的训练以及benchmark测试，并开源了HunyuanOCR和Qwen3-VL系列的[Eagle3权重](https://huggingface.co/collections/AngelSlim/eagle3)。

我们训练的HunyuanOCR和Qwen3-VL系列Eagle3模型的表现可以参见基准测试[benchmarks](../../../performance/speculative_decoding/benchmarks.md)，
其中全部数据都是在单张GPU上使用vLLM推理获得。
## 1. 支持模型列表
- `HunyuanOCR`
- `Qwen3-VL`

## 2. 准备数据

### 2.1 数据组织形式
所有数据需保存在jsonl文件中，训练数据格式可参考:

- 数据示例:

    ```json
    # 文生文
    {"id": "0", "conversations": [{"role": "user", "content": [{"type": "text", "text": "xxx"}]}, {"role": "assistant", "content": [{"type": "text", "text": "xxx"}]}]}

    # 图生文
    {"id": "1", "conversations": [{"role": "user", "content": [{"type": "image", "image": "xxx/images/0001.jpeg"}, {"type": "text", "text": "xxx"}]}, {"role": "assistant", "content": [{"type": "text", "text": "xxxx"}]}]}
    ```

- 典型字段意义如下：
    - id: 对话唯一标识
    - conversations: OpenAI 对话格式
        - image: 对应图像文件路径


### 2.2 数据生成

数据生成包括：1）为目标模型生成采样数据，2）为Eagle3模型离线生成目标模型的hidden states。

#### 2.2.1 为目标模型生成采样数据

生成采样数据为可选项，当有足够数量以及足够质量的目标模型SFT数据时，此步可略过。当训练数据和目标模型不配套时，则需要为目标模型重新采样生成数据。

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


#### 2.2.2 为Eagle3模型生成hidden states

目前仅支持以HF为后端生成hidden states，调用脚本如下：
```shell
# For HunyuanOCR
bash scripts/speculative/hunyuan_ocr/generate_vlm_hidden_for_draft_model.sh
# For Qwen3-VL series
bash scripts/speculative/qwen3_vl/generate_vlm_hidden_for_draft_model.sh
```
> 注意：qwen3_vl系列模型生成hidden states需要更新transformers>=5.0.0,
 或者cherry-pick: https://github.com/huggingface/transformers/pull/42609,
 否则抓取的hidden states不可用！！！

**脚本参数说明：**

在使用前，需要在脚本中配置以下参数：

- `DATASET_PATH`: 输入数据集的HF名称或本地路径
- `MODEL_NAME`: 目标模型的HF名称或本地路径
- `TARGET_BACKEND`: 目标模型后端，目前仅支持HF
- `MODEL_MAX_LENGTH`: 生成数据的上下文长度
- `CHAT_TEMPLATE_TYPE`: 目标模型的目标类型，目前支持qwen3_vl/hunyuan_vl
- `OUTPUT_DIR`: 生成的数据集输出路径


## 3. 训练Eagle3模型

目前支持在线训练和离线训练两种模式：在线训练适合显存足够、目标模型不大、训练上下文长度不要求极长的场景，
离线训练适合大尺寸目标模型、磁盘空间足够、长上下文训练场景。

### 3.1 在线训练

使用下面的脚本进行Eagle3模型的在线训练：

```shell
# For HunyuanOCR
bash scripts/speculative/hunyuan_ocr/train_eagle3_vlm_online.sh
# For Qwen3-VL series
bash scripts/speculative/qwen3_vl/train_eagle3_vlm_online.sh
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

### 3.2 离线训练

在离线训练前，必须要完成`2.2.2` 为Eagle3模型生成hidden states。
使用下面的脚本进行Eagle3模型的离线训练：

```shell
# For HunyuanOCR
bash scripts/speculative/hunyuan_ocr/train_eagle3_vlm_offline.sh
# For Qwen3-VL series
bash scripts/speculative/qwen3_vl/train_eagle3_vlm_offline.sh
```

**脚本参数说明：**

在使用前，需要在脚本中配置以下参数：

- `TARGET_MODEL_NAME_OR_PATH`: 目标模型的HF名称或本地名称
- `DRAFT_MODEL_CONFIG_PATH`: 草稿模型的config路径
- `TRAIN_DATA_PATH`: 训练数据路径,.jsonl格式
- `TRAIN_HIDDEN_PATH`: 训练hidden states数据路径
- `EVAL_HIDDEN_PATH`: 验证hidden states数据路径
- `OUTPUT_DIR`: Eagle3模型输出路径
- `MODEL_MAX_LENGTH`: 训练数据的最大长度
- `CHAT_TEMPLATE_TYPE`: 目标模型的数据模板类型
- `LM_HEAD_KEY`: 目标模型lm head的weight key名称，可以在model.safetensors.index.json中查看，默认为lm_head.weight时可不指定这个参数。当为model.embed_tokens.weight时，需要指定。
- `RUN_NAME`: 当`report_to`设为wand时，可以指定该参数设置wand中的run name。


## 4. 基准测试

AngelSlim提供了HunyuanOCR和Qwen3-VL系列模型vLLM backend的Eagle3基准测试脚本，用于评估投机采样的性能提升。

### 4.1 vLLM基准测试

> vLLM 适配参考: [Support Eagle3 for HunyuanOCR & Qwen3-VL](https://github.com/vllm-project/vllm/pull/32230)

#### 4.1.1 基本用法

使用 `tools/vllm_offline_eagle3_vlm_batch.py` 脚本进行投机采样基准测试：

```shell
python3 tools/vllm_offline_eagle3_vlm_batch.py \
    --base-model-path ${BASE_MODEL_PATH} \
    --eagle-model-path ${EAGLE_MODEL_PATH} \
    --output_file ${OUTPUT_FILE} \
    --use_eagle
```


#### 4.1.2 参数说明

**模型配置参数：**
- `--target_model`: 基础模型路径（必需）
- `--draft_model`: Eagle辅助模型路径（必需）

**基准测试配置：**
- `--dataset`: 基准数据集名称，默认为 `lmms-lab/textvqa`, 可选【`lmms-lab/textvqa`,`MMMU/MMMU`,`Lin-Chen/MMStar`,`opendatalab/OmniDocBench`,`Lin-Chen/MMStar`】
- `--use_eagle`: 运行Eagle3推理，默认为False
- `--output_file`: 输出结果文件路径
- `--num_prompts`: 测试用例数量，默认为100

**生成参数：**
- `--temp`: 采样温度，默认为 0
- `--max_model_len`: 最大上下文长度，默认为 32768
- `--output_len`: 最大生成token数，默认为 1024
- `--max_num_seqs`: 每次迭代的最大序列数，默认为 1
- `--num_spec_tokens`: draft model投机采样token数量，默认为2

**硬件配置：**
- `--tp`: 张量并行大小，默认为1

**其他设置：**
- `--seed`: 随机种子

#### 4.1.3 使用示例


**运行投机采样：**
```shell
python3 tools/vllm_offline_eagle3_vlm_batch.py \
    --target_model Qwen/Qwen3-VL-2B-Instruct \
    --draft_model "$EAGLE_DIR" \
    --use_eagle \
    --num_spec_tokens 4 \
    --dataset "$task" \
    --num_prompts 80 \
    --temp 0 \
    --max_num_seqs 1 \
    --output_len 1024 \
    --output_file "$OUTPUT_FILE"
```

**Baseline基准测试：**
```shell
python3 tools/vllm_offline_eagle3_vlm_batch.py \
    --target_model Qwen/Qwen3-VL-2B-Instruct \
    --num_spec_tokens 4 \
    --dataset "$task" \
    --num_prompts 80 \
    --temp 0 \
    --max_num_seqs 1 \
    --output_len 1024 \
    --output_file "$OUTPUT_FILE"
```

#### 4.1.4 性能报告

运行完成后，工具会自动生成性能报告，包括：
- 投机采样与基线模型的性能对比
- 加速比统计
- 生成质量指标（如果启用）

结果将保存在指定的输出目录中，便于后续分析和比较。

完整的vLLM benchmark结果可见[Benchmark](../../../performance/speculative_decoding/benchmarks.md)。