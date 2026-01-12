# 语音合成模型EAGLE3

[Eagle3](https://arxiv.org/pdf/2503.01840)是目前最常用、加速效果最好的投机采样算法。
本项目包括Eagle3的训练以及benchmark测试，并开源了Fun-CosyVoice3的[Eagle3权重](https://huggingface.co/collections/AngelSlim/eagle3)。

我们训练的Fun-CosyVoice3 Eagle3模型的表现可以参见基准测试[benchmarks](../../performance/speculative_decoding/benchmarks.md)

## 1. 支持模型列表
- `Fun-CosyVoice3`

## 2. 准备数据

### 2.1 数据组织形式

所有数据需保存在`jsonl`文件中，数据格式可参考：
- 原始训练数据示例：`AngelSlim/dataset/tts_fake_data/train.jsonl`
- 重采样训练数据示例：`AngelSlim/dataset/tts_fake_data/train_regenerate.jsonl`

### 2.2 原始训练数据

每行字段意义如下：
- `text`：输入文本
- `audio_path`：真实音频绝对路径
- `instruct`：发音人文本表示
- `instruct_audio_path`：发音人音频绝对路径

### 2.3 重采样训练数据（推荐）

为得到高质量的目标模型SFT数据，建议使用目标模型重新采样训练数据，将LLM生成的语音token保存在`jsonl`文件中，每行字段意义如下：
- `text`：输入文本
- `audio_tokens`：生成的语音token
- `instruct`：发音人文本表示
- `instruct_audio_path`：发音人音频绝对路径


## 3. 训练Eagle3模型

目前仅支持在线训练模式。

### 3.1 在线训练

使用下面的脚本进行Eagle3模型的在线训练：

```shell
bash scripts/speculative/train_eagle3_tts_online.sh
```

**脚本参数说明：**

在使用前，需要在脚本中配置以下参数：

- `TARGET_MODEL_NAME_OR_PATH`: 目标模型的HF名称或本地名称
- `DRAFT_MODEL_CONFIG_PATH`: 草稿模型的config路径
- `TRAIN_DATA_PATH`: 训练数据路径
- `OUTPUT_DIR`: Eagle3模型输出路径
- `MODEL_MAX_LENGTH`: 训练数据的最大长度


## 4. 基准测试

AngelSlim提供了HF backend的Eagle3基准测试脚本，用于评估投机采样的性能提升。

### 4.1 HF基准测试

`Fun-CosyVoice3`仅支持HF测试平均接收长度。

#### 4.1.1 基本用法

使用 `tools/spec_benchmark.py` 脚本进行投机采样基准测试：

```shell
python3 tools/spec_benchmark.py \
    --base-model-path ${BASE_MODEL_PATH} \
    --eagle-model-path ${EAGLE_MODEL_PATH} \
    --model-id ${MODEL_ID} \
    --mode both
```

#### 4.1.2 参数说明

**模型配置参数：**
- `--base-model-path`: 基础模型路径（必需）
- `--eagle-model-path`: Eagle辅助模型路径（必需）
- `--model-id`: 模型标识符（必需）

**基准测试配置：**
- `--bench-name`: 基准数据集名称，可参考【`tts_fake_data`】
- `--mode`: 执行模式，可选 `eagle`（仅投机采样）、`baseline`（仅基线）、`both`（两者都执行），默认为 `both`
- `--output-dir`: 结果输出目录

**生成参数：**
- `--temperature`: 采样温度，默认为 1.0
- `--max-new-token`: 最大生成token数，默认为 1024
- `--total-token`: 草稿树中的总节点数，默认为 60
- `--depth`: 树深度，默认为 5
- `--top-k`: Top-k采样，默认为 10
- `--generate-audio`: 是否生成最终音频

**硬件配置：**
- `--num-gpus-per-model`: 每个模型使用的GPU数量，默认为 1
- `--num-gpus-total`: 总GPU数量，默认为 1
- `--max-gpu-memory`: 每个GPU的最大内存限制

**其他设置：**
- `--seed`: 随机种子，默认为 42
- `--question-begin`: 问题起始索引（用于调试）
- `--question-end`: 问题结束索引（用于调试）
- `--no-metrics`: 跳过自动指标计算

**注意事项：**
- `--bench-name`: 也可以添加自定义测试集，在`AngelSlim/dataset`目录下创建新的子目录并将目录名作为`--bench-name`，在新目录下创建`question.jsonl`，框架会自动读取该文件
- `--temperature`: `Fun-CosyVoice3`在`temperature`为0时容易生成大量重复token，建议测试时使用默认配置

#### 4.1.3 使用示例

**完整基准测试：**
```shell
python3 tools/spec_benchmark.py \
    --base-model-path /path/to/base/model \
    --eagle-model-path /path/to/eagle/model \
    --model-id cosyvoice3 \
    --mode both \
    --output-dir ./results \
    --deploy-backend pytorch_tts \
    --generate-audio
```

**注意事项：**
- `Fun-CosyVoice3`在设置`generate-audio`为`True`时需要额外导入`cosyvoice`包，安装步骤如下：
    ```shell
    git clone https://github.com/FunAudioLLM/CosyVoice
    pip install hyperpyyaml omegaconf conformer diffusers hydra-core lightning gdown matplotlib wget x-transformers pyworld librosa
    ```

    测试脚本参考:
    ```shell
    export PYTHONPATH=/xxx/CosyVoice:/xxx/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH
    python3 tools/spec_benchmark.py [ARGS]
    ```

**不生成音频：**
```shell
python3 tools/spec_benchmark.py \
    --base-model-path /path/to/base/model \
    --eagle-model-path /path/to/eagle/model \
    --model-id cosyvoice3 \
    --mode both \
    --output-dir ./results \
    --deploy-backend pytorch_tts
```