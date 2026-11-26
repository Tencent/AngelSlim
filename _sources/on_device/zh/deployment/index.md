# 部署总览与推理校验

[← 端侧量化与低比特训练](../low_bit.md) ｜ [← 返回总览](../index.md)

压缩后的模型有多条部署路线。本页先说明如何选择路线，并给出所有路线共用的推理校验方法（Transformers、`tools/infer.py`）与服务化方法（vLLM、SGLang）。各路线的详细步骤见本目录其它页面。

## 目录

- [1. 选择部署路线](#1-选择部署路线)
- [2. 推理校验](#2-推理校验)
- [3. vLLM 与 SGLang 服务化](#3-vllm-与-sglang-服务化)

---

## 1. 选择部署路线

| 部署目标 | 推荐路线 | 输入模型 | 页面 |
| --- | --- | --- | --- |
| 开发机快速验证量化结果 | Transformers | Hugging Face 格式的压缩模型 | 本页 [2](#2-推理校验) |
| 单机 GPU 服务化 | vLLM、SGLang | 压缩后的 Hugging Face 格式模型 | 本页 [3](#3-vllm-与-sglang-服务化) |
| PC / 手机 / CPU 端侧推理（HY-MT2、Hy-MT1.5、Hy4-flash） | llama.cpp | GGUF（含 STQ 低比特类型） | [llama.cpp 部署](./llamacpp.md) |
| Hy4-flash 的高性能本地推理（NVIDIA / Apple Silicon） | Strata | Hy4-flash GGUF | [Strata 部署](./strata.md) |
| 超大模型的多机本地运行 | prima.cpp | Hy4 preview GGUF | [多机部署（prima.cpp）](./prima_multi_machine.md) |
| Android 离线应用、Arm SME2 设备 | 官方 Demo / SME2 内核 | Hy-MT1.5 GGUF | [移动端部署](./mobile.md) |

如果你只想先确认压缩结果是否正确，从 [Transformers 校验](#2-推理校验) 开始；确认无误后再选择端侧路线。

---

## 2. 推理校验

在把模型部署到端侧之前，建议先在开发机上用 Transformers 校验模型能否正常加载与生成。

### 2.1 Transformers 加载与校验

量化产出模型的 `config.json` 包含 `quantization_config`。若要通过 Transformers 加载量化模型，可以在量化配置的 `global` 中设置 `deploy_backend: huggingface`，或者手动把 `config.json` 中的 `ignored_layers` 字段改为 `ignore`。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/your/compressed-model"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

也可以使用仓库自带的脚本完成同样的检查：

```shell
python3 scripts/deploy/offline.py /path/to/your/compressed-model "Hello, my name is"
```

### 2.2 tools/infer.py 推理

`tools/infer.py` 支持直接指定模型路径和输入提示词，也可以通过 YAML 配置运行：

```shell
python3 tools/infer.py \
    --model-path /path/to/your/compressed-model \
    --input-prompt "请把下面这句话翻译成英文：今天天气很好。" \
    --save-path ./output/
```

| 参数 | 说明 |
| --- | --- |
| `--model-path` | 压缩后模型路径 |
| `--input-prompt` | 输入提示词 |
| `-c, --config` | 使用 YAML 配置运行（可选） |
| `--save-path` | 输出目录，默认 `./output/` |
| `--output-file` | 额外把生成文本写入该文件 |
| `--print-full` | 打印提示词与生成内容，而不仅是新生成的 Token |

---

## 3. vLLM 与 SGLang 服务化

GPU 服务端或工作站上，可以用 vLLM 或 SGLang 暴露 OpenAI 兼容接口，用于验证服务化路径。

### 3.1 vLLM

```shell
pip install "vllm>=0.8.5.post1"
```

使用仓库脚本启动（会读取 `--model-path`、`--port`、`-d`、`-t`、`-g`、`--max-model-len` 等参数）：

```shell
cd AngelSlim/scripts/deploy
bash run_vllm.sh --model-path /path/to/your/compressed-model \
    --port 8080 \
    -d 0,1 \
    -t 2 \
    -g 0.8 \
    --max-model-len 4096
```

或直接使用 vLLM 的入口：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8080 \
    --model /path/to/your/compressed-model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code
```

### 3.2 SGLang

```shell
cd AngelSlim/scripts/deploy
bash run_sglang.sh --model-path /path/to/your/compressed-model -d 0,1 -t 2 --port 8080
```

### 3.3 调用测试

```shell
cd AngelSlim/scripts/deploy
bash openai.sh -m /path/to/your/compressed-model --port 8080 -p "你好，请介绍一下你自己。"
```

脚本支持 `--max-tokens`、`--temperature`、`--top-p`、`--top-k`、`--repetition-penalty` 与 `--system-prompt` 等采样参数，可通过 `-h` 查看完整说明。

---

下一步：[llama.cpp 部署 →](./llamacpp.md)
