# 压缩模型部署

## 1. 准备压缩模型

首先根据[快速开始文档](../getting_started/quickstrat.md)指引，完成量化并保存模型。量化相关配置会自动存入模型保存路径下的`config.json`文件，用于部署时正确加载量化模型。

根据快速开始文档中的指引，存储的压缩模型路径下config.json文件中，增加以下配置，在部署过程中用于正确加载量化模型。

以`fp8`静态量化为例，量化配置如下：

```
"quantization_config": {
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "input_activations": {
        "num_bits": 8,
        "strategy": "tensor",
        "dynamic": false,
        "type": "float"
      },
      "weights": {
        "num_bits": 8,
        "strategy": "tensor",
        "dynamic": false,
        "type": "float"
      }
    }
  },
  "format": "naive-quantized",
  "ignored_layers": [
    "lm_head"
  ],
  "quant_method": "compressed-tensors",
  "quantization_status": "compressed"
},
```

如果需要通过`transformers`加载量化模型，请在量化模型配置的`global`中设置`deploy_backend: huggingface`，或者直接手动将量化产出模型路径下`config.json`配置中的`ignored_layers`字段改为`ignore`。

测试transformers加载量化模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## 2. 启动服务

### vLLM

- 环境配置：

  为保证vLLM环境的正常运行，请使用以下命令安装所需依赖：

    ```shell
    pip install "vllm>=0.8.5.post1"
    ```

- 启动兼容OpenAI格式的API服务
    
    以下指令将启动兼容OpenAI API格式的服务，默认在 http://localhost:8080 地址进行访问：

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

    注意设置的显存占用比例需要能满足模型最大上下文长度的kvcache需求。

- 多节点启动API服务

  如果需要多节点多卡启动服务，需要每个节点上设置主节点地址`MASTER_IP`和`LOCAL_IP`，在所有节点上通过 ray 启动服务，vLLM提供了多节点服务脚本 [multi-node-serving.sh](https://github.com/vllm-project/vllm/blob/v0.8.5.post1/examples/online_serving/multi-node-serving.sh)。
  
  以两机十六卡上启动服务为例：

  ```shell
  if [ $LOCAL_IP = $MASTER_IP ]; then
      bash multi-node-serving.sh \
          leader --ray_cluster_size=2 --ray_port=6380 --ray_init_timeout=300; \ 
          python3 -m vllm.entrypoints.openai.api_server \
            --port 8080 \
            --model $MODEL_PATH \
            --tensor-parallel-size 8 \
            --pipeline_parallel_size 2 \
            --trust-remote-code \
            --max-model-len 4096
  else
      bash multi-node-serving.sh \
          worker --ray_address=$MASTER_IP --ray_port=6380; \
          python3 -m vllm.entrypoints.openai.api_server \
            --port 8080 \
            --model $MODEL_PATH \
            --tensor-parallel-size 8 \
            --pipeline_parallel_size 2 \
            --trust-remote-code \
            --max-model-len 4096
  fi
  ```

  在两节点都执行上述指令后，可以在主节点的 http://localhost:8080 访问服务。
  
:::{note}
- 如果在部署Qwen3 MOE量化模型时遇到报错 `RuntimeError: CUDA error: no kernel image is available for execution on the device`，尝试将 `vllm` 版本升级到 `0.9.2`
- 相关参考Issue：[链接](https://github.com/vllm-project/vllm/issues/19690)
:::


### SGLang

- 环境配置：

    ```shell
    pip install "sglang[all]>=0.4.6.post1"
    ```

- 启动兼容OpenAI格式的API服务
    
    下面的指令会在本地 http://localhost:8080 启动服务：

    ```shell
    python -m sglang.launch_server \
        --host 0.0.0.0 \
        --port 8080 \
        --model-path $MODEL_PATH \
        --tp 4 \
        --mem-fraction-static 0.9 \
        --context-length 4096 \
        --trust-remote-code
    ```
    其中，模型路径`MODEL_PATH`可以为本地路径或 huggingface 路径，`--tp`设置张量并行数，`--mem-fraction-static`设置显存占用比例，`--max-model-len`指定模型最大上下文长度。

- 多节点启动API服务

  如果需要多节点多卡启动服务，在每个节点上执行启动服务指令，指定主节点地址`MASTER_IP`和端口`PORT`和该节点的`RANK`。

  以两机十六卡上启动服务为例：

```{eval-rst}
.. tabs::

   .. tab:: 主节点
      .. code-block:: bash
         :linenos:

         python3 -m sglang.launch_server \
           --model-path $MODEL_PATH \
           --tp 16 \
           --dist-init-addr $MASTER_IP:$PORT \
           --nnodes 2 \
           --node-rank 0

   .. tab:: 从节点
      .. code-block:: bash
         :linenos:

         python3 -m sglang.launch_server \
           --model-path $MODEL_PATH \
           --tp 16 \
           --dist-init-addr $MASTER_IP:$PORT \
           --nnodes 2 \
           --node-rank 1

```


## 3. API 调用

- 单轮对话（Legacy Completion）

  单轮输入，纯文本提示。

```{eval-rst}
.. tabs::

   .. tab:: Shell
      .. code-block:: bash
         :linenos:

         curl http://localhost:8080/v1/completions \
           -H 'Content-Type: application/json' \
           -d '{
             "model": "'"$MODEL_PATH"'",
             "prompt": "翻译为中文: Hello world",
             "max_tokens": 100,
             "temperature": 0.5,
             "top_p": 0.9,
             "top_k": 5
           }'

   .. tab:: Python
      .. code-block:: python
         :linenos:

         from openai import OpenAI
         client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")
         
         response = client.completions.create(
             model=MODEL_PATH,
             prompt="翻译为中文: Hello world",
             max_tokens=100,
             temperature=0.5,
             top_p=0.9,
             extra_body={"top_k": 5}
         )
         print(response.choices[0].text)

```

- 多轮对话（Chat Completion）

  支持 system/user/assistant 角色。

```{eval-rst}
.. tabs::


   .. tab:: Shell
      .. code-block:: bash
         :linenos:

         curl http://localhost:8080/v1/chat/completions \
           -H 'Content-Type: application/json' \
           -d '{
             "model": "'"$MODEL_PATH"'",
             "messages": [{"role": "user", "content": "翻译为中文: Hello world"}],
             "temperature": 0.5,
             "top_p": 0.9,
             "top_k": 5
           }'

   .. tab:: Python
      .. code-block:: python
         :linenos:

         from openai import OpenAI
         client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")
         
         response = client.chat.completions.create(
             model=MODEL_PATH,
             messages=[{"role": "user", "content": "翻译为中文: Hello world"}],
             temperature=0.5,
             top_p=0.9,
             extra_body={"top_k": 5}
         )
         print(response.choices[0].message.content)

```


- 流式输出

  在请求中设置`stream`字段为`true`，可以流式地返回请求输出：

```{eval-rst}
.. tabs::


   .. tab:: Shell
      .. code-block:: bash
         :linenos:

         curl http://localhost:8080/v1/chat/completions \
           -H 'Content-Type: application/json' \
           -d '{
             "model": "'"$MODEL_PATH"'",
             "messages": [{"role": "user", "content": "写一个故事"}],
             "stream": true,
             "temperature": 0.5,
             "top_p": 0.9,
             "top_k": 5
           }'

   .. tab:: Python
      .. code-block:: python
         :linenos:

         from openai import OpenAI
         client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")
         
         stream = client.chat.completions.create(
             model=MODEL_PATH,
             messages=[{"role": "user", "content": "写一个故事"}],
             stream=True,
             temperature=0.6,
             top_p=0.9,
             extra_body={"top_k": 5}
         )
         for chunk in stream:
             if content := chunk.choices[0].delta.content:
                 print(content, end="", flush=True)
```
