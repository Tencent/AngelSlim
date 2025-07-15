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
from angelSlim import engine
engine.get_supported_compress_method()
```


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
