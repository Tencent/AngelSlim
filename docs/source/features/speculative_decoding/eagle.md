# EAGLE
本项目将集成包括但不限于[Eagle](https://arxiv.org/pdf/2503.01840)系列的投机采样算法。
我们计划将对投机采样算法的代码以及部分开源大模型的Eagle3权重开源。
作为第一批开源内容，我们将提供Qwen3系列的[Eagle3权重](https://huggingface.co/collections/AngelSlim/qwen3-eagle-686787e3258f84fb09019f32)。
后续，更多的代码和其他大模型的权重也将陆续开源，敬请关注。
我们训练的Qwen3系列Eagle3模型的表现可以参见基准测试[benchmarks](../../performance/speculative_decoding/benchmarks.md)，
其中全部数据都是在单张H20上使用pytorch推理获得。

## 快速测试
目前sglang已经支持Qwen3-8B/14B/30B-A3B模型的eagle3部署，你可以选择使用sglang作为推理后端快速验证Eagle3模型的加速效果。
在已经安装sglang的环境中使用以下命令可以快速启动一个兼容Openai的服务，然后即可以通过本地端口进行请求了。
- 启动兼容OpenAI格式的API服务
    
    以下指令将启动兼容OpenAI API格式的服务，默认在 http://0.0.0.0:8080 地址进行访问：

    ```shell
    python3 -m sglang.launch_server \
        --model ${TARGET_MODEL_PATH_OR_NAME} \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path ${EAGLE3_MODEL_PATH} \
        --speculative-num-steps 6 \
        --speculative-eagle-topk 10 \
        --speculative-num-draft-tokens 32 \
        --mem-fraction 0.9  \
        --dtype bfloat16
    ```
    其中:
    - `TARGET_MODEL_PATH_OR_NAME`为本地路径或模型在huggingface上的名字;
    - `EAGLE3_MODEL_PATH`为Eagle3模型路径或在huggingface上的名字;

## 训练及创新
Comming soon.