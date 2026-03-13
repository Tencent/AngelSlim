(gptq)=

# GPTQ量化

本项目采用[GPTQ](https://arxiv.org/abs/2210.17323)算法实现Int4-weight-only量化，该算法逐层处理模型权重，利用少量校准数据最小化量化后的权重重构误差，通过近似Hessian逆矩阵的优化过程逐层调整权重。流程无需重新训练模型，仅需少量校准数据即可量化权重，提升推理效率并降低部署门槛。


您可以量化`AngelSlim/configs`下面带有gptq字段的模型类型，你也可以直接下载我们量化完成的开源模型使用：#TODO [hflink]

## 配置

GPTQ `confg.yaml`文件参数配置，您可以参考`config/model/int4_gptq`路径下的文件，下面是参数信息介绍。


- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`int4_gptq`。
- `quantization.bits`：量化比特数，目前支持4bit和8bit。
- `quantization.quant_method`：主要指定权重的量化粒度，GPTQ为`per-group`。
- `quantization.quant_method.group_size`：对整个权重矩阵量化时的每一组group的大小，通常为128和64。
- `quantization.ignore_layers`：指定模型中不需要量化的层。

## INT4-GPTQ量化

您可以通过下面代码启动GPTQ量化流程
```shell
python3 tools/run.py -c configs/qwen2_5/int4_gptq/qwen2_5-7b_int4_gptq.yaml
```

## 部署
要使用 vLLM 运行 GPTQ 模型，您可以修改`AngelSlim/scripts/deploy/run_vllm.sh`中的`MODEL_PATH`字段后通过以下命令使用：

```shell
cd AngelSlim/scripts/deploy
sh run_vllm.sh
```
