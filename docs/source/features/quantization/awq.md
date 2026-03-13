(awq)=

# AWQ量化

本项目使用[AWQ](https://arxiv.org/abs/2306.00978) 算法来实现Int4-weight-only量化。算法通过少量校准数据（无需训练）统计激活值的幅度，对每个权重通道计算一个缩放系数s，放大重要权重的数值范围，使其在量化时保留更多信息。流程无需重新训练模型，仅需少量校准数据即可确定缩放系数，适合快速部署。



您可以量化`AngelSlim/configs`下面带有awq字段的模型类型，你也可以直接下载我们量化完成的开源模型使用：#TODO [hflink]

## 配置

AWQ `confg.yaml`文件参数配置，您可以参考`config/model/int4_awq`路径下的文件，下面是参数信息介绍。


- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`int4_awq`。
- `quantization.bits`：量化比特数，目前支持4bit和8bit。
- `quantization.quant_method`：主要指定权重和激活的量化粒度，AWQ为`per-group`等。
- `quantization.quant_method.group_size`：对整个权重矩阵量化时的每一组的channel的大小，通常为128和64。
- `quantization.quant_method.zero_point`：是否使用非对称量化，推荐设置`true`开启。
- `quantization.quant_method.mse_range`：是否开启AWQ中的自动clip权重功能，推荐设置`false`。


## 启动Int4-AWQ量化

您可以通过下面代码启动AWQ量化流程
```shell
python3 tools/run.py -c configs/qwen2_5/qwen2_5-7b_int4_awq.yaml
```

##部署
要使用 vLLM 运行 AWQ 模型，您可以修改`AngelSlim/scripts/deploy/run_vllm.sh`中的`MODEL_PATH`字段后通过以下命令使用：

```shell
cd AngelSlim/scripts/deploy
sh run_vllm.sh
```