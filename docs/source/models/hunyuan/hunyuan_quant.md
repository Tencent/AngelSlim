(hunyuan)=

# 混元模型量化

混元模型主要采用 FP8、Int4-GPTQ、Int4-AWQ 三种方案进行模型压缩

## FP8量化

FP8 量化采用 8 位浮点格式，通过少量校准数据（无需训练）预先确定全局缩放因子，将模型权重与激活值永久转换为FP8格式，提升推理效率并降低部署门槛。

您可以量化`AngelSlim/configs/hunyuan`下面带有fp8字段的模型类型，你也可以直接下载我们量化完成的开源模型使用：[Hunyuan](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)

### 配置

FP8 `confg.yaml`文件参数配置，您可以参考`config/hunyuan/fp8_static`路径下的文件，下面是参数信息介绍。


- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`fp8_static`。
- `quantization.bits`：量化比特数，目前支持8bit。
- `quantization.quant_method`：主要指定权重的量化粒度，FP8为`per-tensor`。
- `quantization.ignore_layers`：指定模型中不需要量化的层。

### FP8量化

您可以通过下面代码启动 FP8 量化流程
```shell
python3 tools/run.py -c configs/hunyuan/fp8_static/hunyuan_a13b_fp8_static.yaml
```

### 部署
要使用 vLLM 运行 FP8 模型，您可以修改`AngelSlim/deploy/run_vllm.sh`中的`MODEL_PATH`字段后通过以下命令使用：

```shell
cd AngelSlim/deploy
sh run_vllm.sh
```


## GPTQ量化

Int4-GPTQ 算法实现 Int4-weight-only 量化，该算法逐层处理模型权重，利用少量校准数据最小化量化后的权重重构误差，通过近似Hessian逆矩阵的优化过程逐层调整权重。流程无需重新训练模型，仅需少量校准数据即可量化权重，提升推理效率并降低部署门槛。


您可以量化`AngelSlim/configs/hunyuan`下面带有gptq字段的模型类型，你也可以直接下载我们量化完成的开源模型使用：[Hunyuan](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)

### 配置

GPTQ `confg.yaml`文件参数配置，您可以参考`config/hunyuan/int4_gptq`路径下的文件，下面是参数信息介绍。


- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`int4_gptq`。
- `quantization.bits`：量化比特数，目前支持4bit和8bit。
- `quantization.quant_method`：主要指定权重的量化粒度，GPTQ为`per-group`。
- `quantization.quant_method.group_size`：对整个权重矩阵量化时的每一组group的大小，通常为128和64。
- `quantization.ignore_layers`：指定模型中不需要量化的层。

### INT4-GPTQ量化

您可以通过下面代码启动 GPTQ 量化流程
```shell
python3 tools/run.py -c configs/hunyuan/int4_gptq/hunyuan_a13b_int4_gptq.yaml
```

### 部署
要使用 vLLM 运行 GPTQ 模型，您可以修改`AngelSlim/deploy/run_vllm.sh`中的`MODEL_PATH`字段后通过以下命令使用：

```shell
cd AngelSlim/deploy
sh run_vllm.sh
```


## AWQ量化

Int4-AWQ 算法实现 Int4-weight-only 量化，算法通过少量校准数据（无需训练）统计激活值的幅度，对每个权重通道计算一个缩放系数s，放大重要权重的数值范围，使其在量化时保留更多信息。流程无需重新训练模型，仅需少量校准数据即可确定缩放系数，适合快速部署。

您可以量化`AngelSlim/configs`下面带有awq字段的模型类型，你也可以直接下载我们量化完成的开源模型使用：[Hunyuan](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)

### 配置

AWQ `confg.yaml`文件参数配置，您可以参考`config/hunyuan/int4_awq`路径下的文件，下面是参数信息介绍。


- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`int4_awq`。
- `quantization.bits`：量化比特数，目前支持4bit和8bit。
- `quantization.quant_method`：主要指定权重和激活的量化粒度，AWQ为`per-group`等。
- `quantization.quant_method.group_size`：对整个权重矩阵量化时的每一组的channel的大小，通常为128和64。
- `quantization.quant_method.zero_point`：是否使用非对称量化，推荐设置`true`开启。
- `quantization.quant_method.mse_range`：是否开启AWQ中的自动clip权重功能，推荐设置`false`。

### INT4-AWQ量化

您可以通过下面代码启动 AWQ 量化流程
```shell
python3 tools/run.py -c configs/hunyuan/int4_awq/hunyuan_a13b_int4_awq.yaml
```

### 部署
要使用 vLLM 运行 AWQ 模型，您可以修改`AngelSlim/deploy/run_vllm.sh`中的`MODEL_PATH`字段后通过以下命令使用：

```shell
cd AngelSlim/deploy
sh run_vllm.sh
```
