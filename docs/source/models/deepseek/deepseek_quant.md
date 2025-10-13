# DeepSeek量化

DeepSeekR1模型主要采用FP8-Static和W4A8两种方案进行模型压缩

## FP8量化

DeepSeekR1的FP8量化为per-tensor粒度。

您可以量化`AngelSlim/configs/deepseek_r1`下面带有fp8_static字段的模型类型。

### 配置

FP8 `confg.yaml`文件参数配置，您可以参考`config/deepseek_r1/fp8_static`路径下的文件，下面是参数信息介绍。

#### model配置
- `name`：填写`DeepSeek`。
- `torch_dtype`：权重加载时使用的数据类型。设置为`fp8`时可直接加载HF上的deepseek-ai/DeepSeek-R1-0528模型。设置为`bf16`时需要将fp8权重进行转换。
- `low_cpu_mem_usage`：设置为True时表示使用单卡运行，权重存放在CPU上，同时`quantization.low_memory`也要设置为`true`,并且`model.device_map`填写`cpu`。

#### Compression配置
- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`fp8_static`。
- `quantization.bits`：量化比特数，目前支持8bit。
- `quantization.low_memory`: 设置为`true`表示使用单卡运行。
- `quantization.quant_method`：主要指定权重的量化粒度，FP8为`per-tensor`。
- `quantization.ignore_layers`：指定模型中不需要量化的层。

### FP8量化

您可以通过下面代码启动FP8量化流程。支持多机多卡以及单卡校准。

#### 多卡
多卡模式需要--multi-nodes参数指定。单机8卡H20可实现DeepSeek权重以`fp8`类型加载的量化。
```shell
NNODES={运行的节点数}
NPROC_PER_NODE={每个节点的卡数}
MASTER_ADDR={主节点ip地址}
CONFIG=configs/deepseek_r1/fp8_static/deepseek_r1_fp8_static.yaml

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $NPROC_PER_NODE \
    --node-rank {节点序号} \
    --master-addr $MASTER_ADDR \
    tools/run.py \
    --config $CONFIG \
    --multi-nodes
```

双机16卡运行时，则每台机器上都执行以上脚本。

#### 单卡
```shell
python3 tools/run.py -c configs/deepseek_r1/fp8_static/deepseek_r1_fp8_static_low_memmory.yaml
```

### 部署
vLLM支持DeepSeek的per-tensor量化，已验证vllm==0.8.5可部署。

若使用单机8卡141GB H20，您可以修改`AngelSlim/deploy/run_vllm.sh`中的`MODEL_PATH`字段后通过以下命令使用：

```shell
cd AngelSlim/deploy
sh run_vllm.sh
```

若使用96GB H20，需要双机部署

## W4A8-FP8量化

DeepSeekR1的W4A8-FP8量化，其中权重为per-group的粒度，group-size为128，激活为per-tensor的粒度。

您可以量化`AngelSlim/configs/deepseek_r1`下面带有w4a8_fp8字段的模型类型。

### 配置

W4A8-FP8 `confg.yaml`文件参数配置，您可以参考`config/deepseek_r1/w4a8_fp8`路径下的文件，下面是参数信息介绍。

#### model配置
- `name`：填写`DeepSeek`。
- `torch_dtype`：权重加载时使用的数据类型。设置为`fp8`时可直接加载HF上的deepseek-ai/DeepSeek-R1-0528模型。设置为`bf16`时需要将fp8权重进行转换后再进行量化。
- `low_cpu_mem_usage`：设置为True时表示使用单卡运行，权重存放在CPU上，同时`quantization.low_memory`也要设置为`true`,并且`model.device_map`填写`cpu`。

#### Compression配置
- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`w4a8_fp8`。
- `quantization.bits`：量化比特数，W4A8-FP8下不起作用。
- `quantization.low_memory`: 设置为`true`表示使用单卡运行。
- `quantization.quant_method`：主要指定权重的量化粒度，W4A8-FP8为`per-group`。
- `quantization.group_size`：权重量化分组数。
- `quantization.ignore_layers`：指定模型中不需要量化的层。

### W4A8-FP8量化

您可以通过下面代码启动W4A8-FP8量化流程。支持多机多卡以及单卡校准。

#### 多机多卡
多机多卡模式需要--multi-nodes参数指定
```shell
NNODES={运行的节点数}
NPROC_PER_NODE={每个节点的卡数}
MASTER_ADDR={主节点ip地址}
CONFIG=configs/deepseek_r1/w4a8_fp8/deepseek_r1_w4a8_fp8.yaml

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $NPROC_PER_NODE \
    --node-rank {节点序号} \
    --master-addr $MASTER_ADDR \
    tools/run.py \
    --config $CONFIG \
    --multi-nodes
```

#### 单卡
```shell
python3 tools/run.py -c configs/deepseek_r1/w4a8_fp8/deepseek_r1_w4a8_fp8_low_memmory.yaml
```

## INT4-AWQ量化

DeepSeekR1的INT4-AWQ量化，可使用vllm部署。其中权重为per-group的粒度，group-size可选64/128；激活为动态per-token量化，可支持激活数据类型为int8/fp8。具体可见对应PR:
```shell
https://github.com/vllm-project/vllm/pull/24722
```


您可以量化`AngelSlim/configs/deepseek_r1`下面带有`int4_awq`字段的模型类型。

### 配置

INT4-AWQ `confg.yaml`文件参数配置，您可以参考`config/deepseek_r1/int4_awq`路径下的文件，下面是参数信息介绍。

#### model配置
- `name`：填写`DeepSeek`。
- `torch_dtype`：权重加载时使用的数据类型。设置为`fp8`时可直接加载HF上的deepseek-ai/DeepSeek-R1-0528模型。设置为`bf16`时需要将fp8权重进行转换后再进行量化。
- `device_map`：设置为`cpu`。

#### Compression配置
- `name`：压缩策略，填写`PTQ`。
- `quantization.name`：压缩算法选填`int4_awq`。
- `quantization.bits`：量化比特数，设置为4。
- `quantization.quant_method`：主要指定权重的量化粒度，设置为`per-group`。
- `quantization.group_size`：权重量化分组数。
- `quantization.zero_point`：权重量化偏置，设置为`True`。
- `quantization.ignore_layers`：指定模型中不需要量化的层。

### INT4-AWQ量化

您可以通过下面代码启动INT4-AWQ量化流程:
```shell
python3 tools/run.py --config configs/deepseek_r1/int4_awq/deepseek_r1_int4_awq.yaml
```