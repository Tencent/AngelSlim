(int8)=

# INT8量化

## W8A8-INT8 Dynamic量化

activation采用per-token动态量化，weight离线做per-channel静态量化。

运行示例如下：

```shell
python3 tools/run.py -c configs/qwen3/int8_dynamic/qwen3-0_6b_int8_dynamic.yaml
```

该配置文件中，量化相关参数如下：
- `name`：压缩策略，选填量化`quantization`。
- `quantization.name`：压缩算法选填`int8_dynamic`。
- `quantization.bits`：INT8量化对应填写8bit。
- `quantization.quant_method`：主要指定权重和激活的量化粒度为`per-tensor`。
- `quantization.ignore_layers`：需要忽略不进行量化的线性层。

```yaml
compression:
  name: PTQ
  quantization:
    name: int8_dynamic     # Supported: fp8_static, fp8_dynamic, int4_awq, int4_gptq, int8_dynamic
    bits: 8                # Quantization bits
    quant_method:
      weight: "per-channel"
      activation: "per-token"
    ignore_layers:         # Skip quantization for these layers
      - "lm_head"
```

## 产出模型

每个被量化的线性层保存：

- `weight`：8位INT数，形状为`[input_dim, output_dim]`
- `weight_scale`：用于反量化的scales，形状为`[input_dim, 1]`

配置文件`config.json`中量化相关配置：

```json
"quantization_config": {
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "input_activations": {
        "dynamic": true,
        "num_bits": 8,
        "strategy": "token",
        "type": "int"
      },
      "output_activations": null,
      "weights": {
        "dynamic": false,
        "num_bits": 8,
        "strategy": "channel",
        "type": "int"
      }
    }
  },
  "format": "int-quantized",
  "ignore": [
    "lm_head"
  ],
  "kv_cache_scheme": null,
  "quant_method": "compressed-tensors",
  "quantization_status": "compressed"
}
```

可参阅[vLLM INT8 文档](https://docs.vllm.ai/en/stable/features/quantization/int8.html)加载INT8量化模型配置要求。