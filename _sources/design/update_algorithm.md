# 新增模型与算法


## 新增模型

以`Qwen`为例，所有新定义的模型继承自`BaseModel`，然后使用`@SlimModelFactory.register`来进行注册即可，示例如下：

```python
@SlimModelFactory.register
class Qwen(BaseModel):
    def __init__(
        self,
        model=None,
        model_path=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            model_path=model_path,
            deploy_backend=deploy_backend,
        )
```

在`SlimModelFactory`中会自动创建模型全局列表，调用时直接指定模型的Class名字即可，比如：`Qwen`，创建model如下所示：

```python
self.slim_model = SlimModelFactory.create(
   "Qwen", model=self.model, deploy_backend=deploy_backend
)
```


## 新增压缩算法

量化目前支持`FP8`、`INT8`、`INT4`等策略，压缩通过`CompressorFactory`来统一注册压缩算法，比如PTQ的注册如下：

```python
@CompressorFactory.register
class PTQ:
    def __init__(self, model, slim_config=None):
        self.quant_model = model
        ...
```

在`CompressorFactory`中会自动创建压缩算法全局列表，调用时直接指定压缩算法的Class名字即可，比如创建PTQ，如下所示：

```python
slim_config = {
   "global_config": global_config,
   "compress_config": compress_config,
}
self.compressor = CompressorFactory.create(
   "PTQ", self.slim_model, slim_config=slim_config
)

```
