# 视觉 Token 压缩（端侧 VLM）

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)

视觉语言模型的图像 Token 数量会显著影响预填充时间与 KV Cache 占用。AngelSlim 的 Token 压缩模块可以在推理时减少视觉 Token，例如 IDPruner 算法。

安装多模态依赖与 lmms-eval 的步骤见 [Token 压缩安装指南](../../../features/token_compressor/installation.md)。使用 IDPruner 运行评测的示例如下：

```shell
python tools/run_token_pruning_evaluation.py \
    --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --configs "configs/qwen2_5_vl/pruning/idpruner_r0.75.yaml" \
    --tasks "textvqa" \
    --output_dir "./results/idpruner_test"
```

其中 `ratio` 控制保留的视觉 Token 比例，`mmr_lambda` 平衡重要性与多样性，`selector_path` 指定 Scorer 权重。更多剪枝算法及其新增方式见 [Token 压缩文档](../../../features/token_compressor/index.md)。

---

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)
