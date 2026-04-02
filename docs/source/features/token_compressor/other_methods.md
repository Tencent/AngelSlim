# 其他视觉 Token 压缩方法

除了核心算法 IDPruner 外，AngelSlim 同样集成了多种业界主流的视觉 Token 压缩方案及基准（Baseline）方法：

- [**FastV**](https://arxiv.org/abs/2403.06764)
- [**DART**](https://arxiv.org/abs/2502.11494)
- [**DivPrune**](https://arxiv.org/abs/2503.02175)
- [**VisionSelector**](https://arxiv.org/abs/2510.16598)
- [**HiPrune**](https://arxiv.org/abs/2508.00553)
- [**VisionZip**](https://arxiv.org/abs/2412.04467)
- [**SCOPE**](https://arxiv.org/abs/2510.24214)
- [**VisPruner**](https://arxiv.org/abs/2412.01818)
- **Random** (随机剪枝基准)


## 运行示例

### 1. 功能验证 (Smoke Test)
使用 `tools/test_token_pruning.py` 快速验证指定策略在模型上的适配性与单次推理逻辑：

```bash
python tools/test_token_pruning.py \
    --config configs/qwen2_5_vl/pruning/scope_r0.75.yaml \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct"
```

**Smoke Test 参数说明：**
- `--config`: 指向对应的策略 YAML 配置文件。
- `--model_path`: 原始模型的本地路径或 HuggingFace ID。脚本将根据配置自动完成模型包装。


### 2. 精度评测 (Evaluation)
使用 `tools/run_token_pruning_evaluation.py` 在标准数据集上评估剪枝后的模型性能。以下示例演示如何使用 **SCOPE** 策略在 **TextVQA** 任务上运行评测：

```bash
python tools/run_token_pruning_evaluation.py \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --configs configs/qwen2_5_vl/pruning/scope_r0.75.yaml \
    --tasks textvqa \
    --batch_size 1 \
    --output_dir ./eval_results/scope_textvqa
```

**Evaluation 参数说明：**
- `--model_path`: 原始模型的路径或 HuggingFace ID。
- `--configs`: 指向一个或多个策略 YAML 配置文件（支持批量评测）。
- `--tasks`: 指定 `lmms-eval` 支持的任务名称（如 `textvqa`, `mmmu_val`）。
- `--batch_size`: 推理批大小。当前只支持 batch_size=1。
- `--output_dir`: 评测报告与详细指标的保存路径。