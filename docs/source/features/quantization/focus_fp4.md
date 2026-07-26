# FOCUS FP4 量化

FOCUS 通过优化量化 scale，降低 FP4 量化带来的精度损失。当前支持
Qwen3-4B 的 MXFP4 W4A4 和 NVFP4 W4A4，并提供 DeepSpeed ZeRO-3
训练配置。

## 环境准备

请先按照 [AngelSlim 安装文档](../../getting_started/installation.md) 完成安装，
并确保环境中已安装 `deepspeed`。

默认配置会从 Hugging Face 加载 `Qwen/Qwen3-4B` 和
`Salesforce/wikitext`。运行环境需要能够访问 Hugging Face，或者已经在本地
缓存模型和数据集；也可以通过 `MODEL_PATH` 指定本地模型目录。

从仓库根目录运行以下命令。示例默认使用 2 张 GPU，`NPROC` 应与
`CUDA_VISIBLE_DEVICES` 中的 GPU 数量一致。

## 运行 FOCUS

### MXFP4

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=mxfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh
```

### NVFP4

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=nvfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh
```

脚本默认生成 fake checkpoint。可通过以下环境变量调整运行参数：

- `MODEL_PATH`：基础模型路径，默认使用 `Qwen/Qwen3-4B`。
- `SAVE_PATH`：输出路径；未设置时使用配置文件中的 `global.save_path`。
- `NPROC`：训练进程数，默认值为 `2`。
- `FORMAT`：量化格式，可选 `mxfp4` 或 `nvfp4`。

### 调整 GPU 数量

正式配置使用 global batch size 32：

```text
global batch size = per-device batch size × gradient accumulation steps × NPROC
```

保持配置中的 per-device batch size 不变时，建议按以下方式设置：

- MXFP4（per-device batch size 1）：2/4/8 张 GPU 分别使用
  gradient accumulation steps 16/8/4。
- NVFP4（per-device batch size 2）：2/4/8 张 GPU 分别使用
  gradient accumulation steps 8/4/2。

只修改 `NPROC` 会改变 global batch size。使用非 2-GPU 配置时，请复制对应
YAML、修改 `compression.QAT.hf_args.gradient_accumulation_steps`，再通过
`CONFIG=/path/to/config.yaml` 启动。

## 输出格式

默认的 `SAVE_FORMAT=fake` 会保存包含量化后权重和已优化 scale 的 fake
checkpoint。

如需直接生成 packed compressed-tensors checkpoint，可设置
`SAVE_FORMAT=real`：

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=mxfp4 \
SAVE_FORMAT=real \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
SAVE_PATH=./output/focus-mxfp4-real \
bash scripts/fp4/qwen-3-4b/run_focus_fp4.sh
```

将 `FORMAT` 改为 `nvfp4` 即可导出 NVFP4。`SAVE_PATH` 和
`global.save_path` 表示输出根目录，运行时还会追加配置文件名（不含
`.yaml`）：

- fake checkpoint：
  `<output_root>/<config_stem>_fake_quant_model.pt`
- real checkpoint：
  `<output_root>/<config_stem>/final_quant_checkpoint/`

例如，上面的 real 命令会保存到
`./output/focus-mxfp4-real/qwen3-4b_focus_mxfp4_w4a4_real_zero3/final_quant_checkpoint/`。

## 验证 checkpoint

验证 MXFP4 fake checkpoint 中的权重和 subgroup scale：

```shell
python tools/validate_focus_fp4_checkpoint.py \
  --checkpoint /path/to/mxfp4_fake_quant_model.pt \
  --model-path /path/to/Qwen3-4B \
  --qtype mxfp4 \
  --group-size 32 \
  --num-sub 4
```

验证 NVFP4 时使用 `--qtype nvfp4 --group-size 16 --num-sub 2`。

direct-real checkpoint 没有独立的 fake checkpoint，可执行结构校验：

```shell
python tools/validate_focus_mxfp4_export.py \
  --export-path /path/to/final_quant_checkpoint \
  --model-path /path/to/Qwen3-4B
```

NVFP4 使用 `tools/validate_focus_nvfp4_export.py`。对于由 fake checkpoint
离线导出的 real checkpoint，可额外传入 `--checkpoint` 执行 bit-exact
重打包校验。

## 评测

ZeRO-3 多进程命令仅用于训练和保存。请勿向上述 `torchrun` 启动流程追加
`--ppl-eval` 或 `--lm-eval`；保存完成后，请在独立的单 GPU 进程中加载
checkpoint 并运行 PPL 或 lm-evaluation-harness 评测。

## 配置文件

仓库提供以下 Qwen3-4B 配置：

- `configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_zero3.yaml`
- `configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_zero3.yaml`
- `configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_real_zero3.yaml`
- `configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_real_zero3.yaml`

如需调整训练数据、batch size、训练轮数或输出路径，请复制对应配置后修改，
并通过 `CONFIG=/path/to/config.yaml` 传给启动脚本。
