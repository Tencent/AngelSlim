# FOCUS FP4 量化

FOCUS 通过优化量化 scale，降低 FP4 量化带来的精度损失。当前支持
Qwen3-4B 的 MXFP4 W4A4 和 NVFP4 W4A4，并提供 DeepSpeed ZeRO-3
训练配置。

## 环境准备

请先按照 [AngelSlim 安装文档](../../getting_started/installation.md) 完成安装，
并确保环境中已安装 `deepspeed`。

从仓库根目录运行以下命令。示例默认使用 2 张 GPU，`NPROC` 应与
`CUDA_VISIBLE_DEVICES` 中的 GPU 数量一致。

## 运行 FOCUS

### MXFP4

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=mxfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4_qat.sh
```

### NVFP4

```shell
CUDA_VISIBLE_DEVICES=0,1 \
FORMAT=nvfp4 \
NPROC=2 \
MODEL_PATH=/path/to/Qwen3-4B \
bash scripts/fp4/qwen-3-4b/run_focus_fp4_qat.sh
```

脚本默认生成 fake checkpoint。可通过以下环境变量调整运行参数：

- `MODEL_PATH`：基础模型路径，默认使用 `Qwen/Qwen3-4B`。
- `SAVE_PATH`：输出路径；未设置时使用配置文件中的 `global.save_path`。
- `NPROC`：训练进程数，默认值为 `2`。
- `FORMAT`：量化格式，可选 `mxfp4` 或 `nvfp4`。

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
bash scripts/fp4/qwen-3-4b/run_focus_fp4_qat.sh
```

将 `FORMAT` 改为 `nvfp4` 即可导出 NVFP4。real checkpoint 保存在输出目录的
`final_quant_checkpoint/` 中。

## 配置文件

仓库提供以下 Qwen3-4B 配置：

- `configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_zero3.yaml`
- `configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_zero3.yaml`
- `configs/qwen3/fp4/qwen3-4b_focus_mxfp4_w4a4_real_zero3.yaml`
- `configs/qwen3/fp4/qwen3-4b_focus_nvfp4_w4a4_real_zero3.yaml`

如需调整训练数据、batch size、训练轮数或输出路径，请复制对应配置后修改，
并通过 `CONFIG=/path/to/config.yaml` 传给启动脚本。
