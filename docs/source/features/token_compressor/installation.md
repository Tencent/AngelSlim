# 安装指南

在使用 AngelSlim 的 Token 压缩与评测功能前，请确保您已完成 AngelSlim 的基础安装。本章节将指导您完成针对多模态模型剪枝与评测所需的额外环境配置。

## 1. 安装多模态依赖库

请使用以下命令安装多模态相关的辅助依赖：

```bash
uv pip install -r requirements/requirements_multimodal.txt
```

## 2. 配置集成评测环境 (lmms-eval)

AngelSlim 深度集成了 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 以支持一键式精度评测。评测脚本 `run_pruning_eval.py` 依赖于此套件。

请按照以下指令克隆仓库并以编辑模式安装：

```bash
# 返回您的工作目录
cd ~

# 克隆并安装 lmms-eval
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv pip install -e ".[all]"
```

:::{note}
**注意**：**不能**直接采取 `pip install lmms_eval` 进行安装。
:::
