# 稀疏注意力

稀疏注意力（Sparse Attention）是 AngelSlim 针对长上下文大模型推理开发的 Prefill 加速模块。其核心目标是在推理过程中动态跳过不重要的注意力块，从而显著降低 Prefill 阶段的计算量与延迟。

## 算法状态

当前仓库内置以下 Prefill 稀疏算法，均通过统一的 YAML + `tools/run.py` / `tools/infer.py` 入口调用（`compression.name: Sparsity`）。**目前只有 Stem 提供了完整的用户文档**；其余算法作为实验性能力提供，参数说明可参考 `configs/qwen3/sparse/` 下对应的 YAML 及各算法目录内的 NOTICE / 源码 docstring：

| 算法 | 状态 | 说明 |
| --- | --- | --- |
| Stem | 稳定 | 见下方 `stem` 页。`torch` 后端需要 Triton；`hpc` 后端为内部扩展，非公开可安装。 |
| MInference（a_shape / tri_shape / minference） | 实验性 | 内置 MIT 来源 kernel；`minference` 变体需运行时 CUDA 工具链（JIT 编译索引扩展）。 |
| FlexPrefill | 实验性 | 内置 Triton kernel（MIT 来源）。 |
| XAttention | 实验性 | 内置 kernel（MIT 来源）；head_dim 256 走 torch 参考。 |
| FlashPrefill | 实验性 | clean-room 实现；`alpha<=0` dense 路径需要 `flash_attn`。 |
| VecAttention | 仅参考实现 | 快速 kernel 依赖尚未公开发布的 vLLM-flash-attention fork；外部用户默认走 torch 参考路径（`allow_pseudo_sparse: true`）。 |
| CoSA | 实验性 | proxy 为内置 Triton；attention 依赖内部 `hpc` 扩展（非公开可安装）。 |

通用约束（所有算法）：不支持 padding mask（请用 batch=1 或无 padding 输入）、部分 kernel 要求 batch=1、prefix-cache / chunked prefill 会自动回退到 dense（稀疏仅在真正首填 `k_len==q_len` 时生效）、不支持 sliding-window 层、不支持多卡 / TP（`WORLD_SIZE>1` 会被拒绝）、稀疏 prefill 不支持 `output_attentions=True`、**不能与量化组合**。`allow_pseudo_sparse` 是正确性 / 调试回退路径，不是性能路径。

:::{toctree}
:caption: Contents
:maxdepth: 1

stem
:::
