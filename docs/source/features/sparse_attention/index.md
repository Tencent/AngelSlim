# 稀疏注意力

稀疏注意力（Sparse Attention）是 AngelSlim 针对长上下文大模型推理开发的 Prefill 加速模块。其核心目标是在推理过程中动态跳过不重要的注意力块，从而显著降低 Prefill 阶段的计算量与延迟。

:::{toctree}
:caption: Contents
:maxdepth: 1

stem
:::
