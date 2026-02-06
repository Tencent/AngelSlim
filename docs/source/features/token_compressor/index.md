# Token 压缩

Token 压缩（Token Compressor）是 AngelSlim 针对多模态大模型（MLLM）开发的高效压缩模块。其核心目标是在推理过程中动态减少视觉 Token 数量，从而显著降低 KV Cache 的显存占用，并大幅加速 Prefill 与 Decoding 阶段。

本工具包提供了一套基于元数据驱动的通用适配架构，支持一键式部署多种剪枝（Pruning）与合并（Merging）算法。本工具的核心特点在于实现了压缩算法逻辑与模型底层代码的高度解耦：研究人员无需深入关注特定模型复杂的内部实现，只需专注于开发高效的剪枝或合并策略，即可实现跨模型的快速迁移、部署与算法验证。

此外，我们还在框架中深度集成了 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 评测套件。这意味着，对于任何集成进来的压缩算法，研究人员都可以利用 lmms-eval 的能力，一键启动多维度精度评测，快速获取模型在各种主流基准任务上的性能指标。

:::{toctree}
:caption: 核心算法与方法
:maxdepth: 1

idpruner
other_methods
:::