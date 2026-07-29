# Stem: Rethinking Causal Information Flow in Sparse Attention

**Stem** 是 AngelSlim 的稀疏注意力算法，用于加速长上下文 LLM 的 **Prefill** 阶段。它通过在 block 粒度上估计注意力重要性，动态选择 top-k 关键块执行 block-sparse attention，在保持生成质量的同时大幅降低 Prefill 延迟。

## 1. 算法动机

长上下文推理（如 32K–128K tokens）中，Prefill 阶段的全量 attention 计算是主要瓶颈：

- 计算量随序列长度 **二次增长**，显存和延迟双重压力
- 实际上大部分 attention block 对最终输出贡献极小，存在大量冗余

Stem 的核心思路是：**先用低成本的 block-level scoring 估计每个 attention block 的重要性，再只对重要的 block 执行精确 attention**。

## 2. 技术原理

Stem 的 Prefill 过程分为三步：

### 2.1 Block-Level Scoring

使用 **Triton 加速的 strided group GEMM** 计算下采样的 Q·K^T 分数矩阵，并结合 value-norm bonus 项，得到每个 query-block 对每个 key-block 的重要性估计：

$$\text{score}(Q_i, K_j) = \frac{Q_i \cdot K_j^T}{\sqrt{d} \cdot s \cdot n} + \lambda \cdot \text{ReLU}(\bar{v}_j)$$

其中 $s$ 为 stride 因子，$n$ 为归一化系数，$\bar{v}_j$ 为 value-norm 的标准化对数值。

### 2.2 Top-k Schedule

每层的 per-block top-k budget 由两个独立的旋钮共同决定：

- **`layer_keep_ratios`（分层保留比例）**：给出每层的基础 keep-ratio。默认调度为
  warmup 前 2 层保留 100%（`1.0`），其余层保留稳态比例（`0.2`）；生产环境建议按模型层数显式传入完整列表。
- **`stem_alpha`（budget 衰减因子）**：在**单层内部**沿 query-block 位置对 budget 施加线性衰减，
  默认 `1.0`（不衰减）；示例 YAML 中设为 `0.7` 表示后段 query-block 逐步收紧 budget。可传 list
  按层覆盖。
- 额外保证 **initial blocks**（sink tokens）和 **sliding window** blocks 始终被保留。

### 2.3 Block-Sparse Attention

根据 top-k mask 执行稀疏 attention：

- 如果安装了 `block-sparse-attn` 库，使用真正的 block-sparse kernel
- 否则默认硬失败；仅当显式设置 `allow_pseudo_sparse: true` 时，才把 **最终 attention**
  回退到 dense masked 实现（展开 top-k mask 后做 dense attention）。注意 block
  重要性 scoring 阶段始终使用 Triton kernel，因此该回退**仍需要 Triton/CUDA**，并非纯
  CPU 参考。
- **HPC 后端**支持 bf16 dense prefill 和 fp8 block-sparse prefill（varlen / paged 两种路径）

**Decode 阶段不受影响**，仍使用模型原始的 attention 实现（FlashAttention-2 / eager / SDPA）。

## 3. 支持范围

| 维度 | 支持情况 |
|------|---------|
| **后端** | `torch`（PyTorch + Triton scoring，需 CUDA/Triton）、`hpc`（内部 C++ 扩展，**非公开可安装**） |
| **HPC 精度** | bf16（dense prefill）、fp8（block-sparse prefill，varlen / paged） |
| **序列长度** | 无上限，建议 4K+ tokens 以体现加速效果 |

> **后端说明**：`torch` 是面向社区的默认后端，其 block 重要性 scoring 使用 Triton kernel（因此需要 CUDA + Triton，**不是** 纯 CPU 参考实现）。`hpc` 后端依赖一个未随本仓库发布、外部无法安装的 HPC C++ 扩展，仅供内部使用；外部用户请使用 `backend: torch`。

## 4. 性能评测

> 以下为内部初步评测结果，完整的可复现材料（原始 metric / 数据集 / 模型 commit / 环境 / kernel / 命令）将随后发布。

在长上下文与 Agent 类任务上，我们对 Stem 的精度保持能力做了初步评测。在 **FP8-W8A8 + Stem** 配置下，模型在 LongBench v2、CL-bench、CL-bench Life、SWE-bench Verified、Terminal-Bench 2.0、ClawEval 等 benchmark 上的得分与 BF16 基线基本持平，部分任务（如 ClawEval）略有提升，初步验证了 Stem 稀疏注意力在加速 Prefill 的同时对模型质量影响很小。

:::{image} /assets/stem/benchmark.png
:alt: Stem 在多个 benchmark 上的精度对比（BF16 vs FP8-W8A8+Stem）。
:::

## 5. 快速开始

确保已安装 AngelSlim（`pip install -e .[sparse]`），然后在项目根目录运行。Stem
与 PTQ/QAT 一致，统一走标准的 YAML + `tools/run.py` / `tools/infer.py` 入口。

### Stem（torch 后端）

编辑或复用 `configs/qwen3/sparse/stem/qwen3-8b_stem_torch.yaml`，把 `model.model_path`
指向你的权重，然后：

```bash
# 压缩并保存稀疏检查点
python tools/run.py -c configs/qwen3/sparse/stem/qwen3-8b_stem_torch.yaml

# 用稀疏配置直接推理
python tools/infer.py -c configs/qwen3/sparse/stem/qwen3-8b_stem_torch.yaml
```

### Dense 对照（无 Stem patch）

直接用任意非稀疏 YAML（`compression` 不含 `Sparsity`），或加载未打补丁的模型。

### 切换后端 / 精度

后端与精度通过 YAML 的 `compression.sparsity.attn_kwargs` 设置（见下表），不再用命令行
开关：

```yaml
compression:
  name: Sparsity
  sparsity:
    name: stem
    allow_pseudo_sparse: false      # kernel 缺失则硬失败；置 true 回退 dense masked attention（scoring 仍需 Triton）
    attn_kwargs:
      backend: hpc                  # torch | hpc
      hpc_dtype: bf16               # bf16 | fp8（仅 hpc 后端）
      block_size: 128
```

## 6. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `backend` | `"torch"` | 后端选择：`"torch"` 或 `"hpc"` |
| `hpc_dtype` | `"bf16"` | HPC 后端精度：`"bf16"` 或 `"fp8"` |
| `hpc_fp8_path` | `"varlen"` | FP8 执行路径（仅 hpc 后端）：`"varlen"` 或 `"paged"` |
| `stem_alpha` | `1.0` | 单层内沿 query-block 的 budget 衰减因子，可传 list 分层控制 |
| `block_size` | `128` | attention block 大小 |
| `stride` | `8`（torch）/ `16`（hpc） | scoring 阶段的下采样步长（torch 默认 8；hpc 默认 16，可用 `hpc_stem_stride` 覆盖） |
| `chunk_size` | `2048` | scoring 阶段的分块宽度 |
| `norm` | `1.0` | scoring 阶段的额外归一化系数 |
| `initial_blocks` | `4` | 始终保留的头部 block 数量（sink tokens） |
| `window_size` | `4` | sliding window 保留的尾部 block 数量 |

## 7. 代码结构

```
angelslim/compressor/sparsity/
├── __init__.py                              # 懒加载入口（仅暴露 Sparsity）
├── registry.py                              # SparsityAlgorithmRegistry（统一注册表）
└── algorithms/
    └── stem/
        ├── __init__.py                      # 注册副作用
        ├── algorithm.py                     # Stem(SparsityAlgorithm) 注册类（主入口）
        ├── backends/
        │   ├── dispatcher.py                # torch / hpc 路由
        │   ├── torch_impl.py               # PyTorch + Triton 实现
        │   └── hpc_impl.py                 # HPC C++ 扩展实现
        ├── modules/
        │   └── forward.py                   # patched attention forward
        └── ops/
            └── stem_kernel.py               # Triton kernel

# 入口（统一走标准 YAML 路径）：
tools/run.py    -c configs/qwen3/sparse/stem/*.yaml   # 压缩/保存
tools/infer.py  -c configs/qwen3/sparse/stem/*.yaml   # 推理
```

## 8. Python API

Stem 是一等公民的注册算法，通过统一的 `SparsityAlgorithmRegistry` 构造，或直接走上面
的 YAML 路径：

```python
from angelslim.compressor.sparsity.registry import SparsityAlgorithmRegistry
import angelslim.compressor.sparsity.algorithms  # 触发注册

algo = SparsityAlgorithmRegistry.create("stem", attn_kwargs={
    "backend": "hpc",
    "hpc_dtype": "fp8",
    "stem_alpha": [1.0] * 5 + [0.7] * 31,  # 36 层 Qwen3-8B
})
algo.setup(model)                 # 按模型层数派生 per-instance 配置
# 框架的 patcher 会对每个注意力模块调用 algo.build_attn_forward(...)
```
