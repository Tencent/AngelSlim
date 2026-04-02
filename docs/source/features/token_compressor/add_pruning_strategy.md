# 新增剪枝策略指南

AngelSlim 的核心优势在于将**模型架构包装**与**剪枝算法逻辑**完全解耦。研究人员只需调用底层封装好的工具函数，即可访问模型内部的激活值（如 Attention Map）并实现复杂的剪枝或合并逻辑。

接下来，我们以实现一个 **“Bottom-K Attention 策略”** 为例进行说明。

### 策略目标
从 Vision Encoder 的第 5 层提取注意力图，并保留其中注意力分数**最低**（即最不显著）的 K 个视觉 Token。

---

## 1. 核心开发流程

### 第一步：实现算法逻辑
在 `angelslim/compressor/token_compressor/algorithm` 目录下新建 Python 文件。我们需要利用 `PruningContext` 中捕获的 Q/K 张量来重构注意力分布。

```python
import torch
from ..base.context import PruningContext
from .utils.utils import (
    _extract_and_validate_vision_token_info,
    _recompute_attention_maps_for_all_images,
    _regroup_tensors_by_count
)

def bottom_k_attention_pruning(context: PruningContext, **kwargs) -> torch.Tensor:
    """
    自定义策略实现：选取 Vision Encoder 指定层注意力最低的 Token。
    """
    # 1. 提取超参数与元数据
    ratio = kwargs.get("ratio", 0.5)  # 剪枝比例
    target_layer = 5                  # 目标层索引
    input_ids = context.input_ids
    device = input_ids.device

    # 2. 定位全序列中的视觉 Token 分布
    # vision_indices_global: 所有视觉 Token 的绝对位置索引
    # num_tokens_per_image: 记录序列中每张图像占用的 Token 数量，用于处理 Packed Sequence
    vision_indices_global, non_vision_indices_global, _, num_tokens_per_image = \
        _extract_and_validate_vision_token_info(context)

    if len(vision_indices_global) == 0:
        return torch.ones_like(input_ids, dtype=torch.bool)

    # 3. 从上下文提取 Vision Encoder 第 5 层的 Q 和 K
    # 注意：必须在 YAML 的 requirements 中预先指定捕获该层，否则此处会报错
    q_tensor = context.vit_q[target_layer]
    k_tensor = context.vit_k[target_layer]
    
    if q_tensor is None or k_tensor is None:
        raise RuntimeError(f"Layer {target_layer} tensors not captured. Check requirements in YAML.")

    # 4. 重构注意力图
    # _recompute_attention_maps_for_all_images 会根据 Q/K 计算每张图的内部注意力分布
    # final_scores_list: 包含每张图像注意力权重的列表，元素形状为 [1, N_i]
    final_scores_list, _ = _recompute_attention_maps_for_all_images(q_tensor, k_tensor, context)
    
    # 将物理层面的输出对齐到逻辑层面的图像 Token（处理多图/视频帧拼接场景）
    final_scores_list, _ = _regroup_tensors_by_count(final_scores_list, num_tokens_per_image)

    # 5. 执行 Bottom-K 选择逻辑
    all_kept_indices_global = []
    vision_indices_split = torch.split(vision_indices_global, num_tokens_per_image)

    for scores, global_idx_map in zip(final_scores_list, vision_indices_split):
        n_tokens = scores.shape[1]
        num_to_keep = int(round(n_tokens * (1 - ratio)))
        num_to_keep = max(1, num_to_keep)

        # 核心：使用 torch.topk 选取权重最小的词（largest=False）
        _, bottom_k_local_indices = torch.topk(scores.squeeze(0), k=num_to_keep, largest=False)
        
        # 将局部索引映射回全序列的绝对索引
        all_kept_indices_global.append(global_idx_map[bottom_k_local_indices])

    # 6. 汇总索引并构造 [1, Seq_Len] 的布尔掩码
    kept_indices_tensor = torch.cat(all_kept_indices_global)
    final_indices = torch.cat([non_vision_indices_global.to(device), kept_indices_tensor])

    keep_mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    keep_mask[final_indices] = True
    
    return keep_mask.unsqueeze(0)
```

### 第二步：在工厂类中注册
修改 `angelslim/compressor/token_compressor/factory.py`，将新算法加入注册表：

```python
from .algorithm.my_custom_file import bottom_k_attention_pruning

PRUNING_STRATEGIES: Dict[str, Callable] = {
    "idpruner": idpruner,
    "bottom_k_attn": bottom_k_attention_pruning, # 注册名，用于 YAML 配置
    ...
}
```

---

## 2. 核心通用工具函数详解

在开发过程中，推荐使用以下内置函数以确保算法的跨模型兼容性：

### `_extract_and_validate_vision_token_info`
*   **输入**：`context` (PruningContext)。
*   **输出及形状**：
    1.  `vision_indices`: `[num_vision_tokens]`，视觉 Token 的绝对索引。
    2.  `non_vision_indices`: `[num_non_vision_tokens]`，非视觉 Token 的索引。
    3.  `vision_mask`: `[seq_len]`，全序列布尔掩码。
    4.  `num_tokens_per_image`: `List[int]`，Packed Sequence 中每张图像对应的 Token 数。

### `_recompute_attention_maps_for_all_images`
*   **功能**：屏蔽掉 Packed Sequence 中不相关的 Token（如其他图像或文本），重新计算单张图像内部的自注意力。
*   **输入**：`q_tensor`, `k_tensor` (形状 `[B, H, N, D]`) 及 `context`。
*   **输出**：
    1.  `final_scores_list`: `List[Tensor]`，每个元素为 `[1, N_i]`，即该图像的重要性得分。
    2.  `final_keys_list`: `List[Tensor]`，每个元素为 `[1, N_i, head_dim]`，即该图像的 Key 特征向量。

---

## 3. 运行与验证

在准备 YAML 配置文件时，**必须**在 `requirements` 中声明捕获 Vision Encoder 的第 5 层，否则 Context 中将无法获取对应的张量：

```yaml
# configs/qwen2_5_vl/pruning/bottom_k_test.yaml
compressor:
  requirements:
    vit_q_layers: [5]  # 捕获第 5 层 Query
    vit_k_layers: [5]  # 捕获第 5 层 Key
  strategies:
    global:
      strategy: "bottom_k_attn" # 使用注册的算法名
      params:
        ratio: 0.75
```

**运行单次推理验证：**
```bash
python tools/test_token_pruning.py \
    --config configs/qwen2_5_vl/pruning/bottom_k_test.yaml \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct"
```