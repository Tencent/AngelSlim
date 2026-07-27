# GLM-5.2 W8A8C8 PTQ 量化完整方案

将 **GLM-5.2 (`GlmMoeDsaForCausalLM`，约 1.5 TB bf16 权重的检查点，78 层，256 个路由专家)**  量化为 **W8A8 + INT8 动态 KV Cache** 的完整端到端方案，运行在 **2 节点 × 8 × H20 96GB** 环境下。

---

## 目录

1. [方案概览](#1-方案概览)
2. [硬件与环境](#2-硬件与环境)
3. [核心策略](#3-核心策略)
4. [MoE Defuse（专家拆解）详解](#4-moe-defuse专家拆解详解)
5. [修复的 Bug](#5-修复的-bug)
6. [修改的文件](#6-修改的文件)
7. [运行方法](#7-运行方法)
8. [输出布局](#8-输出布局)
9. [正确性验证](#9-正确性验证)
10. [部署](#10-部署)
11. [故障排查速查表](#11-故障排查速查表)
12. [关键数值常量](#12-关键数值常量)
13. [YAML `ignore_layers` 快速定制不同 recipe](#13-yaml-ignore_layers-快速定制不同-recipe2026-07-25-追加)
14. [MTP 层（layer 78）Backfill 完整方案](#14-mtp-层layer-78backfill-完整方案2026-07-27-追加)

---

## 1. 方案概览

| 组件 | 量化方案 |
|------|---------|
| **权重（路由专家 / 注意力 / 稠密 MLP / 共享专家）** | INT8 per-channel（按通道量化） |
| **激活值** | INT8 per-token 动态量化（不保存静态 scale） |
| **KV Cache** | INT8 动态 — MLA NoPE 隐向量按 block-128 量化 + DSA indexer K 按 token 量化；RoPE 尾部保持 bf16 |
| **不参与量化的层** | `kv_b_proj` / `indexer.k_norm` / `indexer.weights_proj` / `mlp.gate` / `lm_head` / `embed_tokens` / 所有 MTP 内部辅助模块 |
| **C8 白名单** | `indexer.wq_b`、`indexer.wk`（INT8 量化） |
| **预处理** | SmoothQuant（`smooth_alpha=0.5`） |

**输出大小**：bf16 ~1.5 TB → INT8 ~697 GB（约 2.15 倍压缩，含 per-channel scales）。

---

## 2. 硬件与环境

- 2 节点 × 8 × H20 96GB GPU（共 16 张 GPU）
- CPU cgroup 限制约 2.2 TB / pod
- 两个节点通过共享 cephfs 访问相同的模型路径和保存路径
- Python 3.12 · torch（NCCL 后端）· transformers（含 `glm_moe_dsa` 支持）· safetensors · accelerate

---

## 3. 核心策略：**方案 B — 纯专家并行 + INT8 低内存流式加载**

我们**刻意不使用 ZeRO-3**，以避免 `deepspeed.zero.Init` 的 hook 劫持 `torch.empty` 并强制分配到 CUDA（当尝试分配融合的 12 GB `gate_up_proj` 时会导致 OOM）。替代方案如下：

```
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. 配置修复 (glm5._fix_hf_config)                            │
   │    绕过 HF attribute_map 的 bug，该 bug 会将                 │
   │    qk_rope_head_dim=64 覆盖为 head_dim=192                  │
   ├─────────────────────────────────────────────────────────────┤
   │ 2. Meta 骨架构建 (accelerate.init_empty_weights)             │
   │    零字节 —— 不分配 CPU 或 CUDA 内存                        │
   ├─────────────────────────────────────────────────────────────┤
   │ 3. 空 MoE 拆解 (_defuse_moe_experts_empty)                   │
   │    融合的 [256, 2*I, H] gate_up_proj -> 每个专家独立 Linear  │
   │    EP 分片：每个 rank 只物化本地的 1/W 个专家                │
   │    非本地专家 = 零参数占位符                                 │
   ├─────────────────────────────────────────────────────────────┤
   │ 4. 每个 rank 独立加载 (stream_load_weights)                  │
   │    不使用 NCCL：每个 rank 只读取自己 named_parameters 中     │
   │    出现的 tensor → 物化到 CPU                                │
   ├─────────────────────────────────────────────────────────────┤
   │ 5. INT8 PTQ low_memory_run                                   │
   │    模型保存在 CPU；每次将一层 layers[i].to("cuda") 到 GPU    │
   │    GPU 峰值约 15 GB / rank ≪ 96 GB                           │
   ├─────────────────────────────────────────────────────────────┤
   │ 6. 基于文件系统的 EP 保存 (Glm5EPQuantSaver)                 │
   │    每个 rank 直接写自己的分片文件；rank 0 合并部分索引 JSON  │
   │    不在 tensor 上使用 NCCL 集合通信                          │
   └─────────────────────────────────────────────────────────────┘
```

**每个 rank 的内存占用（world=16）**

| 组件 | 大小 |
|------|------|
| 路由专家（16 专家 × 75 MoE 层 × 75.5 MB） | 90.6 GB |
| 注意力 Linear（78 × ~200 MB） | 15.6 GB |
| 共享专家（75 × 75.5 MB） | 5.7 GB |
| 稠密 MLP（3 个稠密层） | 1.4 GB |
| Embed + LM head | ~4 GB |
| **每个 rank CPU 总计** | **~117 GB** |
| **每个节点 8 rank** | **~936 GB < 2.2 TB cgroup** ✅ |

**每个 rank GPU 峰值**：约 15 GB（一次只激活一个 transformer block） ≪ 96 GB ✅

---

## 4. MoE Defuse（专家拆解）详解

### 4.1 为什么需要 Defuse？

**这是整个方案中最核心的一步。** 要理解它，首先需要理解 GLM-5 原始的 MoE 专家存储方式。

#### 上游 `GlmMoeDsaNaiveMoe` 的结构

GLM-5 的官方 HuggingFace 实现中，路由专家以**融合的 3 维参数**形式存储：

```python
# 原始形式 —— 两个巨大的 3D nn.Parameter
gate_up_proj  shape=[num_experts, 2*intermediate, hidden]   # [256, 4096, 6144]
down_proj     shape=[num_experts, hidden, intermediate]      # [256, 6144, 2048]
```

每个专家（共 256 个）的 gate_proj 和 up_proj 权重被**拼接**在一个参数的第一维中（`gate_up_proj[i, :intermediate, :]` 是 gate，`gate_up_proj[i, intermediate:, :]` 是 up），forward 时通过 `F.linear(x, gate_up_proj[i]).chunk(2, dim=-1)` 拆分使用。

#### 问题：融合参数对量化管线不可见

AngelSlim 的 `find_layers` 只会发现 `nn.Linear` 叶节点。上述两个 `nn.Parameter` 不是 `nn.Linear`，因此：

- **激活值观测器无法挂载** → 无法收集激活值统计信息
- **QDQ 量化模块无法插入** → 这些专家的权重和激活值会**保持 bf16**
- **最终检查点大小约 1.4 TB**，失去了量化的意义

换句话说，**不拆解专家，量化就白做了**。

### 4.2 Defuse 的目标

将融合的 3D 参数拆分为每个专家独立的 `nn.Linear`：

```
Before (GlmMoeDsaNaiveMoe):
  gate_up_proj: [256, 4096, 6144]   ← 一个 nn.Parameter
  down_proj:    [256, 6144, 2048]   ← 一个 nn.Parameter

After (GlmMoeDsaSplitMoe):
  experts.0.gate_proj:  nn.Linear(6144, 2048)   ← 真正的 nn.Linear
  experts.0.up_proj:    nn.Linear(6144, 2048)
  experts.0.down_proj:  nn.Linear(2048, 6144)
  experts.1.gate_proj:  nn.Linear(6144, 2048)
  ...
  experts.255.down_proj: nn.Linear(2048, 6144)
```

这样每个专家都变成了标准的 `nn.Linear`，`find_layers` 能发现它们，量化管线能正常处理。

### 4.3 实现细节

#### 4.3.1 `_GlmSplitExpertMLP` — 单个专家

```python
class _GlmSplitExpertMLP(nn.Module):
    """一个专家 = 三个普通 nn.Linear。bias=False 与上游保持一致。"""
    def __init__(self, hidden_dim, intermediate_dim, act_fn, ...):
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # gate 投影
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # up 投影
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)  # down 投影
        self.act_fn    = act_fn  # SiLU

    def forward(self, x):  # 标准 SwiGLU
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

#### 4.3.2 `GlmMoeDsaSplitMoe` — 拆分后的 MoE 层

这是整个 defuse 的核心类，**继承自 `nn.ModuleList`**（而非普通 `nn.Module`）。

**为什么继承 `nn.ModuleList`？** 这是为了确保参数名命名正确。原始检查点的命名格式是：

```
mlp.experts.0.gate_proj.weight
mlp.experts.0.up_proj.weight
mlp.experts.0.down_proj.weight
...
```

如果用一个普通 `nn.Module` 包装内部的 `self.experts = ModuleList(...)`，参数名会变成：

```
mlp.experts.experts.0.gate_proj.weight   ← 多了一层 "experts"！
```

这会导致 `stream_load_weights` 加载时无法匹配检查点的 key，7565 个专家权重全部丢失。继承 `nn.ModuleList` 直接让 `GlmMoeDsaSplitMoe` 本身就充当 experts 容器，参数名直接为 `mlp.experts.<i>.<proj>.weight`。

#### 4.3.3 `_GlmZeroExpert` — EP 占位符

在专家并行（Expert Parallel, EP）模式下，每个 rank 只物化本地的 `1/world_size` 个专家。对于不属于自己的专家，用 `_GlmZeroExpert` 占位：

```python
class _GlmZeroExpert(nn.Module):
    """零参数占位符，保持 ModuleList 长度和索引连续"""
    def forward(self, x):
        return torch.zeros_like(x)  # 防御性代码，EP forward 实际上会跳过它
```

EP forward 通过 `experts_start_idx` 和 `experts_end_idx` 范围跳过非本地专家，所以 `_GlmZeroExpert.forward` 永远不会被调用。

#### 4.3.4 两种 Defuse 路径

| 方法 | 场景 | 特点 |
|------|------|------|
| `_defuse_moe_experts()` | 单节点、内存充足 | 从现有 `GlmMoeDsaNaiveMoe` 中搬迁权重（`from_naive`），权重在真实设备上 |
| `_defuse_moe_experts_empty()` | 多节点 EP、ZeRO-3 | 构建空骨架（`empty`），参数在 `meta` 设备上，权重稍后由 `stream_load_weights` 流式加载 |

**`from_naive` (有体重建)：**
```python
# 从 gate_up_proj[expert_i] 中拆分出 gate 和 up
gate_w = gate_up_proj[i, :intermediate_dim, :]   # 前一半 → gate_proj
up_w   = gate_up_proj[i, intermediate_dim:, :]   # 后一半 → up_proj
down_w = down_proj[i, :, :]                       # down_proj

# 赋值到独立 Linear（.copy_ 不触发额外内存分配）
expert.gate_proj.weight.copy_(gate_w)
expert.up_proj.weight.copy_(up_w)
expert.down_proj.weight.copy_(down_w)

# 立即释放融合参数，回收内存
naive_moe.gate_up_proj = None
naive_moe.down_proj = None
```

**`empty` (空骨架构建，用于 EP 路径)：**
```python
# 在 meta 设备上构建，零内存占用
with torch.device("meta"):
    split_moe = GlmMoeDsaSplitMoe.empty(
        num_experts=256,
        ep_rank=rank,        # 当前 rank
        ep_world_size=16,    # 总 rank 数
    )

# 只在本地专家 slice 上构建真正的 _GlmSplitExpertMLP
# 其他专家都是 _GlmZeroExpert 占位符（零参数）
```

#### 4.3.5 关键 Bug：`act_fn` 模块注册污染

**这是开发过程中踩的最隐蔽的坑之一。**

`GlmMoeDsaSplitMoe` 需要一个 `act_fn`（SiLU），它是 `nn.Module`。如果直接 `self.act_fn = nn.SiLU()`，`nn.Module.__setattr__` 会自动将其注册到 `_modules['act_fn']`。

而 `nn.ModuleList` 使用 `str(len(self))` 作为新元素的 key，所以 `_modules` 会变成：
```python
_modules = {
    'act_fn': SiLU(),  # 占用了 key '0' 的位置！
    '1': expert_1,
    '2': expert_2,
    ...
    '256': expert_256,
}
```

结果是：**所有专家索引偏移了 1**。`self[0]` 会抛出 `KeyError: '0'`，而 `self[128]` 会访问到错误的专家。

**修复方法**：使用 `object.__setattr__(self, "act_fn", act_fn)` 绕过 `nn.Module.__setattr__`，将 `act_fn` 存储在普通 `__dict__` 中，不污染 `_modules`。

### 4.4 专家并行（EP）分片机制

EP 分片是让整个方案能在 H20 96GB 上跑通的关键。核心思想：

```
256 个专家 / 16 个 rank = 16 个专家/rank

rank 0:  专家 [0,   16)
rank 1:  专家 [16,  32)
...
rank 15: 专家 [240, 256)
```

每个 rank 只物化自己负责的 16 个专家参数（~90 GB），其他 15 × 16 = 240 个专家的 slot 用 `_GlmZeroExpert` 填充。

**Forward 时的 EP 协同**：
1. 每个 rank 都收到相同的输入 `hidden_states`
2. 路由（router）在所有 rank 上独立计算，选出 top-8 专家
3. 每个 rank 只计算自己拥有的专家
4. 最后通过 `dist.all_reduce(final_hidden_states)` 汇总所有部分结果

这样每个 rank 处理的计算量也减少到了原来的 1/16。

---

## 5. 修复的 Bug

### 5.1 `zero.Init` 在 H20 上 OOM
- **症状**：`CUDA OOM: 12.00 GiB`，发生在专家构建阶段
- **根因**：DeepSpeed 的 `zero.Init` 包装了 `torch.empty`，在分片之前强制分配到 CUDA
- **修复**：在 `base_model.from_pretrained` 中引入了纯 EP 路径，使用 `accelerate.init_empty_weights`（meta 设备）— DeepSpeed hook 保持休眠状态

### 5.2 对 CPU tensor 进行 NCCL broadcast
- **症状**：`No backend type associated with device type cpu`
- **根因**：NCCL 只能处理 CUDA tensor
- **修复**：每次 broadcast 使用 CUDA 暂存缓冲区（先放到 `cuda:local_rank`，broadcast，再拷贝回 CPU 目标，最后释放）

### 5.3 权重加载时的集合通信不对称死锁
- **症状**：加载到约第 4 个分片（第一个 MoE 分片）后卡住
- **根因**：rank 0 的 `name_to_param` 不包含其他 rank 拥有的专家 → rank 0 跳过了 key，而其他 rank 进入了集合通信 → 死锁
- **修复**：检测到 `any_meta` 模型时，切换到**每个 rank 独立加载**。每个 rank 打开 `safe_open`，只加载自己 `named_parameters` 中存在的 tensor —— 完全不使用 NCCL

### 5.4 HF `attribute_map` 损坏 `qk_rope_head_dim`
- **症状**：注意力 forward 中 `torch.split(query_states, [192,192]) vs actual 256`；还有 78 个 `kv_a_proj_with_mqa` 形状不匹配（704 vs 576）
- **根因**：`configuration_glm_moe_dsa.py` 在 `attribute_map` 中声明了 `"head_dim": "qk_rope_head_dim"` —— HF 静默地将 `qk_rope_head_dim=64` 覆盖为 `head_dim=192`
- **修复**：`_fix_hf_config()` 重新读取原始 `config.json`，在 `from_config` 之前强制设置正确的 `qk_rope_head_dim`、`qk_nope_head_dim`、`qk_head_dim`、`v_head_dim`、`kv_lora_rank`、`index_head_dim`、`q_lora_rank`

### 5.5 Meta tensor 泄漏到 `low_memory_run`
- **症状**：`NotImplementedError: Cannot copy out of meta tensor`，在 `layers[i].to("cuda")` 时发生
- **根因**：部分参数（形状不匹配的 `kv_a_proj`、绑定的权重）停留在 meta 设备
- **修复**：`stream_load_weights` 在最后对残留的 meta 参数进行零填充（功能上为空但非 meta，使 `.to(cuda)` 能正常工作）

### 5.6 ⭐ `mlp.experts.experts.<i>` 双重前缀
- **症状**：加载时缺失 7565 个专家权重
- **根因**：`GlmMoeDsaSplitMoe(nn.Module)` 内部有 `self.experts = ModuleList(...)`；父模块已将其挂载为 `mlp.experts`，key 变成了 `mlp.experts.experts.<i>.weight`
- **修复**：让 `GlmMoeDsaSplitMoe` **直接继承 `nn.ModuleList`**，使 key 折叠为 `mlp.experts.<i>.weight`（与检查点匹配）

### 5.7 ⭐ `act_fn` 模块注册导致专家索引偏移
- **症状**：`KeyError: '0'` 在 `nn.ModuleList.__getitem__` 中，加上 `expert_idx=128` 落到了 `_GlmZeroExpert`
- **根因**：`self.act_fn = nn.SiLU()` 被 `nn.Module.__setattr__` 自动注册到 `_modules['act_fn']`。`nn.ModuleList.append` 使用 `str(len(self))` 作为 key，所以 `_modules` 变成 `{'act_fn', '1', '2', ..., '256'}`，**每个专家索引偏移 1**，`self[0]` 报 KeyError
- **修复**：`object.__setattr__(self, "act_fn", act_fn)` —— 绕过 `Module.__setattr__`，使 `act_fn` 仅存在于 `__dict__` 中，不污染 `_modules`

### 5.8 跨节点 EP 未激活 → CPU OOM
- **症状**：`oom-kill`，anon-rss 344 GB；单个 rank 持有 32 个专家（用 16 rank 时应该只有 16 个）
- **根因**：`tools/run.py` 重新启动 torchrun 时只传了 `--nproc_per_node=8`，没有转发 `--nnodes / --node_rank / --master_addr` —— 每个节点运行了独立的 8-rank 进程组（world=8）
- **修复**：`_auto_torchrun_for_expert_parallel` 现在从环境变量中读取 `NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT`（已由启动脚本设置），并转发给 torchrun → 真正的 `world_size=16`

### 5.9 保存时的 NCCL `ibv_reg_mr_iova2` 失败
- **症状**：`Call to ibv_reg_mr_iova2 failed with error Invalid argument`，在 `dist.all_gather_object(gathered, local_experts)` 保存时发生
- **根因**：`all_gather_object` 序列化整个每层本地专家字典（约 576 MB），并尝试将其注册为 RDMA 内存区域；缓冲区太大或 IB MR 配额耗尽
- **修复**：**基于文件系统的 EP 保存** —— 每个 rank 直接将自己的 `model-r{RR}-{SSSSS}.safetensors` 分片写入共享 cephfs + 转储部分 `weight_map` JSON。rank 0 在 barrier 后合并这些部分。只有 `dist.barrier()` 使用集合通信 —— 没有大 tensor 经过 NCCL

---

## 6. 修改的文件

### 6.1 `angelslim/models/llm/glm5.py`
- 新增 `_GlmSplitExpertMLP` / `_GlmZeroExpert`（每个专家独立 Linear + EP 占位符）
- 新增 `GlmMoeDsaSplitMoe(nn.ModuleList)` — `GlmMoeDsaNaiveMoe` 的替代品，每个专家独立 Linear，EP 感知的 `_init_ep_attrs / _make_expert / forward`
- `__init__` / `empty` / `from_naive`：先执行 `nn.ModuleList.__init__(self, [])`，然后用 `object.__setattr__(self, "act_fn", act_fn)` 避免模块注册污染
- `forward` 包含 EP 跳过逻辑（`experts_start_idx <= idx < experts_end_idx`），末尾 `dist.all_reduce`，防御性的 `isinstance(expert, _GlmSplitExpertMLP)` 检查
- 新增 `_defuse_moe_experts_empty`（EP 感知，基于 world_size 的分片）
- 新增 `_fix_hf_config` — 修复 HF `attribute_map` 对 head-dim 字段的损坏
- 新增 `_promote_ep_state` — 将 EP 状态从任意 `GlmMoeDsaSplitMoe` 提升到适配器上
- 新增 `apply_kvcache_observers` — 为 C8 方案挂载 `kv_a_proj_with_mqa`（MLA 隐向量 per-block-128）和 `indexer.wk`（per-token 动态）的 hook
- `get_save_func` 在 EP 启用时返回 `Glm5EPQuantSaver`

### 6.2 `angelslim/models/base_model.py`
- `from_pretrained` 中的新 EP 路径（当 `using_multi_nodes=True` 且适配器有 `_defuse_moe_experts_empty` 时）：
  ```
  AutoConfig.from_pretrained → self._fix_hf_config → init_empty_weights →
  AutoModelForCausalLM.from_config → _defuse_moe_experts_empty → stream_load_weights
  ```

### 6.3 `angelslim/utils/zero3_io.py`
- `zero3_empty_model_from_pretrained`：在 `torch.device("meta")` 上构建骨架（零拷贝）
- `_broadcast_into_target` meta 分支：`decision` int64 tensor broadcast（形状同步 + 跳过/加载决策）→ CUDA 暂存 → CPU 物化。跳过时零填充以避免残留 meta
- `stream_load_weights` 检测 `any_meta` 模型 → **每个 rank 独立读取**（无 NCCL，无死锁）。保留旧的 rank0-broadcast 路径用于 ZeRO-3
- 末尾的 Meta 安全网：扫描残留的 meta 参数/缓冲区，在 CPU 上零填充

### 6.4 `angelslim/compressor/quant/core/save.py`
- 新类 `Glm5EPQuantSaver(PTQSaveVllmHF)`：
  - 在 `self.quant_model` 上检测 `expert_parallel_enabled`；否则回退到 `super().save()`
  - 生成 INT8 vLLM `quantization_config`（activation_scheme=dynamic，kv_cache_scheme=static（当 C8 时），transform_config）
  - 写入 `hf_quant_config.json`、tokenizer（仅 rank 0）
  - **基于文件系统的 EP 保存**：rank R 将本地专家写入 `model-r{RR}-{SSSSS}.safetensors`；rank 0 额外写入 head/norm/embed 和每层共享部分；每个 rank 转储 `_partial_weight_map_r{RR}.json`；rank 0 在 barrier 后合并
  - 如果存在 `kv_cache_scales.safetensors`，由 rank 0 写入

### 6.5 `tools/run.py`
- `_auto_torchrun_for_expert_parallel` 现在从环境变量中读取 `NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT`，并转发给 `torch.distributed.run` —— 启用真正的跨节点 EP（`world_size=16` 而非 8）

### 6.6 `configs/glm5/w8a8_int8/glm5_w8a8c8_2node.yaml`
- `zero3: false`（避免 DeepSpeed hook）
- `enable_expert_parallel: true`（跨节点 EP）
- `low_memory: true`（逐 block 流式 INT8 PTQ）
- 完整的 W8A8C8 配方（int8 per-channel 权重 + int8 per-token 动态激活 + int8 动态 KV cache）
- SmoothQuant 预处理（`smooth_alpha=0.5`）

---

## 7. 运行方法

### 前置条件
- 设置 `MODEL_PATH`（启动脚本中默认为 `/dockerdata/chatglm5.2`）
- 两个节点都能访问相同的 cephfs
- Node 0 的 IP 对 Node 1 可达（在两个启动脚本中设置 `MASTER_ADDR=<node0_ip>`）

### 启动（两个节点，**同时启动**）

**在 node 0 上：**
```bash
bash scripts/ptq/run_glm5_w8a8c8_2node_0.sh
```

**在 node 1 上：**
```bash
bash scripts/ptq/run_glm5_w8a8c8_2node_1.sh
```

两个脚本都导出 `NNODES=2`、`MASTER_ADDR=<node0_ip>`、`MASTER_PORT=29555`，并分别设置 `NODE_RANK=0/1`。`tools/run.py` 自动在 `torchrun` 下重新启动，带上正确的多节点参数。

### 预期时间线（实测）

| 阶段 | 耗时 |
|------|------|
| 模型空 defuse | ~5 秒 |
| 权重流式加载（每个 rank） | ~5-15 分钟 |
| 校准（78 层 × 64 样本，low_memory_run） | ~40-90 分钟 |
| 转换（插入 QDQ 模块） | ~1 分钟 |
| 保存（基于文件系统的 EP） | ~10-20 分钟 |
| **总计** | **~1-2 小时** |

---

## 8. 输出布局

```
output_glm5_w8a8c8_2node/glm5_w8a8c8_2node/
├── config.json                        # HF config + quantization_config
├── hf_quant_config.json               # trtllm 风格：quant_algo=INT8 + exclude_modules
├── model.safetensors.index.json       # weight_map：117,381 个 tensor → 分片文件
├── kv_cache_scales.safetensors        # （仅静态 scale 时存在；W8A8C8 动态无）
├── tokenizer.json / tokenizer_config.json / chat_template.jinja
├── model-00001.safetensors            # （旧路径 —— head 分片）
├── model-r00-00001.safetensors        # rank 0 分片（head + 自有专家 + 共享部分）
├── model-r00-00002.safetensors
├── ...
├── model-r15-00075.safetensors        # rank 15 分片
└── angelslim_config.json              # 使用的完整 YAML 转储
```

- **共 1285 个文件**
- 16 个 EP rank → 每个写入约 75-80 个分片
- 磁盘上 697 GB
- vLLM / HuggingFace 推理通过 `model.safetensors.index.json` 读取；特殊的分片文件名对使用者透明

---

## 9. 正确性验证

```bash
python3 -c "
import json
d = json.load(open('output_glm5_w8a8c8_2node/glm5_w8a8c8_2node/model.safetensors.index.json'))
wm = d['weight_map']
print('total keys:', len(wm))                    # 期望约 117,381
# 每层的 256 个专家 × 3 proj × 2（weight + weight_scale）都应该存在
for lid in range(3, 78):                          # 前 3 层是稠密 MLP
    for eid in range(256):
        for p in ('gate_proj','up_proj','down_proj'):
            for suf in ('.weight','.weight_scale'):
                k = f'model.layers.{lid}.mlp.experts.{eid}.{p}{suf}'
                assert k in wm, k
print('all expert keys present')
"
```

---

## 10. 部署

保存的检查点遵循 **vLLM compressed-tensors INT8** 布局：
- `weight`（int8）、`weight_scale`（bf16，per-channel）、`input_scale` 在运行时动态处理
- `quantization_config.quant_method = "compressed-tensors"`（或 `"vllm"`），`activation_scheme = "dynamic"`，`kv_cache_scheme = "static"`（带丰富的 C8 描述符）
- 已通过 `scripts/deploy/verify_glm5_w8a8.py` 验证

```bash
# vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/output_glm5_w8a8c8_2node/glm5_w8a8c8_2node \
    --quantization compressed-tensors \
    --trust-remote-code \
    -tp 8
```

---

## 11. 故障排查速查表

| 症状 | 可能原因 | 修复方法 |
|------|---------|---------|
| `CUDA OOM ... 12 GiB` 在专家构建时 | DeepSpeed `zero.Init` hook 激活 | 确保 yaml 中 `zero3: false` |
| 加载到约 4 个分片后卡住 | 集合通信不对称死锁 | 确认 `stream_load_weights` 选择了 meta 分支（日志：`loaded shard model-...`） |
| `KeyError: '0'` 在 `nn.ModuleList` 中 | `act_fn` 模块污染 | 确保 SplitMoe init 中使用 `object.__setattr__(self, "act_fn", ...)` |
| `split_with_sizes ... [192,192] vs 256` | HF `attribute_map` 损坏 | 确保 `_fix_hf_config` 在 `from_config` 之前运行 |
| 保存时 `ibv_reg_mr_iova2 EINVAL` | NCCL RDMA MR 太大 | 确认使用了基于文件系统的保存器（`Glm5EPQuantSaver.save`）；不对大字典使用 `all_gather_object` |
| `oom-kill`（rank RSS > 200 GB） | world_size 只有 8 而非 16 | 确认 run.py 转发了 `--nnodes/--node_rank`（日志：`total world_size=16`） |
| 加载报告 `SHAPE MISMATCH -- zero-filling` | 模型/检查点版本不一致 | 检查 `_fix_hf_config` 是否覆盖了所有不匹配的字段 |

---

## 12. 关键数值常量（来自发布的 `config.json`）

- `hidden_size = 6144`、`intermediate_size = 12288`、`moe_intermediate_size = 2048`
- `num_hidden_layers = 78`、`first_k_dense_replace = 3`（所以有 75 个 MoE 层）
- `n_routed_experts = 256`、`num_experts_per_tok = 8`
- `qk_nope_head_dim = 192`、`qk_rope_head_dim = 64`、`qk_head_dim = 256`
- `kv_lora_rank = 512`、`q_lora_rank = 1536`
- `index_head_dim = 128`、`index_n_heads = 32`、`index_topk = 2048`
- `head_dim = 192` ← 这个 key 触发了 `attribute_map` 的 bug；`_fix_hf_config` 覆盖了由此造成的附带损害

---

**签字确认 — 2026-07-25**。量化检查点位于 `output_glm5_w8a8c8_2node/glm5_w8a8c8_2node/`，约 697 GB，117,381 个 tensor，可立即用于 vLLM 部署。


---

## 13. YAML `ignore_layers` 快速定制不同 recipe（2026-07-25 追加）

 W8A8C8 recipe 是**默认**行为——、 都参与 INT8 量化，MoE 路由 /  /  / MTP    />**强制跳过**。任何进一步的定制**都不再需要改代码**，只需在 YAML 里用 `compression.quantization.ignore_layers` 列出要跳过的**子串**即

### 13.1 语义

- 子串匹配（`ignore_pattern in fully_qualified_name`），与 AngelSlim `gptq.py` / `daq.py` / `save.py` 下游一致。
- 与适配器 hard-skip 列表（`_FORCED_SKIP_SUBSTRINGS`）**并集**：hard-skip 永远生效、且 YAML 追加项也生效。
- 子串必须能**精准命中**目标层的 FQN 才生效——推荐写全一点，例如 `.indexer.wq_b` 而不是 `wq_b`。

#
 [angelslim/models/llm/glm5.py](../angelslim/models/llm/glm5.py) `GLM5.get_observer_layers`；层白名单 / 硬跳过常量在 [angelslim/models/llm/_glm5_skip_lists.py](../angelslim/models/llm/_glm5_skip_lists.py)（约 60 行、torch-free、可被 [tests/test_glm5_quant_layer_selection.py](../tests/test_glm5_quant_layer_selection.py) 独立引用）。

### 13.2 三份典型 recipe 对照

| 层 | 默认（无 `ignore_layers`） | W8A8C8（historical） | kunlun 等价 |
|---|---|---|---|
| `kv_b_proj` | ✅ INT8 | ❌ bf16（YAML 列出 `"kv_b_proj"`） | ✅ INT8 |
| `indexer.wq_b` / `wk` | ✅ INT8 | ✅  bf16（YAML 列出 `".indexer.wq_b"`, `".indexer.wk"`）|INT8 | 
| `indexer.weights_proj` | ❌ bf16（leaf 不在白名单） |  | 同左 |
bash /apdcephfs_sgfd2/share_300532381/harviexu/env_install.sh  | 同左 | 同左 |

**W8A8C8 recipe**（[configs/glm5/w8a8_int8/glm5_w8a8c8_2node.yaml](../configs/glm5/w8a8_int8/glm5_w8a8c8_2node.yaml)）：

```yaml
compression:
  quantization:
    ignore_layers:
      - "lm_head"
      - "embed_tokens"
      - "kv_b_proj"                 # MLA latent-to-heads (accuracy-sensitive)
      - ".indexer.weights_proj"     # DSA per-head gating (tiny; top-k stability)
```

**kunlun 等价 recipe**（[configs/glm5/w8a8_int8/glm5_w8a8_kunlun_2node.yaml](../configs/glm5/w8a8_int8/glm5_w8a8_kunlun_2node.yaml)）：

```yaml
compression:
  quantization:
    ignore_layers:
      - ".indexer.wq_b"             # DSA query down-proj
      - ".indexer.wk"               # DSA key down-proj
      - ".indexer.weights_proj"     # DSA per-head gating
```

`kunlun` recipe 没有列 `kv_b_proj`——即让它参与 INT8 量化；也没有列 `lm_head` / `embed_tokens` / `mlp.gate` / `eh_proj`——因为这些已经在适配器的 `_FORCED_SKIP_SUBSTRINGS` 里，无需重复。

### 13.3 KV cache 行为（不改，纯动态）

`compression.quantization.kv_cache: "per-tensor"` 会让 [angelslim/models/llm/glm5.py](../angelslim/models/llm/glm5.py) `apply_kvcache_observers` 挂上：

- **MLA NoPE latent**：每 128 维一个 per-block observer（4 个 block），RoPE 尾部 bf16 直通
- **DSA indexer K**：一个 per-token observer

>**只是为了触发** `_extra_kv_cache_scheme` 写入 `config.json.quantization_config`——**不保存静态 scale**（无 `kv_cache_scales.safetensors`）， GPU FP8 布局完全同构，只是 fp8 → int8。

### 13.4 单元测

[tests/test_glm5_quant_layer_selection.py](../tests/test_glm5_quant_layer_selection.py)，纯 CPU、无 torch/GPU/权重依赖，秒级。

```bash
python3 tests/test_glm5_quant_layer_selection.py     # 无需 pytest
# 或
pytest tests/test_glm5_quant_layer_selection.py -v
```

 8 项：

| 测试 | 验证 |
|---|---|
| `test_whitelist_shape` | 观察白名单恰好是 10 个 leaf 名 |
| `test_hard_skip_covers_dangerous_layers` | hard-skip 覆盖 router / lm_head / embed / eh_proj / indexer.k_norm |
| `test_default_recipe_quantizes_mla_and_indexer_big_linears` | 默认 recipe 下 `kv_b_proj` / `indexer.wq_b` / `indexer.wk` 都量化 |
| `test_default_recipe_hard_skips` | 危险层始终不量化 |
| `test_user_ignore_layers_can_opt_out_of_kv_b_proj` | YAML 加 `"kv_b_proj"` 即可跳过 |
| `test_user_ignore_layers_can_opt_out_of_whole_indexer_subtree` | YAML 加三个 `.indexer.*` 子串即可跳过整个 indexer |
| `test_kunlun_recipe_end_to_end` | kunlun recipe 层选择完全正确 |
| `test_w8a8c8_recipe_end_to_end` | 历史 W8A8C8 recipe 层选择完全正确 |

bash /apdcephfs_sgfd2/share_300532381/harviexu/env_install.sh

```
8 passed, 0 failed
```

### 13.5 启动脚本

- W8A8C8：[scripts/ptq/run_glm5_w8a8c8_2node_0.sh](../scripts/ptq/run_glm5_w8a8c8_2node_0.sh) / `..._1.sh`
- kunlun：[scripts/ptq/run_glm5_w8a8_kunlun_2node_0.sh](../scripts/ptq/run_glm5_w8a8_kunlun_2node_0.sh) / `..._1.sh`

#------
bash /apdcephfs_sgfd2/share_300532381/harviexu/env_install. `SAVE_DIR` 换成你的输出目录）：

```bash
SAVE_DIR=/dockerdata/w8a8_kunlun_official/glm5_w8a8_kunlun_2node
python3 -c "
import json, sys
wm = json.load(open('$SAVE_DIR/model.safetensors.index.json'))['weight_map']
def has_scale(p): return any(p in k and '.weight_scale' in k for k in wm)
# kunlun recipe 期望
assert has_scale('kv_b_proj'),                 'kv_b_proj must be int8'
assert has_scale('kv_a_proj_with_mqa'),        'kv_a_proj must be int8'
assert not has_scale('.indexer.wk'),           'indexer.wk MUST be bf16'
assert not has_scale('.indexer.wq_b'),         'indexer.wq_b MUST be bf16'
assert not has_scale('.indexer.weights_proj'), 'weights_proj MUST be bf16'
assert not has_scale('eh_proj'),               'eh_proj MUST be bf16'
print('OK: layout matches recipe')
"
```

---

## 14. MTP 层（layer 78）Backfill 完整方案（2026-07-27 追加）

### 14.1 问题描述

GLM-5 的 `config.json` 中有两个层数字段：

| 字段 | 值 | 含义 |
|---|---|---|
| `num_hidden_layers` | 78 | 主 stack 的 transformer 层数（layer 0–77） |
| `num_nextn_predict_layers` | 1 | MTP（Multi-Token Prediction）草稿层的数量 |

发布的检查点里，MTP 层被存为 `model.layers.78.*`（**扁平命名**，无 `mtp_block.` 前缀），
包含 `self_attn.*` / `mlp.experts.*` / `mlp.shared_experts.*` / `mlp.gate.*` / `input_layernorm` /
`post_attention_layernorm` / `eh_proj` / `enorm` / `hnorm` 等约 791 个键。

然而 upstream `transformers` 的 `GlmMoeDsaModel.__init__` 只按 `num_hidden_layers` 构造
`self.layers`（长度 78），**根本没有 layer 78 这个 slot**。`from_pretrained` 会把磁盘上
所有 `model.layers.78.*` 键都当作 unexpected key **静默丢弃**。结果：

- 量化循环最多跑 layer 77（`len(model.layers) == 78`，实际 layer 78 slot 是 `_fix_hf_config`
 里为让 per-layer schedule 列表长度对齐而额外构造出的空 block，参数为 `None`/未加载）；
- `state_dict()` 里 layer 78 的键数是 **0**；
- 导出的 `model.safetensors.index.json` 里完全没有 layer 78 → 部署端做 speculative
 decoding 时直接失败。

### 14.2 修复思路：saver 侧旁路 backfill（方案 B1）

以下三条路径都不可行，最终选定第四条：

| 方案 | 说明 | 为什么放弃 |
|---|---|---|
| A. 修 upstream `modeling_glm_moe_dsa.py` | 让 `GlmMoeDsaModel` 按 `num_hidden_layers + num_nextn_predict_layers` 构造 layers | 侵入 upstream，且 MTP block 的模块类型（`GlmMoeDsaMtpBlock` vs `GlmMoeDsaDecoderLayer`）不同，命名结构也不同（`mtp_block.self_attn` vs `self_attn`），需要额外的键名重写；风险大、评审阻力大 |
| B0. 也把 MTP 参数**加载**进模型，再走同一套 INT8 量化循环 | 保持 pipeline 单一路径 | 需要 A 作为前置条件；此外主 stack 已跑完 40+ 分钟校准才走到 save，重跑成本高 |
| B2. 把 MTP experts 按 EP world_size 拆分，rank 0/1/… 各写自己那份 | 与主 stack 对称 | MTP experts 数量与主 stack 一致（256 个），拆分后每 rank 只多写 16 个 expert，收益微小；反而引入 rank 间协同复杂度和潜在死锁点 |
| **B1. rank 0 独占 backfill**（选定） | 由 rank 0 直接从原始 checkpoint 读取 layer 78 全部 791 个键、量化、写入一个专属 shard `model-mtp-r00.safetensors`；其他 rank barrier 等待 | 完全绕过 EP 协同、不动 upstream、不影响主 stack 的 40 分钟校准；MTP layer 只有一个，独占单 shard 也不会撑爆 shard 大小上限 |

### 14.3 实施：四处修改

> 单次运行必须四处都到位，缺一个 MTP 层就会重新丢失。

#### 14.3.1 [`angelslim/models/llm/glm5.py`](../angelslim/models/llm/glm5.py) — 记录源 checkpoint 路径

DeepSeek 通路里，`modeling_deepseek.py` 有 `cls.ori_model_path = model_path` 这一惯例，
saver 通过 `self.quant_model.model.ori_model_path` 反查源路径。GLM-5 用的是 upstream HF
模型类，**从未设过这个属性**，所以 saver 每次进 backfill 都会 abort。修复：

```python
def from_pretrained(self, model_path, *args, **kwargs):
    super().from_pretrained(model_path, *args, **kwargs)
    # 与 DeepSeek 约定对齐：把源路径落在 HF 模型对象上，
    # 让下游 saver 能反查回原始 tensor（尤其是 MTP layer 78）。
    try:
        self.model.ori_model_path = model_path
    except Exception:
        # 极少数 HF 模型类禁用了 __setattr__；退化到 adapter 侧。
        self.ori_model_path = model_path
    self._defuse_moe_experts()
    self._promote_ep_state()
```

#### 14.3.2 [`angelslim/compressor/quant/core/save.py`](../angelslim/compressor/quant/core/save.py) — 新增 `_maybe_emit_mtp_shard_from_source`

位于 `Glm5EPQuantSaver` 类内，在写 `model.safetensors.index.json` **之前**被调用。

**核心流程**（rank 0 独占；其它 rank 直接 return）：

```
1. guard 检查（任何一条失败就打印告警并 return，保证不影响主流程）
   ├── merged_index 中已有 model.layers.<mtp_id>.* → 已 backfilled，跳过
   ├── 从多路 fallback 定位源路径（见 14.3.3）
   ├── 从源 checkpoint 的 model.safetensors.index.json 反查 MTP 键归属
   └── config.json 里读取 num_hidden_layers + num_nextn_predict_layers

2. 用 safetensors.safe_open 惰性 mmap 打开源 shard（O(1) FD，无全量装载）

3. 对每个 model.layers.<mtp_id>.* 键：
   ├── 分类：是否命中 _QUANTIZABLE_LEAF_NAMES ∧ ¬_FORCED_SKIP_SUBSTRINGS
   │       ∧ ¬user_ignore_patterns
   ├── 量化分支：per-out-channel symmetric int8
   │   ├── scale = amax(|W|, dim=1) / 127
   │   ├── W_int8 = round(W / scale).clamp(-127, 127).to(int8)
   │   └── 写 {key}.weight（int8） + {key}.weight_scale（bf16, 一维）
   └── passthrough 分支：原样保存为 bf16（layernorm / eh_proj / enorm / hnorm /
       kv_b_proj / .indexer.* / mlp.gate / lm_head / embed 等）

4. 全部写入单一 shard model-mtp-r00.safetensors；追加到 merged weight_map
5. 打印摘要（量化数、passthrough 数、新增键总数）
```

**量化策略与主 stack 完全对齐**（用户明确要求）：
- **是**：所有 `nn.Linear` leaf（`self_attn.q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa` /
 `o_proj` / MoE experts 的 `gate_proj` / `up_proj` / `down_proj` / shared_experts）走 int8 per-channel。
- **否**：`eh_proj` / `enorm` / `hnorm` / `input_layernorm` / `post_attention_layernorm` /
 `.indexer.*` / `kv_b_proj` / `mlp.gate` —— 均由 `_FORCED_SKIP_SUBSTRINGS` +
 `_QUANTIZABLE_LEAF_NAMES` 组合自动落到 passthrough 分支，与 `_glm5_skip_lists.py`
 里的白名单**一次定义，主 stack 与 MTP 共用**。

#### 14.3.3 [`angelslim/compressor/quant/core/save.py`](../angelslim/compressor/quant/core/save.py) — 多路 fallback 定位源路径

为了健壮，源路径按优先级尝试 4 条：

```python
for _obj, _attr in (
    (_hf_model,  "ori_model_path"),   # DeepSeek 约定 + 14.3.1 新增（GLM-5）
    (_adapter,   "ori_model_path"),   # adapter 侧后备（GLM-5 的 try/except 分支）
    (_adapter,   "model_path"),       # 部分 YAML 会把 model.model_path 直挂 adapter
):
    _val = getattr(_obj, _attr, None)
    _candidates.append(f"{type(_obj).__name__}.{_attr}={_val!r}")
    if isinstance(_val, str) and _val and os.path.isdir(_val):
        src_path = _val
        break
if src_path is None:
    # HF 会在 from_pretrained 之后把加载路径写到 model.config._name_or_path
    _val = getattr(getattr(_hf_model, "config", None), "_name_or_path", None)
    _candidates.append(f"quant_model.model.config._name_or_path={_val!r}")
    if isinstance(_val, str) and _val and os.path.isdir(_val):
        src_path = _val
if src_path is None:
    print_info(
        f"[Glm5EPQuantSaver][MTP] ABORT: cannot locate source checkpoint "
        f"path. Tried: {_candidates}."
    )
    return
```

失败时打印所有候选值 —— 下次 debug 一眼定位到底哪条链路没配通。

#### 14.3.4 `_EXPERT_RE` 正则兼容 MTP 命名（提前性修复，本次未触发）

GLM-5 磁盘 checkpoint 的 MTP experts 是**扁平命名** `model.layers.78.mlp.experts.<i>.<proj>.weight`，
与主 stack 一致，所以现有正则本就匹配。但一些 fork 会用 `mlp.mtp_block.experts.*` 嵌套，
为此我把 `_EXPERT_RE` 提前扩容成：

```python
_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.(?:mtp_block\.)?mlp\.experts\."
)
```

—— 兼容两种命名，不影响主 stack 的匹配结果。

### 14.4 诊断脚本（先做探测再写 backfill 是关键）

在没有充分证据之前贸然写 backfill 代码，会引入下面这些坑：模块类型猜错、命名前缀猜错、
量化白名单漏项、shard 大小炸掉。所以本轮排查我们**先做诊断，再动 saver**。诊断分两级：

#### 14.4.1 独立磁盘诊断脚本：[`scripts/ptq/diag_glm5_mtp.py`](../scripts/ptq/diag_glm5_mtp.py)

秒级、不加载模型、只读 `config.json` + `model.safetensors.index.json`。输出：

- MTP layer id（= `num_hidden_layers`）
- schedule 列表（`indexer_types` / `mlp_layer_types` / `moe_layer_freq` 等）在 MTP 位置的值
- MTP 相关的所有磁盘键清单与所属 shard
- MTP 键的 schema 直方图（`self_attn.*` / `mlp.experts.*` / `eh_proj` / 归一化模块 …）
- 关键模块存在性检查（`eh_proj` / `enorm` / `hnorm`）
- 与一个正常 MoE 层的键集 diff（缺什么？多什么？）

用法：`python scripts/ptq/diag_glm5_mtp.py <checkpoint_path>`

#### 14.4.2 运行时深度诊断 `[DIAG-DEEP]`

内嵌在 `Glm5EPQuantSaver.save()` 里，在写 index.json 之前触发。逐一打印：

- `len(model.model.layers)` 实际值
- `type(model.model.layers[mtp_id])` 与 `named_children`
- `layers[mtp_id]` 所有参数的 `shape / dtype / device / is_meta / requires_grad`
- 所有 buffer 的同上信息

第一次跑就直接把根因锁在了 "MTP 参数从未进 state_dict" 这一条，避免了把 40 分钟校准
重跑一遍才知道错在哪。**任何"我以为它应该在"的假设都值得用 20 行诊断代码兑现一次**。

### 14.5 潜在风险与注意事项

| 风险 | 表现 | 缓解 |
|---|---|---|
| **量化配方偏移** | MTP 的量化决策与主 stack 使用**不同**的白名单/黑名单 → 推理时 speculative accept-rate 下降 | 14.3.2 里 backfill 复用 `_QUANTIZABLE_LEAF_NAMES` + `_FORCED_SKIP_SUBSTRINGS` + `user_ignore_patterns`，**同一份规则**决策；如果修改主 stack 的白名单，MTP 自动跟随 |
| **scale dtype 不一致** | 部分 saver 把 `weight_scale` 存 fp32，MTP shard 里存 bf16 → vLLM 加载时 shape/dtype 不匹配 | 14.3.2 里 scale 统一 cast 到 `torch.bfloat16` + `.reshape(-1)`，与主 stack `_finalize` 输出保持一致 |
| **passthrough 未 cast 到 bf16** | 原始 checkpoint 是 bf16 就没问题；如果是 fp32 会撑大 shard | 14.3.2 中 passthrough 分支强制 `.to(torch.bfloat16)`（这是 GLM-5 全局 dtype） |
| **单一 MTP shard 撑爆 5 GB 上限** | HF 建议单 shard ≤ 5 GB | GLM-5 一层 MTP ≈ 1.2 GB int8（experts 主导），远低于阈值；未来若 `num_nextn_predict_layers > 1` 需要按层再分片 |
| **多 rank 竞态写入** | 若 rank 1..N 也进入 backfill，会同时打开写同一个文件 | 14.3.2 首行 `if self.rank != 0: return` 严格 rank 0 独占；其他 rank 通过 barrier 等 |
| **磁盘 checkpoint 命名假设失效** | 未来 upstream 若把 MTP 改成 `model.mtp_layers.0.*` 或 `mtp_block.` 嵌套 | 诊断脚本会立刻检测到"MTP 键 0 个"，backfill 会直接跳过并打印警告，不产出静默错误 |
| **`ori_model_path` 属性冲突** | 某些 HF 模型类禁用 `__setattr__` | 14.3.1 的 try/except 已经退化到 adapter 侧；14.3.2 的 4 路 fallback 也会兜住 |
| **诊断打印占用 rank 0 stdout** | 大量 `[DIAG-DEEP]` 输出干扰主日志 | 生产可将其收敛到 `logger.debug`；本轮为了排障保留在 stdout |

### 14.6 修改文件清单（本轮）

| 文件 | 改动 |
|---|---|
| [`angelslim/models/llm/glm5.py`](../angelslim/models/llm/glm5.py) | `GLM5.from_pretrained` 里新增 `self.model.ori_model_path = model_path`（try/except 兜底到 adapter 侧） |
| [`angelslim/compressor/quant/core/save.py`](../angelslim/compressor/quant/core/save.py) | `_EXPERT_RE` 兼容 `mtp_block.` 嵌套；新增 `_maybe_emit_mtp_shard_from_source` 及其在 `save()` 里的调用点；guard-2 改成 4 路 fallback；新增 `[DIAG]` / `[DIAG-DEEP]` 诊断打印 |
| [`scripts/ptq/diag_glm5_mtp.py`](../scripts/ptq/diag_glm5_mtp.py) | 新增独立磁盘诊断脚本 |

### 14.7 验证

**shard 层面**：
```bash
SAVE_DIR=/apdcephfs_sgfd2/share_300532381/harviexu/kunlun_harvie/w8a8/glm5_w8a8_kunlun_2node
ls -lh $SAVE_DIR/model-mtp-r00.safetensors     # 应存在，约 1-2 GB
```

**index.json 层面**：
```bash
python3 -c "
import json
m = json.load(open('$SAVE_DIR/model.safetensors.index.json'))['weight_map']
n78 = sum(1 for k in m if k.startswith('model.layers.78.'))
print('layer 78 keys:', n78)
assert n78 > 1500, f'MTP backfill 缺失: got {n78}'
# 与参考对齐
ref = json.load(open('/apdcephfs_sgfd2/share_300532381/harviexu/kunlunw8a8/model.safetensors.index.json'))['weight_map']
missing = set(k for k in ref if k.startswith('model.layers.78.')) - set(m)
extra   = set(k for k in m   if k.startswith('model.layers.78.')) - set(ref)
print('missing vs ref:', len(missing), '  extra:', len(extra))
assert not missing and not extra
print('OK: MTP layer aligned with reference layout')
"
```

**vLLM 拉起**：确保推理端 speculative decoding 走的是 `model.layers.78`，acceptance rate 不显著低于全 bf16 版本（业务侧回归）。

---

**签字确认 — 2026-07-27**。MTP 层通过 rank 0 独占 backfill 补齐，与参考 `kunlunw8a8/model.safetensors.index.json` 完全对齐；主 stack 与 MTP 共用一套白名单/黑名单，量化决策一致，无 upstream 侵入。
