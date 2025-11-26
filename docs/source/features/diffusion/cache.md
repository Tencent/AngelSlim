# Diffusion模型Cache

AngelSlim 提供高效灵活的 Diffusion Transformer (DiT) 模型 Cache 能力。通过复用中间计算结果，显著减少推理步骤中的重复计算，从而加速图像生成过程。

## 支持的Cache类型

AngelSlim 支持以下三种 Cache 策略：

- **DeepCache**：基于块级别的缓存控制，可以灵活指定哪些块和步骤需要缓存，提供细粒度的缓存管理
- **TeaCache**：基于残差的缓存策略，通过存储输入输出的残差（差值）来高效复用缓存，适合连续步骤间变化较小的场景
- **TaylorCache**：基于泰勒展开的缓存策略，使用泰勒级数预测未来输出，将张量分解为低频和高频分量以提高近似精度

## 配置

### CacheHelper 基类

所有 Cache 策略的基类，提供基础的缓存管理功能：

#### 构造函数参数

- `double_blocks`（List, 可选）：需要缓存的 double block 模块列表
- `single_blocks`（List, 可选）：需要缓存的 single block 模块列表
- `no_cache_steps`（Set[int], 可选）：指定不使用缓存的步骤集合

#### 主要方法

- `enable()`：启用缓存功能
- `disable()`：禁用缓存功能并恢复原始前向方法
- `reset_states()`：重置所有内部状态
- `clear_states()`：清除缓存状态但保留函数字典

### DeepCacheHelper

继承自 CacheHelper，提供块级别的缓存控制：

#### 额外参数

- `no_cache_block_id`（Dict[str, Set[int]], 可选）：字典，映射块类型（"double_blocks"、"single_blocks"）到不应缓存的块 ID 集合

### TeaCacheHelper

继承自 CacheHelper，实现基于残差的缓存策略：

#### 额外参数

- `cache_name`（str, 可选）：kwargs 中要缓存的输入字段名称（默认："img"）

### TaylorCacheHelper

继承自 CacheHelper，实现基于泰勒展开的缓存策略：

#### 额外参数

- `max_order`（int, 可选）：泰勒展开的最大阶数（默认：2）
- `low_freqs_order`（int, 可选）：计算低频导数的阶数（默认：2）
- `high_freqs_order`（int, 可选）：计算高频导数的阶数（默认：2）

## 使用方法

### HunyuanVideo-1.5

```python
import torch
from hyvideo.commons.infer_state import get_infer_state
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
from angelslim.compressor.diffusion import DeepCacheHelper, TeaCacheHelper, TaylorCacheHelper

# 创建 CacheHelper 并启用缓存
transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
    os.path.join(cached_folder, "transformer", transformer_version), torch_dtype=transformer_dtype, 
    low_cpu_mem_usage=True,
).to(transformer_init_device)

infer_state = get_infer_state()
if infer_state.enable_cache:
    no_cache_steps = list(range(0, infer_state.cache_start_step)) + list(range(infer_state.cache_start_step, infer_state.cache_end_step, infer_state.cache_step_interval)) + list(range(infer_state.cache_end_step, infer_state.total_steps))
    cache_type = infer_state.cache_type
    if cache_type == 'deepcache':
        no_cache_block_id = {"double_blocks":infer_state.no_cache_block_id}
        cache_helper = DeepCacheHelper(
            double_blocks=transformer.double_blocks,
            no_cache_steps=no_cache_steps,
            no_cache_block_id=no_cache_block_id,
        )
    elif cache_type == 'teacache':
        cache_helper = TeaCacheHelper(
            double_blocks=transformer.double_blocks,
            no_cache_steps=no_cache_steps,
        )
    elif cache_type == 'taylorcache':
        cache_helper = TaylorCacheHelper(
            double_blocks=transformer.double_blocks,
            no_cache_steps=no_cache_steps,
        )
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    cache_helper.enable()
else:
    cache_helper = None

# 修改 pipeline 文件
# 修改pipeline推理函数，在每个时间步iteration过程中把当前时间步赋值给cache_helper.cur_timestep，并清理cache的状态。
class HunyuanVideo_1_5_Pipeline(DiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        ...,
        **kwargs,
    ):
        cache_helper = getattr(self, 'cache_helper', None)
        if cache_helper is not None:
            cache_helper.clear_states()
            assert num_inference_steps == get_infer_state().total_steps
        # Denoising loop， set cur_timestep
        for i, t in enumerate(timesteps):
            if cache_helper is not None:
                cache_helper.cur_timestep = i

# 加载模型
pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
    pretrained_model_name_or_path=args.model_path,
    transformer_version=transformer_version,
    enable_offloading=args.offloading,
    enable_group_offloading=args.group_offloading,
    overlap_group_offloading=args.overlap_group_offloading,
    create_sr_pipeline=enable_sr,
    force_sparse_attn=args.sparse_attn,
    transformer_dtype=transformer_dtype,
)

# 运行推理
out = pipe(
    enable_sr=enable_sr,
    prompt=args.prompt,
    aspect_ratio=args.aspect_ratio,
    num_inference_steps=args.num_inference_steps,
    sr_num_inference_steps=None,
    video_length=args.video_length,
    negative_prompt=args.negative_prompt,
    seed=args.seed,
    output_type="pt",
    prompt_rewrite=enable_rewrite,
    return_pre_sr_video=args.save_pre_sr_video,
    **extra_kwargs,
)
```
