# EAGLE

[Eagle3](https://arxiv.org/pdf/2503.01840)是目前最常用、加速效果最好的投机采样算法。
本项目包括Eagle3的训练以及benchmark测试，并开源了Hunyuan、HunyuanOCR、Qwen3、Qwen3-VL、Qwen2Audio、Fun-CosyVoice3等模型的[Eagle3权重](https://huggingface.co/collections/AngelSlim/eagle3)。

我们训练的Eagle3模型的表现可以参见基准测试[benchmarks](../../../performance/speculative_decoding/benchmarks.md)，
其中全部数据都是在单张H20上使用vLLM推理获得。

:::{toctree}
:caption: Contents
:maxdepth: 1

eagle
vlm_eagle
audio_asr_eagle
audio_tts_eagle
:::
