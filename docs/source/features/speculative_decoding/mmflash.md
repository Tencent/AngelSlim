# MMFlash

**MMFlash** is a speculative decoding method for vision-language models (VLMs), from the paper **MMFlash: Faster Speculative Decoding for Vision-Language Models**. It builds on [DFlash](https://github.com/z-lab/dflash) block-diffusion drafting: the draft reads hidden states from five layers of the target and proposes a block of 16 tokens in one forward pass, and the target verifies the block in one forward pass. MMFlash keeps the DFlash draft architecture and is trained on multimodal data with a loss weighted by per-token visual-grounding (visual-KL) scores.

[[GitHub]](https://github.com/fscdc/MMFlash) | [[Project page]](https://fscdc.github.io/MMFlash/) | [[Hugging Face]](https://huggingface.co/collections/FSCCS/mmflash-6af09b489ba5237bda91177f)

---

## 📦 Draft models

| Target model | MMFlash draft | Block size | Target hidden states |
|---|---|---|---|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [FSCCS/Qwen3.5-4B-MMFlash](https://huggingface.co/FSCCS/Qwen3.5-4B-MMFlash) | 16 | layers 1, 8, 15, 22, 29 (of 32) |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [FSCCS/Qwen3.5-9B-MMFlash](https://huggingface.co/FSCCS/Qwen3.5-9B-MMFlash) | 16 | layers 1, 8, 15, 22, 29 (of 32) |

The checkpoints use the DFlash draft format, so they load with AngelSlim's `QwenDFlashDraftModel` without conversion. More drafts are listed in the [Hugging Face collection](https://huggingface.co/collections/FSCCS/mmflash-6af09b489ba5237bda91177f).

## ⚡ Quick Start

### Inference and evaluation (transformers)

`tools/mmflash_benchmark.py` runs MMFlash speculative decoding with Hugging Face transformers. It decodes every question twice with the same loop, once with block size 1 (the target alone, autoregressive) and once with the draft's block size, and reports the decoding speedup, the average acceptance length and its histogram, and how many outputs are identical to autoregressive decoding.

```shell
# fast Gated DeltaNet kernels for the Qwen3.5 linear-attention layers (strongly recommended)
pip install flash-linear-attention

python tools/mmflash_benchmark.py \
    --model-name-or-path Qwen/Qwen3.5-4B \
    --draft-name-or-path FSCCS/Qwen3.5-4B-MMFlash \
    --dataset chartqa \
    --max-samples 64 \
    --max-new-tokens 2048
```

One image and prompt:

```shell
python tools/mmflash_benchmark.py \
    --model-name-or-path Qwen/Qwen3.5-4B \
    --draft-name-or-path FSCCS/Qwen3.5-4B-MMFlash \
    --image /path/to/image.png \
    --prompt "Describe this image."
```

Multi-GPU evaluation (questions are sharded across ranks, results are gathered on rank 0):

```shell
torchrun --nproc_per_node=8 --master_port=29600 tools/mmflash_benchmark.py \
    --model-name-or-path Qwen/Qwen3.5-9B \
    --draft-name-or-path FSCCS/Qwen3.5-9B-MMFlash \
    --dataset mathvista \
    --max-samples 256
```

Notes:

- Without `flash-linear-attention`, transformers runs the Qwen3.5 linear-attention layers with its PyTorch reference kernels. Decoding still works, but a 16-token verification pass becomes much slower: in our ChartQA run with Qwen3.5-4B the speedup was 1.3× instead of about 3×.
- Supported datasets: `chartqa` (HuggingFaceM4/ChartQA test), `mmstar` (Lin-Chen/MMStar val) and `mathvista` (AI4Math/MathVista testmini), with the prompts of the MMFlash evaluation. `--max-samples` selects a fixed random subset (seed 0).
- `--block-size` defaults to the draft config (16). `--enable-thinking` turns on Qwen3.5's thinking mode (off by default, as in the MMFlash evaluation). `--output-file` writes every output and its timing to a JSON file.
- The average acceptance length is the mean number of tokens committed per verification step (accepted drafts plus the target's own token), averaged over questions, as in `tools/dflash_benchmark.py`.
- In bfloat16, a 16-token verification pass and a 1-token decoding step round differently, so a greedy output can leave the autoregressive one at a near-tied token, as in other serving engines. `tests/test_mmflash_inference.py` checks on a small float32 model that the outputs are identical for every acceptance length.
- The script disables PyTorch's cuDNN attention backend (`torch.backends.cuda.enable_cudnn_sdp(False)`, also in the Python example below). That backend rebuilds its execution plan whenever the key/value length changes, i.e. at every decoding step. With it enabled, autoregressive decoding ran about 8× and MMFlash decoding about 28× slower on an H200.

### Python API

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from angelslim.compressor.speculative.inference.mmflash import MMFlashGenerator
from angelslim.compressor.speculative.train.models.draft.qwen_dflash import QwenDFlashDraftModel

torch.backends.cuda.enable_cudnn_sdp(False)
target = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16)
target = target.cuda().eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
draft = QwenDFlashDraftModel.from_pretrained("FSCCS/Qwen3.5-4B-MMFlash", dtype=torch.bfloat16)
draft = draft.cuda().eval()

messages = [{"role": "user", "content": [
    {"type": "image", "image": "/path/to/image.png"},
    {"type": "text", "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True,
    return_tensors="pt", enable_thinking=False,
)
generator = MMFlashGenerator(target, draft)
out = generator.generate(inputs, max_new_tokens=1024,
                         stop_token_ids=processor.tokenizer.convert_tokens_to_ids(["<|im_end|>"]))
print(processor.decode(out.output_ids[0, out.num_input_tokens:], skip_special_tokens=True))
print("tokens per verification step:", sum(out.acceptance_lengths) / len(out.acceptance_lengths))
```

`generate(..., block_size=1)` decodes with the target alone, which is the baseline the speedup is measured against.

### How the decoding loop handles vision-language targets

`MMFlashGenerator` (`angelslim/compressor/speculative/inference/mmflash.py`) extends the DFlash loop of `tools/dflash_benchmark.py`:

- **Multimodal prefill.** The processor outputs (`pixel_values`, `image_grid_thw`, `mm_token_type_ids`) go into the first target forward. The multimodal RoPE offset that the target computes for the prompt (`rope_deltas`) is applied to the positions of every verification block.
- **Hybrid attention.** Qwen3.5 interleaves Gated DeltaNet (linear-attention) layers with full attention. When a draft suffix is rejected, it has to be removed from the convolution and recurrent states of the linear-attention layers, which `Cache.crop` cannot do. The loop saves those states before each verification forward and records the outputs of the layers' input projections during it. After acceptance it recomputes the states of the accepted prefix in one batched pass over all layers, with the flash-linear-attention kernel when it is installed. This is exact, because in a causal model the inputs at the accepted positions do not depend on the rejected tokens. Full-attention layers are cropped to the accepted length.
- The loop decodes one request at a time and runs on transformers 5.8 and later (tested with 5.8.1, 5.15.0 and 5.23.0). Targets with sliding-window attention are not supported. For serving, use SGLang or vLLM (below).

### Serving with SGLang and vLLM

SGLang and vLLM serve MMFlash drafts with their native DFlash support:

```shell
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path FSCCS/Qwen3.5-4B-MMFlash \
    --speculative-dflash-block-size 16 \
    --attention-backend fa3 \
    --mm-attention-backend sdpa \
    --disable-radix-cache \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

```shell
vllm serve Qwen/Qwen3.5-4B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --attention-backend FLASH_ATTN \
    --reasoning-parser qwen3 \
    --speculative-config '{"method": "dflash", "model": "FSCCS/Qwen3.5-4B-MMFlash", "num_speculative_tokens": 15}'
```

## 📈 Results

`tools/mmflash_benchmark.py` on one NVIDIA H200 (transformers 5.15, flash-linear-attention 0.6, bfloat16, SDPA, greedy, `--max-new-tokens 2048`; a random subset of each dataset). The speedup is the ratio of the mean time per output token, as in `tools/dflash_benchmark.py`. Eager transformers decoding is bound by CPU launch overhead (about 29 ms per step for both targets), so these numbers describe this loop rather than a serving engine.

| Target | Dataset | Questions | Avg. acceptance length | Autoregressive (tok/s) | MMFlash (tok/s) | Speedup |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | ChartQA | 20 | 5.74 | 34.4 | 105.8 | 3.12× |
| Qwen3.5-4B | MMStar | 20 | 4.48 | 34.2 | 75.9 | 2.29× |
| Qwen3.5-4B | MathVista | 20 | 6.67 | 34.2 | 98.0 | 3.45× |
| Qwen3.5-9B | ChartQA | 10 | 6.08 | 33.7 | 95.8 | 3.34× |
| Qwen3.5-9B | MMStar | 10 | 4.58 | 33.9 | 70.5 | 2.28× |

On SGLang with one NVIDIA H200 (temperature 0, concurrency 1), averaged over ChartQA, CharXiv, MMStar, MMBench, DynaMath, MathVista and MathVerse:

| Target | Average acceptance length | Decoding speedup |
|---|---|---|
| Qwen3.5-4B | 5.70 | 3.00× |
| Qwen3.5-9B | 5.60 | 3.05× |

Per-benchmark results and concurrency scaling are in the [4B](https://huggingface.co/FSCCS/Qwen3.5-4B-MMFlash) and [9B](https://huggingface.co/FSCCS/Qwen3.5-9B-MMFlash) model cards and the [GitHub repository](https://github.com/fscdc/MMFlash).

## 📜 Citation

```bibtex
@article{feng2026mmflash,
  title   = {MMFlash: Faster Speculative Decoding for Vision-Language Models},
  author  = {Feng, Sicheng and Chen, Zigeng and Wang, Song and Fang, Gongfan and Wang, Xinchao},
  journal = {arXiv preprint},
  year    = {2026}
}
```
