# Qwen3-4B Rollout Distillation Data

This directory documents how to build a Qwen3-4B teacher-rollout dataset for distillation. The generated JSONL files are intentionally not committed to the repository.

## Files

- `prepare_qwen3_rollout_data.py`: prompt sampling and vLLM rollout script.
- `README.md`: dataset construction notes.
- `.gitignore`: prevents generated JSONL data from being committed.

Generated files:

- `prompts_10k.jsonl`: sampled prompts before teacher rollout.
- `qwen3_4b_rollout_10k.jsonl`: final chat-style SFT data in AngelSlim `TextDataset` format.

## Construction Workflow

### 1. Build prompts

```bash
python prepare_qwen3_rollout_data.py \
  --mode build_prompts \
  --model-path Qwen/Qwen3-4B \
  --output-dir qwen3_distill_rollout_staging \
  --num-samples 10000
```

### 2. Generate teacher rollouts

```bash
python prepare_qwen3_rollout_data.py \
  --mode rollout \
  --model-path Qwen/Qwen3-4B \
  --output-dir qwen3_distill_rollout_staging \
  --tensor-parallel-size 8 \
  --batch-size 64 \
  --max-model-len 8192 \
  --max-new-tokens 2048
```

### 3. Use the generated data

Point `dataset.data_path` to the generated JSONL and enable SFT masking:

```yaml
dataset:
  name: TextDataset
  data_path: qwen3_distill_rollout_staging/qwen3_4b_rollout_10k.jsonl
  max_seq_length: 8192
  batch_size: 1
  is_sft_data: true
```

With `is_sft_data: true`, `TextDataset` masks prompt tokens with `-100` and supervises only the final assistant response.

## Prompt Sources

Prompts are sampled from public datasets:

- `HuggingFaceH4/ultrachat_200k`: general instruction
- `teknium/OpenHermes-2.5`: general instruction
- `TIGER-Lab/MathInstruct`: reasoning and math
- `ise-uiuc/Magicoder-Evol-Instruct-110K`: code
- `Yukang/LongAlpaca-12k`: long context

## Filtering

The script applies the following filters before writing the final data:

- Normalized exact deduplication.
- Prompt token length bounded by `--max-prompt-tokens`.
- Total prompt plus response length bounded by `--max-model-len`.
- Obvious benchmark names such as `MMLU`, `GSM8K`, `ARC`, `HellaSwag`, and `Winogrande` are filtered from prompts.
- Empty, too-short, or refusal-template responses are removed.

## Example Data Statistics

One 10k-prompt rollout run produced:

- Prompt candidates: `10000`
- Final valid rollout samples: `9887`
- Categories:
  - `general_instruction`: `4887`
  - `reasoning_math`: `2000`
  - `code`: `1500`
  - `long_context`: `1500`
- Thinking mode:
  - `true`: `3500`
  - `false`: `6387`
- Token stats:
  - Prompt tokens: min `26`, p50 `86`, p95 `709`, max `6154`
  - Response tokens: min `16`, p50 `869`, p95 `2048`, max `2051`
  - Total tokens: min `52`, p50 `1044`, p95 `2174`, max `8153`

## JSONL Format

Each generated row uses chat-style messages:

```json
{
  "id": "reasoning_math_000001",
  "source": "TIGER-Lab/MathInstruct",
  "category": "reasoning_math",
  "rollout_model": "Qwen/Qwen3-4B",
  "thinking": true,
  "prompt_tokens": 128,
  "response_tokens": 512,
  "total_tokens": 640,
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
