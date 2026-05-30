# Distillation

AngelSlim Distill trains a full-precision student with an independent full-precision teacher. It is the fp-only distillation path: it does not initialize PTQ, QAT plugins, or quantized save logic. Use QAD when the student should be quantized during distillation.

## Features

- Load an independent teacher from `compression.Distill.teacher_model_path`.
- Train all full-precision student parameters with HuggingFace `Seq2SeqTrainer`.
- Combine supervised CausalLM loss and knowledge distillation loss with `lm_loss_weight` and `kd_loss_weight`.
- Support `kl`, `rkl`, `mse`, `kd`, `cakld`, `kl_top`, and `rkl_top` loss variants.
- Pass HuggingFace trainer options through `compression.Distill.hf_args`, including DeepSpeed ZeRO configs.
- Save the final student with HuggingFace `save_pretrained`.

## Example

This example distills a Qwen3-1.7B full-precision student from a Qwen3-4B full-precision teacher.

```bash
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/distill/fp/qwen3-1_7b_fp_distill_cakld_from_qwen3-4b_zero2.yaml
```

Key fields:

```yaml
model:
  model_path: Qwen/Qwen3-1.7B

compression:
  name: Distill
  Distill:
    teacher_model_path: Qwen/Qwen3-4B
    student_type: fp
    trainable_parameters: all
    save_format: hf
    loss_type: cakld
    lm_loss_weight: 1.0
    kd_loss_weight: 1.0
```

## Experiment Results

The following benchmark compares a Qwen3-1.7B base model with a Qwen3-1.7B full-precision student distilled from a Qwen3-4B teacher. PPL is not included in this table.

Experiment setting:

- Teacher: Qwen3-4B full-precision model.
- Student: Qwen3-1.7B full-precision model.
- Training data: Qwen3-4B teacher rollouts generated from public instruction datasets. See `dataset/qwen3_4b_rollout_10k/README.md` for the data construction workflow.
- Sequence length: `8192`.
- Global batch size: `32` with 8 GPUs, per-device batch size `1`, and gradient accumulation steps `4`.
- Loss: CausalLM loss plus CAKLD loss, both with weight `1.0`.
- Evaluation: generation-based benchmark with vLLM. IFEval generation is reported without the official strict scorer.

| Group | Task | Base | Distilled | Delta | Samples |
|---|---:|---:|---:|---:|---:|
| General | PIQA | 0.6638 | 0.7383 | +0.0745 | 1838 |
| General | ARC Easy | 0.8930 | 0.8912 | -0.0018 | 570 |
| General | ARC Challenge | 0.7258 | 0.7224 | -0.0034 | 299 |
| General | HellaSwag | 0.5908 | 0.6257 | +0.0349 | 10042 |
| General | Winogrande | 0.5446 | 0.5304 | -0.0142 | 1267 |
| General | MMLU | 0.5291 | 0.5096 | -0.0195 | 14042 |
| Reasoning | GSM8K | 0.7991 | 0.7612 | -0.0379 | 1319 |
| Reasoning | MATH subset | 0.6081 | 0.6040 | -0.0041 | 500 |
| Reasoning | BBH subset | 0.7000 | 0.8000 | +0.1000 | 250 |

## Dataset Format

`TextDataset` supports plain language-modeling data and chat-style SFT data. For chat-style JSONL data, set `is_sft_data: true`; prompt tokens are masked with `-100`, and only the final assistant response contributes to the loss.

```json
{
  "messages": [
    {"role": "user", "content": "Explain knowledge distillation."},
    {"role": "assistant", "content": "Knowledge distillation trains a smaller student model to match a larger teacher model."}
  ]
}
```

## Main Fields

```yaml
compression:
  name: Distill
  Distill:
    teacher_model_path: Qwen/Qwen3-4B
    teacher_torch_dtype: auto
    teacher_device_map: null
    student_type: fp
    trainable_parameters: all
    save_format: hf           # hf/full/real
    loss_type: cakld          # origin, kl, rkl, kd, cakld, mse, kl_top, rkl_top
    kd_temperature: 1.0
    lm_loss_weight: 1.0
    kd_loss_weight: 1.0
    hf_args:
      deepspeed: configs/qwen3/distill/fp/ds_config_zero2.json
```

Use `loss_type: origin` with `kd_loss_weight: 0.0` to run a supervised fine-tuning baseline with the same trainer path.
