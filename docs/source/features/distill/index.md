# Distillation

AngelSlim Distill trains a student model with a separate full-precision teacher model. The student can be either a full-precision HuggingFace model or a quantized QAT-style model.

## Features

- Load an independent teacher from `compression.Distill.teacher_model_path`.
- Support full-precision students with HuggingFace `save_pretrained` output.
- Support quantized students by reusing QAT quantization, learnable-scale plugins, conversion, and save paths.
- Select trainable parameters with `trainable_parameters: all` or `trainable_parameters: quant`.
- Combine supervised CausalLM loss and knowledge distillation loss with `lm_loss_weight` and `kd_loss_weight`.
- Pass HuggingFace trainer options through `compression.Distill.hf_args`, including DeepSpeed ZeRO configs.

## Example 1: Distill a Smaller Full-Precision Model

This example distills a Qwen3-1.7B full-precision student from a Qwen3-4B full-precision teacher.

```bash
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/distill/fp/qwen3-1_7b_fp_distill_from_qwen3-4b_zero2.yaml
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

## Example 2: Distill a Quantized Model

This example distills a W4A8-FP8 Qwen3-4B student from a full-precision Qwen3-4B teacher. The quantized student reuses QAT learnable-scale plugins and trains only quantization parameters.

```bash
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/distill/w4a8_fp8/qwen3-4b_w4a8_fp8_distill_zero2.yaml
```

Key fields:

```yaml
compression:
  name: Distill
  quantization:
    name: w4a8_fp8
  Distill:
    teacher_model_path: Qwen/Qwen3-4B
    student_type: quantized
    trainable_parameters: quant
    save_format: real
    plugin_config:
      enable_scale: true
```

## Example 3: Distill with Special Weight Quantizers

The special weight quantizer path keeps the standard `QuantLinear` wrapper and switches only the weight quantizer implementation through config. Six demo configs are provided:

```text
configs/qwen3/distill/special/qwen3-1_7b_sherry_distill_from_qwen3-4b_zero2.yaml
configs/qwen3/distill/special/qwen3-1_7b_absmean_distill_from_qwen3-4b_zero2.yaml
configs/qwen3/distill/special/qwen3-1_7b_twn_distill_from_qwen3-4b_zero2.yaml
configs/qwen3/distill/special/qwen3-1_7b_lsq_distill_from_qwen3-4b_zero2.yaml
configs/qwen3/distill/special/qwen3-1_7b_seq_distill_from_qwen3-4b_zero2.yaml
configs/qwen3/distill/special/qwen3-1_7b_dlt_distill_from_qwen3-4b_zero2.yaml
```

Run one method by selecting its config:

```bash
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/distill/special/qwen3-1_7b_sherry_distill_from_qwen3-4b_zero2.yaml
```

A Hunyuan translation-style 2-bit SEQ distillation demo is also provided:

```text
configs/hunyuan/distill/special/hunyuan_seq_2bit_distill_zero2.yaml
```

Replace `model.model_path`, `compression.Distill.teacher_model_path`, and `dataset.data_path` with local model and translation-data paths before running it.

Key fields:

```yaml
plugin_config:
  enable_scale: true
  quant_config:
    use_weight_quant: true
    use_activation_quant: false
    weight_quantizer: special
    special:
      quant_method: sherry  # sherry, absmean, twn, lsq, seq, or dlt
      granularity: per_group
      group_size: 128
      w_bits: 1
      N: 3
      M: 4
```

For the 2-bit SEQ demo, the special weight quantizer uses per-channel scaling:

```yaml
plugin_config:
  enable_scale: true
  quant_config:
    use_weight_quant: true
    use_activation_quant: false
    weight_quantizer: special
    special:
      quant_method: seq
      granularity: per_channel
      w_bits: 2
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

## Main Distill Fields

```yaml
compression:
  name: Distill
  Distill:
    teacher_model_path: Qwen/Qwen3-4B
    teacher_torch_dtype: auto
    teacher_device_map: null
    student_type: fp          # fp or quantized
    trainable_parameters: all # all or quant
    save_format: hf           # hf/full/real for fp; real/fake for quantized paths
    loss_type: cakld          # origin, kl, rkl, kd, cakld, mse, kl_top, rkl_top
    kd_temperature: 1.0
    lm_loss_weight: 1.0
    kd_loss_weight: 1.0
    hf_args:
      deepspeed: configs/qwen3/distill/w4a8_fp8/ds_config_zero2.json
```

Use `loss_type: origin` with `kd_loss_weight: 0.0` to run a supervised fine-tuning baseline with the same trainer path.
