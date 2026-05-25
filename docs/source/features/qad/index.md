# Quantization-Aware Distillation

QAD trains a quantized student with an independent full-precision teacher. It is the bridge between the QAT and Distill paths: QAD reuses QAT quantization modules, learnable-scale plugins, conversion, and save logic, while sharing the Distill trainer and distillation losses.

Use `Distill` for full-precision students. Use `QAD` when the student should be quantized during training.

## Features

- Build the student with the same quantization configuration used by QAT.
- Load an independent teacher from `compression.QAD.teacher_model_path`.
- Train all student parameters or only quantization-related parameters with `trainable_parameters`.
- Reuse QAT plugin configuration, including learnable weight, activation, KV, norm, and LWC parameters.
- Support the same supervised and distillation loss composition as Distill.
- Save quantized outputs through QAT save formats such as `fake`, `real`, and `real_and_kvcache`.

## W4A8-FP8 Example

This example distills a W4A8-FP8 Qwen3-4B student from a full-precision Qwen3-4B teacher and trains only quantization parameters.

```bash
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/qad/w4a8_fp8/qwen3-4b_w4a8_fp8_qad_zero2.yaml
```

Key fields:

```yaml
compression:
  name: QAD
  quantization:
    name: w4a8_fp8
  QAD:
    teacher_model_path: Qwen/Qwen3-4B
    student_type: quantized
    trainable_parameters: quant
    save_format: real
    plugin_config:
      enable_scale: true
```

## Special Weight Quantizers

The special weight quantizer path keeps the standard `QuantLinear` wrapper and switches only the weight quantizer implementation through config. The Qwen3 examples are:

```text
configs/qwen3/qad/special/qwen3-1_7b_sherry_qad_from_qwen3-4b_zero2.yaml
configs/qwen3/qad/special/qwen3-1_7b_absmean_qad_from_qwen3-4b_zero2.yaml
configs/qwen3/qad/special/qwen3-1_7b_twn_qad_from_qwen3-4b_zero2.yaml
configs/qwen3/qad/special/qwen3-1_7b_lsq_qad_from_qwen3-4b_zero2.yaml
configs/qwen3/qad/special/qwen3-1_7b_seq_qad_from_qwen3-4b_zero2.yaml
configs/qwen3/qad/special/qwen3-1_7b_dlt_qad_from_qwen3-4b_zero2.yaml
```

Run one method by selecting its config:

```bash
torchrun --nproc_per_node=8 \
  tools/run.py \
  -c configs/qwen3/qad/special/qwen3-1_7b_sherry_qad_from_qwen3-4b_zero2.yaml
```

Key fields:

```yaml
plugin_config:
  enable_scale: true
  quant_config:
    use_weight_quant: true
    use_activation_quant: false
    weight_quantizer: special
    special:
      quant_method: sherry
      granularity: per_group
      group_size: 128
      w_bits: 1
      N: 3
      M: 4
```

A Hunyuan translation-style 2-bit SEQ QAD demo is also provided:

```text
configs/hunyuan/qad/special/hunyuan_seq_2bit_qad_zero2.yaml
```

Replace `model.model_path`, `compression.QAD.teacher_model_path`, and `dataset.data_path` with local model and dataset locations before running it.

## Main Fields

```yaml
compression:
  name: QAD
  quantization:
    name: w4a8_fp8
  QAD:
    teacher_model_path: Qwen/Qwen3-4B
    teacher_torch_dtype: auto
    teacher_device_map: cuda
    student_type: quantized
    trainable_parameters: quant  # all or quant
    save_format: real            # fake, real, save_kvcache_only, real_and_kvcache
    loss_type: cakld
    lm_loss_weight: 1.0
    kd_loss_weight: 1.0
    plugin_config:
      enable_scale: true
    hf_args:
      deepspeed: configs/qwen3/qad/w4a8_fp8/ds_config_zero2.json
```
