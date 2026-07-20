# MCoreQAD Backend

MCoreQAD is an isolated Megatron-Core backend for distributed, scale-only
quantization-aware training and distillation. It is maintained under
`angelslim/compressor/mcore_qad` and does not replace AngelSlim's existing
Hugging Face/DeepSpeed `QAT` or `QAD` implementations.

The backend freezes the original BF16 weights, applies fake quantization, and
optimizes only quantizer scales. When distillation is enabled, the same frozen
model with quantization disabled supplies the teacher logits.

## Support Matrix

- Models: Qwen3-MoE (`model_type: qwen3_moe`) and HunYuan-3
  (`model_type: hy_v3`).
- Formats: `nvfp4`, `nvfp4a16`, `w4a16`, `w8a8`, `fp8`, and `w4afp8`.
- Parallelism: TP, EP, CP, SP, and DP. Pipeline parallelism must remain `1`.
- Output: per-rank scale checkpoints and `angelslim_config.json`.
- Deployment export is intentionally not part of this integration stage.

## Installation

Use Python 3.12 and CUDA 12.8. Install the reference PyTorch wheel before
building Transformer Engine, because its PyTorch extension must be compiled
against the active torch/CUDA environment:

```bash
pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install megatron-core==0.18.0
pip install --no-build-isolation "transformer_engine[pytorch]>=2.16"
pip install -e ".[mcore-qad]"
```

The reference environment uses PyTorch 2.10, Megatron-Core 0.18, Transformer
Engine 2.16 or newer, and H100 GPUs with NVLink.

## Megatron Checkpoint Preparation

MCoreQAD trains from a reshardable Megatron distributed checkpoint. Convert a
supported Hugging Face safetensors checkpoint once. `tools/run.py` performs this
conversion automatically when `compression.MCoreQAD.checkpoint_path` is missing
or empty. Under `torchrun`, global rank zero converts while the other ranks wait.
Multi-node jobs must therefore use a checkpoint path on shared storage.

The converter can also be run explicitly:

```bash
python -m angelslim.compressor.mcore_qad.tools.hf_to_megatron \
  --hf /path/to/hf-model \
  --out /path/to/mcore-checkpoint
```

Add `--cpu`, or set `checkpoint_conversion_cpu: true` in YAML, when the full
model cannot be constructed on one GPU.

## Train from YAML

Update one of the supplied examples:

- `configs/qwen3/mcore_qad/qwen3_moe_nvfp4.yaml`
- `configs/hunyuan/mcore_qad/hy_v3_nvfp4.yaml`

Then launch with `torchrun`:

```bash
torchrun --nproc_per_node=8 tools/run.py \
  -c configs/qwen3/mcore_qad/qwen3_moe_nvfp4.yaml
```

The `torchrun` world size must be compatible with the TP, CP, and EP values in
the YAML. These MoE models require SP whenever TP is greater than one (and SP is
invalid with TP=1). CP requires `seq_len` to be divisible by
`2 * context_parallel`, and top-k distillation currently requires TP=1.

The main backend section is:

```yaml
compression:
  name: MCoreQAD
  MCoreQAD:
    checkpoint_path: /path/to/mcore-checkpoint
    checkpoint_conversion_cpu: false
    format: nvfp4
    lm_weight: 1.0
    distill_weight: 1.0
    distill_type: cakld
    seq_len: 2048
    micro_batch_size: 8
    train_iters: 128
    parallel:
      tensor_parallel: 1
      pipeline_parallel: 1
      expert_parallel: 8
      context_parallel: 1
      sequence_parallel: false
    optim:
      lr: 2.0e-4
      weight_decay: 0.0
      betas: [0.9, 0.95]
      grad_clip: 1.0
```

`model.model_path` points to the original Hugging Face directory,
`dataset.data_path` is optional (omitting it uses random tokens), and
`global.save_path` controls the scale-checkpoint output directory.
