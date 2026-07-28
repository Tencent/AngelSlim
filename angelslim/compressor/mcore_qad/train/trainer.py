# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer: assemble the mcore QAT/QAD pipeline from a TrainConfig and run it.

Ties together: parallel init -> build mcore model (per-model adapter config) -> load
reshardable dist-checkpoint (FP weights) -> inject fake-quant -> (optional) load PTQ
scales -> training over ShareGPT/random data with LM(+distill) loss -> save scales.

Loss: total = lm_weight*lm + distill_weight*distill. When distill_weight>0 a quant-OFF
teacher forward is run (frozen weights => the FP model is the teacher).
"""

from __future__ import annotations

import torch

from angelslim.compressor.mcore_qad.train.config import TrainConfig
from angelslim.compressor.mcore_qad.train.optimizer import (
    build_optimizer,
    collect_scale_parameters,
)


def _hms(sec: float) -> str:
    sec = int(max(sec, 0.0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.rank = 0

    def setup(self) -> "Trainer":
        from megatron.core import parallel_state as ps

        from angelslim.compressor.mcore_qad.checkpoint import (
            load_dist_checkpoint,
            load_initial_scales,
        )
        from angelslim.compressor.mcore_qad.mcore.dist import init_model_parallel
        from angelslim.compressor.mcore_qad.mcore.model import build_gpt_model
        from angelslim.compressor.mcore_qad.mcore.quantize import quantize_mcore_model
        from angelslim.compressor.mcore_qad.models.base import (
            auto_config,
            load_hf_config,
        )
        from angelslim.compressor.mcore_qad.quant.presets import get_format
        from angelslim.compressor.mcore_qad.train.loss import build_loss

        p = self.cfg.parallel
        assert (
            p.pipeline_parallel == 1
        ), "Trainer uses a direct QAD loop (pp=1). For pp>1 use the pipeline schedule."
        if p.context_parallel > 1:
            assert (
                self.cfg.seq_len % (2 * p.context_parallel) == 0
            ), "context parallel needs seq_len divisible by 2*cp (zigzag load balancing)."
        self.rank, self.world, local = init_model_parallel(
            tp=p.tensor_parallel,
            pp=p.pipeline_parallel,
            ep=p.expert_parallel,
            cp=p.context_parallel,
            etp=1,
        )  # experts: EP-only, grouped
        self.device = torch.device("cuda", local)

        hf_cfg = load_hf_config(self.cfg.hf_path)
        self.model_type = hf_cfg["model_type"]
        cfg, meta = auto_config(
            hf_cfg,
            tp=p.tensor_parallel,
            pp=p.pipeline_parallel,
            ep=p.expert_parallel,
            cp=p.context_parallel,
            sequence_parallel=p.sequence_parallel,
            params_dtype=torch.bfloat16,
        )
        cfg.expert_tensor_parallel_size = 1  # experts sharded by EP only (grouped path)
        if self.cfg.recompute:  # full per-layer activation recompute
            cfg.recompute_granularity = "full"
            cfg.recompute_method = "uniform"
            cfg.recompute_num_layers = 1
        self.meta = meta
        pre = ps.is_pipeline_first_stage() if p.pipeline_parallel > 1 else True
        post = ps.is_pipeline_last_stage() if p.pipeline_parallel > 1 else True
        # TE FlashAttention is required for context parallelism and very long sequences.
        # With CP=1 the local attention core is sufficient and avoids a hard TE dependency.
        use_te_attention = p.context_parallel > 1
        self.model = build_gpt_model(
            cfg, meta, pre_process=pre, post_process=post, te_core_attention=use_te_attention
        ).to(self.device)
        load_dist_checkpoint(self.model, self.cfg.ckpt_path)
        # Quantize the non-expert linears (skip routed experts -- replaced just below), then
        # swap each MoE layer's per-expert SequentialMLP for stacked grouped-GEMM experts with
        # one-shot NVFP4 fake-quant: the fast, memory-light MoE path (no-op on dense models).
        from angelslim.compressor.mcore_qad.mcore.grouped_experts import (
            replace_moe_experts_with_grouped,
        )

        weight_spec, act_spec = get_format(self.cfg.fmt)
        if self.cfg.experts_only:
            # Quantize ONLY the routed experts (grouped path below): freeze the backbone and
            # skip every non-expert linear so attention / dense MLP / shared expert / lm_head
            # stay BF16 (the shared expert is MoELayer.shared_experts, untouched by grouping).
            self.model.requires_grad_(False)
            n_quant = 0
        else:
            n_quant = quantize_mcore_model(
                self.model,
                weight_spec,
                act_spec,
                skip_substr=("router", "output_layer", "local_experts"),
            )
        n_grouped = replace_moe_experts_with_grouped(self.model, weight_spec, act_spec)
        if self.rank == 0:
            print(
                f"[setup] quantized {n_quant} linears + grouped experts on {n_grouped} MoE "
                f"layers (experts_only={self.cfg.experts_only})",
                flush=True,
            )
        if self.cfg.init_scales_path:
            load_initial_scales(self.model, self.cfg.init_scales_path)
        if self.cfg.recompute:
            # Frozen backbone => the input embedding output has no grad, and activation
            # checkpointing keys its recomputed output's requires_grad off its INPUTS, so the
            # graph would detach. HF-style: make the input-embedding output require grad (a
            # gradient entry point; no parameter's frozen state changes).
            from angelslim.compressor.mcore_qad.mcore.model import (
                enable_input_require_grads,
            )

            enable_input_require_grads(self.model)

        # TP>1: keep logits TP-sharded and use vocab-parallel loss (no full-vocab gather) --
        # numerically identical to the dense loss, and essential for long context. TP=1 logits
        # are already full-vocab, so the dense loss is used directly.
        self.vocab_parallel = p.tensor_parallel > 1
        loss_kw = dict(
            lm_weight=self.cfg.lm_weight,
            distill_weight=self.cfg.distill_weight,
            distill_type=self.cfg.distill_type,
            temperature=self.cfg.distill_temperature,
            distill_topk=self.cfg.distill_topk,
        )
        if self.vocab_parallel:
            from angelslim.compressor.mcore_qad.train.loss_vp import VocabParallelLoss

            self.loss_fn = VocabParallelLoss(**loss_kw)
        else:
            self.loss_fn = build_loss(**loss_kw)
        self.optimizer = build_optimizer(self.model, self.cfg.optim)
        self.params = collect_scale_parameters(self.model)
        self.named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        from angelslim.compressor.mcore_qad.train.flops import fwd_flops_per_token

        self.fwd_flops_tok = fwd_flops_per_token(cfg, meta.vocab_size, self.cfg.seq_len)
        return self

    def _prepare(self, ids, labels):
        """Shift to per-position next-token targets, then (if CP) zigzag-shard the
        sequence across context-parallel ranks. Shifting MUST precede the CP split."""
        from megatron.core import parallel_state as ps

        b, s = ids.shape
        targets = labels.clone()
        targets[:, :-1] = labels[:, 1:]
        targets[:, -1] = -100  # last position has no next token
        pos = torch.arange(s, device=self.device).unsqueeze(0).expand(b, s).contiguous()
        # attention_mask=None: TE FlashAttention applies the causal mask internally (no [s,s]
        # score matrix), which is what makes 256k context feasible.
        if self.cfg.parallel.context_parallel > 1:
            from megatron.core.utils import get_pretrain_batch_on_this_cp_rank

            batch = get_pretrain_batch_on_this_cp_rank(
                {"input_ids": ids, "position_ids": pos, "labels": targets},
                ps.get_context_parallel_group(),
            )
            return (
                {
                    "input_ids": batch["input_ids"],
                    "position_ids": batch["position_ids"],
                    "attention_mask": None,
                },
                batch["labels"],
            )
        return {"input_ids": ids, "position_ids": pos, "attention_mask": None}, targets

    def _step(self, ids, labels):
        from angelslim.compressor.mcore_qad.train.switch import quant_disabled

        inputs, targets = self._prepare(ids, labels)
        gather = not self.vocab_parallel  # vocab-parallel loss keeps logits TP-sharded
        teacher_logits = None
        if self.loss_fn.needs_teacher:  # quant-OFF frozen-weight teacher
            with torch.no_grad(), quant_disabled(self.model):
                teacher_logits = self.model(**inputs, runtime_gather_output=gather)
        student_logits = self.model(**inputs, runtime_gather_output=gather)
        return self.loss_fn(student_logits, teacher_logits, targets)

    def _data_iter(self):
        """Data-parallel sharded iterator: each DP rank sees distinct batches, while
        ranks sharing a model-parallel (TP/PP/CP) group see identical inputs."""
        from megatron.core import parallel_state as ps

        from angelslim.compressor.mcore_qad.dataset import (
            load_hy_applied_batches,
            load_sharegpt_batches,
        )

        c = self.cfg
        # pure DP (exclude CP): CP ranks in a DP group share the full batch, then split it.
        dp_rank = ps.get_data_parallel_rank(with_context_parallel=False)
        dp_size = ps.get_data_parallel_world_size(with_context_parallel=False)
        if c.data_path:  # yields (ids, labels) batches
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(c.hf_path)
            pool = max(1, min(c.train_iters, 256)) * dp_size  # deterministic shared pool
            # hy_v3 ships pre-rendered `applied_message` jsonl (assistant-only loss); every
            # other model uses the ShareGPT chat-template loader.
            loader = (
                load_hy_applied_batches if self.model_type == "hy_v3" else load_sharegpt_batches
            )
            batches = loader(c.data_path, tok, c.seq_len, c.micro_batch_size, pool, self.device)
            i = dp_rank
            while True:
                yield batches[i % len(batches)]
                i += dp_size  # disjoint stride per DP rank
        else:
            g = torch.Generator().manual_seed(1234 + dp_rank)  # distinct data per DP rank
            while True:
                ids = torch.randint(
                    0, self.meta.vocab_size, (c.micro_batch_size, c.seq_len), generator=g
                ).to(self.device)
                yield ids, ids  # labels = ids (no padding)

    # fixed schema so the all-reduce tensor has the SAME size on every rank/step, even when a
    # CP shard is empty (n_tokens==0 -> metrics={}); a varying size would deadlock the collective.
    _METRIC_KEYS = ("loss", "lm", "fkl", "rkl", "cakld")

    def _reduce_metrics(self, loss, n_tokens: int, metrics: dict) -> dict:
        """Token-weighted means of loss + each metric across DP+CP, in ONE all-reduce."""
        import torch.distributed as dist
        from megatron.core import parallel_state as ps

        vals = {"loss": float(loss.detach()), **{k: float(v) for k, v in metrics.items()}}
        stat = torch.tensor(
            [vals.get(k, 0.0) * n_tokens for k in self._METRIC_KEYS] + [float(n_tokens)],
            device=self.device,
        )
        grp = ps.get_data_parallel_group(with_context_parallel=True)
        if grp is not None and dist.get_world_size(grp) > 1:
            dist.all_reduce(stat, group=grp)
        n = stat[-1].item()
        return {k: (stat[i] / n).item() if n > 0 else 0.0 for i, k in enumerate(self._METRIC_KEYS)}

    def train(self) -> None:
        import time

        from megatron.core import parallel_state as ps

        from angelslim.compressor.mcore_qad.checkpoint import save_scales
        from angelslim.compressor.mcore_qad.parallel.grad_sync import (
            all_reduce_data_parallel_grads,
        )

        c = self.cfg
        it = self._data_iter()
        # whole-model FLOPs/step = factor * (fwd/token) * tokens. Per student step: 1 fwd + 2 bwd
        # (+1 recomputed fwd if activation recompute is on); +1 fwd for the quant-off teacher.
        # CP splits the same tokens (no extra work) -> count full seq once over pure-DP groups.
        factor = 3.0 + (1.0 if c.recompute else 0.0) + (1.0 if self.loss_fn.needs_teacher else 0.0)
        dp_size = ps.get_data_parallel_world_size(with_context_parallel=False)
        tokens_per_step = c.micro_batch_size * c.seq_len * dp_size
        flops_per_step = factor * self.fwd_flops_tok * tokens_per_step
        if self.rank == 0:
            print(
                f"[train] iters={c.train_iters} tokens/step={tokens_per_step} "
                f"world={self.world} flops/step={flops_per_step:.3e} "
                f"(fwd/token={self.fwd_flops_tok:.3e}, factor={factor:g})",
                flush=True,
            )

        total_sanitized = 0
        torch.cuda.synchronize(self.device)
        t_start = t_prev = time.time()
        ema_dt = 0.0
        for step in range(c.train_iters):
            ids, labels = next(it)
            loss, n_tok, metrics = self._step(ids, labels)
            self.optimizer.zero_grad()
            loss.backward()
            all_reduce_data_parallel_grads(self.named_params)  # average grads over DP+CP
            # sanitize non-finite scale grads (nan/inf -> 0) and count how many params hit it.
            n_bad = 0
            for p in self.params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
                    n_bad += 1
            total_sanitized += n_bad
            gnorm = torch.nn.utils.clip_grad_norm_(self.params, c.optim.grad_clip)
            self.optimizer.step()

            red = self._reduce_metrics(loss, n_tok, metrics)  # token-weighted across DP+CP
            torch.cuda.synchronize(self.device)
            now = time.time()
            dt = now - t_prev
            t_prev = now
            ema_dt = dt if step <= 1 else 0.9 * ema_dt + 0.1 * dt  # seed past the warmup step
            if self.rank == 0:
                remaining = c.train_iters - step - 1
                tflops = flops_per_step / self.world / dt / 1e12
                toks = tokens_per_step / dt
                mem = torch.cuda.max_memory_allocated(self.device) / 1e9
                keys = ("lm", "fkl", "rkl", "cakld") if self.loss_fn.needs_teacher else ("lm",)
                m = " ".join(f"{k} {red[k]:.4f}" for k in keys)
                msg = (
                    f"iter {step + 1:>4}/{c.train_iters} | loss {red['loss']:.4f} | {m} "
                    f"| gnorm {float(gnorm):.2e} | {tflops:.1f} TFLOPS/gpu | {dt:.2f}s/it "
                    f"| {toks:.0f} tok/s "
                    f"| elapsed {_hms(now - t_start)} | eta {_hms(ema_dt * remaining)} "
                    f"| mem {mem:.1f}G"
                )
                if n_bad:
                    msg += f" | SANITIZED {n_bad}"
                print(msg, flush=True)
            if c.save_every and c.save_path and (step + 1) % c.save_every == 0:
                save_scales(self.model, c.save_path, tag=f"step{step + 1}")
        if self.rank == 0:
            print(
                f"[train] done in {_hms(time.time() - t_start)}; "
                f"non-finite-grad sanitizations: {total_sanitized}",
                flush=True,
            )
        if c.save_path:
            save_scales(self.model, c.save_path)
