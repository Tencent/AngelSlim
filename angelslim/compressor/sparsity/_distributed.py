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

"""Distributed-runtime detection for sparse hard-fails.

Sparse is single-rank, single-node only. These detectors
return a best-effort suspected world size / parallelism signal; the patcher
hard-fails (``IncompatibleConfigError``) when any of them fires, rather than
silently producing a dense-labelled-sparse model.
"""

from __future__ import annotations

import os


def detect_world_size() -> int:
    """Return the distributed world size from the environment (1 if unset)."""
    for var in ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def detect_tp(model) -> int:
    """Best-effort **tensor-parallel** degree detection.

    Returns a suspected TP degree; >1 means the patcher must
    hard-fail. The distinction that matters: real TENSOR
    parallelism shards a single layer's weights across devices (each rank holds
    a slice of q_proj/o_proj), so the patched ``attn.forward`` would see only a
    shard and the all-reduce contract would break. In contrast, accelerate's
    ``device_map="auto"`` places **whole layers** on different devices and runs
    them sequentially in one process — the patched forward is called normally,
    so device-map layer sharding is fully compatible with sparse and must NOT
    be refused. The old code counted distinct device_map devices and so
    false-positived on every >=2-GPU ``device_map="auto"`` load (the flagship
    Stem YAML hard-failed on any multi-GPU box).

    Signals (any positive => refuse):
      1. transformers' own TP state: ``model._tp_size`` (set to the TP degree
         when loaded with ``tp_plan`` / a device mesh; ``None`` otherwise).
      2. torch.distributed initialised with world size > 1 AND no accelerate
         device_map (i.e. a genuine multi-process parallel run, not single-proc
         layer sharding).

    DeepSpeed-AutoTP and bare "vllm in sys.modules" vectors were REMOVED:
    they were dead / false-positive (read a class attr that is always None;
    fired on a mere import). The runtime guard ``detect_cuda_graph_or_compile``
    still catches the real vLLM-serving signal via an env var.
    """
    # 1. transformers' tensor-parallel size (the real, precise signal).
    tp_size = getattr(model, "_tp_size", None)
    if isinstance(tp_size, int) and tp_size > 1:
        return tp_size

    # 2. genuine multi-process torch.distributed run (not device_map sharding).
    dm = getattr(model, "hf_device_map", None)
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            if ws > 1 and not (isinstance(dm, dict) and len(dm) > 0):
                return ws
    except Exception:
        pass

    return 1


def detect_cuda_graph_or_compile(model) -> str:
    """Return a non-empty reason string if a CUDA-Graph / compile path is active.

    Vertical-slash style kernels use data-dependent indexing,
    incompatible with CUDA-graph stream capture and ``torch.compile(fullgraph=True)``.

    Two **real** signals are checked (a third, dead
    one — ``torch._dynamo.config.fullgraph``, an attribute that does not exist,
    so the old check was always False and advertised a protection we did not
    have). There is no reliable *static* signal for an active
    ``fullgraph=True`` compile at patch time (``torch.compiler.is_compiling()``
    is only true mid-trace); if a caller wraps the patched forward in
    ``torch.compile(fullgraph=True)`` it will fail loudly at first call — we do
    not pretend to pre-detect it.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "_compile_static_kv_cache", False):
        return (
            "model.config._compile_static_kv_cache is set; sparse attention is "
            "incompatible with CUDA graphs (data-dependent indexing). Disable "
            "static-KV-cache compilation or remove sparsity."
        )
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"):
        return (
            "vLLM serving worker detected; vLLM constructs its own attention "
            "modules and will not call the patched HF forward. Use the HF "
            "generate path for sparse."
        )
    return ""
