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


"""GPTQ Hessian collection in vLLM workers: accumulate H = E[x x^T] under the forward hooks,
then dump it. Inactive unless ``VLLM_COLLECT_HESSIAN=1``.

Linear inputs come from forward hooks, routed-expert inputs from the fused-MoE wrapper in
``hooks`` (they never cross a module boundary). Independent of the min/max stats.
The CLI lives in ``hessian_collect``: the hook functions below are shipped to vLLM workers by
reference, so they must not be defined in the module that runs as ``__main__``."""

import json
import os
import re
import zlib

import torch

from . import hooks
from .hooks import patch_image_embeds
from .llm import get_draft_model_from_worker

__all__ = [
    "dump_hessian_stats",
    "dump_hessians",
    "get_collector",
    "hessian_enabled",
    "observe",
    "observe_moe",
    "observe_per_expert",
    "remove_hessian_hooks",
    "setup_hessian_hooks",
    "setup_hessian_hooks_on_draft",
]

# Default: every module the GPTQ quantizer handles; narrow via VLLM_HESSIAN_MODULES.
DEFAULT_MODULE_RE = (
    r"(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|indexer\.(wk|wq_b)"
    r"|gate_proj|up_proj|down_proj|gate_up_proj)"
)


def hessian_enabled():
    return os.getenv("VLLM_COLLECT_HESSIAN", "0") == "1"


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class HessianCollector:
    """Per-worker accumulator for the H groups this rank owns. Only MoE ``gate_up_proj``
    shares one layer-level H across experts; everything else gets its own."""

    def __init__(self):
        self.module_re = re.compile(os.getenv("VLLM_HESSIAN_MODULES", DEFAULT_MODULE_RE))
        self.max_tokens = _env_int("VLLM_HESSIAN_MAX_TOKENS", 400_000)
        # Per-expert budget: routing is skewed, so the group cap would leave most experts
        # under the rank floor.
        self.max_rows_per_expert = _env_int("VLLM_HESSIAN_ROWS_PER_EXPERT", 20_000)
        # Fraction of experts that must reach that budget before the group stops.
        self.expert_quota = float(os.getenv("VLLM_HESSIAN_EXPERT_QUOTA", "0.9"))
        self.row_stride = max(_env_int("VLLM_HESSIAN_ROW_STRIDE", 1), 1)
        # Groups with too few rows are dropped (rank deficient) so the quantizer falls back.
        self.min_tokens_factor = _env_int("VLLM_HESSIAN_MIN_TOKENS_FACTOR", 2)

        self.H = {}  # group -> [d, d] or [n_experts, d, d] float32, owner only
        self.counts = {}  # group -> int, or group -> [n_experts] list
        self.dims = {}
        self.per_expert = set()
        # Online smooth: down_proj weight has s baked in, so serving multiplies the input by 1/s;
        # replay it in place so H / input_scale see the same activation as serving.
        self.smooth_path = os.getenv("VLLM_MOE_ONLINE_SMOOTH_PATH", "")
        self._smooth_cpu = None  # {layer_idx: [n_experts, intermediate] fp32}, lazy
        self._smooth_dev = {}  # {layer_idx: tensor on the activation's device}
        self._smooth_warned = set()
        # Layer-aggregated min/max for moe_expert_stats.json; not subject to the H token cap.
        self.moe_stats = {}  # group -> {"min": float, "max": float}
        # Experts a shared MoE group stands in for (fans one H out at dump time).
        self.expert_fanout = {}
        self.bytes = 0
        # Kept on every rank: the token cap must be decided identically or gathers desync.
        self.seen = {}
        self.seen_per_expert = {}

        self.tp = None
        self.tp_size = 1
        self.tp_rank = 0
        try:
            from vllm.distributed.parallel_state import get_tp_group

            self.tp = get_tp_group()
            self.tp_size = self.tp.world_size
            self.tp_rank = self.tp.rank_in_group
        except Exception:
            self.tp = None

    def owner(self, group):
        """Stable owner rank, round-robin by layer index offset by a hash of the group kind
        to balance size. crc32, not hash(): hash() is salted per process."""
        if self.tp_size == 1:
            return 0
        m = re.search(r"\.(?:mtp_)?layers\.(\d+)\.", group)
        if m is None:
            return zlib.crc32(group.encode()) % self.tp_size
        kind = group[: m.start()] + group[m.end() :]
        return (int(m.group(1)) + zlib.crc32(kind.encode())) % self.tp_size

    def _admits(self, group, rows):
        """Whether to accumulate, decided identically on every rank."""
        if rows <= 0 or not self.module_re.search(group):
            return False
        seen = self.seen.get(group, 0)
        if self.max_tokens and seen >= self.max_tokens:
            return False
        self.seen[group] = seen + rows
        return True

    def _admits_per_expert(self, group, ids, n_experts):
        """Same symmetric decision, budgeted per expert (ids are TP-replicated)."""
        if not self.module_re.search(group) or ids.numel() == 0:
            return False
        seen = self.seen_per_expert.get(group)
        if seen is None:
            seen = torch.zeros(n_experts, dtype=torch.long)
            self.seen_per_expert[group] = seen
        if self.max_rows_per_expert:
            done = (seen >= self.max_rows_per_expert).float().mean().item()
            if done >= self.expert_quota:
                return False
        seen += torch.bincount(ids.clamp(min=0).to(torch.long), minlength=n_experts).cpu()[
            :n_experts
        ]
        return True

    def _prepare(self, group, x, sharded):
        """Flatten, subsample and (if sharded) all_gather x; the owner's [rows, d] fp32, else None.
        Every rank must reach the collective, so it runs before the owner check."""
        flat = x.reshape(-1, x.shape[-1])
        if self.row_stride > 1:
            flat = flat[:: self.row_stride]
        owner = self.owner(group)
        if sharded and self.tp is not None and self.tp_size > 1:
            flat = self.tp.all_gather(flat.contiguous(), dim=-1)
        if self.tp_rank != owner:
            return None
        return flat.float()

    def _alloc(self, group, shape, dim, device=None):
        """Allocate an accumulator and log the rank's running footprint."""
        H = torch.zeros(*shape, dtype=torch.float32, device=device)
        self.dims[group] = dim
        self.bytes += H.numel() * 4
        print(
            f"[hessian] rank {self.tp_rank} owns {group} {tuple(shape)} "
            f"({H.numel() * 4 / 2**30:.2f} GiB, {self.bytes / 2**30:.2f} GiB total)",
            flush=True,
        )
        return H

    def _accumulate(self, group, target, xf, count_key=None):
        """Running mean of x x^T (not a sum) to keep fp32 precision bounded."""
        rows = xf.shape[0]
        prev = self.counts[group] if count_key is None else self.counts[group][count_key]
        total = prev + rows
        target.mul_(prev / total).add_(xf.T @ xf, alpha=1.0 / total)
        if count_key is None:
            self.counts[group] = total
        else:
            self.counts[group][count_key] = total

    def _rows_used(self, x):
        rows = x.numel() // x.shape[-1]
        return (rows + self.row_stride - 1) // self.row_stride

    @staticmethod
    def _layer_idx(group):
        """Layer index from a group name, or None (MTP / dense / non-layer)."""
        m = re.search(r"\.layers\.(\d+)\.", group)
        return int(m.group(1)) if m else None

    def _smooth_for_layer(self, layer_idx, device, dtype):
        """Cached [n_experts, intermediate] smooth tensor for a layer, or None if off/absent."""
        if not self.smooth_path:
            return None
        if self._smooth_cpu is None:
            from safetensors.torch import load_file

            raw = load_file(self.smooth_path, device="cpu")
            self._smooth_cpu = {}
            for k, v in raw.items():
                m = re.search(r"\.layers\.(\d+)\.mlp\.experts\.w2_smooth_scale$", k)
                if m:
                    self._smooth_cpu[int(m.group(1))] = v.float()
            print(
                f"[hessian] online smooth: loaded {len(self._smooth_cpu)} layer scales "
                f"from {self.smooth_path}",
                flush=True,
            )
        cpu = self._smooth_cpu.get(layer_idx)
        if cpu is None:
            return None
        cached = self._smooth_dev.get(layer_idx)
        if cached is None or cached.device != device or cached.dtype != dtype:
            cached = cpu.to(device=device, dtype=dtype)
            self._smooth_dev[layer_idx] = cached
        return cached

    def apply_smooth(self, group, x, flat_ids):
        """In-place multiply the TP-sharded down_proj input by this rank's slice of 1/s."""
        layer = self._layer_idx(group)
        if layer is None:
            return x
        local = x.shape[-1]
        full = self._smooth_for_layer(layer, x.device, x.dtype)
        if full is None:
            return x
        n_experts, inter = full.shape
        if local * self.tp_size != inter:
            if layer not in self._smooth_warned:
                print(
                    f"[hessian] online smooth: layer {layer} shard mismatch "
                    f"local={local} tp={self.tp_size} inter={inter}; skipping",
                    flush=True,
                )
                self._smooth_warned.add(layer)
            return x
        start = self.tp_rank * local
        shard = full[:, start : start + local]  # [n_experts, local]
        ids = flat_ids.reshape(-1).clamp(min=0, max=n_experts - 1).long()
        x.mul_(shard.index_select(0, ids))  # [rows, local]
        return x

    def observe_moe_stats(self, group, x):
        """Update layer-aggregated min/max; the driver reduces across ranks (max of amaxes)."""
        if not self.module_re.search(group):
            return
        st = self.moe_stats.get(group)
        if st is None:
            st = {"min": float("inf"), "max": float("-inf")}
            self.moe_stats[group] = st
        with torch.no_grad():
            xf = x.float()
            st["min"] = min(st["min"], float(xf.min()))
            st["max"] = max(st["max"], float(xf.max()))

    def observe(self, group, x, sharded, expert_fanout=None):
        """Accumulate one H over ``x``; ``expert_fanout`` makes it serve that many experts."""
        if not self._admits(group, self._rows_used(x)):
            return
        if expert_fanout:
            self.expert_fanout[group] = expert_fanout
        xf = self._prepare(group, x, sharded)
        if xf is None:
            return
        dim = xf.shape[-1]
        if group not in self.H:
            self.H[group] = self._alloc(group, (dim, dim), dim, xf.device)
            self.counts[group] = 0
        self._accumulate(group, self.H[group], xf)

    def observe_per_expert(self, group, x, expert_ids, n_experts, sharded):
        """Accumulate one H per expert (MoE ``down_proj``), splitting ``x`` by ``expert_ids``."""
        ids = expert_ids.reshape(-1)
        if self.row_stride > 1:
            ids = ids[:: self.row_stride]
        if not self._admits_per_expert(group, ids, n_experts):
            return
        xf = self._prepare(group, x, sharded)
        if xf is None:
            return
        dim = xf.shape[-1]
        if group not in self.H:
            self.H[group] = self._alloc(group, (n_experts, dim, dim), dim, xf.device)
            self.counts[group] = [0] * n_experts
            self.per_expert.add(group)
        for e in ids.unique().tolist():
            if e < 0 or e >= n_experts:
                continue
            rows_e = xf[ids == e]
            if rows_e.shape[0] == 0:
                continue
            self._accumulate(group, self.H[group][e], rows_e, count_key=e)

    def dump(self, out_dir):
        """Write this rank's H to ``out_dir`` and return the meta for merging."""
        os.makedirs(out_dir, exist_ok=True)
        from safetensors.torch import save_file

        groups = {}
        modules = {}
        for group, H in self.H.items():
            dim = self.dims[group]
            fname = "h_" + re.sub(r"[^A-Za-z0-9]+", "_", group) + ".safetensors"
            floor = self.min_tokens_factor * dim
            if group in self.per_expert:
                counts = self.counts[group]
                keep = [e for e, c in enumerate(counts) if c >= floor]
                if not keep:
                    continue
                save_file({"H": H.cpu().contiguous()}, os.path.join(out_dir, fname))
                groups[group] = {
                    "file": fname,
                    "dim": dim,
                    "per_expert": True,
                    "tokens": int(min(counts[e] for e in keep)),
                    "experts_kept": len(keep),
                    "experts_total": len(counts),
                }
                base, _, stage = group.rpartition(".")
                for e in keep:
                    modules[f"{base}.{e}.{stage}"] = {"group": group, "expert": e}
                srt = sorted(counts)
                print(
                    f"[hessian] rank {self.tp_rank} {group}: kept {len(keep)}/{len(counts)} "
                    f"experts (floor {floor} rows), rows min/p10/median/max = "
                    f"{srt[0]}/{srt[len(srt) // 10]}/{srt[len(srt) // 2]}/{srt[-1]}",
                    flush=True,
                )
            else:
                if self.counts[group] < floor:
                    continue
                save_file({"H": H.cpu().contiguous()}, os.path.join(out_dir, fname))
                groups[group] = {
                    "file": fname,
                    "dim": dim,
                    "per_expert": False,
                    "tokens": int(self.counts[group]),
                }
                for key in self._module_keys(group):
                    modules[key] = {"group": group, "expert": None}

        meta = {"groups": groups, "modules": modules, "moe_stats": self.moe_stats}
        with open(os.path.join(out_dir, f"hessian_meta.rank{self.tp_rank}.json"), "w") as f:
            json.dump(meta, f)
        return meta

    def _module_keys(self, group):
        """Every imatrix key this non-per-expert group serves (fans out only for MoE gate_up)."""
        fanout = self.expert_fanout.get(group)
        if fanout:
            base, _, stage = group.rpartition(".")
            return [f"{base}.{e}.{stage}" for e in range(fanout)]
        return [group]


_COLLECTOR_ATTR = "_angelslim_hessian_collector"


def get_collector():
    """The one collector for this worker process, anchored on ``builtins`` so aliased
    imports of this package do not split it into separate (empty) instances."""
    import builtins

    collector = getattr(builtins, _COLLECTOR_ATTR, None)
    if collector is None:
        collector = HessianCollector()
        setattr(builtins, _COLLECTOR_ATTR, collector)
    return collector


def observe(group, x, sharded=False, expert_fanout=None):
    if hessian_enabled():
        get_collector().observe(group, x, sharded, expert_fanout=expert_fanout)


def observe_per_expert(group, x, expert_ids, n_experts, sharded=False):
    if hessian_enabled():
        get_collector().observe_per_expert(group, x, expert_ids, n_experts, sharded)


def dump_hessians(out_dir):
    if not hessian_enabled():
        return None
    return get_collector().dump(out_dir)


class LinearHessianHook:
    """Forward hook accumulating H over a linear layer's input (a class to stay pickle-safe)."""

    def __init__(self, layer_name, sharded):
        self.layer_name = layer_name
        # RowParallelLinear(input_is_parallel) sees only its slice of x; the cross-rank
        # blocks of E[x x^T] need a gather inside the hook.
        self.sharded = sharded

    def __call__(self, module, args, output):
        act = args[0] if isinstance(args, tuple) else args
        if not isinstance(act, torch.Tensor):
            return
        if act.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return
        with torch.no_grad():
            observe(self.layer_name, act, sharded=self.sharded)


def _gptq_moe_observer(stage, hidden_states, topk_ids, global_num_experts, w1):
    name = getattr(w1, "_gptq_layer_name", None)
    observe_moe(stage, hidden_states, topk_ids, global_num_experts, name)


def setup_hessian_hooks(model):
    """``apply_model`` entry point: register Linear hooks and tag each MoE layer's
    ``w13_weight`` (the only layer object ``TritonExperts.apply`` receives) with its name."""
    from vllm.model_executor.layers.linear import LinearBase

    if not hessian_enabled():
        return "VLLM_COLLECT_HESSIAN != 1, nothing registered"
    if getattr(model, "_gptq_hessian_hooks", None):
        return "hessian hooks already registered"

    # Swap the vision tower for the offline embedding cache when VLLM_IMAGE_EMBEDS_DIR is set.
    print("[hessian] " + patch_image_embeds(model), flush=True)

    col = get_collector()
    handles = []
    skipped = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, (torch.nn.Linear, LinearBase)):
            continue
        # Pre-filter so the count is accurate and unwanted layers pay nothing per forward.
        if not col.module_re.search(name):
            skipped += 1
            continue
        handles.append(
            mod.register_forward_hook(
                LinearHessianHook(name, bool(getattr(mod, "input_is_parallel", False)))
            )
        )

    moe = 0
    for name, mod in hooks.find_moe_layers(model).items():
        w13 = getattr(mod, "w13_weight", None)
        if w13 is None:
            print(f"[hessian] {name}: no w13_weight, in-kernel hook cannot fire", flush=True)
            continue
        w13._gptq_layer_name = name
        moe += 1

    hooks.install()
    hooks.register_observer(_gptq_moe_observer)

    model._gptq_hessian_hooks = handles
    return (
        f"hessian: {len(handles)} linear hooks ({skipped} linears filtered out), "
        f"{moe} MoE layers tagged, owner rank {col.tp_rank}/{col.tp_size}"
    )


def remove_hessian_hooks(model):
    for handle in getattr(model, "_gptq_hessian_hooks", []) or []:
        handle.remove()
    model._gptq_hessian_hooks = []
    return "hessian hooks removed"


def setup_hessian_hooks_on_draft(worker):
    """``collective_rpc`` entry point: arm the MTP draft block (unreachable via ``apply_model``).
    Group names keep the draft form ``layers.{num_hidden_layers+i}``, not ``mtp_layers.{i}``."""
    draft_model = get_draft_model_from_worker(worker)
    if draft_model is None:
        return None
    return "draft " + str(setup_hessian_hooks(draft_model))


def observe_moe(stage, hidden_states, topk_ids, global_num_experts, layer_name):
    """Called from ``TritonExperts.apply``: one shared H per layer for ``gate_up_proj`` (expert
    inputs agree closely; 2.8 TB -> 11 GB), one H per expert for ``down_proj`` (they do not)."""
    if not hessian_enabled() or layer_name is None:
        return
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(topk_ids, torch.Tensor):
        return
    # FP8 checkpoints quantize the gate_up input before this point; skip.
    if hidden_states.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return

    group = f"{layer_name}.{stage}"
    with torch.no_grad():
        rows_hs = hidden_states.shape[0]
        rows_topk, top_k = topk_ids.shape[0], topk_ids.shape[1]
        flat_ids = topk_ids.reshape(-1)

        if rows_hs == rows_topk:
            # gate_up_proj: [num_tokens, hidden]; expand to top_k rows to match flat_ids.
            flat_hidden = (
                hidden_states.unsqueeze(1)
                .expand(-1, top_k, -1)
                .reshape(-1, hidden_states.shape[-1])
            )
        elif rows_hs == rows_topk * top_k:
            # down_proj: already one row per (token, expert) pair.
            flat_hidden = hidden_states
        else:
            # Shape we don't recognise: skipping is right, guessing is not.
            return

        if stage == "down_proj":
            # Replay serving's online smooth (1/s) IN PLACE: the weight is baked W_down*s, so
            # the real forward needs it and H/absmax must see the smoothed activation.
            col = get_collector()
            col.apply_smooth(group, flat_hidden, flat_ids)
            col.observe_moe_stats(group, flat_hidden)
            observe_per_expert(group, flat_hidden, flat_ids, global_num_experts, sharded=True)
        else:
            # gate_up_proj input is TP-replicated and unsmoothed; still record absmax.
            get_collector().observe_moe_stats(group, flat_hidden)
            observe(group, flat_hidden, sharded=False, expert_fanout=global_num_experts)


def dump_hessian_stats(model):
    """``apply_model`` entry point: write this rank's H to ``VLLM_HESSIAN_DIR``, return only meta
    (the tensors run to hundreds of GB and must not travel back to the driver)."""
    del model  # the collector is a process-global, not model state
    out_dir = os.getenv("VLLM_HESSIAN_DIR", "")
    if not out_dir:
        return None
    return dump_hessians(out_dir)
