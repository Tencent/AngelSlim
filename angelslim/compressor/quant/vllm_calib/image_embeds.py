"""Offline ViT embeddings for Hy4-VL calibration: precompute them and patch vLLM to look them up.
``VLLM_IMAGE_EMBEDS_DIR`` arms the patch, removing the multi-image ViT activation spike.
CLI: ``python -m angelslim.compressor.quant.vllm_calib.image_embeds -c <yaml>``."""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import torch

__all__ = ["engine_kwargs", "patch_image_embeds", "precompute"]

_patched = False
_f = None  # safetensors handle (opened lazily)
_manifest = None
_verify_left = 0
_keys = None  # cached key set; ``safe_open.keys()`` re-reads the header
_warned_miss = False


def _hash_one(pv):
    """Key of one image: hash of its bf16 ``pixel_values`` (shared by precompute and lookup)."""
    pv = pv.to(torch.bfloat16).contiguous().cpu()
    return hashlib.blake2b(pv.view(torch.uint16).numpy().tobytes()).hexdigest()[:32]


class _TowerCall:
    """The tower with its unpatched ``forward``, for misses and VERIFY."""

    def __init__(self, visual, forward):
        self._visual, self._forward = visual, forward

    @property
    def dtype(self):
        return self._visual.dtype

    def __call__(self, pv, grid_thw):
        return self._forward(self._visual, pv, grid_thw)


def _hy_v4_vision():
    """Vision module of the ``vllm.models.hy_v4`` build, or None on the older layout."""
    try:
        from vllm.models.hy_v4.nvidia import vision
    except ImportError:
        return None
    return vision


def engine_kwargs():
    """``LLM`` kwargs the cache needs. Builds that normalise pixels on device hand the
    tower raw uint8, so the lookup keys (processor-normalised pixels) would never hit."""
    if not os.getenv("VLLM_IMAGE_EMBEDS_DIR", ""):
        return {}
    from vllm.engine.arg_utils import EngineArgs

    if "mm_device_do_normalize" in EngineArgs.__dataclass_fields__:
        return {"mm_device_do_normalize": False}
    return {}


def patch_image_embeds(model):
    """Install the embedding lookup (idempotent; no-op if unset or text-only); returns a status."""
    global _patched, _f, _manifest, _verify_left, _keys

    d = os.getenv("VLLM_IMAGE_EMBEDS_DIR", "")
    if not d:
        return "image-embeds: disabled (VLLM_IMAGE_EMBEDS_DIR unset)"
    if _patched:
        return "image-embeds: already patched"
    if getattr(model, "visual", None) is None:
        # Text-only checkpoint or no mm limit: no vision tower to bypass.
        return "image-embeds: skipped (model has no vision tower)"

    st = os.path.join(d, "image_embeds.safetensors")
    mf = os.path.join(d, "manifest.json")
    for p in (st, mf):
        if not os.path.isfile(p):
            raise RuntimeError(
                f"[image-embeds] {p} missing -- run "
                f"python -m angelslim.compressor.quant.vllm_calib.image_embeds first, or unset "
                f"VLLM_IMAGE_EMBEDS_DIR to use the live vision tower"
            )
    with open(mf, encoding="utf-8") as f:
        _manifest = json.load(f)

    from safetensors import safe_open

    _f = safe_open(st, framework="pt")
    _keys = set(_f.keys())
    _verify_left = 2 if os.getenv("VLLM_IMAGE_EMBEDS_VERIFY", "") == "1" else 0

    def _lookup(visual, pv, grid):
        global _verify_left, _warned_miss

        # On any miss, run the real tower for the whole request (never mix cached and live).
        hits, off, all_hit = [], 0, True
        try:
            for t, h, w in grid:
                n = int(t) * int(h) * int(w)
                key = _hash_one(pv[off : off + n])
                if key not in _keys:
                    all_hit = False
                    break
                hits.append(key)
                off += n
        except Exception:
            all_hit = False

        if not all_hit:
            if not _warned_miss:
                _warned_miss = True
                print(
                    "[image-embeds] cache miss -- falling back to the live "
                    "vision tower for this and any further uncached request "
                    "(re-run vllm_calib.image_embeds on the current "
                    "calibration data to restore the low-memory path)",
                    flush=True,
                )
            real = visual(pv, grid_thw=grid)
            return list(real) if isinstance(real, (list, tuple)) else [real]

        # safetensors reads land on CPU: move to the worker's device and tower dtype.
        outs = [_f.get_tensor(k).to(device=pv.device, dtype=visual.dtype) for k in hits]

        if _verify_left > 0:
            real = visual(pv, grid_thw=grid)
            real_list = list(real) if isinstance(real, (list, tuple)) else [real]
            cat_real = torch.cat(real_list)
            cat_emb = torch.cat(outs)
            diff = (cat_real.float() - cat_emb.float()).abs()
            relmax = (diff.max() / cat_real.float().abs().max().clamp(min=1e-9)).item()
            print(
                f"[image-embeds] VERIFY-DIAG: relmax={relmax:.3e} "
                f"maxdiff={diff.max().item():.6g} "
                f"meandiff={diff.mean().item():.6g} shape={tuple(cat_real.shape)}",
                flush=True,
            )
            if relmax > 0.05:
                raise RuntimeError(
                    f"[image-embeds] VERIFY FAILED: relmax {relmax:.3e} >> bf16 "
                    f"noise -- pipeline mismatch between precompute and engine"
                )
            _verify_left -= 1

        # One tensor per image: ``sanity_check_mm_encoder_outputs`` asserts len == #images.
        return outs

    # Patch the class, not the instance, since vLLM may hold several model references.
    if _hy_v4_vision() is not None:
        # vllm.models.hy_v4 build: embed_multimodal calls ``self.visual(pv, grids)`` directly.
        tower_cls = type(model.visual)
        tower_forward = tower_cls.forward

        def forward(visual, pixel_values, grid_thw):
            if torch.is_tensor(grid_thw):
                grid_thw = grid_thw.tolist()
            return _lookup(_TowerCall(visual, tower_forward), pixel_values, grid_thw)

        tower_cls.forward = forward
    else:
        # HYV4VL inherits ``_process_image_input`` from V3, so patching the V3 class covers both.
        from vllm.model_executor.models import hunyuan_v3_vision as m

        def _process_image_input(self, image_input):
            if image_input["type"] == "image_embeds":
                return image_input["image_embeds"].type(self.visual.dtype)
            grid = image_input["image_grid_thw"].tolist()
            return _lookup(self.visual, image_input["pixel_values"], grid)

        m.HYV3VLForConditionalGeneration._process_image_input = _process_image_input
    _patched = True
    return (
        f"image-embeds: patched ({len(_manifest)} embeddings, dir={d}, "
        f"verify={'on' if _verify_left else 'off'})"
    )


def collect_image_names(data_path):
    """Ordered, de-duplicated image file names referenced by the JSONL."""
    names, seen = [], set()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line) if line.strip() else {}
            except json.JSONDecodeError:
                continue
            for n in record.get("images") or []:
                if isinstance(n, str) and n not in seen:
                    seen.add(n)
                    names.append(n)
    return names


_VLLM_CONFIG_CTX = None


def build_vision_tower(model_path, device):
    """Standalone ``HYV4VisionTransformer`` on one device, with the engine's SDPA backend."""
    for k, v in (
        ("EP_SUPPRESS_NCCL_CHECK", "1"),
        ("EP_NCCL_ROOT_DIR", "/usr/local/tccl"),
        ("EP_NVSHMEM_ROOT_DIR", "/usr/local/nvshmem"),
        ("NVSHMEM_DIR", "/usr/local/nvshmem"),
        ("DEEPEP_NUM_CPU_TIMEOUT_SECONDS", "600"),
        ("DEEPEP_NUM_GPU_TIMEOUT_CYCLES_IN_BILLIONS", "400"),
    ):
        os.environ.setdefault(k, v)
    ldl = ":".join(
        ["/usr/local/tccl/lib", "/usr/local/nvshmem/lib", "/usr/local/cuda/lib64",
         "/usr/local/nvidia/lib64"]
    )  # fmt: skip
    old = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = ldl + (":" + old if old else "")

    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.attention.mm_encoder_attention as _mm
    import vllm.model_executor.models.vision as _vv
    from safetensors import safe_open
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from .llm import hf_config

    _hv = _hy_v4_vision()
    if _hv is None:
        import vllm.model_executor.models.hunyuan_v4_vision as _hv

    # Held in a module global: dropping the context manager lets GC run its finally,
    # which clears the config the tower's CustomOp layers read at init.
    global _VLLM_CONFIG_CTX
    _VLLM_CONFIG_CTX = set_current_vllm_config(VllmConfig())
    _VLLM_CONFIG_CTX.__enter__()

    # The engine runs the ViT with TORCH_SDPA; the name is bound in three modules.
    def _sdpa_backend(head_size, dtype):
        return AttentionBackendEnum.TORCH_SDPA

    _vv.get_vit_attn_backend = _hv.get_vit_attn_backend = _sdpa_backend
    _mm.get_vit_attn_backend = _sdpa_backend

    for k, v in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29555"), ("RANK", "0"),
                 ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):  # fmt: skip
        os.environ.setdefault(k, v)
    ps.init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="gloo"
    )
    ps.initialize_model_parallel(tensor_model_parallel_size=1)

    cfg = hf_config(model_path)
    visual = _hv.HYV4VisionTransformer(cfg.vision_config, quant_config=None, prefix="visual")
    visual = visual.to(torch.bfloat16).to(device).eval()
    # The engine keeps the position embedding in fp32.
    pos = visual.embeddings.position_embedding.weight
    pos.data = pos.data.to(torch.float32)

    with open(os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8") as f:
        idx = json.load(f)["weight_map"]
    shard_keys = defaultdict(list)
    for k in idx:
        if k.startswith("vit."):
            shard_keys[idx[k]].append(k)
    if not shard_keys:
        sys.exit(f"ERROR: no vit.* tensors in {model_path}; is this a VLM checkpoint?")

    def renamed_weights():
        for shard, ks in shard_keys.items():
            with safe_open(os.path.join(model_path, shard), framework="pt") as f:
                for k in ks:
                    yield k[len("vit.") :], f.get_tensor(k)

    loaded = visual.load_weights(renamed_weights())
    n_params = len(dict(visual.named_parameters()))
    print(f"tower weights loaded: {len(loaded)}/{n_params} params")
    if len(loaded) != n_params:
        sys.exit("ERROR: some tower parameters failed to load")
    return visual


def precompute(model_path, data_path, out_dir, image_dir=None, device="cuda:0"):
    """Run the tower one image at a time and write ``image_embeds.safetensors`` + manifest."""
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    image_dir = image_dir or os.path.join(os.path.dirname(os.path.abspath(data_path)), "images")
    names = collect_image_names(data_path)
    if not names:
        sys.exit(f"ERROR: no image references in {data_path}")
    print(f"images referenced: {len(names)}")

    visual = build_vision_tower(model_path, device)
    if _hy_v4_vision() is not None:
        # The processor the engine builds; transformers has no HYV4VL mapping on this build.
        from vllm.transformers_utils.processors.hy_v4 import HYV4VLProcessor

        proc = HYV4VLProcessor.from_pretrained(model_path)
    else:
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    tensors, manifest, n_fail = {}, {}, 0
    for i, name in enumerate(names, 1):
        try:
            # Must open images exactly like llm.build_prompts, or every hash misses.
            with Image.open(os.path.join(image_dir, name)) as im:
                rgb = im.convert("RGB")
            out = proc.image_processor(images=[rgb], return_tensors="pt")
            pv, grid = out["pixel_values"], out["image_grid_thw"].tolist()
            with torch.no_grad():
                emb = visual(pv.to(torch.bfloat16).to(device), grid_thw=grid)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            h = _hash_one(pv)
            if h in manifest:  # pixel-identical files share one embedding
                manifest[h]["images"].append(name)
            else:
                tensors[h] = emb.to(torch.bfloat16).cpu().contiguous()
                manifest[h] = {"image": name, "images": [name], "T": int(emb.shape[0]),
                               "grid": [list(g) for g in grid]}  # fmt: skip
        except Exception as e:
            n_fail += 1
            print(f"  WARN: {name}: {e}", file=sys.stderr)
        if i % 50 == 0 or i == len(names):
            print(f"  [{i}/{len(names)}] unique={len(tensors)} fail={n_fail}", flush=True)
    if not tensors:
        sys.exit("ERROR: no embeddings produced")

    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "image_embeds.safetensors"))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"done: {len(tensors)} embeddings -> {out_dir} ({n_fail} failed, live ViT for those)")


def main(argv=None):
    from .llm import parse_with_yaml

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", help="flat or nested AngelSlim YAML")
    p.add_argument("--model-path")
    p.add_argument("--data-path", dest="ptq_data_path")
    p.add_argument("--image-embeds-dir", help="output directory")
    p.add_argument("--image-dir", help="default: <data dir>/images")
    p.add_argument("--device")
    p.add_argument("--force", action="store_true", default=None, help="recompute the cache")
    args = parse_with_yaml(
        p, {"device": "cuda:0"}, argv, required=("model_path", "ptq_data_path", "image_embeds_dir")
    )
    out = args.image_embeds_dir
    if not args.force and os.path.isfile(os.path.join(out, "image_embeds.safetensors")):
        print(f"reusing {out} (--force to recompute)")
        return 0
    precompute(args.model_path, args.ptq_data_path, out, args.image_dir, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
