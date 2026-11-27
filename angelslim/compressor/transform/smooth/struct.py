"""Per-model offline Smooth pairs and runtime collection boundaries."""

import re

LAYER_PREFIXES = ("model.layers", "language_model.model.layers")
DEFAULT_MODULES = r"^model\.layers\.\d+\.self_attn\.(linear_gate|o_proj)$"
MODEL_PAIRS = {
    "hy4": {
        "attn_input": dict(
            first="input_layernorm",
            second=(
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "linear_gate",
                "indexer.wk",
                "indexer.weights_proj",
            ),
        ),
        "q_input": dict(first="self_attn.q_a_layernorm", second=("q_b_proj", "indexer.wq_b")),
        "kv_input": dict(first="self_attn.kv_a_layernorm", second=("kv_b_proj",)),
        "o_input": dict(first="self_attn.kv_b_proj", second=("o_proj",)),
    },
}


def resolve_model_name(model_name=None, *, profile=None):
    if not model_name:
        raise ValueError("Offline Smooth requires model.name from the YAML config")
    name = model_name.lower()
    if name not in MODEL_PAIRS:
        raise ValueError(f"Offline Smooth is not supported for model.name={name!r}")
    if profile not in (None, "offline", f"{name}_attention"):
        raise ValueError(f"Smooth profile {profile!r} does not match model.name={name!r}")
    return name


def get_smooth_pairs(model_name, reader):
    """Expand the model's pairs to the existing per-layer group dictionary format."""
    rules = MODEL_PAIRS[resolve_model_name(model_name)]
    pairs = []
    for layer in range(int(reader.config["num_hidden_layers"])):
        base = f"{reader.layer_prefix}.{layer}"
        attn = f"{base}.self_attn"
        for kind, rule in rules.items():
            first = f"{base}.{rule['first']}.weight"
            if not reader.has(first):
                continue
            consumers = [
                f"{attn}.{leaf}.weight"
                for leaf in rule["second"]
                if reader.has(f"{attn}.{leaf}.weight")
            ]
            pair = dict(
                name=f"layer{layer}.{kind}",
                consumers=consumers,
            )
            if kind == "o_input":
                pair.update(
                    producer=first,
                    selector=dict(
                        kind="head_strided_rows",
                        heads=int(reader.config["num_attention_heads"]),
                        head_width=int(reader.config["qk_nope_head_dim"])
                        + int(reader.config["v_head_dim"]),
                        offset=int(reader.config["qk_nope_head_dim"]),
                        length=int(reader.config["v_head_dim"]),
                    ),
                )
            else:
                pair["norm"] = first
            pairs.append(pair)
    return pairs


def get_hook_spec(model_name, name, modules):
    """Resolve a target input to a callable module boundary in the running model."""
    resolve_model_name(model_name)
    match = re.fullmatch(r"(model\.layers\.\d+\.self_attn)\.(.+)", name)
    if match is None:
        raise ValueError(f"Unsupported Smooth input: {name}")
    attn, leaf = match.groups()
    source, output = name, False
    if leaf == "linear_gate" and getattr(modules.get(attn), "use_hpc_gated_mla", False):
        source = attn
    elif leaf == "kv_b_proj" and attn + ".mla_attn" in modules:
        # MLA may absorb KV_b or apply it to cached tokens rather than this forward's input.
        source, output = attn + ".kv_a_layernorm", True
    elif name not in modules:
        aliases = {
            "q_a_proj": "fused_qkv_a_proj",
            "kv_a_proj_with_mqa": "fused_qkv_a_proj",
            "indexer.wk": "indexer.wk_weights_proj",
            "indexer.weights_proj": "indexer.wk_weights_proj",
            "q_b_proj": "q_a_layernorm",
            "kv_b_proj": "kv_a_layernorm",
        }
        source = attn + "." + aliases.get(leaf, leaf)
        output = leaf in ("q_b_proj", "kv_b_proj")
    if source not in modules:
        raise ValueError(f"No runtime input source for Smooth target {name}: {source}")
    mode = "concat" if getattr(modules[source], "input_is_parallel", False) else "max"
    return source, output, mode
