"""Hy4 trunk attention: fold channel scales into Norms or KV_b V rows."""
import re

MODEL_TYPES = ("hy_v4", "hy_v4_vl", "hy_v4_internal", "hy_v4_vl_internal")
LEGACY_PROFILE = "hy4_attention"
LAYER_PREFIXES = ("model.layers", "language_model.model.layers")
DEFAULT_MODULES = r"^model\.layers\.\d+\.self_attn\.(linear_gate|o_proj)$"
OFFLINE_SMOOTH_GROUPS = {
    "attn_input": dict(
        consumers=("q_a_proj", "kv_a_proj_with_mqa", "linear_gate",
                   "indexer.wk", "indexer.weights_proj"),
        stat="q_a_proj", norm="input_layernorm",
    ),
    "q_input": dict(consumers=("q_b_proj", "indexer.wq_b"),
                    stat="q_b_proj", norm="self_attn.q_a_layernorm"),
    "kv_input": dict(consumers=("kv_b_proj",),
                     stat="kv_b_proj", norm="self_attn.kv_a_layernorm"),
    "o_input": dict(consumers=("o_proj",), stat="o_proj", producer="kv_b_proj"),
}
_INPUT_ALIASES = {"fused_qkv_a_proj": "q_a_proj"}
_OUTPUT_ALIASES = {"kv_a_layernorm": "kv_b_proj", "q_a_layernorm": "q_b_proj"}
_DEFAULT_STATS = ("q_a_proj", "kv_b_proj", "o_proj")


def group_specs(reader):
    for layer in range(int(reader.config["num_hidden_layers"])):
        base = f"{reader.layer_prefix}.{layer}"
        attn, stat_base = f"{base}.self_attn", f"model.layers.{layer}.self_attn"
        for kind, rule in OFFLINE_SMOOTH_GROUPS.items():
            consumers = [f"{attn}.{leaf}.weight" for leaf in rule["consumers"]
                         if reader.has(f"{attn}.{leaf}.weight")]
            spec = dict(name=f"layer{layer}.{kind}", stat=f"{stat_base}.{rule['stat']}",
                        consumers=consumers)
            if "norm" in rule:
                spec["norm"] = f"{base}.{rule['norm']}.weight"
                if not reader.has(spec["norm"]):
                    continue
            else:
                spec.update(
                    producer=f"{attn}.{rule['producer']}.weight",
                    producer_stat=f"{stat_base}.{rule['producer']}",
                    selector=dict(kind="head_strided_rows",
                                  heads=int(reader.config["num_attention_heads"]),
                                  head_width=int(reader.config["qk_nope_head_dim"])
                                  + int(reader.config["v_head_dim"]),
                                  offset=int(reader.config["qk_nope_head_dim"]),
                                  length=int(reader.config["v_head_dim"])),
                )
                if not reader.has(spec["producer"]):
                    continue
            yield spec


def hook_spec(runtime_name, module, num_hidden_layers, stat_names):
    if any(part.startswith(("mtp", "draft", "dfly", "visual", "vit"))
           for part in runtime_name.split(".")):
        return None
    match = re.search(r"(?:^|\.)(model\.layers\.(\d+)\.self_attn\.)([^.]+)$", runtime_name)
    if match is None or (num_hidden_layers is not None and int(match[2]) >= num_hidden_layers):
        return None
    leaf = match[3]
    output = leaf in _OUTPUT_ALIASES
    if output:
        leaf = _OUTPUT_ALIASES[leaf]
    elif leaf in _INPUT_ALIASES:
        leaf = _INPUT_ALIASES[leaf]
    elif leaf not in ("q_a_proj", "o_proj"):
        return None
    name = match[1] + leaf
    if stat_names is not None:
        if name not in stat_names:
            return None
    elif leaf not in _DEFAULT_STATS:
        return None
    mode = "concat" if leaf == "o_proj" and getattr(module, "input_is_parallel", False) else "max"
    return name, output, mode
