import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

json_path = "/apdcephfs_gy7/share_303171455/linchuanxie/code_0403_kvcache_tensor/kv_cache_tuned_scales.json"  # noqa: E501
st_path = "/apdcephfs_gy7/share_303171455/linchuanxie/code_0403_fp8/kv_cache_scales.safetensors"
out_path = "/apdcephfs_gy7/share_303171455/linchuanxie/code_0403_fp8/kv_cache_scales.safetensors"

# 读取 JSON
with open(json_path) as f:
    json_scales = json.load(f)

# 读取 safetensors
tensors = {}
with safe_open(st_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key).clone()

# 替换对应层的值
replaced = 0
skipped = 0
for key, val in json_scales.items():
    if key in tensors:
        tensors[key] = torch.tensor([val], dtype=tensors[key].dtype)
        replaced += 1
    else:
        print(f"[SKIP] key not found in safetensors: {key}")
        skipped += 1

print(f"替换: {replaced} 个 key，跳过: {skipped} 个 key")

# 保存
save_file(tensors, out_path)
print(f"已保存到: {out_path}")

# 在 out_path 同目录下的 config.json 中添加 attn_quant_config
config_path = os.path.join(os.path.dirname(out_path), "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

config["attn_quant_config"] = {
    "kv_cache_quant": {
        "dtype": "fp8_e4m3",
        "k_quant": {"scheme": "static", "granularity": "per_tensor"},
        "v_quant": {"scheme": "static", "granularity": "per_tensor"},
    },
    "q_quant": {"dtype": "fp8_e4m3", "scheme": "dynamic", "granularity": "per_token_per_head"},
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"已更新 config.json: {config_path}")
