# KV-cache 校准与 scale 替换

两个独立入口，均在 AngelSlim 仓库根目录执行，vLLM 无需打补丁：

| 入口 | 作用 | 阶段 |
| --- | --- | --- |
| `python3 -m angelslim.compressor.quant.vllm_calib.calibrate --kv-only` | 只挂 Attention KV hook 的 vLLM 校准（不统计 Linear / MoE），可选搜索每层或每头的 KV scale 倍率 | 校准 |
| `python3 -m angelslim.compressor.quant.ckpt_convert.kv_scales` | 用校准产出的 JSON 改写已量化模型的 `kv_cache_scales.safetensors`，并更新 `config.json` 中的 `attn_quant_config` | 量化后 |

## 1. KV-only 校准

```bash
python3 -m angelslim.compressor.quant.vllm_calib.calibrate --kv-only \
    -c configs/Hy3/ptq/fp8/Hy3_kvcache_calibrate.yaml
```

YAML 键名与命令行参数的 `dest` 一致（如 `kv_granularity: per-head`、`search_kv_scale: true`）。
一键脚本：`scripts/ptq/run_kvcache_calibrate_for_Hy3.sh`。

产出（写入 `output_dir`）：

| 文件 | 条件 |
| --- | --- |
| `activation_stats.json` | per-tensor |
| `activation_stats_per_head.json` | per-head |
| `kv_scale_multipliers[_per_head].json`、`kv_cache_tuned_scales[_per_head].json` | `search_kv_scale: true` |

## 2. 替换 KV scale

```bash
# per-head（默认）
python3 -m angelslim.compressor.quant.ckpt_convert.kv_scales \
    --granularity per-head \
    --json   /path/to/kv_cache_tuned_scales_per_head.json \
    --src    /path/to/model_dir/kv_cache_scales.safetensors \
    --output /path/to/output_dir/kv_cache_scales.safetensors

# per-tensor
python3 -m angelslim.compressor.quant.ckpt_convert.kv_scales \
    --granularity per-tensor \
    --json   /path/to/kv_cache_tuned_scales.json \
    --src    /path/to/model_dir/kv_cache_scales.safetensors \
    --output /path/to/output_dir/kv_cache_scales.safetensors
```

省略 `--output` 时原地覆盖 `--src`，并自动生成 `.bak` 备份。
