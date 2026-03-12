cd /cfs_cloud_code/jiebinzhang/AngelSlim

PYTHONPATH=/cfs_cloud_code/jiebinzhang/AngelSlim:/cfs_cloud_code/jiebinzhang/SpecForge \
python -m pdb tools/debug_dflash_compare.py \
    --jsonl /cfs_cloud_code/jiebinzhang/SpecForge/cache/dataset/regen_qwen3_4b.jsonl \
    --target_model /apdcephfs_gy5_303770945/share_303770945/jiebin/hf_models/Qwen/Qwen3-4B \
    --draft_config configs/qwen3_dflash.json \
    --sample_idx 0 \
    --seed 42