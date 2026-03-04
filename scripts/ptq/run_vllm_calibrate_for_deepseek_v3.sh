# Allow function serialization for apply_model in vLLM v1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Enable MoE expert statistics collection
export VLLM_MOE_COLLECT_STATS=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1
# Disable verbose MoE stats logging
export VLLM_MOE_COLLECT_STATS_VERBOSE=0
# Enable per-expert statistics collection
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1


run_name=
model_path=
ptq_data_path=
output_dir=
tp_size=32
batch_size=128
num_samples=512
max_length=16384

# Boolean flags (non-empty to enable, empty to disable)
skip_weight_loading=""  # set to "" to disable
verbose=""              # set to "" to disable

python3 tools/run_vllm_calibrate.py \
    --model-path $model_path \
    --ptq-data-path $ptq_data_path \
    --output-dir $output_dir \
    --tp-size $tp_size \
    --batch-size $batch_size \
    --num-samples $num_samples \
    --max-length $max_length \
    $skip_weight_loading \
    $verbose \
    2>&1 | tee logs/${run_name}.log

