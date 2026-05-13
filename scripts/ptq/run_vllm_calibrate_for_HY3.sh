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

export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

run_name=log_name
model_path=/path/to/model
ptq_data_path=/path/to/dataset
output_dir=/path/to/output

tp_size=16
batch_size=4
num_samples=512
max_length=16384

# Boolean flags (non-empty to enable, empty to disable)
skip_weight_loading=""  # set to "--skip-weight-loading" to enable debug mode
verbose=""              # set to "--verbose" to enable

# KV-cache granularity: none | per-tensor | per-head
kv_granularity="per-head"

# KV cache scale search settings
search_kv_scale="--search-kv-scale"   # set to "--search-kv-scale" to enable scale search
search_kv_num_samples=64               # number of samples used for the search
search_kv_min_multiplier=0.8           # lower bound of multiplier search range
search_kv_max_multiplier=16.0          # upper bound of multiplier search range
search_kv_num_steps=50                 # number of log-uniform grid points

python3 tools/run_vllm_calibrate.py \
    --model-path $model_path \
    --ptq-data-path $ptq_data_path \
    --output-dir $output_dir \
    --tp-size $tp_size \
    --batch-size $batch_size \
    --num-samples $num_samples \
    --max-length $max_length \
    --kv-granularity $kv_granularity \
    $skip_weight_loading \
    $verbose \
    $search_kv_scale \
    --search-kv-num-samples $search_kv_num_samples \
    --search-kv-min-multiplier $search_kv_min_multiplier \
    --search-kv-max-multiplier $search_kv_max_multiplier \
    --search-kv-num-steps $search_kv_num_steps \
    2>&1 | tee logs/${run_name}.log
