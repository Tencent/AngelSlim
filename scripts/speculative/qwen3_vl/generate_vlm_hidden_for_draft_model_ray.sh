#!/bin/bash
set -x

# ============================================================================
# Ray-based VLM hidden state generation script (for vLLM backend)
#
# This script handles:
#   1. Starting the Ray cluster (head node + worker nodes)
#   2. Waiting for all nodes to be ready
#   3. Running the Python generation script on the head node
#   4. Stopping the Ray cluster after completion
# ============================================================================


TARGET_MODEL_PATH=
TRAIN_DATASET_PATH=
TRAIN_HIDDEN_PATH=
MODEL_MAX_LENGTH=8192
# Set defaults
SAMPLE_NUM=50000
CHAT_TEMPLATE_TYPE=
MAX_PIXELS=153664
MIN_PIXELS=3136
TP_SIZE=1
MAX_MODEL_LEN=8192
LIMIT_MM_PER_PROMPT='{"image": 10, "video": 10}'
GPU_MEMORY_UTILIZATION=0.85


CONFIG_DIR=angelslim/compressor/speculative/train/configs
DRAFT_MODEL_CONFIG_PATH="$CONFIG_DIR/qwen3-vl-4b-eagle3-mrope.json"

# ============================================================================
# Ray cluster configuration
# ============================================================================
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
RAY_HEAD_PORT=${MASTER_PORT:-6379}
export OPENBLAS_NUM_THREADS=64
export OMP_NUM_THREADS=64

NUM_CPUS_PER_NODE=${NUM_CPUS_PER_NODE:-64}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
TOTAL_GPUS=$(($NNODES*$NUM_GPUS_PER_NODE))

if [[ "$NODE_RANK" -eq '0' ]]; then
  rm -rf $TRAIN_HIDDEN_PATH/.ready
  rm -rf $TRAIN_HIDDEN_PATH
  mkdir -p $TRAIN_HIDDEN_PATH
  touch "$TRAIN_HIDDEN_PATH/.ready"
else
  sleep 10s
  while [ ! -f "$TRAIN_HIDDEN_PATH/.ready" ]; do
    echo "Waiting for node 0 to create the directory $TRAIN_HIDDEN_PATH..."
    sleep 5
  done
fi

CURR_RUNTIME=$(date +"%Y%m%d_%H%M")
RAY_CHECK_DIR="$TRAIN_HIDDEN_PATH/RAY_CHECK_${CURR_RUNTIME:0:-1}0"
# Pre-create output directory and log file so Ray startup logs can be captured
HIDDEN_LOG_PATH="$TRAIN_HIDDEN_PATH/log_ray_${RANK}.txt"
touch $HIDDEN_LOG_PATH

# ============================================================================
# Export environment variables BEFORE starting Ray.
# Ray's daemon process (raylet) is started in Step 1, and all Ray Worker
# Actors are spawned by this raylet.  Workers inherit the environment that
# existed when raylet was created.  If we export MAX_PIXELS / MIN_PIXELS
# after `ray start`, the workers will NOT see them -- they will get None.
# ============================================================================
export MAX_PIXELS=$MAX_PIXELS
export MIN_PIXELS=$MIN_PIXELS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
SHARED_HF_CACHE="$(dirname $TRAIN_HIDDEN_PATH)/.hf_cache"
export HF_HOME="$SHARED_HF_CACHE"
export HF_DATASETS_CACHE="$SHARED_HF_CACHE/datasets"
mkdir -p "$HF_DATASETS_CACHE"


# ============================================================================
# Step 1: Start Ray cluster
# ============================================================================
# Always stop any existing Ray cluster first to release GPU memory from
# previous runs
if ray status &>/dev/null; then
    echo "[RANK $NODE_RANK] Stopping existing Ray cluster to release GPU memory..." | tee -a $HIDDEN_LOG_PATH
    ray stop --force 2>&1 | tee -a $HIDDEN_LOG_PATH
    sleep 5
fi


if [ "$NODE_RANK" -eq "0" ]; then
    # ---- Head node ----
    rm -rf ${RAY_CHECK_DIR}
    mkdir -p ${RAY_CHECK_DIR}
    echo "[HEAD $NODE_RANK] Starting Ray head node..." | tee -a $HIDDEN_LOG_PATH
    NODE_IP=$MASTER_ADDR
    echo "$NODE_IP" > ${RAY_CHECK_DIR}/head_ip.txt
    echo "[HEAD] IP is $NODE_IP, written to ${RAY_CHECK_DIR}/head_ip.txt" | tee -a $HIDDEN_LOG_PATH
    echo "head" > ${RAY_CHECK_DIR}/node_${NODE_RANK}_$(hostname -I | awk '{print $1}')

    ray start --head \
        --node-ip-address=${NODE_IP} \
        --port=${RAY_HEAD_PORT} \
        --num-cpus=${NUM_CPUS_PER_NODE} \
        --num-gpus=${NUM_GPUS_PER_NODE} 2>&1 | tee -a $HIDDEN_LOG_PATH

    touch ${RAY_CHECK_DIR}/RANK_${NODE_RANK}.ready
    echo "[HEAD] Waiting for all ${NNODES} nodes to become ready..." | tee -a $HIDDEN_LOG_PATH
    while [ $(ls ${RAY_CHECK_DIR}/RANK_*.ready 2>/dev/null | wc -l) -lt $NNODES ]; do
        sleep 1
    done
    echo "[HEAD] All ${NNODES} nodes are ready. Waiting for cluster resources to settle..." | tee -a $HIDDEN_LOG_PATH
    sleep 10
    echo "[HEAD] Cluster status:" | tee -a $HIDDEN_LOG_PATH
    ray status 2>&1 | tee -a $HIDDEN_LOG_PATH
else
    # ---- Worker node ----
    echo "[WORKER $NODE_RANK] Waiting for head IP at ${RAY_CHECK_DIR}/head_ip.txt" | tee -a $HIDDEN_LOG_PATH
    sleep 10
    while [ ! -f ${RAY_CHECK_DIR}/head_ip.txt ]; do
        sleep 5
    done
    sleep 10
    RAY_HEAD_ADDR=$(cat ${RAY_CHECK_DIR}/head_ip.txt)
    echo "[WORKER $NODE_RANK] Read RAY_HEAD_ADDR=${RAY_HEAD_ADDR}, connecting to Ray head..." | tee -a $HIDDEN_LOG_PATH
    ray start --address=${RAY_HEAD_ADDR}:${RAY_HEAD_PORT} \
        --num-gpus=${NUM_GPUS_PER_NODE} \
        --num-cpus=${NUM_CPUS_PER_NODE} 2>&1 | tee -a $HIDDEN_LOG_PATH
    touch ${RAY_CHECK_DIR}/RANK_${NODE_RANK}.ready
    echo "worker" > ${RAY_CHECK_DIR}/node_${NODE_RANK}_$(hostname -I | awk '{print $1}')
    echo "[WORKER $NODE_RANK] Joined cluster." | tee -a $HIDDEN_LOG_PATH
fi


# ============================================================================
# Step 2: Run the generation task (head node only)
# ============================================================================
if [ "$NODE_RANK" -eq "0" ]; then
    # Build optional arguments
    OPTIONAL_ARGS=""
    if [ -n "$CHAT_TEMPLATE_TYPE" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --chat_template_type $CHAT_TEMPLATE_TYPE"
    fi
    if [ -n "$TOTAL_GPUS" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --total_gpus $TOTAL_GPUS"
    fi
    if [ -n "$LIMIT_MM_PER_PROMPT" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --limit_mm_per_prompt '$LIMIT_MM_PER_PROMPT'"
    fi
    if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --gpu_memory_utilization $GPU_MEMORY_UTILIZATION"
    fi

    # Build the full command, log it, then execute
    python tools/ray_generate_hidden_for_draft_model.py \
        --modal_type VLM \
        --dataset_path $TRAIN_DATASET_PATH \
        --target_model_name_or_path $TARGET_MODEL_PATH \
        --draft_model_config_path $DRAFT_MODEL_CONFIG_PATH \
        --target_backend vllm \
        --torch_dtype bfloat16 \
        --model_max_length $MODEL_MAX_LENGTH \
        --outdir $TRAIN_HIDDEN_PATH \
        --num_proc 16 \
        --sample_num $SAMPLE_NUM \
        --tensor_parallel_size $TP_SIZE \
        --max_model_len $MAX_MODEL_LEN \
        $OPTIONAL_ARGS
fi
