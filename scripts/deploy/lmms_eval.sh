#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHON_MULTIPROCESSING_METHOD=spawn
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="~/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG

INFERENCE_TP_SIZE=4
BATCH_SIZE=16

TASKS=("mmmu_val" "docvqa_val" "chartqa")

# Check if model paths are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path1> <model_path2> ..."
    exit 1
fi

# Iterate over all provided model paths
for MODEL_PATH in "$@"; do
    # Extract model name from path (last directory name)
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "======================================================"
    echo "Evaluating model: $MODEL_NAME"
    echo "Model path: $MODEL_PATH"
    echo "======================================================"

    # Create dedicated result directory for the model
    RESULT_PATH="./results/$MODEL_NAME"
    mkdir -p "$RESULT_PATH"

    for TASK in "${TASKS[@]}"; do
        echo "=============================================="
        echo "Evaluating model: $MODEL_NAME"
        echo "Evaluating task: $TASK"
        echo "=============================================="
        
        python3 -m lmms_eval \
            --model vllm \
            --model_args model_version=$MODEL_PATH,gpu_memory_utilization=0.9,tensor_parallel_size=$INFERENCE_TP_SIZE \
            --tasks $TASK \
            --batch_size $BATCH_SIZE \
            --log_samples \
            --log_samples_suffix vllm \
            --output_path "$RESULT_PATH/$TASK.json" 2>&1 | tee "$RESULT_PATH/$TASK.log"
        
        echo "Evaluation completed for $TASK"
        echo "Results saved to: $RESULT_PATH"
    done

    echo "Evaluation completed for $MODEL_NAME"
done

echo "All model evaluations finished!"
