#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHON_MULTIPROCESSING_METHOD=spawn
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL=1

INFERENCE_TP_SIZE=4

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
    
    # Evaluate ceval, mmlu, gsm8k
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_PATH,add_bos_token=True,gpu_memory_utilization=0.9,tensor_parallel_size=$INFERENCE_TP_SIZE \
        --tasks ceval-valid \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$RESULT_PATH/ceval_results.json" 2>&1 | tee "$RESULT_PATH/ceval.log"
    
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_PATH,add_bos_token=True,gpu_memory_utilization=0.9,tensor_parallel_size=$INFERENCE_TP_SIZE \
        --tasks mmlu \
        --num_fewshot 4 \
        --batch_size 1 \
        --output_path "$RESULT_PATH/mmlu_results.json" 2>&1 | tee "$RESULT_PATH/mmlu.log"
    
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_PATH,add_bos_token=True,gpu_memory_utilization=0.9,tensor_parallel_size=$INFERENCE_TP_SIZE \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$RESULT_PATH/gsm8k_results.json" 2>&1 | tee "$RESULT_PATH/gsm8k.log"
    
    # Evaluate humaneval
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_PATH,add_bos_token=True,gpu_memory_utilization=0.9,tensor_parallel_size=$INFERENCE_TP_SIZE \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size auto \
        --confirm_run_unsafe_code \
        --output_path "$RESULT_PATH/humaneval_results.json" 2>&1 | tee "$RESULT_PATH/humaneval.log"
    
    echo "Evaluation completed for $MODEL_NAME"
    echo "Results saved to: $RESULT_PATH"
done

echo "All model evaluations finished!"
