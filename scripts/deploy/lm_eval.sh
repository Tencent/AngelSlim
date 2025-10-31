#!/bin/bash

usage() {
    cat << EOF
Usage: $0 [OPTIONS] <model_path1> <model_path2> ...

Options:
  -d, --devices DEVICES          CUDA devices to use (default: 0,1,2,3)
  -t, --tensor-parallel SIZE     Tensor parallel size (default: 4)
  -g, --gpu-memory UTILIZATION   GPU memory utilization (default: 0.9)
  -r, --result-dir DIR           Base result directory (default: ./results)
  -b, --batch-size SIZE          Batch size for auto tasks (default: auto)
  --tasks TASK1,TASK2,...        Comma-separated list of tasks to evaluate (default: ceval-valid,mmlu,gsm8k,humaneval)
  -n, --num-fewshot NUM          Number of few-shot examples (default: 0)
  -h, --help                     Show this help message

Examples:
  bash $0 -d 0,1 -t 2 --gpu-memory 0.8 /path/to/model1 /path/to/model2
  bash $0 --tasks ceval-valid,mmlu,gsm8k,humaneval /path/to/model1
EOF
}

CUDA_VISIBLE_DEVICES="0,1,2,3"
INFERENCE_TP_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
RESULT_BASE_DIR="./results"
BATCH_SIZE="auto"
TASKS=("ceval-valid" "mmlu" "gsm8k" "humaneval")
NUM_FEWSHOT=0

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--devices)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        -t|--tensor-parallel)
            INFERENCE_TP_SIZE="$2"
            shift 2
            ;;
        -g|--gpu-memory)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        -r|--result-dir)
            RESULT_BASE_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tasks)
            IFS=',' read -ra TASKS <<< "$2"
            shift 2
            ;;
        -n|--num-fewshot)
            NUM_FEWSHOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*|--*)
            echo "Error: Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

# Check if model paths are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path1> <model_path2> ..."
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHON_MULTIPROCESSING_METHOD=spawn
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL=1

echo "======================================================"
echo "           Model Evaluation Configuration"
echo "======================================================"
echo "CUDA Visible Devices:      $CUDA_VISIBLE_DEVICES"
echo "Tensor Parallel Size:      $INFERENCE_TP_SIZE"
echo "GPU Memory Utilization:    $GPU_MEMORY_UTILIZATION"
echo "Result Base Directory:     $RESULT_BASE_DIR"
echo "Batch Size:                $BATCH_SIZE"
echo "Number of Few-shot:        $NUM_FEWSHOT"
echo "Tasks to Evaluate:         ${TASKS[*]}"
echo "Number of Models:          $#"
echo "Model Paths:"
for model_path in "$@"; do
    echo "  - $model_path"
done
echo "======================================================"
echo

# Iterate over all provided model paths
for MODEL_PATH in "$@"; do
    # Extract model name from path (last directory name)
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "======================================================"
    echo "Evaluating model: $MODEL_NAME"
    echo "Model path: $MODEL_PATH"
    
    # Create dedicated result directory for the model
    RESULT_PATH="$RESULT_BASE_DIR/$MODEL_NAME"
    mkdir -p "$RESULT_PATH"
    
    for TASK in "${TASKS[@]}"; do
        echo "=============================================="
        echo "Evaluating task: $TASK"
        echo "Number of few-shot: $NUM_FEWSHOT"
        echo "=============================================="
        if [[ "$TASK" == *"humaneval"* ]]; then
            # Evaluate humaneval
            lm_eval --model vllm \
                --model_args pretrained=$MODEL_PATH,add_bos_token=True,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,tensor_parallel_size=$INFERENCE_TP_SIZE \
                --tasks $TASK \
                --num_fewshot $NUM_FEWSHOT \
                --batch_size $BATCH_SIZE \
                --confirm_run_unsafe_code \
                --output_path "$RESULT_PATH/$TASK.json" 2>&1 | tee "$RESULT_PATH/$TASK.log"
        else
            lm_eval --model vllm \
                --model_args pretrained=$MODEL_PATH,add_bos_token=True,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,tensor_parallel_size=$INFERENCE_TP_SIZE \
                --tasks $TASK \
                --num_fewshot $NUM_FEWSHOT \
                --batch_size $BATCH_SIZE \
                --output_path "$RESULT_PATH/$TASK.json" 2>&1 | tee "$RESULT_PATH/$TASK.log"
        fi
    done

    echo "Evaluation completed for $MODEL_NAME"
    echo "Results saved to: $RESULT_PATH"
done

echo "All model evaluations finished!"
