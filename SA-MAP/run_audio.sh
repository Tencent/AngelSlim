#!/usr/bin/env bash
# Modified from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/run_audio.sh

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    # Kill all background processes
    pkill -P $$  # Kill all child processes of the current script
    exit 0
}

# Set trap to catch SIGINT (Ctrl+C) and SIGTERM signals
trap cleanup SIGINT SIGTERM

# Help function
print_help() {
    echo "Usage: bash $0 [options] --model MODEL_NAME --data DATASET_NAME"
    echo
    echo "Required parameters:"
    echo "  --model MODEL_NAME       Model name"
    echo "  --model-path MODEL_PATH  Model PATH"
    echo "  --data DATASET_NAME      Dataset name"
    echo "  --work-dir DIR          Working directory (default: eval_result)"
    echo "Optional parameters:"
    echo "  --eval-method METHOD    Evaluation method (default: default)"
    echo "  --eval-file FILE        Path to evaluation result file (default: auto)"
    echo
    echo "Control parameters:"
    echo "  --force-reinfer         Force re-inference"
    echo "  --reeval                Re-evaluate"
    echo "  --skip-eval             Skip evaluation"
    echo "  --debug                 Debug mode"
    echo "  --long-debug            Debug mode"
    echo "  --subset SUBSET_NAME     Subset name"
}

EVAL_METHOD="default"
EVAL_FILE="auto"
MODEL=""
MODEL_PATH=""
DATA=""
DEFAULT_WORK_DIR="eval_result"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_help
            exit 0
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data)
            DATA="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --eval-method)
            EVAL_METHOD="$2"
            shift 2
            ;;
        --eval-file)
            EVAL_FILE="$2"
            shift 2
            ;;
        --force-reinfer)
            FORCE_REINFER="true"
            shift
            ;;
        --reeval)
            REEVAL="true"
            shift
            ;;
        --skip-eval)
            SKIP_EVAL="true"
            shift
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        --long-debug)
            LONG_DEBUG="true"
            shift
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            print_help
            exit 1
            ;;
    esac
done

# Set default work-dir
if [ -z "$WORK_DIR" ]; then
    WORK_DIR="$DEFAULT_WORK_DIR"
fi

# Ensure work-dir exists
if [ ! -d "$WORK_DIR" ]; then
    mkdir -p "$WORK_DIR"
    echo "Created work directory: $WORK_DIR"
fi

# GPU groups for different configurations

# Get available GPU count
get_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l
    else
        echo "0"
    fi
}

# Dynamically construct GPU groups
if [ -n "$DEBUG" ]; then
    TOTAL_GPUS=1
elif [ -n "$LONG_DEBUG" ]; then
    TOTAL_GPUS=1
else
    TOTAL_GPUS=$(get_gpu_count)
fi
echo "Detected $TOTAL_GPUS GPUs"

# Construct single GPU groups
GPU_1_GROUPS=()
for ((i=0; i<TOTAL_GPUS; i++)); do
    GPU_1_GROUPS+=($i)
done

# Construct 4-GPU groups
declare -A GPU_4_GROUPS
group_idx=0
for ((i=0; i<TOTAL_GPUS; i+=4)); do
    if ((i+3 < TOTAL_GPUS)); then
        GPU_4_GROUPS[$group_idx]="${i} $((i+1)) $((i+2)) $((i+3))"
        ((group_idx++))
    fi
done

# If no GPU is detected, show warning
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "No GPU devices detected, can only run evaluation"
fi

# If GPU count is less than 4 but trying to use StepAudio model, show warning
if [ "$TOTAL_GPUS" -lt 4 ] && [[ " ${models[@]} " =~ "StepAudio" ]]; then
    echo "Warning: StepAudio requires at least 4x80G GPUs, but only $TOTAL_GPUS GPUs available"
    exit 1
fi

# Split MODEL string into array by space
IFS=' ' read -r -a models <<< "$MODEL"

# If reeval is specified, no GPU is needed, single process inference is sufficient
if [ -n "$REEVAL" ]; then
    echo "> $WORK_DIR"
    for model in "${models[@]}"; do
        echo "Running reeval for model: $model"
        CMD="python run_audio.py \
        --model $model \
        --model-path $MODEL_PATH \
        --data $DATA \
        --work-dir $WORK_DIR \
        --reeval"

        [ "$EVAL_FILE" != "auto" ] && CMD="$CMD --eval-file $EVAL_FILE"
        [ "$EVAL_METHOD" != "default" ] && CMD="$CMD --eval-method $EVAL_METHOD"
        [ -n "$SUBSET" ] && CMD="$CMD --subset $SUBSET"
        echo "Executing command: $CMD"
        eval "$CMD"
    done
    exit 0
fi

# Loop through each model and dataset
for model in "${models[@]}"; do
    if [[ $model == "StepAudio" ]]; then
        NUM_GPUS=4
        GPU_GROUPS=("${GPU_4_GROUPS[@]}")
    else
        NUM_GPUS=1
        GPU_GROUPS=("${GPU_1_GROUPS[@]}")
    fi

    for i in "${!GPU_GROUPS[@]}"; do
        echo "> $WORK_DIR"
    done

    for i in "${!GPU_GROUPS[@]}"; do
        rank=$i
        # Convert space-separated GPU list to comma-separated string
        if [[ $NUM_GPUS == 4 ]]; then
                CUDA_DEVICES=$(echo ${GPU_GROUPS[$i]} | tr ' ' ',')
        else
                CUDA_DEVICES=${GPU_GROUPS[$i]}
        fi
        WORLD_SIZE=${#GPU_GROUPS[@]}

        echo "Running inference for model: $model and dataset: $data on GPU group: $CUDA_DEVICES"
        CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python run_audio.py \
        --model $model \
        --model-path $MODEL_PATH \
        --data $DATA \
        --work-dir $WORK_DIR \
        --rank $rank \
        --world-size $WORLD_SIZE"

        # Add optional parameters
        [ "$EVAL_METHOD" != "default" ] && CMD="$CMD --eval-method $EVAL_METHOD"
        [ -n "$FORCE_REINFER" ] && CMD="$CMD --force-reinfer"
        [ -n "$SKIP_EVAL" ] && CMD="$CMD --skip-eval"
        [ -n "$DEBUG" ] && CMD="$CMD --debug"
        [ -n "$SUBSET" ] && CMD="$CMD --subset $SUBSET"

        echo "Executing command: $CMD"
        if [ -n "$DEBUG" ]; then
            eval "$CMD"
        elif [ -n "$LONG_DEBUG" ]; then
            eval "$CMD"
        else
            eval "$CMD &"
        fi
    done
    wait
    echo "Inference for model: $model completed."

done
