#!/bin/bash

usage() {
    cat << EOF
Usage: $0 <model_path1> [OPTIONS]

Options:
  --model-path                   Model path (Need)
  --port PORT                    Servive port (default: 8080)
  -d, --devices DEVICES          CUDA devices to use (default: 0,1,2,3)
  -t, --tensor-parallel SIZE     Tensor parallel size (default: 4)
  -p, --pipeline-parallel-size   Pipline parallel size (default: 1)
  -g, --gpu-memory UTILIZATION   GPU memory utilization (default: 0.9)
  --max-model-len                Max model len (default: 4096)
  -h, --help                     Show this help message

Examples:
  bash $0 --model-path /path/to/model -d 0,1 -t 2 --gpu-memory-utilization 0.8
EOF
}

CUDA_VISIBLE_DEVICES="0,1,2,3"
INFERENCE_TP_SIZE=4
PIPELINE_PARALLEL_SIZE=1
PORT=8080
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--devices)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        -t|--tensor-parallel)
            INFERENCE_TP_SIZE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -g|--gpu-memory)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --pipeline-parallel-size)
            PIPELINE_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
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

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --pipeline_parallel_size ${PIPELINE_PARALLEL_SIZE} \
    --tensor-parallel-size ${INFERENCE_TP_SIZE} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN}
