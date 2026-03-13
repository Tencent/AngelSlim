#!/bin/bash

usage() {
    cat << EOF
Usage: $0 <model_path1> [OPTIONS]

Options:
  --model-path                   Model path (Need)
  --port PORT                    Servive port (default: 8080)
  -d, --devices DEVICES          CUDA devices to use (default: 0,1,2,3)
  -t, --tensor-parallel SIZE     Tensor parallel size (default: 4)
  -g, --gpu-memory UTILIZATION   GPU memory utilization (default: 0.9)
  -h, --help                     Show this help message

Examples:
  bash $0 --model-path /path/to/model -d 0,1 -t 2 --gpu-memory-utilization 0.8
EOF
}

CUDA_VISIBLE_DEVICES="0,1,2,3"
INFERENCE_TP_SIZE=4
PORT=8080
GPU_MEMORY_UTILIZATION=0.9

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

python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model-path $MODEL_PATH \
    --tp $INFERENCE_TP_SIZE \
    --mem-fraction-static $GPU_MEMORY_UTILIZATION \
    --trust-remote-code