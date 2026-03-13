#!/bin/bash

PROMPT="一种零件的内径尺寸在图纸上是30±0.02(单位：毫米） 表示这种零件的标准尺寸是30毫米．加工要求最大不超过标准尺寸__毫米 最小不低于标准尺寸__毫米。"
PORT=8080
MAX_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
REPETITION_PENALTY=1.05
SYSTEM_PROMPT="You are a helpful assistant."

usage() {
    cat << EOF
Usage: $0 -m /path/to/model [OPTIONS]

Options:
  -m,  --model                   Model path(Needed)
  -p, --prompt PROMPT            Prompt text to send to the model
  --port PORT                    API server port (default: 8080)
  --max-tokens TOKENS            Maximum tokens to generate (default: 2048)
  --temperature TEMP             Sampling temperature (default: 0.7)
  --top-p TOP_P                  Top-p sampling parameter (default: 0.8)
  --top-k TOP_K                  Top-k sampling parameter (default: 20)
  --repetition-penalty PENALTY   Repetition penalty (default: 1.05)
  --system-prompt PROMPT         System prompt (default: "You are a helpful assistant.")
  -h, --help                     Show this help message

Examples:
  bash $0 -m /path/to/model --port 8000 -p "你的提示词" 
EOF
}

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --repetition-penalty)
            REPETITION_PENALTY="$2"
            shift 2
            ;;
        --system-prompt)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
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

curl http://0.0.0.0:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [
            {
                "role": "system",
                "content": "'"$SYSTEM_PROMPT"'"
            },
            {
                "role": "user",
                "content": "'"$PROMPT"'"
            }
        ],
        "max_tokens": '"$MAX_TOKENS"',
        "temperature": '"$TEMPERATURE"',
        "top_p": '"$TOP_P"',
        "top_k": '"$TOP_K"',
        "repetition_penalty": '"$REPETITION_PENALTY"'
    }'