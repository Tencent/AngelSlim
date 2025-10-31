export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH=$1  # your/path/to/model
PORT=8080
INFERENCE_TP_SIZE=4

python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model-path $MODEL_PATH \
    --tp $INFERENCE_TP_SIZE \
    --mem-fraction-static 0.9 \
    --trust-remote-code