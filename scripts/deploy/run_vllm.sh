export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH=$1  # your/path/to/model
PORT=8080
INFERENCE_TP_SIZE=4

python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --pipeline_parallel_size 1 \
    --tensor-parallel-size ${INFERENCE_TP_SIZE} \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
