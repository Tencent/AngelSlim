export MODEL_NAME=
export MODEL_LOCAL_PATH=
export GPU_NUM=8

# Start vLLM server
for i in $(seq 0 $((GPU_NUM-1))); do
    cmd="CUDA_VISIBLE_DEVICES=${i} nohup vllm serve ${MODEL_LOCAL_PATH} --port $((6000 + i)) 2>&1 > ./logs/$(echo ${MODEL_NAME} | sed 's/\//-/g')_${i}.log &"
    echo $cmd
    eval $cmd
done