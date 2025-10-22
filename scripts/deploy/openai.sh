MODEL_PATH=$1
PORT=8080
curl http://0.0.0.0:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "一种零件的内径尺寸在图纸上是30±0.02(单位：毫米） 表示这种零件的标准尺寸是30毫米．加工要求最大不超过标准尺寸__毫米 最小不低于标准尺寸__毫米。"
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05
    }'