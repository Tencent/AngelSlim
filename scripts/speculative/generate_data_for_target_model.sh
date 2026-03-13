export DATA_NAME_OR_PATH=
export OUTPUT_DIR=
export DATA_FORMAT=sharegpt
export DATA_SHARD_SIZE=50000
export BASE_PORT=6000
export NUM_THREADS=256


# Generate data
python3 ./tools/generate_data_for_target_model.py \
    --data_name_or_path $DATA_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --data_format $DATA_FORMAT \
    --data_shard_size $DATA_SHARD_SIZE \
    --base_port $BASE_PORT \
    --num_threads $NUM_THREADS
