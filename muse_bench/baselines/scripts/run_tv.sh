CORPUS='news'

FORGET="../../../harmfulness/data/unlearning_data_AdvBench_3B.json"

TARGET_DIR='meta-llama/Llama-3.2-3B-Instruct'
LLAMA_DIR='meta-llama/Llama-3.2-3B-Instruct'

MAX_LEN=1024
PER_DEVICE_BATCH_SIZE=8
FT_EPOCHS=5
FT_LR='1e-5'


python ../unlearn.py \
    --algo 'tv' \
    --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
    --data_file $FORGET \
    --out_dir "tv" \
    --max_len $MAX_LEN --epochs $FT_EPOCHS --lr $FT_LR \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --alpha 2 \

