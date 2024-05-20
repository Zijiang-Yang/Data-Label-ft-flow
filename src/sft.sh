MASTER_PORT=$(shuf -n 1 -i 8000-65535)
PYTHON_PATH=$1
MODEL_PATH=$2
TEMPLATE=$3
LORA_TARGET=$4
LORA_PATH=$5
DS_CONFIG_PATH=$6
MAX_LEN=$7
NUM_EPOCH=$8
TRAIN_BATCH=$9

deepspeed --include localhost:4,5,6,7 --master_port $MASTER_PORT $PYTHON_PATH \
    --stage sft \
    --model_name_or_path $MODEL_PATH\
    --do_train \
    --dataset custom_train_dataset\
    --template $TEMPLATE \
    --flash_attn true \
    --finetuning_type lora \
    --no_resume_lora_training \
    --lora_target $LORA_TARGET \
    --output_dir $LORA_PATH \
    --per_device_train_batch_size $TRAIN_BATCH \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 4 \
    --max_length $MAX_LEN \
    --cutoff_len $MAX_LEN \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --warmup_steps 100 \
    --learning_rate 3e-4 \
    --max_grad_norm 0.5 \
    --num_train_epochs $NUM_EPOCH \
    --val_size 0.1 \
    --plot_loss \
    --overwrite_output_dir \
    --bf16 \
    --deepspeed $DS_CONFIG_PATH