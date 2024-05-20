PYTHON_PATH=$1
MODEL_PATH=$2
TEMPLATE=$3
LORA_PATH=$4
OUTPUT_PATH=$5
MAX_LEN=$6
TEMPERATURE=$7
EVAL_BATCH_SIZE=$8

deepspeed --include localhost:4,5,6,7 --master_port=9901 $PYTHON_PATH \
    --model_name_or_path $MODEL_PATH \
    --stage sft \
    --do_predict \
    --dataset custom_pred_dataset \
    --template $TEMPLATE \
    --finetuning_type lora \
    --checkpoint_dir $LORA_PATH \
    --output_dir $OUTPUT_PATH \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --max_len $MAX_LEN \
    --cutoff_len $MAX_LEN \
    --temperature $TEMPERATURE \
    --max_samples 10000 \
    --predict_with_generate