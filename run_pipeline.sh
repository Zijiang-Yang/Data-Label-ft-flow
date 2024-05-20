#!/bin/bash

#repo本地地址，需要修改
REPO_DIR_PATH=/sdata/zhijiang/ai-project/data_label_ft_flow
#LLaMA Factory本地地址，需要修改
LLAMA_FACTORY_DIR_PATH=/sdata/zhijiang/llmfile/LLaMA-Factory2
#项目名称，需要修改，跟transform_dataset_2_timestamp_format中的PROJECT_NAME需保持一致
PROJECT_NAME=customer_service_gen_answer
#model本地地址，根据使用模型可以修改
#LOCAL_MODEL_DIR_PATH=/data/disk4/tmp/Qwen-72B-Chat-Int8
LOCAL_MODEL_DIR_PATH=/data/disk4/home/zhijiang/tmp/Qwen-7B-Chat-Int8
#num_epoch, default 3
NUM_EPOCH=6
#per_device_train_batch_size, default 1
TRAIN_BATCH=1
#temperature, default 0.01
TEMPERATURE=0.01
#is_fp16, store true, default using bf16
is_fp16=false
#pred文件中如果包含非instruction, input, output的信息并想整理在输出时，需要在此处写上key的名称
#如果没有则为none, 如果有多个，每个key之间需要用英文逗号分隔
ORGANIZE_KEYS=none
#enable_zero3, store true, default use zero2
#pred_only，如果只做pred，需要传入pred_only,否则不要传入

export CUDA_VISIBLE_DEVICES=2,1,0,3,6,5,4,7

python $REPO_DIR_PATH/src/pipeline.py \
    --repo_dir_path $REPO_DIR_PATH \
    --llama_factory_dir_path  $LLAMA_FACTORY_DIR_PATH \
    --project_name $PROJECT_NAME \
    --model_dir_path $LOCAL_MODEL_DIR_PATH \
    --max_length 1024 \
    --per_device_train_batch_size $TRAIN_BATCH \
    --num_epoch $NUM_EPOCH \
    --temperature $TEMPERATURE \
    --pred_dataset_organize_keys $ORGANIZE_KEYS \
    # --pred_only \
    # --enable_zero3 \
