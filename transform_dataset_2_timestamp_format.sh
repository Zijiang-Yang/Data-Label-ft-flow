#!/bin/bash

#repo本地地址，需要修改
REPO_DIR_PATH=/sdata/zhijiang/ai-project/data_label_ft_flow
#需要转化格式的文件地址，需要修改
INPUT_FILE_PATH=/data/disk4/home/yaosong/QA_LLM/cutomer_service/model_data/pred.json
#项目名称，需要修改
PROJECT_NAME=customer_service_gen_answer
#train或者pred，根据需要修改
DATA_TYPE=pred


python $REPO_DIR_PATH/data_process/transform_2_timestamp_format.py \
    --repo_dir_path $REPO_DIR_PATH \
    --input_file_path $INPUT_FILE_PATH \
    --project_name $PROJECT_NAME \
    --data_type $DATA_TYPE