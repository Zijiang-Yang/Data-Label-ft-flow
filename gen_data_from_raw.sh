#!/bin/bash

#repo本地地址
REPO_DIR_PATH="/sdata/zhijiang/ai-project/data_label_ft_flow"
#项目名称
PROJECT_NAME="sales_question_classify"


python $REPO_DIR_PATH/data_process/generate_from_raw.py \
    --repo_dir_path $REPO_DIR_PATH \
    --project_name $PROJECT_NAME 