# 负责处理从数据到标注到微调的全流程


## 安装依赖
    git clone https://git.woniubaoxian.com/ai-project/data_label_ft_flow.git
    proxychains git clone https://github.com/hiyouga/LLaMA-Factory.git
    proxychains conda create -n llm python=3.10
    conda activate llm
    cd data_label_ft_flow
    proxychains pip install -r requirements.txt


## 开始
    #一键微调+推理
    sh run_pipeline.sh

    #已有数据增加时间戳并上传云端
    sh transform_dataset_2_timestamp_format.sh

    #处理原始数据，生成带时间戳的数据并上传云端
    sh gen_data_from_raw.sh


## 需要修改参数
- run_pipeline.sh
  - REPO_DIR_PATH
  - LLAMA_FACTORY_DIR_PATH
  - PROJECT_NAME
- transform_dataset_2_timestamp_format
  - REPO_DIR_PATH
  - INPUT_FILE_PATH
  - PROJECT_NAME
  - DATA_TYPE
- gen_data_from_raw
  - REPO_DIR_PATH
  - PROJECT_NAME