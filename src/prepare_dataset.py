import os
import json
import oss2
from setup_bucket import bucket
from validation import validate_train, validate_pred


class TimeStampException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def get_timestamp_from_filename(filename):
    timestamp_str = filename.split('_')[0]
    try:
        timestamp = int(timestamp_str)
        return timestamp
    except:
        raise TimeStampException(f"无法解析时间戳：{timestamp_str}")


def get_bucket_latest(project_name, data_type):
    dir_path = os.path.join("data_label_ft_flow", project_name, "data", data_type) + "/"
    file_paths = []
    for obj in oss2.ObjectIterator(bucket, prefix = dir_path, delimiter = '/'):
        file_path = obj.key.split("/")[-1]
        file_paths.append(file_path)
    bucket_latest_file = max(file_paths, key=lambda file: get_timestamp_from_filename(file))
    return bucket_latest_file


def get_latest_dataset_path(local_data_dir_path, project_name, data_type):
    if not os.path.exists(local_data_dir_path):
        os.makedirs(local_data_dir_path)
    file_paths = [file_name for file_name in os.listdir(local_data_dir_path)]
    local_latest_file = max(file_paths, key = lambda file_path: get_timestamp_from_filename(file_path)) if len(file_paths) > 0 else None
    bucket_latest_file = get_bucket_latest(project_name, data_type)
    if local_latest_file is None or get_timestamp_from_filename(bucket_latest_file) > get_timestamp_from_filename(local_latest_file):
        download_bucket_file_path = os.path.join("data_label_ft_flow", project_name, "data", data_type, bucket_latest_file)
        download_local_file_path = os.path.join(local_data_dir_path, bucket_latest_file)
        bucket.get_object_to_file(download_bucket_file_path, download_local_file_path)
        local_latest_file = download_local_file_path
    return os.path.join(local_data_dir_path, local_latest_file)


def write_data_info(llama_factory_dir_path, train_data_path, pred_data_path):
    data_info_dir_path = os.path.join(llama_factory_dir_path, "data")
    data_info_file_path = os.path.join(data_info_dir_path, "dataset_info.json")
    with open(data_info_file_path, "r", encoding="utf-8") as file:
        data_info = json.load(file)  #data_info_file_path=llama_factory_dir_path/data/dataset_info.json
    
    #write_backup
    data_info_backup_path = os.path.join(data_info_dir_path, "data_info_backup.json")
    with open(data_info_backup_path, "w", newline="", encoding="utf-8") as file:
        json.dump(data_info, file, ensure_ascii=False, indent=4)

    #add_dataset
    data_info["custom_train_dataset"] = {
        "file_name": train_data_path,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": ""
        }}
    data_info["custom_pred_dataset"] = {
        "file_name": pred_data_path,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": ""
        }}
    with open(data_info_file_path, "w", newline="", encoding="utf-8") as file:
        json.dump(data_info, file, ensure_ascii=False, indent=4)


def prepare_dataset(args):
    latest_train_path = get_latest_dataset_path(args.train_data_dir_path, args.project_name, "train")
    validate_train(latest_train_path)

    args.train_dataset_path = latest_train_path
    model_name = args.model_dir_path.split("/")[-1]
    new_model_lora_dir_path = os.path.join(args.model_lora_dir_path, str(get_timestamp_from_filename(latest_train_path.split("/")[-1])) + "_" + model_name)
    if not os.path.exists(new_model_lora_dir_path):
        os.makedirs(new_model_lora_dir_path)
    args.model_lora_dir_path = new_model_lora_dir_path

    latest_pred_path = get_latest_dataset_path(args.pred_data_dir_path, args.project_name, "pred")
    validate_pred(latest_pred_path)

    args.pred_dataset_path = latest_pred_path
    new_model_pred_dir_path = os.path.join(args.model_pred_dir_path, str(get_timestamp_from_filename(latest_pred_path.split("/")[-1])) + "_" + model_name)
    if not os.path.exists(new_model_pred_dir_path):
        os.makedirs(new_model_pred_dir_path)
    args.model_pred_dir_path = new_model_pred_dir_path
    
    write_data_info(args.llama_factory_dir_path, latest_train_path, latest_pred_path)


if __name__ == "__main__":
    llama_path = r"/data/disk4/home/yaosong/llm_code/LLaMA-Factory"
    train_dir = r"/data/disk4/home/yaosong/data_label_ft_flow/projects/sales_question_classify/data/train"
    pred_dir = r"/data/disk4/home/yaosong/data_label_ft_flow/projects/sales_question_classify/data/pred"
    project_name = "sales_question_classify"

    prepare_dataset(llama_path, train_dir, pred_dir, project_name)
