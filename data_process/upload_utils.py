import os
import json
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import time


os.environ['OSS_ACCESS_KEY_ID'] = 'xxxx'
os.environ['OSS_ACCESS_KEY_SECRET'] = 'xxxx'

auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, 'xxxx', 'xxxx')


def write_upload_file(res_list, repo_dir_path, project_name, data_type):
    local_dir_path = os.path.join(repo_dir_path, "projects", project_name, "data", data_type)
    if not os.path.exists(local_dir_path):
        os.makedirs(local_dir_path)
    time_stamp = int(time.time())
    match data_type:
        case "train":
            file_name = f"{time_stamp}_ft.json"
        case "pred":
            file_name = f"{time_stamp}_pred.json"
    local_file_path = os.path.join(local_dir_path, file_name)
    with open(local_file_path, "w", newline="", encoding="utf-8") as file:
        json.dump(res_list, file, ensure_ascii=False, indent=4)
    
    bucket_file_path = os.path.join("data_label_ft_flow", project_name, "data", data_type, file_name)
    bucket.put_object_from_file(bucket_file_path, local_file_path)