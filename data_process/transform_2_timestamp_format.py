import os
import json
import argparse
from upload_utils import write_upload_file


def init_args():
    parser = argparse.ArgumentParser(description='transform existing train and predict data to timestamp format')
    parser.add_argument('--repo_dir_path', type=str, help='git repo dir path')
    parser.add_argument('--input_file_path', type=str, help='path of file needed to be transformed')
    parser.add_argument('--project_name', type=str, help='project name')
    parser.add_argument('--data_type', type=str, choices=["train", "pred"], help='train or pred data')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_args()
    with open(args.input_file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
    
    write_upload_file(dataset, args.repo_dir_path, args.project_name, args.data_type)
