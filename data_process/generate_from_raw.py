import os
import json
import argparse
from typing import List, Union, TypedDict
from upload_utils import write_upload_file
import time


def init_args():
    parser = argparse.ArgumentParser(description='generate train and predict data')
    parser.add_argument('--repo_dir_path', type=str, help='git repo dir path')
    parser.add_argument('--project_name', type=str, help='project name')
    args = parser.parse_args()
    return args


def get_process_func(project_name: str):
    match project_name:
        case "sales_question_classify":
            from process_raw_by_project.sales_question_classify import get_train_list, get_pred_list
        case "insurance_liability_clause_extract":
            from process_raw_by_project.insurance_liability_clause_extract import get_train_list, get_pred_list
    return get_train_list, get_pred_list


def handle_task(args):
    get_train_list, get_pred_list = get_process_func(args.project_name)

    train_list = get_train_list()
    write_upload_file(train_list, args.repo_dir_path, args.project_name, "train")

    pred_list = get_pred_list()
    write_upload_file(pred_list, args.repo_dir_path, args.project_name, "pred")

    return


if __name__ == "__main__":
    args = init_args()
    handle_task(args)