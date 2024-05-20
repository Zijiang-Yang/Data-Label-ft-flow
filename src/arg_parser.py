import os
import argparse


model_name_2_info = {"yi": ["llama2", "q_proj,k_proj"], "llama2-chinese":["llama2", "q_proj,k_proj"],
                     "llama2": ["llama2", "q_proj,k_proj"], "chatglm3": ["chatglm3", "query_key_value"], 
                     "qwen": ["qwen", "c_attn"], "baichuan2":["baichuan2", "W_pack"],
                     "internlm2": ["intern2", "wqkv"], "phi": [None, "Wqkv"]}


class ModelChoiceException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _init_relative_args(args):
    args.sft_script_path = os.path.join(args.repo_dir_path, "src/sft.sh")
    args.predict_script_path = os.path.join(args.repo_dir_path, "src/predict.sh")
    args.train_data_dir_path = os.path.join(args.repo_dir_path, "projects", args.project_name, "data", "train")
    args.pred_data_dir_path = os.path.join(args.repo_dir_path, "projects", args.project_name, "data", "pred")
    args.model_lora_dir_path = os.path.join(args.repo_dir_path, "projects", args.project_name, "lora")
    args.model_pred_dir_path = os.path.join(args.repo_dir_path, "projects", args.project_name, "data", "pred_out")
    if len(args.pred_dataset_organize_keys) <= 0 or args.pred_dataset_organize_keys.lower() == "none":
        args.pred_dataset_organize_keys = None
        
    model_name = None
    available_models = ["yi", "llama2-chinese", "llama2", "chatglm3", "qwen", "baichuan2", "phi", "internlm2"]
    for abrev_name in available_models:
        if abrev_name in args.model_dir_path.lower():
            model_name = abrev_name
    if not model_name:
        raise ModelChoiceException("model should only be chosen from {}".format(", ".join(available_models)))
    args.template, args.lora_target = model_name_2_info[model_name]


def init_args():
    parser = argparse.ArgumentParser(description='lora sft and predict using LLaMA Factory')
    parser.add_argument('--repo_dir_path', type=str, help='git repo dir path')
    parser.add_argument('--llama_factory_dir_path', type=str, help='llama factory dir path')
    parser.add_argument('--model_dir_path', type=str, help='base model weights dir path')
    parser.add_argument('--project_name', type=str, help='project name')
    parser.add_argument('--num_epoch', type=int, nargs='?', default=3, help='number of epochs in traning')
    parser.add_argument('--per_device_train_batch_size', type=int, nargs='?', default=1, help='train batch size')
    parser.add_argument('--max_length', type=int, nargs='?', default=1024, help='train data max length')
    parser.add_argument('--temperature', type=float, nargs='?', default=0.01, help='generation temperature')
    parser.add_argument('--is_fp16', action='store_true', help='use fp16 for model param')
    parser.add_argument('--enable_zero3', action='store_true', help='use zero3 deepspeed for training')
    parser.add_argument('--pred_only', action='store_true', help='only do predict')
    parser.add_argument('--pred_dataset_organize_keys', type=str, default=None, help='keys to be organized to xlsx after prediction (default: None)')
    args = parser.parse_args()
    _init_relative_args(args)
    return args