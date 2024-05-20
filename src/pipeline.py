import os
import subprocess
from prepare_dataset import prepare_dataset
from arg_parser import init_args
from validation import validate_llama_factory
from reorganize_format import reorganize_format
from setup_bucket import bucket, bucket_viewer


def upload_lora(local_lora_dir_path, project_name):
    time_stamp = local_lora_dir_path.split("/")[-1]
    bucket_lora_dir_path = os.path.join("data_label_ft_flow", project_name, "lora", time_stamp)
    for file_name in os.listdir(local_lora_dir_path):
        local_file_path = os.path.join(local_lora_dir_path, file_name)
        bucket_file_path = os.path.join(bucket_lora_dir_path, file_name)
        bucket.put_object_from_file(bucket_file_path, local_file_path)


def upload_pred_out(local_pred_out_dir_path, project_name):
    time_stamp = local_pred_out_dir_path.split("/")[-1]
    bucket_pred_out_dir_path = os.path.join("data_label_ft_flow", project_name, "data", "pred_out", time_stamp)
    for file_name in os.listdir(local_pred_out_dir_path):
        local_file_path = os.path.join(local_pred_out_dir_path, file_name)
        bucket_file_path = os.path.join(bucket_pred_out_dir_path, file_name)
        bucket.put_object_from_file(bucket_file_path, local_file_path)


def upload_res(local_pred_out_dir_path, local_lora_dir_path, project_name):
    upload_lora(local_lora_dir_path, project_name)
    upload_pred_out(local_pred_out_dir_path, project_name)
    print("\nlora权重，推理结果已上传\n云端该项目现有文件如下：\n")
    bucket_viewer(project_name)


if __name__ == "__main__":
    args = init_args()
    prepare_dataset(args)
    validate_llama_factory(args.llama_factory_dir_path, args.enable_zero3)

    os.chdir(args.llama_factory_dir_path)

    python_path = os.path.join(args.llama_factory_dir_path, "src", "train_bash.py")
    if not args.pred_only:
        ds_config_path = os.path.join(args.llama_factory_dir_path, "ds_config_zero3.json") if args.enable_zero3 else os.path.join(args.llama_factory_dir_path, "ds_config.json")
        stf_return_code = subprocess.call(["sh", args.sft_script_path,
                                        python_path,
                                        args.model_dir_path,
                                        args.template,
                                        args.lora_target,
                                        args.model_lora_dir_path,
                                        ds_config_path,
                                        str(args.max_length),
                                        str(args.num_epoch),
                                        str(args.per_device_train_batch_size)])
        if stf_return_code != 0 and not args.enable_zero3:
            print("-----------------------\n训练未成功，尝试使用deepspeed zero3训练\n-----------------------")
            args.enable_zero3 = True
            validate_llama_factory(args.llama_factory_dir_path, args.enable_zero3)
            ds_config_path = os.path.join(args.llama_factory_dir_path, "ds_config_zero3.json")
            stf_return_code = subprocess.call(["sh", args.sft_script_path,
                                            python_path,
                                            args.model_dir_path,
                                            args.template,
                                            args.lora_target,
                                            args.model_lora_dir_path,
                                            ds_config_path,
                                            str(args.max_length),
                                            str(args.num_epoch),
                                            str(args.per_device_train_batch_size)])
        if stf_return_code != 0:
            print("-----------------------\n训练未成功，程序已退出\n-----------------------")
            print(f"return code: {stf_return_code}")
            exit()
        print("-----------------------\n训练成功，现在开始推理\n-----------------------")
    pred_return_code = subprocess.call(["sh", args.predict_script_path,
                                        python_path,
                                        args.model_dir_path,
                                        args.template,
                                        args.model_lora_dir_path,
                                        args.model_pred_dir_path,
                                        str(args.max_length),
                                        str(args.temperature),
                                        "8"])
    if pred_return_code != 0:
        print("-----------------------\n推理未成功，尝试降低eval batch size为1\n-----------------------")
        pred_return_code = subprocess.call(["sh", args.predict_script_path,
                                            python_path,
                                            args.model_dir_path,
                                            args.template,
                                            args.model_lora_dir_path,
                                            args.model_pred_dir_path,
                                            str(args.max_length),
                                            str(args.temperature),
                                            "1"])
    if pred_return_code != 0:
        print("-----------------------\n推理未成功，程序已退出\n-----------------------")
        print(f"return code: {pred_return_code}")
        exit()
    print("-----------------------\n推理成功，现在开始转换推理结果格式\n-----------------------")
    reorganize_format(args.model_pred_dir_path, args.pred_dataset_path, args.pred_dataset_organize_keys)
    print("-----------------------\n转换推理结果格式成功，现在开始上传lora权重，推理结果\n-----------------------")
    upload_res(args.model_pred_dir_path, args.model_lora_dir_path, args.project_name)