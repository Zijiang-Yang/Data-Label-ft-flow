import os
from setup_bucket import bucket


def validate_train(file_path):
    return


def validate_pred(file_path):
    return


def validate_llama_factory(llama_factory_dir_path, enable_zero3):
    if "ds_config.json" not in os.listdir(llama_factory_dir_path):
        bucket.get_object_to_file('data_label_ft_flow/llama_factory_setting/ds_config.json', os.path.join(llama_factory_dir_path, "ds_config.json"))
    if enable_zero3 and "ds_config_zero3.json" not in os.listdir(llama_factory_dir_path):
        bucket.get_object_to_file('data_label_ft_flow/llama_factory_setting/ds_config_zero3.json', os.path.join(llama_factory_dir_path, "ds_config_zero3.json"))
    