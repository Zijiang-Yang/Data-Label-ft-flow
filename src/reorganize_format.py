import os
import json
import pandas as pd

def _load_pred_dataset(pred_dataset_path, pred_dataset_organize_keys):
    organize_keys = pred_dataset_organize_keys.split(",")
    output_2_keys_dict = dict()
    with open(pred_dataset_path, "r", encoding="utf-8") as file:
        pred_list = json.load(file)

    for info_dict in pred_list:
        output = info_dict["output"]
        output_2_keys_dict[output] = dict()
        for key, value in info_dict.items():
            if key not in organize_keys:
                continue
            output_2_keys_dict[output][key] = value
    return output_2_keys_dict

def reorganize_format(pred_out_dir_path, pred_dataset_path, pred_dataset_organize_keys):
    if pred_dataset_organize_keys is not None:
        organize_keys = pred_dataset_organize_keys.split(",")
        output_2_keys_dict = _load_pred_dataset(pred_dataset_path, pred_dataset_organize_keys)
    else:
        organize_keys = []
        output_2_keys_dict = dict()
    result_jsonl_path = os.path.join(pred_out_dir_path, "generated_predictions.jsonl")
    res_list = []
    with open(result_jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            info_dict = eval(line)
            output = info_dict["label"]
            single_pred_res = [output, info_dict["predict"]]
            for organize_key in organize_keys:
                value = output_2_keys_dict[output][organize_key]
                single_pred_res.append(value)
            res_list.append(single_pred_res)
        df = pd.DataFrame(res_list, columns=["input", "predict"] + organize_keys)
        output_file_path = os.path.join(pred_out_dir_path, pred_out_dir_path.split("/")[-1] + "_predict_output.xlsx")
        df.to_excel(output_file_path, index=False)


if __name__ == "__main__":
    pred_dir_path = r"/data/disk4/home/yaosong/QA_LLM/question_classify_db/child_insurance_label_pred_res_01"
    # reorganize_format(pred_dir_path, "", None)

    pred_data_file_path = r"/data/disk4/home/yaosong/QA_LLM/question_classify_db/pred_questions_child_insurance.json"
    reorganize_format(pred_dir_path, pred_data_file_path, "sim_questions")
