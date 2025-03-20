import os

from WillMindS.utils.io import read_JSON, write_JSON

def process_gsm8k_xl_test_dataset(input_dataset_path, output_dataset_path):
    input_dataset = read_JSON(input_dataset_path)
    output_dataset = []
    for idx in range(len(input_dataset)):
        output_data = {}
        input_data = input_dataset[idx]

        question = input_data["question"]
        enhanced_values = input_data["enhanced_v"]
        result = input_data["enhanced_result"]

        for i in range(len(enhanced_values)):
            question = question.replace(f"{{v_{i+1}}}", str(enhanced_values[i]))

        output_data["dataset_name"] = "gsm8k_xl"
        output_data["case_idx"] = idx
        output_data["question"] = question
        output_data["result"] = result

        output_dataset.append(output_data)
    write_JSON(output_dataset_path, output_dataset)

def process_funcqa_test_dataset(input_dataset_path, output_dataset_path):
    input_dataset = read_JSON(input_dataset_path)
    output_dataset = []
    for idx in range(len(input_dataset)):
        output_data = {}
        input_data = input_dataset[idx]

        question = input_data["question"]
        result = input_data["answer"]

        output_data["dataset_name"] = "funcqa"
        output_data["case_idx"] = idx
        output_data["question"] = question
        output_data["result"] = result

        output_dataset.append(output_data)
    write_JSON(output_dataset_path, output_dataset)

def process_kamel_test_dataset(input_dataset_path, output_dataset_path):
    input_dataset = read_JSON(input_dataset_path)
    output_dataset = []
    for idx in range(len(input_dataset)):
        output_data = {}
        input_data = input_dataset[idx]

        output_data["dataset_name"] = "kamel"
        output_data["case_idx"] = idx
        output_data["question"] = input_data["question"]
        output_data["tool"] = input_data["api"]
        output_data["result"] = input_data["answer"][0]["chosen"]
        output_dataset.append(output_data)
    write_JSON(output_dataset_path, output_dataset)

if __name__ == "__main__":
    input_dataset_path = "./data/raw_dataset_in_each_format/ToolkenGPT/gsm8k-xl/test.json"
    output_dataset_path = "./data/gsm8k_xl/test.jsonl"
    process_gsm8k_xl_test_dataset(input_dataset_path, output_dataset_path)

    input_dataset_path = "./data/raw_dataset_in_each_format/ToolkenGPT/funcqa/funcqa_oh.json"
    output_dataset_path = "./data/funcqa/test_oh.jsonl"
    process_funcqa_test_dataset(input_dataset_path, output_dataset_path)
    input_dataset_path = "./data/raw_dataset_in_each_format/ToolkenGPT/funcqa/funcqa_mh.json"
    output_dataset_path = "./data/funcqa/test_mh.jsonl"
    process_funcqa_test_dataset(input_dataset_path, output_dataset_path)

    # input_dataset_path_list = [
    #     "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/test_first_10.json",
    #     "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/test_first_20.json",
    #     "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/test_first_30.json",
    #     "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/test_first_60.json",
    #     "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/test_first_100.json",
    #     "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/test_first_234.json"
    # ]
    # output_dataset_path_list = [
    #     "./data/kamel/test_first_10.jsonl",
    #     "./data/kamel/test_first_20.jsonl",
    #     "./data/kamel/test_first_30.jsonl",
    #     "./data/kamel/test_first_60.jsonl",
    #     "./data/kamel/test_first_100.jsonl",
    #     "./data/kamel/test_first_234.jsonl",
    # ]
    # for path_idx in range(len(input_dataset_path_list)):
    #     process_kamel_test_dataset(input_dataset_path_list[path_idx], output_dataset_path_list[path_idx])