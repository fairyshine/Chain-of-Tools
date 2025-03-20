import os

from WillMindS.utils.io import read_JSON, write_JSON

def dev_2_test(source_path, target_path):
    source_dataset = read_JSON(source_path)
    target_dataset = []
    for index in range(len(source_dataset)):
        source_data = source_dataset[index]
        target_data = {}
        target_data["dataset_name"] = source_data["dataset_name"]
        target_data["case_idx"] = index
        target_data["question"] = source_data["text"].split("Question: ")[-1].split("\nAnswer:")[0]
        target_data["tool"] = source_data["tool_calling"][0]["call_tool"][1:-1]
        target_dataset.append(target_data)
    write_JSON(target_path, target_dataset)
    

def test_2_dev(source_path, target_path):
    source_dataset = read_JSON(source_path)
    target_dataset = []
    for index in range(len(source_dataset)):
        source_data = source_dataset[index]
        target_data = {}
        target_data["dataset_name"] = source_data["dataset_name"]
        target_data["text"] = "Question: " + source_data["question"] + "\nAnswer: The answer is _"
        target_data["start_str_idx"] = [len(target_data["text"])-2]
        target_data["tool_calling"] = [{"call_tool": "<"+source_data["tool"]+">", "call_param": ["_"], "return": ["_"]}]
        target_data["result"] = ["_"]
        target_dataset.append(target_data)
    write_JSON(target_path, target_dataset)


if __name__ == "__main__":
    dev_source_path = "./data/kamel/dev.jsonl"
    dev_target_path = "./data/kamel_switch/dev.jsonl"
    test_source_path = "./data/kamel/test_first_234.jsonl"
    test_target_path = "./data/kamel_switch/test.jsonl"


    dev_2_test(dev_source_path, test_target_path)
    test_2_dev(test_source_path, dev_target_path)