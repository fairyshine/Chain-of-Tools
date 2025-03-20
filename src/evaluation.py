import json

from tool_hub.arithmetic import *

def read_jsonl(data_path):
    dataset=[]
    with open(data_path,'r', encoding='UTF-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def accuracy(pred, true, type = "exact"):
    if len(pred) < len(true):
        true = true[:len(pred)]

    correct = 0
    for p, t in zip(pred, true):
        try:
            if type == "exact":
                if float(p) == float(t):
                    correct += 1
            elif type == "round":
                if round(float(p), 2) == custom_round(float(t), 2):
                    correct += 1
            elif type == "approx":
                # 1% error tolerance, e.g. 1000 -> 990 ~ 1010
                if abs(float(p) - float(t)) <= abs(float(t)) * 0.001:
                    correct += 1
            elif type == "approx_100":
                # 1% error tolerance, e.g. 1000 -> 990 ~ 1010
                if abs(float(p) - float(t)) <= abs(float(t)) * 0.01:
                    correct += 1
        except ValueError:
            pass
    return correct / len(pred)

if __name__ == "__main__":
    output_dataset = read_jsonl("./output/LLaMA_2024-05-23 21:06:00/gsm8k_xltest_result.jsonl")
    pred_list = [data["pred_result"] for data in output_dataset]
    gold_list = [data["gold_result"] for data in output_dataset]

    print("Round Accuracy: "+str(accuracy(pred_list, gold_list, type="round")))
    print("Approx Accuracy: "+str(accuracy(pred_list, gold_list, type="approx")))
    print("Approx100 Accuracy: "+str(accuracy(pred_list, gold_list, type="approx_100")))