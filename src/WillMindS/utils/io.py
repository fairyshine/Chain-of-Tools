import json
import pickle

def read_json(data_path):
    with open(data_path,'r', encoding='UTF-8') as f:
        dataset = json.load(f)
    return dataset


def read_jsonl(data_path):
    dataset=[]
    with open(data_path,'r', encoding='UTF-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def write_json(data_path, dataset, indent=0):
    with open(data_path,'w', encoding='UTF-8') as f:
            if indent != 0:
                json.dump(dataset, f, ensure_ascii=False, indent=indent)
            else:
                json.dump(dataset, f, ensure_ascii=False)


def write_jsonl(data_path, dataset, indent=0):
    with open(data_path,'w', encoding='UTF-8') as f:
        for data in dataset:
            if indent != 0:
                f.writelines(json.dumps(data, ensure_ascii=False, indent=indent))
            else:
                f.writelines(json.dumps(data, ensure_ascii=False))
            f.write('\n')


def read_JSON(data_path):
    if data_path.split('.')[-1].lower() == "json":
        try:
            return read_json(data_path)
        except:
            return read_jsonl(data_path)
    elif data_path.split('.')[-1].lower() == "jsonl":
        try:
            return read_jsonl(data_path)
        except:
            return read_json(data_path)
    else:
        print("data_path error !!!")


def write_JSON(data_path, dataset, indent=0):
    if data_path.split('.')[-1].lower() == "json":
        try:
            return write_json(data_path, dataset, indent)
        except:
            return write_jsonl(data_path, dataset, indent)
    elif data_path.split('.')[-1].lower() == "jsonl":
        try:
            return write_jsonl(data_path, dataset, indent)
        except:
            return write_json(data_path, dataset, indent)
    else:
        print("data_path error !!!")

def read_pickle(data_path):
    with open(data_path,'rb') as f:
        dataset = pickle.load(f)
    return dataset

def write_pickle(data_path, dataset):
    with open(data_path,'wb') as f:
        pickle.dump(dataset, f)