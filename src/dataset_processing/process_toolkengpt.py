import os

from transformers import AutoTokenizer

from WillMindS.utils.io import read_JSON, write_JSON

def process_gsm8k_xl(tokenizer, input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    tools_bag = read_JSON(output_dir_path+'tool.json')
    tools_list = [key for key in tools_bag["tool_dict"]]

    # 处理训练集
    raw_train_dataset = read_JSON(input_dir_path + "train.json")
    processed_train_dataset = []
    for raw_train_data in raw_train_dataset:
        processed_train_data = dict()

        text = raw_train_data["text"]
        start_str_idx_list = []
        tool_calling_list = []
        for idx in range(len(raw_train_data["start_token_idx"])):
            start_str_idx_list.append(len(tokenizer.decode(tokenizer(text,add_special_tokens=False).input_ids[:raw_train_data["start_token_idx"][idx]-1])))

            tool_calling = raw_train_data["tar_eq"][idx]
            tool_calling_dict = {}

            tool_name = tool_calling.split('(')[0]
            if tool_name[-1] == ">":
                tool_name = tool_name.strip('<>')
            else:
                tool_name = tool_name.split('>')[-1]
            tool_name = [tools_list[index] for index, string in enumerate(tools_list) if "<"+tool_name+">" == string][0]

            param_list = tool_calling.split('(')[-1].split(')')[0].split(',')
            param_list = [ param.strip(' ') for param in param_list]

            return_value = tool_calling.split('=')[-1].split('<')[0]

            tool_calling_dict["call_tool"] = tool_name
            tool_calling_dict["call_param"] = param_list
            tool_calling_dict["return"] = [return_value]
            assert tool_calling_dict["return"][0] == raw_train_data["tar_number"][idx]
            tool_calling_list.append(tool_calling_dict)

        processed_train_data["dataset_name"] = "gsm8k_xl"
        processed_train_data["text"] = text
        processed_train_data["start_str_idx"] = start_str_idx_list
        processed_train_data["tool_calling"] = tool_calling_list
        processed_train_data["result"] = raw_train_data["tar_number"]
        processed_train_dataset.append(processed_train_data)
    write_JSON(output_dir_path+"train.jsonl", processed_train_dataset[:-1000])
    write_JSON(output_dir_path+"dev.jsonl", processed_train_dataset[-1000:])

    # 处理测试集
    # raw_test_dataset = read_JSON(input_dir_path + "test.json")
    # processed_test_dataset = []
    # for raw_test_data in raw_test_dataset:
    #     processed_test_data = dict()

    #     processed_test_dataset.append(raw_test_data)
    # write_JSON(output_dir_path+"test.jsonl", processed_test_dataset)


def process_funcqa(tokenizer, input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    tools_bag = read_JSON(output_dir_path+'tool.json')
    tools_list = [key for key in tools_bag["tool_dict"]]

    # 处理训练集
    raw_train_dataset = read_JSON(input_dir_path + "train.json")
    processed_train_dataset = []
    for raw_train_data in raw_train_dataset:
        processed_train_data = dict()

        text = raw_train_data["text"]
        start_str_idx_list = []
        tool_calling_list = []
        for idx in range(len(raw_train_data["start_token_idx"])):
            start_str_idx_list.append(len(tokenizer.decode(tokenizer(text,add_special_tokens=False).input_ids[:raw_train_data["start_token_idx"][idx]-1])))

            tool_calling = raw_train_data["tar_eq"][idx]
            tool_calling_dict = {}

            tool_name = tool_calling.split('(')[0]
            if tool_name[-1] == ">":
                tool_name = tool_name.strip('<>')
            else:
                tool_name = tool_name.split('>')[-1]
            tool_name = [tools_list[index] for index, string in enumerate(tools_list) if "<"+tool_name+">" == string][0]

            param_list = tool_calling.split('(')[-1].split(')')[0].split(',')
            param_list = [ param.strip(' ') for param in param_list]

            return_value = tool_calling.split('=')[-1].split('<')[0]

            tool_calling_dict["call_tool"] = tool_name
            tool_calling_dict["call_param"] = param_list
            tool_calling_dict["return"] = [return_value]
            assert tool_calling_dict["return"][0] == raw_train_data["tar_number"][idx]
            tool_calling_list.append(tool_calling_dict)

        processed_train_data["dataset_name"] = "funcqa"
        processed_train_data["text"] = text
        processed_train_data["start_str_idx"] = start_str_idx_list
        processed_train_data["tool_calling"] = tool_calling_list
        processed_train_data["result"] = raw_train_data["tar_number"]
        processed_train_dataset.append(processed_train_data)
    write_JSON(output_dir_path+"train.jsonl", processed_train_dataset[:-39])
    write_JSON(output_dir_path+"dev.jsonl", processed_train_dataset[-39:])


def process_kamel(tokenizer, input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    tools_bag = read_JSON(output_dir_path+'tool.json')
    tools_list = [key for key in tools_bag["tool_dict"]]

    # 处理训练集 1 金标
    raw_train_dataset = read_JSON(input_dir_path + "kamel_id_train.json")
    processed_train_dataset = []
    for raw_train_data in raw_train_dataset:
        processed_train_data = dict()

        text = raw_train_data["text"]
        start_str_idx_list = []
        tool_calling_list = []
        for idx in range(len(raw_train_data["start_token_idx"])):
            start_str_idx_list.append(len(tokenizer.decode(tokenizer(text,add_special_tokens=False).input_ids[:raw_train_data["start_token_idx"][idx]-1])))

            tool_calling = raw_train_data["call"][idx]
            tool_calling_dict = {}

            tool_name = tool_calling.split('(')[0]
            if tool_name[-1] == ">":
                tool_name = tool_name.strip('<>')
            else:
                tool_name = tool_name.split('>')[-1]
            tool_name = [tools_list[index] for index, string in enumerate(tools_list) if "<"+tool_name+">" == string][0]
            assert tool_name[1:-1] == raw_train_data["api"]

            param_list = tool_calling.split('(')[-1].split(')')[0].split(',')
            param_list = [ param.strip(' "') for param in param_list]

            return_value = tool_calling.split('=')[-1].split('<')[0]

            tool_calling_dict["call_tool"] = tool_name
            tool_calling_dict["call_param"] = param_list
            tool_calling_dict["return"] = [return_value]
            assert tool_calling_dict["return"][0] == raw_train_data["return"][idx]
            tool_calling_list.append(tool_calling_dict)

        processed_train_data["dataset_name"] = "kamel"
        processed_train_data["text"] = text
        processed_train_data["start_str_idx"] = start_str_idx_list
        processed_train_data["tool_calling"] = tool_calling_list
        processed_train_data["result"] = raw_train_data["return"]
        processed_train_dataset.append(processed_train_data)
    write_JSON(output_dir_path+"train.jsonl", processed_train_dataset[:-1000])
    write_JSON(output_dir_path+"dev.jsonl", processed_train_dataset[-1000:])

def process_kamel_GPT(input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    tools_bag = read_JSON(output_dir_path+'tool.json')
    tools_list = [key for key in tools_bag["tool_dict"]]

    # 处理训练集 2 LLM生成
    raw_train_dataset = read_JSON(input_dir_path + "train_clean.json")
    processed_train_dataset = []
    miss_counter = 0
    for raw_train_data in raw_train_dataset:
        try:
            processed_train_data = dict()

            text = raw_train_data["text"]
            start_str_idx_list = []
            result_list = []
            tool_calling_list = []
            for idx in range(len(raw_train_data["call"])):
                # start_str_idx_list.append(len(tokenizer.decode(tokenizer(text,add_special_tokens=False).input_ids[:raw_train_data["start_token_idx"][idx]-1])))
                # result_list.append(tokenizer.decode(tokenizer(text,add_special_tokens=True).input_ids[raw_train_data["start_token_idx"][idx]:raw_train_data["end_token_idx"][idx]]))
                tool_calling = raw_train_data["call"][idx]
                tool_calling_dict = {}

                tool_name = tool_calling.split('(')[0]
                if tool_name[-1] == ">":
                    tool_name = tool_name.strip('<>')
                else:
                    tool_name = tool_name.split('>')[-1]
                tool_name = [tools_list[index] for index, string in enumerate(tools_list) if "<"+tool_name+">" == string][0]
                assert tool_name[1:-1] == raw_train_data["api"]

                param_list = tool_calling.split('(')[-1].split(')')[0].split(',')
                param_list = [ param.strip(' "') for param in param_list]

                return_value = tool_calling.split('=')[-1].split('<')[0].strip('"')

                tool_calling_dict["call_tool"] = tool_name
                tool_calling_dict["call_param"] = param_list
                tool_calling_dict["return"] = [return_value]
                tool_calling_list.append(tool_calling_dict)

                result_list.append(tool_calling_dict["return"][0])
                start_str_idx_list.append(len(text.split(result_list[-1])[0]))
                if result_list[-1][0].isalpha():
                    start_str_idx_list[-1] -= 1

            processed_train_data["dataset_name"] = "kamel"
            processed_train_data["text"] = text
            processed_train_data["start_str_idx"] = start_str_idx_list
            processed_train_data["tool_calling"] = tool_calling_list
            processed_train_data["result"] = result_list
            processed_train_dataset.append(processed_train_data)
        except:
            miss_counter += 1
            pass
    print("Missing: ",miss_counter)
    write_JSON(output_dir_path+"train.jsonl", processed_train_dataset[:-1000])
    write_JSON(output_dir_path+"dev.jsonl", processed_train_dataset[-1000:])

def process_kamel_strip(tokenizer, input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    tools_bag = read_JSON(output_dir_path+'tool.json')
    tools_list = [key for key in tools_bag["tool_dict"]]

    # 处理训练集 1 金标
    raw_train_dataset = read_JSON(input_dir_path + "kamel_id_train.json")
    processed_train_dataset = []
    for raw_train_data in raw_train_dataset:
        processed_train_data = dict()

        text = raw_train_data["text"]
        start_str_idx_list = []
        tool_calling_list = []
        for idx in range(len(raw_train_data["start_token_idx"])):
            start_str_idx_list.append(len(tokenizer.decode(tokenizer(text,add_special_tokens=False).input_ids[:raw_train_data["start_token_idx"][idx]-1])))

            tool_calling = raw_train_data["call"][idx]
            tool_calling_dict = {}

            tool_name = tool_calling.split('(')[0]
            if tool_name[-1] == ">":
                tool_name = tool_name.strip('<>')
            else:
                tool_name = tool_name.split('>')[-1]
            tool_name = [tools_list[index] for index, string in enumerate(tools_list) if "<"+tool_name+">" == string][0]
            assert tool_name[1:-1] == raw_train_data["api"]

            param_list = tool_calling.split('(')[-1].split(')')[0].split(',')
            param_list = [ param.strip(' "') for param in param_list]

            return_value = tool_calling.split('=')[-1].split('<')[0]

            tool_calling_dict["call_tool"] = tool_name
            tool_calling_dict["call_param"] = param_list
            tool_calling_dict["return"] = [return_value]
            assert tool_calling_dict["return"][0] == raw_train_data["return"][idx]
            tool_calling_list.append(tool_calling_dict)

            if return_value[0].isdigit():
                start_str_idx_list[-1] -= 1

        processed_train_data["dataset_name"] = "kamel"
        processed_train_data["text"] = text
        processed_train_data["start_str_idx"] = start_str_idx_list
        processed_train_data["tool_calling"] = tool_calling_list
        processed_train_data["result"] = raw_train_data["return"]
        processed_train_dataset.append(processed_train_data)
    write_JSON(output_dir_path+"train.jsonl", processed_train_dataset[:-1000])
    write_JSON(output_dir_path+"dev.jsonl", processed_train_dataset[-1000:])

def process_kamel_GPT_strip(input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    tools_bag = read_JSON(output_dir_path+'tool.json')
    tools_list = [key for key in tools_bag["tool_dict"]]

    # 处理训练集 2 LLM生成
    raw_train_dataset = read_JSON(input_dir_path + "train_clean.json")
    processed_train_dataset = []
    miss_counter = 0
    for raw_train_data in raw_train_dataset:
        try:
            processed_train_data = dict()

            text = raw_train_data["text"]
            start_str_idx_list = []
            result_list = []
            tool_calling_list = []
            for idx in range(len(raw_train_data["call"])):
                # start_str_idx_list.append(len(tokenizer.decode(tokenizer(text,add_special_tokens=False).input_ids[:raw_train_data["start_token_idx"][idx]-1])))
                # result_list.append(tokenizer.decode(tokenizer(text,add_special_tokens=True).input_ids[raw_train_data["start_token_idx"][idx]:raw_train_data["end_token_idx"][idx]]))
                tool_calling = raw_train_data["call"][idx]
                tool_calling_dict = {}

                tool_name = tool_calling.split('(')[0]
                if tool_name[-1] == ">":
                    tool_name = tool_name.strip('<>')
                else:
                    tool_name = tool_name.split('>')[-1]
                tool_name = [tools_list[index] for index, string in enumerate(tools_list) if "<"+tool_name+">" == string][0]
                assert tool_name[1:-1] == raw_train_data["api"]

                param_list = tool_calling.split('(')[-1].split(')')[0].split(',')
                param_list = [ param.strip(' "') for param in param_list]

                return_value = tool_calling.split('=')[-1].split('<')[0].strip('"')

                tool_calling_dict["call_tool"] = tool_name
                tool_calling_dict["call_param"] = param_list
                tool_calling_dict["return"] = [return_value]
                tool_calling_list.append(tool_calling_dict)

                result_list.append(tool_calling_dict["return"][0])
                start_str_idx_list.append(len(text.split(result_list[-1])[0]))

                start_str_idx_list[-1] -= 1

            processed_train_data["dataset_name"] = "kamel"
            processed_train_data["text"] = text
            processed_train_data["start_str_idx"] = start_str_idx_list
            processed_train_data["tool_calling"] = tool_calling_list
            processed_train_data["result"] = result_list
            processed_train_dataset.append(processed_train_data)
        except:
            miss_counter += 1
            pass
    print("Missing: ",miss_counter)
    write_JSON(output_dir_path+"train.jsonl", processed_train_dataset[:-1000])
    write_JSON(output_dir_path+"dev.jsonl", processed_train_dataset[-1000:])

def process_vh(tokenizer, input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    # 处理训练集
    raw_train_dataset = read_JSON(input_dir_path + "legal_train_v4_embedding.json")
    processed_train_dataset = []
    for raw_train_data in raw_train_dataset:
        processed_train_data = dict()

        text = raw_train_data["text"]
        start_str_idx_list = []
        tool_calling_list = []
        for idx in range(len(raw_train_data["start_token_idx"])):
            start_str_idx_list.append(len(tokenizer.decode(tokenizer(text,add_special_tokens=False).input_ids[:raw_train_data["start_token_idx"][idx]-1])))
            tool_calling_list.append({"call_tool":raw_train_data["tar_eq"][idx]})

        processed_train_data["dataset_name"] = "vh"
        processed_train_data["text"] = text
        processed_train_data["start_str_idx"] = start_str_idx_list
        processed_train_data["tool_calling"] = tool_calling_list
        processed_train_data["result"] = None
        processed_train_dataset.append(processed_train_data)
    write_JSON(output_dir_path+"train.jsonl", processed_train_dataset[:-47])
    write_JSON(output_dir_path+"dev.jsonl", processed_train_dataset[-47:])

# def combine_toolkengpt_trainset(input_dir_dict, output_dir_path, output_file_name):
#     dataset_test_len_dict = {
#         "gsm8k_xl": 1000,
#         "funcqa": 39,
#         "vh": 47,
#         "kamel": 1000
#     }
#     # 初始化
#     if not os.path.exists(output_dir_path):
#         os.mkdir(output_dir_path)
    
#     combined_dataset = []
#     for dataset_name in input_dir_dict:
#         dataset_load = read_JSON(input_dir_dict[dataset_name])[:-dataset_test_len_dict[dataset_name]]
#         for i in range(len(dataset_load)):
#             dataset_load[i]["dataset_name"] = dataset_name
#         combined_dataset.extend(dataset_load)

#     write_JSON(output_dir_path+output_file_name, combined_dataset)

    



if __name__ == "__main__":
    Checkpoint = "/public/home/jhfang/mswu_wlchen/PTM/Llama-2-7b-chat-hf"
    Tokenizer = AutoTokenizer.from_pretrained(Checkpoint)

    # input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/gsm8k-xl/"
    # output_dir_path = "./data/gsm8k_xl/"
    # process_gsm8k_xl(Tokenizer, input_dir_path, output_dir_path)

    # input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/funcqa/"
    # output_dir_path = "./data/funcqa/"
    # process_funcqa(Tokenizer, input_dir_path, output_dir_path)

    # input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/"
    # output_dir_path = "./data/kamel/"
    # process_kamel(Tokenizer, input_dir_path, output_dir_path)

    # input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/vh/"
    # output_dir_path = "./data/vh/"
    # process_vh(Tokenizer, input_dir_path, output_dir_path)


    # input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/"
    # output_dir_path = "./data/kamel_GPT/"
    # process_kamel_GPT(input_dir_path, output_dir_path)


    input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/"
    output_dir_path = "./data/kamel_strip/"
    process_kamel_strip(Tokenizer, input_dir_path, output_dir_path)

    input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/"
    output_dir_path = "./data/kamel_GPT_strip/"
    process_kamel_GPT_strip(input_dir_path, output_dir_path)