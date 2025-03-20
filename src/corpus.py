import torch
from torch.utils.data import Dataset

from prompt import prompt_tool_retriever

class judge_Collater():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self,batch_data_list):
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = False
        collated_dict = {}
        input_ids_list = []
        judge_labels_list = []
        for raw_data in batch_data_list:
            input_ids = self.tokenizer(raw_data["text"]).input_ids
            judge_labels = [0]*len(input_ids)
            start_token_idx_list = []
            for start_str_idx in raw_data["start_str_idx"]:
                start_token_idx = len(self.tokenizer(raw_data["text"][:start_str_idx], add_special_tokens=False).input_ids)
                start_token_idx_list.append(start_token_idx)

            for start_token_idx in start_token_idx_list:
                judge_labels[start_token_idx] = 1  #+1-1    +1是因为开头有<s> -1是因为用前一个token预测 相互抵消

            input_ids_list.append(input_ids)
            judge_labels_list.append(judge_labels)

        collated_dict["input_ids"] = torch.tensor(input_ids_list)
        collated_dict["judge_labels"] = torch.tensor(judge_labels_list)
        return collated_dict

class judge_Dataset(Dataset):
    def __init__ (self, input_dataset):
        super().__init__()
        self.raw_dataset = input_dataset
        self.processed_dataset = self.data_process()

    def data_process(self):
        processed_dataset = []
        for raw_data in self.raw_dataset:
            processed_data = dict()
            text = raw_data["text"]
            start_str_idx_list = raw_data["start_str_idx"]

            processed_data["text"] = text
            processed_data["start_str_idx"] = start_str_idx_list

            processed_dataset.append(processed_data)
        return processed_dataset

    def __getitem__(self, index):
        return self.processed_dataset[index]

    def __len__(self):
        return len(self.processed_dataset)


class retriever_Collater():
    def __init__(self, tokenizer, tool_database):
        self.tokenizer = tokenizer
        self.tool_database = tool_database

    def __call__(self,batch_data_list):
        self.tokenizer.padding_side = 'right'
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = True
        collated_dict = {}
        dataset_name_list = []
        query_list = []
        tool_name_list = []
        tool_name_in_batch_list = []
        # tool_vectors_list = []
        tool_information_in_batch_list = []
        gold_label_in_batch_list = []

        for raw_data in batch_data_list:
            dataset_name_list.append(raw_data["dataset_name"])
            query_list.append(raw_data["text"])
            tool_name_list.append(raw_data["tool_name"])

        for i in range(len(tool_name_list)):
            search_tool_text = dataset_name_list[i]+'_'+tool_name_list[i]
            if search_tool_text in tool_name_in_batch_list:
                gold_label_in_batch_list.append(tool_name_in_batch_list.index(search_tool_text))
            else: 
                tool_name_in_batch_list.append(search_tool_text)
                # tool_vectors_list.append(self.tool_database[dataset_name_list[i]]["vec_lib"][tool_name_list[i]]["tool_vectors"][0])
                tool_information_in_batch_list.append(prompt_tool_retriever(tool_name_list[i], self.tool_database[dataset_name_list[i]]["tool_information"][tool_name_list[i]]["tool_description"]))
                gold_label_in_batch_list.append(len(tool_name_in_batch_list)-1)

        tokenized_query = self.tokenizer(query_list, add_special_tokens=True, padding=True, return_tensors='pt', return_attention_mask=True)
        tokenized_tool = self.tokenizer(tool_information_in_batch_list, add_special_tokens=True, padding=True, return_tensors='pt', return_attention_mask=True)

        collated_dict["query_tokens"] = tokenized_query
        # collated_dict["tool_vectors_in"] = torch.stack(tool_vectors_list, 0)
        collated_dict["tool_tokens"] = tokenized_tool
        collated_dict["gold_labels"] = torch.tensor(gold_label_in_batch_list)

        collated_dict["dataset_name"] = dataset_name_list
        collated_dict["tool_name"] = tool_name_list

        return collated_dict

class retriever_Dataset(Dataset):
    def __init__ (self, input_dataset):
        super().__init__()
        self.raw_dataset = input_dataset
        self.processed_dataset = self.data_process()

    def data_process(self):
        processed_dataset = []
        for raw_data in self.raw_dataset:
            processed_data = dict()
            text = raw_data["text"]
            for idx in range(len(raw_data["tool_calling"])):
                split_text = text[:raw_data["start_str_idx"][idx]]

                processed_data["dataset_name"] = raw_data["dataset_name"]
                processed_data["text"] = split_text
                processed_data["tool_name"] = raw_data["tool_calling"][idx]["call_tool"]

                processed_dataset.append(processed_data)
        return processed_dataset


    def __getitem__(self, index):
        return self.processed_dataset[index]

    def __len__(self):
        return len(self.processed_dataset)