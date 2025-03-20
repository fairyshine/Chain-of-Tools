import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.activations import ACT2FN

from WillMindS.utils.io import read_JSON, read_pickle, write_pickle

from prompt import prompt_tool_retriever

class Tensor_Weighting(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, vector):
        return self.weight * vector

class MLPLayer(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size
        self.gate_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LLM_with_tools(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

        self.foundation_model = AutoModelForCausalLM.from_pretrained(self.config.checkpoint, device_map="auto")
        if self.config.half_quantized:
            self.foundation_model = self.foundation_model.half()
        self.hidden_size = self.foundation_model.config.hidden_size
        try:
            self.intermediate_size = self.foundation_model.config.intermediate_size
        except:
            self.intermediate_size = self.hidden_size * 4

        self.retrieval_size = self.hidden_size

        self.tool_database = {}
        self.tool_range = self.config.tool_range

        # 判别是否调用工具
        self.tool_judge = MLPLayer(self.hidden_size, self.intermediate_size, 1).to("cuda:0") #.to("cuda:{}".format(torch.cuda.device_count()-1))

        # 选择调用什么工具
        if self.config.cal_seq:
            self.retriever_query = MambaBlock(self.hidden_size).to("cuda:0")
            self.retriever_tool_selection = MambaBlock(self.hidden_size).to("cuda:1")
            # self.retriever_tool_parameter = MambaBlock(self.hidden_size).to("cuda:1")
        else:
            if self.config.use_mamba:
                self.retriever_query = MambaBlock(self.hidden_size).to("cuda:0")
                self.retriever_tool_selection = MambaBlock(self.hidden_size).to("cuda:1")
                # self.retriever_tool_parameter = MambaBlock(self.hidden_size).to("cuda:1")
            else:
                self.retriever_query = MLPLayer(self.hidden_size, self.intermediate_size, self.retrieval_size).to("cuda:0")
                self.retriever_tool_selection = MLPLayer(self.hidden_size, self.intermediate_size, self.retrieval_size).to("cuda:1")
                # self.retriever_tool_parameter = MLPLayer(self.hidden_size, self.intermediate_size, self.retrieval_size).to("cuda:1")
        
        self.w_tensor = Tensor_Weighting(self.hidden_size)

    def sl_judge(self, dir, mode, message=""):
        if message != "":
            dir_new = dir + str(message) + "/"
            if not os.path.exists(dir_new):
                os.makedirs(dir_new)
            path = dir_new + 'tool_judge.pt'
        else:
            path = dir + 'tool_judge.pt'
        if mode == "save":
            torch.save(self.tool_judge.state_dict(), path)
        else:
            assert mode == "load"
            self.tool_judge.load_state_dict(torch.load(path))

    def sl_retriever(self, dir, mode, message=""):
        if message != "":
            dir_new = dir + str(message) + "/"
            if not os.path.exists(dir_new):
                os.makedirs(dir_new)
            path_list = [dir_new+"retriever_query.pt",
                            dir_new+"retriever_tool.pt",
                            dir_new+"retriever_param.pt",
                            dir_new+"retriever_tensor_weighting.pt",]
        else:
            path_list = [dir+"retriever_query.pt",
                            dir+"retriever_tool.pt",
                            dir+"retriever_param.pt",
                            dir+"retriever_tensor_weighting.pt",]
        if mode == "save":
            torch.save(self.retriever_query.state_dict(), path_list[0])
            torch.save(self.retriever_tool_selection.state_dict(), path_list[1])
            # torch.save(self.retriever_tool_parameter.state_dict(), path_list[2])
            if self.config.tensor_weighting:
                torch.save(self.w_tensor.state_dict(), path_list[3])
        else:
            assert mode == "load"
            self.retriever_query.load_state_dict(torch.load(path_list[0]))
            self.retriever_tool_selection.load_state_dict(torch.load(path_list[1]))
            # self.retriever_tool_parameter.load_state_dict(torch.load(path_list[2]))
            if self.config.tensor_weighting:
                self.w_tensor.load_state_dict(torch.load(path_list[3]))

    def get_tensor_filtering_index(self):
        assert self.config.tensor_weighting
        weight_list = self.w_tensor.weight.tolist()
        filter_index_list = []
        for w_index in range(len(weight_list)):
            if weight_list[w_index] >= self.config.tensor_filtering_threshold:
                filter_index_list.append(w_index)
        self.tensor_filtering_index = torch.tensor(filter_index_list)


    def read_tool_database_file(self, dataset_name, data_path):
        self.tool_database[dataset_name] = read_pickle(data_path)

    def save_tool_database_file(self, dataset_name, data_path):
        write_pickle(data_path, self.tool_database[dataset_name])

    def load_tool_database(self, data_dir_dict):
        if self.config.database_vector_calculate:
            for dataset_name in data_dir_dict:
                self.init_database(dataset_name, data_dir_dict[dataset_name]+'tool.json')
            for dataset_name in data_dir_dict:
                self.save_tool_database_file(dataset_name, data_dir_dict[dataset_name]+'tool_database.pickle')
        else:
            for dataset_name in data_dir_dict:
                self.read_tool_database_file(dataset_name, data_dir_dict[dataset_name]+'tool_database.pickle')

    def init_database(self, dataset_name, data_path):
        self.tool_database[dataset_name] = read_JSON(data_path)
        self.tool_database[dataset_name]["vec_lib"] = {}
        for tool_name in self.tool_database[dataset_name]["tool_information"]:
            self.tool_database[dataset_name]["vec_lib"][tool_name] = {"tool_vectors":[None, None],"param_vectors":{}}

    def calculate_query_vector(self, hidden_states, calculated_tensor=None):
        if self.config.cal_seq:
            query_vector = calculated_tensor + hidden_states.to(calculated_tensor.device)
        else:
            raw_dim = hidden_states.dim()
            if raw_dim == 1:
                hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
            elif raw_dim == 2:
                hidden_states = hidden_states.unsqueeze(0)
            else:
                assert raw_dim == 3
            query_vector = self.retriever_query(hidden_states.to(next(self.retriever_query.parameters()).device))
            query_vector += hidden_states.to(query_vector.device)
            query_vector = query_vector.squeeze()
            if raw_dim == 2 and query_vector.dim() == 1:
                query_vector = query_vector.unsqueeze(0)
        if self.config.tensor_weighting:
            query_vector = self.w_tensor(query_vector.to(self.w_tensor.weight.device))
        return query_vector

    def calculate_tool_vector(self, hidden_states, calculated_tensor=None):
        if self.config.cal_seq:
            tool_vector = calculated_tensor + hidden_states.to(calculated_tensor.device)
        else:
            raw_dim = hidden_states.dim()
            if raw_dim == 1:
                hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
            elif raw_dim == 2:
                hidden_states = hidden_states.unsqueeze(0)
            else:
                assert raw_dim == 3
            tool_vector = self.retriever_tool_selection(hidden_states.to(next(self.retriever_tool_selection.parameters()).device))
            tool_vector += hidden_states.to(tool_vector.device)
            tool_vector = tool_vector.squeeze()
            if raw_dim == 2 and tool_vector.dim() == 1:
                tool_vector = tool_vector.unsqueeze(0)
        if self.config.tensor_weighting:
            tool_vector = self.w_tensor(tool_vector.to(self.w_tensor.weight.device))
        return tool_vector

    def calculate_param_vector(self, hidden_states):
        pass

    def calculate_database(self,dataset_name_list):
        self.tokenizer.padding_side = 'right'
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = True
        for dataset_name in dataset_name_list:
            for tool_name in self.tool_database[dataset_name]["tool_information"]:
                tool_information_text = prompt_tool_retriever(tool_name, self.tool_database[dataset_name]["tool_information"][tool_name]["tool_description"])
                tokenized_tool = self.tokenizer(
                                    tool_information_text,
                                    add_special_tokens=True, padding=True, return_tensors='pt', return_attention_mask=True).to('cuda')
                # 计算tool vector
                with torch.no_grad():
                    self.foundation_model.eval( )
                    tool_output = self.foundation_model(input_ids=tokenized_tool.input_ids, attention_mask=tokenized_tool.attention_mask, output_hidden_states=True)
                    tool_pos_id = torch.where(tokenized_tool.input_ids[0] == self.tokenizer.eos_token_id)[0][0].item()
                    hidden_state = tool_output.hidden_states[-1][0][tool_pos_id]
                    if self.config.cal_seq:
                        calculated_states = self.retriever_tool_selection(tool_output.hidden_states[-1].to(next(self.retriever_tool_selection.parameters()).device))
                        calculated_tensor = calculated_states[0][tool_pos_id]
                        tool_vector_out = self.calculate_tool_vector(hidden_state, calculated_tensor=calculated_tensor)
                    else:
                        tool_vector_out = self.calculate_tool_vector(hidden_state)

                self.tool_database[dataset_name]["vec_lib"][tool_name]["tool_vectors"] = [hidden_state, tool_vector_out]

    # def calculate_database_output_vectors(self):
    #     for dataset_name in self.tool_database:
    #         for tool_name in self.tool_database[dataset_name]["vec_lib"]:
    #             # 计算tool vector
    #             tool_vector_in = self.tool_database[dataset_name]["vec_lib"][tool_name]["tool_vectors"][0]
    #             tool_vector_out = self.calculate_tool_vector(tool_vector_in)
    #             self.tool_database[dataset_name]["vec_lib"][tool_name]["tool_vectors"][1] = tool_vector_out

    def tool_judging(self, hidden_states):
        probs = self.tool_judge(hidden_states.to(next(self.tool_judge.parameters()).device))
        if list(probs.size()) == [1]:
            single_probs = F.sigmoid(probs)[0]
            if single_probs > 0.5:
                return True, probs
            else:
                return False, probs
        else:
            probs = torch.squeeze(probs, dim=-1)
            return probs
    
    @torch.no_grad()
    def tool_judging_with_query(self, query):
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = False
        input_ids_list = self.tokenizer(query, add_special_tokens=True).input_ids
        model_output = self.foundation_model(input_ids=torch.tensor([input_ids_list]), output_hidden_states=True)
        hidden_states = model_output.hidden_states[-1][0][-1]
        tool_judge_signal, _ = self.tool_judging(hidden_states)
        return tool_judge_signal

    def tool_selection(self, dataset_name, hidden_states):
        assert len(list(hidden_states.size())) == 3

        if self.config.cal_seq:
            calculated_states = self.retriever_query(hidden_states.to(next(self.retriever_query.parameters()).device))
            query_vector = self.calculate_query_vector(hidden_states[0][-1], calculated_tensor=calculated_states[0][-1])
        else:
            query_vector = self.calculate_query_vector(hidden_states)
            query_vector = query_vector.squeeze()

            query_vector = torch.nn.functional.normalize(query_vector,p=2,dim=-1)
            if self.config.tensor_weighting and self.config.tensor_filtering:
                query_vector = query_vector[self.tensor_filtering_index] 

        probs = [0] * len(self.tool_database[dataset_name]["tool_dict"])
        tool_name_list = [None] * len(probs)
        for tool_name,tool_idx in self.tool_database[dataset_name]["tool_dict"].items():
            tool_vector = self.tool_database[dataset_name]["vec_lib"][tool_name]["tool_vectors"][1]

            tool_vector = torch.nn.functional.normalize(tool_vector,p=2,dim=-1)
            if self.config.tensor_weighting and self.config.tensor_filtering:
                tool_vector = tool_vector[self.tensor_filtering_index]

            query_vector = query_vector.to(tool_vector.device)
            if not self.config.similarity_norm:
                probs[tool_idx] = torch.matmul(query_vector, tool_vector)
            else:
                probs[tool_idx] = torch.matmul(query_vector, tool_vector) / ((torch.norm(query_vector) + torch.norm(tool_vector))/2)**2
            tool_name_list[tool_idx] = tool_name
        # 限制工具检索的搜索范围
        if self.tool_range > 0:
            probs = probs[:self.tool_range]
        tool_idx = probs.index(max(probs))
        top_k = min(5, len(probs))
        _ , top_k_indices = torch.topk(torch.tensor(probs), top_k)
        return tool_name_list[tool_idx], probs[tool_idx], [tool_name_list[i] for i in top_k_indices]

    @torch.no_grad()
    def tool_selection_with_query(self, dataset_name, query):
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = True
        input_ids_list = self.tokenizer(query, add_special_tokens=True).input_ids
        model_output = self.foundation_model(input_ids=torch.tensor([input_ids_list]), output_hidden_states=True)
        hidden_states = model_output.hidden_states[-1]
        if self.config.cal_seq:
            tool_name, _, top_k_tool_name_list = self.tool_selection(dataset_name, hidden_states)
        else:
            tool_name, _, top_k_tool_name_list = self.tool_selection(dataset_name, hidden_states[0][-1].unsqueeze(dim=0).unsqueeze(dim=0))
        return tool_name, top_k_tool_name_list

    def parameter_selection(self):
        pass

    @torch.no_grad()
    def generate_tool_calling_with_query(self, query, max_length=200): # 返回参数调用字符串 "(...)"
        def find_tool_calling_parameters(input_text): # 输入为 <tool name>(|......)去掉开头的部分，直接从参数开始
            bracket_signal = 1
            quote_signal = ''
            end_idx = -1
            for idx in range(len(input_text)):
                if quote_signal != '':
                    if input_text[idx] == quote_signal:
                        quote_signal = ''
                else:
                    if input_text[idx] in ["'", '"']:
                        quote_signal = input_text[idx]
                    else:
                        if input_text[idx] == '(':
                            bracket_signal += 1
                        elif input_text[idx] == ')':
                            bracket_signal -= 1
                        if bracket_signal == 0:
                            end_idx = idx + 1
                            break
            if bracket_signal == 0:
                return True, '('+input_text[:end_idx]
            else:
                return False, ''

        temperature = self.config.temperature
        top_p = self.config.top_p

        past_kv = None

        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = False
        input_token_list = self.tokenizer(query,add_special_tokens=True).input_ids
        generated_token_num = 0
        while generated_token_num < max(max_length, len(query)):
            if past_kv != None:
                model_output = self.foundation_model(input_ids=torch.tensor([[input_token_list[-1]]]), past_key_values=past_kv, use_cache=True)
            else:
                model_output = self.foundation_model(input_ids=torch.tensor([input_token_list]), use_cache=True)
            past_kv = model_output.past_key_values
            logits = model_output.logits[0][-1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_id = sample_top_p(probs, top_p)
            else:
                next_token_id = torch.argmax(logits, dim=-1).tolist()
            if next_token_id == self.tokenizer.eos_token_id:
                break
            else:
                input_token_list.append(next_token_id)
                generated_token_num += 1
                if generated_token_num%10==0:
                    output_text = self.tokenizer.decode(input_token_list, skip_special_tokens=True)[len(query):]
                    end_flag, parameter_text = find_tool_calling_parameters(output_text)
                    if end_flag:
                        return parameter_text
        output_text = self.tokenizer.decode(input_token_list, skip_special_tokens=True)[len(query):]
        end_flag, parameter_text = find_tool_calling_parameters(output_text)
        if end_flag:
            return parameter_text
        else:
            return "()"

    @torch.no_grad()
    def generate_text_basic(self, input_text, max_length=100):
        temperature = self.config.temperature
        top_p = self.config.top_p

        generate_flag = True
        generated_token_num = 0
        tool_judge_signal = False
        tool_judge_prob = 0
        eos_signal = False
        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = False
        input_token_list = self.tokenizer(input_text,add_special_tokens=True).input_ids

        while generate_flag and (generated_token_num < max_length):
            model_output = self.foundation_model(input_ids=torch.tensor([input_token_list]), output_hidden_states=True)
            logits = model_output.logits[0][-1]
            last_token_hidden_states = model_output.hidden_states[-1][0][-1]
            tool_judge_signal, tool_judge_prob = self.tool_judging(last_token_hidden_states.to(next(self.tool_judge.parameters()).device))
            # * 判断是否调用工具 
            if tool_judge_signal:
                generate_flag = False
                break
            # * 不掉用工具，生成下一个token 
            else:
                # 随机性采样
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token_id = sample_top_p(probs, top_p)
                # 确定性生成
                else:
                    next_token_id = torch.argmax(logits, dim=-1).tolist()
                # * 判断是否生成终止符</s> 
                if next_token_id == self.tokenizer.eos_token_id:
                    generate_flag = False
                    eos_signal = True
                    break
                # * 正常生成token，继续 
                else:
                    input_token_list.append(next_token_id)
                    generated_token_num += 1

        output_text = self.tokenizer.decode(input_token_list, skip_special_tokens=True)
        # * 停止输出条件：需要调用工具 ； 生成结束符 ； 达到长度上限
        return {"output_text": output_text, 
                "tool_judge_signal": tool_judge_signal, 
                "tool_judge_prob": tool_judge_prob, 
                "eos_signal": eos_signal}

    @torch.no_grad()
    def generate_text_in_detailed_prompt(self, prompt, prompt_tool, question_text, answer_text, max_length=500, allow_instant_call=True, start_with_limit=False, disabled_token=[]):
        temperature = self.config.temperature
        top_p = self.config.top_p

        output_answer_text = answer_text

        generated_token_num = 0
        tool_judge_signal = False
        eos_signal = False

        self.tokenizer.add_bos_token = True
        self.tokenizer.add_eos_token = False
        input_token_list = self.tokenizer(prompt.format(question_text, output_answer_text),add_special_tokens=True).input_ids

        dot_token_id = self.tokenizer.get_vocab()["."]

        past_kv = None

        start_flag = True

        generate_flag = True
        while generate_flag and (generated_token_num < max_length):
            tool_judge_signal = self.tool_judging_with_query(prompt_tool.format(question_text, output_answer_text.replace("$","")))
            # * 判断是否调用工具 
            if tool_judge_signal and allow_instant_call:
                    generate_flag = False
                    break
            elif tool_judge_signal and output_answer_text!="": # * 回答开头不调用工具，先生成thought
                    generate_flag = False
                    break
            # * 不掉用工具，生成下一个token 
            else:
                if past_kv != None:
                    model_output = self.foundation_model(input_ids=torch.tensor([[input_token_list[-1]]]), past_key_values=past_kv, use_cache=True)
                else:
                    model_output = self.foundation_model(input_ids=torch.tensor([input_token_list]), use_cache=True)
                past_kv = model_output.past_key_values
                logits = model_output.logits[0][-1]
                # 随机性采样
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token_id = sample_top_p(probs, top_p)
                # 确定性生成
                else:
                    next_token_id = torch.argmax(logits, dim=-1).tolist()
                    if start_flag:
                        if start_with_limit:
                            top_k_num = max(10,2*len(disabled_token))
                            top_k_values, top_k_indices = torch.topk(logits, top_k_num)
                            top_k_idx = 0
                            while(True):
                                if top_k_indices[top_k_idx] not in disabled_token:
                                    break
                                top_k_idx += 1
                            next_token_id = top_k_indices[top_k_idx]
                        if next_token_id != dot_token_id:
                            start_flag = False
                # * 判断是否生成终止符</s> 
                if next_token_id == self.tokenizer.eos_token_id:
                    generate_flag = False
                    eos_signal = True
                    break
                # * 正常生成token，继续 
                else:
                    input_token_list.append(next_token_id)
                    output_answer_text = self.tokenizer.decode(input_token_list, skip_special_tokens=True)[len(prompt.format(question_text, "")):]
                    generated_token_num += 1

        # * 停止输出条件：需要调用工具 ； 生成结束符 ； 达到长度上限
        return {"answer_text": output_answer_text, 
                "tool_judge_signal": tool_judge_signal, 
                "eos_signal": eos_signal}

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token[0].tolist()


# == == == ==
import math
from einops import rearrange, repeat, einsum

class MambaBlock(nn.Module):
    def __init__(self,
                 d_model,
                 bias = False,
                 d_conv = 4,
                 conv_bias = True,
                 dt_rank = "auto",
                 d_state = 16,
                 expand = 2
                 ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y