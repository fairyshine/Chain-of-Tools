import heapq

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint_path = "/public/home/jhfang/mswu_wlchen/PTM/Llama-2-7b-chat-hf"

def main():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,device_map="auto").half().cuda()

    text = "What are the main topic of Declaration of the Clergy of France?"
    # text = "A pizza parlor has 5935 pizzas and each pizza can be cut into 3583 slices. If each person can have only one slice, how many slices of pizza will be left?"

    output, probs = generate(tokenizer,model,text,500)
    print("output: ", output)
    print("probs: ",probs)
    prob_list = [[ inform for t_inform in token_inform for inform in t_inform] for token_inform in probs]
    df=pd.DataFrame(prob_list)
    df.to_excel('result.xlsx', index=True)

def generate(tokenizer, model, raw_text, max_len):
    # 初始化
    SOFTMAX = torch.nn.Softmax(dim=0)
    input_text = raw_text
    probs = list()
    raw_input_id_list = tokenizer(raw_text).input_ids

    # 判断token长度是否超过最大长度限制
    if max_len <= len(raw_input_id_list):
        return input_text, None
    
    # 主循环体，逐token生成
    for pos in range(len(raw_input_id_list),max_len):
        tokenized = tokenizer(input_text, return_tensors="pt")
        generated = model(tokenized.input_ids.cuda(), output_hidden_states=True)
        logits_last_token_list = generated.logits[0][-1].tolist()
        softmax_logits_last_token_list = SOFTMAX(generated.logits[0][-1]).tolist()
        new_token_id = logits_last_token_list.index(max(logits_last_token_list))
        new_text = tokenizer.decode([new_token_id])
        input_text = tokenizer.decode(tokenized.input_ids[0][1:].tolist()+[new_token_id])

        largest_logits = heapq.nlargest(5, logits_last_token_list)
        # 找到这些元素的索引 
        largest_indexs = [logits_last_token_list.index(num) for num in largest_logits]
        largest_softmax_logits = [softmax_logits_last_token_list[index] for index in largest_indexs]
        largest_tokens = [tokenizer.decode([index]) for index in largest_indexs]
        probs.append([[new_text], largest_tokens, largest_logits, largest_softmax_logits])

        if "</s>" in input_text:
            break
    return input_text, probs
 
def debug():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,device_map="auto").cuda()

    text = "What work has strongly inspired The Secret of Queen Anne or Musketeers Thirty Years After?"

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(inputs.input_ids.cuda(),output_hidden_states=True)
    logits = model.lm_head(outputs.hidden_states[-1])
    print("inputs.input_ids:", inputs.input_ids.size())
    print("outouts.logits:", outputs.logits.size())
    print("outouts.hidden_states:", outputs.hidden_states[-1].size())
    print("logits:",logits.size())
    print(outputs.logits[0][0])
    print(logits[0][0])
    ''' 返回结果
    inputs.input_ids: torch.Size([1, 21])
    outouts.logits: torch.Size([1, 21, 32000])
    outouts.hidden_states: torch.Size([1, 21, 4096])
    logits: torch.Size([1, 21, 32000])
    tensor([ 0.1039, -0.2220,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
        device='cuda:0', grad_fn=<SelectBackward0>)
    tensor([ 0.1039, -0.2220,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
        device='cuda:0', dtype=torch.float16, grad_fn=<SelectBackward0>)
    '''

if __name__ == "__main__":
    main()