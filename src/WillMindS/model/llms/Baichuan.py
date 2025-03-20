# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# class Baichuan():  #1代
#     def __init__(self,checkpoint_path=''):
#         self.checkpoint_path = checkpoint_path

#         self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
#         self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, trust_remote_code=True).half().cuda()
#         self.model_name = 'Baichuan'
#         self.model.eval()

#     def answer(self,input_text):
#         try:
#             inputs = self.tokenizer(input_text, return_tensors="pt").to('cuda')
#             generate_ids = self.model.generate(**inputs, max_new_tokens=4096,repetition_penalty=1.1)
#             output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#             return output, output[len(input_text):]
#         except:
#             output = ''
#             return output, output

#     def chat(self):
#         pass


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
class Baichuan():  #2代
    def __init__(self,checkpoint_path=''):
        self.checkpoint_path = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True).half().cuda()
        self.model.generation_config = GenerationConfig.from_pretrained(self.checkpoint_path)
        self.model_name = 'Baichuan'
        self.model.eval()

    def answer(self,input_text):
        # try:
        messages =[{"role": "user", "content": input_text}]
        response = self.model.chat(self.tokenizer, messages)
        return response, response
        # except:
        #     output = ''
        #     return output, output

    def chat(self):
        pass


# from vllm import LLM
# class Baichuan():  #2代
#     def __init__(self,checkpoint_path=''):
#         self.checkpoint_path = checkpoint_path
#         self.model = LLM(model=self.checkpoint_path, trust_remote_code=True, max_num_batched_tokens=4096, dtype="float16")  # Name or path of your model
#         self.model_name = 'Baichuan'
#         # self.model.eval()

#     def answer(self,input_text):
#         output = self.model.generate(input_text)
#         return str(output), output[0].outputs[0].text

#     def chat(self):
#         pass