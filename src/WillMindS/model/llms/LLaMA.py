from transformers import AutoTokenizer, LlamaForCausalLM
class LLaMA():
    def __init__(self,checkpoint_path=''):
        self.checkpoint_path = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        # self.model = LlamaForCausalLM.from_pretrained(self.checkpoint_path, trust_remote_code=True).half().cuda()
        self.model = LlamaForCausalLM.from_pretrained(self.checkpoint_path, trust_remote_code=True, device_map="auto")
        self.model_name = 'LLaMA'
        self.model.eval()

    def answer(self,input_text):
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt")
            generate_ids = self.model.generate(inputs.input_ids.cuda(), max_new_tokens=256)#2048
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return output, output[len(input_text):]
        except:
            output = ''
            return output, output

    def chat(self):
        pass


# from vllm import LLM
# class LLaMA():
#     def __init__(self,checkpoint_path=''):
#         self.checkpoint_path = checkpoint_path
#         self.model = LLM(model=self.checkpoint_path, trust_remote_code=True, max_num_batched_tokens=4096, dtype="float16")  # Name or path of your model
#         self.model_name = 'LLaMA'
#         # self.model.eval()

#     def answer(self,input_text):
#         output = self.model.generate(input_text)
#         return str(output), output[0].outputs[0].text

#     def chat(self):
#         pass