from transformers import AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

class LLaMA_LoRA():
    def __init__(self,checkpoint_path=''):
        self.checkpoint_path = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained("/public/home/jhfang/mswu_wlchen/PTM/Llama-2-7b-hf")
        self.base_model = LlamaForCausalLM.from_pretrained("/public/home/jhfang/mswu_wlchen/PTM/Llama-2-7b-hf").half().cuda()
        peft_config = LoraConfig(inference_mode=True,
                        r=8, 
                        lora_alpha=32, 
                        lora_dropout=0.1)
        self.model = get_peft_model(self.base_model, peft_config)
        self.model = self.model.from_pretrained(self.base_model, self.checkpoint_path)
        self.model_name = 'LLaMA_LoRA'
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
