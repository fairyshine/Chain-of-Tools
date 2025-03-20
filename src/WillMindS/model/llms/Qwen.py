from transformers import AutoTokenizer, AutoModelForCausalLM

class Qwen():
    def __init__(self,checkpoint_path=''):
        self.checkpoint_path = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, trust_remote_code=True).half().cuda()
        self.model_name = 'Qwen'
        self.model.eval()

    def answer(self,input_text):
        try:
            output, _ = self.model.chat(self.tokenizer, input_text, history=[])
        except:
            output = ''
        return output, output

    def chat(self):
        pass