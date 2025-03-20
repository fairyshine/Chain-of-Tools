from transformers import AutoTokenizer, AutoModelForCausalLM

class CPM_Bee():
    def __init__(self,checkpoint_path=''):
        self.checkpoint_path = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, trust_remote_code=True).half().cuda()
        self.model_name = 'CPM_Bee'
        self.model.eval()

    def answer(self,input_text):
        try:
            result = self.model.generate({"input": input_text, "<ans>": ""}, self.tokenizer)
            output = result[0]["<ans>"]
            return output, output
        except:
            output = ''
            return output, output

    def chat(self):
        pass