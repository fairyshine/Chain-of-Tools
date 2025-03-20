import re

from rich.progress import track
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM

from WillMindS.config import Config
from WillMindS.log import Log_Init
from WillMindS.utils.io import read_JSON, write_JSON

from tool_hub.arithmetic import *  # custom_round

# prompt_gsm8k_xl_zero_shot = '''Solve the following math problem step by step, and then provide the final answer in the format: ‘So, the answer is xxx.’

# Question: [QUESTION]
# Answer:'''

prompt_gsm8k_xl_zero_shot = '''Solve the following math problem step by step, and then provide the final answer in the format: ‘So, the answer is xxx.’

[QUESTION]
'''

prompt_gsm8k_xl_CoT = '''Answer the following questions step by step.

Question: Mark has 3 tanks for pregnant fish. Each tank has 4 pregnant fish and each fish gives birth to 20 young. How many young fish does he have at the end?
Answer: He has 4*3=12 pregnant fish They give birth to 12*20=240 fish #### 240

Question: The math questions in a contest are divided into three rounds: easy, average, and hard. There are corresponding points given for each round. That is 2, 3, and 5 points for every correct answer in the easy, average, and hard rounds, respectively. Suppose Kim got 6 correct answers in the easy; 2 correct answers in the average; and 4 correct answers in the difficult round, what are her total points in the contest?
Answer: Kim got 6 points/round x 2 round = 12 points in the easy round. She got 2 points/round x 3 rounds = 6 points in the average round. She got 4 points/round x 5 rounds = 20 points in the difficult round. So her total points is 12 points + 6 points + 20 points = 38 points. #### 38

Question: A clothing store sells 20 shirts and 10 pairs of jeans. A shirt costs $10 each and a pair of jeans costs twice as much. How much will the clothing store earn if all shirts and jeans are sold?
Answer: Twenty shirts amount to $10 x 20 = $200. The cost of each pair of jeans is $10 x 2 = $20. So 10 pairs of jeans amount to $20 x 10 = $200. Therefore, the store will earn $200 + $200 = $400 if all shirts and jeans are sold. #### 400

Question: Arnold’s collagen powder has 18 grams of protein for every 2 scoops. His protein powder has 21 grams of protein per scoop. And his steak has 56 grams of protein. If he has 1 scoop of collagen powder, 1 scoop of protein powder and his steak, how many grams of protein will he consume?
Answer: 2 scoops of collagen powder have 18 grams of protein and he only has 1 scoop so he consumes 18/2 = 9 grams of protein He has 9 grams collagen powder, 21 grams of protein powder and 56 grams in his steak for a total of 9+21+56 = 86 grams of protein #### 86

Question: [QUESTION]
Answer:'''

# prompt_funcqa_oh_zero_shot = '''Solve the following math problem, and then provide the final answer in the format: ‘So, the answer is xxx.’

# Question: [QUESTION]
# Answer:'''

prompt_funcqa_oh_zero_shot = '''Solve the following math problem, and then provide the final answer in the format: ‘So, the answer is xxx.’

[QUESTION]
'''

prompt_funcqa_oh_CoT = '''Q: If Amy’s income increases by 4% annually, how many times will it multiply in 11 years?
A: In 11 years, Amy’s income will increase by 1.04^11=1.54 times. So, the answer is 1.54.

Q: If a store sells 147 bananas today and 354 more bananas tomorrow, how many bananas does the store sell in total?
A: The store sells 147 bananas today and 354 more bananas tomorrow, so the total number of bananas sold is 147+354=501. So, the answer is 501.

Q: A man had 789.4 dollars in his wallet. He spent 11.99 dollars on a movie ticket. How much money does he have left now?
A: The man had 789.4 dollars in his wallet and spent 11.99 dollars on a movie ticket, so he has 789.4-11.99=777.41 dollars left. So, the answer is 777.41 dollars.

Q: If a cake weighs 3.77 pounds and is divided into 13 equal pieces, how much does each piece weight?
A: Each piece of the cake weighs 3.77/13=0.29 pounds. So, the answer is 0.29 pounds.

Q: [QUESTION]
A:'''

# prompt_funcqa_mh_zero_shot = '''Solve the following math problem step by step, and then provide the final answer in the format: ‘So, the answer is xxx.’

# Question: [QUESTION]
# Answer:'''

prompt_funcqa_mh_zero_shot = '''Solve the following math problem step by step, and then provide the final answer in the format: ‘So, the answer is xxx.’

[QUESTION]
'''

prompt_funcqa_mh_CoT = '''Answer the following questions step by step:

Question: A coin is tossed 8 times, what is the probability of getting exactly 7 heads ?
Answer: The total number of possible outcomes to toss a coin 8 times is 2^8=256. The number of ways of getting exactly 7 heads is 8C7=8. The probability of getting exactly 7 heads is 8/256=0.03125. #### 0.03125

Question: If paint costs $3.2 per quart, and a quart covers 12 square feet, how much will it cost to paint the outside of a cube 10 feet on each edge?
Answer: The total surface area of the 10 ft cube is 6*10^2=6*100=600 square feet. The number of quarts needed is 600/12=50. The cost is 50*3.2=160. #### 160

Question: log(x)=2, log(y)=0.1, what is the value of log(x-y) ?
Answer: log(x)=2, so x=10^2=100; log(y)=0.1, so y=10^0.1=1.26; x-y=100-1.26=98.74, so log(x-y)=log(98.74)=1.99. #### 1.99

Question: How many degrees does the hour hand travel when the clock goes 246 minutes?
Answer: The hour hand travels 360 degrees in 12 hours, so every hour it travels 360/12=30 degrees. 246 minutes is 246/60=4.1 hours. The hour hand travels 4.1*30=123 degrees. #### 123

Question: [QUESTION]
Answer:'''

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
            generate_ids = self.model.generate(inputs.input_ids.cuda(), max_new_tokens=512)#2048
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return output
        except Exception as e:
            logger.info("生成时报错: " + str(e))
            output = ''
            return output

class Mistral():
    def __init__(self,checkpoint_path=''):
        self.checkpoint_path = checkpoint_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, trust_remote_code=True).half().cuda()
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, trust_remote_code=True, device_map="auto")
        self.model_name = 'Mistral'
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer(self,input_text):
        try:
            inputs = self.tokenizer([input_text], return_tensors="pt").to("cuda")
            generate_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=True)#2048
            output = self.tokenizer.batch_decode(generate_ids)[0]
            return output
        except Exception as e:
            logger.info("生成时报错: " + str(e))
            output = ''
            return output

def infer_arithmetic(logger, model, dataset_name, test_dataset, prompt, question_type, answer_type, output_name):
    def parse_answer(answer, pattern:str="####"):
        if pattern=="####":
            answer = answer.split("####")[-1]
            answer = answer.strip().strip("\n").strip('\\n')
            # 32,333 -> 32333
            answer = answer.replace(",", "")

            # get the last number
            try:
                answer = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", answer)[-1]
            except:
                answer = 0
        elif pattern=="answer is":
            answer = answer.split("answer is")[-1]
            answer = answer.strip().strip("\n").strip('\\n')

            # 32,333 -> 32333
            answer = answer.replace(",", "")

            # get the last number
            try:
                answer = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", answer)[-1]
            except:
                answer = 0
        return answer
    def accuracy(pred, true, type = "exact"):
        if len(pred) < len(true):
            true = true[:len(pred)]

        correct = 0
        for p, t in zip(pred, true):
            try:
                if type == "exact":
                    if float(p) == float(t):
                        correct += 1
                elif type == "round":
                    if round(float(p), 2) == custom_round(float(t), 2):
                        correct += 1
                elif type == "approx":
                    # 1% error tolerance, e.g. 1000 -> 990 ~ 1010
                    if abs(float(p) - float(t)) <= abs(float(t)) * 0.001:
                        correct += 1
            except ValueError:
                pass
        
        return correct / len(pred)

    output_dataset = []
    logger.info("====== "+ output_name +" ======")
    for step, data in track(enumerate(test_dataset), description='Infering {}...'.format(dataset_name)):
        # 初始化
        output_data = {}
        question_text = data["question"]

        answer_text_full = model.answer(prompt.replace("[QUESTION]", question_text))
        answer_text = answer_text_full[len(prompt.replace("[QUESTION]", question_text)):]

        output_data["dataset_name"] = dataset_name
        output_data["case_idx"] = data["case_idx"]
        output_data["question"] = question_text
        output_data["generated_answer"] = answer_text
        logger.info("case_idx: "+str(output_data["case_idx"])+"  generated_answer: "+output_data["generated_answer"])

        answer_text_split = answer_text.split(question_type)[0]
        output_data["pred_result"] = parse_answer(answer_text_split, pattern=answer_type)

        output_data["gold_result"] = data["result"]
        output_dataset.append(output_data)

        write_JSON(config.model_dir+"/"+output_name+"_result.jsonl",output_dataset)

    pred_list = [data["pred_result"] for data in output_dataset]
    gold_list = [data["gold_result"] for data in output_dataset]
    logger.info("Round Accuracy: "+str(accuracy(pred_list, gold_list, type="round")))
    logger.info("Approx Accuracy: "+str(accuracy(pred_list, gold_list, type="approx")))

def remove_english_chars(string):
    new_string = ""
    for char in string:
        if not char.isalpha():
            new_string += char
    return new_string


if __name__ == "__main__":
    config = Config()
    logger = Log_Init(config)
    config.log_print_config(logger)

    # checkpoint = "/public/home/jhfang/mswu_wlchen/PTM/Llama-2-7b-chat-hf/"
    checkpoint = "/public/home/jhfang/mswu_wlchen/PTM/Mistral-7B-Instruct-v0.2/"
    model = Mistral(checkpoint)

    test_dataset = read_JSON("./data/gsm8k_xl/test.jsonl")
    infer_arithmetic(logger, model, "gsm8k_xl", test_dataset, prompt_gsm8k_xl_zero_shot, "Question: ", "answer is", "prompt_gsm8k_xl_zero_shot")
    infer_arithmetic(logger, model, "gsm8k_xl", test_dataset, prompt_gsm8k_xl_CoT, "Question: ", "####", "prompt_gsm8k_xl_CoT")

    test_dataset = read_JSON("./data/funcqa/test_oh.jsonl")
    infer_arithmetic(logger, model, "funcqa", test_dataset, prompt_funcqa_oh_zero_shot, "Question: ", "answer is", "prompt_funcqa_oh_zero_shot")
    infer_arithmetic(logger, model, "funcqa", test_dataset, prompt_funcqa_oh_CoT, "Q: ", "answer is", "prompt_funcqa_oh_CoT")

    test_dataset = read_JSON("./data/funcqa/test_mh.jsonl")
    infer_arithmetic(logger, model, "funcqa", test_dataset, prompt_funcqa_mh_zero_shot, "Question: ", "answer is", "prompt_funcqa_mh_zero_shot")
    infer_arithmetic(logger, model, "funcqa", test_dataset, prompt_funcqa_mh_CoT, "Question: ", "####", "prompt_funcqa_mh_CoT")


