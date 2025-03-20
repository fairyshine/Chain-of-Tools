import re

from rich.progress import track

from WillMindS.utils.io import read_JSON, write_JSON

from prompt import prompt_gsm8k_xl_infer, prompt_gsm8k_xl_tool_mode, \
                    prompt_funcqa_infer_oh, prompt_funcqa_infer_mh, prompt_funcqa_tool_mode, \
                    prompt_kamel_infer, prompt_kamel_tool_mode, \
                    prompt_SQ_tool_mode
from tool_hub.arithmetic import *

def infer_with_tools(config, logger, model, dataset_name, test_file_path):
    model.eval()
    logger.info("============ Inferring {} ============".format(dataset_name))
    test_dataset = read_JSON(test_file_path)
    if dataset_name == "gsm8k_xl" or dataset_name == "funcqa":
        infer_arithmetic(config, logger, model, dataset_name, test_dataset, test_file_path)
    if dataset_name == "kamel":
        infer_kamel(config, logger, model, dataset_name, test_dataset, test_file_path)
    if dataset_name == "SimpleQuestionsv2":
        infer_SQ_retrieval_only(config, logger, model, dataset_name, test_dataset, test_file_path)


def infer_arithmetic(config, logger, model, dataset_name, test_dataset, test_file_path):
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

    def rm_comma_between_num_in_string(input_string):
        output_string = input_string
        for i in range(len(input_string)-2,1,-1):
            if output_string[i]== "," and output_string[i+1].isdigit() and output_string[i-1].isdigit():
                output_string = output_string[:i] + output_string[i+1:]
        return output_string

    if dataset_name == "gsm8k_xl":
        prompt_infer = prompt_gsm8k_xl_infer
        prompt_tool_mode = prompt_gsm8k_xl_tool_mode
    else:
        assert dataset_name == "funcqa"
        if "oh" in test_file_path:
            prompt_infer = prompt_funcqa_infer_oh
        else:
            assert "mh" in test_file_path
            prompt_infer = prompt_funcqa_infer_mh
        prompt_tool_mode = prompt_funcqa_tool_mode

    disabled_str_list = ["0","1","2","3","4","5","6","7","8","9"]
    disabled_token_list = []
    token_vocabulary = model.tokenizer.get_vocab()
    for str_in in disabled_str_list:
        if str_in in token_vocabulary:
            disabled_token_list.append(token_vocabulary[str_in])

    output_dataset = []
    for step, data in track(enumerate(test_dataset), description='Infering {}...'.format(dataset_name)):
        # 初始化
        output_data = {}
        question_text = data["question"]
        answer_text = ""
        tool_call_list = []
        tool_call_position = []
        tool_judge_signal = False

        tool_calling_error_signal = False


        # 生成回答
        generate_flag = True
        start_with_limit = False
        while(generate_flag):
            # 【模式：工具调用】
            if tool_judge_signal:
                # 检索所需工具
                tool_name, _ = model.tool_selection_with_query(dataset_name, prompt_tool_mode.format(question_text, answer_text))
                template_id = model.tool_database[dataset_name]["tool_template_id"][tool_name][0]
                template = model.tool_database[dataset_name]["template_list"][template_id]
                # 将之前的Tool Calling结果添加到Prompt中
                answer_text_with_tool = answer_text
                if tool_call_list:
                    bias = 0
                    for i in range(len(tool_call_position)):
                        start_pos = tool_call_position[i]
                        end_pos = start_pos + len(tool_call_list[i].split(" = ")[-1])
                        answer_text_with_tool = answer_text_with_tool[:start_pos+bias] +tool_call_list[i] + answer_text_with_tool[end_pos+bias:]
                        bias += len(tool_call_list[i]) - (end_pos - start_pos)
                tool_calling_prompt = template.replace("[QUESTION]", question_text)+answer_text_with_tool.rstrip().replace("$","")+" "+tool_name+"("
                # tool_calling_prompt = rm_comma_between_num_in_string(tool_calling_prompt)
                parameter_text = model.generate_tool_calling_with_query(tool_calling_prompt)
                status, tool_calling, tool_calling_result = call_arithmetic_tools(tool_name, parameter_text)
                if status:
                    tool_call_list.append(tool_calling)
                    answer_text += " " if (answer_text[-1]!=" " and answer_text[-1]!="$") else ""
                    tool_call_position.append(len(answer_text))
                    answer_text += tool_calling_result
                    tool_calling_error_signal = False
                    start_with_limit = True
                else:
                    if tool_calling_error_signal:
                        break
                    tool_calling_error_signal = True
            else:
                start_with_limit = False
            # 【模式：文本生成】
            if start_with_limit:
                generate_output = model.generate_text_in_detailed_prompt(prompt_infer, prompt_tool_mode, question_text, answer_text, allow_instant_call=False, start_with_limit=start_with_limit,disabled_token=disabled_token_list)
            else:
                generate_output = model.generate_text_in_detailed_prompt(prompt_infer, prompt_tool_mode, question_text, answer_text, allow_instant_call=False)
            answer_text = generate_output["answer_text"]
            # * 判断是否生成过多内容
            if ("Question: " in answer_text) or ("Q: " in answer_text):
                break
            tool_judge_signal = generate_output["tool_judge_signal"]
            if generate_output["eos_signal"]:
                generate_flag = False
            if not tool_judge_signal:
                generate_flag = False
            if len(answer_text) >= 1000:
                generate_flag = False

        output_data["dataset_name"] = dataset_name
        output_data["case_idx"] = data["case_idx"]
        output_data["question"] = question_text
        output_data["generated_answer"] = answer_text
        logger.info("case_idx: "+str(output_data["case_idx"])+"  generated_answer: "+output_data["generated_answer"])
        output_data["tool_calling_list"] = tool_call_list
        output_data["tool_calling_position"] = tool_call_position
        if dataset_name == "gsm8k_xl":
            answer_text_split = answer_text.split("Question: ")[0]
            output_data["pred_result"] = parse_answer(answer_text_split, pattern="####")
        if dataset_name == "funcqa":
            if "oh" in test_file_path:
                answer_text_split = answer_text.split("Q: ")[0]
                output_data["pred_result"] = parse_answer(answer_text_split, pattern="answer is")
            else:
                assert "mh" in test_file_path
                answer_text_split = answer_text.split("Question: ")[0]
                output_data["pred_result"] = parse_answer(answer_text_split, pattern="####")
        output_data["gold_result"] = data["result"]
        output_dataset.append(output_data)

        write_JSON(config.model_dir+dataset_name+test_file_path.split(".")[0].split("/")[-1]+"_result.jsonl",output_dataset)

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

def call_arithmetic_tools(tool_name, parameter_text):
    args = parameter_text.replace("((", "(").replace("))", ")").replace("$", "").replace("=","")
    if ", " in args:
        args = args.replace(", ", ";").replace(",", "").replace(";", ", ")
    args = args.replace(" ", "")
    # handle %
    if '%' in args:
        temp = args[1:-1].split(",")
        for arg_i, arg in enumerate(temp):
            # if have percentage, convert to decimal
            if "%" in arg:
                arg = remove_english_chars(arg.replace("%", "").split("/")[0].strip())
                arg = str(float(arg) / 100)
            temp[arg_i] = arg
        args = f"({','.join(temp)})"
    try:
        res = eval(f"{tool_name[1:-1]}_{args}")
        tool_calling = f"{tool_name}{args} = {res}"
        return True, tool_calling, res
    except Exception as e:
        return False, str(e), None


# def infer_kamel_retrieval_only(config, logger, model, dataset_name, test_dataset, test_file_path):
#     output_dataset = []
#     for step, data in track(enumerate(test_dataset), description='Infering {}...'.format(dataset_name)):
#         output_data = {}
#         question_text = data["question"]

#         # if data["result"][0].isdigit():
#         #     answer_text = " "
#         # else:
#         answer_text = ""
#         tool_name, top_k_tool_name_list = model.tool_selection_with_query(dataset_name, prompt_kamel_tool_mode.format(question_text, answer_text))

#         output_data["dataset_name"] = dataset_name
#         output_data["case_idx"] = data["case_idx"]
#         output_data["pred_tool"] = tool_name
#         output_data["pred_topk"] = top_k_tool_name_list
#         output_data["gold_tool"] = data["tool"]
#         output_dataset.append(output_data)

#         write_JSON(config.model_dir+dataset_name+str(len(test_dataset))+"_result.jsonl", output_dataset)

#     acc_top1_counter = 0
#     acc_topk_counter = 0
#     for data in output_dataset:
#         if data["pred_tool"] == "<"+data["gold_tool"]+">":
#             acc_top1_counter += 1
#         if "<"+data["gold_tool"]+">" in data["pred_topk"]:
#             acc_topk_counter += 1

#     logger.info("TOP1 ACC: "+str(acc_top1_counter/len(output_dataset)))
#     logger.info("TOPk ACC: "+str(acc_topk_counter/len(output_dataset)))


def infer_kamel(config, logger, model, dataset_name, test_dataset, test_file_path):
    output_dataset = []
    for step, data in track(enumerate(test_dataset), description='Infering {}...'.format(dataset_name)):
        # 初始化
        output_data = {}
        question_text = data["question"]
        answer_text = ""
        tool_judge_signal = False

        tool_name = ""
        top_k_tool_name_list = []

        # 生成回答
        generate_flag = True
        while(generate_flag):
            # 【模式：工具调用】
            if tool_judge_signal:
                # 检索所需工具
                tool_name, top_k_tool_name_list = model.tool_selection_with_query(dataset_name, prompt_kamel_tool_mode.format(question_text, answer_text))
                generate_flag = False
                break
            # 【模式：文本生成】
            generate_output = model.generate_text_in_detailed_prompt(prompt_kamel_infer, prompt_kamel_tool_mode, question_text, answer_text)
            answer_text = generate_output["answer_text"]
            tool_judge_signal = generate_output["tool_judge_signal"]
            if generate_output["eos_signal"]:
                generate_flag = False
            if not tool_judge_signal:
                generate_flag = False
            if len(answer_text) >= 1000:
                generate_flag = False

        output_data["dataset_name"] = dataset_name
        output_data["case_idx"] = data["case_idx"]
        output_data["pred_tool"] = tool_name
        output_data["pred_topk"] = top_k_tool_name_list
        output_data["gold_tool"] = data["tool"]
        output_data["generated_answer"] = answer_text
        output_dataset.append(output_data)

        write_JSON(config.model_dir+dataset_name+str(len(test_dataset))+"_result.jsonl", output_dataset)

    acc_top1_counter = 0
    acc_topk_counter = 0
    for data in output_dataset:
        if data["pred_tool"] == "<"+data["gold_tool"]+">":
            acc_top1_counter += 1
        if "<"+data["gold_tool"]+">" in data["pred_topk"]:
            acc_topk_counter += 1

    logger.info("TOP1 ACC: "+str(acc_top1_counter/len(output_dataset)))
    logger.info("TOPk ACC: "+str(acc_topk_counter/len(output_dataset)))

def infer_SQ_retrieval_only(config, logger, model, dataset_name, test_dataset, test_file_path):
    output_dataset = []
    for step, data in track(enumerate(test_dataset), description='Infering {}...'.format(dataset_name)):
        output_data = {}
        question_text = data["question"]

        # if data["result"][0].isdigit():
        #     answer_text = " "
        # else:
        answer_text = ""
        tool_name, top_k_tool_name_list = model.tool_selection_with_query(dataset_name, prompt_SQ_tool_mode.format(question_text, answer_text))

        output_data["dataset_name"] = dataset_name
        output_data["case_idx"] = data["case_idx"]
        output_data["pred_tool"] = tool_name
        output_data["pred_topk"] = top_k_tool_name_list
        output_data["gold_tool"] = data["tool"]
        output_dataset.append(output_data)

        write_JSON(config.model_dir+dataset_name+str(len(test_dataset))+"_result.jsonl", output_dataset)

    acc_top1_counter = 0
    acc_topk_counter = 0
    for data in output_dataset:
        if data["pred_tool"] == "<"+data["gold_tool"]+">":
            acc_top1_counter += 1
        if "<"+data["gold_tool"]+">" in data["pred_topk"]:
            acc_topk_counter += 1

    logger.info("TOP1 ACC: "+str(acc_top1_counter/len(output_dataset)))
    logger.info("TOPk ACC: "+str(acc_topk_counter/len(output_dataset)))