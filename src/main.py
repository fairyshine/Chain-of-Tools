from WillMindS.config import Config
from WillMindS.log import Log_Init

from model import LLM_with_tools
from train import tool_judge_train, tool_retriever_train
from inference import *


def main(config, logger):
    model = LLM_with_tools(config)
    model.tokenizer.pad_token = model.tokenizer.eos_token # LLaMA2 Mistral

    # * 加载模型参数 
    if config.load_toolcalling_checkpoint:
        # model.sl_judge(config.judge_checkpoint_dir, "load")
        model.sl_retriever(config.retriever_checkpoint_dir, "load")

    # dataset_dir_dict = config.dataset_dir
    # dataset_dir_dict = {"gsm8k_xl":config.dataset_dir["gsm8k_xl"]}
    # dataset_dir_dict = {"funcqa":config.dataset_dir["funcqa"]}
    # dataset_dir_dict = {"gsm8k_xl":config.dataset_dir["gsm8k_xl"],
    #                     "funcqa":config.dataset_dir["funcqa"]}
    # dataset_dir_dict = {"kamel":config.dataset_dir["kamel"]}
    # dataset_dir_dict = {"kamel":"./data/kamel_GPT/"}
    # dataset_dir_dict = {"SimpleQuestionsv2":config.dataset_dir["SimpleQuestionsv2"]}
    dataset_dir_dict = {"SimpleQuestionsv2":"./data/SimpleToolQuestions_unseen/"}


    # * 加载工具库 
    model.load_tool_database(dataset_dir_dict)
    model.calculate_database([dataset_name for dataset_name in dataset_dir_dict])

    # * 训练judge 
    # tool_judge_train(config, logger, model, dataset_dir_dict, mode="train+test")
    # #judge一起训练
    # tool_judge_train(config, logger, model, config.dataset_dir, mode="train+test")
    # #保存judger
    # model.sl_judge(config.model_dir, "save")
    
    # * 训练retriever 
    # dataset_dir_without_vh = {key:config.dataset_dir[key] for key in config.dataset_dir if key!='vh'}
    # #retriever分开训练
    # for dataset_name in dataset_dir_without_vh:
    #     tool_retriever_train(config, logger, model, {dataset_name:dataset_dir_without_vh[dataset_name]}, mode="train+test")
    #retriever一起训练
    # tool_retriever_train(config, logger, model, dataset_dir_without_vh, mode="train+test")

    # tool_retriever_train(config, logger, model, dataset_dir_dict, mode="train+test")

    #保存retriever
    # model.sl_retriever(config.model_dir, "save")

    # * 评测 
    # infer_with_tools(config, logger, model, "gsm8k_xl", "./data/gsm8k_xl/test.jsonl")

    # infer_with_tools(config, logger, model, "funcqa", "./data/funcqa/test_oh.jsonl")
    # infer_with_tools(config, logger, model, "funcqa", "./data/funcqa/test_mh.jsonl")

    # model.tool_range = 30
    # infer_with_tools(config, logger, model, "kamel", "./data/kamel/test_first_30.jsonl")
    # model.tool_range = 60
    # infer_with_tools(config, logger, model, "kamel", "./data/kamel/test_first_60.jsonl")
    # model.tool_range = 100
    # infer_with_tools(config, logger, model, "kamel", "./data/kamel/test_first_100.jsonl")
    # model.tool_range = -1
    # infer_with_tools(config, logger, model, "kamel", "./data/kamel/test_first_234.jsonl")

    # model.tool_range = 200
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test_200.jsonl")
    # model.tool_range = 400
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test_400.jsonl")
    # model.tool_range = 600
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test_600.jsonl")
    # model.tool_range = 800
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test_800.jsonl")
    # model.tool_range = 999
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test.jsonl")
    # model.tool_range = -1
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test_ex.jsonl")

    # model.tool_range = 1099
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_100.jsonl")
    # model.tool_range = 1199
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_200.jsonl")
    # model.tool_range = 1299
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_300.jsonl")
    # model.tool_range = 1399
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_400.jsonl")
    # model.tool_range = 1499
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_500.jsonl")
    # model.tool_range = 1599
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_600.jsonl")
    # model.tool_range = 1699
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_700.jsonl")
    # model.tool_range = 1836
    # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_837.jsonl")

    model.tool_range = 100
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_100.jsonl")
    model.tool_range = 200
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_200.jsonl")
    model.tool_range = 300
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_300.jsonl")
    model.tool_range = 400
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_400.jsonl")
    model.tool_range = 500
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_500.jsonl")
    model.tool_range = 600
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_600.jsonl")
    model.tool_range = 700
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_700.jsonl")
    model.tool_range = 837
    infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions_unseen/test_unseen_837.jsonl")



def multiple_run(config, logger):
    model = LLM_with_tools(config)
    model.tokenizer.pad_token = model.tokenizer.eos_token # LLaMA2专供
    # model.sl_judge("./output/LLaMA_2024-05-19 22:15:15/epoch_1/", "load") # gsm8k
    # model.sl_judge("./output/judge_joint/", "load") # joint
    # checkpoint_list = ["./output/retriever_kamel_gold_mistral/epoch_{}/".format(epoch) for epoch in range(3,11)]
    # checkpoint_list = ["./output/retriever_kamel_gpt_mistral/epoch_{}/".format(epoch) for epoch in range(3,11)]
    checkpoint_list = ["./output/retriever_simplequestions_mistral/epoch_{}/".format(epoch) for epoch in range(3,11)]
    for checkpoint in checkpoint_list:
        logger.info(checkpoint)
        model.sl_retriever(checkpoint, "load")
        model.load_tool_database(config.dataset_dir)
        # model.calculate_database(["kamel"])
        model.calculate_database(["SimpleQuestionsv2"])
        # model.calculate_database(["gsm8k_xl"])
        if config.tensor_weighting and config.tensor_filtering:
            model.get_tensor_filtering_index()
        # infer_with_tools(config, logger, model, "kamel", "./data/kamel/test_first_234.jsonl")
        # infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test.jsonl")
        infer_with_tools(config, logger, model, "SimpleQuestionsv2", "./data/SimpleToolQuestions/test_ex.jsonl")
        # infer_with_tools(config, logger, model, "gsm8k_xl", "./data/gsm8k_xl/test.jsonl")

if __name__ == "__main__":
    config = Config()
    logger = Log_Init(config)
    config.log_print_config(logger)

    # main(config, logger)

    multiple_run(config, logger)