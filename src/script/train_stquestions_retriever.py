from WillMindS.config import Config
from WillMindS.log import Log_Init

from model import LLM_with_tools
from train import tool_judge_train, tool_retriever_train
from inference import *


def main(config, logger):
    model = LLM_with_tools(config)
    model.tokenizer.pad_token = model.tokenizer.eos_token # LLaMA2 Mistral

    dataset_dir_dict = {"SimpleQuestionsv2":config.dataset_dir["SimpleQuestionsv2"]}

    # * 加载工具库 
    model.load_tool_database(dataset_dir_dict)
    model.calculate_database([dataset_name for dataset_name in dataset_dir_dict])

    # * 训练retriever 
    tool_retriever_train(config, logger, model, dataset_dir_dict, mode="train+test")


if __name__ == "__main__":
    config = Config()
    logger = Log_Init(config)
    config.log_print_config(logger)

    main(config, logger)