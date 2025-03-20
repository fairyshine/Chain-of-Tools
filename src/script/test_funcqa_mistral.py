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
        model.sl_judge(config.judge_checkpoint_dir, "load")
        model.sl_retriever(config.retriever_checkpoint_dir, "load")

    dataset_dir_dict = {"funcqa":config.dataset_dir["funcqa"]}

    # * 加载工具库 
    model.load_tool_database(dataset_dir_dict)
    model.calculate_database([dataset_name for dataset_name in dataset_dir_dict])

    # * 评测 
    infer_with_tools(config, logger, model, "funcqa", "./data/funcqa/test_oh.jsonl")
    infer_with_tools(config, logger, model, "funcqa", "./data/funcqa/test_mh.jsonl")


def multiple_run(config, logger):
    model = LLM_with_tools(config)
    model.tokenizer.pad_token = model.tokenizer.eos_token # LLaMA2专供
    model.sl_judge("./output/Mistral_2024-06-09 02:06:38/epoch_3/", "load")
    checkpoint_list = ["./output/Mistral_2024-06-08 16:13:29/epoch_{}/".format(epoch) for epoch in range(3,21)]
    for checkpoint in checkpoint_list:
        logger.info(checkpoint)
        model.sl_retriever(checkpoint, "load")
        model.load_tool_database(config.dataset_dir)
        model.calculate_database(["funcqa"])
        if config.tensor_weighting and config.tensor_filtering:
            model.get_tensor_filtering_index()
        infer_with_tools(config, logger, model, "funcqa", "./data/funcqa/test_oh.jsonl")
        infer_with_tools(config, logger, model, "funcqa", "./data/funcqa/test_mh.jsonl")

if __name__ == "__main__":
    config = Config()
    logger = Log_Init(config)
    config.log_print_config(logger)

    # main(config, logger)

    multiple_run(config, logger)