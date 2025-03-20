from WillMindS.model.llms.ChatGPT import ChatGPT
from WillMindS.model.llms.ChatGLM import ChatGLM
from WillMindS.model.llms.MOSS import MOSS
from WillMindS.model.llms.ChatYuan import ChatYuan
from WillMindS.model.llms.CPM_Bee import CPM_Bee
from WillMindS.model.llms.LLaMA import LLaMA
# from WillMindS.model.llms.LLaMA_LoRA import LLaMA_LoRA
from WillMindS.model.llms.Baichuan import Baichuan
from WillMindS.model.llms.InternLM import InternLM
from WillMindS.model.llms.Qwen import Qwen

__all__ = [
    "Auto_Model",
    "ChatGPT",
    "ChatGLM",
    "MOSS",
    "ChatYuan",
    "CPM_Bee",
    "LLaMA",
    "LLaMA_LoRA",
    "Baichuan",
    "InternLM",
    "Qwen"
]

def Auto_Model(model_name,model_path,distribute=False):
    match model_name:
        case "ChatGLM":
            return ChatGLM(model_path,distribute=distribute)
        case "MOSS":
            return MOSS(model_path)
        case "ChatYuan":
            return ChatYuan(model_path)
        case "CPM_Bee":
            return CPM_Bee(model_path)
        case "LLaMA":
            return LLaMA(model_path)
        # case "LLaMA_LoRA":
        #     return LLaMA_LoRA(model_path)
        case "Baichuan":
            return Baichuan(model_path)
        case "InternLM":
            return InternLM(model_path)
        case "Qwen":
            return Qwen(model_path)