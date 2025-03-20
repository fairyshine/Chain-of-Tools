import os

from WillMindS.utils.io import read_JSON, write_JSON


def process_gsm8k_xl_tools(input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    tool_dataset = {}

    tool_dict = read_JSON(input_dir_path+"func_dict.json")

    tool_information = {}
    tool_information["<add>"] = {
        "tool_name": "<add>",
        "tool_description": "This tool adds two or more numbers together and returns the sum."
    }
    tool_information["<subtract>"] = {
        "tool_name": "<subtract>",
        "tool_description": "This tool subtracts one or more numbers from the first number provided, returning the difference."
    }
    tool_information["<multiply>"] = {
        "tool_name": "<multiply>",
        "tool_description": "This tool multiplies two or more numbers together, returning the product."
    }
    tool_information["<divide>"] = {
        "tool_name": "<divide>",
        "tool_description": "This tool divides the first number by the second number, returning the quotient. Take note of dividing by zero."
    }

    tool_template_id = {}
    template_list = []
    for name in os.listdir(input_dir_path+"template/"):
        with open(input_dir_path+f"template/{name}") as f:
            tool_name = name.split("_")[-1].replace(".txt", "")
            if tool_name in tool_dict:
                template_list.append(f.read())
                tool_template_id[tool_name] = [len(template_list)-1]

    tool_dataset["tool_dict"] = tool_dict
    tool_dataset["tool_information"] = tool_information
    tool_dataset["tool_template_id"] = tool_template_id
    tool_dataset["template_list"] = template_list
    write_JSON(output_dir_path+"tool.json", tool_dataset, indent=4)

def process_funcqa_tools(input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    tool_dataset = {}

    tool_dict = read_JSON(input_dir_path+"func_dict.json")

    tool_information = {}
    tool_information["<add>"] = {
        "tool_name": "<add>",
        "tool_description": "This tool adds two or more numbers together and returns the sum."
    }
    tool_information["<subtract>"] = {
        "tool_name": "<subtract>",
        "tool_description": "This tool subtracts one or more numbers from the first number provided, returning the difference."
    }
    tool_information["<multiply>"] = {
        "tool_name": "<multiply>",
        "tool_description": "This tool multiplies two or more numbers together, returning the product."
    }
    tool_information["<divide>"] = {
        "tool_name": "<divide>",
        "tool_description": "This tool divides the first number by the second number, returning the quotient. Take note of dividing by zero."
    }
    tool_information["<power>"] = {
        "tool_name": "<power>",
        "tool_description": "This tool raises a base number to the exponent provided, returning the power."
    }
    tool_information["<sqrt>"] = {
        "tool_name": "<sqrt>",
        "tool_description": "This tool calculates the square root of a number, returning the principal square root."
    }
    tool_information["<log>"] = {
        "tool_name": "<log>",
        "tool_description": "This tool calculates the logarithm of a number to a specified base, returning the log value."
    }
    tool_information["<ln>"] = {
        "tool_name": "<ln>",
        "tool_description": "This tool calculates the natural logarithm (base e) of a number, returning the value of ln."
    }
    tool_information["<lcm>"] = {
        "tool_name": "<lcm>",
        "tool_description": "This tool finds the least common multiple of two or more numbers, returning the smallest common multiple."
    }
    tool_information["<gcd>"] = {
        "tool_name": "<gcd>",
        "tool_description": "This tool calculates the greatest common divisor of two or more numbers, returning the largest common divisor."
    }
    tool_information["<remainder>"] = {
        "tool_name": "<remainder>",
        "tool_description": "This tool divides one number by another and returns the remainder of the division."
    }
    tool_information["<choose>"] = {
        "tool_name": "<choose>",
        "tool_description": "This tool calculates the number of ways to choose k items from n items without repetition and without order."
    }
    tool_information["<permutate>"] = {
        "tool_name": "<permutate>",
        "tool_description": "This tool calculates the number of ways to arrange n items into a sequence where order matters."
    }

    tool_template_id = {}
    template_list = []
    for name in os.listdir(input_dir_path+"template_oh/"):
        with open(input_dir_path+f"template_oh/{name}") as f:
            tool_name = name.split("_")[-1].replace(".txt", "")
            if tool_name in tool_dict:
                    template_list.append(f.read())
                    tool_template_id[tool_name] = [len(template_list)-1]

    for name in os.listdir(input_dir_path+"template_mh/"):
        with open(input_dir_path+f"template_mh/{name}") as f:
            tool_name = name.split("_")[-1].replace(".txt", "")
            if tool_name in tool_dict:
                    template_list.append(f.read())
                    tool_template_id[tool_name].append(len(template_list)-1)

    tool_dataset["tool_dict"] = tool_dict
    tool_dataset["tool_information"] = tool_information
    tool_dataset["tool_template_id"] = tool_template_id
    tool_dataset["template_list"] = template_list
    write_JSON(output_dir_path+"tool.json", tool_dataset, indent=4)

def process_kamel_tools(input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    tool_dataset = {}

    tool_dict = {}
    raw_tool_dict = read_JSON(input_dir_path+"func_dict.json")
    for raw_tool in raw_tool_dict:
        tool_dict["<"+raw_tool+">"] = raw_tool_dict[raw_tool]

    tool_information = {}
    raw_tool_information = read_JSON(input_dir_path+"api_desc.json")
    for raw_tool in raw_tool_information:
        tool_information["<"+raw_tool+">"] = {
            "tool_name": "<"+raw_tool+">",
            "tool_description": raw_tool_information[raw_tool]
        }

    tool_template_id = {}
    template_list = []
    for name in ["kamel_first_30.txt", "kamel_first_ood30.txt", "kamel_first_ood60.txt", "kamel_general.txt"]:
        with open(input_dir_path+f"template/{name}") as f:
                template_list.append(f.read())

    for tool_name in tool_information:
        tool_template_id[tool_name] = [0,1,2,3]

    tool_dataset["tool_dict"] = tool_dict
    tool_dataset["tool_information"] = tool_information
    tool_dataset["tool_template_id"] = tool_template_id
    tool_dataset["template_list"] = template_list
    write_JSON(output_dir_path+"tool.json", tool_dataset, indent=4)

def process_vh_tools(input_dir_path, output_dir_path):
    # 初始化
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    tool_dataset = {}

    tool_dict = read_JSON(input_dir_path+"func_dict.json")

    tool_information = {}

    tool_template_id = {}
    template_list = []
    with open(input_dir_path+"template/vh_special_v4.txt") as f:
            template_list.append(f.read())

    for tool_name in tool_information:
        tool_template_id[tool_name] = [0]

    tool_dataset["tool_dict"] = tool_dict
    tool_dataset["tool_information"] = tool_information
    tool_dataset["tool_template_id"] = tool_template_id
    tool_dataset["template_list"] = template_list
    write_JSON(output_dir_path+"tool.json", tool_dataset, indent=4)



if __name__ == "__main__":
    input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/gsm8k-xl/"
    output_dir_path = "./data/gsm8k_xl/"
    process_gsm8k_xl_tools(input_dir_path, output_dir_path)

    input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/funcqa/"
    output_dir_path = "./data/funcqa/"
    process_funcqa_tools(input_dir_path, output_dir_path)

    input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/kamel/"
    output_dir_path = "./data/kamel/"
    process_kamel_tools(input_dir_path, output_dir_path)

    input_dir_path = "./data/raw_dataset_in_each_format/ToolkenGPT/vh/"
    output_dir_path = "./data/vh/"
    process_vh_tools(input_dir_path, output_dir_path)