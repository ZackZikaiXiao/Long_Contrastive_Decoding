# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

import numpy as np
from torch import nn
from pathlib import Path
from tqdm import tqdm
import random
import json
import torch
import os
from transformers import AutoConfig
import re
import json
import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser(description='Generate probs')

# 设置默认的 model_path 和 data_path
parser.add_argument('--model_path', type=str, default='/home/zikaixiao/zikaixiao/LongLoRA-main/models/llama-3-8B-262k', help='Path to the model directory')
parser.add_argument('--data_path', type=str, default='./motivation/data_32k', help='Path to the data directory')


args = parser.parse_args()
    
model_path = args.model_path
data_path = args.data_path



def load_jsonl(file_path):
    """
    Load a .jsonl file where each JSON object may span multiple lines.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        List[dict]: A list where each element is a dictionary representing a JSON object from the file.
    """
    data = []
    current_json = ""

    with open(file_path, 'r') as file:
        for line in file:
            current_json += line.strip()  # Add each line to the current JSON object, removing leading/trailing whitespace
            
            # Check if we have reached the end of a JSON object
            if line.strip().endswith("}"):
                try:
                    data.append(json.loads(current_json))  # Parse the JSON object
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {current_json}")
                    print(f"Error: {e}")
                current_json = ""  # Reset for the next JSON object
    
    return data


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
config = AutoConfig.from_pretrained(model_path)

device = torch.device('cuda')
tokenizer.padding_side = "left"

DEFAULT_PAD_TOKEN = "[PAD]"

special_tokens_dict = dict()
special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)



def mean_reciprocal_rank(ranks):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a list of ranks.
    
    Args:
    ranks (list of int): A list where each entry represents the rank of the first correct answer for a query.
                         The rank should be a positive integer. If no correct answer is found, 
                         that entry can be set to a very large number (e.g., infinity).
    
    Returns:
    float: The Mean Reciprocal Rank (MRR).
    """
    # Calculate the reciprocal rank for each query
    reciprocal_ranks = [1.0 / rank for rank in ranks]
    # reciprocal_ranks = [rank for rank in ranks]
    
    # Calculate the mean of the reciprocal ranks
    mrr = sum(reciprocal_ranks) / len(ranks)
    
    return mrr

# 遍历目录下的所有文件
datasets_path = []
for filename in os.listdir(data_path):
    # 只处理 .jsonl 文件
    if filename.endswith(".jsonl") and "pred" in filename:
        datasets_path.append(os.path.join(data_path, filename))


datasets_path = sorted(datasets_path, key=lambda x: int(re.search(r'(\d+)_pred\.jsonl$', x).group(1)))

ignore_punctuations = ["\"", ' ', ':', "\\", "\\\""]

draw_mrrs = []
draw_file_names = []
for file_id in range(len(datasets_path)):
    dataset_path = datasets_path[file_id]
    
    dataset = load_jsonl(dataset_path)
    mrr_ranks = []
    for eg in dataset:
        for token_id in range(len(eg["logits"])):
            if tokenizer.decode(eg["logits"][token_id][0][0]) in ignore_punctuations:
                continue
            for prob_id in range(len(eg["logits"][token_id][0])):
                decoded_token = tokenizer.decode(eg["logits"][token_id][0][prob_id])
                if eg["answer"].startswith(decoded_token):
                # if tokenizer.decode(eg["logits"][token_id][0][prob_id]) in eg["answer"]:
                    break
            mrr_ranks.append(prob_id+1)
            break
    # Removing all zero values from the list
    mrr_ranks = [value for value in mrr_ranks if value != 1]
    mrr = mean_reciprocal_rank(mrr_ranks)
    file_name = os.path.splitext(os.path.basename(dataset_path))[0]
    draw_mrrs.append(mrr)
    draw_file_names.append(file_name)
    print(f"Mean Reciprocal Rank (MRR) of {file_name}: {mrr}")

# 提取文件名中的长度并取对数
lengths = [int(name.split('_')[-2]) for name in draw_file_names]
log_lengths = np.log(lengths)

# 生成新的标签，格式为 "x.x k"，保留一位小数
labels = [f"{length / 1000:.1f}k" for length in lengths]

# 画图
plt.figure(figsize=(10, 6))
plt.plot(log_lengths, draw_mrrs, marker='o', linestyle='-', color='b')

# 设置横轴标签
plt.xticks(log_lengths, labels)

plt.xlabel('Length of Context')
plt.ylabel('Mean Reciprocal Rank (MRR)')
plt.title('Length-Induced Ranking Degradation')
plt.grid(True)

# 保存为 PD
plt.savefig(data_path + '/mrr_vs_log_length.pdf', format='pdf')

# 显示图表
plt.show()
    