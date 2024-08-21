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
import argparse


parser = argparse.ArgumentParser(description='Generate probs')

# 设置默认的 model_path 和 data_path
parser.add_argument('--model_path', type=str, default='/home/zikaixiao/zikaixiao/LongLoRA-main/models/llama-3-8B-262k', help='Path to the model directory')
parser.add_argument('--data_path', type=str, default='./motivation/data/', help='Path to the data directory')

args = parser.parse_args()
    
model_path = args.model_path
data_path = args.data_path


dataset_name = "kv_retrieval"
TRUNCATE_LEN = 131072

DATA_NAME_TO_MAX_NEW_TOKENS = {
    "kv_retrieval": 100,   
    "math_calc": 2048,
    "variable_tracking": 100   
}


DATA_NAME_TO_DATA_SELECTION = {
    "kv_retrieval": 100,
    "math_calc": 10,
    "variable_tracking": 100
}


MODEL_TO_PROMPT_TEMPLATE = {
    "kv_retrieval": "Given the JSON object below, extract and return only the value corresponding to the specified key.\n\n{context}\n\n{input}. Return only the value and do not include any additional text in your response:",  # noqa
    "math_calc": "Calculate the numerical expression and provide intermediate results only, for example, for the expression 1 + 3 + 10 - 8, output 4, 14, 6 without displaying the steps.\n\nCalculate the value of the expression below: \n\n{context}\n\nDo not copy the first number; instead, start outputting from the result of the operation between the first two numbers.{input}",
    "variable_tracking": """\n\n{context} Your response should consist solely of listing all the variables in the specified format, such as 'AAA, BBB, CCC, DDD, EEE'; do not include any additional text in your response."""
}

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None

def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def create_prompt(eg, dataset_name, MODEL_TO_PROMPT_TEMPLATE):
    template = MODEL_TO_PROMPT_TEMPLATE[dataset_name]
    if dataset_name == "variable_tracking":
        format_dict = {
            "context": eg["instruction"],
        }
    else:
        format_dict = {
            "context": eg["context"],
            "input": eg["input"],
        }
    prompt = template.format(**format_dict)
    return prompt

def get_answer(eg: dict, data_name: str):
    if data_name in ["code_debug", "longbook_choice_eng"]:
        OPTIONS = "ABCD"
        if isinstance(eg["answer"], str):
            ret = [eg["answer"], OPTIONS[eg['options'].index(eg["answer"])]]
        elif isinstance(eg["answer"], list):
            if len(eg["answer"]) == 1:
                ret = [eg["answer"][0], OPTIONS[eg['options'].index(eg["answer"][0])]]
            elif len(eg["answer"]) == 2 and eg["answer"][1] in ['A', 'B', 'C', 'D']:
                ret = eg['answer']
            else:
                raise ValueError
        else:
            raise ValueError
        return ret

    return eg["answer"]


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
config = AutoConfig.from_pretrained(model_path)

device = torch.device('cuda')
tokenizer.padding_side = "left"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=False,
# )
# model = LlamaForCausalLM.from_pretrained(model_path,
#                                     config=config,
#                                     torch_dtype=torch.bfloat16,   
#                                     attn_implementation="flash_attention_2",    # flash_attention_2
#                                     quantization_config=bnb_config,
#                                     device_map='auto'
#                                     )

model = LlamaForCausalLM.from_pretrained(model_path,
                                    config=config,
                                    torch_dtype=torch.bfloat16,   
                                    attn_implementation="flash_attention_2",    # flash_attention_2
                                    device_map='auto'
                                    )

DEFAULT_PAD_TOKEN = "[PAD]"

special_tokens_dict = dict()
special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


def generate(model, tokenizer, prompts, temperature=1.0, max_new_tokens=20):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    outputs = model.generate(
        prompts,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample = False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,  # 设置terminators
        use_cache=True,
        return_dict_in_generate=True, 
        output_scores=True
    )
    
    logits = outputs.scores
    # topk_values, topk_indices = torch.topk(logits[0], config.vocab_size, dim=-1)
    response = outputs.sequences[0][input_ids.shape[-1]:]
    generated_texts = tokenizer.decode(response, skip_special_tokens=True)

    return generated_texts, logits

def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def load_json(fname):
    return json.load(open(fname))


def dump_jsonl(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def dump_json(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def load_data(data_path: str, data_dir: str = "../data/InfiniteBench/"):
    path = data_path
    fname = Path(data_dir, path)
    return list(iter_jsonl(fname))



# 遍历目录下的所有文件
datasets_path = []
for filename in os.listdir(data_path):
    # 只处理 .jsonl 文件
    if filename.endswith(".jsonl"):
        datasets_path.append(os.path.join(data_path, filename))


datasets_path = sorted(datasets_path, key=lambda x: int(re.search(r'(\d+)', x).group()))


for i in range(len(datasets_path)):
    preds = []
    dataset_path = datasets_path[i]
    model_name = os.path.basename(model_path)

    max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[dataset_name]
    
    dataset = load_data(dataset_path, data_dir="")
    
    # dataset = dataset[0:2]

    for eg in dataset:
        prompts =  create_prompt(eg, dataset_name, MODEL_TO_PROMPT_TEMPLATE)
        input_text = truncate_by_tokens(prompts, tokenizer, TRUNCATE_LEN)

        messages  = [{'role': 'user', 'content': input_text}]
        
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
 
        pred, logits = generate(model, tokenizer, input_ids, temperature=1, max_new_tokens=max_new_tokens)

        top_k_pred = []
        for next_token_logits_base in logits:
            k = 1000
            _, topk_indices = torch.topk(next_token_logits_base, k, dim=-1)
            top_k_pred.append(topk_indices)
        
        eg['pred'] = pred
        eg['logits'] = [tensor.tolist() for tensor in top_k_pred][0:5]
        # eg['logits'] = logits.tolist()  # 将 logits 转换为可序列化的列表格式
        # 删除不需要的字段
        if 'context' in eg:
            del eg['context']
        if 'input' in eg:
            del eg['input']
        print("label", eg['answer'])
        print("pred", pred)
        preds.append(eg)
    # 将更新后的数据集保存为 JSON 文件
    file_name, file_extension = os.path.splitext(dataset_path)
    # 添加 "pred" 后缀
    new_dataset_path = f"{file_name}_pred{file_extension}"

    with open(new_dataset_path, 'w') as f:
        # 逐行写入每个字典对象
        for eg in preds:
            json_str = json.dumps(eg, indent=4)  # 将字典转换为JSON字符串
            f.write(json_str + '\n')  # 每个字典对象写一行，并在后面加一个换行符