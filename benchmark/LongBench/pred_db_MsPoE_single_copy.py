import os
import sys
from datasets import load_dataset
import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from filelock import FileLock
import socket

from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import barrier
import torch.distributed as dist

from typing import Optional, Tuple

import os
import sys
import pdb
import math
import copy
import time 
import types
import numpy as np 
from scipy.stats import entropy

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import importlib.util
file_path = '/home/zikaixiao/zikaixiao/LongLoRA-main/generate_replace_every_step_pos_permu.py'
spec = importlib.util.spec_from_file_location('generate_replace_module', file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.generate_replace()

MsPoE_enable = False

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/zikaixiao/zikaixiao/LongLoRA-main/models/llama-3-8B-262k", help="Absolute path to the evaluation model")
    parser.add_argument('--model_name', type=str, default="llama-3-8B-262k", help="Name of the model to save evaluation results")
    parser.add_argument('--longbench_dir', type=str, default="/home/zikaixiao/zikaixiao/LongLoRA-main/benchmark/LongBench", help="Directory to save the evaluation results")
    parser.add_argument('--data_path', type=str, default="/home/zikaixiao/zikaixiao/LongLoRA-main/benchmark/LongBench/data", help="Directory to load the data files")
    parser.add_argument('--maxlen', type=int, default=7500, help="Maximum input length for the model")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)



# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama" in model_name:
        if "llama-3" in model_name:
            prompt = f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(data, max_length, max_gen, prompt_format, dataset, device, model_name, model_path, out_path):
    # Your training/inference logic here
    # device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_path, model_name)
    tokenizer.padding_side = "right"

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                pad_token_id=tokenizer.eos_token_id,
                # eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)


        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path) 
        import sys
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(path)
        if MsPoE_enable:
            from attention.modeling_llama_MsPoE import MsPoELlamaForCausalLM
            print('Using Ms-PoE Positional Embedding')
            config.apply_layers = list(int(x) for x in "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31".split(','))
            config.compress_ratio_min = 1.2
            config.compress_ratio_max = 1.8
            config.head_type = "normal"
            config._attn_implementation = "flash_attention_2"
            model = MsPoELlamaForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16, device_map='auto')
        else:
            
            # Add the directory containing the module to the Python path
            module_path = '/home/zikaixiao/zikaixiao/LongLoRA-main/attention'
            if module_path not in sys.path:
                sys.path.append(module_path)
            # Now you can import the module
            from modeling_llama_official_copy import LlamaForCausalLM
            # model = LlamaForCausalLM.from_pretrained(path, config=config, device_map='auto')
            model = LlamaForCausalLM.from_pretrained(model_path,
                                    config=config,
                                    torch_dtype=torch.bfloat16,   
                                    attn_implementation="flash_attention_2",    # flash_attention_2
                                    device_map='auto'
                                    )

    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model_name
    model_path = args.model_path
    max_length = args.maxlen
    longbench_dir = args.longbench_dir
    data_path = args.data_path

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("./benchmark/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./benchmark/LongBench/config/dataset2maxlen.json", "r"))
 
    if not os.path.exists(longbench_dir):
        os.makedirs(longbench_dir)
    if not os.path.exists(f"{longbench_dir}/pred"):
        os.makedirs(f"{longbench_dir}/pred")
    if not os.path.exists(f"{longbench_dir}/pred_e"):
        os.makedirs(f"{longbench_dir}/pred_e")

    for dataset in datasets:
        if args.e:
            data = [json.loads(line) for line in open(os.path.join(data_path, f"{dataset}_e.jsonl"), encoding="utf-8")]
            # data = [json.loads(line) for line in open(f"./LongBench/data/{dataset}_e.jsonl", encoding="utf-8")]
            if not os.path.exists(f"{longbench_dir}/pred_e/{model_name}"):
                os.makedirs(f"{longbench_dir}/pred_e/{model_name}")
            out_path = f"{longbench_dir}/pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = [json.loads(line) for line in open(os.path.join(data_path, f"{dataset}.jsonl"), encoding="utf-8")]
            # data = [json.loads(line) for line in open(f"./LongBench/data/{dataset}.jsonl", encoding="utf-8")]
            if not os.path.exists(f"{longbench_dir}/pred/{model_name}"):
                os.makedirs(f"{longbench_dir}/pred/{model_name}")
            out_path = f"{longbench_dir}/pred/{model_name}_rope_permu/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        # data_subsets = [data_all[i::world_size] for i in range(world_size)]
        get_pred(data_all, max_length, max_gen, prompt_format, dataset, device, model_name, model_path, out_path)
