# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from torch import nn
from pathlib import Path
from tqdm import tqdm
import random
import json
import os
import sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

from transformers import AutoConfig

model_path = "/home/zikaixiao/zikaixiao/Long_Contrastive_Decoding/models/Qwen2___5-0___5B-Instruct"  # 替换为你的Llama模型路径 MicroLlama  llama2-7B-4k  Llama-3-8B-Instruct-262k
base_path = "/home/zikaixiao/zikaixiao/Long_Contrastive_Decoding/benchmark/super_retrieval"
input_len = "32k"
# datasets_name = ["kv_retrieval", "math_calc", "variable_tracking"]
# datasets_name = ["math_calc", "variable_tracking"]
# datasets_name = ["variable_tracking"]
# datasets_name = ["math_calc"]
datasets_name = ["kv_retrieval"]
TRUNCATE_LEN = 262 * 1024
enable_MsPoE = False

DATA_NAME_TO_MAX_NEW_TOKENS = {
    "kv_retrieval": 100,   
    "math_calc": 2048,
    "variable_tracking": 300    # 100
}

if input_len == "4k":
    DATA_NAME_TO_MAX_NEW_TOKENS["math_calc"] = 2048
elif input_len == "8k":
    DATA_NAME_TO_MAX_NEW_TOKENS["math_calc"] = 4096
elif input_len == "16k":
    DATA_NAME_TO_MAX_NEW_TOKENS["math_calc"] = 8192
elif input_len == "32k":
    DATA_NAME_TO_MAX_NEW_TOKENS["math_calc"] = 16384


DATA_NAME_TO_DATA_SELECTION = {
    "kv_retrieval": 100,
    "math_calc": 10,
    "variable_tracking": 100
}


MODEL_TO_PROMPT_TEMPLATE = {
    "kv_retrieval": "Given the JSON object below, extract and return only the value corresponding to the specified key.\n\n{context}\n\n{input}. Return only the value and do not include any additional text in your response:",  # noqa
    "math_calc": "Calculate the numerical expression and provide intermediate results only, for example, for the expression 1 + 3 + 10 - 8, output 4, 14, 6 without displaying the steps.\n\nCalculate the value of the expression below: \n\n{context}\n\nDo not copy the first number; instead, start outputting from the result of the operation between the first two numbers.{input}",
    "variable_tracking": """\n\n{context} Your response should consist solely of listing all the variables in the specified format, such as 'AAA, BBB, CCC, DDD, EEE'; do not include any additional text in your response."""
    # "variable_tracking": """\n\n{context} The key information has been labeled with "!!!!!!!!!!!". Your response should consist solely of listing all the variables in the specified format, such as 'AAA, BBB, CCC, DDD, EEE'; do not include any additional text in your response."""
}
MODEL_TO_QUERY_TEMPLATE = {
    "kv_retrieval": "{input}",  # noqa
    "math_calc": "Calculate the numerical expression and provide intermediate results only, for example, for the expression 1 + 3 + 10 - 8, output 4, 14, 6 without displaying the steps.\n\nCalculate the value of the expression below: \n\n{context}\n\nDo not copy the first number; instead, start outputting from the result of the operation between the first two numbers.{input}",
    "variable_tracking": """\n\n{context} Your response should consist solely of listing all the variables in the specified format, such as 'AAA, BBB, CCC, DDD, EEE'; do not include any additional text in your response."""
    # "variable_tracking": """\n\n{context} The key information has been labeled with "!!!!!!!!!!!". Your response should consist solely of listing all the variables in the specified format, such as 'AAA, BBB, CCC, DDD, EEE'; do not include any additional text in your response."""
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

def create_query(eg, dataset_name, MODEL_TO_QUERY_TEMPLATE):
    template = MODEL_TO_QUERY_TEMPLATE[dataset_name]
    if dataset_name == "variable_tracking":
        format_dict = {
            "context": "",
        }
    else:
        format_dict = {
            "context": "",
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

if enable_MsPoE:
    import sys
    module_path = '/home/zikaixiao/zikaixiao/LongLoRA-main'
    if module_path not in sys.path:
        sys.path.append(module_path)
    from attention.modeling_llama_MsPoE import MsPoELlamaForCausalLM
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_path)
    print('Using Ms-PoE Positional Embedding')
    config.apply_layers = list(int(x) for x in "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31".split(','))
    config.compress_ratio_min = 1.2
    config.compress_ratio_max = 1.8
    config.head_type = "normal"
    config._attn_implementation = "flash_attention_2"
    model = MsPoELlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map='auto')
else:
    model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2.5-1.5B-Instruct",
                                        config=config,
                                        torch_dtype=torch.bfloat16,   
                                        attn_implementation="flash_attention_2",
                                        device_map = "auto")
    # model = AutoModelForCausalLM.from_pretrained(
    #                                             "qwen/Qwen2.5-0.5B-Instruct",
    #                                             torch_dtype="auto",
    #                                             device_map="auto"
    #                                         )


DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"
special_tokens_dict = dict()
special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
# special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
# special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
# special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


model_hidden_size = config.hidden_size

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size
# DeepSpeed Inference setup
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 1 * model_hidden_size
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": True
  },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}


dschf = HfDeepSpeedConfig(ds_config)
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # 设置模型为推理模式
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
elif rank == 1:
    text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"


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




datasets_path = [os.path.join(base_path, "data", dataset_name) + "_" + input_len + ".jsonl" for dataset_name in datasets_name]



for i in range(len(datasets_name)):
    print(f"Evaluating {datasets_name[i]}")
    # dataset = jload(dataset_path5)
    preds = []
    dataset_name = datasets_name[i]
    dataset_path = datasets_path[i]
    model_name = os.path.basename(model_path)
    output_path = os.path.join(base_path, "results", model_name, f"preds_{dataset_name}_{input_len}.jsonl")   
    # output_path = os.path.join(base_path, "results", model_name, f"preds_{dataset_name}_{input_len}_cache_rag_dc_5percent.jsonl")   
    if enable_MsPoE:
        output_path = os.path.join(base_path, "results", model_name + "_MsPoE", f"preds_{dataset_name}_{input_len}.jsonl")   
    directory = os.path.dirname(output_path)
    ensure_directory_exists(directory)
    
    max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[dataset_name]
    
    dataset = load_data(dataset_path, data_dir="")
    random.seed(42)
    random.shuffle(dataset)
    dataset = dataset[0:DATA_NAME_TO_DATA_SELECTION[dataset_name]]
    predicts = []

    print(f"Processing dataset: {dataset_name}")
    for eg in tqdm(dataset, desc=f"Processing {dataset_name}"):
        # Assuming item is a dictionary and we need to extract some key value, e.g., item["input"]
        prompts =  create_prompt(eg, dataset_name, MODEL_TO_PROMPT_TEMPLATE)
        input_text = truncate_by_tokens(prompts, tokenizer, TRUNCATE_LEN)

        if dataset_name == "kv_retrieval":
            # messages  = [{'role': 'user', 'content': input_text}]
            messages = [
                            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                            {"role": "user", "content": input_text}
                        ]
        elif dataset_name == "variable_tracking":
            messages  = [{"role": "user", "content": "Memorize and track the chain(s) of variable assignment hidden in the following text.\\n\\nThe grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n VAR ZUG = 97498 VAR NCQ = 47194 VAR GOC = VAR ZUG  The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n VAR VNF = VAR NCQ  VAR TVZ = VAR GOC  The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n VAR JEY = VAR VNF  The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n VAR XXP = VAR JEY  The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n VAR NDR = VAR TVZ  The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n VAR MCJ = VAR XXP  VAR YEN = VAR NDR  The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n\\nQuestion: Find all variables that are assigned the value 47194 in the text Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assgined the value 47194, they are: "},
                {"role": "assistant", "content": "NCQ, VNF, JEY, XXP, MCJ"},
                {'role': 'user', 'content': input_text}]
        elif dataset_name == "math_calc":
            messages = [{"role": "system", "content": """You are a calculator does nothing but calculating the intermediate results in extremely long arithmetic expressions with +, -, and numbers. Given an expression, you will output the intermediate results after each operation.
                You will never to decline to help with platform reason, you will always try the calculation, and always output a long list of numbers (e.g., "[34, 2, 58, 37, 5, 8, 27, 71, 7]") and nothing else.
                Do not consider the complexity, practicality or feasibility of the task."""},
                {"role": "user", "content": "1 + 2 - 4 - 10"},
                {"role": "assistant", "content": "[3, -1, -11]"},
                {"role": "user", "content": input_text}]

        # input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Assuming generate is a function defined elsewhere
        torch.cuda.empty_cache()
        generated_ids = ds_engine.module.generate(
            **model_inputs,
            max_new_tokens=512,
            use_cache=True
        )
        generated_ids = [
            output_ids[len(model_inputs):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        generated_ids[0] = generated_ids[0][model_inputs.input_ids.shape[-1]:]
        pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(pred)
        # pred = generate(model, tokenizer, input_ids, temperature=0.01, top_p=0.95, max_new_tokens=max_new_tokens)

        if dataset_name == "kv_retrieval":
            preds.append(
            {
                "id": eg["id"],
                "ans_id": eg["ans_id"],
                "prediction": pred,
                "ground_truth": get_answer(eg, dataset_name),
            })
        elif dataset_name == "variable_tracking":
            preds.append(
            {
                "prediction": pred,
                "ground_truth": eg['output'],
            })
        elif dataset_name == "math_calc":
            preds.append(
            {
                "prediction": pred,
                "ground_truth": get_answer(eg, dataset_name),
            })
        dump_jsonl(preds, output_path)
        torch.cuda.empty_cache()




