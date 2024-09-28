# # from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import LlamaForCausalLM, AutoTokenizer
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# import numpy as np
# from torch import nn
# from pathlib import Path
# from tqdm import tqdm
# import random
# import json
# import torch
# import os
# from transformers import AutoConfig
# import re
# import argparse
# import matplotlib.pyplot as plt
# import random
# import re
# from collections import OrderedDict

# parser = argparse.ArgumentParser(description='Generate probs')

# # 设置默认的 model_path 和 data_path
# parser.add_argument('--model_path', type=str, default='/home/zikaixiao/zikaixiao/LongLoRA-main/models/llama-3-8B-262k', help='Path to the model directory')
# parser.add_argument('--data_path', type=str, default='./motivation/kv_retrieval_noise_26008_wrong.jsonl', help='Path to the data directory')

# args = parser.parse_args()
    
# model_path = args.model_path
# dataset_path = args.data_path


# dataset_name = "kv_retrieval"
# TRUNCATE_LEN = 131072

# DATA_NAME_TO_MAX_NEW_TOKENS = {
#     "kv_retrieval": 100,   
#     "math_calc": 2048,
#     "variable_tracking": 100   
# }


# DATA_NAME_TO_DATA_SELECTION = {
#     "kv_retrieval": 100,
#     "math_calc": 10,
#     "variable_tracking": 100
# }


# MODEL_TO_PROMPT_TEMPLATE = {
#     "kv_retrieval": "Given the JSON object below, extract and return only the value corresponding to the specified key.\n\n{context}\n\n{input}. Return only the value and do not include any additional text in your response:",  # noqa
#     "math_calc": "Calculate the numerical expression and provide intermediate results only, for example, for the expression 1 + 3 + 10 - 8, output 4, 14, 6 without displaying the steps.\n\nCalculate the value of the expression below: \n\n{context}\n\nDo not copy the first number; instead, start outputting from the result of the operation between the first two numbers.{input}",
#     "variable_tracking": """\n\n{context} Your response should consist solely of listing all the variables in the specified format, such as 'AAA, BBB, CCC, DDD, EEE'; do not include any additional text in your response."""
# }

# def ensure_directory_exists(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def truncate_input(input: list, max_length: int, manner="middle"):
#     if len(input) <= max_length:
#         return input
#     if manner == "middle":
#         split = max_length // 2
#         return input[0:split] + input[-split:]
#     else:
#         return None

# def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
#     tokens = tok.encode(input)
#     len_before = len(tokens)
#     print(f"# tokens before: {len_before}")
#     tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
#     len_after = len(tokens)  # type: ignore
#     print(f"# tokens after: {len_after}")
#     assert len_after <= len_before
#     assert len_after <= max_tokens
#     return tok.decode(tokens, skip_special_tokens=True)


# def create_prompt(eg, dataset_name, MODEL_TO_PROMPT_TEMPLATE):
#     template = MODEL_TO_PROMPT_TEMPLATE[dataset_name]
#     if dataset_name == "variable_tracking":
#         format_dict = {
#             "context": eg["instruction"],
#         }
#     else:
#         format_dict = {
#             "context": eg["context"],
#             "input": eg["input"],
#         }
#     prompt = template.format(**format_dict)
#     return prompt

# def get_answer(eg: dict, data_name: str):
#     if data_name in ["code_debug", "longbook_choice_eng"]:
#         OPTIONS = "ABCD"
#         if isinstance(eg["answer"], str):
#             ret = [eg["answer"], OPTIONS[eg['options'].index(eg["answer"])]]
#         elif isinstance(eg["answer"], list):
#             if len(eg["answer"]) == 1:
#                 ret = [eg["answer"][0], OPTIONS[eg['options'].index(eg["answer"][0])]]
#             elif len(eg["answer"]) == 2 and eg["answer"][1] in ['A', 'B', 'C', 'D']:
#                 ret = eg['answer']
#             else:
#                 raise ValueError
#         else:
#             raise ValueError
#         return ret

#     return eg["answer"]


# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
# config = AutoConfig.from_pretrained(model_path)

# device = torch.device('cuda')
# tokenizer.padding_side = "left"

# # bnb_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_use_double_quant=False,
# # )
# # model = LlamaForCausalLM.from_pretrained(model_path,
# #                                     config=config,
# #                                     torch_dtype=torch.bfloat16,   
# #                                     attn_implementation="flash_attention_2",    # flash_attention_2
# #                                     quantization_config=bnb_config,
# #                                     device_map='auto'
# #                                     )

# model = LlamaForCausalLM.from_pretrained(model_path,
#                                     config=config,
#                                     torch_dtype=torch.bfloat16,   
#                                     attn_implementation="flash_attention_2",    # flash_attention_2
#                                     device_map='auto'
#                                     )

# DEFAULT_PAD_TOKEN = "[PAD]"

# special_tokens_dict = dict()
# special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

# num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))


# def generate(model, tokenizer, prompts, temperature=1.0, max_new_tokens=20):
#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>"),
#     ]
#     outputs = model.generate(
#         prompts,
#         max_new_tokens=max_new_tokens,
#         num_beams=1,
#         do_sample = False,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=terminators,  # 设置terminators
#         use_cache=True,
#         return_dict_in_generate=True, 
#         output_scores=True
#     )
    
#     logits = outputs.scores
#     # topk_values, topk_indices = torch.topk(logits[0], config.vocab_size, dim=-1)
#     response = outputs.sequences[0][input_ids.shape[-1]:]
#     generated_texts = tokenizer.decode(response, skip_special_tokens=True)

#     return generated_texts, logits

# def load_jsonl(file_path):
#     """
#     Load a .jsonl file where each JSON object may span multiple lines.

#     Args:
#         file_path (str): The path to the .jsonl file.

#     Returns:
#         List[dict]: A list where each element is a dictionary representing a JSON object from the file.
#     """
#     data = []
#     current_json = ""

#     with open(file_path, 'r') as file:
#         for line in file:
#             current_json += line.strip()  # Add each line to the current JSON object, removing leading/trailing whitespace
            
#             # Check if we have reached the end of a JSON object
#             if line.strip().endswith("}"):
#                 try:
#                     data.append(json.loads(current_json))  # Parse the JSON object
#                 except json.JSONDecodeError as e:
#                     print(f"Error decoding JSON: {current_json}")
#                     print(f"Error: {e}")
#                 current_json = ""  # Reset for the next JSON object
    
#     return data



# preds = []

# model_name = os.path.basename(model_path)

# max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[dataset_name]

# dataset = load_jsonl(dataset_path)

# for i in range(len(dataset)):
#     eg = dataset[i]
#     prompts_init =  create_prompt(eg, dataset_name, MODEL_TO_PROMPT_TEMPLATE)
    
#     # 从 prompts 中提取 JSON 对象
#     json_data = re.search(r'JSON data:\n(.*)\n\n\nKey:', prompts_init).group(1)
#     json_data_dict = json.loads(json_data, object_pairs_hook=OrderedDict)

#     # 自动提取关键的 key
#     important_key = re.search(r'Key: "(.*)"', prompts_init).group(1)

#     # 提取噪声 key-value 对（有序）
#     noisy_pairs = [(k, v) for k, v in json_data_dict.items() if k != important_key]

#     # 生成不同噪声比例的 prompts
#     for noise_level in range(100, 0, -10):  # 从90%到10%，每次减少10%
#         # 计算要保留的噪声对数
#         num_noisy_pairs_to_keep = int(len(noisy_pairs) * (noise_level / 100.0))

#         # 随机选择要保留的噪声对，并按原顺序排列
#         selected_noisy_pairs = sorted(random.sample(noisy_pairs, num_noisy_pairs_to_keep), key=lambda x: noisy_pairs.index(x))

#         # 构建有序字典，保留原始顺序
#         reduced_json_data_dict = OrderedDict()
#         for k, v in json_data_dict.items():
#             if k == important_key or k in dict(selected_noisy_pairs):
#                 reduced_json_data_dict[k] = v

#         # 将字典转换为 JSON 字符串
#         reduced_json_data_str = json.dumps(reduced_json_data_dict, separators=(', ', ': '))

#         # 生成新的 prompts
#         prompts = f'Given the JSON object below, extract and return only the value corresponding to the specified key.\n\nJSON data:\n{reduced_json_data_str}\n\n\nKey: "{important_key}"\nThe value associated with the specified key is: . Return only the value and do not include any additional text in your response:'

#         # 输出或保存新的 prompts
#         # print(f"Noise Level: {noise_level}%\n{new_prompts}\n")
    
    
    
#         # 开始分析
#         input_text = truncate_by_tokens(prompts, tokenizer, TRUNCATE_LEN)

#         messages  = [{'role': 'user', 'content': input_text}]
        
#         input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)

#         with torch.no_grad():
#             pred, logits = generate(model, tokenizer, input_ids, temperature=1, max_new_tokens=max_new_tokens)

#         topk_indices = []
#         topk_values = []
#         for next_token_logits_base in logits:
#             k = 100
#             topk_value, topk_indice = torch.topk(next_token_logits_base, k, dim=-1)
#             topk_indices.append(topk_indice)
#             topk_values.append(topk_value)
#         topk_indices = [tensor.tolist() for tensor in topk_indices][0:10]
#         topk_values = [tensor.tolist() for tensor in topk_values][0:10]

        
#         ignore_punctuations = ["\"", ' ', ':', "\\", "\\\""]
#         for token_id in range(len(topk_indices)):
#                 if tokenizer.decode(topk_indices[token_id][0][0]) in ignore_punctuations:
#                     continue
#                 for prob_id in range(len(topk_indices[token_id][0])):
#                     decoded_token = tokenizer.decode(topk_indices[token_id][0][prob_id])
#                     if eg["answer"].startswith(decoded_token):
#                     # if tokenizer.decode(eg["logits"][token_id][0][prob_id]) in eg["answer"]:
#                         break
#                 break
        
#         # 计算 anchor_token 和 anchor_logits
#         anchor_token = topk_indices[token_id][0]
#         anchor_logits = [logits[token_id].tolist()[0][i] for i in anchor_token]
#         length = input_ids.shape[1]

#         # 获取 label_token
#         label_token = anchor_token[prob_id]

#         # 对 anchor_token 进行排序，并根据排序的顺序调整 anchor_logits
#         sorted_indices = sorted(range(len(anchor_token)), key=lambda k: anchor_token[k])
#         sorted_anchor_token = [anchor_token[i] for i in sorted_indices]
#         sorted_anchor_logits = [anchor_logits[i] for i in sorted_indices]

#         # 找到 label_token 在排序后的列表中的索引
#         label_index = sorted_anchor_token.index(label_token)

#         # x 轴设置为 0 到 n-1
#         x_axis = list(range(len(sorted_anchor_token)))

#         # 创建文件夹路径
#         save_path = './motivation/cause_of_distraction'
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         # 保存文件的完整路径
#         file_name = f'{save_path}/{i}_{length}.jpg'

#         # 画图
#         plt.figure(figsize=(10, 6))
#         plt.scatter(x_axis, sorted_anchor_logits, color='blue', marker='o', s=5)  # 调整点的大小

#         # 强调 label_token 对应的点
#         plt.scatter(x_axis[label_index], sorted_anchor_logits[label_index], color='red', marker='o', s=50, label='Label Token')

#         # 添加标题和标签
#         plt.title('Anchor Token vs Logits')
#         plt.xlabel('Sorted Index')
#         plt.ylabel('Logits')
#         plt.grid(True)

#         # 添加图例
#         plt.legend()

#         # 保存为 PDF 格式
#         plt.savefig(file_name, format='jpg', dpi=300)

#         # 释放显存
#         del input_ids, logits, topk_indices, topk_values, sorted_anchor_logits, sorted_anchor_token, x_axis
#         torch.cuda.empty_cache()


import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

folder_path = './motivation/cause_of_distraction'
# 自动读取文件夹中的所有图片文件
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 提取样本编号并组织文件
samples = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    sample_id, length = os.path.splitext(file_name)[0].split('_')
    if sample_id not in samples:
        samples[sample_id] = []
    samples[sample_id].append((file_path, length))

# 排序文件
for sample_id in samples:
    samples[sample_id].sort(key=lambda x: int(x[1]))

# 为每个样本生成并保存动图
for sample_id, paths in samples.items():
    # 获取第一张图片的尺寸，以设置图形的大小
    img = Image.open(paths[0][0])
    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100 + 1))  # 调整figsize增加空间显示长度

    # 初始化进度条
    progress_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # 进度条的位置和大小
    progress_ax.set_xlim(0, len(paths) - 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    progress_line, = progress_ax.plot([], [], color='blue', lw=6)

    def animate(i):
        img_path, length = paths[i]
        img = Image.open(img_path)
        ax.clear()
        ax.imshow(img, interpolation='nearest')
        ax.axis('off')  # 关闭坐标轴

        if i == 0:
            # 如果是第一帧，添加“开始”标记
            ax.text(10, 10, 'START', fontsize=80, color='red', verticalalignment='top', horizontalalignment='left')

        ax.set_title(f'Length: {length}', fontsize=80, pad=10)  # 在上方显示长度

        # 更新进度条
        progress_line.set_data([0, i+1], [0.5, 0.5])

    ani = animation.FuncAnimation(fig, animate, frames=len(paths), interval=3000)

    # 保存动图，以"样本ID.gif"命名
    gif_name = f'{sample_id}.gif'
    ani.save(gif_name, writer='imagemagick', fps=1)
    plt.close(fig)  # 关闭图表，防止过多打开的图表占用内存

print("动图生成完毕。")