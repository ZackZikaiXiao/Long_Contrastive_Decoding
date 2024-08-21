#!/bin/bash

# 设置路径变量
base_path="/home/zikaixiao/zikaixiao/LongLoRA-main"
model_path="$base_path/models/llama2-7B-4k"
model_name="/llama2-7B-4k-prompt-engineering-context-recalling"
data_path="$base_path/LongBench/data"
longbench_dir="$base_path/LongBench"
maxlen=3500
port=12341

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行 pred_db.py
python "$longbench_dir/pred_db.py" --model_path "$model_path" --model_name "$model_name" --data_path "$data_path" --longbench_dir "$longbench_dir" --maxlen "$maxlen" --port "$port"

# # 运行 eval_db.py
python "$longbench_dir/eval_db.py" --model_name "$model_name" --longbench_dir "$longbench_dir"

# 运行 analysis_db.py
python "$longbench_dir/analysis_db.py" --longbench_dir "$longbench_dir"