import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# 读取JSONL文件并转换为DataFrame
file_path = '/home/zikaixiao/zikaixiao/LongLoRA-main/benchmark/super_retrieval/results/llama-3-8B-262k/preds_kv_retrieval_16k_vis.jsonl'

# 读取JSONL文件
data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 转换为DataFrame
df = pd.DataFrame(data)

# 判断prediction和ground_truth是否相等
df['correct'] = df['prediction'] == df['ground_truth']

# 计算每个ans_id的正确率
accuracy = df.groupby('ans_id')['correct'].mean().reset_index()

# 将ans_id分成20组
num_groups = 20
accuracy['group'] = pd.cut(accuracy['ans_id'], bins=num_groups, labels=np.arange(num_groups))

# 计算每组的平均正确率
grouped_accuracy = accuracy.groupby('group')['correct'].mean()

# 可视化预测正确程度和分组后的ans_id的关系
plt.figure(figsize=(12, 6))
plt.bar(grouped_accuracy.index.astype(str), grouped_accuracy.values)
plt.xlabel('Answer ID Group')
plt.ylabel('Accuracy')
plt.title('Prediction Accuracy by Answer ID Group')
plt.ylim(0, 1)
plt.grid(axis='y')

# 保存图像
output_path = 'prediction_accuracy_by_ans_id_group.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)  # 保存图像时自动调整边界
plt.show()

print(f"图像已保存到 {output_path}")
