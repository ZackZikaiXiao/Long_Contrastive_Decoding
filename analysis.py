import json
import matplotlib.pyplot as plt

# 假设你的JSON数据存储在文件中，首先读取文件
with open('ori_pqal.json', 'r') as file:
    data = json.load(file)

# 初始化一个列表来存储每个CONTEXTS的字符数量
contexts_lengths = []

# 遍历JSON数据
for entry in data.values():
    # 确保'CONTEXTS'字段存在
    if 'CONTEXTS' in entry:
        # 遍历CONTEXTS列表
        for context in entry['CONTEXTS']:
            # 计算每个CONTEXT的字符数量并添加到列表中
            contexts_lengths.append(len(context))

# 绘制分布图
plt.figure(figsize=(10, 5))
plt.hist(contexts_lengths, bins=20, color='blue', edgecolor='black')
plt.title('Distribution of CONTEXTS Lengths')
plt.xlabel('Length of CONTEXTS')
plt.ylabel('Frequency')

# 保存高清图像
plt.savefig('contexts_length_distribution.png', format='png', dpi=300)

# 显示图表
plt.show()