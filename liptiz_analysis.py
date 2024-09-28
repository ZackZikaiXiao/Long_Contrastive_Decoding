import torch

logits_catch_1 = torch.load('/home/zikaixiao/zikaixiao/Long_Contrastive_Decoding/logit_weights_2831_layer1.pt')
logits_catch_2 = torch.load('/home/zikaixiao/zikaixiao/Long_Contrastive_Decoding/logit_weights_3831_layer1.pt')

# 计算绝对值差异
absolute_difference = torch.abs(logits_catch_1 - logits_catch_2)

# 你可以选择对绝对值差异进行求和或取平均
sum_of_differences = absolute_difference.sum()
mean_of_differences = absolute_difference.mean()

# 打印结果
print("Sum of absolute differences:", sum_of_differences)
print("Mean of absolute differences:", mean_of_differences)