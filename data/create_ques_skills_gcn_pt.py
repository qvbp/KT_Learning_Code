# import pandas as pd
# import numpy as np
# import torch

# # 1. 读取 ques_skill.csv 文件
# ques_skill_path = './assist2009/ques_skills.csv'  # 替换为你的文件路径
# df = pd.read_csv(ques_skill_path)

# # 2. 获取唯一的 ques（问题），统计问题的数量
# ques_ids = df['ques'].unique()  # 假设问题ID列是 'ques'
# ques_count_unique = len(ques_ids)  # 统计问题的数量

# print(f"ques_count_unique: {ques_count_unique}")

# # 创建问题的索引映射，保持原始顺序
# ques_to_index = {ques: i for i, ques in enumerate(ques_ids)}

# # 初始化邻接矩阵，维度为 (ques_count_unique, ques_count_unique)
# adj_matrix = np.zeros((ques_count_unique, ques_count_unique))

# # 填充邻接矩阵
# for _, row in df.iterrows():
#     ques_id = row['ques']
#     skill_id = row['skill']
    
#     # 获取问题和技能对应的行和列索引
#     ques_idx = ques_to_index[ques_id]
#     # skill_idx = ques_to_index.get(skill_id, None)  # 获取技能的索引位置
    
#     adj_matrix[ques_idx, skill_id] = 1  # 问题和技能之间有联系，置为1

# # 归一化邻接矩阵
# non_zero_indices = adj_matrix == 1
# adj_matrix_normalized = np.copy(adj_matrix)
# adj_matrix_normalized[non_zero_indices] = np.random.uniform(0.1, 1, size=np.sum(non_zero_indices))
# adj_matrix_normalized = (adj_matrix_normalized + adj_matrix_normalized.T) / 2  # 保持对称性

# # 转换为 PyTorch 张量
# adj_tensor = torch.tensor(adj_matrix_normalized, dtype=torch.float32)

# # 保存邻接矩阵
# torch.save(adj_tensor, './assist2009/ques_skill_gcn_adj.pt')

# # 输出邻接矩阵的一些信息
# print("邻接矩阵的维度:", adj_tensor.shape)
# print("邻接矩阵的最小值:", adj_tensor.min().item())
# print("邻接矩阵的最大值:", adj_tensor.max().item())
# print("邻接矩阵的均值:", adj_tensor.mean().item())

import pandas as pd
import numpy as np
import json
import torch

# 加载映射 JSON 文件
json_file_path = './bridge2algebra2006/keyid2idx.json'  # 替换为你的 JSON 文件路径
with open(json_file_path, 'r') as file:
    mapping = json.load(file)

ques_mapping = mapping['questions']  # 问题的映射
skill_mapping = mapping['concepts']    # 技能的映射

# 加载 CSV 文件
csv_file_path = './bridge2algebra2006/ques_skills.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_file_path)

# 2. 获取唯一的 ques（问题），统计问题的数量
ques_ids = df['ques'].unique()  # 假设问题ID列是 'ques'
ques_count_unique = len(ques_ids)  # 统计问题的数量

print(f"ques_count_unique: {ques_count_unique}")

# 获取矩阵的维度
adj_matrix = np.zeros((ques_count_unique, ques_count_unique))

# 填充邻接矩阵
# cnt = 0
for _, row in df.iterrows():
    ques_id = str(row['ques'])  # 确保问题ID是字符串
    skill_id = str(row['skill'])  # 确保技能ID是字符串
    
    # 利用映射找到矩阵索引
    ques_idx = ques_mapping.get(ques_id, None)
    skill_idx = skill_mapping.get(skill_id, None)
    
    if ques_idx is not None and skill_idx is not None:
        adj_matrix[ques_idx, skill_idx] = 1
        # if cnt < 20:
        #     print("-"*100)
        #     print(f"ques_id: {ques_id}         ques_idx: {ques_idx}")
        #     print(f"skill_id: {skill_id}       skill_idx: {skill_idx}")
        #     print("-"*100)
        #     cnt += 1
        # else:
        #     continue


# # 归一化邻接矩阵
# non_zero_indices = adj_matrix == 1
# adj_matrix_normalized = np.copy(adj_matrix)
# adj_matrix_normalized[non_zero_indices] = np.random.uniform(0.1, 1, size=np.sum(non_zero_indices))
# adj_matrix_normalized = (adj_matrix_normalized + adj_matrix_normalized.T) / 2

# # 转换为 PyTorch 张量
# adj_tensor = torch.tensor(adj_matrix_normalized, dtype=torch.float32)

# # 保存邻接矩阵
# torch.save(adj_tensor, './assist2009/ques_skill_gcn_adj.pt')

# # 输出邻接矩阵的信息
# print("邻接矩阵维度:", adj_tensor.shape)
# print("邻接矩阵最小值:", adj_tensor.min().item())
# print("邻接矩阵最大值:", adj_tensor.max().item())
# print("邻接矩阵均值:", adj_tensor.mean().item())

# 归一化邻接矩阵（行归一化）
def row_normalize(adj_matrix):
    """
    对邻接矩阵进行行归一化
    """
    row_sum = adj_matrix.sum(axis=1)  # 每一行的和
    row_sum[row_sum == 0] = 1  # 防止除以 0
    normalized_adj = adj_matrix / row_sum[:, np.newaxis]  # 每行归一化
    return normalized_adj

# 调用行归一化
adj_matrix_normalized = row_normalize(adj_matrix)

# 转换为 PyTorch 张量 && 保存邻接矩阵
# adj_tensor = torch.tensor(adj_matrix_normalized, dtype=torch.float32)
# torch.save(adj_tensor, './algebra2005/ques_skill_gcn_adj.pt')

# 稀疏矩阵转换 && 保存稀疏矩阵
adj_tensor = torch.tensor(adj_matrix_normalized, dtype=torch.float32).to_sparse()
torch.save(adj_tensor, './bridge2algebra2006/ques_skill_gcn_adj.pt')

# 获取稀疏张量的非零值
non_zero_values = adj_tensor.values()

# 输出非零元素的信息
print("非零元素的最小值:", non_zero_values.min().item())
print("非零元素的最大值:", non_zero_values.max().item())
print("非零元素的均值:", non_zero_values.mean().item())



