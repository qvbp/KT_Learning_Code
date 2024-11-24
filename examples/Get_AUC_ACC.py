import json
import ast
import numpy as np
import statistics
import argparse
import os

# 创建参数解析器
parser = argparse.ArgumentParser(description='Calculate ACC and AUC from test result files.')
parser.add_argument('--dir', type=str, required=True, help='The directory path containing the result files.')

# 解析参数
args = parser.parse_args()
directory = args.dir

# 定义列表
window_testauc_list = []
window_testacc_list = []
windowauclate_mean_list = []
windowacclate_mean_list = []
is_fusion = True

# 遍历指定路径下的文件
for i in range(0, 5):
    # 文件路径
    file_path = os.path.join(directory, f'output_{i}.txt')

    # 读取文件的最后一行
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()

    # 使用 ast.literal_eval() 解析字典
    results = ast.literal_eval(last_line)

    # 提取需要的实验结果并分别放入各自的列表中
    window_testauc_list.append(results['window_testauc'])
    window_testacc_list.append(results['window_testacc'])
    if 'windowauclate_mean' in results:
        windowauclate_mean_list.append(results['windowauclate_mean']) 
        windowacclate_mean_list.append(results['windowacclate_mean'])
    else:
        is_fusion = False 

    # 输出结果，查看是否正确
    print("-"*100)
    print(f"Fold {i}:")
    print("window_testauc_list:", window_testauc_list[i])
    print("window_testacc_list:", window_testacc_list[i])
    if is_fusion:
        print("windowauclate_mean_list:", windowauclate_mean_list[i])
        print("windowacclate_mean_list:", windowacclate_mean_list[i])

print("-"*100)
# 计算 没有fusion的 AUC、ACC 的平均值和标准差
# AUC
window_testauc_mean = np.mean(window_testauc_list)
window_testauc_std = statistics.stdev(window_testauc_list)

# ACC
window_testacc_mean = np.mean(window_testacc_list)
window_testacc_std = statistics.stdev(window_testacc_list)

# 格式化输出
print(end='\n')
print("+"*100)
print(f"window_testauc: {window_testauc_mean:.4f}±{window_testauc_std:.4f}")
print(f"window_testacc: {window_testacc_mean:.4f}±{window_testacc_std:.4f}")

# 计算 有fusion的 AUC、ACC 的平均值和标准差
if is_fusion:
    # AUC
    windowauclate_mean = np.mean(windowauclate_mean_list)
    windowauclate_mean_std = statistics.stdev(windowauclate_mean_list)
    
    # ACC
    windowacclate_mean = np.mean(windowacclate_mean_list)
    windowacclate_mean_std = statistics.stdev(windowacclate_mean_list)

    # 格式化输出
    print(f"windowauclate_mean: {windowauclate_mean:.4f}±{windowauclate_mean_std:.4f}")
    print(f"windowacclate_mean: {windowacclate_mean:.4f}±{windowacclate_mean_std:.4f}")
    print("+"*100)