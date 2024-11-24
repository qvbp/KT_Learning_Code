import numpy as np
import statistics

# 数据
num_auc = [0.77984, 1, 1, 1, 1]
num_acc = [1, 1, 1, 1, 1]

# 计算 AUC 的平均值和标准差
auc_mean = np.mean(num_auc)
auc_std = statistics.stdev(num_auc)

# 计算 ACC 的平均值和标准差
acc_mean = np.mean(num_acc)
acc_std = statistics.stdev(num_acc)

# 格式化输出
print(f"AUC: {auc_mean:.4f}±{auc_std:.4f}")
print(f"ACC: {acc_mean:.4f}±{acc_std:.4f}")
