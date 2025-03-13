import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA  # 使用推荐的模块导入FPCA
import os

# -------------------------------
# 设置随机种子以确保结果可复现
# -------------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)

# -------------------------------
# 数据读取及预处理
# -------------------------------
# Load CSV files
death_df = pd.read_csv('patients_state.csv')
panda_df = pd.read_csv('P_and_A_state.csv')
workplace_df = pd.read_csv('workplace_state.csv')

# 用0填充workplace_df中的缺失值
workplace_df = workplace_df.fillna(0)

# 重命名第一列为 'state_id'
death_df = death_df.rename(columns={death_df.columns[0]: 'state_id'})
panda_df = panda_df.rename(columns={panda_df.columns[0]: 'state_id'})
workplace_df = workplace_df.rename(columns={workplace_df.columns[0]: 'state_id'})

# 按 'state_id' 合并数据
merged_df = death_df.merge(panda_df, on='state_id').merge(workplace_df, on='state_id')

# 转换时间序列数据为 numpy 数组
death_ts = merged_df[death_df.columns[1:]].values.astype(np.float32)
workplace_ts = merged_df[workplace_df.columns[1:]].values.astype(np.float32)

# 静态特征：人口密度和60岁以上老人比例（假设列名分别为 'population density P' 和 'proportion A'）
static_features = merged_df[['population density P', 'proportion A']].values.astype(np.float32)

merged_df.to_csv("me1.csv", index=False)

# -------------------------------
# 划分时间序列：前60%用于 FPCA，后40%用于预测
# -------------------------------
T_total = death_ts.shape[1]
T1 = int(T_total * 0.6)  # 前60%时间步
T2 = T_total - T1  # 后40%时间步

# 前60%数据用于 FPCA 特征提取，后40%作为预测目标
death_input = death_ts[:, :T1]  # 形状: (num_counties, T1)
death_target = death_ts[:, T1:]  # 形状: (num_counties, T2)

# -------------------------------
# 将郡随机划分为 70% 训练集和 30% 测试集
# -------------------------------
num_counties = death_ts.shape[0]
indices = np.arange(num_counties)
np.random.shuffle(indices)
train_size = int(0.7 * num_counties)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

death_input_train = death_input[train_indices, :]  # 训练集 FPCA输入
death_target_train = death_target[train_indices, :]  # 训练集预测目标
death_input_test = death_input[test_indices, :]
death_target_test = death_target[test_indices, :]

# -------------------------------
# FPCA 在训练数据上进行拟合
# -------------------------------
# 假设时间网格均匀分布于 0 到 1
grid_points = np.linspace(0, 1, T1)
fd_train = FDataGrid(data_matrix=death_input_train, grid_points=grid_points)

# 设置 FPCA 主成分个数（可调）
n_components = 10

# 拟合 FPCA
fpca = FPCA(n_components=n_components)
fpca.fit(fd_train)
# 将训练数据转换为 FPCA 得分
scores_train = fpca.transform(fd_train)  # 形状: (train_size, n_components)

# -------------------------------
# 建立回归模型进行预测
# -------------------------------
regressor = LinearRegression()
regressor.fit(scores_train, death_target_train)

# -------------------------------
# 对测试数据进行 FPCA 得分转换及预测
# -------------------------------
fd_test = FDataGrid(data_matrix=death_input_test, grid_points=grid_points)
scores_test = fpca.transform(fd_test)
predictions_test = regressor.predict(scores_test)  # 形状: (num_test, T2)

num_test = death_input_test.shape[0]
print("绘制测试集中所有郡的FPCA预测曲线")
for j in range(num_test):
    # 获取全局索引，便于从原始数据中提取对应样本
    global_index = test_indices[j]

    plt.figure(figsize=(10, 6))
    # 提取完整的死亡时间序列（全时间步）
    full_death = death_ts[global_index, :]
    plt.plot(range(T_total), full_death, label='Observed Patients Count', marker='o')
    # 使用在测试集中对应位置的真实后40%数据
    plt.plot(range(T1, T_total), death_target_test[j], label='True Patients Count (Last 40%)', linestyle='--')
    # 使用预测结果中对应位置的数据
    plt.plot(range(T1, T_total), predictions_test[j], label='Predicted Patients Count', marker='o', linestyle='--')

    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Death Count')
    # 获取当前样本对应的 state_id
    state_id = merged_df["state_id"].iloc[global_index]
    plt.title(f'State {state_id} (FPCA Prediction)')
    plt.legend()
    plt.grid(True)

    # -------------------------------
    # 保存图片到指定文件夹下
    # -------------------------------
    save_dir = "fig/patients/state/fpca"
    save_path = os.path.join(save_dir, f"patients_state_{state_id}_fpca.png")  # 定义保存路径
    plt.savefig(save_path)  # 保存当前图像
    plt.show()

