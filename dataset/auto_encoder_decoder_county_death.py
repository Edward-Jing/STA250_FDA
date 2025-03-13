import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import random

# -------------------------------
# 设置随机种子以确保结果可复现
# -------------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# -------------------------------
# 数据读取及预处理
# -------------------------------
# Load CSV files
death_df = pd.read_csv('death_county.csv')
panda_df = pd.read_csv('P_and_A_county.csv')
workplace_df = pd.read_csv('workplace.csv')

# 用0替换workplace_df中的缺失值
workplace_df = workplace_df.fillna(0)

# 重命名第一列为 'county_id'
death_df = death_df.rename(columns={death_df.columns[0]: 'county_id'})
panda_df = panda_df.rename(columns={panda_df.columns[0]: 'county_id'})
workplace_df = workplace_df.rename(columns={workplace_df.columns[0]: 'county_id'})

# 按'county_id'合并数据
merged_df = death_df.merge(panda_df, on='county_id').merge(workplace_df, on='county_id')

# 转换时间序列数据为numpy数组
death_ts = merged_df[death_df.columns[1:]].values.astype(np.float32)
workplace_ts = merged_df[workplace_df.columns[1:]].values.astype(np.float32)

# 静态特征：人口密度和60岁以上老人比例（假设列名分别为 'population density P' 和 'proportion A'）
static_features = merged_df[['population density P', 'proportion A']].values.astype(np.float32)

merged_df.to_csv("me1.csv", index=False)

# -------------------------------
# 划分时间序列：前60%用于编码器训练，后40%用于预测
# -------------------------------
T_total = death_ts.shape[1]
T1 = int(T_total * 0.6)  # 前60%的时间步
T2 = T_total - T1  # 后40%的时间步

# 自编码阶段使用前60%的死亡数据作为重构目标
death_input = death_ts[:, :T1]
# 预测阶段使用后40%的死亡数据作为预测目标
death_target = death_ts[:, T1:]
# workplace数据取前T1个时间步
workplace_input = workplace_ts[:, :T1]

# 拼接输入特征：death_input + workplace_input + 静态特征
# 输入维度 = T1 (death) + T1 (workplace) + 2 (静态特征)
X = np.concatenate([death_input, workplace_input, static_features], axis=1)  # shape: (num_samples, 2*T1 + 2)


# -------------------------------
# 定义Dataset
# -------------------------------
class CountyDataset(Dataset):
    def __init__(self, X, death_input, death_target):
        # 将numpy数组转换为torch张量
        self.X = torch.from_numpy(X)
        self.death_input = torch.from_numpy(death_input)
        self.death_target = torch.from_numpy(death_target)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.death_input[idx], self.death_target[idx]


dataset = CountyDataset(X, death_input, death_target)
# DataLoader用于阶段一训练全体数据
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# -------------------------------
# 定义模型组件
# -------------------------------
# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x  # 返回潜在特征


# 自编码解码器：用于重构前60%的死亡数据
class ReconstructionDecoder(nn.Module):
    def __init__(self, hidden_dim, recon_output_dim):
        super(ReconstructionDecoder, self).__init__()
        # Decoder for reconstructing the input (first T1 time steps)
        self.fc = nn.Linear(hidden_dim, recon_output_dim)

    def forward(self, latent):
        out = self.fc(latent)
        return out


# 预测解码器：用于预测后40%的死亡数据
class ForecastingDecoder(nn.Module):
    def __init__(self, hidden_dim, forecast_output_dim):
        super(ForecastingDecoder, self).__init__()
        # Decoder for forecasting future (last T2 time steps)
        self.fc = nn.Linear(hidden_dim, forecast_output_dim)

    def forward(self, latent):
        out = self.fc(latent)
        return out


# -------------------------------
# 初始化模型组件
# -------------------------------
input_dim = X.shape[1]  # 2*T1 + 2
hidden_dim = 128  # 超参数，可根据需要调整
recon_output_dim = T1  # 重构目标维度
forecast_output_dim = T2  # 预测目标维度

encoder = Encoder(input_dim, hidden_dim)
recon_decoder = ReconstructionDecoder(hidden_dim, recon_output_dim)
forecast_decoder = ForecastingDecoder(hidden_dim, forecast_output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)
recon_decoder = recon_decoder.to(device)
forecast_decoder = forecast_decoder.to(device)

# -------------------------------
# 阶段一：使用自编码任务训练编码器
# -------------------------------
num_epochs_stage1 = 300
criterion = nn.MSELoss()
optimizer_stage1 = optim.Adam(list(encoder.parameters()) + list(recon_decoder.parameters()), lr=0.001)

print("阶段一：训练编码器和自编码解码器（仅使用前60%的数据重构）")
for epoch in range(num_epochs_stage1):
    encoder.train()
    recon_decoder.train()
    epoch_loss = 0
    for batch_X, batch_death_input, _ in dataloader:
        batch_X = batch_X.to(device)
        batch_death_input = batch_death_input.to(device)
        optimizer_stage1.zero_grad()
        latent = encoder(batch_X)
        recon_output = recon_decoder(latent)
        loss = criterion(recon_output, batch_death_input)
        loss.backward()
        optimizer_stage1.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch + 1}/{num_epochs_stage1}, Reconstruction Loss: {epoch_loss:.4f}")

# -------------------------------
# 阶段二：冻结编码器，仅训练预测解码器
# 并对郡级数据划分训练集(70%)与测试集(30%)
# -------------------------------
# 冻结编码器参数
for param in encoder.parameters():
    param.requires_grad = False

# 随机划分70%训练和30%测试的郡
num_counties = len(dataset)
indices = np.arange(num_counties)
np.random.shuffle(indices)
train_size = int(0.7 * num_counties)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_epochs_stage2 = 300
optimizer_stage2 = optim.Adam(forecast_decoder.parameters(), lr=0.001)

print("阶段二：训练预测解码器（仅使用训练集70%的郡进行预测训练）")
for epoch in range(num_epochs_stage2):
    encoder.eval()  # 保证编码器处于冻结状态
    forecast_decoder.train()
    epoch_loss = 0
    for batch_X, _, batch_death_target in train_dataloader:
        batch_X = batch_X.to(device)
        batch_death_target = batch_death_target.to(device)
        optimizer_stage2.zero_grad()
        latent = encoder(batch_X)
        forecast_output = forecast_decoder(latent)
        loss = criterion(forecast_output, batch_death_target)
        loss.backward()
        optimizer_stage2.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(train_dataset)
    print(f"Epoch {epoch + 1}/{num_epochs_stage2}, Forecasting Loss: {epoch_loss:.4f}")

# -------------------------------
# 模型预测及结果绘图（在测试集中随机选取5个郡进行绘图）
# -------------------------------
encoder.eval()
forecast_decoder.eval()
# 从测试集中提取所有样本数据
X_test = []
death_ts_test = []
for idx in test_indices:
    sample_X, _, _ = dataset[idx]
    X_test.append(sample_X.numpy())
    death_ts_test.append(death_ts[idx])
X_test = np.stack(X_test, axis=0)  # shape: (num_test, input_dim)
death_ts_test = np.stack(death_ts_test, axis=0)  # 原始死亡时间序列

with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).to(device)
    latent_test = encoder(X_test_tensor)
    test_predictions = forecast_decoder(latent_test).cpu().numpy()  # shape: (num_test, T2)

# 随机选择5个测试集中的郡进行绘图
num_test = X_test.shape[0]
selected = np.random.choice(np.arange(num_test), size=5, replace=False)

target_counties = [17167, 27015, 21149, 13033, 21209]

print("绘制测试集中随机5个郡的预测曲线")
for county in target_counties:
    # 查找 county_id 等于目标值的样本索引
    target_indices = merged_df.index[merged_df['county_id'] == county].tolist()

    if len(target_indices) == 0:
        print(f"未找到 county {county} 的数据。")
        continue  # 若未找到，则跳过当前郡

    target_idx = target_indices[0]  # 获取第一个匹配项

    # 提取目标样本的输入特征和死亡时间序列数据
    X_target = X[target_idx]  # 输入特征，形状为 (2*T1 + 2,)
    death_ts_target = death_ts[target_idx]  # 原始死亡时间序列，形状为 (T_total,)

    # 将输入特征转换为 tensor，并增加 batch 维度
    X_target_tensor = torch.from_numpy(X_target).unsqueeze(0).to(device)

    # 利用训练好的模型生成预测结果
    encoder.eval()
    forecast_decoder.eval()
    with torch.no_grad():
        latent_target = encoder(X_target_tensor)
        forecast_prediction = forecast_decoder(latent_target).cpu().numpy().flatten()  # 形状为 (T2,)

    # 绘制图像
    plt.figure(figsize=(10, 6))
    # 绘制全部时间步的观测死亡数据
    plt.plot(range(T_total), death_ts_target, label='Observed Death Count', marker='o')
    # 绘制后40%时间步的真实死亡数据
    plt.plot(range(T1, T_total), death_ts_target[T1:], label='True Death Count (Last 40%)', linestyle='--')
    # 绘制后40%时间步的预测死亡数据
    plt.plot(range(T1, T_total), forecast_prediction, label='Predicted Death Count', marker='o', linestyle='--')

    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Death Count')
    plt.title(f'County {county}')
    plt.legend()
    plt.grid(True)
    # 添加代码：保存图片到fig文件夹下
    # -------------------------------
    save_path = f"fig/death/county/auto_decoder/death_county_{county}_auto_decoder.png"  # 定义保存路径
    plt.savefig(save_path)  # 保存当前图像到指定路径
    plt.show()
