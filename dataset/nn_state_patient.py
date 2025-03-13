import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Load CSV files
death_df = pd.read_csv('patients_state.csv')
panda_df = pd.read_csv('P_and_A_state.csv')
workplace_df = pd.read_csv('workplace_state.csv')

# Replace NA in workplace_df with 0
workplace_df = workplace_df.fillna(0)

# Rename the first column to 'county_id' for consistency.
death_df = death_df.rename(columns={death_df.columns[0]: 'state_id'})
panda_df = panda_df.rename(columns={panda_df.columns[0]: 'state_id'})
workplace_df = workplace_df.rename(columns={workplace_df.columns[0]: 'state_id'})

# Merge the dataframes on 'state_id'
merged_df = death_df.merge(panda_df, on='state_id').merge(workplace_df, on='state_id')

# convert the time series matrix
death_ts = merged_df[death_df.columns[1:]].values.astype(np.float32)
workplace_ts = merged_df[workplace_df.columns[1:]].values.astype(np.float32)

# 静态特征：人口密度（population density）和60岁以上老人的比例（proportion_60plus）
# 假设 panda_df 中对应的列名分别为 'population density P' 和 'proportion A'
static_features = merged_df[['population density P', 'proportion A']].values.astype(np.float32)

merged_df.to_csv("me1.csv", index=False)

# -------------------------------
# Split Time Series into Input and Target
# -------------------------------

# death_ts 每一行为一个郡，列数为总时间步数
T_total = death_ts.shape[1]
T1 = int(T_total * 0.6)  # 前60%作为输入和重构目标
T2 = T_total - T1  # 后40%作为预测目标

# 取死亡数的前60%作为模型输入，同时作为重构目标；后40%作为最终预测目标
death_input = death_ts[:, :T1]  # 用于计算loss（重构部分）
death_target = death_ts[:, T1:]  # 用于最终预测评估

# 对于 workplace 数据，同样取前 T1 个时间步作为输入特征
workplace_input = workplace_ts[:, :T1]

# 将时间序列数据进行展平，然后与静态特征拼接
# 每个样本的输入维度 = death_input (T1) + workplace_input (T1) + static_features (2)
X = np.concatenate([death_input, workplace_input, static_features], axis=1)  # shape: (num_samples, 2*T1 + 2)


# -------------------------------
# Define Dataset and DataLoader
# -------------------------------

# 新的Dataset返回X、重构目标（前60%死亡数）和预测目标（后40%死亡数）
class CountyDataset(Dataset):
    def __init__(self, X, death_input, death_target):
        # Convert numpy arrays to torch tensors
        self.X = torch.from_numpy(X)
        self.death_input = torch.from_numpy(death_input)
        self.death_target = torch.from_numpy(death_target)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.death_input[idx], self.death_target[idx]


dataset = CountyDataset(X, death_input, death_target)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# -------------------------------
# Define Neural Network Model with Dual Outputs
# -------------------------------

class DeathPredictionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, recon_output_dim, forecast_output_dim):
        super(DeathPredictionNet, self).__init__()
        # 英文注释: First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # 英文注释: Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 英文注释: Reconstruction branch for the input (first T1 time steps)
        self.fc_recon = nn.Linear(hidden_dim, recon_output_dim)
        # 英文注释: Forecast branch for predicting future (last T2 time steps)
        self.fc_forecast = nn.Linear(hidden_dim, forecast_output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # 英文注释: Obtain reconstruction and forecast outputs
        out_recon = self.fc_recon(x)  # 用于重构前60%时间步
        out_forecast = self.fc_forecast(x)  # 用于预测后40%时间步
        return out_recon, out_forecast


input_dim = X.shape[1]  # 2*T1 + 2
hidden_dim = 64  # 超参数，可根据需要调整
recon_output_dim = T1  # 重构目标维度：前60%时间步
forecast_output_dim = T2  # 预测目标维度：后40%时间步

model = DeathPredictionNet(input_dim, hidden_dim, recon_output_dim, forecast_output_dim)

# -------------------------------
# Set Device for GPU Computation
# -------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# -------------------------------
# Define Loss and Optimizer
# -------------------------------

# 这里只使用重构分支计算loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------

num_epochs = 50  # 根据需要调整训练轮数

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_death_input, _ in dataloader:
        batch_X = batch_X.to(device)
        batch_death_input = batch_death_input.to(device)

        optimizer.zero_grad()
        # 英文注释: Forward pass returns reconstruction and forecast outputs
        recon_pred, _ = model(batch_X)
        loss = criterion(recon_pred, batch_death_input)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= len(dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 模型训练完成后，即可用训练好的模型对新数据进行预测，
# 输出为对应郡后40%时间步的累计死亡人数预测结果

# -------------------------------
# 预测并绘制各郡的结果图
# -------------------------------

model.eval()
with torch.no_grad():
    X_tensor = torch.from_numpy(X).to(device)
    # 英文注释: 使用预测分支获得后40%的预测结果
    _, predictions = model(X_tensor)
    predictions = predictions.cpu().numpy()

num_counties = death_ts.shape[0]

# 根据需要，若郡数量较多，可选择只绘制部分郡，比如指定一些索引
plot_indices = [ 2,43,35,23,40,10,22,18,49,20,7,42,14,28,38]
for i in plot_indices:
    if i >= num_counties:
        continue  # 避免索引超出范围
    plt.figure(figsize=(10, 6))
    # 英文注释: Plot the observed death counts for the entire time series.
    plt.plot(range(T_total), death_ts[i], label='Observed Patients Count', marker='o')
    # 英文注释: Plot the true death counts for the last 40% time steps.
    plt.plot(range(T1, T_total), death_ts[i, T1:], label='True Death Count (Last 40%)', linestyle='--')
    # 英文注释: Plot the predicted death counts for the last 40% time steps.
    plt.plot(range(T1, T_total), predictions[i], label='Predicted Patients Count', marker='o', linestyle='--')

    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Patients Count')
    plt.title(f'State {merged_df["state_id"].iloc[i]}')
    plt.legend()
    plt.grid(True)
    # 添加代码：保存图片到fig文件夹下
    # -------------------------------
    save_path = f"fig/patients/state/nn/patients_county_'{merged_df['state_id'].iloc[i]}'_nn.png"  # 定义保存路径
    plt.savefig(save_path)  # 保存当前图像到指定路径
    plt.show()
