# ===================== 1. 导入依赖库 =====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')
# 全局基础配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# 项目路径自动创建
DATA_DIR = "./data"
MODEL_DIR = "./models"
PLOT_DIR = "./plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
# ===================== 2. 加载并拼接三份高度风速数据集 =====================
def load_and_merge_data():
    ds_10m = load_dataset("Antajitters/WindSpeed_10m", split="train")
    ds_50m = load_dataset("Antajitters/WindSpeed_50m", split="train")
    ds_100m = load_dataset("Antajitters/WindSpeed_100m", split="train")
    df10 = ds_10m.to_pandas()
    df50 = ds_50m.to_pandas()
    df100 = ds_100m.to_pandas()
    df10["height"] = 10
    df50["height"] = 50
    df100["height"] = 100
    df = pd.concat([df10, df50, df100], ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print(f"合并总数据量: {df.shape[0]} 行, {df.shape[1]} 列")
    print("数据前5行：")
    print(df.head())
    return df
# ===================== 3. 数据清洗：缺失值+IQR异常值剔除 =====================
def data_clean(df):
    df_clean = df.copy()
    fea_cols = ["Wind Speed", "Wind Direction", "Temperature", "Pressure", "Humidity", "height"]
    target_col = "Wind Speed"
    df_clean = df_clean[fea_cols + ["Timestamp"]]
    # 缺失值填充
    print(f"清洗前缺失值统计：\n{df_clean.isnull().sum()}")
    for col in fea_cols:
        if df_clean[col].isnull().sum() > 0:
            med = df_clean[col].median()
            df_clean[col].fillna(med, inplace=True)
    print(f"缺失值处理完毕，剩余缺失值总数：{df_clean.isnull().sum().sum()}")
    # IQR剔除异常值
    def filter_outlier(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return series[(series >= lower) & (series <= upper)]
    for col in fea_cols:
        df_clean = df_clean[df_clean[col].isin(filter_outlier(df_clean[col]))]
    df_clean = df_clean.reset_index(drop=True)
    print(f"异常值剔除后剩余数据量：{df_clean.shape[0]} 行")
    return df_clean, fea_cols, target_col
# ===================== 4. 特征工程 + 时序7:2:1时间分割 =====================
def feature_eng_and_split(df_clean, fea_cols, target_col):
    df_fea = df_clean.copy()
    # 构造时间衍生特征
    df_fea["hour"] = df_fea["Timestamp"].dt.hour
    df_fea["month"] = df_fea["Timestamp"].dt.month
    fea_cols.extend(["hour", "month"])
    # 标准化
    scaler = StandardScaler()
    X = df_fea[fea_cols].values
    y = df_fea[target_col].values.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X)
    # 时序严格按时间分割，禁止随机打乱
    total = len(X_scaled)
    train_split = int(total * 0.7)
    val_split = int(total * 0.2)
    test_split = total - train_split - val_split
    X_train, y_train = X_scaled[:train_split], y[:train_split]
    X_val, y_val = X_scaled[train_split:train_split+val_split], y[train_split:train_split+val_split]
    X_test, y_test = X_scaled[train_split+val_split:], y[train_split+val_split:]
    print(f"训练集:{X_train.shape}, 验证集:{X_val.shape}, 测试集:{X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, fea_cols
# ===================== 5. 构造时序样本（滑动窗口） =====================
def create_seq_dataset(X, y, seq_len=8, pred_len=1):
    seq_X, seq_y = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        seq_X.append(X[i:i+seq_len])
        seq_y.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(seq_X), np.array(seq_y)
# ===================== 6. 自定义数据集加载器 =====================
class WindTimeDataset(Dataset):
    def __init__(self, x_arr, y_arr):
        self.x = torch.FloatTensor(x_arr)
        self.y = torch.FloatTensor(y_arr)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
# ===================== 7. 模型1：线性回归 =====================
def train_linear_reg(X_train_seq, y_train_seq, X_test_seq, y_test_seq):
    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
    y_train_flat = y_train_seq.reshape(y_train_seq.shape[0], -1)
    y_test_flat = y_test_seq.reshape(y_test_seq.shape[0], -1)
    lr_model = LinearRegression()
    lr_model.fit(X_train_flat, y_train_flat)
    y_pred = lr_model.predict(X_test_flat)
    mse = mean_squared_error(y_test_flat, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred)
    r2 = r2_score(y_test_flat, y_pred)
    res_dict = {"MSE": round(mse,4), "RMSE": round(rmse,4), "MAE": round(mae,4), "R2": round(r2,4)}
    print("===== 线性回归评估指标 =====")
    print(res_dict)
    return lr_model, y_pred, res_dict
# ===================== 8. 模型2：LSTM =====================
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.out_fc(out[:, -1, :])
def train_lstm_model(train_loader, val_loader, test_loader, input_dim, pred_len):
    model = LSTMNet(input_dim=input_dim, hidden_dim=64, layer_num=2, pred_len=pred_len).to(DEVICE)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epoch_num = 20
    for epoch in range(epoch_num):
        model.train()
        train_loss_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).squeeze(-1)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_func(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).squeeze(-1)
                pred = model(batch_x)
                val_loss_total += loss_func(pred, batch_y).item()
        if (epoch + 1) % 5 == 0:
            print(f"LSTM Epoch{epoch+1} | TrainLoss:{train_loss_total/len(train_loader):.4f} | ValLoss:{val_loss_total/len(val_loader):.4f}")
    # 测试集预测
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            pred_val = model(batch_x).cpu().numpy()
            true_val = batch_y.squeeze(-1).cpu().numpy()
            y_pred_list.extend(pred_val)
            y_true_list.extend(true_val)
    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)
    res_dict = {"MSE": round(mse,4), "RMSE": round(rmse,4), "MAE": round(mae,4), "R2": round(r2,4)}
    print("===== LSTM评估指标 =====")
    print(res_dict)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pth"))
    return model, y_pred_arr, res_dict
# ===================== 9. 模型3：Transformer =====================
class TransformerNet(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, layer_num=2, pred_len=1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, layer_num)
        self.output_linear = nn.Linear(d_model, pred_len)
    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        return self.output_linear(x[:, -1, :])
def train_transformer_model(train_loader, val_loader, test_loader, input_dim, pred_len):
    model = TransformerNet(input_dim=input_dim, pred_len=pred_len).to(DEVICE)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epoch_num = 20
    for epoch in range(epoch_num):
        model.train()
        train_loss_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).squeeze(-1)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_func(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).squeeze(-1)
                pred = model(batch_x)
                val_loss_total += loss_func(pred, batch_y).item()
        if (epoch + 1) % 5 == 0:
            print(f"Transformer Epoch{epoch+1} | TrainLoss:{train_loss_total/len(train_loader):.4f} | ValLoss:{val_loss_total/len(val_loader):.4f}")
    model.eval()
    y_true_list, y_p