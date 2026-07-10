# ====================== 【代码第一行，优先配置HF国内镜像】 ======================
import os

# 强制使用hf镜像域名，避免访问国外地址超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "./hf_datasets_cache"
# 屏蔽软链接红色警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ====================== 全部依赖库导入 ======================
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ====================== 全局配置 ======================
# 固定随机种子，保证实验可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# 设备自动识别
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")
# 创建存储文件夹
dir_list = ["./data", "./models", "./plots", "./hf_datasets_cache"]
for d in dir_list:
    if not os.path.exists(d):
        os.makedirs(d)
# 绘图中文支持
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 时序超参数
HISTORY_WINDOW = 8
PRED_STEP = 1
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
# 数据集地址与对应高度
dataset_info = [
    {"url": "hf://datasets/Antajitters/WindSpeed_10m/", "height": 10},
    {"url": "hf://datasets/Antajitters/WindSpeed_50m/", "height": 50},
    {"url": "hf://datasets/Antajitters/WindSpeed_100m/", "height": 100}
]
splits = {'train': 'data/train-00000-of-00001.parquet', 'val': 'data/val-00000-of-00001.parquet',
          'test': 'data/test-00000-of-00001.parquet'}
# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


# ====================== 1. 读取并合并三组风速数据（你提供的三段读取代码整合） ======================
def load_and_merge_data():
    df_list = []
    for info in dataset_info:
        print(f"正在在线读取数据集：{info['url']}")
        # 你给出的读取代码，只读取train训练集
        df_single = pd.read_parquet(info["url"] + splits["train"])
        # 新增高度标签区分数据集
        df_single["height"] = info["height"]
        df_list.append(df_single)
        print(f"{info['height']}m 数据读取完成，数据量：{len(df_single)}")

    # 合并10/50/100米全部数据
    total_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("全部列名：", total_df.columns.tolist())
    # 时间字段转换、按时间升序排序
    total_df["Date & Time Stamp"] = pd.to_datetime(total_df["Date & Time Stamp"])
    total_df = total_df.sort_values("Date & Time Stamp").reset_index(drop=True)
    print(f"数据集合并完成，总行数：{total_df.shape[0]}，总列数：{total_df.shape[1]}")
    print("前5行预览：")
    print(total_df.head())
    return total_df


# ====================== 2. 数据清洗（匹配数据集真实字段） ======================
def data_clean(raw_df):
    keep_cols = [
        "Date & Time Stamp",
        "SpeedAvg",
        "SpeedMax",
        "DirectionAvg",
        "TemperatureAvg",
        "TemperatureMax",
        "PressureAvg",
        "height"
    ]
    df = raw_df[keep_cols].copy()
    print("\n清洗前缺失值统计：")
    print(df.isnull().sum())
    # 数值缺失值用中位数填充
    fill_columns = ["SpeedAvg", "SpeedMax", "DirectionAvg", "TemperatureAvg", "TemperatureMax", "PressureAvg"]
    for col in fill_columns:
        df[col] = df[col].fillna(df[col].median())
    total_null = df.isnull().sum().sum()
    print(f"缺失值填充完成，剩余缺失值总数：{total_null}")
    # IQR剔除异常值
    feature_cols = ["SpeedAvg", "SpeedMax", "DirectionAvg", "TemperatureAvg", "TemperatureMax", "PressureAvg"]
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    df = df.reset_index(drop=True)
    print(f"IQR异常值剔除完成，剩余样本量：{df.shape[0]}")
    return df


# ====================== 3. 特征工程 + 数据集时序划分 ======================
def feature_split_dataset(clean_df):
    df = clean_df.copy()
    # 提取时间衍生特征（hour、month在这里生成）
    df["hour"] = df["Date & Time Stamp"].dt.hour
    df["month"] = df["Date & Time Stamp"].dt.month
    # 输入特征、预测目标（平均风速SpeedAvg）
    feature_columns = ["SpeedMax", "DirectionAvg", "TemperatureAvg", "TemperatureMax", "PressureAvg", "height", "hour",
                       "month"]
    target_col = "SpeedAvg"
    X_raw = df[feature_columns].values
    y_raw = df[target_col].values.reshape(-1, 1)
    # 时序切分（不打乱顺序）
    total_len = len(X_raw)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = train_end + int(total_len * VAL_RATIO)

    X_train_raw = X_raw[:train_end]
    y_train_raw = y_raw[:train_end]
    X_val_raw = X_raw[train_end:val_end]
    y_val_raw = y_raw[train_end:val_end]
    X_test_raw = X_raw[val_end:]
    y_test_raw = y_raw[val_end:]
    # 标准化（仅用训练集均值方差，防止数据泄露）
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    X_val = scaler_x.transform(X_val_raw)
    X_test = scaler_x.transform(X_test_raw)

    y_train = scaler_y.fit_transform(y_train_raw)
    y_val = scaler_y.transform(y_val_raw)
    y_test = scaler_y.transform(y_test_raw)

    print(f"\n数据集划分完成：")
    print(f"训练集 X:{X_train.shape}, y:{y_train.shape}")
    print(f"验证集 X:{X_val.shape}, y:{y_val.shape}")
    print(f"测试集 X:{X_test.shape}, y:{y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y, feature_columns, df


# ====================== 4. 滑动窗口构造时序样本 ======================
def create_sequence_data(X, y, seq_len, pred_len):
    seq_x, seq_y = [], []
    sample_num = len(X)
    for i in range(sample_num - seq_len - pred_len + 1):
        seq_x.append(X[i:i + seq_len])
        seq_y.append(y[i + seq_len:i + seq_len + pred_len])
    seq_x = np.array(seq_x, dtype=np.float32)
    seq_y = np.array(seq_y, dtype=np.float32).squeeze(-1)
    return seq_x, seq_y


# 自定义数据集类
class WindDataset(Dataset):
    def __init__(self, x_arr, y_arr):
        self.x = torch.from_numpy(x_arr)
        self.y = torch.from_numpy(y_arr)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ====================== 5. EDA可视化绘图 ======================
def draw_eda_plot(clean_df, feat_cols):
    # 平均风速分布直方图
    plt.figure(figsize=(10, 5))
    plt.hist(clean_df["SpeedAvg"], bins=50, alpha=0.7, color="#1f77b4", density=True)
    clean_df["SpeedAvg"].plot.kde(color="#d62728", linewidth=2)
    plt.title("清洗后平均风速 SpeedAvg 分布直方图", fontsize=14)
    plt.xlabel("风速 m/s")
    plt.ylabel("密度")
    plt.savefig("./plots/wind_speed_dist.png", dpi=300, bbox_inches="tight")
    plt.close()
    # 特征相关性热力图
    corr_list = feat_cols + ["SpeedAvg"]
    corr_mat = clean_df[corr_list].corr()
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr_mat.columns)), corr_mat.columns, rotation=45)
    plt.yticks(range(len(corr_mat.columns)), corr_mat.columns)
    # 标注相关系数
    for i in range(len(corr_mat.columns)):
        for j in range(len(corr_mat.columns)):
            plt.text(j, i, f"{corr_mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("特征相关性热力图", fontsize=14)
    plt.savefig("./plots/corr_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("EDA可视化图片已保存至 ./plots 文件夹")


# ====================== 6. 模型定义 & 训练函数 ======================
# 线性回归训练
def train_linear_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler_y):
    train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
    lr_model = LinearRegression()
    lr_model.fit(train_flat, y_train_seq)
    y_pred_scaled = lr_model.predict(test_flat)
    # 反标准化还原真实风速
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).squeeze()
    y_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).squeeze()
    # 评价指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("\n===== 线性回归模型评价指标 =====")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    # 绘制预测对比曲线
    show_num = 200
    plt.figure(figsize=(12, 5))
    plt.plot(range(show_num), y_true[:show_num], label="真实SpeedAvg风速", c="#1f77b4", linewidth=2)
    plt.plot(range(show_num), y_pred[:show_num], label="预测SpeedAvg风速", c="#9467bd", linestyle="--", linewidth=2)
    plt.title("线性回归-测试集真实值vs预测值曲线", fontsize=14)
    plt.xlabel("时序样本序号")
    plt.ylabel("风速 m/s")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("./plots/linear_predict.png", dpi=300, bbox_inches="tight")
    plt.close()
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, y_pred, y_true


# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layer_num=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last_out = out[:, -1, :]
        pred = self.fc(last_out)
        return pred.squeeze(-1)


def train_lstm_model(train_loader, val_loader, test_x_seq, test_y_seq, input_dim, scaler_y):
    model = LSTMModel(input_dim=input_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if epoch % 5 == 0:
            print(f"[LSTM Epoch {epoch:2d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    # 保存权重
    torch.save(model.state_dict(), "./models/lstm_model.pth")
    # 测试集预测
    model.eval()
    test_x_tensor = torch.from_numpy(test_x_seq).to(device)
    with torch.no_grad():
        pred_scaled = model(test_x_tensor).cpu().numpy()
    # 反标准化
    y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).squeeze()
    y_true = scaler_y.inverse_transform(test_y_seq.reshape(-1, 1)).squeeze()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("\n===== LSTM模型评价指标 =====")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    # 绘图
    show_num = 200
    plt.figure(figsize=(12, 5))
    plt.plot(range(show_num), y_true[:show_num], label="真实SpeedAvg风速", c="#1f77b4", linewidth=2)
    plt.plot(range(show_num), y_pred[:show_num], label="LSTM预测风速", c="#9467bd", linestyle="--", linewidth=2)
    plt.title("LSTM-测试集真实值vs预测值曲线", fontsize=14)
    plt.xlabel("时序样本序号")
    plt.ylabel("风速 m/s")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("./plots/lstm_predict.png", dpi=300, bbox_inches="tight")
    plt.close()
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, y_pred, y_true


# Transformer时序模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embed(x)
        feat = self.encoder(x)
        last_feat = feat[:, -1, :]
        out = self.head(last_feat)
        return out.squeeze(-1)


def train_transformer_model(train_loader, val_loader, test_x_seq, test_y_seq, input_dim, scaler_y):
    model = TransformerModel(input_dim=input_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if epoch % 5 == 0:
            print(f"[Transformer Epoch {epoch:2d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), "./models/transformer_model.pth")
    # 测试预测
    model.eval()
    test_x_tensor = torch.from_numpy(test_x_seq).to(device)
    with torch.no_grad():
        pred_scaled = model(test_x_tensor).cpu().numpy()
    y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).squeeze()
    y_true = scaler_y.inverse_transform(test_y_seq.reshape(-1, 1)).squeeze()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("\n===== Transformer模型评价指标 =====")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    show_num = 200
    plt.figure(figsize=(12, 5))
    plt.plot(range(show_num), y_true[:show_num], label="真实SpeedAvg风速", c="#1f77b4", linewidth=2)
    plt.plot(range(show_num), y_pred[:show_num], label="Transformer预测风速", c="#9467bd", linestyle="--", linewidth=2)
    plt.title("Transformer-测试集真实值vs预测值曲线", fontsize=14)
    plt.xlabel("时序样本序号")
    plt.ylabel("风速 m/s")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("./plots/transformer_predict.png", dpi=300, bbox_inches="tight")
    plt.close()
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, y_pred, y_true


# ====================== 主运行入口【已修复执行顺序】 ======================
if __name__ == "__main__":
    print("========== 多高度风速时序预测实验（pandas hf://在线读取数据集）==========")
    # 1. 读取合并数据（使用你提供的三段parquet读取代码）
    raw_data = load_and_merge_data()
    # 2. 清洗数据
    clean_data = data_clean(raw_data)
    # 3. 先运行特征工程，生成hour、month列（关键修复点）
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y, feat_cols, full_df = feature_split_dataset(
        clean_data)
    # 4. 再执行EDA绘图，传入full_df（包含hour/month）
    draw_eda_plot(full_df, feat_cols)
    # 5. 构造时序滑动窗口样本
    train_seq_x, train_seq_y = create_sequence_data(X_train, y_train, HISTORY_WINDOW, PRED_STEP)
    val_seq_x, val_seq_y = create_sequence_data(X_val, y_val, HISTORY_WINDOW, PRED_STEP)
    test_seq_x, test_seq_y = create_sequence_data(X_test, y_test, HISTORY_WINDOW, PRED_STEP)
    input_dim = train_seq_x.shape[-1]
    print(f"\n时序样本维度：")
    print(f"训练集序列 X:{train_seq_x.shape}, y:{train_seq_y.shape}")
    print(f"验证集序列 X:{val_seq_x.shape}, y:{val_seq_y.shape}")
    print(f"测试集序列 X:{test_seq_x.shape}, y:{test_seq_y.shape}")
    # 6. 构建dataloader给深度学习模型
    train_dataset = WindDataset(train_seq_x, train_seq_y)
    val_dataset = WindDataset(val_seq_x, val_seq_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 7. 训练三类模型
    print("\n---------- 开始训练：线性回归基准模型 ----------")
    linear_metrics, _, _ = train_linear_model(train_seq_x, train_seq_y, test_seq_x, test_seq_y, scaler_y)

    print("\n---------- 开始训练：LSTM 模型 ----------")
    lstm_metrics, _, _ = train_lstm_model(train_loader, val_loader, test_seq_x, test_seq_y, input_dim, scaler_y)

    print("\n---------- 开始训练：Transformer 模型 ----------")
    trans_metrics, _, _ = train_transformer_model(train_loader, val_loader, test_seq_x, test_seq_y, input_dim, scaler_y)

    # 8. 汇总所有模型指标对比表格
    print("\n==================== 全部模型指标汇总对比 ====================")
    print(f"{'模型名称':<12}{'MSE':<10}{'RMSE':<10}{'MAE':<10}{'R²':<10}")
    print("-" * 52)
    print(
        f"{'线性回归':<12}{linear_metrics['mse']:<10.4f}{linear_metrics['rmse']:<10.4f}{linear_metrics['mae']:<10.4f}{linear_metrics['r2']:<10.4f}")
    print(
        f"{'LSTM':<12}{lstm_metrics['mse']:<10.4f}{lstm_metrics['rmse']:<10.4f}{lstm_metrics['mae']:<10.4f}{lstm_metrics['r2']:<10.4f}")
    print(
        f"{'Transformer':<12}{trans_metrics['mse']:<10.4f}{trans_metrics['rmse']:<10.4f}{trans_metrics['mae']:<10.4f}{trans_metrics['r2']:<10.4f}")
    print("实验全部完成！图表、模型权重已保存至对应文件夹")