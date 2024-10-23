import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from function import *
from function_preparation import *
from classModel import *

#region (1) Data Preparation---------------------------------------------------------------
#1.1 Xử lý df
eth_path = r"E:\MyPythonCode\Crypto\data\csv\binance_ETHUSDT.csv"
df = pd.read_csv(eth_path)
df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df = df.resample('1h', on='Open Time').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'Quote Volume': 'sum',
    'Number of Trades': 'sum',
    'Taker Buy Base Volume': 'sum',
    'Taker Buy Quote Volume': 'sum',
}).dropna()
df['SMA_10'] = calculate_sma(df, period=10)
df['EMA_10'] = calculate_ema(df, period=10)
df['RSI_14'] = calculate_rsi(df, period=14)
df['MACD'], df['Signal_Line'] = calculate_macd(df)
df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df, period=20)
df['ATR'] = calculate_atr(df, period=14)
df['Stochastic_Oscillator'] = calculate_stochastic_oscillator(df, period=14)
df['OBV'] = calculate_obv(df)
df['ROC'] = calculate_roc(df, period=12)
df['CMF'] = calculate_cmf(df, period=20)
df['ADX_14'] = calculate_adx(df, period=14)
df['SAR'] = calculate_sar(df, acceleration=0.02, maximum=0.2)
df = df.dropna()
df = df.iloc[20:].reset_index(drop=True)
# print(df)
# df.to_excel('file_name.xlsx', index=False)

#1.2 Tạo đầu vào cho mô hình
timesteps = 24 * 7  # 7 ngày, mỗi ngày 24 giờ
future_steps = 4  # Dự đoán 4 bước thời gian tiếp theo
X = []
y = []

# Giả sử df có các cột: ['Open', 'High', 'Low', 'Close', 'Volume']
for i in range(len(df) - timesteps - future_steps):
    # Lấy toàn bộ các đặc trưng (trong 7 ngày)
    X.append(df.iloc[i:i + timesteps].values)

    # Lấy 4 bước thời gian tiếp theo cho các cột 'Close' và 'Volume' làm y
    y.append(df.iloc[i + timesteps:i + timesteps + future_steps][['Close', 'Volume']].values)

# Chuyển đổi X và y thành numpy array để dùng cho LSTM
X = np.array(X)
y = np.array(y)
print_comboXy_shape(X, y)
# example_idx = 3
# print_comboXy_in_index(X, y, example_idx)

X_tensor = torch.tensor(X, dtype=torch.float32)  # Đầu vào X (kích thước [samples, timesteps, features])
y_tensor = torch.tensor(y, dtype=torch.float32)  # Nhãn y (kích thước [samples,])

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

#1.3 Kiểm tra kết quả chuẩn bị
# check_dataset_splits(dataset, train_dataset, val_dataset, test_dataset)
# check_batch_in_loader(train_loader) # print ra khá dài
# check_unique_labels_in_dataset(y_tensor)
# check_label_distribution(train_dataset, val_dataset, test_dataset)
# check_dataloader_functionality(train_loader, test_loader)
#endregion

#region (2) Model Preparation-----------------------------------------------------------
# Khởi tạo model LSTM Regression
input_size = X.shape[2]  # Số lượng đặc trưng đầu vào
hidden_size = 64         # Kích thước của hidden layer
output_size = 2          # Số lượng feature đầu ra
num_layers = 2           # Số lớp LSTM
ahead = 4               # Dự đoán 4 bước thời gian
dropout = 0.2
model_LSTM_regression = LSTM_Regression(input_size, hidden_size, output_size, num_layers, ahead, dropout)
print("Thông tin về Model Regression: \n", model_LSTM_regression)
# test_model_on_batch(model_LSTM_regression, train_loader)
#endregion

# #region (3) Cụm Code kiểm tra outputs.shape & targets.shape-----------------------------
count = 0  # Khởi tạo biến đếm
for inputs, targets in train_loader:
    # In giá trị của targets
    # print("Targets:", targets)
    print("Targets.shape: ", targets.shape)
    # targets = targets.long()

    outputs = model_LSTM_regression(inputs)
    print("Outputs.shape : ", outputs.shape)
    # outputs đã được .squeeze() ở lúc tạo model LSTM

    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, targets)
    print(loss)

    # Tiến hành các bước tiếp theo...
    count += 1  # Tăng biến đếm lên 1
    if count >= 1:  # Dừng sau khi thưc hiện vòng lặp 1 lần
        break
# #endregion

#region (4) ModelManager Preparation-----------------------------------------------------
#1. Chọn 1 trong 2 cách cập nhật learning rate
optimizer, scheduler_reduce_lr, scheduler_step_lr = create_optimizer_and_schedulers(model_LSTM_regression)

#2. Khởi tạo class manager cho Regression
MM_Regression = ModelManagerRegression(
    model=model_LSTM_regression,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    lr=0.001,
    patience=50,
    criterion=torch.nn.MSELoss(),
    ahead=ahead
)
#print("Thông tin về ModelManager Classification: \n", MM_Regression)
#endregion

#region (5) First Execution----------------------------------------------------------------------
#5.1 Huấn luyện mô hình Regression không dùng scheduler
MM_Regression.train(num_epochs=1, save_dir='models')

#5.2 Huấn luyện mô hình Regression với scheduler
# MM_Regression.train(num_epochs=50, save_dir='models', scheduler=scheduler_reduce_lr)
#endregion