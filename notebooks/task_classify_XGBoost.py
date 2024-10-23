import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from function import *
from function_preparation import *
from classML import *
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

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
# Tạo nhãn 0 hoặc 1 dựa trên sự thay đổi của giá trị 'Close'
df['y'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()
df = df.iloc[20:].reset_index(drop=True)
# print(df)
df.to_excel('file_name.xlsx', index=False)

#1.2 Tạo đầu vào cho mô hình
timesteps = 24*7 
X = []
y = []
for i in range(len(df) - timesteps):
    # Giống như việc quét chọn 1 khu vực trong bảng tính, chỉ định index từ đâu đến đâu    

    # Lấy các cột đặc trưng làm X
    X.append(df.iloc[i:i+timesteps].values)  
    # Tất cả cột là đầu vào
    # *** Giải thích ***
    # [tức là lấy hàng từ (i) đến (i+timesteps-1), và cột giá trị là cả df
        
    # Lấy cột 'Label' ở vị trí i+timesteps làm y
    y.append(df.iloc[i+timesteps]['y'])  
    # *** Giải thích ***
    # [tức là lấy hàng (i+timesteps), và cột giá trị là 'Close']

X, y = np.array(X), np.array(y)
print_comboXy_shape(X, y)

# Chia dữ liệu thành train (70%), validation (15%), và test (15%)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

# Chia ra train, validation, và test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, shuffle=False)
# print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}')
# check_label_distribution_Xy(y_train, y_val, y_test)

# Reshape X_train, X_val, X_test từ 3D sang 2D
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test= X_test.reshape(X_test.shape[0], -1)

# Kiểm tra lại shape
print("Sau khi reshape cho phù hợp mới MachineLearning")
print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}')

# #1.1-------------------------------------------------------
# from sklearn.ensemble import RandomForestClassifier

# # Huấn luyện mô hình RandomForest cho bài toán phân loại
# RF_classifier = RandomForestClassifier(n_estimators=500, max_depth=10)
# RF_classifier.fit(X_train, y_train)

# #1.2 -----------------------------------------------------
# # Dự đoán với mô hình RandomForest
# y_pred = RF_classifier.predict(X_test)

# #1.3 Kiểm tra và đánh giá
# # Gọi hàm evaluate_model
# metrics = evaluate_model(y_test, y_pred)


# y_pred = RF_classifier.predict(X_val)
# metrics = evaluate_model(y_val, y_pred)

# #2.1------------------------------------------------------
# from xgboost import XGBClassifier

# Hoặc huấn luyện với XGBoost cho bài toán phân loại (chưa chèn 6 tham số)
XGB_classifier = XGBClassifier(n_estimators=50, 
                               max_depth=3, 
                               learning_rate=0.1,
                               reg_alpha=0.5,
                               reg_lambda=20, 
                               use_label_encoder=False, 
                               eval_metric='logloss')
eval_set = [(X_train, y_train)]
eval_set.append((X_val, y_val))
XGB_classifier.fit(X_train, y_train,
                   eval_set=eval_set,  # Đặt tập đánh giá để theo dõi
                    verbose=True)  # Hiển thị thông tin trong quá trình huấn luyện)

#2.2 ----------------------------------------------------
# Dự đoán với mô hình XGBoost
y_pred = XGB_classifier.predict(X_test)

#2.3 Kiểm tra và đánhgias
# Gọi hàm evaluate_model
metrics = evaluate_model(y_test, y_pred)

