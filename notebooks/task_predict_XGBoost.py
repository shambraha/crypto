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
    y.append(df.iloc[i+timesteps]['Close'])  
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
print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}')

# Reshape X_train, X_val, X_test từ 3D sang 2D
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test= X_test.reshape(X_test.shape[0], -1)

# Kiểm tra lại shape
print("Sau khi reshape cho phù hợp mới MachineLearning")
print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}')

# #region (1) Dùng XGBoost ------------------------------------------------------------------------------------------ 
# # Khởi tạo ModelManager
# xgb_manager = MachineLearningManager(model_type="xgboost", n_estimators=100, max_depth=6, learning_rate=0.1)

# # Huấn luyện mô hình
# xgb_manager.train(X_train, y_train, X_val, y_val)

# # # Đánh giá mô hình
# metrics = xgb_manager.evaluate(X_test, y_test)
# print(metrics)

# # Lưu mô hình
# xgb_manager.save_model("xgboost_model.pkl")

# # # Dự đoán dữ liệu mới
# # y_pred = xgb_manager.predict(X_test)
# #endregion

#region (2) Dùng random-forest---------------------------------------------
# # Khởi tạo ModelManager cho Random Forest
rf_manager = MachineLearningManager(model_type="random_forest", n_estimators=100, max_depth=10)

# # Huấn luyện mô hình
# rf_manager.train(X_train, y_train)

# # Đánh giá mô hình
# metrics = rf_manager.evaluate(X_test, y_test)
# print(metrics)

# Lưu mô hình
# đã tự động lưu sau khi train nên không cần bước này nữa

# Dự đoán dữ liệu mới
path = r"E:\MyPythonCode\Crypto\notebooks\modelsML\best-RandomForestRegressor-20240927-153944.pkl"
RF_loaded = joblib.load(path)

#2.3.2 Chọn bộ dữ liệu để test
mode = 'test'
if mode == 'test':
    checkX = X_test
    checky = y_test
else:
    checkX = X_val
    checky = y_val
    
#2.3.3 Thay thế cặp X_test, y_test thành X_val, y_val
y_pred = RF_loaded.predict(checkX)        
                
# Kiểm tra hiệu suất của mô hình
mse = mean_squared_error(checky, y_pred)
rmse = mean_squared_error(checky, y_pred, squared=False)
mae = mean_absolute_error(checky, y_pred)
print("Trung bình bình phương của sai số: ", mse)
print("Sai số thực tế (with outliers): ", rmse)
print("Sai số trung bình (without outliers): ", mae)
#endregion

#region (3) Thủ công Random-Forest-------------------------------------------
# Thủ công phải dùng notebooks để xem kết quả và điều chỉnh tiếp liên tục
# RF = RandomForestRegressor(n_estimators=10, max_depth=6)
# RF.fit(X_train, y_train)
# joblib.dump(RF, 'modelsML\\random_forest_model.pkl') # Lưu lại liền

# # Dự đoán trên tập validation
# y_val_pred = RF.predict(X_val)

# # Tính toán RMSE và MAE trên tập validation
# rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
# mae_val = mean_absolute_error(y_val, y_val_pred)

# print(f"Validation RMSE: {rmse_val}")
# print(f"Validation MAE: {mae_val}")

