import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from function import *
from function_preparation import *
from classML import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

#region (1) Data Preparation---------------------------------------------------------------

# def load_and_prepare_data(file_path):
#     df = pd.read_csv(file_path)
#     df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
#     df = df.resample('1h', on='Open Time').agg({
#         'Open': 'first',
#         'High': 'max',
#         'Low': 'min',
#         'Close': 'last',
#         'Volume': 'sum',
#         'Quote Volume': 'sum',
#         'Number of Trades': 'sum',
#         'Taker Buy Base Volume': 'sum',
#         'Taker Buy Quote Volume': 'sum',
#     }).dropna()
#     # Reset lại index để 'Open Time' trở thành cột
#     df = df.reset_index()
#     df = calculate_indicators(df)  # Add indicators
#     df = df.dropna()
#     df = df.iloc[20:].reset_index(drop=True)
#     return df

def calculate_indicators(df):
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
    return df

def load_and_prepare_data(file_path, timeframe='1h'):
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    
    # Resample dữ liệu dựa trên timeframe được chọn
    df = df.set_index('Open Time').resample(timeframe).agg({
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

    # Reset lại index để 'Open Time' trở thành cột
    df = df.reset_index()
    
    # Tính toán các chỉ báo kỹ thuật dựa trên timeframe
    df = calculate_indicators(df)
    
    # Loại bỏ các hàng có giá trị NaN sau khi thêm các chỉ báo
    df = df.dropna()
    #df = df.iloc[20:].reset_index(drop=True)
    
    return df

# def calculate_indicators(df, timeframe='1h'):
#     # Đặt chu kỳ của các chỉ báo dựa trên timeframe
#     period_factor = {'1h': 1, '4h': 4, '1d': 24, '1w': 168}  # Sử dụng hệ số cho các chu kỳ lớn hơn

#     # Điều chỉnh các chu kỳ dựa trên timeframe
#     period_sma = 10 * period_factor[timeframe]
#     period_ema = 10 * period_factor[timeframe]
#     period_rsi = 14 * period_factor[timeframe]
#     period_bollinger = 20 * period_factor[timeframe]
#     period_atr = 14 * period_factor[timeframe]
#     period_stochastic = 14 * period_factor[timeframe]
#     period_roc = 12 * period_factor[timeframe]
#     period_cmf = 20 * period_factor[timeframe]
#     period_adx = 14 * period_factor[timeframe]
    
#     # Tính toán các chỉ báo với các chu kỳ điều chỉnh
#     df['SMA_10'] = calculate_sma(df, period=period_sma)
#     df['EMA_10'] = calculate_ema(df, period=period_ema)
#     df['RSI_14'] = calculate_rsi(df, period=period_rsi)
#     df['MACD'], df['Signal_Line'] = calculate_macd(df)
#     df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df, period=period_bollinger)
#     df['ATR'] = calculate_atr(df, period=period_atr)
#     df['Stochastic_Oscillator'] = calculate_stochastic_oscillator(df, period=period_stochastic)
#     df['OBV'] = calculate_obv(df)
#     df['ROC'] = calculate_roc(df, period=period_roc)
#     df['CMF'] = calculate_cmf(df, period=period_cmf)
#     df['ADX_14'] = calculate_adx(df, period=period_adx)
#     df['SAR'] = calculate_sar(df, acceleration=0.02, maximum=0.2)
    
#     return df

#region (2) Model Input Creation-----------------------------------------------------------

def create_model_inputs(df, timesteps=168):
    X, y = [], []
    for i in range(len(df) - timesteps):
        X.append(df.iloc[i:i+timesteps].values)
        y.append(df.iloc[i+timesteps]['Close'])
    X, y = np.array(X), np.array(y)
    return X, y

#region (3) Data Splitting-----------------------------------------------------------------

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    test_size = len(X) - train_size - val_size

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def reshape_for_ml(X_train, X_val, X_test):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, X_val, X_test

#region (4) Model Training and Evaluation--------------------------------------------------

def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test):
    rf_manager = MachineLearningManager(model_type="random_forest", n_estimators=100, max_depth=10)
    rf_manager.train(X_train, y_train)
    metrics = rf_manager.evaluate(X_test, y_test)
    print("Random Forest Metrics:", metrics)
    return rf_manager

def load_model(path):
    return joblib.load(path)

#region (5) Prediction and Performance Evaluation------------------------------------------

def predict_and_evaluate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    return mse, rmse, mae

#region (6) Các hàm cho hàm main hoàn chỉnh
def create_new_df_with_indicators(path, timeframe):
    df = load_and_prepare_data(path, timeframe)
    df.to_csv('..//data//df_with_indicator.csv', index=False)  # Optional: save to Excel
    return df

def read_df_and_choose_timeframe(path, timeframe):
    df = pd.read_csv(path)
    # df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df = df.resample(timeframe, on='Open Time')
    return df

#region Data Preparation
# (Separate Step 1 of 2) 1h 4h 1d 1w
df = create_new_df_with_indicators(r"E:\MyPythonCode\Crypto\data\csv\binance_ETHUSDT.csv", '1w')
# (Separate Step 2 of 2)
# df = read_df_and_choose_timeframe('..//data//df_with_indicator.csv', '1d')
print(df)
#endregion

# # Create model inputs
# timesteps = 24 * 7
# X, y = create_model_inputs(df, timesteps)
# print_comboXy_shape(X, y)

# # Split data
# X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
# X_train, X_val, X_test = reshape_for_ml(X_train, X_val, X_test)

# # Train Random Forest model and evaluate
# rf_manager = train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)

# # Load model and predict on validation or test set
# path = r"E:\MyPythonCode\Crypto\notebooks\modelsML\best-RandomForestRegressor-20240927-153944.pkl"
# RF_loaded = load_model(path)
# mode = 'test'  # Change 'test' or 'val' based on what you want to check
# checkX, checky = (X_test, y_test) if mode == 'test' else (X_val, y_val)
    
# # Predict and evaluate
# predict_and_evaluate(RF_loaded, checkX, checky)

