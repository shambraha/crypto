# Import thư viện chuẩn
import os
import sys
import time

# Import thư viện bên thứ ba
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import plotly.graph_objects as go

# Import các hàm và mô-đun nội bộ của dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import csv_folder_path
from main.interact_dataframe import extract_symbols_from_local_path, get_df_from_name
# from main.interact_binanceApi import get_binance_data_to_gdrive, get_binance_data, color_schemes
# from main.interact_streamlit import add_trace_candlestick, add_trace_line, update_yaxis_layout, ...
from function import *
from function_preparation import *
# from classML import *
from classML_upgrade import *

# Cấu hình trang Streamlit
st.set_page_config(layout="wide")

# Hàm tính toán các chỉ báo kỹ thuật
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

# Hàm tải và chuẩn bị dữ liệu theo timeframe
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

# Hàm tạo các cặp X-y làm đầu vào
def create_model_inputs(df, timesteps=168):
    X, y = [], []

    # Loại bỏ cột 'Open Time' khỏi X trước khi thêm vào mảng
    df = df.drop(columns=['Open Time'])

    for i in range(len(df) - timesteps):
        X.append(df.iloc[i:i+timesteps].values)
        y.append(df.iloc[i+timesteps]['Close'])
    X, y = np.array(X), np.array(y)
    return X, y

# Hàm chia train, val, test
def split_data(X, y, train_ratio=0.85, val_ratio=0.1):
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    test_size = len(X) - train_size - val_size

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Dùng cho XGBoost và Random Forest
#.. chuyển đầu vào từ 3D (100, 168, 35) sang 2D (100, 168*35) 
def reshape_for_ml(X_train, X_val, X_test):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, X_val, X_test

# Hàm tổng Split Data
def split_data_and_reshape():
    # Chia dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(st.session_state.X, st.session_state.y)
    
    # Lưu các biến vào session state
    st.session_state.X_train, st.session_state.X_val, st.session_state.X_test = X_train, X_val, X_test
    st.session_state.y_train, st.session_state.y_val, st.session_state.y_test = y_train, y_val, y_test

    # Lưu kích thước trước khi reshape vào session_state
    st.session_state.shape_before_reshape = {
        "X_train": X_train.shape,
        "X_val": X_val.shape,
        "X_test": X_test.shape,
        "y_train": y_train.shape,
        "y_val": y_val.shape,
        "y_test": y_test.shape
    }

    # Thực hiện reshape và lưu kết quả vào session_state
    st.session_state.X_train, st.session_state.X_val, st.session_state.X_test = reshape_for_ml(X_train, X_val, X_test)

    # Lưu kích thước sau khi reshape vào session_state
    st.session_state.shape_after_reshape = {
        "X_train": st.session_state.X_train.shape,
        "X_val": st.session_state.X_val.shape,
        "X_test": st.session_state.X_test.shape,
        "y_train": st.session_state.y_train.shape,
        "y_val": st.session_state.y_val.shape,
        "y_test": st.session_state.y_test.shape
    }

#*********************************************************************************#
# Giao diện Streamlit
st.title("Analyze Crypto using ML Model")
#*********************************************************************************#

# Phần Data ----------------------------------------------------------------------
# Lấy danh sách symbol từ local_folder
available_symbols_local = extract_symbols_from_local_path(csv_folder_path)
selected_symbol = st.sidebar.selectbox("Select Symbol", available_symbols_local)
timeframe = st.sidebar.selectbox("Chọn timeframe:", ['1h', '4h', '1d', '1w'])
timesteps = st.sidebar.number_input("Timesteps:", min_value=1, max_value=1000, value=168)

 # Tạo đường dẫn đến file dữ liệu
file_path = os.path.join(csv_folder_path, f'binance_{selected_symbol}USDT.csv')

st.session_state.timeframe = timeframe
# Kiểm tra session state cho df, X, y và model
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None

# Nút 1: Tải và xử lý dữ liệu
if st.button("Add Indicators to Data"):
    st.session_state.df = load_and_prepare_data(file_path, st.session_state.timeframe)

# Hiển thị bảng `df` nếu đã được tải
if st.session_state.df is not None:
    st.write("Dữ liệu sau khi thêm Indicators:")
    st.write(st.session_state.df.shape)
    st.write(st.session_state.df)    

# Nút 2: Tạo đầu vào cho mô hình
if st.session_state.df is not None and st.button("Create Model Inputs"):
    st.session_state.X, st.session_state.y = create_model_inputs(st.session_state.df, timesteps)

# Hiển thị X và y sau khi tạo đầu vào cho mô hình
if st.session_state.X is not None and st.session_state.y is not None:
    st.write(f"Kích thước của X: {st.session_state.X.shape}")
    st.write(f"Kích thước của y: {st.session_state.y.shape}")

# Nút 3: Chia dữ liệu
if st.session_state.X is not None and st.session_state.y is not None and st.button("Split Data"):
    split_data_and_reshape()
    
# Hiển thị kích thước trước và sau khi reshape ở ngoài phần xử lý của nút
if "shape_before_reshape" in st.session_state:
    st.write("Kích thước trước khi reshape:")
    st.write(f"X_train: {st.session_state.shape_before_reshape['X_train']}")
    st.write(f"X_val: {st.session_state.shape_before_reshape['X_val']}")
    st.write(f"X_test: {st.session_state.shape_before_reshape['X_test']}")

if "shape_after_reshape" in st.session_state:
    st.write("Kích thước sau khi reshape:")
    st.write(f"X_train: {st.session_state.shape_after_reshape['X_train']}, y_train: {st.session_state.shape_after_reshape['y_train']}")
    st.write(f"X_val: {st.session_state.shape_after_reshape['X_val']}, y_val: {st.session_state.shape_after_reshape['y_val']}")
    st.write(f"X_test: {st.session_state.shape_after_reshape['X_test']}, y_test: {st.session_state.shape_after_reshape['y_test']}")

# Phần Model ----------------------------------------------------------------------
# Khởi tạo biến session_state cho các biến cần thiết
if "X_train" not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_val = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_val = None
    st.session_state.y_test = None
    st.session_state.model_manager = None
    st.session_state.training_status = ""
# Khởi tạo các biến trong st.session_state nếu chưa tồn tại
if "model_manager" not in st.session_state:
    st.session_state.model_manager = None
if "progress_bar" not in st.session_state:
    st.session_state.progress_bar = st.progress(0)
if "status_text" not in st.session_state:
    st.session_state.status_text = st.empty()  # Khởi tạo là st.empty() để có thể gọi .text() sau này
if "best_score_text" not in st.session_state:
    st.session_state.best_score_text = st.empty()  # Khởi tạo là st.empty() để có thể gọi .text()
if "training_status" not in st.session_state:
    st.session_state.training_status = ""
if "final_best_score" not in st.session_state:
    st.session_state.final_best_score = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

# Chọn loại mô hình và loại bài toán
model_type = st.sidebar.selectbox("Chọn loại mô hình:", ["random_forest", "xgboost"])
task_type = st.sidebar.selectbox("Chọn loại bài toán:", ["regression", "classification"])

# Nhập các tham số của mô hình
n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100)
max_depth = st.sidebar.slider("max_depth", 1, 20, 10)

# Thêm learning_rate nếu mô hình là xgboost
learning_rate = None
if model_type == "xgboost":
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01)

# Nút 4: Khởi tạo và hiển thị mô hình khi nhấn nút Initialize Model
if st.button("Initialize Model"):
    model_kwargs = {
        "n_estimators": n_estimators,
        "max_depth": max_depth
    }
    if learning_rate is not None:
        model_kwargs["learning_rate"] = learning_rate
    
    st.session_state.model_manager = MachineLearningManager(
        model_type=model_type,
        task_type=task_type,
        **model_kwargs
    )
    # Lưu chuỗi thông tin mô hình vào session_state để hiển thị sau
    st.session_state.model_summary = st.session_state.model_manager.get_model_summary()

# Hiển thị thông tin mô hình nếu đã khởi tạo
if "model_summary" in st.session_state:
    st.write("Model initialized:", st.session_state.model_summary)

# Thêm tùy chọn số lượng epochs vào sidebar
num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=500, value=100, step=1)

# Nút 5: Huấn luyện mô hình và hiển thị tiến trình
if st.session_state.model_manager is not None and st.button("Train Model"):
    # Tạo thanh tiến trình và hiển thị trạng thái
    progress_bar = st.progress(0)
    status_text = st.empty()
    best_score_text = st.empty()
    
    st.session_state.training_status = "Training in progress..."
    status_text.text(st.session_state.training_status)

    # Huấn luyện mô hình
    # num_epochs = 100  # Giả định 100 lần lặp
    for epoch in range(1, num_epochs + 1):
        # Train model và cập nhật model tốt nhất
        st.session_state.model_manager.train_with_best_model(
            st.session_state.X_train, st.session_state.y_train, 
            st.session_state.X_val, st.session_state.y_val)
        
        # Cập nhật thanh tiến trình và trạng thái
        progress_bar.progress(int(epoch / num_epochs * 100))
        status_text.text(f"Training: {epoch}/{num_epochs} epochs complete")
        best_score_text.text(f"Best Validation Score (MSE): {st.session_state.model_manager.best_score:.4f}")
        
        # Dừng nhẹ để hiển thị tiến trình mượt hơn
        time.sleep(0.1)
    
    # Cập nhật trạng thái sau khi hoàn tất
    st.session_state.training_status = "Training completed!"
    st.session_state.status_text.text(st.session_state.training_status)
    st.session_state.final_best_score = st.session_state.model_manager.best_score  # Lưu điểm tốt nhất sau huấn luyện

# Hiển thị thông tin huấn luyện và kết quả tốt nhất bên ngoài nút
if "training_status" in st.session_state:
    st.write("Training Status:", st.session_state.training_status)
if "final_best_score" in st.session_state:
    st.write("Mô hình tốt nhất đã được lưu với điểm số MSE thấp nhất:", st.session_state.final_best_score)

# Nút 6: Đánh giá mô hình sau khi huấn luyện
if st.session_state.model_manager is not None and st.button("Evaluate Model"):
    # Đánh giá mô hình trên tập kiểm tra
    st.session_state.evaluation_results = st.session_state.model_manager.evaluate(
        st.session_state.X_test, st.session_state.y_test
    )

# Hiển thị kết quả đánh giá nếu đã có trong session_state
if "evaluation_results" in st.session_state:
    st.write("Evaluation Results:", st.session_state.evaluation_results)

# Phần Trực quan hoá----------------------------------------------------------------
# Vẽ biểu đồ nến cho dữ liệu lịch sử
def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df['Open Time'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title="Candlestick Chart", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig)

# Vẽ biểu đồ đường so sánh giá trị thực tế và dự đoán
def plot_actual_vs_predicted(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_true))), y=y_true, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title="Actual vs Predicted", xaxis_title="Time Step", yaxis_title="Price")
    st.plotly_chart(fig)

# Vẽ biểu đồ phân phối lỗi
def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=50)])
    fig.update_layout(title="Error Distribution", xaxis_title="Error", yaxis_title="Frequency")
    st.plotly_chart(fig)

# Biểu đồ nến: Có thể gọi sau khi dữ liệu được xử lý xong (sau phần “Add Indicators to Data”).
if st.session_state.df is not None:
    plot_candlestick(st.session_state.df)

# Biểu đồ Actual vs Predicted và Error Distribution: Gọi sau khi mô hình được đánh giá.
if st.session_state.model_manager is not None and st.button("Plot"):
    # Đánh giá mô hình trên tập kiểm tra
    st.session_state.evaluation_results = st.session_state.model_manager.evaluate(
        st.session_state.X_test, st.session_state.y_test
    )    

    # Lấy giá trị dự đoán từ mô hình và lưu vào session_state
    st.session_state.y_pred = st.session_state.model_manager.predict(st.session_state.X_test)
    st.session_state.y_test_actual = st.session_state.y_test
    
def display_plots():
    if "y_pred" in st.session_state and "y_test_actual" in st.session_state:
        # Hiển thị biểu đồ so sánh giá trị thực tế và dự đoán
        plot_actual_vs_predicted(st.session_state.y_test_actual, st.session_state.y_pred)
        
        # Hiển thị biểu đồ phân phối lỗi
        plot_error_distribution(st.session_state.y_test_actual, st.session_state.y_pred)

# Gọi hàm display_plots để hiển thị biểu đồ nếu có dữ liệu
display_plots()    

# Hàm dự đoán cho n time steps tiếp theo
def predict_future_steps(model, X_test, n_steps):
    predictions = []
    current_input = X_test[-1]  # Lấy tập dữ liệu kiểm tra cuối cùng làm input ban đầu

    for _ in range(n_steps):
        # Dự đoán bước tiếp theo
        pred = model.predict(current_input.reshape(1, -1))
        predictions.append(pred[0])

        # Cập nhật current_input cho bước tiếp theo bằng cách loại bỏ phần tử đầu và thêm giá trị dự đoán vào cuối
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred  # Thêm giá trị dự đoán vào cuối

    return predictions

# Sử dụng hàm predict_future_steps và vẽ biểu đồ
if st.session_state.model_manager is not None and st.button("Predict Future Steps"):
    ### Giả sử mô hình đã được lưu trong st.session_state.model_manager
    # Dự đoán trên tập kiểm tra hiện tại
    y_pred = st.session_state.model_manager.predict(st.session_state.X_test)
    
    # Dự đoán thêm n bước tiếp theo
    n_future_steps = st.sidebar.number_input("Số bước dự đoán tiếp theo:", min_value=1, max_value=100, value=10)
    future_predictions = predict_future_steps(st.session_state.model_manager, st.session_state.X_test, n_future_steps)
    
    # Ghép `y_test` với các dự đoán mới cho việc vẽ biểu đồ
    y_combined = np.concatenate((y_pred, future_predictions), axis=0)
    time_combined = list(range(len(y_pred) + len(future_predictions)))
    
    # Vẽ biểu đồ với dự đoán mới
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=list(range(len(st.session_state.y_test))), y=st.session_state.y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=time_combined[-len(future_predictions):], y=future_predictions, mode='lines', name='Future Predictions', line=dict(dash='dash')))
    
    fig.update_layout(title="Actual, Predicted, and Future Predictions", xaxis_title="Time Step", yaxis_title="Price")
    st.plotly_chart(fig)

# Sử dụng hàm predict_future_steps và vẽ biểu đồ với Timestamp
if st.session_state.model_manager is not None and st.button("Predict Future Steps with Timestamp"):
    # Dự đoán trên tập kiểm tra hiện tại
    y_pred = st.session_state.model_manager.predict(st.session_state.X_test)
    
    # Dự đoán thêm n bước tiếp theo
    n_future_steps = st.sidebar.number_input("Số bước dự đoán tiếp theo:", min_value=1, max_value=100, value=10)
    future_predictions = predict_future_steps(st.session_state.model_manager, st.session_state.X_test, n_future_steps)
    
    # Tạo danh sách thời gian gốc cho y_test và y_pred dựa trên thời gian trong df
    original_time_index = st.session_state.df['Open Time'].iloc[-len(st.session_state.y_test):].tolist()
    # future_time_index = pd.date_range(start=original_time_index[-1], periods=n_future_steps + 1, freq=timeframe)[1:]  # Bỏ khoảng thời gian đầu
    future_time_index = pd.date_range(
        start=original_time_index[-1],
        periods=n_future_steps + 1,
        freq=st.session_state.timeframe
    )[1:]

    # Kết hợp thời gian của y_test và các dự đoán mới
    time_combined = original_time_index + list(future_time_index)
    y_combined = np.concatenate((y_pred, future_predictions), axis=0)
    
    # Vẽ biểu đồ với thời gian gốc
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_time_index, y=y_pred, mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=original_time_index, y=st.session_state.y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=time_combined[-len(future_predictions):], y=future_predictions, mode='lines', name='Future Predictions', line=dict(dash='dash')))
    
    fig.update_layout(title="Actual, Predicted, and Future Predictions (with Timestamps)", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig)
