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
from classModel_upgrade import *

# Cấu hình trang Streamlit
st.set_page_config(layout="wide")

# Hàm (1) tính toán các chỉ báo kỹ thuật
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

# Hàm (2) tải và chuẩn bị dữ liệu theo timeframe
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

# Hàm (3) tạo các cặp X-y làm đầu vào
# Hàm (3.1 of 3)
def create_model_inputs(df, timesteps=168):
    X, y = [], []

    # Loại bỏ cột 'Open Time' khỏi X trước khi thêm vào mảng
    df = df.drop(columns=['Open Time'])

    for i in range(len(df) - timesteps):
        X.append(df.iloc[i:i+timesteps].values)
        y.append(df.iloc[i+timesteps]['Close'])
    X, y = np.array(X), np.array(y)
    return X, y

# Hàm (3.2 of 3)
def create_model_inputs_with_ahead(df, timesteps=168, ahead=1):
    X, y = [], []
    df = df.drop(columns=['Open Time'])  # Giả định cột 'Open Time' không cần cho X

    for i in range(len(df) - timesteps - ahead + 1):
        X.append(df.iloc[i:i + timesteps].values)  # Chuỗi đầu vào với độ dài `timesteps`
        y.append(df.iloc[i + timesteps: i + timesteps + ahead]['Close'].values)  # Chuỗi dự đoán `ahead` bước

    X, y = np.array(X), np.array(y)
    return X, y

# Hàm (3.3 of 3)
def create_model_inputs_with_ahead_and_outputSize(df, timesteps=168, output_size=1, ahead=1, target_columns=None):
    """
    target_columns: Danh sách các cột đặc trưng mà bạn muốn dự đoán.
    """
    X, y = [], []
    
    # Xác định các cột đặc trưng đầu ra
    if target_columns is None:
        target_columns = ['Close']  # Mặc định nếu không cung cấp thì dự đoán cột 'Close'
    
    # Loại bỏ cột 'Open Time' khỏi X trước khi thêm vào mảng
    df = df.drop(columns=['Open Time'])
    
    for i in range(len(df) - timesteps - ahead + 1):
        # Chuỗi đầu vào với độ dài `timesteps`
        X.append(df.iloc[i:i + timesteps].values)
        
        # Chuỗi đầu ra với các cột đặc trưng đã chọn và `ahead` bước
        y.append(df.iloc[i + timesteps: i + timesteps + ahead][target_columns].values)

    X, y = np.array(X), np.array(y)
    return X, y


# Hàm (4) chia train, val, test
def split_data(X, y, train_ratio=0.85, val_ratio=0.1):
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    test_size = len(X) - train_size - val_size

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

# (5) Dùng cho XGBoost và Random Forest
#.. chuyển đầu vào từ 3D (100, 168, 35) sang 2D (100, 168*35) 
# def reshape_for_ml(X_train, X_val, X_test):
#     X_train = X_train.reshape(X_train.shape[0], -1)
#     X_val = X_val.reshape(X_val.shape[0], -1)
#     X_test = X_test.reshape(X_test.shape[0], -1)
#     return X_train, X_val, X_test

# (6) Hàm tổng Split Data
def split_data_without_reshape():
    # Chia dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(st.session_state.X, st.session_state.y)
    
    # Chuyển đổi dữ liệu sang Tensor để nạp vào LSTM
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Lưu các biến vào session state
    st.session_state.X_train, st.session_state.X_val, st.session_state.X_test = X_train, X_val, X_test
    st.session_state.y_train, st.session_state.y_val, st.session_state.y_test = y_train, y_val, y_test

    # Lưu kích thước trước khi reshape vào session_state
    st.session_state.shape_lstm = {
        "X_train": X_train.shape,
        "X_val": X_val.shape,
        "X_test": X_test.shape,
        "y_train": y_train.shape,
        "y_val": y_val.shape,
        "y_test": y_test.shape
    }
   
#*********************************************************************************#
# Giao diện Streamlit
st.title("Analyze Crypto using ML Model")
#*********************************************************************************#

# A. Phần Data ----------------------------------------------------------------------
st.sidebar.subheader("A. Phần Data")
# Lấy danh sách symbol từ local_folder
available_symbols_local = extract_symbols_from_local_path(csv_folder_path)
selected_symbol = st.sidebar.selectbox("Select Symbol", available_symbols_local)
timeframe = st.sidebar.selectbox("Chọn timeframe:", ['1h', '4h', '1d', '1w'])
timesteps = st.sidebar.number_input("Timesteps:", min_value=1, max_value=1000, value=168)

# Thêm output_size và ahead vào session_state nếu chưa tồn tại
if "ahead" not in st.session_state:
    st.session_state.ahead = 1  # Số bước dự đoán trước, giá trị mặc định là 1
st.session_state.ahead = st.sidebar.number_input("Steps Ahead", min_value=1, max_value=10, value=st.session_state.ahead)
if "output_size" not in st.session_state:
    st.session_state.output_size = 1  # Giá trị mặc định cho hồi quy
st.session_state.output_size = st.sidebar.number_input("Output Size (PHẢI chọn cột features tương ứng ở Nút 2 - Line 220)",
                                                    min_value=1, max_value=10,
                                                    value=st.session_state.output_size)

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

# Cho phép người dùng chọn các cột đầu ra dự đoán
# all_columns = list(st.session_state.df.columns[1:])  # Lấy tất cả các cột trừ 'Open Time'
# target_columns = st.sidebar.multiselect("Chọn các cột đầu ra (phải bằng Output Size)", all_columns, default=all_columns[:st.session_state.output_size])

# Kiểm tra nếu `st.session_state.df` đã được khởi tạo
if "df" in st.session_state and st.session_state.df is not None:
    # Lấy tất cả các cột trừ 'Open Time'
    all_columns = list(st.session_state.df.columns[1:])
else:
    all_columns = []  # Nếu `df` chưa được khởi tạo, đặt `all_columns` thành danh sách rỗng

# Cho phép người dùng chọn các cột đầu ra dự đoán, chỉ hiển thị nếu `all_columns` không rỗng
if all_columns:
    target_columns = st.sidebar.multiselect("Chọn các cột đầu ra (phải bằng Output Size)", all_columns, default=all_columns[:st.session_state.output_size])
else:
    st.warning("Cần tải dữ liệu trước khi chọn các cột đầu ra.")
    target_columns = []

# Kiểm tra số lượng cột đầu ra
if len(target_columns) != st.session_state.output_size:
    st.warning(f"Output Size ({st.session_state.output_size}) và số lượng cột đầu ra ({len(target_columns)}) phải khớp.")
else:
    # Nút 2: Tạo đầu vào cho mô hình
    if st.session_state.df is not None and st.button("Create Model Inputs"):
        st.session_state.X, st.session_state.y = create_model_inputs_with_ahead_and_outputSize(
                                                    st.session_state.df, timesteps,
                                                    output_size=st.session_state.output_size,
                                                    ahead=st.session_state.ahead,
                                                    target_columns=target_columns)
        
# # Nút 2: Tạo đầu vào cho mô hình
# if st.session_state.df is not None and st.button("Create Model Inputs"):
#     # st.session_state.X, st.session_state.y =            create_model_inputs(st.session_state.df, timesteps)
#     # st.session_state.X, st.session_state.y = create_model_inputs_with_ahead(st.session_state.df, timesteps, st.session_state.ahead)
#     output_size = st.session_state.output_size # con số phải tương ứng với số lượng cột trong target_columns
#     target_columns = ['Close', 'Volume', 'High']
#     st.session_state.X, st.session_state.y = create_model_inputs_with_ahead_and_outputSize(
#                                                     st.session_state.df, timesteps,
#                                                     output_size=output_size,
#                                                     ahead=st.session_state.ahead,
#                                                     target_columns=target_columns)

# Hiển thị X và y sau khi tạo đầu vào cho mô hình
if st.session_state.X is not None and st.session_state.y is not None:
    st.write(f"Kích thước của X: {st.session_state.X.shape}")
    st.write(f"Kích thước của y: {st.session_state.y.shape}")

# Nút 3: Chia dữ liệu
if st.session_state.X is not None and st.session_state.y is not None and st.button("Split Data"):
    split_data_without_reshape()
    
# Hiển thị kích thước without reshape ở ngoài phần xử lý của nút
if "shape_lstm" in st.session_state:
    st.write("Kích thước sau khi reshape:")
    st.write(f"X_train: {st.session_state.shape_lstm['X_train']}, y_train: {st.session_state.shape_lstm['y_train']}")
    st.write(f"X_val: {st.session_state.shape_lstm['X_val']}, y_val: {st.session_state.shape_lstm['y_val']}")
    st.write(f"X_test: {st.session_state.shape_lstm['X_test']}, y_test: {st.session_state.shape_lstm['y_test']}")

# B. Phần Model ----------------------------------------------------------------------
# Khởi tạo biến session_state cho các biến cần thiết nếu chưa có
if "X_train" not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_val = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_val = None
    st.session_state.y_test = None
    st.session_state.model_manager = None
    st.session_state.training_status = ""

# Khởi tạo các biến khác trong st.session_state nếu chưa tồn tại
if "model_manager" not in st.session_state:
    st.session_state.model_manager = None
if "progress_bar" not in st.session_state:
    st.session_state.progress_bar = st.progress(0)
if "status_text" not in st.session_state:
    st.session_state.status_text = st.empty()  # Khởi tạo st.empty() để có thể gọi .text() sau này
if "best_score_text" not in st.session_state:
    st.session_state.best_score_text = st.empty()
if "training_status" not in st.session_state:
    st.session_state.training_status = ""
if "final_best_score" not in st.session_state:
    st.session_state.final_best_score = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

st.sidebar.subheader("B. Cấu hình Mô hình LSTM")
# Chọn loại mô hình và loại bài toán
task_type = st.sidebar.selectbox("Chọn loại bài toán:", ["regression", "classification"])

# Nhập và Lưu các tham số mô hình LSTM vào session_state
if "hidden_size" not in st.session_state:
    st.session_state.hidden_size = 64
st.session_state.hidden_size = st.sidebar.slider("Hidden Size", 10, 200, st.session_state.hidden_size)
if "num_layers" not in st.session_state:
    st.session_state.num_layers = 2
st.session_state.num_layers = st.sidebar.slider("Number of Layers", 1, 4, st.session_state.num_layers)
if "dropout_rate" not in st.session_state:
    st.session_state.dropout_rate = 0.0
st.session_state.dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, st.session_state.dropout_rate)
if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 0.001
st.session_state.learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=st.session_state.learning_rate, step=0.0001, format="%.4f")

# Nếu là bài toán classification, hiển thị num_classes với key duy nhất
if task_type == "classification":
    st.session_state.num_classes = st.sidebar.number_input("Number of Classes", min_value=2, max_value=10, value=2, key="num_classes_key")
else:
    st.session_state.num_classes = None  # Đặt None khi không phải bài toán classification




# Nút khởi tạo và hiển thị mô hình LSTM khi nhấn nút Initialize Model
if st.button("Initialize Model"):
    # Khởi tạo mô hình LSTM
    input_size = st.session_state.X_train.shape[2] if st.session_state.X_train is not None else None
    if input_size is None:
        st.warning("Cần khởi tạo và chuẩn bị dữ liệu trước khi tạo mô hình!")
    else:
        if task_type == "regression":
            model = LSTM_Regression(input_size=input_size,
                                hidden_size=st.session_state.hidden_size,
                                num_layers=st.session_state.num_layers,
                                dropout_rate=st.session_state.dropout_rate,
                                output_size=st.session_state.output_size,
                                ahead=st.session_state.ahead)
            st.session_state.model_manager = ModelManagerRegression(
                model=model,
                criterion=nn.MSELoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=st.session_state.learning_rate)
            )
            # Tạo chuỗi model_summary cho Regression
            st.session_state.model_summary = (
                f"**Model LSTM Regression**  \n"
                f"- Input Size: {input_size}  \n"   
                f"- Hidden Size: {st.session_state.hidden_size}  \n"
                f"- Layers: {st.session_state.num_layers}  \n"
                f"- Dropout: {st.session_state.dropout_rate}  \n"
                f"- Learning Rate: {st.session_state.learning_rate}  \n"
                f"- Output Size: {st.session_state.output_size}  \n"
                f"- Steps Ahead: {st.session_state.ahead}  \n"
            )

        elif task_type == "classification":
            # Lấy num_classes từ session_state khi task_type là classification
            model = LSTMClassification(input_size=input_size,
                                hidden_size=st.session_state.hidden_size,
                                num_layers=st.session_state.num_layers,
                                dropout_rate=st.session_state.dropout_rate,
                                num_classes=st.session_state.num_classes)
            st.session_state.model_manager = ModelManagerClassification(
                model=model,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=st.session_state.learning_rate)
            )
            # Tạo chuỗi model_summary cho Classification
            st.session_state.model_summary = (
                f"**Model LSTM Classification**  \n"
                f"- Input Size: {input_size}  \n"   
                f"- Hidden Size: {st.session_state.hidden_size}  \n"
                f"- Layers: {st.session_state.num_layers}  \n"
                f"- Dropout: {st.session_state.dropout_rate}  \n"
                f"- Learning Rate: {st.session_state.learning_rate}  \n"
                f"- Classes: {st.session_state.num_classes}  \n"
            )
               
# Hiển thị thông tin mô hình nếu đã khởi tạo
if "model_summary" in st.session_state:
    st.markdown(st.session_state.model_summary)

# C. Phần Huấn luyện ----------------------------------------------------------------------
st.sidebar.subheader("C. Training")
# Lưu các tham số num_epochs và batch_size vào session_state
if "num_epochs" not in st.session_state:
    st.session_state.num_epochs = 100
st.session_state.num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=500, value=st.session_state.num_epochs, step=1)
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 32
st.session_state.batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=128, value=st.session_state.batch_size, step=1)

# D. readme.md
# Khi bạn nạp dữ liệu vào mô hình LSTM, kích thước của các tensor đầu vào nên như sau:
# X_train: (batch_size, sequence_length, input_size)
# y_train: (batch_size, output_size), nếu output_size > 1, hoặc (batch_size,) nếu output_size = 1
# sequence_length chính là timesteps
# còn input_size chính là số lượng features đầu vào
# còn output_size là số features muốn dự đoán