# Import thư viện chuẩn
import os
import sys
import time

# Import thư viện bên thứ ba
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

# (5) CHỈ DÙNG CHO XGBoost và Random Forest
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

# def normalize_data(X_train, X_val, X_test, normalization_type):
#     if normalization_type == "Min-Max Scaling":
#         scaler = MinMaxScaler()
#     elif normalization_type == "Z-score Normalization":
#         scaler = StandardScaler()
#     else:
#         return X_train, X_val, X_test  # Không chuẩn hóa
    
#     # Chuẩn hóa dữ liệu train
#     X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    
#     # Áp dụng scaler lên dữ liệu val và test
#     X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
#     X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
#     return X_train_scaled, X_val_scaled, X_test_scaled

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# def normalize_data(X, y, normalization_type):
#     if normalization_type == "Min-Max Scaling":
#         x_scaler = MinMaxScaler()
#         y_scaler = MinMaxScaler()
#     elif normalization_type == "Z-score Normalization":
#         x_scaler = StandardScaler()
#         y_scaler = StandardScaler()
#     else:
#         return X, y  # Không chuẩn hóa
    
#     # Chuẩn hóa X
#     X_scaled = x_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
#     # Chuẩn hóa y
#     y_scaled = y_scaler.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
    
#     return X_scaled, y_scaled, x_scaler, y_scaler  # Trả về scaler để dùng cho giải chuẩn hóa nếu cần

def normalize_data(X, y, normalization_type):
    if normalization_type == "Min-Max Scaling":
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    elif normalization_type == "Z-score Normalization":
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
    else:
        return X, y  # Không chuẩn hóa

    # Chuẩn hóa X và giữ kích thước 3 chiều
    X_scaled = x_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Chuẩn hóa y và giữ kích thước 3 chiều
    y_scaled = y_scaler.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
    
    return X_scaled, y_scaled, x_scaler, y_scaler

#*********************************************************************************#
# Giao diện Streamlit
st.title("Analyze Crypto using LSTM Model")
#*********************************************************************************#

# A. Phần Data ----------------------------------------------------------------------
st.sidebar.subheader("A. Phần Data")
# Lấy danh sách symbol từ local_folder
available_symbols_local = extract_symbols_from_local_path(csv_folder_path)
selected_symbol = st.sidebar.selectbox("Select Symbol", available_symbols_local)
timeframe = st.sidebar.selectbox("Chọn timeframe:", ['1h', '4h', '1d', '1w'])
timesteps = st.sidebar.number_input("Timesteps:", min_value=1, max_value=1000, value=168)

# Lưu selected_symbol vào session_state nếu nó thay đổi
if "selected_symbol" not in st.session_state or st.session_state.selected_symbol != selected_symbol:
    st.session_state.selected_symbol = selected_symbol

# Thêm output_size và ahead vào session_state nếu chưa tồn tại
if "ahead" not in st.session_state:
    st.session_state.ahead = 1  # Số bước dự đoán trước, giá trị mặc định là 1
st.session_state.ahead = st.sidebar.number_input("Steps Ahead", min_value=1, max_value=10, value=st.session_state.ahead)
if "output_size" not in st.session_state:
    st.session_state.output_size = 1  # Giá trị mặc định cho hồi quy
st.session_state.output_size = st.sidebar.number_input("Output Size (PHẢI chọn cột features tương ứng ở Nút 2 - Line 220)",
                                                    min_value=1, max_value=50,
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
if st.button("A.1 Add Indicators to Data"):
    st.session_state.df = load_and_prepare_data(file_path, st.session_state.timeframe)

# Hiển thị bảng `df` nếu đã được tải
if st.session_state.df is not None:
    st.write("Dữ liệu sau khi thêm Indicators:")
    st.write(st.session_state.df.shape)
    st.write(st.session_state.df)    

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




# # Nút Normalize Data
# if st.sidebar.button("Normalize Data"):
#     if normalization_type != "None":
#         X_train, X_val, X_test = normalize_data(X_train, X_val, X_test, normalization_type)
#         st.success(f"Data normalized using {normalization_type}")
#     else:
#         st.warning("Please select a normalization type before proceeding.")


# Kiểm tra số lượng cột đầu ra
if len(target_columns) != st.session_state.output_size:
    st.warning(f"Output Size ({st.session_state.output_size}) và số lượng cột đầu ra ({len(target_columns)}) phải khớp.")
else:
    # Nút 2: Tạo đầu vào cho mô hình
    if st.session_state.df is not None and st.button("A.2 Create Model Inputs"):
        st.session_state.X, st.session_state.y = create_model_inputs_with_ahead_and_outputSize(
                                                    st.session_state.df, timesteps,
                                                    output_size=st.session_state.output_size,
                                                    ahead=st.session_state.ahead,
                                                    target_columns=target_columns)

# Hiển thị X và y sau khi tạo đầu vào cho mô hình
if st.session_state.X is not None and st.session_state.y is not None:
    st.write(f"Kích thước của X: {st.session_state.X.shape}")
    st.write(f"Kích thước của y: {st.session_state.y.shape}")

# Thêm tùy chọn Normalization trong Sidebar
st.sidebar.subheader("Data Normalization")
normalization_type = st.sidebar.selectbox(
    "Select Normalization Type",
    options=["None", "Min-Max Scaling", "Z-score Normalization"]
)

def normalize_and_update_state(normalization_type):
    if normalization_type != "None":
        # Thực hiện chuẩn hóa khi có lựa chọn kiểu normalize
        st.session_state.X, st.session_state.y, st.session_state.x_scaler, st.session_state.y_scaler = normalize_data(
            st.session_state.X, st.session_state.y, normalization_type
        )
        st.success(f"Data normalized using {normalization_type}")
    else:
        # Nếu không chọn kiểu normalize, tiếp tục mà không chuẩn hóa
        st.info("Proceeding without normalization.")

# Nút Normalize
if st.button("A.3 Normalize"):
    normalize_and_update_state(normalization_type)


# Nút 3: Chia dữ liệu
if st.session_state.X is not None and st.session_state.y is not None and st.button("A.4 Split Data"):
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
# Cập nhật task_type vào session_state
if "task_type" not in st.session_state or st.session_state.task_type != task_type:
    st.session_state.task_type = task_type

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
st.session_state.learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.05, value=st.session_state.learning_rate, step=0.0001, format="%.4f")

# Nếu là bài toán classification, hiển thị num_classes với key duy nhất
if task_type == "classification":
    st.session_state.num_classes = st.sidebar.number_input("Number of Classes", min_value=2, max_value=10, value=2, key="num_classes_key")
else:
    st.session_state.num_classes = None  # Đặt None khi không phải bài toán classification


# Nút 4: Khởi tạo và hiển thị mô hình LSTM khi nhấn nút Initialize Model
if st.button("B.1 Initialize Model"):
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

# Chạy dữ liệu qua mô hình một lần với một batch dữ liệu nhỏ và hiển thị kết quả đầu ra..
# ..giúp chúng ta xác nhận rằng các cài đặt và kiến trúc mô hình phù hợp trước khi thực hiện huấn luyện chính thức 
def demo_train_step(model, X_sample, y_sample):
        model.eval()  # Đặt mô hình ở chế độ đánh giá (evaluation mode) cho demo
        with torch.no_grad():  # Tắt tính năng tính gradient
            y_pred = model(X_sample)
        return y_pred, y_sample

# Nút 5: Thực hiện train demo với 1 batch nhỏ
if "model_summary" in st.session_state and st.button("B.2 Demo Train"):
    # Kiểm tra nếu X_train và y_train đã được khởi tạo
    if st.session_state.X_train is not None and st.session_state.y_train is not None:
        # Chọn một sample từ dữ liệu huấn luyện
        X_sample = st.session_state.X_train[:1]  # Lấy một batch nhỏ (ở đây là batch size = 1)
        y_sample = st.session_state.y_train[:1]
        
        # Đưa dữ liệu vào mô hình và chạy demo train
        y_pred, y_sample = demo_train_step(st.session_state.model_manager.model, X_sample, y_sample)
        
        # Hiển thị kết quả
        st.write("Đầu ra của mô hình (y_pred):", y_pred)
        st.write("Giá trị thực tế (y_sample):", y_sample)
    else:
        st.warning("Cần chuẩn bị dữ liệu trước khi thực hiện demo train.")


# C. Phần Huấn luyện ----------------------------------------------------------------------
st.sidebar.subheader("C. Training Parameters")

if "num_epochs" not in st.session_state:
    st.session_state.num_epochs = 100  # Số epoch mặc định
st.session_state.num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=10000, value=st.session_state.num_epochs)
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 32  # Kích thước batch mặc định
st.session_state.batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=128, value=st.session_state.batch_size)

# Khởi tạo cờ trong session_state nếu chưa có
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False

# Hàm vẽ lịch sử huấn luyện cho hồi quy
def plot_regression_history(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Regression Training History")
    ax.legend()
    st.pyplot(fig)

# Hàm vẽ lịch sử huấn luyện cho phân loại
def plot_classification_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Vẽ Loss trên ax1
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Classification Loss History")
    ax1.legend()

    # Vẽ Accuracy trên ax2
    ax2.plot(history["train_accuracy"], label="Train Accuracy")
    ax2.plot(history["val_accuracy"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy History")
    ax2.legend()

    st.pyplot(fig)

# Hàm huấn luyện đầy đủ
def train_model():
    # Kiểm tra nếu mô hình và dữ liệu đã sẵn sàng
    if st.session_state.model_manager and st.session_state.X_train is not None and st.session_state.y_train is not None:
        # Chuyển đổi dữ liệu sang DataLoader để huấn luyện theo batch
        train_data = torch.utils.data.TensorDataset(st.session_state.X_train, st.session_state.y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=st.session_state.batch_size, shuffle=True)
        
        val_data = torch.utils.data.TensorDataset(st.session_state.X_val, st.session_state.y_val)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=st.session_state.batch_size, shuffle=False)

        # Thiết lập thanh tiến trình và thông báo trạng thái
        progress_bar = st.progress(0)
        status_text = st.empty()  
        epoch_status = st.empty()

        # Bên trong hàm huấn luyện
        for epoch in range(st.session_state.num_epochs):
            st.session_state.model_manager.train(train_loader, val_loader, epochs=1)
            train_loss = st.session_state.model_manager.history["train_loss"][-1]
            val_loss = st.session_state.model_manager.history["val_loss"][-1]

            # Cập nhật trạng thái epoch duy nhất
            epoch_status.text(f"Epoch {epoch + 1}/{st.session_state.num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            progress_bar.progress((epoch + 1) / st.session_state.num_epochs)
         
        # Sau khi hoàn tất huấn luyện
        st.success("Training complete!")
        
        # Gọi hàm vẽ tương ứng dựa trên session_state.task_type
        if st.session_state.task_type == "regression":
            plot_regression_history(st.session_state.model_manager.history)
        elif st.session_state.task_type == "classification":
            plot_classification_history(st.session_state.model_manager.history)          

        # Đặt cờ hoàn thành huấn luyện
        st.session_state.training_complete = True
    else:
        st.warning("Cần khởi tạo mô hình và dữ liệu trước khi huấn luyện!")

# Nút 6: Bắt đầu huấn luyện thực sự với tham số num_epochs và batch_size
if st.button("C.1 Start Training"):
    train_model()

# Chỉ hiển thị các nút Save và Load khi training_complete là True
if st.session_state.training_complete:
    # Sử dụng selected_symbol trong đường dẫn model_path
    model_path = st.sidebar.text_input("Model Path", value=f"modelsML/{st.session_state.selected_symbol}.pth")
    # Cập nhật learning rate mới từ sidebar
    new_learning_rate = st.sidebar.number_input("New Learning Rate for Fine-tuning", min_value=0.0001, max_value=0.1, step=0.001, format="%.4f")

    if st.button("C.2 Save Model (rememberCheck Model Path)"):
        if model_path:
            st.session_state.model_manager.save_model(model_path)
            st.success(f"Model saved successfully at {model_path}!")
        else:
            st.warning("Please enter a valid model path.")

    if st.button("C.3 Load Model (rememberCheck new_learning_rate)"):
        if model_path:
            st.session_state.model_manager.load_model(model_path)
            st.success(f"Model loaded successfully from {model_path}!")

            # Thiết lập optimizer với learning rate mới
            for param_group in st.session_state.model_manager.optimizer.param_groups:
                param_group['lr'] = new_learning_rate

            st.write(f"Updated learning rate to {new_learning_rate}")
        else:
            st.warning("Please enter a valid model path.")

# D. Phần Đánh giá ----------------------------------------------------------------------
# Thêm lựa chọn loại dữ liệu đánh giá
st.sidebar.subheader("D. Evaluation Options")
evaluation_option = st.sidebar.selectbox(
    "Select Evaluation Type",
    options=["Test Dataset", "Validation Dataset", "Forecast n Steps Ahead"]
)

# Nếu chọn Forecast, cho phép chọn n bước thời gian tiếp theo
n_steps = 0
if evaluation_option == "Forecast n Steps Ahead":
    n_steps = st.sidebar.number_input("Number of Steps Ahead to Forecast", min_value=1, max_value=100, value=10)
   
# Hàm dự đoán và trả về [predictions, true_values] để thực hiện vẽ biểu đồ
def evaluate_and_predict(evaluation_option, model_manager, n_steps=0):
    model_manager.model.eval()
    predictions, true_values = None, None
    steps_ahead = st.session_state.ahead  # Sử dụng bước ahead từ session state
    
    with torch.no_grad():
        if evaluation_option == "Test Dataset":
            X_data = st.session_state.X_test
            y_data = st.session_state.y_test
            predictions = model_manager.model(X_data).cpu().numpy()
            true_values = y_data.cpu().numpy()

        elif evaluation_option == "Validation Dataset":
            X_data = st.session_state.X_val
            y_data = st.session_state.y_val
            predictions = model_manager.model(X_data).cpu().numpy()
            true_values = y_data.cpu().numpy()

        # Phần này viết lại sau
        # elif evaluation_option == "Forecast n Steps Ahead":
        #     # Lấy dữ liệu gần nhất để dự báo
        #     X_last = st.session_state.X_test[-1:]  # Dùng sample cuối cùng của tập test để dự đoán n bước
        #     predictions = []
        #     for _ in range(n_steps):
        #         y_pred = model_manager.model(X_last)
        #         predictions.append(y_pred.cpu().numpy())
                
        #         # Cập nhật X_last để tiếp tục dự đoán cho bước tiếp theo
        #         X_last = torch.cat((X_last[:, 1:, :], y_pred.unsqueeze(1)), dim=1)

        #     predictions = np.array(predictions).squeeze()
        elif evaluation_option == "Forecast n Steps Ahead":
            # Dùng mẫu cuối cùng của X_test để bắt đầu dự báo
            X_last = st.session_state.X_test[-1:].clone()  # Giữ nguyên kích thước (1, 7, 34) cho đầu vào
            predictions = []
            steps_to_predict = n_steps

            while steps_to_predict > 0:
                # Dự báo cho số bước tiếp theo (steps_ahead)
                y_pred = model_manager.model(X_last)  # Kích thước (1, steps_ahead, 34)
                y_pred_np = y_pred.cpu().numpy().squeeze()  # Chuyển sang NumPy và bỏ kích thước không cần thiết
                
                # Lưu y_pred cho mỗi bước
                predictions.append(y_pred_np)
                
                # Cập nhật X_last: lấy steps_ahead bước cuối cùng làm đầu vào mới
                # Nếu còn ít hơn steps_ahead bước để dự đoán, chỉ lấy phần cần thiết
                if steps_to_predict >= steps_ahead:
                    X_last = torch.cat((X_last[:, steps_ahead:, :], y_pred), dim=1)  # Giữ lại 7 bước
                else:
                    X_last = torch.cat((X_last[:, steps_ahead:, :], y_pred[:, :steps_to_predict, :]), dim=1)
                
                # Giảm số bước còn lại cần dự báo
                steps_to_predict -= steps_ahead
            
            # Chuyển predictions thành mảng NumPy
            predictions = np.concatenate(predictions[:n_steps], axis=0)  # Kích thước (n_steps, 34)

    return predictions, true_values

# Hàm vẽ kết quả cho Test Dataset và Validation Dataset với Plotly
def plot_test_validation_results(true_values, predictions, evaluation_option, selected_features, step_ahead):
    fig = go.Figure()

    for i, feature in enumerate(selected_features):
        # Thêm đường True Value cho mỗi đặc trưng
        fig.add_trace(go.Scatter(
            x=list(range(true_values.shape[0])),
            y=true_values[:, step_ahead, i],  # Sử dụng step_ahead đã chọn
            mode='lines',
            name=f"True {feature} - Step {step_ahead + 1}",
            hoverinfo="y"
        ))

        # Thêm đường Predicted Value cho mỗi đặc trưng
        fig.add_trace(go.Scatter(
            x=list(range(predictions.shape[0])),
            y=predictions[:, step_ahead, i],  # Sử dụng step_ahead đã chọn
            mode='lines',
            name=f"Predicted {feature} - Step {step_ahead + 1}",
            line=dict(dash='dash'),
            hoverinfo="y"
        ))

    # Cài đặt tiêu đề và trục
    fig.update_layout(
        title=f"{evaluation_option} - Step {step_ahead + 1} Ahead Prediction",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
        hovermode="x"
    )

    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)


# Hàm vẽ kết quả cho Forecast n Steps Ahead
# def plot_forecast_results(predictions, n_steps):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     for feature in range(34):  # Hoặc chỉ chọn các feature quan trọng để dễ xem hơn
#         ax.plot(range(len(predictions)), predictions[:, feature], linestyle='--', label=f"Predicted Feature {feature+1}")
#     ax.set_title(f"Forecast {n_steps} Steps Ahead")
#     ax.set_xlabel("Future Steps")
#     ax.set_ylabel("Predicted Values")
#     ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
#     st.pyplot(fig)
# Hàm vẽ kết quả cho Forecast n Steps Ahead

# def plot_forecast_results(predictions, n_steps, selected_features):
#     fig = go.Figure()

#     for i, feature in enumerate(selected_features):
#         fig.add_trace(go.Scatter(
#             x=list(range(n_steps)),
#             y=predictions[:n_steps, i],  # Dữ liệu dự báo cho từng đặc trưng
#             mode='lines',
#             name=f"Predicted {feature}",
#             hoverinfo="y"
#         ))

#     fig.update_layout(
#         title=f"Forecast {n_steps} Steps Ahead",
#         xaxis_title="Future Steps",
#         yaxis_title="Predicted Values",
#         legend_title="Legend",
#         hovermode="x"
#     )

#     st.plotly_chart(fig)

import plotly.graph_objects as go

# Hàm vẽ kết quả cho Forecast n Steps Ahead với dữ liệu X_last
def plot_forecast_results(X_last, predictions, n_steps, selected_features):
    fig = go.Figure()

    # Thêm dữ liệu X_last vào biểu đồ
    X_last_len = X_last.shape[1]  # Số bước trong X_last

    for i, feature in enumerate(selected_features):
        # Hiển thị X_last (dữ liệu ban đầu)
        fig.add_trace(go.Scatter(
            x=list(range(X_last_len)),
            # y=X_last[0, :, i].cpu().numpy(),  # Chuyển đổi tensor X_last thành numpy array
            y=X_last[0, :, i],  # Không cần gọi `.cpu()` nếu đã là numpy array
            mode='lines',
            name=f"Initial {feature}",
            hoverinfo="y",
            line=dict(color='blue')
        ))

        # Hiển thị dự báo tiếp theo cho feature
        fig.add_trace(go.Scatter(
            x=list(range(X_last_len, X_last_len + n_steps)),
            y=predictions[:n_steps, i],  # Dữ liệu dự báo cho từng đặc trưng
            mode='lines',
            name=f"Predicted {feature}",
            hoverinfo="y",
            line=dict(dash='dash', color='red')
        ))

    fig.update_layout(
        title=f"Forecast {n_steps} Steps Ahead with Initial Data",
        xaxis_title="Time",
        yaxis_title="Values",
        legend_title="Legend",
        hovermode="x"
    )

    st.plotly_chart(fig)


# Lựa chọn đặc trưng trong sidebar
selected_features = st.sidebar.multiselect(
    "Select Features to Display",
    options=[f"Feature {i+1}" for i in range(34)],
    default=["Feature 1", "Feature 2", "Feature 3"]  # Chọn mặc định một số đặc trưng quan trọng
)
# Thanh trượt chọn step ahead
# step_ahead = st.sidebar.slider("Select Step Ahead", min_value=1, max_value=st.session_state.ahead, value=1) - 1  # Trừ 1 để dùng làm chỉ số
# Kiểm tra nếu ahead lớn hơn 1 thì mới hiển thị thanh trượt
if st.session_state.ahead > 1:
    # Thanh trượt chọn step ahead (nếu ahead > 1)
    step_ahead = st.sidebar.slider("Select Step Ahead", min_value=1, max_value=st.session_state.ahead, value=1) - 1
else:
    # Đặt step_ahead mặc định là 0 nếu ahead = 1
    step_ahead = 1
    st.sidebar.write("Only 1 step ahead available, no selection needed.")

# Nút 7: Plot Result theo 3 trường hợp tuỳ chọn
if st.button("Plot Result"):
    # # Lấy kết quả dự đoán
    # predictions, true_values = evaluate_and_predict(evaluation_option, st.session_state.model_manager, n_steps)   
    
    # # Giải chuẩn hóa cả X và y nếu đã normalize
    # if 'y_scaler' in st.session_state:
    #     predictions = st.session_state.y_scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
    #     true_values = st.session_state.y_scaler.inverse_transform(true_values.reshape(-1, true_values.shape[-1])).reshape(true_values.shape)
              
    # # Lấy các chỉ số của đặc trưng được chọn
    # selected_feature_indices = [int(f.split()[1]) - 1 for f in selected_features]

    # Gọi hàm vẽ tương ứng
    if evaluation_option in ["Test Dataset", "Validation Dataset"]:

        # copy khối ngoài bỏ vào đây************************
        # Lấy kết quả dự đoán
        predictions, true_values = evaluate_and_predict(evaluation_option, st.session_state.model_manager, n_steps)   
        
        # Giải chuẩn hóa cả X và y nếu đã normalize
        if 'y_scaler' in st.session_state:
            predictions = st.session_state.y_scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
            true_values = st.session_state.y_scaler.inverse_transform(true_values.reshape(-1, true_values.shape[-1])).reshape(true_values.shape)
                
        # Lấy các chỉ số của đặc trưng được chọn
        selected_feature_indices = [int(f.split()[1]) - 1 for f in selected_features]        
        # copy khối ngoài bỏ vào đây***********************

        # Lọc true_values và predictions theo các đặc trưng đã chọn
        # selected_feature_indices = [int(f.split()[1]) - 1 for f in selected_features]

        # true_values = true_values[:, :, selected_feature_indices]
        # predictions = predictions[:, :, selected_feature_indices]

        ##############################################################
        # có vẻ đây là chìa khoá cho vụ ahead = 1
        # Lọc theo các đặc trưng đã chọn dựa trên số chiều
        if predictions.ndim == 3:
            predictions = predictions[:, :, selected_feature_indices]
            true_values = true_values[:, :, selected_feature_indices] if true_values is not None else None
        else:
            predictions = predictions[:, selected_feature_indices]


        # # Kiểm tra số chiều của true_values và predictions
        # if true_values.ndim == 2:
        #     true_values = true_values[:, selected_feature_indices]
        #     predictions = predictions[:, selected_feature_indices]
        # else:
        #     true_values = true_values[:, :, selected_feature_indices]
        #     predictions = predictions[:, :, selected_feature_indices]

        # Gọi hàm vẽ với các đặc trưng đã chọn
        # plot_test_validation_results(true_values, predictions, evaluation_option)
        # plot_test_validation_results(true_values, predictions, evaluation_option, selected_features)
        plot_test_validation_results(true_values, predictions, evaluation_option, selected_features, step_ahead)

    elif evaluation_option == "Forecast n Steps Ahead":
        # plot_forecast_results(predictions, n_steps)
        # predictions = predictions[:, selected_feature_indices]
        # plot_forecast_results(predictions, n_steps, selected_features)

        # copy khối ngoài bỏ vào đây************************
        # Lấy kết quả dự đoán
        predictions, true_values = evaluate_and_predict(evaluation_option, st.session_state.model_manager, n_steps)   
        
        # Giải chuẩn hóa cả X và y nếu đã normalize
        if 'y_scaler' in st.session_state:
            predictions = st.session_state.y_scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
            # true_values = st.session_state.y_scaler.inverse_transform(true_values.reshape(-1, true_values.shape[-1])).reshape(true_values.shape)
                
        # Lấy các chỉ số của đặc trưng được chọn
        selected_feature_indices = [int(f.split()[1]) - 1 for f in selected_features]        
        # copy khối ngoài bỏ vào đây***********************


        # Lấy mẫu cuối cùng từ X_test để làm X_last
        # X_last = st.session_state.X_test[-1:, :, selected_feature_indices]  # Lọc các đặc trưng đã chọn

        # Lấy mẫu cuối cùng từ X_test để làm X_last (không lọc đặc trưng lúc này)
        X_last = st.session_state.X_test[-1:, :, :]  # Lấy toàn bộ đặc trưng trước khi giải chuẩn hóa

        # Giải chuẩn hóa X_last nếu x_scaler có tồn tại
        if 'x_scaler' in st.session_state:
            X_last = st.session_state.x_scaler.inverse_transform(
                X_last.reshape(-1, X_last.shape[-1])
            ).reshape(X_last.shape)

        # Lọc X_last và predictions theo các đặc trưng đã chọn sau khi giải chuẩn hóa
        X_last = X_last[:, :, selected_feature_indices]

        predictions = predictions[:, selected_feature_indices]        

        # predictions = predictions[:, selected_feature_indices]
        plot_forecast_results(X_last, predictions, n_steps, selected_features)


# D. readme.md
# Khi bạn nạp dữ liệu vào mô hình LSTM, kích thước của các tensor đầu vào nên như sau:
# X_train: (batch_size, sequence_length, input_size)
# y_train: (batch_size, output_size), nếu output_size > 1, hoặc (batch_size,) nếu output_size = 1
# sequence_length chính là timesteps
# còn input_size chính là số lượng features đầu vào
# còn output_size là số features muốn dự đoán

