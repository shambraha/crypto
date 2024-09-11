import os
import pandas as pd
import numpy as np
# Thư viện numpy để tính toán RMSE và MAPE
# Các thư viện từ scikit-learn để tính toán MSE, MAE, R²
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Thư viện dtaidistance để tính toán Dynamic Time Warping (DTW)
from dtaidistance import dtw
from config.config import csv_folder_path

#folder_path = r'E:\MyPythonCode\newestCSV'
# Đường dẫn tương đối tới thư mục 'csv' nằm trong thư mục 'data' nằm trong vị trí hiện tại
# folder_path = os.path.join(os.getcwd(), 'data', 'csv')
#print(folder_path)

# Veriable 1-----------------------------------------------
# color_schemes = {
#     'green_red': {'increasing': '#28a745', 'decreasing': '#dc3545'},  # Xanh lục - Đỏ
#     'blue_orange': {'increasing': '#007bff', 'decreasing': '#fd7e14'},  # Xanh da trời - Cam
#     'yellow_purple': {'increasing': '#ffc107', 'decreasing': '#6f42c1'},  # Vàng - Tím
#     'darkblue_pink': {'increasing': '#17a2b8', 'decreasing': '#e83e8c'},  # Xanh dương đậm - Hồng
#     'darkgreen_brown': {'increasing': '#28a745', 'decreasing': '#795548'},  # Xanh lá cây đậm - Nâu
#     'orange_lightblue': {'increasing': '#ff851b', 'decreasing': '#39cccc'}   # Cam đậm - Xanh nhạt
# }
color_schemes = {
    '1': {'increasing': '#28a745', 'decreasing': '#dc3545'},  # Xanh lục - Đỏ
    '2': {'increasing': '#007bff', 'decreasing': '#fd7e14'},  # Xanh da trời - Cam
    '3': {'increasing': '#ffc107', 'decreasing': '#6f42c1'},  # Vàng - Tím
    '4': {'increasing': '#17a2b8', 'decreasing': '#e83e8c'},  # Xanh dương đậm - Hồng
    '5': {'increasing': '#28a745', 'decreasing': '#795548'},  # Xanh lá cây đậm - Nâu
    '6': {'increasing': '#ff851b', 'decreasing': '#39cccc'}   # Cam đậm - Xanh nhạt
}


# Function1------------------------------------------------
def chooseFormattimeColumn(df, column_name):
    df['Formated Time'] = pd.to_datetime(df[column_name], unit='ms')
    #df['Formated Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    
# Function2-----------------------------------------------
def get_df_from_name(symbol):    
    file_name = os.path.join(csv_folder_path, f'binance_{symbol}USDT.csv')
    df = pd.read_csv(file_name)
    chooseFormattimeColumn(df, 'Open Time')
    return df

# Function3----------------------------------------------
def showinfo_dtypes_head_tail(df):
    print(df.dtypes)
    print(df.head())
    print(df.tail())

# Function 4---------------------------------------------
# Lọc dff theo khung giờ (timeframe) và khoảng thời gian tuỳ chỉnh
def filter_and_resample_df(df, timeframe, start_time, end_time):
    # Filter df based on user inputs [start_time, end_time]
    df1_filtered = df[(df['Formated Time'] >= pd.to_datetime(start_time)) & (df['Formated Time'] <= pd.to_datetime(end_time))]
    #df2_filtered = df_ETH[(df_ETH['Formated Time'] >= pd.to_datetime(start_time)) & (df_ETH['Formated Time'] <= pd.to_datetime(end_time))]

    # Resample the df data based on user inputs [timeframe]
    # resample_rule = {"1h": "1H", "4h": "4H", "12h": "12H"}[timeframe]
    resample_rule = timeframe
    df1_resampled = df1_filtered.resample(resample_rule, on='Formated Time').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    # df2_resampled = df2_filtered.resample(resample_rule, on='Formated Time').agg({
    #     'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    # }).dropna()
    return df1_resampled

# Function 5----------------------------------------------
# Hàm liệt kê các tệp CSV trong thư mục cục bộ và trích xuất symbols
def extract_symbols_from_local_path(local_folder_path):
    symbols = []
    # Duyệt qua tất cả các tệp trong thư mục
    for file_name in os.listdir(local_folder_path):
        # Kiểm tra định dạng tệp để đảm bảo chỉ lấy những tệp có định dạng binance_<symbol>USDT.csv
        if file_name.startswith('binance_') and file_name.endswith('USDT.csv'):
            # Lấy phần symbol từ tên tệp, ví dụ: binance_BTCUSDT.csv -> BTC
            symbol = file_name.split('_')[1].replace('USDT.csv', '')
            symbols.append(symbol)
    return symbols

# Function 6----------------------------------------------
# Hàm tính toán sự tương quan hoặc MSE giữa hai tuần
# def compare_weeks(week1_df, week2_df, method='MSE'):
#     if len(week1_df) != len(week2_df):
#         return "Weeks have different number of data points, comparison not possible."
    
#     if method == 'MSE':
#         mse = mean_squared_error(week1_df['Close'], week2_df['Close'])
#         return f"Mean Squared Error (MSE) between two weeks: {mse}"
#     elif method == 'Correlation':
#         correlation = week1_df['Close'].corr(week2_df['Close'])
#         return f"Correlation between two weeks: {correlation}"

# def calculate_mape(week1_df, week2_df):
#     """Tính MAPE giữa hai tuần"""
#     return np.mean(np.abs((week1_df['Close'] - week2_df['Close']) / week2_df['Close'])) * 100

# def calculate_dtw(week1_df, week2_df):
#     """Tính khoảng cách DTW giữa hai tuần"""
#     return dtw.distance(week1_df['Close'].values, week2_df['Close'].values)

# # Hàm tính Correlation với xử lý NaN
# def calculate_correlation(week1_df, week2_df):
#     """Tính Correlation giữa hai tuần với xử lý NaN"""
#     return week1_df['Close'].corr(week2_df['Close'], method='pearson')

# # Hàm tính MAPE với kiểm tra để tránh chia cho 0
# def calculate_mape(week1_df, week2_df):
#     """Tính MAPE giữa hai tuần với kiểm tra tránh chia cho 0"""
#     close_week2 = week2_df['Close'].replace(0, np.nan)  # Thay 0 bằng NaN để tránh chia cho 0
#     return np.mean(np.abs((week1_df['Close'] - week2_df['Close']) / close_week2)) * 100

# # Hàm normalize Min-Max
# def normalize_min_max(df):
#     return (df - df.min()) / (df.max() - df.min())

# # Hàm normalize Z-score (Standardization)
# def normalize_z_score(df):
#     return (df - df.mean()) / df.std()

# # Tạo từ điển chứa mô tả cho từng phương pháp
# method_descriptions = {
#     'MSE': '''**Mean Squared Error (MSE)** đo lường bình phương trung bình của sai số giữa các giá trị thực và dự đoán.
#     \n-  **Ý nghĩa**: MSE phạt nặng các sai số lớn hơn vì sai số được bình phương, do đó nhạy cảm hơn với các ngoại lệ (outliers).''',

#     'MAE': '''**Mean Absolute Error (MAE)** đo lường giá trị tuyệt đối trung bình của các sai số giữa các giá trị thực và dự đoán.
#     \n- **Ý nghĩa**: MAE ít nhạy cảm với các ngoại lệ vì nó chỉ đo lường sai số theo giá trị tuyệt đối.''',

#     'RMSE': '''**Root Mean Squared Error (RMSE)** là căn bậc hai của MSE, giúp đo lường sai số với đơn vị giống với dữ liệu gốc.
#     \n- **Ý nghĩa**: RMSE dễ hiểu hơn MSE vì nó đưa sai số về cùng đơn vị với dữ liệu gốc.''',

#     'Correlation': '''**Correlation (Tương quan)** đo lường mức độ tương quan tuyến tính giữa hai tập dữ liệu.
#     \n- **Ý nghĩa**: Tương quan thể hiện mối quan hệ tuyến tính, với giá trị gần 1 biểu thị mối tương quan tích cực mạnh và gần -1 là mối tương quan âm mạnh.''',

#     'R²': '''**R² (Hệ số xác định)** đo lường mức độ mà một biến dự đoán có thể giải thích biến khác.
#     \n- **Ý nghĩa**: R² cho biết tỷ lệ phần trăm sự biến thiên của biến phụ thuộc được giải thích bởi biến độc lập.''',

#     'MAPE': '''**Mean Absolute Percentage Error (MAPE)** đo lường sai số trung bình theo phần trăm giữa giá trị thực và dự đoán.
#     \n- **Ý nghĩa**: MAPE giúp so sánh sai số dưới dạng phần trăm, dễ hiểu trong các ngữ cảnh tài chính.''',

#     'DTW': '''**Dynamic Time Warping (DTW)** đo lường khoảng cách giữa hai chuỗi thời gian cho dù có sự thay đổi về thời gian.
#     \n- **Ý nghĩa**: DTW hữu ích khi so sánh các chuỗi thời gian có biến đổi khác nhau trong thời gian, ví dụ, chuỗi có thể bị giãn hoặc co nhưng vẫn có hình dạng tương tự.''',
# }


# # Hàm tính toán sự tương quan hoặc MSE giữa hai tuần
# def compare_weeks(week1_df, week2_df, method='MSE'):
#     if len(week1_df) != len(week2_df):
#         return "Weeks have different number of data points, comparison not possible."
    
#     if method == 'MSE':
#         return f"Mean Squared Error (MSE) between two weeks: {mean_squared_error(week1_df['Close'], week2_df['Close'])}"
#     elif method == 'MAE':
#         return f"Mean Absolute Error (MAE) between two weeks: {mean_absolute_error(week1_df['Close'], week2_df['Close'])}"
#     elif method == 'RMSE':
#         return f"Root Mean Squared Error (RMSE) between two weeks: {np.sqrt(mean_squared_error(week1_df['Close'], week2_df['Close']))}"
#     elif method == 'Correlation':
#         correlation = calculate_correlation(week1_df, week2_df)
#         if pd.isna(correlation):
#             return "Correlation is NaN. Please check the data for any inconsistencies."
#         return f"Correlation between two weeks: {correlation}"
#     elif method == 'R²':
#         return f"R² score between two weeks: {r2_score(week1_df['Close'], week2_df['Close'])}"
#     elif method == 'MAPE':
#         return f"Mean Absolute Percentage Error (MAPE) between two weeks: {calculate_mape(week1_df, week2_df)}%"
#     elif method == 'DTW':
#         return f"Dynamic Time Warping (DTW) distance between two weeks: {calculate_dtw(week1_df, week2_df)}"
