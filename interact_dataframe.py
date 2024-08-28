import os
import pandas as pd
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
