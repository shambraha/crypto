import os
import pandas as pd
import numpy as np
# Thư viện numpy để tính toán RMSE và MAPE
# Các thư viện từ scikit-learn để tính toán MSE, MAE, R²
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Thư viện dtaidistance để tính toán Dynamic Time Warping (DTW)
from dtaidistance import dtw
from config.config import csv_folder_path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyts.approximation import PiecewiseAggregateApproximation

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

# Function 4A---------------------------------------------
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

# Function 4B---------------------------------------------
# đúng như tên hàm, resample df theo timeframe chỉ định
def resample_df(df, timeframe):
    resample_rule = timeframe
    df_resampled = df.resample(resample_rule, on='Formated Time').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return df_resampled

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
# Tạo danh sách các cửa sổ trượt trên df với window_size chỉ định
def create_sliding_windows(df, window_size):
    windows = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        windows.append(window)
    return windows

def create_overlapping_windows(df_resampled, window_size, overlap_size):
    step_size = window_size - overlap_size  # Kích thước bước giảm đi do chồng lấn
    overlapping_windows = []
    for i in range(0, len(df_resampled) - window_size + 1, step_size):
        window = df_resampled[i:i + window_size]
        overlapping_windows.append(window)
    return overlapping_windows

def create_paa_segments_old(df_resampled, num_segments):
    paa = PiecewiseAggregateApproximation(n_segments=num_segments)
    paa_values = paa.fit_transform([df_resampled['Close'].values])
    return paa_values

def create_paa_segments(df, num_segments):
    """
    Chia dữ liệu thành các phân đoạn bằng PAA (Piecewise Aggregate Approximation).
    Args:
        df: DataFrame chuẩn hóa.
        num_segments: Số lượng phân đoạn mong muốn.

    Returns:
        List chứa các phân đoạn đã áp dụng PAA.
    """
    n = len(df)  # Số lượng điểm dữ liệu
    segment_size = n // num_segments  # Kích thước mỗi phân đoạn

    paa_segments = []
    for i in range(0, n, segment_size):
        segment = df.iloc[i:i + segment_size].mean()  # Tính trung bình mỗi phân đoạn
        paa_segments.append(segment)
    
    return np.array(paa_segments)  # Trả về mảng PAA

def create_variable_windows_old(df_resampled, min_size, max_size, step_size):
    variable_windows = []
    for size in range(min_size, max_size, step_size):  # Thử các kích thước cửa sổ khác nhau
        windows = create_sliding_windows(df_resampled, size)
        variable_windows.extend(windows)
    return variable_windows

def create_variable_windows(df, min_size, max_size, step_size):
    """
    Tạo các cửa sổ với kích thước thay đổi từ min_size đến max_size.
    Args:
        df: DataFrame chuẩn hóa.
        min_size: Kích thước nhỏ nhất của cửa sổ.
        max_size: Kích thước lớn nhất của cửa sổ.
        step_size: Bước nhảy giữa các cửa sổ.

    Returns:
        List chứa các cửa sổ có kích thước thay đổi.
    """
    windows = []
    for size in range(min_size, max_size + 1, step_size):
        for i in range(0, len(df) - size + 1, step_size):
            window = df.iloc[i:i + size]
            windows.append(window)
    
    return windows


def create_pyramid_windows(df_resampled, base_window_size, num_levels):
    pyramid_windows = []
    for level in range(num_levels):
        window_size = base_window_size // (2 ** level)  # Tăng kích thước cửa sổ theo cấp độ
        windows_at_level = create_sliding_windows(df_resampled, window_size)
        pyramid_windows.extend(windows_at_level)
    return pyramid_windows

# Function 7A----------------------------------------------
# Tính toán khoảng cách DTW cho từng đoạn trượt
def compare_with_dtw(df_resampled, sliding_windows):
    dtw_distances = []
    for window in sliding_windows:
        # Chuyển các chuỗi về dạng mảng numpy để tính DTW
        resampled_values = df_resampled['Close'].values
        window_values = window['Close'].values
        
        # Tính khoảng cách DTW
        distance = dtw.distance(resampled_values, window_values)
        dtw_distances.append(distance)
    
    return dtw_distances
# Function 7B----------------------------------------------
def compare_with_pmk(df_resampled, sliding_windows, num_levels=5):
    pmk_distances = []
    
    # Chuyển cột 'Close' trong df_resampled thành chuỗi 1D
    resampled_values = df_resampled['Close'].values
    
    for window in sliding_windows:
        # Chuyển cột 'Close' trong mỗi window thành chuỗi 1D
        window_values = window['Close'].values
        
        # Tính khoảng cách PMK giữa df_resampled và sliding window
        distance = compute_pmk(resampled_values, window_values, num_levels=num_levels)
        pmk_distances.append(distance)
    
    return pmk_distances

def compute_pmk(ts_a, ts_b, num_levels=3):
    total_distance = 0
    
    # Loop qua các cấp độ pyramid (từ chi tiết đến tổng quát)
    for level in range(num_levels):
        # Xác định kích thước bin (mức phân đoạn) cho cấp độ này
        bin_size = 2 ** level
        
        # Tạo histogram cho mỗi chuỗi ở cấp độ này
        hist_a = create_histogram(ts_a, bin_size)
        hist_b = create_histogram(ts_b, bin_size)
        
        # Tính số điểm tương đồng ở cấp độ này
        matches = np.minimum(hist_a, hist_b).sum()
        
        # Tính trọng số cho các điểm tương đồng ở cấp độ này (càng cao càng chi tiết)
        total_distance += matches / bin_size
    
    return total_distance

def create_histogram(ts, bin_size):
    # Chia chuỗi thành các bins và đếm số lượng điểm trong mỗi bin
    return np.histogram(ts, bins=np.arange(0, len(ts), bin_size))[0]

# Function 8----------------------------------------------
# Lựa chọn đoạn có khoảng cách DTW nhỏ nhất, và lấy chỉ số của slicing_windows tương ứng
def find_best_match(dtw_distances, sliding_windows):
    min_distance = min(dtw_distances)
    best_match_index = dtw_distances.index(min_distance)
    best_match_window = sliding_windows[best_match_index]
    
    return best_match_window, best_match_index, min_distance
def find_worst_match(dtw_distances, sliding_windows):
    max_distance = max(dtw_distances)
    best_match_index = dtw_distances.index(max_distance)
    best_match_window = sliding_windows[best_match_index]
    
    return best_match_window, best_match_index, max_distance

# Function 9----------------------------------------------
# 2 methos chuẩn hoá dữ liệu trước khi so sánh dtw
# Giả sử df_resampled đã có các cột Open, High, Low, Close, Volume
def normalize_min_max(df):
    # Tạo một bản sao của df để không thay đổi dữ liệu gốc
    df_normalized = df.copy()
    # Chuẩn hóa các cột liên quan đến giá trị
    scaler = MinMaxScaler()
    df_normalized[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df_normalized[['Open', 'High', 'Low', 'Close', 'Volume']])
    # Trả về DataFrame đã chuẩn hóa
    return df_normalized

def normalize_z_score(df):
    df_normalized = df.copy()
    scaler = StandardScaler()
    # Chỉ chuẩn hóa các cột liên quan đến giá trị
    df_normalized[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df_normalized[['Open', 'High', 'Low', 'Close', 'Volume']])
    return df_normalized

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
