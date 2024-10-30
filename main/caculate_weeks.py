# Thư viện numpy để tính toán RMSE và MAPE
import numpy as np
# Các thư viện từ scikit-learn để tính toán MSE, MAE, R²
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Thư viện dtaidistance để tính toán Dynamic Time Warping (DTW)
from dtaidistance import dtw
import pandas as pd
import plotly.graph_objects as go
import re

# Function tính tương quan----------------------------------------------------------------------
def calculate_mape(week1_df, week2_df):
    """Tính MAPE giữa hai tuần"""
    return np.mean(np.abs((week1_df['Close'] - week2_df['Close']) / week2_df['Close'])) * 100

def calculate_dtw(week1_df, week2_df):
    """Tính khoảng cách DTW giữa hai tuần"""
    return dtw.distance(week1_df['Close'].values, week2_df['Close'].values)

# Hàm tính Correlation với xử lý NaN
def calculate_correlation(week1_df, week2_df):
    """Tính Correlation giữa hai tuần với xử lý NaN"""
    return week1_df['Close'].corr(week2_df['Close'], method='pearson')

# Hàm tính MAPE với kiểm tra để tránh chia cho 0
def calculate_mape(week1_df, week2_df):
    """Tính MAPE giữa hai tuần với kiểm tra tránh chia cho 0"""
    close_week2 = week2_df['Close'].replace(0, np.nan)  # Thay 0 bằng NaN để tránh chia cho 0
    return np.mean(np.abs((week1_df['Close'] - week2_df['Close']) / close_week2)) * 100

# Function normalize---------------------------------------------------------------------------
# Hàm normalize Min-Max
def normalize_min_max(df):
    return (df - df.min()) / (df.max() - df.min())

# Hàm normalize Z-score (Standardization)
def normalize_z_score(df):
    return (df - df.mean()) / df.std()

# Hàm để lấy số từ chuỗi kết quả của compare_weeks---------------------------------------------
def extract_number_from_result(result):
    """Lọc số từ chuỗi kết quả trả về từ compare_weeks."""
    # Sử dụng regex để lấy phần số từ chuỗi (bao gồm cả số thập phân)
    match = re.search(r"[-+]?\d*\.\d+|\d+", result)
    if match:
        return float(match.group())  # Trả về số ở dạng float
    return 0  # Nếu không tìm thấy số, trả về nguyên câu


# Tạo từ điển chứa mô tả cho từng phương pháp--------------------------------------------------
method_descriptions = {
    'MSE': '''**Mean Squared Error (MSE)** đo lường bình phương trung bình của sai số giữa các giá trị thực và dự đoán.
    \n- **Ý nghĩa**: MSE phạt nặng các sai số lớn hơn vì sai số được bình phương, do đó nhạy cảm hơn với các ngoại lệ (outliers).
    \n- **Kết quả**: MSE càng thấp càng tốt, giá trị càng nhỏ biểu thị sự khác biệt nhỏ giữa giá trị dự đoán và thực tế. Thường dao động từ 0 đến vô cùng, giá trị 0 là lý tưởng.''',

    'MAE': '''**Mean Absolute Error (MAE)** đo lường giá trị tuyệt đối trung bình của các sai số giữa các giá trị thực và dự đoán.
    \n- **Ý nghĩa**: MAE ít nhạy cảm với các ngoại lệ vì nó chỉ đo lường sai số theo giá trị tuyệt đối.
    \n- **Kết quả**: MAE càng thấp càng tốt, giá trị càng nhỏ biểu thị dự đoán gần với thực tế. Dao động từ 0 đến vô cùng.''',

    'RMSE': '''**Root Mean Squared Error (RMSE)** là căn bậc hai của MSE, giúp đo lường sai số với đơn vị giống với dữ liệu gốc.
    \n- **Ý nghĩa**: RMSE dễ hiểu hơn MSE vì nó đưa sai số về cùng đơn vị với dữ liệu gốc.
    \n- **Kết quả**: RMSE càng thấp càng tốt, giá trị thấp biểu thị sai số nhỏ. Giá trị lý tưởng là 0.''',

    'Correlation': '''**Correlation (Tương quan)** đo lường mức độ tương quan tuyến tính giữa hai tập dữ liệu.
    \n- **Ý nghĩa**: Tương quan thể hiện mối quan hệ tuyến tính, với giá trị gần 1 biểu thị mối tương quan tích cực mạnh và gần -1 là mối tương quan âm mạnh.
    \n- **Kết quả**: Giá trị dao động từ -1 đến 1. Giá trị gần 1 là tốt nếu hai tập dữ liệu có mối tương quan tích cực.''',

    'R²': '''**R² (Hệ số xác định)** đo lường mức độ mà một biến dự đoán có thể giải thích biến khác.
    \n- **Ý nghĩa**: R² cho biết tỷ lệ phần trăm sự biến thiên của biến phụ thuộc được giải thích bởi biến độc lập.
    \n- **Kết quả**: Dao động từ 0 đến 1. Giá trị gần 1 là tốt, biểu thị mô hình có thể giải thích tốt sự biến thiên của dữ liệu.''',

    'MAPE': '''**Mean Absolute Percentage Error (MAPE)** đo lường sai số trung bình theo phần trăm giữa giá trị thực và dự đoán.
    \n- **Ý nghĩa**: MAPE giúp so sánh sai số dưới dạng phần trăm, dễ hiểu trong các ngữ cảnh tài chính.
    \n- **Kết quả**: MAPE càng thấp càng tốt. Dao động từ 0% đến 100% hoặc cao hơn. MAPE dưới 10% được coi là rất tốt.''',

    'DTW': '''**Dynamic Time Warping (DTW)** đo lường khoảng cách giữa hai chuỗi thời gian cho dù có sự thay đổi về thời gian.
    \n- **Ý nghĩa**: DTW hữu ích khi so sánh các chuỗi thời gian có biến đổi khác nhau trong thời gian, ví dụ, chuỗi có thể bị giãn hoặc co nhưng vẫn có hình dạng tương tự.
    \n- **Kết quả**: Giá trị càng thấp càng tốt, biểu thị hai chuỗi càng tương tự nhau.''',
}


# Function 1-----------------------------------------------------------------------------
# Hàm tính toán sự tương quan hoặc MSE giữa hai tuần
def compare_weeks(week1_df, week2_df, method='MSE'):
    if len(week1_df) != len(week2_df):
        return "Weeks have different number of data points, comparison not possible."
    
    if method == 'MSE':
        return f"Mean Squared Error (MSE) between two weeks: {mean_squared_error(week1_df['Close'], week2_df['Close'])}"
    elif method == 'MAE':
        return f"Mean Absolute Error (MAE) between two weeks: {mean_absolute_error(week1_df['Close'], week2_df['Close'])}"
    elif method == 'RMSE':
        return f"Root Mean Squared Error (RMSE) between two weeks: {np.sqrt(mean_squared_error(week1_df['Close'], week2_df['Close']))}"
    elif method == 'Correlation':
        correlation = calculate_correlation(week1_df, week2_df)
        if pd.isna(correlation):
            return "Correlation is NaN. Please check the data for any inconsistencies."
        return f"Correlation between two weeks: {correlation}"
    elif method == 'R²':
        return f"R² score between two weeks: {r2_score(week1_df['Close'], week2_df['Close'])}"
    elif method == 'MAPE':
        return f"Mean Absolute Percentage Error (MAPE) between two weeks: {calculate_mape(week1_df, week2_df)}%"
    elif method == 'DTW':
        return f"Dynamic Time Warping (DTW) distance between two weeks: {calculate_dtw(week1_df, week2_df)}"

# Function 2-----------------------------------------------------------------------------
def compare_with_all_weeks(df, week1_df, year1, week1, available_weeks, normalize_method, comparison_method, st):
    """Hàm so sánh week1 với tất cả các tuần còn lại, áp dụng normalization và trả về bảng kết quả cùng heatmap."""
    
    # Normalize dữ liệu của week1 nếu cần
    if normalize_method == 'Min-Max':
        week1_df['Close'] = normalize_min_max(week1_df['Close'])
    elif normalize_method == 'Z-score':
        week1_df['Close'] = normalize_z_score(week1_df['Close'])

    # Tạo một DataFrame để lưu kết quả so sánh
    comparison_results = []

    # Lặp qua tất cả các tuần để so sánh với week1
    for _, row in available_weeks.iterrows():
        year2, week2 = row['Year'], row['Week']
        if year1 == year2 and week1 == week2:
            continue  # Bỏ qua nếu tuần đang so sánh là chính tuần 1

        # Lọc dữ liệu của từng tuần còn lại
        week2_df = df[(df['Year'] == year2) & (df['Week'] == week2)]

        # Normalize dữ liệu của week2 nếu cần
        if normalize_method == 'Min-Max':
            week2_df['Close'] = normalize_min_max(week2_df['Close'])
        elif normalize_method == 'Z-score':
            week2_df['Close'] = normalize_z_score(week2_df['Close'])

        # Tính toán chỉ số so sánh theo phương pháp đã chọn
        comparison_result = compare_weeks(week1_df, week2_df, method=comparison_method)
        # Lọc lấy giá trị số từ kết quả
        comparison_value = extract_number_from_result(comparison_result)

        # Lưu kết quả (năm, tuần, giá trị so sánh)
        comparison_results.append((f"{year2}-W{week2}", comparison_value))

    # Chuyển đổi kết quả thành DataFrame
    comparison_df = pd.DataFrame(comparison_results, columns=['Week', comparison_method])
    # Sắp xếp DataFrame theo cột giá trị so sánh (ví dụ: MSE, MAE, v.v.)
    comparison_df = comparison_df.sort_values(by=comparison_method, ascending=True)  # Sắp xếp tăng dần theo giá trị so sánh

    # return comparison_df
    # Hiển thị bảng kết quả với kích thước tùy chỉnh
    st.write("### Comparison Results")
    st.dataframe(comparison_df, height=600, width=600)