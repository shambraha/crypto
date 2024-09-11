import streamlit as st
from interact_dataframe import extract_symbols_from_local_path, get_df_from_name
from caculate_weeks import normalize_min_max, normalize_z_score
from caculate_weeks import method_descriptions
from caculate_weeks import compare_weeks, compare_with_all_weeks
from interact_streamlit import plot_weeks_comparison
from sklearn.metrics import mean_squared_error
import numpy as np

from config.config import csv_folder_path
st.set_page_config(layout="wide")


# Lấy danh sách symbol từ local_folder
available_symbols_local = extract_symbols_from_local_path(csv_folder_path)

# Sử dụng selectbox thay cho multiselect để chỉ chọn 1 symbol
selected_symbol = st.sidebar.selectbox("Select Symbol", available_symbols_local)

# Đọc file df từ lựa chọn
df = get_df_from_name(selected_symbol)

# Bạn có thể tiếp tục sử dụng df cho các thao tác khác
st.write(f"You have selected: {selected_symbol}")
st.dataframe(df)  # Hiển thị dữ liệu của symbol đã chọn

# Load dữ liệu
df.set_index('Formated Time', inplace=True)
df['DayofWeek'] = df.index.dayofweek
df['Week'] = df.index.isocalendar().week
df['Year'] = df.index.year

# Tạo danh sách các tuần có trong dữ liệu và sắp xếp theo thứ tự giảm dần (từ gần nhất đến xa nhất)
available_weeks = df[['Year', 'Week']].drop_duplicates().sort_values(by=['Year', 'Week'], ascending=[False, False])

# # Chọn tuần đầu tiên và tuần thứ hai
# week1 = st.sidebar.selectbox("Select First Week", available_weeks.apply(lambda x: f"{x['Year']}-W{x['Week']}", axis=1))
# week2 = st.sidebar.selectbox("Select Second Week", available_weeks.apply(lambda x: f"{x['Year']}-W{x['Week']}", axis=1))

# Tìm tuần gần nhất (tuần hiện tại)
latest_year = df['Year'].max()
latest_week = df[df['Year'] == latest_year]['Week'].max()
latest_week_str = f"{latest_year}-W{latest_week}"

# Chọn tuần đầu tiên và tuần thứ hai, với tuần gần nhất làm giá trị mặc định
week1 = st.sidebar.selectbox(
    "Select First Week", 
    available_weeks.apply(lambda x: f"{x['Year']}-W{x['Week']}", axis=1), 
    index=available_weeks.apply(lambda x: f"{x['Year']}-W{x['Week']}", axis=1).tolist().index(latest_week_str)
)

week2 = st.sidebar.selectbox(
    "Select Second Week", 
    available_weeks.apply(lambda x: f"{x['Year']}-W{x['Week']}", axis=1),
    index=available_weeks.apply(lambda x: f"{x['Year']}-W{x['Week']}", axis=1).tolist().index(latest_week_str)
)

# Tách thông tin năm và tuần từ lựa chọn của người dùng
year1, week1 = map(int, week1.split('-W'))
year2, week2 = map(int, week2.split('-W'))

# Lọc dữ liệu của hai tuần được chọn
week1_df = df[(df['Year'] == year1) & (df['Week'] == week1)]
week2_df = df[(df['Year'] == year2) & (df['Week'] == week2)]

# Vẽ biểu đồ so sánh giữa 2 tuần được chọn
plot_weeks_comparison(week1_df, year1, week1, week2_df, year2, week2)

# Chọn kiểu Normalize trong Streamlit
normalize_method = st.sidebar.selectbox("Select Normalization Method", ['None', 'Min-Max', 'Z-score'])
# Normalize dữ liệu nếu cần
if normalize_method == 'Min-Max':
    week1_df['Close'] = normalize_min_max(week1_df['Close'])
    week2_df['Close'] = normalize_min_max(week2_df['Close'])
elif normalize_method == 'Z-score':
    week1_df['Close'] = normalize_z_score(week1_df['Close'])
    week2_df['Close'] = normalize_z_score(week2_df['Close'])

# Tạo selectbox để chọn phương pháp so sánh
comparison_method = st.sidebar.selectbox(
    "Select Comparison Method", 
    list(method_descriptions.keys())
)
# Hiển thị thông tin về phương pháp so sánh
st.markdown(f"### {comparison_method}")
st.write(method_descriptions[comparison_method])

# Hiển thị kết quả của phương pháp so sánh
comparison_result = compare_weeks(week1_df, week2_df, method=comparison_method)
st.write(comparison_result)


# Lưu trạng thái so sánh với tất cả các tuần trong session_state
if 'show_all_weeks' not in st.session_state:
    st.session_state['show_all_weeks'] = False  # Khởi tạo trạng thái mặc định

# Nút Compare với tất cả các tuần
compare_with_all = st.sidebar.button("Compare Week1 with AllWeeks")
if compare_with_all:
    # Kích hoạt trạng thái so sánh tất cả các tuần
    st.session_state['show_all_weeks'] = True
    # Gọi hàm để so sánh và hiển thị bảng kết quả cùng heatmap
    compare_with_all_weeks(df, week1_df, year1, week1, available_weeks, normalize_method, comparison_method, st=st)

# Hiển thị lại kết quả so sánh nếu đã bấm Compare Week1 with AllWeeks trước đó
if st.session_state['show_all_weeks']:
    compare_with_all_weeks(df, week1_df, year1, week1, available_weeks, normalize_method, comparison_method, st=st)
