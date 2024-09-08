import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from interact_dataframe import get_df_from_name, filter_and_resample_df
from interact_dataframe import extract_symbols_from_local_path
from interact_streamlit import create_sidebar_for_userinputs
from config.config import csv_folder_path
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Tăng kích thước của sidebar */
    [data-testid="stSidebar"] {
        width: 450px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Lấy danh sách symbol từ local_folder
available_symbols_local = extract_symbols_from_local_path(csv_folder_path)

# Bắt đầu giao diện Streamlit
st.title("Gainner/Loser Calculator")

# Phần chọn thời gian
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now()
if 'end_time' not in st.session_state:
    st.session_state['end_time'] = datetime.now()

# Nút "Calculate" ở phía trên
calculate = st.sidebar.button("Calculate")

# Phần chọn start_time và end_time thủ công
st.header("Select Time Range")
# st.session_state['start_time'] = st.sidebar.date_input("Select Start Date", value=st.session_state['start_time'])
# st.session_state['end_time'] = st.sidebar.date_input("Select End Date", value=st.session_state['end_time'])
st.session_state['start_time'] = st.date_input("Select Start Date", value=st.session_state['start_time'])
st.session_state['end_time'] = st.date_input("Select End Date", value=st.session_state['end_time'])

# Nút "+1" để tăng start_time và end_time thêm 1 ngày
if st.sidebar.button("+1 Day"):
    st.session_state['start_time'] += timedelta(days=1)
    st.session_state['end_time'] += timedelta(days=1)

# Sử dụng thời gian đã chọn hoặc đã thay đổi trong session_state
start_time = st.session_state['start_time']
end_time = st.session_state['end_time']

# Hiển thị thời gian đã chọn
st.write(f"Start Time: {start_time}")
st.write(f"End Time: {end_time}")

# Checkbox để chọn tất cả symbols
select_all = st.sidebar.checkbox("Select All Symbols", value=False)

# Chọn symbols từ danh sách có sẵn
if select_all:
    selected_symbols = st.sidebar.multiselect("Select Symbols", available_symbols_local, default=available_symbols_local)
else:
    selected_symbols = st.sidebar.multiselect("Select Symbols", available_symbols_local)

# Checkbox để chọn gainer hoặc loser
gain_or_lose = st.sidebar.radio("Show", ('Gainners', 'Losers'))

# Kiểm tra nếu nút "Calculate" được nhấn
if calculate:
    recover_rates = {}

    # Duyệt qua từng symbol được chọn
    for symbol in selected_symbols:
        # Đọc DataFrame cho symbol
        df = get_df_from_name(symbol)

        # Lọc DataFrame theo thời gian bắt đầu và kết thúc
        df_filtered = filter_and_resample_df(df, "1d", start_time, end_time)

        # Nếu DataFrame sau khi lọc không có dữ liệu, bỏ qua symbol này
        if df_filtered.empty:
            continue

        # Lấy giá Close đầu tiên và cuối cùng
        close_start = df_filtered['Close'].iloc[0]
        close_end = df_filtered['Close'].iloc[-1]

        # Tính toán Recover Rate
        recover_rate = (close_end / close_start) - 1  # Công thức: (giá cuối / giá đầu) - 1
        recover_rates[symbol] = recover_rate

    # Kiểm tra nếu không có symbol nào có dữ liệu
    if not recover_rates:
        st.write("No data available for the selected symbols and time range.")
    else:
        # Sắp xếp recover_rates dựa vào gainer hoặc loser
        sort_ascending = gain_or_lose == 'Losers'  # Nếu chọn "Losers", sắp xếp tăng dần, ngược lại giảm dần
        sorted_recover_rates = sorted(recover_rates.items(), key=lambda x: x[1], reverse=not sort_ascending)

        # Chuyển kết quả sang DataFrame để hiển thị dưới dạng bảng
        df_results = pd.DataFrame(sorted_recover_rates, columns=['Symbol', 'Recover Rate'])

        # Hiển thị bảng có thể tương tác như Excel với kích thước tuỳ chỉnh
        st.dataframe(df_results.style.format({'Recover Rate': '{:.2%}'}), height=800, width=400)
