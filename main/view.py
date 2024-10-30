import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from interact_binanceApi import get_binance_data_to_gdrive, get_binance_data
from interact_dataframe import get_df_from_name, showinfo_dtypes_head_tail, filter_and_resample_df
from interact_dataframe import color_schemes, extract_symbols_from_local_path
from interact_streamlit import add_trace_candlestick, add_trace_line, update_yaxis_layout
from interact_streamlit import create_sidebar_for_userinputs, update_symbol_progress
from interact_drive import extract_symbols_from_drive, get_df_from_name_gdrive, authorize_and_create_drive
from interact_data import read_csv_to_dict_pandas, list_all_tags, get_symbols_with_tag
st.set_page_config(layout="wide")
from config.config import csv_folder_path


#region Phần chọn [1/3 of 3]: Sử dụng Local
# st.sidebar.header("Local Data Operations")
# # Thêm một ô nhập symbol cho việc tải xuống local
# input_symbol_local = st.sidebar.text_input("Enter Symbol for Local", value="")
# # Nút để tải csv xuống Local
# if st.sidebar.button("downLocal"):
#     if input_symbol_local:
#         # Sử dụng st.spinner để hiển thị vòng xoay khi đang tải
#         with st.spinner(f"Downloading {input_symbol_local} to Local..."):
#             get_binance_data(input_symbol_local, 365)
#         st.success(f"Download completed for {input_symbol_local}!")
#     else:
#         st.write("Please enter a symbol to download.")
# # Thêm lựa chọn nhiều symbol vào sidebar
# # available_symbols = ['BTC', 'ETH', 'SOL', 'BNB']
# available_symbols_local = extract_symbols_from_local_path(csv_folder_path)
# selected_symbols = st.sidebar.multiselect("Select Symbols from Local", available_symbols_local, default=['BTC'])
# # Đọc file df từ các lựa chọn local
# dfs = {symbol: get_df_from_name(symbol) for symbol in selected_symbols}
#endregion

#region Phần chọn [2/3 of 3]: Sử dụng Google Drive
# st.sidebar.header("Google Drive Data Operations")
# # Thêm một ô nhập symbol cho việc tải xuống Google Drive
# input_symbol_drive = st.sidebar.text_input("Enter Symbol for Google Drive", value="")
# # Nút để tải csv xuống gDrive
# if st.sidebar.button("downgDrive"):
#     if input_symbol_drive:
#         # Sử dụng st.spinner để hiển thị vòng xoay khi đang tải
#         with st.spinner(f"Downloading {input_symbol_drive} to Google Drive..."):
#             get_binance_data_to_gdrive(input_symbol_drive, 365)
#         st.success(f"Download completed for {input_symbol_drive}!")
#     else:
#         st.write("Please enter a symbol to download.")
# # Thêm lựa chọn nhiều symbol vào sidebar
# available_symbols_drive = extract_symbols_from_drive()
# selected_symbols = st.sidebar.multiselect("Select Symbols from Google Drive", available_symbols_drive, default=['BTC', 'ETH'])
# # Bước 2: Đọc file df từ các lựa chọn Google Drive
# drive = authorize_and_create_drive()
# dfs = {symbol: get_df_from_name_gdrive(drive, symbol) for symbol in selected_symbols}
#endregion

#region Phần chọn [3/3 of 3]: Thêm tính năng symbol-by-tags
st.sidebar.header("Local Data Operations")

# Đọc toàn bộ tags từ CSV và tạo symbol_tags
csv_path = '.\\data\\symbols_with_tags.csv'
symbol_tags = read_csv_to_dict_pandas(csv_path)

# Thêm một ô nhập symbol cho việc tải xuống local
input_symbol_local = st.sidebar.text_input("Enter Symbol for Local", value="")
# Nút để tải csv xuống Local
if st.sidebar.button("downLocal"):
    if input_symbol_local:
        # Sử dụng st.spinner để hiển thị vòng xoay khi đang tải
        with st.spinner(f"Downloading {input_symbol_local} to Local..."):
            get_binance_data(input_symbol_local, 365)
        st.success(f"Download completed for {input_symbol_local}!")
    else:
        st.write("Please enter a symbol to download.")

if st.sidebar.button("Update Symbols"):
    # Lấy danh sách các symbols có sẵn
    available_symbols_local = extract_symbols_from_local_path(csv_folder_path)

    if available_symbols_local:
        update_symbol_progress(available_symbols_local)  # Gọi hàm cập nhật với tiến độ
    else:
        st.sidebar.write("No symbols found to update.")


# Thêm lựa chọn nhiều symbol từ local
# available_symbols = ['BTC', 'ETH', 'SOL', 'BNB']
available_symbols_local = extract_symbols_from_local_path(csv_folder_path)
selected_symbols = st.sidebar.multiselect("Select Symbols from Local", available_symbols_local, default=['BTC'])
# Đọc file df từ các lựa chọn local
dfs = {symbol: get_df_from_name(symbol) for symbol in selected_symbols}

st.sidebar.header("Add Symbols by Tag")
# Lựa chọn sàn giao dịch (Binance, Bybit, v.v.)
selected_exchange = st.sidebar.selectbox("Select Exchange", options=["binance", "bybit"])

# Liệt kê tất cả các tags của sàn đã chọn
available_tags = list_all_tags(symbol_tags, selected_exchange)
selected_tag = st.sidebar.selectbox("Select Tag", available_tags)

# Nút để thêm symbols theo tag vào dfs
if st.sidebar.button("Download Symbols by Tag"):
    # Lấy các symbols có tag đã chọn cho exchange đã chọn
    symbols_with_tag = get_symbols_with_tag(symbol_tags, selected_exchange, selected_tag)
    
    # In ra danh sách symbols tìm thấy
    st.write(f"Found symbols with tag '{selected_tag}' in '{selected_exchange}': {symbols_with_tag}")
    
    # Thêm các symbols đó vào dfs
    for symbol in symbols_with_tag:
        if symbol not in dfs:  # Đảm bảo không thêm trùng lặp
            if symbol not in available_symbols_local:  # Kiểm tra xem symbol có trong dữ liệu local không
                with st.spinner(f"Downloading {symbol} to Local..."):
                    get_binance_data(symbol, 365)  # Tải dữ liệu về nếu chưa có
                st.success(f"Download completed for {symbol}!")

                # Cập nhật danh sách symbols local sau khi tải về
                available_symbols_local = extract_symbols_from_local_path(csv_folder_path)
            # Sau khi chắc chắn đã tải về hoặc có sẵn trong local, thêm vào dfs
            dfs[symbol] = get_df_from_name(symbol)
            st.success(f"Added {symbol} to data frames.")          

#endregion

#region the rest
# Bước 3: Lọc dataframe theo các tiêu chí--------------------------------------------------------
timeframe, start_time, end_time = create_sidebar_for_userinputs()
dfs_filtered = {symbol: filter_and_resample_df(dfs[symbol], timeframe, start_time, end_time) for symbol in selected_symbols}

# Bước 4: Create the overlay chart--------------------------------------------------------------
fig = go.Figure()

# 4.1 Tạo trace cho từng datafram
#bonus Thêm lựa chọn loại biểu đồ vào sidebar
chart_type = st.sidebar.radio("Select Chart Type", ('Candlestick', 'Line'))

for i, symbol in enumerate(selected_symbols):
    yaxis = f'y{i+1}'  # Đặt tên cho yaxis tương ứng, ví dụ: y1, y2, y3, ...
    color_scheme = color_schemes[str(i+1)]  # Lấy màu sắc từ từ điển dựa trên thứ tự i+1
    if chart_type == 'Candlestick':
        add_trace_candlestick(fig, dfs_filtered[symbol], f"Crypto {symbol}", yaxis=yaxis, color_scheme=color_scheme)
    elif chart_type == 'Line':
        add_trace_line(fig, dfs_filtered[symbol], f"Crypto {symbol}", yaxis=yaxis, color_scheme=color_scheme)

# 4.2 Layout settings
fig.update_layout(title="Overlay of Multi Crypto Charts",
                xaxis_title="Time",
                yaxis_title="Price",
                width=2000,   # Chiều rộng (pixels)
                height=1000,   # Chiều cao (pixels)
                # autosize=True,  # Tự động điều chỉnh kích thước
                # margin=dict(l=10, r=10, t=30, b=10),
                xaxis_rangeslider_visible=False, # Không hiện cái thanh ở dưới
                dragmode='zoom',  # Kích hoạt chế độ kéo để phóng to
                hovermode='x unified' # Hiển thị giá trị hover cho cả trục X
                )

# 4.3 Cập nhật layout cho từng trục Y để không bị chồng lên nhau
for i in range(len(selected_symbols)):
    yaxis = f'yaxis{i+2}'  # Tên trục yaxis1, yaxis2, yaxis3, ...
    update_yaxis_layout(fig, yaxis)

# 4.4 Hiển thị lên Streamlit
st.plotly_chart(fig, use_container_width=True)

# Bước 5: Tính toán chỉ số Recover Rate cho từng symbol--------------------------------------------
recover_rates = {}
for symbol in selected_symbols:
    # Lấy dataframe đã lọc cho symbol hiện tại
    df_filtered = dfs_filtered[symbol]

    # Lấy giá close đầu tiên và cuối cùng trong khoảng thời gian đã chọn
    close_start = df_filtered['Close'].iloc[0]
    close_end = df_filtered['Close'].iloc[-1]

    # Tính toán Recover Rate
    recover_rate = (close_end / close_start) - 1  # Công thức: (giá cuối / giá đầu) - 1
    recover_rates[symbol] = recover_rate

# Hiển thị Recover Rate trên Streamlit
st.sidebar.header("Recover Rate")
sort_ascending = st.sidebar.checkbox("Sort")

# Sắp xếp recover_rates theo giá trị từ cao đến thấp hoặc thấp đến cao dựa vào checkbox
sorted_recover_rates = sorted(recover_rates.items(), key=lambda x: x[1], reverse=not sort_ascending)
for symbol, rate in sorted_recover_rates:
    st.sidebar.write(f"{symbol}: {rate:.2%}")
#endregion the rest

#readme.md
# HƯỚNG DẪN DÙNG CÁC CHỨC NĂNG CHÍNH
# 1. Tải về klines
# 1.1 downLocal: nhập 1 symbol và tải về Local
# 1.2 Update Symbols: cập nhật toàn bộ symbols trong thư mục Local
# 1.3 Download Symbols by Tags: tải 1 nhóm symbol về Local [Lưu ý: cần phải refresh để sử dụng]
# 2. Tác dụng chính của view.py:
# 2.1 Overlay các chart lên nhau để so sánh độ mạnh yếu: [a. trong nội bộ nhóm] và [b. giữa các con trưởng nhóm với nhau]
# 2.2 kèm theo chỉ số Recover Rate để có số liệu chính xác