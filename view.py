import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from interact_dataframe import get_df_from_name, showinfo_dtypes_head_tail, filter_and_resample_df
from interact_dataframe import color_schemes, extract_symbols_from_local_path
from interact_streamlit import create_sidebar_for_userinputs, add_trace_candlestick, update_yaxis_layout
from interact_drive import extract_symbols_from_drive, get_df_from_name_gdrive, authorize_and_create_drive
st.set_page_config(layout="wide")
from config.config import csv_folder_path
from interact_binanceApi import get_binance_data_to_gdrive, get_binance_data

#region Phần chọn [1/2 of 2]: Sử dụng Local
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
# selected_symbols = st.sidebar.multiselect("Select Symbols from Local", available_symbols_local, default=['BTC', 'ETH'])
# # Đọc file df từ các lựa chọn local
# dfs = {symbol: get_df_from_name(symbol) for symbol in selected_symbols}
#endregion

#region Phận chọn [2/2 of 2]: Sử dụng Google Drive
st.sidebar.header("Google Drive Data Operations")
# Thêm một ô nhập symbol cho việc tải xuống Google Drive
input_symbol_drive = st.sidebar.text_input("Enter Symbol for Google Drive", value="")
# Nút để tải csv xuống gDrive
if st.sidebar.button("downgDrive"):
    if input_symbol_drive:
        # Sử dụng st.spinner để hiển thị vòng xoay khi đang tải
        with st.spinner(f"Downloading {input_symbol_drive} to Google Drive..."):
            get_binance_data_to_gdrive(input_symbol_drive, 365)
        st.success(f"Download completed for {input_symbol_drive}!")
    else:
        st.write("Please enter a symbol to download.")
# Thêm lựa chọn nhiều symbol vào sidebar
available_symbols_drive = extract_symbols_from_drive()
selected_symbols = st.sidebar.multiselect("Select Symbols from Google Drive", available_symbols_drive, default=['BTC', 'ETH'])
# Bước 2: Đọc file df từ các lựa chọn Google Drive
drive = authorize_and_create_drive()
dfs = {symbol: get_df_from_name_gdrive(drive, symbol) for symbol in selected_symbols}
#endregion

# # Bước 0: Tải chart về Local hoặc gDrive----------------------------------------------------------
# # Thêm một ô nhập symbol
# input_symbol = st.sidebar.text_input("Enter Symbol", value="")  # Ô nhập symbol

# # Nút để tải csv xuống gDrive
# if st.sidebar.button("downgDrive"):
#     if input_symbol:
#         # Sử dụng st.spinner để hiển thị vòng xoay khi đang tải
#         with st.spinner(f"Downloading {input_symbol} to Google Drive..."):
#             get_binance_data_to_gdrive(input_symbol, 365)
#         st.success(f"Download completed for {input_symbol}!")
#     else:
#         st.write("Please enter a symbol to download.")

# # Nút để tải csv xuống Local
# if st.sidebar.button("downLocal"):
#     if input_symbol:
#         # Sử dụng st.spinner để hiển thị vòng xoay khi đang tải
#         with st.spinner(f"Downloading {input_symbol} to Local..."):
#             get_binance_data(input_symbol, 365)
#         st.success(f"Download completed for {input_symbol}!")
#     else:
#         st.write("Please enter a symbol to download.")

#region Chọn [1/2trong2] Dùng local
# Bước 1: Thêm lựa chọn nhiều symbol vào sidebar
# available_symbols = ['BTC', 'ETH', 'SOL', 'BNB']
# available_symbols = extract_symbols_from_local_path(csv_folder_path)
# selected_symbols = st.sidebar.multiselect("Select Symbols", available_symbols, default=['BTC', 'ETH'])
# Bước 2: Đọc file df từ các lựa chọn
# symbols = selected_symbols
# dfs = {symbol: get_df_from_name(symbol) for symbol in symbols}
#endregion

# #region Chọn [2/2trong2] Dùng gDrive--------------------------------------------------------------
# # Bước 1: Thêm lựa chọn nhiều symbol vào sidebar
# available_symbols = extract_symbols_from_drive()
# selected_symbols = st.sidebar.multiselect("Select Symbols", available_symbols, default=['BTC', 'ETH'])
# # Bước 2: Đọc file df từ các lựa chọn
# drive = authorize_and_create_drive()
# dfs = {symbol: get_df_from_name_gdrive(drive, symbol) for symbol in selected_symbols}
# #endregion

# Bước 3: Lọc dataframe theo các tiêu chí--------------------------------------------------------
timeframe, start_time, end_time = create_sidebar_for_userinputs()
dfs_filtered = {symbol: filter_and_resample_df(dfs[symbol], timeframe, start_time, end_time) for symbol in selected_symbols}

# Bước 4: Create the overlay chart--------------------------------------------------------------
fig = go.Figure()

# 4.1 Tạo trace cho từng datafram
for i, symbol in enumerate(selected_symbols):
    yaxis = f'y{i+1}'  # Đặt tên cho yaxis tương ứng, ví dụ: y1, y2, y3, ...
    color_scheme = color_schemes[str(i+1)]  # Lấy màu sắc từ từ điển dựa trên thứ tự i+1
    add_trace_candlestick(fig, dfs_filtered[symbol], f"Crypto {i+1}", yaxis=yaxis, color_scheme=color_scheme)

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