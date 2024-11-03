import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import csv_folder_path

from interact_dataframe import get_df_from_name, filter_and_resample_df
from interact_dataframe import extract_symbols_from_local_path
from interact_streamlit import create_sidebar_for_userinputs
from main.caculate_gainners import price_diff_statistic_and_plot, percent_diff_statistic_and_plot

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

# Phần 1: Chuẩn bị giao diện *************************************************************
# Lấy danh sách symbol từ local_folder
available_symbols_local = extract_symbols_from_local_path(csv_folder_path)

# Bắt đầu giao diện Streamlit
st.title("Gainner/Loser Calculator")

# Checkbox để chọn tất cả symbols
select_all = st.sidebar.checkbox("Select All Symbols", value=False)
#timeframe = st.sidebar.selectbox("Select Timeframe", ["1H", "4H", "1D", "3D", "1W"])

# Chọn symbols từ danh sách có sẵn
if select_all:
    selected_symbols = st.sidebar.multiselect("Select Symbols", available_symbols_local, default=available_symbols_local)
else:
    selected_symbols = st.sidebar.multiselect("Select Symbols", available_symbols_local)

# Checkbox để chọn gainer hoặc loser
gain_or_lose = st.sidebar.radio("Show", ('Gainners', 'Losers'))

# Nút "+1" để tăng start_time và end_time thêm 1 ngày
if st.sidebar.button("+1 Day"):
    st.session_state['start_time'] += timedelta(days=1)
    st.session_state['end_time'] += timedelta(days=1)

# Đọc trạng thái start_time và end_time từ session_state, nếu k có thì gán datetime.now()
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now()
if 'end_time' not in st.session_state:
    st.session_state['end_time'] = datetime.now()

# Sử dụng thời gian đã chọn hoặc đã thay đổi trong session_state
start_time = st.session_state['start_time']
end_time = st.session_state['end_time']

# Hiển thị thời gian đã chọn
st.write(f"Start Time: {start_time}")
st.write(f"End Time: {end_time}")

# Phần chọn start_time và end_time thủ công
# vì đặt ở sidebar không đủ chỗ nên đặt ở bên st
st.header("Select Time Range")
# st.session_state['start_time'] = st.sidebar.date_input("Select Start Date", value=st.session_state['start_time'])
# st.session_state['end_time'] = st.sidebar.date_input("Select End Date", value=st.session_state['end_time'])
st.session_state['start_time'] = st.date_input("Select Start Date", value=st.session_state['start_time'])
st.session_state['end_time'] = st.date_input("Select End Date", value=st.session_state['end_time'])

# Phần 2: Các button chức năng bên sidebar
# Nút "Calculate" ở phía trên
calculate = st.sidebar.button("Calculate Gainers/Losers")
# Thêm nút mới
statistic_button = st.sidebar.button("Hourly Difference Statistics")
# Thêm nút mới để hiển thị % chênh lệch giữa 2 giá close liên tiếp
# statistic_button_percentage = st.sidebar.button("Hourly Difference Statistics Percent ")

# Thêm nút mới để tính toán giá trị chênh lệch giữa giá max và giá min
statistic_button_max_min = st.sidebar.button("Max-min Hourly Difference Statistics")
# Thêm nút mới để hiển thị % chênh lệch giữa giá max và giá min
# statistic_button_percentage_maxmin = st.sidebar.button("Max-min Hourly Difference Statistics Percent ")

# Phần 3: Chức năng các button
# Kiểm tra nếu nút "Calculate Gainers/Losers" được nhấn (không cần timeframe)
if calculate:
    recover_rates = {}

    # Duyệt qua từng symbol được chọn
    for symbol in selected_symbols:
        # Đọc DataFrame cho symbol
        df = get_df_from_name(symbol)

        # Lọc DataFrame theo thời gian bắt đầu và kết thúc
        df_filtered = filter_and_resample_df(df, "1d", start_time, end_time)
        # df_filtered = filter_and_resample_df(df, timeframe, start_time, end_time)

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

# # Kiểm tra nếu nút "Hourly Difference Statistics" được nhấn
if statistic_button:
    # dùng hàm Price_diff để tính thay đổi giá trị
    price_diff_statistic_and_plot(
        selected_symbols, 
        start_time, 
        end_time, 
        lambda df: df['Close'].diff()
    )
    # thêm vào
    percent_diff_statistic_and_plot(
        selected_symbols, 
        start_time, 
        end_time, 
        lambda df: df['Close'].pct_change() * 100  # Tính % thay đổi cho giá đóng cửa
    )

# # Kiểm tra nếu nút "Max-min Hourly Difference Statistics" được nhấn
if statistic_button_max_min:
    # dùng hàm Price_diff để tính thay đổi giá trị
    price_diff_statistic_and_plot(
        selected_symbols, 
        start_time, 
        end_time, 
        lambda df: df['High'] - df['Low']
    )
    # thêm vào
    percent_diff_statistic_and_plot(
        selected_symbols, 
        start_time, 
        end_time, 
        lambda df: (df['High'] - df['Low']) / df['Low'] * 100  # Tính % thay đổi
    )

# if statistic_button_percentage:
#     # dùng hàm Percent_diff để tính thay đổi phần trăm
#     percent_diff_statistic_and_plot(
#         selected_symbols, 
#         start_time, 
#         end_time, 
#         lambda df: df['Close'].pct_change() * 100  # Tính % thay đổi cho giá đóng cửa
#     )

# # # Kiểm tra nếu nút "Percentage Change Statistics" được nhấn
# if statistic_button_percentage_maxmin:
#     # dùng hàm Percent_diff để tính thay đổi phần trăm
#     percent_diff_statistic_and_plot(
#         selected_symbols, 
#         start_time, 
#         end_time, 
#         lambda df: (df['High'] - df['Low']) / df['Low'] * 100  # Tính % thay đổi
#     )

# # Thêm nút mới cho tính toán mức độ biến động vol/price
# calculate_vol_price_volatility = st.sidebar.button("Calculate Vol/Price Volatility")

# # Xử lý khi nút "Calculate Vol/Price Volatility" được nhấn
# if calculate_vol_price_volatility:
#     vol_price_volatilities = {}
#     vol_price_pct_changes = {}

#     # Duyệt qua từng symbol được chọn
#     for symbol in selected_symbols:
#         # Đọc DataFrame cho symbol
#         df = get_df_from_name(symbol)

#         # Lọc DataFrame theo thời gian bắt đầu và kết thúc
#         df_filtered = filter_and_resample_df(df, "1d", start_time, end_time)

#         # Nếu DataFrame sau khi lọc không có dữ liệu, bỏ qua symbol này
#         if df_filtered.empty:
#             continue

#         # Cách tính 1: Tính toán vol/price cho từng ngày
#         # df_filtered['Vol/Price'] = df_filtered['Volume'] / df_filtered['Close']
#         # Tính mức độ biến động của vol/price (sử dụng độ lệch chuẩn)
#         # volatility = df_filtered['Vol/Price'].std()  # Hoặc sử dụng phương pháp khác như phần trăm thay đổi trung bình
#         # vol_price_volatilities[symbol] = volatility

#         # Cách tính 2: Tính toán vol/price cho từng ngày
#         df_filtered['Vol/Price'] = df_filtered['Volume'] / df_filtered['Close']
#         # Tính phần trăm thay đổi của vol/price
#         df_filtered['Vol/Price % Change'] = df_filtered['Vol/Price'].pct_change() * 100
#         # Tính phần trăm thay đổi trung bình (bỏ qua giá trị NaN đầu tiên)
#         avg_pct_change = df_filtered['Vol/Price % Change'].iloc[1:].mean()  # Bỏ qua giá trị đầu tiên bị NaN
#         vol_price_pct_changes[symbol] = avg_pct_change


#     # # Kiểm tra nếu không có symbol nào có dữ liệu
#     # if not vol_price_volatilities:
#     #     st.write("No data available for the selected symbols and time range.")
#     # else:
#     #     # Sắp xếp vol_price_volatilities theo mức độ biến động (giảm dần)
#     #     sorted_vol_price_volatilities = sorted(vol_price_volatilities.items(), key=lambda x: x[1], reverse=True)

#     #     # Chuyển kết quả sang DataFrame để hiển thị dưới dạng bảng
#     #     df_results = pd.DataFrame(sorted_vol_price_volatilities, columns=['Symbol', 'Vol/Price Volatility'])

#     #     # Hiển thị bảng có thể tương tác như Excel với kích thước tuỳ chỉnh
#     #     st.dataframe(df_results.style.format({'Vol/Price Volatility': '{:.4f}'}), height=800, width=400)

#     # Kiểm tra nếu không có symbol nào có dữ liệu
#     if not vol_price_pct_changes:
#         st.write("No data available for the selected symbols and time range.")
#     else:
#         # Sắp xếp vol_price_pct_changes theo phần trăm thay đổi trung bình (giảm dần)
#         sorted_vol_price_pct_changes = sorted(vol_price_pct_changes.items(), key=lambda x: x[1], reverse=True)

#         # Chuyển kết quả sang DataFrame để hiển thị dưới dạng bảng
#         df_results = pd.DataFrame(sorted_vol_price_pct_changes, columns=['Symbol', 'Avg Vol/Price % Change'])

#         # Hiển thị bảng có thể tương tác như Excel với kích thước tuỳ chỉnh
#         st.dataframe(df_results.style.format({'Avg Vol/Price % Change': '{:.2f}%'}), height=800, width=400)


# # Thêm nút vẽ biểu đồ
# plot_vol_price_charts = st.sidebar.button("Plot Vol/Price % Change Charts")

# # Xử lý khi nút "Plot Vol/Price % Change Charts" được nhấn
# if plot_vol_price_charts:
#     for symbol in selected_symbols:
#         # Đọc DataFrame cho symbol
#         df = get_df_from_name(symbol)

#         # Lọc DataFrame theo thời gian bắt đầu và kết thúc
#         df_filtered = filter_and_resample_df(df, "1d", start_time, end_time)

#         if df_filtered.empty:
#             st.write(f"No data available for {symbol} in the selected time range.")
#             continue

#         # Tính toán vol/price và phần trăm thay đổi của nó
#         df_filtered['Vol/Price'] = df_filtered['Volume'] / df_filtered['Close']
#         df_filtered['Vol/Price % Change'] = df_filtered['Vol/Price'].pct_change() * 100

#         # Kiểm tra dữ liệu sau khi tính toán
#         if 'Vol/Price % Change' not in df_filtered.columns or df_filtered['Vol/Price % Change'].isnull().all():
#             st.write(f"Insufficient data for calculating Vol/Price % Change for {symbol}.")
#             continue

#         # Tạo biểu đồ cho từng symbol
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=df_filtered.index, 
#             y=df_filtered['Vol/Price % Change'], 
#             mode='lines+markers',
#             name=symbol
#         ))

#         fig.update_layout(
#             title=f"Vol/Price % Change for {symbol}",
#             xaxis_title="Date",
#             yaxis_title="Vol/Price % Change (%)",
#             template="plotly_white"
#         )

#         # Hiển thị biểu đồ trong Streamlit
#         st.plotly_chart(fig)

import plotly.graph_objs as go
from scipy.stats import linregress

# Tạo selectbox cho phương pháp phân tích
analysis_method = st.sidebar.selectbox(
    "Select Analysis Method",
    ["Average % Change", "Standard Deviation", "Percentiles", "Linear Trend", "Days Exceeding Threshold", "Cumulative Sum"]
)

# Tạo nút tính toán dựa vào phương pháp phân tích đã chọn
calculate_vol_price_volatility = st.sidebar.button("Calculate Vol/Price Volatility")

# Tích hợp phân tích vào nút "Calculate Vol/Price Volatility"
if calculate_vol_price_volatility:
    results = {}

    for symbol in selected_symbols:
        df = get_df_from_name(symbol)
        df_filtered = filter_and_resample_df(df, "1d", start_time, end_time)

        if df_filtered.empty:
            st.write(f"No data available for {symbol} in the selected time range.")
            continue

        # Tính toán tỷ lệ Vol/Price cho từng ngày
        df_filtered['Vol/Price'] = df_filtered['Volume'] / df_filtered['Close']

        # Thực hiện phương pháp phân tích được chọn
        if analysis_method == "Average % Change":
            df_filtered['Vol/Price % Change'] = df_filtered['Vol/Price'].pct_change() * 100
            results[symbol] = df_filtered['Vol/Price % Change'].iloc[1:].mean()

        elif analysis_method == "Standard Deviation":
            df_filtered['Vol/Price % Change'] = df_filtered['Vol/Price'].pct_change() * 100
            results[symbol] = df_filtered['Vol/Price % Change'].std()

        elif analysis_method == "Percentiles":
            q25 = df_filtered['Vol/Price'].quantile(0.25)
            q50 = df_filtered['Vol/Price'].median()
            q75 = df_filtered['Vol/Price'].quantile(0.75)
            results[symbol] = (q25, q50, q75)

        elif analysis_method == "Linear Trend":
            slope, intercept, r_value, p_value, std_err = linregress(range(len(df_filtered)), df_filtered['Vol/Price'].fillna(0))
            results[symbol] = slope

        elif analysis_method == "Days Exceeding Threshold":
            df_filtered['Vol/Price % Change'] = df_filtered['Vol/Price'].pct_change() * 100
            threshold = 10  # Điều chỉnh ngưỡng tại đây
            results[symbol] = (df_filtered['Vol/Price % Change'].abs() > threshold).sum()

        # elif analysis_method == "Z-score":
        #     mean = df_filtered['Vol/Price % Change'].mean()
        #     std = df_filtered['Vol/Price % Change'].std()

        #     # Chỉ tính Z-score nếu độ lệch chuẩn khác 0
        #     if std != 0:
        #         df_filtered['Z-score'] = (df_filtered['Vol/Price % Change'] - mean) / std
        #     else:
        #         st.write(f"Standard deviation is zero for {symbol}; Z-score calculation not possible.")

        #     # df_filtered['Z-score'] = (df_filtered['Vol/Price % Change'] - mean) / std
        #     # results[symbol] = df_filtered['Z-score'].abs().mean()

        elif analysis_method == "Cumulative Sum":
            df_filtered['Vol/Price % Change'] = df_filtered['Vol/Price'].pct_change() * 100
            results[symbol] = df_filtered['Vol/Price % Change'].cumsum().iloc[-1]

    # Hiển thị kết quả
    df_results = pd.DataFrame(results.items(), columns=['Symbol', analysis_method])
    st.dataframe(df_results)

# Tích hợp vẽ biểu đồ theo phương pháp phân tích đã chọn vào nút "Plot Vol/Price % Change Charts"
plot_vol_price_charts = st.sidebar.button("Plot Vol/Price % Change Charts")

if plot_vol_price_charts:
    for symbol in selected_symbols:
        df = get_df_from_name(symbol)
        df_filtered = filter_and_resample_df(df, "1d", start_time, end_time)

        if df_filtered.empty:
            st.write(f"No data available for {symbol} in the selected time range.")
            continue

        # Tính toán vol/price và phần trăm thay đổi
        df_filtered['Vol/Price'] = df_filtered['Volume'] / df_filtered['Close']
        df_filtered['Vol/Price % Change'] = df_filtered['Vol/Price'].pct_change() * 100

        # Tạo biểu đồ cho từng symbol với nội dung tùy thuộc vào phương pháp phân tích
        fig = go.Figure()

        if analysis_method == "Average % Change":
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Vol/Price % Change'], mode='lines+markers', name=symbol))

        elif analysis_method == "Standard Deviation":
            fig.add_trace(go.Scatter(x=df_filtered.index, y=[df_filtered['Vol/Price % Change'].std()] * len(df_filtered), mode='lines', name=symbol))

        elif analysis_method == "Percentiles":
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Vol/Price'], mode='lines+markers', name=symbol))

        elif analysis_method == "Linear Trend":
            slope, intercept, *_ = linregress(range(len(df_filtered)), df_filtered['Vol/Price'].fillna(0))
            trendline = intercept + slope * np.arange(len(df_filtered))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=trendline, mode='lines', name=symbol))

        elif analysis_method == "Days Exceeding Threshold":
            threshold = 10
            exceed_days = df_filtered['Vol/Price % Change'].apply(lambda x: x if abs(x) > threshold else None)
            fig.add_trace(go.Scatter(x=df_filtered.index, y=exceed_days, mode='markers', name=symbol))

        # elif analysis_method == "Z-score":
        #     mean = df_filtered['Vol/Price % Change'].mean()
        #     std = df_filtered['Vol/Price % Change'].std()

            # # Chỉ tính Z-score nếu độ lệch chuẩn khác 0
            # if std != 0:
            #     df_filtered['Z-score'] = (df_filtered['Vol/Price % Change'] - mean) / std
            # else:
            #     st.write(f"Standard deviation is zero for {symbol}; Z-score calculation not possible.")
                
            # df_filtered['Z-score'] = (df_filtered['Vol/Price % Change'] - mean) / std
            # fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Z-score'], mode='lines+markers', name=symbol))

        elif analysis_method == "Cumulative Sum":
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Vol/Price % Change'].cumsum(), mode='lines+markers', name=symbol))

        fig.update_layout(title=f"{analysis_method} for {symbol}", xaxis_title="Date", yaxis_title=analysis_method, template="plotly_white")
        st.plotly_chart(fig)



#readme.md
# HƯỚNG DẪN DÙNG CÁC CHỨC NĂNG CHÍNH
# bonus Button +1 để tăng window lên 1 đơn vị (có thể tuỳ chỉnh +7 thay vì +1 ở trong code (line 60))
# 1. Caculater Gainers/Loss: tái tạo danh sách gainners/lossers theo khoảng thời gian tuỳ chọn
# 2. Difference Statistic:
# 2.1 Theo df['Close'].diff()      và  df['Close'].pct_change() * 100
# 2.2 Theo df['High'] - df['Low']  và  (df['High'] - df['Low']) / df['Low'] * 100
# biểu đồ line 24h và biểu đồ heatmap 7*24
