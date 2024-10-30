import streamlit as st
import pandas as pd
from interact_dataframe import get_df_from_name, filter_and_resample_df, resample_df, compare_with_dtw
from interact_dataframe import normalize_min_max, find_best_match, find_worst_match, create_overlapping_windows
from interact_dataframe import create_sliding_windows, create_variable_windows, create_pyramid_windows, create_paa_segments
import plotly.graph_objects as go

# Sidebar cho các input từ người dùng
def create_sidebar_for_userinputs():
    st.sidebar.title("Chart Options")
    timeframe = st.sidebar.selectbox("Select Timeframe", ["1H", "4H", "1D", "3D", "1W"])
    start_time = st.sidebar.date_input("From Date")
    end_time = st.sidebar.date_input("To Date")
    
    # Dropdown để chọn cách tạo sliding window
    window_type = st.sidebar.selectbox("Select Window Type", 
                                       ["Sliding Windows", "Overlapping Windows", "Pyramid Windows", "Variable Windows ***", "PAA Segments ***"])

    return timeframe, start_time, end_time, window_type

# Hàm vẽ biểu đồ với Streamlit
def visualize_charts(df_part, dtw_distances, best_match_window1, worst_match_window2, dfETH_resampled, match_index1, window_size):
    st.subheader("Candlestick Chart for Selected Range")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_part.index,
                                 open=df_part['Open'], high=df_part['High'],
                                 low=df_part['Low'], close=df_part['Close'],
                                 name="Candlestick"))
    st.plotly_chart(fig)

    st.subheader("DTW Distances")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(dtw_distances))), 
                              y=dtw_distances, mode='lines', name='DTW Distances'))
    st.plotly_chart(fig2)

    st.subheader("Best Match Window (Normalized Data)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=best_match_window1.index, y=best_match_window1['Close'], mode='lines', name="Best Match"))
    st.plotly_chart(fig3)

    st.subheader("Worst Match Window (Normalized Data)")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=worst_match_window2.index, y=worst_match_window2['Close'], mode='lines', name="Worst Match"))
    st.plotly_chart(fig4)

    st.subheader("Best Match Window (Original Data)")
    best_match_window_original = dfETH_resampled.iloc[match_index1:match_index1 + window_size]
    fig5 = go.Figure()
    fig5.add_trace(go.Candlestick(x=best_match_window_original.index,
                                  open=best_match_window_original['Open'], high=best_match_window_original['High'],
                                  low=best_match_window_original['Low'], close=best_match_window_original['Close'],
                                  name="Best Match Original"))
    st.plotly_chart(fig5)

# MAIN FUNCTION
def main():
    st.title("Crypto Chart Analyzer with DTW")

    # Bước lấy dữ liệu từ người dùng
    timeframe, start_time, end_time, window_type = create_sidebar_for_userinputs()

    # Chuyển đổi date input của Streamlit sang string để sử dụng trong phân tích
    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

    # Nút phân tích
    if st.button("Analyze"):
        # Load data (thay thế với dữ liệu của bạn)
        dfETH = get_df_from_name('ETH')

        # Xử lý dữ liệu
        df_part = filter_and_resample_df(dfETH, timeframe, start_time, end_time)
        dfETH_resampled = resample_df(dfETH, timeframe)

        # Chuẩn hóa dữ liệu
        dfETH_resampled_norm = normalize_min_max(dfETH_resampled)
        df_part_norm = normalize_min_max(df_part)

        # Tạo các sliding windows dựa trên lựa chọn của người dùng-----OLD----
        window_size = len(df_part)
        if window_type == "Sliding Windows":
            sliding_windows = create_sliding_windows(dfETH_resampled_norm, window_size)
        elif window_type == "Overlapping Windows":
            sliding_windows = create_overlapping_windows(dfETH_resampled_norm, window_size, overlap_size=5)
        elif window_type == "Variable Windows":
            sliding_windows = create_variable_windows(dfETH_resampled_norm, min_size=10, max_size=len(df_part), step_size=1)
        elif window_type == "Pyramid Windows":
            base_window_size = len(df_part)
            sliding_windows = create_pyramid_windows(dfETH_resampled_norm, base_window_size, num_levels=3)
        elif window_type == "PAA Segments":
            num_segments = 10
            sliding_windows = create_paa_segments(dfETH_resampled_norm, num_segments)
        # Tạo các sliding windows dựa trên lựa chọn của người dùng-----OLD----

        # Tạo các sliding windows dựa trên lựa chọn của người dùng----NEW----
        # if window_type == "Sliding Windows":
        #     # Với Sliding Windows, window_size có thể bằng len(df_part)
        #     window_size = len(df_part)
        #     sliding_windows = create_sliding_windows(dfETH_resampled_norm, window_size)
        # elif window_type == "Overlapping Windows":
        #     # Với Overlapping Windows, window_size có thể giống Sliding Windows
        #     # Thêm overlap_size cho các cửa sổ chồng lấn
        #     overlap_size = 5  # Bạn có thể cho người dùng nhập overlap_size tùy chọn
        #     window_size = len(df_part)
        #     sliding_windows = create_overlapping_windows(dfETH_resampled_norm, window_size, overlap_size)
        # elif window_type == "Variable Windows":
        #     # Với Variable Windows, cần có kích thước biến đổi (min_size và max_size)
        #     min_size = 10  # Kích thước tối thiểu
        #     max_size = len(df_part)  # Kích thước tối đa là len(df_part)
        #     step_size = 1  # Bước nhảy giữa các cửa sổ
        #     if max_size > min_size:  # Đảm bảo điều kiện này đúng
        #         sliding_windows = create_variable_windows(dfETH_resampled_norm, min_size, max_size, step_size)
        #     else:
        #         st.error("Max size phải lớn hơn min size!")
        #         return  # Kết thúc sớm nếu điều kiện không thỏa     
        # elif window_type == "Pyramid Windows":
        #     # Với Pyramid Windows, cần base_window_size
        #     base_window_size = len(df_part)  # Bạn có thể thay đổi thành thông số tùy chọn
        #     num_levels = 3  # Có thể cho người dùng chọn số level
        #     window_size = base_window_size
        #     sliding_windows = create_pyramid_windows(dfETH_resampled_norm, base_window_size, num_levels)
        # elif window_type == "PAA Segments":
        #     # Với PAA, kích thước cửa sổ phụ thuộc vào số segment
        #     num_segments = 10  # Có thể cho người dùng chọn số segment
        #     window_size = num_segments
        #     sliding_windows = create_paa_segments(dfETH_resampled_norm, num_segments)      
        # Tạo các sliding windows dựa trên lựa chọn của người dùng----NEW----

        # Tính toán DTW và tìm best match, worst match
        with st.spinner("Đang tính toán DTW..."):
            dtw_distances = compare_with_dtw(df_part_norm, sliding_windows)
        best_match_window1, match_index1, min_distance = find_best_match(dtw_distances, sliding_windows)
        worst_match_window2, match_index2, max_distance = find_worst_match(dtw_distances, sliding_windows)

        # Trực quan hóa biểu đồ
        visualize_charts(df_part, dtw_distances, best_match_window1, worst_match_window2, dfETH_resampled, match_index1, window_size)

if __name__ == "__main__":
    main()
