import streamlit as st
import pandas as pd
from interact_dataframe import get_df_from_name, filter_and_resample_df, resample_df, compare_with_dtw
from interact_dataframe import normalize_min_max, find_best_match, find_worst_match, create_overlapping_windows
from interact_dataframe import create_sliding_windows, create_variable_windows, create_pyramid_windows, create_paa_segments
from interact_dataframe import extract_symbols_from_local_path
import plotly.graph_objects as go

from config.config import csv_folder_path
st.set_page_config(layout="wide")

def create_sidebar_for_userinputs():
    st.sidebar.title("Chart Options")
    timeframe = st.sidebar.selectbox("Select Timeframe", ["1H", "4H", "1D", "3D", "1W"])
    start_time = st.sidebar.date_input("From Date")
    end_time = st.sidebar.date_input("To Date")

    # Lấy danh sách symbol từ local_folder
    available_symbols_local = extract_symbols_from_local_path(csv_folder_path)
    # Sử dụng selectbox thay cho multiselect để chỉ chọn 1 symbol
    # selected_symbol = st.sidebar.selectbox("Select Symbol", available_symbols_local)

    # Selectbox để chọn symbol chính (biểu đồ chính)
    # main_symbol = st.sidebar.selectbox("Select Main Symbol", ['ETH', 'BTC', 'AVAX', 'AAVE'])
    main_symbol = st.sidebar.selectbox("Select Main Symbol", available_symbols_local)
    
    # Dropdown để chọn cách tạo sliding window
    window_type = st.sidebar.selectbox("Select Window Type", 
                                       ["Sliding Windows", "Overlapping Windows", "Pyramid Windows", "Variable Windows ***", "PAA Segments ***"])

    # Thêm multiselect để chọn nhiều symbol
    # symbols = st.sidebar.multiselect("Select Symbols for Comparison", ['ETH', 'BTC', 'AVAX', 'AAVE'])
    symbols = st.sidebar.multiselect("Select Symbols for Comparison", available_symbols_local)

    return timeframe, start_time, end_time, main_symbol, window_type, symbols

def main():
    st.title("Crypto Chart Analyzer with DTW")

    # Bước lấy dữ liệu từ người dùng
    timeframe, start_time, end_time, main_symbol, window_type, symbols = create_sidebar_for_userinputs()

    # Chuyển đổi date input của Streamlit sang string để sử dụng trong phân tích
    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

    # Nút phân tích
    if st.button("Analyze"):
        # Load data cho symbol chính dựa trên lựa chọn của người dùng
        df_main_symbol = get_df_from_name(main_symbol)

        # Xử lý dữ liệu cho symbol chính
        df_part = filter_and_resample_df(df_main_symbol, timeframe, start_time, end_time)
        df_main_symbol_resampled = resample_df(df_main_symbol, timeframe)

        # Chuẩn hóa dữ liệu
        df_main_symbol_resampled_norm = normalize_min_max(df_main_symbol_resampled)
        df_part_norm = normalize_min_max(df_part)

        # Tạo sliding windows dựa trên lựa chọn của người dùng
        if window_type == "Sliding Windows":
            window_size = len(df_part)
            sliding_windows = create_sliding_windows(df_main_symbol_resampled_norm, window_size)
        elif window_type == "Overlapping Windows":
            overlap_size = 5
            window_size = len(df_part)
            sliding_windows = create_overlapping_windows(df_main_symbol_resampled_norm, window_size, overlap_size)
        elif window_type == "Variable Windows":
            min_size = 10
            max_size = len(df_part)
            step_size = 1
            if max_size > min_size:
                sliding_windows = create_variable_windows(df_main_symbol_resampled_norm, min_size, max_size, step_size)
                window_size = len(sliding_windows[0])
            else:
                st.error("Max size phải lớn hơn min size!")
                return
        elif window_type == "Pyramid Windows":
            base_window_size = len(df_part)
            num_levels = 3
            window_size = base_window_size
            sliding_windows = create_pyramid_windows(df_main_symbol_resampled_norm, base_window_size, num_levels)
        elif window_type == "PAA Segments":
            num_segments = 10
            window_size = num_segments
            sliding_windows = create_paa_segments(df_main_symbol_resampled_norm, num_segments)

        # So sánh DTW với từng symbol được chọn
        results = []
        for symbol in symbols:
            df_other_symbol = get_df_from_name(symbol)

            # Resample và chuẩn hóa dữ liệu của các symbol khác
            df_other_symbol_resampled = resample_df(df_other_symbol, timeframe)
            df_other_symbol_resampled_norm = normalize_min_max(df_other_symbol_resampled)

            # Tính toán DTW giữa symbol chính và symbol khác
            dtw_distances = compare_with_dtw(df_part_norm, create_sliding_windows(df_other_symbol_resampled_norm, window_size))
            best_match_window, match_index, min_distance = find_best_match(dtw_distances, create_sliding_windows(df_other_symbol_resampled_norm, window_size))
            worst_match_window, _, max_distance = find_worst_match(dtw_distances, create_sliding_windows(df_other_symbol_resampled_norm, window_size))

            # Lưu kết quả cho symbol
            results.append((symbol, min_distance, max_distance, best_match_window, worst_match_window))

        # Hiển thị kết quả
        st.subheader("Comparison Results")
        result_df = pd.DataFrame(results, columns=['Symbol', 'Min DTW Distance', 'Max DTW Distance', 'Best Match', 'Worst Match'])
        st.write(result_df)

        # Trực quan hóa biểu đồ của từng symbol
        for symbol, min_distance, max_distance, best_match_window, worst_match_window in results:
            st.subheader(f"Best and Worst Match for {symbol}")

            # Vẽ biểu đồ Best Match
            fig_best = go.Figure()
            fig_best.add_trace(go.Scatter(x=best_match_window.index, y=best_match_window['Close'], mode='lines', name="Best Match"))
            st.plotly_chart(fig_best)

            # Vẽ biểu đồ Worst Match
            # fig_worst = go.Figure()
            # fig_worst.add_trace(go.Scatter(x=worst_match_window.index, y=worst_match_window['Close'], mode='lines', name="Worst Match"))
            # st.plotly_chart(fig_worst)

if __name__ == "__main__":
    main()

#readme.md
# HƯỚNG DẪN DÙNG CÁC CHỨC NĂNG CHÍNH
# 1. chọn time, main_symbol, compare_symbols
# 2. Mục đích:
# 2.1 tìm [các cặp] lặp lại khoảng thời gian biến động giống nhau
# 2.2 tìm [khoảng thời gian] có hành động tương tự trong quá khứ

