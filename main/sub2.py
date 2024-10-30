from interact_dataframe import get_df_from_name, chooseFormattimeColumn, filter_and_resample_df
from interact_dataframe import resample_df, create_sliding_windows, compare_with_dtw, find_best_match
from interact_dataframe import normalize_min_max, normalize_z_score, find_worst_match, compare_with_pmk
from interact_dataframe import create_overlapping_windows, create_variable_windows, create_pyramid_windows
from interact_dataframe import create_paa_segments
import plotly.graph_objects as go
from interact_streamlit import add_trace_candlestick, add_trace_line, update_yaxis_layout
import webbrowser

def process_data(dfETH, start_time, end_time, timeframe):
    # Cắt đoạn cần so sánh và resample
    df_part = filter_and_resample_df(dfETH, timeframe, start_time, end_time)
    print("Data part:")
    print(df_part)

    # Resample dữ liệu gốc theo timeframe đã chọn
    dfETH_resampled = resample_df(dfETH, timeframe)

    # Chuẩn hóa dữ liệu
    dfETH_resampled_norm = normalize_min_max(dfETH_resampled)
    df_part_norm = normalize_min_max(df_part)

    # Tạo các đoạn trượt
    window_size = len(df_part)
    print(f"Length of window (window_size): {window_size}")

    # Chọn 1 trong các cách slicing, xong nhớ để nó vào return
    # Cách 1:---------------------------------------------------------------------
    # sliding_windows = create_sliding_windows(dfETH_resampled_norm, window_size)
    # print(f"Number of sliding windows created: {len(sliding_windows)}")
    # Cách 2:--------------------------------------------------------------------
    overlapping_windows = create_overlapping_windows(dfETH_resampled_norm, window_size, overlap_size=5)
    print(f"Number of overlapping windows created: {len(overlapping_windows)}")
    # Cách 3:--------------------------------------------------------------------   
    # variable_windows = create_variable_windows(dfETH_resampled_norm, min_size=10, max_size=len(df_part), step_size=1)
    # print(f"Number of variable windows created: {len(variable_windows)}")
    # Cách 4:-------------------------------------------------------------------
    # base_window_size = len(df_part)
    # pyramid_windows = create_pyramid_windows(dfETH_resampled_norm, base_window_size, num_levels=3)
    # print(f"Number of pyramid windows created: {len(pyramid_windows)}")
    # Cách 5:-------------------------------------------------------------------
    # num_segments = 10
    # paa_segments = create_paa_segments(dfETH_resampled_norm, num_segments)

    # NHỚ TRẢ NÓ RA ĐÂY
    sliding_windows = overlapping_windows
    return df_part, dfETH_resampled, df_part_norm, sliding_windows

# Hàm tính toán DTW và PMK
def calculate_distances(df_part_norm, sliding_windows):
    # Tính toán khoảng cách DTW
    dtw_distances = compare_with_dtw(df_part_norm, sliding_windows)

    # Tìm best match và worst match
    best_match_window1, match_index1, min_distance = find_best_match(dtw_distances, sliding_windows)
    worst_match_window2, match_index2, max_distance = find_worst_match(dtw_distances, sliding_windows)

    print(f"Best matching window with normalized DTW distance {min_distance} tại vị trí {match_index1}")
    print(best_match_window1)
    print(f"Worst matching window with normalized DTW distance {max_distance} tại vị trí {match_index2}")
    print(worst_match_window2)

    return dtw_distances, best_match_window1, worst_match_window2, match_index1

# Hàm vẽ biểu đồ upgrade
def visualize_charts(df_part, dtw_distances, best_match_window1, worst_match_window2, dfETH_resampled, match_index1, window_size):
    # Biểu đồ 1: Dữ liệu ban đầu
    fig = go.Figure()
    add_trace_candlestick(fig, df_part, name="Crypto Candlestick")
    add_trace_line(fig, df_part, name="Crypto Line Chart", yaxis='y2')
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            showline=True
        )
    )
    fig.show()
    
    # Biểu đồ 2: DTW Distances
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(dtw_distances))), y=dtw_distances, mode='lines', name='DTW Distances'))
    fig2.update_layout(title='DTW Distances for Sliding Windows', xaxis_title='Sliding Window Index', yaxis_title='DTW Distance', showlegend=True)
    fig2.show()

    # Biểu đồ 3: Best match window (từ dữ liệu đã chuẩn hóa)
    fig3 = go.Figure()
    add_trace_line(fig3, best_match_window1, name="best_match")
    fig3.show()

    # Biểu đồ 4: Worst match window (từ dữ liệu đã chuẩn hóa)
    fig4 = go.Figure()
    add_trace_line(fig4, worst_match_window2, name="worst_match")
    fig4.show()

    # Biểu đồ 5: Dữ liệu gốc của best match
    best_match_window_original = dfETH_resampled.iloc[match_index1:match_index1 + window_size]
    fig5 = go.Figure()
    add_trace_candlestick(fig5, best_match_window_original, name="Best Match Original Candlestick")
    add_trace_line(fig5, best_match_window_original, name="Best Match Original Line", yaxis='y2')
    fig5.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            showline=True
        )
    )
    fig5.show()

# MAIN
# Load data
dfETH = get_df_from_name('ETH')
print(dfETH)

# Cấu hình thông số
start_time = '2023-09-19 00:00:00'
end_time = '2023-09-21 00:00:00'
timeframe = '1H'

# Xử lý dữ liệu
df_part, dfETH_resampled, df_part_norm, sliding_windows = process_data(dfETH, start_time, end_time, timeframe)

# Tính toán DTW và tìm best match, worst match
dtw_distances, best_match_window1, worst_match_window2, match_index1 = calculate_distances(df_part_norm, sliding_windows)

# Trực quan hóa biểu đồ, bao gồm cả dữ liệu gốc best_match
visualize_charts(df_part, dtw_distances, best_match_window1, worst_match_window2, dfETH_resampled, match_index1, len(df_part))
