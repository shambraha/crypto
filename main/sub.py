from interact_dataframe import get_df_from_name, chooseFormattimeColumn, filter_and_resample_df
from interact_dataframe import resample_df, create_sliding_windows, compare_with_dtw, find_best_match
from interact_dataframe import normalize_min_max, normalize_z_score, find_worst_match, compare_with_pmk
import plotly.graph_objects as go
from interact_streamlit import add_trace_candlestick, add_trace_line, update_yaxis_layout
import webbrowser

dfETH = get_df_from_name('ETH')
print(dfETH)

# Cắt đoạn cần so sánh
start_time = '2023-09-09 07:00:00'
end_time = '2023-09-16 07:00:00'
timeframe = '4H'  # resample mỗi 1H 1D 1W 1M, thay 1 bằng any number

df_part = filter_and_resample_df(dfETH, timeframe, start_time, end_time)
print(df_part)

# Tạo Plotly Figure
fig = go.Figure()

# Thêm cả biểu đồ nến và biểu đồ đường vào figure
add_trace_candlestick(fig, df_part, name="Crypto Candlestick")

# Nếu muốn thêm biểu đồ đường thì dùng trục y khác (y2)
add_trace_line(fig, df_part, name="Crypto Line Chart", yaxis='y2')

# Cấu hình trục y2
fig.update_layout(
    yaxis2=dict(
        overlaying='y',  # overlay lên trục y ban đầu
        side='right',    # trục y2 nằm ở bên phải
        showline=True    # hiển thị đường trục
    )
)

# Hiển thị biểu đồ
fig.show()


#region tính toán với norm
#1.1 resampled df gốc theo timeframe đã chọn
dfETH_resampled = resample_df(dfETH, timeframe)
#1.2 norm cả cục to đó
dfETH_resampled_norm = normalize_min_max(dfETH_resampled)
#1.3 # Số lượng hàng của df_part làm kích thước cửa sổ
window_size = len(df_part)  
print(f"Length of window (window_size): {window_size}")
#1.3 Tạo các đoạn trượt chuẩn hoá
sliding_windows = create_sliding_windows(dfETH_resampled_norm, window_size)
print(f"Number of window_size created: {len(sliding_windows)}")

df_part_norm = normalize_min_max(df_part)
#1.4 Tính toán khoảng cách DTW với dữ liệu đã chuẩn hóa
dtw_distances = compare_with_dtw(df_part_norm, sliding_windows)
# dtw_distances = compare_with_pmk(df_part_norm, sliding_windows)

# Vẽ chart distances để trực quan so sánh
fig2 = go.Figure()
# Thêm biểu đồ đường cho DTW distances
fig2.add_trace(go.Scatter(x=list(range(len(dtw_distances))), 
                         y=dtw_distances,
                         mode='lines',
                         name='DTW Distances'))
# Cập nhật bố cục cho biểu đồ
fig2.update_layout(
    title='DTW Distances for Sliding Windows',
    xaxis_title='Sliding Window Index',
    yaxis_title='DTW Distance',
    showlegend=True
)
# Hiển thị biểu đồ
fig2.show()

# In số liệu chi tiết
best_match_window1, match_index1, min_distance = find_best_match(dtw_distances, sliding_windows)
worst_match_window2, match_index2, max_distance = find_worst_match(dtw_distances, sliding_windows)
print(f"Best matching window with normalized DTW distance {min_distance} tại vị trí {match_index1}")
print(best_match_window1)
print(f"Worst matching window with normalized DTW distance {max_distance} tại vị trí {match_index2}")
print(worst_match_window2)

# Tạo Plotly Figure
fig3 = go.Figure()
add_trace_line(fig3, best_match_window1, name="best_match")
fig3.show()

fig4 = go.Figure()
add_trace_line(fig4, worst_match_window2, name="worst_match")
fig4.show()

#endregion
