import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from interact_binanceApi import update_symbol_single

# Function 1---------------------------------------------
# Tạo sidebar lựa chọn bên trái
def create_sidebar_for_userinputs():
    # Sidebar for user inputs
    st.sidebar.title("Chart Options")
    timeframe = st.sidebar.selectbox("Select Timeframe", ["1H", "4H", "1D", "3D", "1W"])
    start_time = st.sidebar.date_input("From Date")
    end_time = st.sidebar.date_input("To Date")
    return timeframe, start_time, end_time

# Function 2---------------------------------------------
# Tạo trace tương ứng cho mỗi dff
#...mỗi loại bar, scatter, candlestick sẽ có hàm riêng tương ứng
def add_trace_candlestick(fig, df_resampled, name, yaxis='y', color_scheme=None):
    # Thiết lập màu mặc định nếu không có color_scheme được truyền vào
    if color_scheme is None:
        color_scheme = {'increasing': '#00cc96', 'decreasing': '#ff3b30'}  # Màu mặc định

    fig.add_trace(go.Candlestick(x=df_resampled.index, 
                            open=df_resampled['Open'], high=df_resampled['High'],
                            low=df_resampled['Low'], close=df_resampled['Close'],
                            name=name,                            
                            yaxis=yaxis,                                                       
                            increasing_line_color=color_scheme['increasing'],  # Màu cho nến tăng
                            decreasing_line_color=color_scheme['decreasing']  # Màu cho nến giảm
                        ))    

def add_trace_line(fig, df_resampled, name, yaxis='y', color_scheme=None):
    # Thiết lập màu mặc định nếu không có color_scheme được truyền vào
    if color_scheme is None:
        color_scheme = {'increasing': '#00cc96', 'decreasing': '#ff3b30'}  # Màu mặc định

    fig.add_trace(go.Scatter(x=df_resampled.index, 
                            y=df_resampled['Close'],
                            mode='lines', # có thể thay bằng 'markers' hoặc 'lines+markers'
                            name=name,                            
                            yaxis=yaxis,      
                            line=dict(color=color_scheme['increasing'])           
                        ))   

# Function 3---------------------------------------------
def update_yaxis_layout(fig, yaxis_name):
    fig.update_layout({
        yaxis_name: dict(
            overlaying='y',
            showline=True,
            side='right' 
        )
    })


# Function 4---------------------------------------------
# Hàm để vẽ biểu đồ so sánh giữa hai tuần
def plot_weeks_comparison(week1_df, year1, week1, week2_df, year2, week2):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Week {week1} ({year1})', 
                                                        f'Week {week2} ({year2})'))

    trace1 = go.Scatter(
        x=week1_df.index,
        y=week1_df['Close'],
        mode='lines',
        name=f'Week {week1} ({year1})',
        line=dict(color='blue')
    )

    trace2 = go.Scatter(
        x=week2_df.index,
        y=week2_df['Close'],
        mode='lines',
        name=f'Week {week2} ({year2})',
        line=dict(color='orange')
    )

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    fig.update_layout(
        title_text="So sánh giá Close giữa hai tuần",
        dragmode='zoom',
        height=600,
        width=1200,
        showlegend=True
    )

    st.plotly_chart(fig)

# Function 5---------------------------------------------
def update_symbol_progress(available_symbols_local):
    """
    Hàm cập nhật từng symbol với tiến độ hiển thị ở màn hình chính.
    
    available_symbols_local: Danh sách các symbols có sẵn trong thư mục CSV.
    """
    num_symbols = len(available_symbols_local)
    
    # Thanh tiến độ
    progress_bar = st.progress(0)
    
    # Danh sách lưu kết quả
    symbols_success = []
    symbols_error = []

    # Vùng hiển thị thông tin cập nhật
    status_area = st.empty()

    for i, symbol in enumerate(available_symbols_local):
        # Hiển thị tiến độ và cập nhật symbol
        with st.spinner(f"Updating {symbol} ({i+1}/{num_symbols})..."):
            result, error = update_symbol_single(symbol)

            # Lưu kết quả cập nhật
            if result:
                symbols_success.append(symbol)
            else:
                symbols_error.append((symbol, error))
        
        # Cập nhật thanh tiến độ
        progress_bar.progress((i + 1) / num_symbols)

        # Hiển thị cập nhật ngay sau khi xử lý xong mỗi symbol
        status_area.text(f"Updated {i + 1}/{num_symbols} symbols...")

    # Hiển thị danh sách các symbols đã cập nhật thành công và gặp lỗi sau khi hoàn tất
    st.write(f"Updated {len(symbols_success)} symbols successfully.")
    st.text_area("Symbols Updated", value="\n".join(symbols_success), height=100)
    
    if symbols_error:
        st.write(f"Errors with {len(symbols_error)} symbols.")
        for symbol, error_msg in symbols_error:
            st.write(f"{symbol}: {error_msg}")
    
    st.write("Update completed!")