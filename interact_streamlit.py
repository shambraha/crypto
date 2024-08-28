import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

# Function 3---------------------------------------------
def update_yaxis_layout(fig, yaxis_name):
    fig.update_layout({
        yaxis_name: dict(
            overlaying='y',
            showline=True,
            side='right' 
        )
    })
