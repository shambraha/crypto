import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
from interact_dataframe import get_df_from_name, filter_and_resample_df

# Function 1-------------------------------------------------------------------------------------------
# Hàm để xử lý và hiển thị bảng và biểu đồ
def price_diff_statistic_and_plot(symbols, start_time, end_time, price_diff_func):
    for symbol in symbols:
        # Đọc DataFrame cho symbol
        df = get_df_from_name(symbol)

        # Lọc DataFrame theo thời gian đã chọn
        df_filtered = filter_and_resample_df(df, "1h", start_time, end_time)

        # Nếu DataFrame sau khi lọc không có dữ liệu, bỏ qua symbol này
        if df_filtered.empty:
            continue

        # Thêm cột ngày trong tuần và giờ vào DataFrame
        df_filtered['DayofWeek'] = df_filtered.index.dayofweek
        df_filtered['Hour'] = df_filtered.index.hour

        # Tính toán chênh lệch giá theo hàm được truyền vào (price_diff_func)
        df_filtered['Price_Diff'] = price_diff_func(df_filtered)

        # #region bảng 24*7-------------------------------------------------
        # # Tạo một bảng 24x7 (giờ là hàng, ngày trong tuần là cột)
        # price_diff_matrix = np.zeros((24, 7))

        # # Điền vào bảng với chênh lệch giá giữa các giờ
        # for hour in range(24):
        #     for day in range(7):
        #         mask = (df_filtered['DayofWeek'] == day) & (df_filtered['Hour'] == hour)
        #         price_diff = df_filtered[mask]['Price_Diff'].mean()
        #         if not np.isnan(price_diff):
        #             price_diff_matrix[hour, day] = price_diff
        
        # # Hiển thị bảng chênh lệch giá 24x7 dưới dạng heatmap
        # st.write(f"Bảng chênh lệch giá cho {symbol}")
        # st.dataframe(pd.DataFrame(price_diff_matrix, index=[f"{h}:00" for h in range(24)],
        #                           columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']).style.format("{:.2f}"))
        #endregion bảng 24*7------

        # #region bảng 7*24------------------------------------------------------
        # Tạo một bảng 7x24 cho giá price_diff
        price_diff_matrix = np.zeros((7, 24))

        # Điền vào bảng với giá trị chênh lệch
        for day in range(7):
            for hour in range(24):
                mask = (df_filtered['DayofWeek'] == day) & (df_filtered['Hour'] == hour)
                price_diff = df_filtered[mask]['Price_Diff'].mean()
                if not np.isnan(price_diff):
                    price_diff_matrix[day, hour] = price_diff

        # Hiển thị bảng chênh lệch giá 7x24 dưới dạng heatmap
        st.write(f"Bảng chênh lệch giá cho {symbol}")
        st.dataframe(pd.DataFrame(price_diff_matrix, index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                  columns=[f"{h}:00" for h in range(24)]).style.format("{:.2f}"))
        #endregion bảng 7*24------

        #region Tạo một bảng 7x24 cho giá Close----------------------------------------
        close_price_matrix = np.zeros((7, 24))

        # Điền vào bảng với giá Close
        for day in range(7):
            for hour in range(24):
                mask = (df_filtered['DayofWeek'] == day) & (df_filtered['Hour'] == hour)
                close_price = df_filtered[mask]['Close'].mean()
                if not np.isnan(close_price):
                    close_price_matrix[day, hour] = close_price

        # Hiển thị bảng giá Close 7x24 dưới dạng heatmap
        st.write(f"Bảng giá Close cho {symbol}")
        st.dataframe(pd.DataFrame(close_price_matrix, index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                  columns=[f"{h}:00" for h in range(24)]).style.format("{:.2f}"))
        #endregion bảng 7*24 giá Close-----

        # #region Vẽ 7 đường thẳng với Plotly cho Price_Diff-------------------------------
        # st.write(f"Biểu đồ chênh lệch giá theo giờ cho {symbol}")
        
        # Tạo biểu đồ Plotly
        fig = go.Figure()

        # Vẽ 7 đường cho từng ngày trong tuần
        days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for day in range(7):
            fig.add_trace(go.Scatter(
                x=[f"{h}:00" for h in range(24)],
                y=price_diff_matrix[day],
                mode='lines',
                name=days_of_week[day]
            ))

        # Thêm tiêu đề và nhãn trục
        fig.update_layout(
            title=f"Biểu đồ chênh lệch giá theo giờ cho {symbol}",
            xaxis_title="Giờ trong ngày",
            yaxis_title="Chênh lệch giá",
            legend_title="Ngày trong tuần"
        )

        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig)
        #endregion vẽ 7 đường thẳng với Plotly---------------


# Function 2 ------------------------------------------------------------------------------------------
# Hàm để xử lý và hiển thị bảng và biểu đồ heatmap
def percent_diff_statistic_and_plot(symbols, start_time, end_time, price_diff_func):
    for symbol in symbols:
        # Đọc DataFrame cho symbol
        df = get_df_from_name(symbol)

        # Lọc DataFrame theo thời gian đã chọn
        df_filtered = filter_and_resample_df(df, "1h", start_time, end_time)

        # Nếu DataFrame sau khi lọc không có dữ liệu, bỏ qua symbol này
        if df_filtered.empty:
            continue

        # Thêm cột ngày trong tuần và giờ vào DataFrame
        df_filtered['DayofWeek'] = df_filtered.index.dayofweek
        df_filtered['Hour'] = df_filtered.index.hour

        # Tính toán % thay đổi theo hàm được truyền vào (price_diff_func)
        df_filtered['Price_Diff'] = price_diff_func(df_filtered)

        # #region bảng 7*24------------------------------------------------------
        # Tạo một bảng 7x24 cho giá price_diff
        price_diff_matrix = np.zeros((7, 24))

        # Điền vào bảng với giá trị chênh lệch
        for day in range(7):
            for hour in range(24):
                mask = (df_filtered['DayofWeek'] == day) & (df_filtered['Hour'] == hour)
                price_diff = df_filtered[mask]['Price_Diff'].mean()
                if not np.isnan(price_diff):
                    price_diff_matrix[day, hour] = price_diff

        # Hiển thị heatmap % thay đổi giá
        st.write(f"Heatmap % thay đổi cho {symbol}")

        fig = px.imshow(price_diff_matrix, 
                        labels=dict(x="Giờ trong ngày", y="Ngày trong tuần", color="% Thay đổi"),
                        x=[f"{h}:00" for h in range(24)],
                        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        aspect="auto")

        fig.update_layout(
            title=f"Heatmap % thay đổi theo giờ cho {symbol}",
            xaxis_title="Giờ trong ngày",
            yaxis_title="Ngày trong tuần",
            coloraxis_colorbar=dict(title="% Thay đổi")
        )

        # Hiển thị heatmap trong Streamlit
        st.plotly_chart(fig)
        #endregion bảng 7*24------