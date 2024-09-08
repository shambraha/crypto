import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
import re
from interact_drive import authorize_and_create_drive, upload_to_drive, create_folder_if_not_exists
from config.config import csv_folder_path

# Function 1---------------------------------------------------------------------------
def get_binance_data(symbol, days):
    # Mã thực thi của hàm

    # Khai báo API Key và API Secret Key của bạn
    api_key = '19gntwOXmLa218sFvQggcxHS9mjvSXzC85932i4UnGL8WjNxmyIN0d7oUeML9RpQ'
    api_secret = 'wepu0GtS1M2rmehVylpCAHuRqEw8o8x6PpXhiyh9sogYYE8JT84rmDAJTCxXM060'

    # Tạo đối tượng Client với thông tin xác thực API
    client = Client(api_key, api_secret)

    # Lấy thời điểm hiện tại và thời điểm trước đó 1 tuần
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    # Chuyển đổi thời gian sang milliseconds
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    # Lấy thông tin klines từ API của Binance
    pairsymbol = f'{symbol}USDT'
    interval = Client.KLINE_INTERVAL_1HOUR
    klines = client.get_historical_klines(pairsymbol, interval, start_str=start_time_ms, end_str=end_time_ms)

    # Chuyển đổi klines thành một DataFrame
    columns = ['Open Time',
               'Open',
               'High',
               'Low',
               'Close',
               'Volume',
               'Close Time',
               'Quote Volume',
               'Number of Trades',
               'Taker Buy Base Volume',
               'Taker Buy Quote Volume',
               'zero']
    df = pd.DataFrame(klines, columns=columns)

    # Lưu DataFrame vào file Excel    
    # folder_path = os.path.join(os.getcwd(), 'data', 'csv')
    file_path = f'{csv_folder_path}/binance_{pairsymbol}.csv'
    df.to_csv(file_path, index=False, float_format='%.2f')
    print(f"DataFrame đã được lưu vào {file_path}")

# Function 2----------------------------------------------------------------------
def get_binance_data_to_gdrive(symbol, days):
    # Mã thực thi của hàm

    # Khai báo API Key và API Secret Key của bạn
    api_key = '19gntwOXmLa218sFvQggcxHS9mjvSXzC85932i4UnGL8WjNxmyIN0d7oUeML9RpQ'
    api_secret = 'wepu0GtS1M2rmehVylpCAHuRqEw8o8x6PpXhiyh9sogYYE8JT84rmDAJTCxXM060'

    # Tạo đối tượng Client với thông tin xác thực API
    client = Client(api_key, api_secret)

    # Lấy thời điểm hiện tại và thời điểm trước đó 1 tuần
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    # Chuyển đổi thời gian sang milliseconds
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    # Lấy thông tin klines từ API của Binance
    pairsymbol = f'{symbol}USDT'
    interval = Client.KLINE_INTERVAL_1HOUR
    klines = client.get_historical_klines(pairsymbol, interval, start_str=start_time_ms, end_str=end_time_ms)

    # Chuyển đổi klines thành một DataFrame
    columns = ['Open Time',
               'Open',
               'High',
               'Low',
               'Close',
               'Volume',
               'Close Time',
               'Quote Volume',
               'Number of Trades',
               'Taker Buy Base Volume',
               'Taker Buy Quote Volume',
               'zero']
    df = pd.DataFrame(klines, columns=columns)

    # Tải DataFrame df lên gDrive
    drive = authorize_and_create_drive()
    if drive:        
        folder_id = create_folder_if_not_exists(drive, 'binanceCSV')
        upload_to_drive(drive, folder_id, df=df, file_name=f'binance_{pairsymbol}.csv')
    


# # Get Klines Datas from Binance using API, input (string: symbol, number: days)
# available_symbols = ['BTC', 'ETH']
# for symbol in available_symbols:
#     full_symbol = f"{symbol}USDT"  # Chuyển đổi BTC thành BTCUSDT, ETH thành ETHUSDT, ...
#     #get_binance_data(symbol, 365)
#     get_binance_data_to_gdrive(symbol, 365)