import pandas as pd
import numpy as np
import talib
import torch

# 1. Danh sách chỉ số kỹ thuật ---------------------------------------------
# 1.1 Simple Moving Average (SMA)
# Được tính bằng trung bình cộng của giá trong một khoảng thời gian xác định.
def calculate_sma(df, period=10, column='Close'):
    return df[column].rolling(window=period).mean()

# 1.2 Exponential Moving Average (EMA)
# Đường trung bình động có trọng số cao hơn cho các giá gần đây.
def calculate_ema(df, period=10, column='Close'):
    return df[column].ewm(span=period, adjust=False).mean()

# 1.3 Relative Strength Index (RSI)
# Được sử dụng để đánh giá mức quá mua hoặc quá bán của tài sản.
def calculate_rsi(df, period=14):
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 1.4 MACD (Moving Average Convergence Divergence)
# Đo lường mối quan hệ giữa hai đường trung bình động.
def calculate_macd(df):
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

# 1.5 Bollinger Bands
# Dựa trên SMA và độ lệch chuẩn, nó cung cấp các dải để đánh giá độ biến động.
def calculate_bollinger_bands(df, period=20):
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# 1.6 Average True Range (ATR)
# Đo lường mức độ biến động của tài sản.
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

# 1.7 Stochastic Oscillator
# Chỉ số này so sánh giá đóng cửa của một tài sản với dải giá của nó trong một khoảng thời gian nhất định.
def calculate_stochastic_oscillator(df, period=14):
    lowest_low = df['Low'].rolling(window=period).min()
    highest_high = df['High'].rolling(window=period).max()
    return ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

# 1.8 On-Balance Volume (OBV)
# Đánh giá sự thay đổi khối lượng giao dịch theo hướng giá để xác định áp lực mua hoặc bán.
def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

# 1.9 Rate of Change (ROC)
# Chỉ số đo lường tỷ lệ thay đổi của giá so với một khoảng thời gian trước đó.
def calculate_roc(df, period=12):
    return df['Close'].pct_change(periods=period) * 100

# 1.10 Chaikin Money Flow (CMF)
# Đo lường dòng tiền vào hoặc ra khỏi thị trường trong một khoảng thời gian.
def calculate_cmf(df, period=20):
    cmf = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    return cmf.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()

# 1.11 Directional Movement Index (DMI)
# Chỉ báo ADX đo lường sức mạnh của xu hướng giá.
def calculate_adx(df, period=14):
    # Tính các chỉ số +DM và -DM
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    # Tính True Range (TR)
    df['TR'] = pd.concat([df['High'] - df['Low'], 
                          abs(df['High'] - df['Close'].shift(1)), 
                          abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    
    # Tính Smoothed +DM, -DM và TR
    df['+DM_Smoothed'] = df['+DM'].rolling(window=period).mean()
    df['-DM_Smoothed'] = df['-DM'].rolling(window=period).mean()
    df['TR_Smoothed'] = df['TR'].rolling(window=period).mean()
    
    # Tính chỉ số +DI và -DI
    df['+DI'] = 100 * (df['+DM_Smoothed'] / df['TR_Smoothed'])
    df['-DI'] = 100 * (df['-DM_Smoothed'] / df['TR_Smoothed'])
    
    # Tính DX (Directional Index)
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    
    # Tính ADX bằng cách làm mượt giá trị DX
    adx = df['DX'].rolling(window=period).mean()
    
    return adx

# 1.12 Parabolic SAR
# Chỉ báo Parabolic SAR giúp xác định điểm dừng và đảo chiều xu hướng giá.
def calculate_sar(df, acceleration=0.02, maximum=0.2):
    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=acceleration, maximum=maximum)
    return df['SAR']

# 2. Tạo các cặp X và y ----------------------------------------------
def create_sequences_in_numpy(df, timesteps):
    X = []
    y = []

    for i in range(len(df) - timesteps):
        # Giống như việc quét chọn 1 khu vực trong bảng tính, chỉ định index từ đâu đến đâu    

        # Lấy các cột đặc trưng làm X
        X.append(df.iloc[i:i+timesteps, :-1].values)  # Tất cả cột trừ cột 'Label' là đầu vào
        # *** Giải thích ***
        # [tức là lấy hàng từ (i) đến (i+timesteps)-1, và cột giá trị là :-1] (lấy all từ cột cuối)
        
        # Lấy cột 'Label' ở vị trí i+timesteps làm y
        y.append(df.iloc[i+timesteps, -1])  # Cột 'Label' là đầu ra
        # *** Giải thích ***
        # [tức là lấy hàng (i+timesteps), và cột giá trị là -1] (lấy cột cuối)

    return np.array(X), np.array(y)

# 3. In thử một cặp X-y tuỳ chọn
def print_comboXy_in_index(X, y, example_idx):
    # Chọn một cặp thử (ví dụ, cặp đầu tiên)
    # example_idx = 2
    # In thử X và y tại index được chọn
    print("X[{}]:\n".format(example_idx), X[example_idx])
    print("\ny[{}]:\n".format(example_idx), y[example_idx])
def print_comboXy_shape(X, y):
    # Kiểm tra kích thước của X và y
    print("X shape:", X.shape)  # (số lượng mẫu, timesteps, số đặc trưng)
    print("y shape:", y.shape)  # (số lượng mẫu,)

# 4. Test thử model với batch bất kỳ
def test_model_on_batch(lstm_upgrade_model, train_loader):
    """
    Hàm kiểm tra mô hình LSTM trên một batch dữ liệu từ train_loader.

    Parameters:
    train_loader: DataLoader của tập huấn luyện chứa các batch dữ liệu.
    lstm_upgrade_model: Mô hình LSTM được thử nghiệm.
    """
    # Lấy một batch từ train_loader
    data_iter = iter(train_loader)
    X_batch, y_batch = next(data_iter)

    print("-----------------------------------------------")
    print("Chạy thử model với 1 batch để xem output_shape")
    # Kiểm tra kích thước của batch đầu vào và nhãn
    print(f"Kích thước của batch đầu vào X_batch: {X_batch.shape}")
    print(f"Kích thước của nhãn y_batch: {y_batch.shape}")

    # Đặt mô hình ở chế độ đánh giá
    lstm_upgrade_model.eval()  

    # Tắt tính toán gradient để giảm bộ nhớ và tăng tốc độ
    with torch.no_grad():  
        # Chạy forward pass qua mô hình LSTM
        outputs = lstm_upgrade_model(X_batch)

    # Kiểm tra kích thước và nội dung của đầu ra
    print(f"Kích thước đầu ra của mô hình: {outputs.shape}")
    print(f"Đầu ra của mô hình (outputs): {outputs}")
