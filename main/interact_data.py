import pandas as pd

# Function 1-----------------------------------------------------------------------------------
# Đọc danh sách symbols-exchange-tags vào một dict symbol_tags
def read_csv_to_dict_pandas(csv_file):
    # Đọc dữ liệu từ file CSV vào DataFrame bằng pandas
    df = pd.read_csv(csv_file)
    
    # Tạo một dictionary trống để lưu trữ thông tin tags cho từng symbol
    symbol_tags = {}
    
    # Sử dụng vòng lặp để duyệt qua từng dòng (row) của DataFrame
    for _, row in df.iterrows():
        symbol = row['Name']  # Lấy giá trị từ cột 'Name' làm symbol
        bybit = row['Sector']  # Lấy giá trị từ cột 'Sector' để đại diện cho exchange 'bybit'
        binance = row['Zones']  # Lấy giá trị từ cột 'Zones' để đại diện cho exchange 'binance'

        # Khởi tạo một dictionary trống cho symbol hiện tại
        symbol_tags[symbol] = {}

        # Kiểm tra xem cột 'Sector' (bybit) có dữ liệu hay không
        if pd.notna(bybit):
            # Nếu có, tách các giá trị trong cột 'Sector' bằng dấu phẩy, loại bỏ khoảng trắng
            symbol_tags[symbol]['bybit'] = [tag.strip() for tag in bybit.split(',')]
        else:
            # Nếu không có dữ liệu, gán một danh sách trống
            symbol_tags[symbol]['bybit'] = []

        # Kiểm tra xem cột 'Zones' (binance) có dữ liệu hay không
        if pd.notna(binance):
            # Nếu có, tách các giá trị trong cột 'Zones' bằng dấu phẩy, loại bỏ khoảng trắng
            symbol_tags[symbol]['binance'] = [tag.strip() for tag in binance.split(',')]
        else:
            # Nếu không có dữ liệu, gán một danh sách trống
            symbol_tags[symbol]['binance'] = []

    # Trả về dictionary symbol_tags chứa các tags đã được tổ chức theo symbol và sàn giao dịch
    return symbol_tags

# Gọi hàm
csv_path = '.\\data\\symbols_with_tags.csv'
symbol_tags = read_csv_to_dict_pandas(csv_path)

# Xem kết quả
# print(symbol_tags)

# In theo key cụ thể
# print(symbol_tags.get('AAVE'))

# In một phần bất kỳ bằng cách lấy lát cắt từ keys
# keys_to_print = list(symbol_tags.keys())[:5]  # Lấy 5 symbol đầu tiên
# for key in keys_to_print:
#     print(f"{key}: {symbol_tags[key]}")

# Function 2--------------------------------------------------------------------------
# lọc ra list các symbol theo exchange và tag mong muốn
def get_symbols_with_tag(symbol_tags, exchange, tag):
    # Tạo một danh sách để lưu các symbol thỏa điều kiện
    symbols_with_tag = []
    
    # Duyệt qua tất cả các symbol trong symbol_tags
    for symbol, exchanges in symbol_tags.items():
        # Kiểm tra nếu exchange (ví dụ: 'binance') có trong từ điển của symbol và chứa tag mong muốn
        if exchange in exchanges and tag in exchanges[exchange]:
            symbols_with_tag.append(symbol)
    
    return symbols_with_tag

# # Gọi hàm
# exchange_name = 'binance'
# tag_name = 'Storage'
# symbols_with_tag = get_symbols_with_tag(symbol_tags, exchange_name, tag_name)
# print(symbols_with_tag)

# Function 3-------------------------------------------------------------------------
# in ra danh sách tags
def list_all_tags(symbol_tags, exchange):
    # Sử dụng một set để lưu trữ các tags, đảm bảo không trùng lặp
    all_tags = set()
    
    # Duyệt qua tất cả các symbol trong symbol_tags
    for _, exchanges in symbol_tags.items():
        # Kiểm tra nếu exchange (ví dụ: 'binance' hoặc 'bybit') có trong từ điển của symbol
        if exchange in exchanges:
            # Thêm tất cả các tags của exchange vào set
            all_tags.update(exchanges[exchange])
    
    return all_tags

# Liệt kê tất cả tags của Bybit-------------------
# bybit_tags = list_all_tags(symbol_tags, 'bybit')
# print("Tất cả các tags của Bybit:", bybit_tags)
# Liệt kê tất cả tags của Binance-----------------
# binance_tags = list_all_tags(symbol_tags, 'binance')
# print("Tất cả các tags của Binance:", binance_tags)



