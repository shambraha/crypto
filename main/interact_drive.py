import json
import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import pandas as pd
from interact_dataframe import chooseFormattimeColumn

# Đường dẫn đến tệp credentials và client_secret
CREDENTIALS_PATH = 'data/credentials.json'
CLIENT_SECRET_PATH = 'data/client_secret.json'
# E:\MyPythonCode\Crypto\data\csv\client_secret.json

# Function phụ------------------------------------------------------
# Hàm đọc tệp và trả về dữ liệu JSON
def read_credentials(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function 1--------------------------------------------------------
# Hàm ủy quyền và khởi tạo GoogleDrive
def authorize_and_create_drive():
    # Kiểm tra xem tệp client_secret.json có tồn tại không
    if not os.path.exists(CLIENT_SECRET_PATH):
        print('File client_secret.json not found. Please download it from Google Developer Console.')
        return None

    # Tạo đối tượng GoogleAuth
    gauth = GoogleAuth()

    # Thiết lập đường dẫn tệp client_secret.json
    gauth.DEFAULT_SETTINGS['client_config_file'] = CLIENT_SECRET_PATH

    # Kiểm tra xem tệp credentials.json có tồn tại không
    if os.path.exists(CREDENTIALS_PATH):
        # Tải thông tin xác thực từ tệp credentials.json
        gauth.LoadCredentialsFile(CREDENTIALS_PATH)
        
        # Kiểm tra nếu token đã hết hạn, cần xác thực lại
        if gauth.access_token_expired:
            print("Token expired, re-authenticating...")
            gauth.LocalWebserverAuth()  # Xác thực lại
            gauth.SaveCredentialsFile(CREDENTIALS_PATH)  # Lưu token mới vào credentials.json
        else:
            gauth.Authorize()  # Đăng nhập sử dụng token hiện tại
            print("Token is valid.")
    else:
        # Nếu không có tệp credentials.json, cần xác thực lần đầu
        print("No credentials found, starting authentication...")
        gauth.LocalWebserverAuth()  # Mở trình duyệt để người dùng xác thực
        gauth.SaveCredentialsFile(CREDENTIALS_PATH)  # Lưu token vào credentials.json

    # Tạo đối tượng GoogleDrive
    drive = GoogleDrive(gauth)
    print("Authentication successful. You can now use 'drive' to interact with Google Drive.")
    return drive

# Function 2-------------------------------------------------------------------------
# Hàm kiểm tra và tạo thư mục trên Google Drive, trả về folder['id]
def create_folder_if_not_exists(drive, folder_name):
    # Tìm kiếm thư mục theo tên
    folder_list = drive.ListFile({'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
    
    if folder_list:
        # Nếu thư mục tồn tại, trả về ID của thư mục
        print(f"Folder '{folder_name}' already exists.")
        return folder_list[0]['id']
    else:
        # Nếu thư mục không tồn tại, tạo thư mục mới
        folder_metadata = {'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        print(f"Folder '{folder_name}' created.")
        return folder['id']
    
# Function 3--------------------------------------------------------------------------
# Hàm upload tệp hoặc DataFrame lên Google Drive
def upload_to_drive(drive, folder_id, file_path=None, df=None, file_name='uploaded_file.csv'):
    if file_path:
        # Tạo tệp Google Drive từ file_path
        file = drive.CreateFile({'title': os.path.basename(file_path), 'parents': [{'id': folder_id}]})
        file.SetContentFile(file_path)
        file.Upload()
        print(f"File '{file_path}' uploaded to Google Drive.")
    elif df is not None:
        # Lưu DataFrame thành tệp CSV tạm thời
        temp_csv_path = 'temp.csv'
        df.to_csv(temp_csv_path, index=False)

        # Tạo tệp Google Drive từ DataFrame
        file = drive.CreateFile({'title': file_name, 'parents': [{'id': folder_id}]})
        file.SetContentFile(temp_csv_path)
        file.Upload()
        print(f"DataFrame uploaded as '{file_name}' to Google Drive.")

        # Xóa tệp CSV tạm thời
        file = None
        os.remove(temp_csv_path)
    else:
        print("No file or DataFrame provided to upload.")

# Function 4----------------------------------------------------------------
# Hàm liệt kê các tệp CSV trong thư mục Google Drive
def list_csv_files_in_drive_folder(drive, folder_id):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false and mimeType='text/csv'"}).GetList()
    return file_list

# Function 5-old----------------------------------------------------------------
# Hàm đọc tệp CSV từ Google Drive và trả về DataFrame
def read_csv_from_drive(drive, file_id):
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile('temp_download.csv')  # Tải tệp xuống với tên tạm thời
    df = pd.read_csv('temp_download.csv')  # Đọc tệp CSV vào DataFrame
    os.remove('temp_download.csv')  # Xóa tệp tạm thời sau khi đọc
    return df
# Function 5-upgrade------------------------------------------------------------
def get_df_from_name_gdrive(drive, symbol):
    folder_id = create_folder_if_not_exists(drive, 'binanceCSV')
    csv_files = list_csv_files_in_drive_folder(drive, folder_id)
    for csv_file in csv_files:
        if csv_file['title'] == f'binance_{symbol}USDT.csv':
            print(f"Reading file: {csv_file['title']}")
            df = read_csv_from_drive(drive, csv_file['id'])
            chooseFormattimeColumn(df, 'Open Time')
            return df
    print(f"No CSV file found for symbol {symbol}")
    return None

# Function 6--------------------------------------------------------------------
# Tạo danh sách symbols từ file_list
def extract_symbols_from_file_list(file_list):
    symbols = []
    for file in file_list:
        # Lấy tiêu đề của tệp
        file_title = file['title']
        # Kiểm tra định dạng tệp để đảm bảo chỉ lấy những tệp có định dạng binance_<symbol>USDT.csv
        if file_title.startswith('binance_') and file_title.endswith('USDT.csv'):
            # Lấy phần symbol từ tiêu đề tệp, ví dụ: binance_BTCUSDT.csv -> BTC
            symbol = file_title.split('_')[1].replace('USDT.csv', '')
            symbols.append(symbol)
    return symbols

# Function gộp
# Tạo danh sách symbols
def extract_symbols_from_drive():
    drive = authorize_and_create_drive()
    if drive:
        folder_id = create_folder_if_not_exists(drive, 'binanceCSV')
        file_list = list_csv_files_in_drive_folder(drive, folder_id)
        available_symbols = extract_symbols_from_file_list(file_list)
    return available_symbols

# # Sử dụng 
# # Gọi hàm authorize_and_create_drive để khởi tạo Google Drive
# drive = authorize_and_create_drive()
# if drive:
    # Thực hiện các thao tác với drive, ví dụ liệt kê các tệp
    # file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    # for file in file_list:
    #     print('Title: %s, ID: %s' % (file['title'], file['id']))

    # Kiểm tra và tạo thư mục 'binanceCSV' nếu chưa có
    # folder_id = create_folder_if_not_exists(drive, 'binanceCSV')

    # Ví dụ sử dụng hàm upload: upload tệp từ đường dẫn
    # file_path = r'E:\MyPythonCode\Crypto\data\csv\binance_TRXUSDT.csv'
    # upload_to_drive(drive, folder_id, file_path=file_path)

    # Ví dụ sử dụng hàm upload: upload DataFrame
    # data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    # df = pd.DataFrame(data)
    # upload_to_drive(drive, folder_id, df=df, file_name='example_df2.csv')

    # folder_id = create_folder_if_not_exists(drive, 'binanceCSV')
    # csv_files = list_csv_files_in_drive_folder(drive, folder_id)

    # for csv_file in csv_files:
    #     print(f"Reading file: {csv_file['title']}")
    #     df = read_csv_from_drive(drive, csv_file['id'])
    #     # print(df.head()) 

    # Đoạn này là ổn nè, thay thế cho đọc file từ thư mục máy tính-----------
    # folder_id = create_folder_if_not_exists(drive, 'binanceCSV')
    # symbol = 'BTC'  # Thay đổi symbol theo ý muốn
    # df = get_df_from_name_gdrive(drive, symbol)
    # if df is not None:
    #     print(df.tail())

    # folder_id = create_folder_if_not_exists(drive, 'binanceCSV')
    # file_list = list_csv_files_in_drive_folder(drive, folder_id)
    # available_symbols = extract_symbols_from_file_list(file_list)
    # print(available_symbols)


