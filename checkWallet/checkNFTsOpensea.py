import requests
import pandas as pd

# Hướng dẫn lấy Opensea-APIKEY : https://docs.opensea.io/reference/api-keys
# Hướng dẫn tương tác Opensea-API: https://docs.opensea.io/reference/list_nfts_by_account

# Hàm lấy danh sách NFT từ OpenSea
def get_nfts_by_opensea(chain, wallet_address, api_key):
    url = f"https://api.opensea.io/api/v2/chain/{chain}/account/{wallet_address}/nfts"
    headers = {
        "Accept": "application/json",
        "X-API-KEY": api_key
    }
    params = {
        "limit": 50  # Giới hạn số lượng NFT trả về
    }

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Lỗi khi truy vấn địa chỉ {wallet_address}: {response.status_code}")
        return None

def get_nft_by_polyscan(address, api_key, page=1, offset=100):
    url = "https://api.polygonscan.com/api"
    params = {
        "module": "account",
        "action": "tokennfttx",
        "address": address,
        "page": page,
        "offset": offset,
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "1":
        return data["result"]
    else:
        print(f"Lỗi: {data['message']}")
        return None


# # Sử dụng hàm
# address = "0x5f487dCa84Ad776Ba9849bECf8bA757443625F02"
# api_key_polyscan = "7FGMY5H38U723M87666D89H1MGHC64G4H3"
# nfts = get_nft_by_polyscan(address, api_key_polyscan)
# # Duyệt qua từng giao dịch và in tên của NFT
# if nfts:
#     for nft in nfts:
#         print(f"Tên NFT: {nft.get('tokenName')}")

# # Đọc dữ liệu từ file CSV và tạo danh sách địa chỉ
# data = pd.read_csv('../data/smartcat.csv')
# address_list = data['address'].tolist()

# # API Key và chuỗi blockchain
# api_key_opensea = "46b23127b3664216b7ee1f59f112fb3a"
# chain = "ethereum"  # "matic" "klaytn"

# # Tạo một cột mới để lưu tên NFT
# data[chain] = ""

# # Duyệt qua từng địa chỉ và lấy tên NFT
# for index, address in enumerate(address_list):
#     print(f"Đang xử lý địa chỉ {index + 1}/{len(address_list)}: {address}")
#     nfts = get_nfts_by_opensea(chain, address, api_key_opensea)
#     nft_names = []
#     if nfts:
#         for nft in nfts.get('nfts', []):
#             nft_names.append(nft.get('name'))
#     # Thêm danh sách tên NFT vào cột 'nft_names' (dạng chuỗi)
#     # data.at[index, 'nft_names'] = ', '.join([str(name) if name is not None else "" for name in nft_names]) if nft_names else "Không có NFT"
#     # Loại bỏ các mục trống trước khi nối chuỗi
#     data.at[index, chain] = ', '.join([str(name) for name in nft_names if name]) if nft_names else "Không có NFT"

# # Ghi lại dữ liệu vào file CSV
# data.to_csv('../data/smartcat_with_nfts.csv', index=False)
# print(f"Đã ghi dữ liệu vào file CSV mới với cột tên {chain}.")


