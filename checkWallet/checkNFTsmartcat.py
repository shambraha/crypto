import pandas as pd
import requests

# Hàm lấy danh sách NFT từ OpenSea
def get_nfts_by_opensea(chain, wallet_address, api_key):
    url = f"https://api.opensea.io/api/v2/chain/{chain}/account/{wallet_address}/nfts"
    headers = {
        "Accept": "application/json",
        "X-API-KEY": api_key
    }
    params = {
        "limit": 50
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Lỗi khi truy vấn OpenSea cho địa chỉ {wallet_address}: {response.status_code}")
        return None

# Hàm lấy danh sách NFT từ PolygonScan
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
        print(f"Lỗi từ PolygonScan: {data['message']}")
        return None

# Hàm lấy danh sách NFT từ MintChain
def get_nft_by_mintchain(address_hash):
    url = f"https://explorer.mintchain.io/api/v2/addresses/{address_hash}/nft"
    params = {
        "type": "ERC-721,ERC-404,ERC-1155"  # Truy vấn cả 3 loại NFT
    }
    headers = {
        "Accept": "application/json"
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Lỗi: {response.status_code}")
        return None

# CỤM CODE THỬ NGHIỆM VỚI 1 ADDRESS-------------------------------------------------------------------------
# # Sử dụng hàm
# address_hash = "0xB58b4b61B8C2981c2216FE30b2bccd797EaC6225"
# nfts = get_nft_by_mintchain(address_hash)

# # Hiển thị kết quả
# if nfts and 'items' in nfts:
#     for nft in nfts['items']:
#         name = nft.get('metadata', {}).get('name', 'N/A')
#         print(f"Tên NFT: {name}")
# else:
#     print("Không có NFT nào được tìm thấy.")


# CỤM CODE TỔNG HỢP-----------------------------------------------------------------------------------------
# Đọc dữ liệu từ file CSV và tạo danh sách địa chỉ
data = pd.read_csv('../data/smartcat.csv')
address_list = data['address'].tolist()

# API Keys và chuỗi blockchain
api_key_opensea = "46b23127b3664216b7ee1f59f112fb3a"
api_key_polyscan = "7FGMY5H38U723M87666D89H1MGHC64G4H3"
chain = "ethereum"  #  [only for def get_nfts_by_opensea] hỗ trợ các chuỗi khác nhau: "matic", "klaytn", v.v.

# Tạo cột mới để lưu kết quả
column_OpenSea = 'OpenSea_NFTs ETH'
column_PolygonScan = 'PolygonScan_NFTs'
column_MintChain = 'MintChain_NFTs'
data[column_OpenSea] = ""
data[column_PolygonScan] = ""
data[column_MintChain] = ""

# Duyệt qua từng địa chỉ và lấy tên NFT từ OpenSea và PolygonScan
for index, address in enumerate(address_list):
    print(f"Đang xử lý địa chỉ {index + 1}/{len(address_list)}: {address}")

    # Gọi hàm từ OpenSea
    opensea_nfts = get_nfts_by_opensea(chain, address, api_key_opensea)
    opensea_names = [nft.get('name') for nft in opensea_nfts.get('nfts', []) if nft.get('name')] if opensea_nfts else []
    data.at[index, column_OpenSea] = ', '.join(opensea_names) if opensea_names else "Không có NFT"

    # Gọi hàm từ PolygonScan
    polyscan_nfts = get_nft_by_polyscan(address, api_key_polyscan)
    polyscan_names = [nft.get('tokenName') for nft in polyscan_nfts if nft.get('tokenName')] if polyscan_nfts else []
    data.at[index, column_PolygonScan] = ', '.join(polyscan_names) if polyscan_names else "Không có NFT"

    # Gọi hàm từ MintChain
    mintchain_nfts = get_nft_by_mintchain(address)
    mintchain_names = [nft.get('metadata', {}).get('name') for nft in mintchain_nfts.get('items', []) if nft.get('metadata', {}).get('name')] if mintchain_nfts else []
    data.at[index, column_MintChain] = ', '.join(mintchain_names) if mintchain_names else "Không có NFT"

# Ghi dữ liệu vào file CSV mới
data.to_csv('../data/smartcat_with_nfts.csv', index=False)
print(f"Đã ghi dữ liệu vào file CSV mới với cột tên từ {column_OpenSea} và {column_PolygonScan} và {column_MintChain}.")
