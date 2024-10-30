import pandas as pd

def phan_loai_tier():
    # Đọc file CSV
    df = pd.read_csv(r'E:\JS2024\improveCode\data\record_plumepassport_full.csv')

    # Loại bỏ dấu phẩy và chuyển cột 'point' thành kiểu số nguyên, Thay thế NaN bằng 0 trong cột 'point'
    df['point'] = df['point'].fillna('0').str.replace(',', '').astype(int)

    # Định nghĩa phân loại cho cột 'tier' dựa trên giá trị của 'point'
    def categorize_tier(point):
        if point < 750000:
            return 'None'
        elif 750000 <= point < 1800000:
            return 'Plus'
        elif 1800000 <= point < 2500000:
            return 'Premium'
        elif 2500000 <= point <= 5000000:
            return 'Business'

    # Áp dụng hàm để tạo cột 'tier'
    df['tier'] = df['point'].apply(categorize_tier)

    # Lưu hoặc hiển thị DataFrame đã cập nhật
    # print(df)
    df.to_csv('../data/plume_with_tier.csv', index=False)

def thong_ke_tier():
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(r'E:\MyPythonCode\Crypto\data\plume_with_tier.csv')

    # Đảm bảo tất cả hạng được liệt kê (bao gồm hạng không có trong dữ liệu)
    tier_counts = df['tier'].value_counts().reindex(['Business', 'Premium', 'Plus', 'None'], fill_value=0)
    print(tier_counts)

# phan_loai_tier()
thong_ke_tier()