import os
import pandas as pd
import torch
import numpy as np


# Function Check1--------------------------------------
def check_dataset_splits_old(train_dataset, test_dataset, dataset):
    """
    Hàm kiểm tra số lượng mẫu trong tập huấn luyện và tập kiểm thử.
    
    Parameters:
    train_dataset: Tập dữ liệu huấn luyện.
    test_dataset: Tập dữ liệu kiểm thử.
    dataset: Tập dữ liệu ban đầu để đảm bảo tính toàn vẹn.
    """
    print("Check1: Kiểm tra số lượng mẫu trong train_loader và test_loader")
    print(f"Số mẫu trong tập huấn luyện: {len(train_dataset)}")
    print(f"Số mẫu trong tập kiểm thử: {len(test_dataset)}")
    
    # Đảm bảo tổng số mẫu trong train_dataset và test_dataset bằng với tổng số mẫu ban đầu
    assert len(train_dataset) + len(test_dataset) == len(dataset), "Tổng số mẫu không khớp!"
    print("---------------------------------------------------------------")

def check_dataset_splits(dataset, train_dataset, val_dataset, test_dataset=None):
    """
    Hàm kiểm tra số lượng mẫu trong tập huấn luyện, tập validation và tập kiểm thử (nếu có).
    
    Parameters:
    train_dataset: Tập dữ liệu huấn luyện.
    val_dataset: Tập dữ liệu validation.
    dataset: Tập dữ liệu ban đầu để đảm bảo tính toàn vẹn.
    test_dataset: (Tùy chọn) Tập dữ liệu kiểm thử. Nếu không có thì chỉ kiểm tra train và val.
    """
    print("Check1: Kiểm tra số lượng mẫu trong train_dataset, val_dataset và test_dataset (nếu có)")
    print(f"Số mẫu trong tập huấn luyện: {len(train_dataset)}")
    print(f"Số mẫu trong tập validation: {len(val_dataset)}")

    if test_dataset is not None:
        print(f"Số mẫu trong tập kiểm thử: {len(test_dataset)}")
    
    # Đảm bảo tổng số mẫu trong train_dataset, val_dataset và test_dataset (nếu có) bằng với tổng số mẫu ban đầu
    total = len(train_dataset) + len(val_dataset)
    
    if test_dataset is not None:
        total += len(test_dataset)
    
    assert total == len(dataset), "Tổng số mẫu không khớp!"
    print("---------------------------------------------------------------")


# Function Check2--------------------------------------
def check_batch_in_loader(train_loader):
    """
    Hàm kiểm tra kích thước và dữ liệu của một batch trong train_loader.
    
    Parameters:
    train_loader: DataLoader của tập huấn luyện chứa các batch dữ liệu.
    """
    print("Check2: Kiểm tra kích thước và dữ liệu của một batch trong train_loader")
    
    # Lấy một batch từ train_loader để kiểm tra
    data_iter = iter(train_loader)
    X_batch, y_batch = next(data_iter)
    
    # Kiểm tra kích thước và nội dung của batch
    print(f"Kích thước batch X: {X_batch.shape}")  # Kiểm tra kích thước batch
    print(f"Kích thước batch y: {y_batch.shape}")
    print(f"Một batch dữ liệu X: {X_batch}")
    print(f"Một batch nhãn y: {y_batch}")
    print("---------------------------------------------------------------")

# Function Check3--------------------------------------
def check_unique_labels_in_dataset(y_tensor):
    """
    Hàm kiểm tra tính đúng đắn của nhãn trong một batch và toàn bộ dataset.

    Parameters:
    y_batch: Nhãn của một batch từ DataLoader.
    y_tensor: Tất cả các nhãn của toàn bộ dataset dưới dạng tensor.
    """
    print("Check3: Kiểm tra tính đúng đắn của nhãn")
    # print(f"Nhãn tương ứng với một batch: {y_batch}")
    
    # Kiểm tra các nhãn duy nhất trong toàn bộ dataset
    print(f"Tất cả các nhãn duy nhất: {torch.unique(y_tensor)}")
    print("---------------------------------------------------------------")

# Function Check4-------------------------------------
def check_label_distribution_old(train_dataset, test_dataset):
    """
    Hàm kiểm tra phân phối nhãn giữa tập huấn luyện và tập kiểm thử.

    Parameters:
    train_dataset: Tập dữ liệu huấn luyện (train_dataset).
    test_dataset: Tập dữ liệu kiểm thử (test_dataset).
    """
    print("Check4: Kiểm tra phân phối nhãn giữa train/test")
    
    # Lấy nhãn từ train_dataset và test_dataset
    train_labels = [y for _, y in train_dataset]
    test_labels = [y for _, y in test_dataset]
    
    # Kiểm tra phân phối nhãn
    print(f"Phân phối nhãn trong tập huấn luyện: {np.bincount(train_labels)}")
    print(f"Phân phối nhãn trong tập kiểm thử: {np.bincount(test_labels)}")
    print("---------------------------------------------------------------")

def check_label_distribution(train_dataset, val_dataset, test_dataset=None):
    """
    Hàm kiểm tra phân phối nhãn giữa tập huấn luyện, tập validation và tập kiểm thử (nếu có).

    Parameters:
    train_dataset: Tập dữ liệu huấn luyện (train_dataset).
    val_dataset: Tập dữ liệu validation (val_dataset).
    test_dataset: (Tùy chọn) Tập dữ liệu kiểm thử (test_dataset).
    """
    print("Check4: Kiểm tra phân phối nhãn giữa train/val/test")

    # Lấy nhãn từ train_dataset và val_dataset
    train_labels = [y for _, y in train_dataset]
    val_labels = [y for _, y in val_dataset]

    # Kiểm tra phân phối nhãn cho train và val
    print(f"Phân phối nhãn trong tập huấn luyện: {np.bincount(train_labels)}")
    print(f"Phân phối nhãn trong tập validation: {np.bincount(val_labels)}")

    # Nếu có test_dataset, kiểm tra thêm phân phối nhãn của nó
    if test_dataset is not None:
        test_labels = [y for _, y in test_dataset]
        print(f"Phân phối nhãn trong tập kiểm thử: {np.bincount(test_labels)}")

    print("---------------------------------------------------------------")


# Function Check5----------------------------------------------
def check_dataloader_functionality(train_loader, test_loader):
    """
    Hàm kiểm tra tính hoạt động của DataLoader, in ra kích thước của các batch trong train_loader và test_loader.

    Parameters:
    train_loader: DataLoader của tập huấn luyện.
    test_loader: DataLoader của tập kiểm thử.
    """
    print("Check5: Kiểm tra DataLoader hoạt động tốt")
    
    # Kiểm tra train_loader
    print("Kiểm tra train_loader")
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx+1} - Kích thước X: {X_batch.shape}, Kích thước y: {y_batch.shape}")
        if batch_idx == 1:  # In 2 batch đầu tiên
            break
    
    # Kiểm tra test_loader
    print("Kiểm tra test_loader")
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        print(f"Batch {batch_idx+1} - Kích thước X: {X_batch.shape}, Kích thước y: {y_batch.shape}")
        if batch_idx == 1:  # In 2 batch đầu tiên
            break

# Function Test1--------------------------------------------
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

