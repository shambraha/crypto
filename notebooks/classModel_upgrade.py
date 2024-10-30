import os
import time
import torch
import torch.nn as nn
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod

# Phần 1: Khởi tạo Model LSTM Base
# class BaseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout_rate, output_size):
#         super(BaseLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(hidden_size, output_size)  # Lớp đầu ra linh hoạt theo `output_size`
        
#     def forward(self, x):
#         out, _ = self.lstm(x)  # LSTM trả về `out` và `hidden states`
#         out = self.dropout(out[:, -1, :])  # Lấy đầu ra của bước cuối cùng và áp dụng dropout
#         out = self.fc(out)  # Lớp fully connected
#         return out
# Phần 1: Khởi tạo Model LSTM Base hỗ trợ output_size và ahead
class BaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, output_size, ahead=1):
        super(BaseLSTM, self).__init__()
        self.ahead = ahead  # Số bước dự đoán trước
        self.output_size = output_size  # Số lượng đặc trưng đầu ra
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size * ahead)  # Lớp fully connected chung

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM trả về `out` và `hidden states`
        out = self.dropout(out[:, -1, :])  # Lấy đầu ra của bước cuối cùng trong chuỗi
        out = self.fc(out)  # Lớp fully connected
        return out
    
# # Phần 1.1: chi tiết LSTM cho Regression
# class LSTM_Regression(BaseLSTM):
#     def __init__(self, input_size, hidden_size=50, num_layers=2, dropout_rate=0):
#         # Lớp đầu ra có `output_size=1` cho bài toán hồi quy
#         super(LSTM_Regression, self).__init__(input_size, hidden_size, num_layers, dropout_rate, output_size=1)
    
#     def forward(self, x):
#         out = super().forward(x)
#         return out  # Đầu ra không cần softmax cho hồi quy
# Phần 1.1: Lớp LSTM cho Regression hỗ trợ output_size và ahead
class LSTM_Regression(BaseLSTM):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout_rate=0, output_size=1, ahead=1):
        super(LSTM_Regression, self).__init__(input_size, hidden_size, num_layers, dropout_rate, output_size, ahead)
    
    def forward(self, x):
        out = super().forward(x)
        # Điều chỉnh đầu ra cho phù hợp với (batch_size, ahead, output_size)
        if self.ahead > 1:
            out = out.view(out.size(0), self.ahead, self.output_size)
        return out  # Không cần softmax cho hồi quy
# class LSTM_Regression(BaseLSTM):
#     def __init__(self, input_size, hidden_size=50, num_layers=2, dropout_rate=0, output_size=1, ahead=1):
#         super(LSTM_Regression, self).__init__(input_size, hidden_size, num_layers, dropout_rate, output_size, ahead)
    
#     def forward(self, x):
#         out = super().forward(x)
#         return out  # Không cần softmax cho hồi quy



# # Phần 1.2: chi tiết LSTM cho Classification
# class LSTMClassification(BaseLSTM):
#     def __init__(self, input_size, hidden_size=50, num_layers=2, dropout_rate=0.2, num_classes=2):
#         # Lớp đầu ra có `output_size=num_classes` cho bài toán phân loại
#         super(LSTMClassification, self).__init__(input_size, hidden_size, num_layers, dropout_rate, output_size=num_classes)
    
#     def forward(self, x):
#         out = super().forward(x)
#         return torch.softmax(out, dim=1)  # Áp dụng softmax để chuẩn hóa xác suất cho bài toán phân loại
# Phần 1.2: Lớp LSTM cho Classification hỗ trợ output_size và ahead
class LSTMClassification(BaseLSTM):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout_rate=0.2, num_classes=2, ahead=1):
        # `output_size=num_classes` để dự đoán xác suất cho từng lớp
        super(LSTMClassification, self).__init__(input_size, hidden_size, num_layers, dropout_rate, output_size=num_classes, ahead=ahead)
    
    def forward(self, x):
        out = super().forward(x)
        
        # Áp dụng softmax để chuẩn hóa xác suất cho từng lớp khi ahead > 1
        if self.ahead > 1:
            out = torch.softmax(out, dim=2)  # Softmax trên chiều output_size khi dự đoán nhiều bước
        else:
            out = torch.softmax(out, dim=1)  # Softmax cho dự đoán một bước
        
        return out
# class LSTMClassification(BaseLSTM):
#     def __init__(self, input_size, hidden_size=50, num_layers=2, dropout_rate=0.2, num_classes=2, ahead=1):
#         # `output_size=num_classes` để dự đoán xác suất cho từng lớp
#         super(LSTMClassification, self).__init__(input_size, hidden_size, num_layers, dropout_rate, output_size=num_classes, ahead=ahead)
    
#     def forward(self, x):
#         out = super().forward(x)
        
#         # Áp dụng softmax để chuẩn hóa xác suất cho từng bước thời gian nếu ahead > 1
#         if self.ahead > 1:
#             out = torch.softmax(out, dim=2)  # Softmax trên chiều output_size khi dự đoán nhiều bước
#         else:
#             out = torch.softmax(out, dim=1)  # Softmax cho dự đoán một bước
        
#         return out
    
# Hướng dẫn sử dụng **************************************************************************************
# Đối với bài toán hồi quy
# input_size = X_train.shape[2]  # Số features
# model_regression = LSTM_Regression(input_size=input_size, hidden_size=50, num_layers=2, dropout_rate=0.2)

# Đối với bài toán phân loại
# num_classes = 3  # Ví dụ có 3 lớp phân loại
# model_classification = LSTMClassification(input_size=input_size, hidden_size=50, num_layers=2, dropout_rate=0.2, num_classes=num_classes)

# Phần 2: Khởi tạo Manager Base
class BaseModelManager(ABC):
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = {"train_loss": [], "val_loss": []}

    @abstractmethod
    def train(self, *args, **kwargs):
        pass  # Phương thức train sẽ được triển khai riêng trong lớp con

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass  # Phương thức evaluate sẽ được triển khai riêng trong lớp con

    def early_stopping(self, val_loss, patience=10):
        """Dừng huấn luyện sớm nếu không có cải thiện về loss trong một số epoch"""
        if not hasattr(self, "_best_loss"):
            self._best_loss = val_loss
            self._patience_counter = 0
        elif val_loss < self._best_loss:
            self._best_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            if self._patience_counter >= patience:
                print("Early stopping triggered.")
                return True
        return False

    def save_model(self, file_path):
        """Lưu mô hình đã huấn luyện"""
        torch.save(self.model.state_dict(), file_path)
        print(f"Mô hình đã được lưu tại {file_path}")

    def load_model(self, file_path):
        """Tải lại mô hình đã lưu"""
        self.model.load_state_dict(torch.load(file_path))
        print(f"Mô hình đã được tải từ {file_path}")

    def plot_training_history(self):
        """Vẽ biểu đồ lịch sử loss của quá trình huấn luyện"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.show()

# Phần 2.1: chi tiết Manager cho Regression
# Điều chỉnh ModelManagerRegression với output_size > 1 và ahead > 1
class ModelManagerRegression(BaseModelManager):
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        """Huấn luyện mô hình cho bài toán hồi quy"""
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
                
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)

                # Điều chỉnh kích thước của y_pred nếu cần thiết để khớp với y_batch
                if y_pred.shape != y_batch.shape:
                    y_pred = y_pred.view_as(y_batch)

                # Tính toán loss
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Tính loss trung bình
            train_loss /= len(train_loader)
            val_loss = self.evaluate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Kiểm tra early stopping
            if self.early_stopping(val_loss, patience):
                break

            # Cập nhật scheduler nếu có
            if self.scheduler:
                self.scheduler.step()

    def evaluate(self, data_loader):
        """Đánh giá mô hình hồi quy trên tập validation hoặc kiểm tra"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
                y_pred = self.model(X_batch)

                # Điều chỉnh kích thước của y_pred nếu cần thiết để khớp với y_batch
                if y_pred.shape != y_batch.shape:
                    y_pred = y_pred.view_as(y_batch)

                # Tính toán loss
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)

# class ModelManagerRegression(BaseModelManager):
#     def train(self, train_loader, val_loader, epochs=100, patience=10):
#         """Huấn luyện mô hình cho bài toán hồi quy"""
#         for epoch in range(epochs):
#             self.model.train()
#             train_loss = 0.0
#             for X_batch, y_batch in train_loader:
#                 X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
                
#                 self.optimizer.zero_grad()
#                 y_pred = self.model(X_batch)

#                 # Điều chỉnh kích thước nếu cần thiết
#                 if y_pred.shape != y_batch.shape:
#                     y_pred = y_pred.view_as(y_batch)

#                 loss = self.criterion(y_pred, y_batch)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 train_loss += loss.item()
            
#             train_loss /= len(train_loader)
#             val_loss = self.evaluate(val_loader)
#             self.history["train_loss"].append(train_loss)
#             self.history["val_loss"].append(val_loss)
            
#             print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
#             if self.early_stopping(val_loss, patience):
#                 break

#             if self.scheduler:
#                 self.scheduler.step()

#     def evaluate(self, data_loader):
#         """Đánh giá mô hình hồi quy trên tập validation hoặc kiểm tra"""
#         self.model.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for X_batch, y_batch in data_loader:
#                 X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
#                 y_pred = self.model(X_batch)

#                 # Điều chỉnh kích thước nếu cần thiết
#                 if y_pred.shape != y_batch.shape:
#                     y_pred = y_pred.view_as(y_batch)

#                 loss = self.criterion(y_pred, y_batch)
#                 total_loss += loss.item()
        
#         return total_loss / len(data_loader)

# class ModelManagerRegression(BaseModelManager):
#     def train(self, train_loader, val_loader, epochs=100, patience=10):
#         """Huấn luyện mô hình cho bài toán hồi quy"""
#         for epoch in range(epochs):
#             self.model.train()
#             train_loss = 0.0
#             for X_batch, y_batch in train_loader:
#                 X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
                
#                 self.optimizer.zero_grad()
#                 y_pred = self.model(X_batch).squeeze()
#                 loss = self.criterion(y_pred, y_batch)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 train_loss += loss.item()
            
#             # Tính loss trên tập huấn luyện và validation
#             train_loss /= len(train_loader)
#             val_loss = self.evaluate(val_loader)
#             self.history["train_loss"].append(train_loss)
#             self.history["val_loss"].append(val_loss)
            
#             # In ra tiến trình
#             print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
#             # Kiểm tra early stopping
#             if self.early_stopping(val_loss, patience):
#                 break

#             # Cập nhật scheduler (nếu có)
#             if self.scheduler:
#                 self.scheduler.step()

#     def evaluate(self, data_loader):
#         """Đánh giá mô hình hồi quy trên tập validation hoặc kiểm tra"""
#         self.model.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for X_batch, y_batch in data_loader:
#                 X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
#                 y_pred = self.model(X_batch).squeeze()
#                 loss = self.criterion(y_pred, y_batch)
#                 total_loss += loss.item()
        
#         return total_loss / len(data_loader)
    
# Phần 2.2 chi tiết Manager cho Classification
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
class ModelManagerClassification(BaseModelManager):
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        """Huấn luyện mô hình cho bài toán phân loại"""
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
                
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)

                # Nếu `ahead > 1`, điều chỉnh `y_pred` và `y_batch` để tương thích
                if y_pred.shape[:-1] != y_batch.shape:
                    y_batch = y_batch.view(y_pred.shape[:-1])

                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss = self.evaluate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.early_stopping(val_loss, patience):
                break

            if self.scheduler:
                self.scheduler.step()

    def evaluate(self, data_loader):
        """Đánh giá mô hình phân loại trên tập validation hoặc kiểm tra"""
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # Dự đoán và so sánh với `y_batch` khi có `ahead > 1`
                _, predicted = torch.max(outputs, -1 if self.model.ahead > 1 else 1)
                y_true.extend(y_batch.cpu().numpy().flatten())
                y_pred.extend(predicted.cpu().numpy().flatten())
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        return total_loss / len(data_loader)

# class ModelManagerClassification(BaseModelManager):
#     def train(self, train_loader, val_loader, epochs=100, patience=10):
#         """Huấn luyện mô hình cho bài toán phân loại"""
#         for epoch in range(epochs):
#             self.model.train()
#             train_loss = 0.0
#             for X_batch, y_batch in train_loader:
#                 X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
                
#                 self.optimizer.zero_grad()
#                 y_pred = self.model(X_batch)
#                 loss = self.criterion(y_pred, y_batch)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 train_loss += loss.item()
            
#             train_loss /= len(train_loader)
#             val_loss = self.evaluate(val_loader)
#             self.history["train_loss"].append(train_loss)
#             self.history["val_loss"].append(val_loss)
            
#             # In ra tiến trình
#             print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
#             if self.early_stopping(val_loss, patience):
#                 break

#             if self.scheduler:
#                 self.scheduler.step()

#     def evaluate(self, data_loader):
#         """Đánh giá mô hình phân loại trên tập validation hoặc kiểm tra"""
#         self.model.eval()
#         total_loss = 0.0
#         y_true, y_pred = [], []
        
#         with torch.no_grad():
#             for X_batch, y_batch in data_loader:
#                 X_batch, y_batch = X_batch.to(self.model.device), y_batch.to(self.model.device)
#                 outputs = self.model(X_batch)
#                 loss = self.criterion(outputs, y_batch)
#                 total_loss += loss.item()
                
#                 _, predicted = torch.max(outputs, 1)
#                 y_true.extend(y_batch.cpu().numpy())
#                 y_pred.extend(predicted.cpu().numpy())
        
#         accuracy = accuracy_score(y_true, y_pred)
#         print(f"Accuracy: {accuracy:.4f}")
        
#         return total_loss / len(data_loader)

#     def plot_confusion_matrix(self, y_true, y_pred):
#         """Vẽ ma trận nhầm lẫn cho bài toán phân loại"""
#         cm = confusion_matrix(y_true, y_pred)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.title("Confusion Matrix")
#         plt.show()

# Hướng dẫn sử dụng *************************************************************************
# Đối với bài toán hồi quy
# regression_manager = ModelManagerRegression(model, criterion, optimizer, scheduler)
# regression_manager.train(train_loader, val_loader, epochs=100, patience=10)
# regression_manager.evaluate(test_loader)

# Đối với bài toán phân loại
# classification_manager = ModelManagerClassification(model, criterion, optimizer, scheduler)
# classification_manager.train(train_loader, val_loader, epochs=100, patience=10)
# classification_manager.evaluate(test_loader)