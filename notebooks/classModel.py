import os
import time
import torch
import torch.nn as nn
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# class LSTM thuần chủng, chỉ là cấu trúc mô hình
#... chưa có dùng [hàm kích hoạt] để xử lý đầu ra
#region class LSTM----------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, ahead=1):
        """
        Class LSTM dùng chung cho cả bài toán phân loại và hồi quy.
        :param input_size: Số lượng đặc trưng đầu vào.
        :param hidden_size: Kích thước trạng thái ẩn.
        :param output_size: Số lượng lớp đầu ra (phân loại hoặc hồi quy).
        :param num_layers: Số lớp trong mạng LSTM.
        :param ahead: Số bước thời gian dự đoán (mặc định là 1).
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ahead = ahead
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size * ahead)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output of the last time step
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)

        return out    
    
#endregion

#region MM_Regression--------------------------------------------------------------
# có thêm parameter ahead để [reshape the outputs]
class ModelManagerRegression:
    def __init__(self, model, train_loader, val_loader, optimizer, lr=0.001, patience=50, criterion=torch.nn.MSELoss(), ahead=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.ahead = ahead

        self.lr = lr
        self.patience = patience
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_losses = []
        # self.train_accuracies = []
        # self.val_accuracies = []

    def __repr__(self):
        return (f"ModelManager(model={self.model.__class__.__name__}, "
                f"lr={self.lr}, patience={self.patience}, "
                f"criterion={self.criterion.__class__.__name__}, "
                f"optimizer={self.optimizer.__class__.__name__})")
    
    def train(self, num_epochs, save_dir='.', scheduler=None):
        #1 Dòng này để lưu model
        os.makedirs(save_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(save_dir, f'best-{self.model.__class__.__name__}-{current_time}.pth')

        #2 Duyệt qua các epochs và cặp (inputs, targets) trong train.loader
        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()  # Set the model to training mode
            total_train_loss = 0

            for inputs, targets in self.train_loader:
                #2.1 Forward pass
                outputs = self.model(inputs)  # outputs shape: [batch_size, ahead, output_size]

                # Reshape the outputs for regression if needed (ahead=1)
                outputs = outputs.view(-1, self.ahead, 1)  # Reshape to [batch_size, ahead, output_size]
                # Note: if ahead=1, this won't change the shape, so it's safe for regression

                # Đảm bảo targets cũng có cùng shape
                assert outputs.shape == targets.shape, f"Mismatch in shape: {outputs.shape} vs {targets.shape}"

                #2.2 Compute loss
                loss = self.criterion(outputs, targets)
                total_train_loss += loss.item()

                #2.3 Không có Compute accuracies

                #2.4 Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            # regression không cần tính accuracy

            #2.5 Đánh giá mô hình trên tập validation
            val_loss = self.evaluate(loader=self.val_loader) if self.val_loader else None

            #2.6 Kiểm tra loại scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Với ReduceLROnPlateau, cập nhật dựa trên val_loss
                    if val_loss is not None:
                        scheduler.step(val_loss)
                else:
                    # Với các scheduler khác (StepLR, CosineAnnealingLR,...), chỉ cần gọi step()
                    scheduler.step()
            
            #2.7 Lưu lại loss và accuracy cho từng epoch
            self.train_losses.append(avg_train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)

            #2.8 In thông tin sau mỗi epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], time: {int(time.time() - start_time)}s, ' +
                  f'loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}' if val_loss is not None else '')

            #2.9 Kiểm tra dừng sớm (early stopping)
            if val_loss and self.early_stopping(val_loss, save_path):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        #3. Tải lại mô hình tốt nhất sau khi hoàn thành quá trình huấn luyện
        self.load_model(save_path)

    def evaluate(self, loader):
        """Đánh giá mô hình trên tập validation."""
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)

                # Reshape outputs nếu cần thiết cho bài toán multi-step
                outputs = outputs.view(-1, self.ahead, 1)  # Điều chỉnh để phù hợp với targets nếu ahead > 1

                # Tính toán loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        return avg_loss
    
    def early_stopping(self, val_loss, save_path):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model(save_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Model loaded from {load_path}')

    def plot_training_history(self):
        """Hiển thị biểu đồ quá trình huấn luyện."""
        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def predict(self, input_data):
        """
        Hàm dự đoán cho ModelManagerRegression.
        :param input_data: Dữ liệu đầu vào (có thể là DataLoader hoặc tensor).
        :return: predicted_values: Giá trị dự đoán.
        """
        self.model.eval()  # Đặt model ở chế độ đánh giá
        with torch.no_grad():
            if isinstance(input_data, torch.utils.data.DataLoader):
                predictions = []
                for inputs, _ in input_data:
                    outputs = self.model(inputs)
                    predictions.append(outputs)
                predicted_values = torch.cat(predictions)
            else:
                predicted_values = self.model(input_data)
            return predicted_values

    def plot_y_yhat(self, y, yhat, feature_names=None, save_dir='.', save_plots=True, num_elements=None):
            if feature_names is None:
                feature_names = [f'Feature {i + 1}' for i in range(y.shape[2])]

            if num_elements is not None:
                y = y[:num_elements]
                yhat = yhat[:num_elements]

            for feature_index, feature_name in enumerate(feature_names):
                plt.figure(figsize=(10, 5))
                plt.plot(y[:, :, feature_index].flatten(), label='y', linestyle='-')
                plt.plot(yhat[:, :, feature_index].flatten(), label='y_hat', linestyle='-')

                plt.title(feature_name)
                plt.xlabel('Time Step')
                plt.ylabel('Values')
                plt.legend()

                if save_plots:
                    os.makedirs(os.path.join(save_dir, self.model.__class__.__name__), exist_ok=True)
                    save_path = os.path.join(save_dir, self.model.__class__.__name__, f'{feature_name}.png')
                    plt.savefig(save_path)

                plt.show()
                plt.close()  # Close the plot to avoid overlapping in saved images

#endregion MM_Regression

#region MM_Classification----------------------------------------------------------
class ModelManagerClassification:
    def __init__(self, model, train_loader, val_loader, optimizer, lr=0.001, patience=50, criterion=torch.nn.CrossEntropyLoss()):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

        self.lr = lr
        self.patience = patience
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def __repr__(self):
        return (f"ModelManager(model={self.model.__class__.__name__}, "
                f"lr={self.lr}, patience={self.patience}, "
                f"criterion={self.criterion.__class__.__name__}, "
                f"optimizer={self.optimizer.__class__.__name__})")
    
    def train(self, num_epochs, save_dir='.', scheduler=None):
        #1 Dòng này để lưu model
        os.makedirs(save_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(save_dir, f'best-{self.model.__class__.__name__}-{current_time}.pth')
        
        #2 Duyệt qua các epochs và cặp (inputs, targets) trong train.loader
        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()  # Set model to training mode
            total_train_loss = 0
            total_correct = 0 # Only in classification
            total_samples = 0 # Only in classification

            for inputs, targets in self.train_loader:
                #2.1 Forward pass
                outputs = self.model(inputs)  # outputs shape: [batch_size, num_classes]
                # Apply softmax to get class probabilities 
                # ... bởi vì ở class LSTM thuần, không xử lý output
                outputs = torch.softmax(outputs, dim=1)  # [batch_size, num_classes]

                # Điều chỉnh targets nếu dùng CrossEntropyLoss
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    if targets.dim() == 2 and targets.size(1) > 1:  # Nếu targets đang ở dạng one-hot
                        targets = torch.argmax(targets, dim=1)  # Chuyển về dạng số nguyên cho CrossEntropyLoss
                targets = targets.long()

                #2.2 Compute loss
                loss = self.criterion(outputs, targets)
                total_train_loss += loss.item()

                #2.3 Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                #2.4 Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_train_loss / len(self.train_loader)
            train_accuracy = total_correct / total_samples

            #2.5 Đánh giá mô hình trên tập validation
            val_loss, val_accuracy = self.evaluate(loader=self.val_loader) if self.val_loader else (None, None)

            #2.6 Kiểm tra loại scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Với ReduceLROnPlateau, cập nhật dựa trên val_loss
                    if val_loss is not None:
                        scheduler.step(val_loss)
                else:
                    # Với các scheduler khác (StepLR, CosineAnnealingLR,...), chỉ cần gọi step()
                    scheduler.step()
            
            #2.7 Lưu lại loss và accuracy cho từng epoch
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            if val_loss is not None:
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

            #2.8  In thông tin sau mỗi epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], time: {int(time.time() - start_time)}s, ' +
                  f'loss: {avg_train_loss:.4f}, accuracy: {train_accuracy:.4f}, ' +
                  f'val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}' if val_loss is not None else '')

            #2.9 Kiểm tra dừng sớm (early stopping)
            if val_loss and self.early_stopping(val_loss, save_path):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        #3. Tải lại mô hình tốt nhất sau khi hoàn thành quá trình huấn luyện
        self.load_model(save_path)

    def evaluate(self, loader):
        """Đánh giá mô hình trên tập validation."""
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                # Điều chỉnh outputs nếu dùng softmax hoặc CrossEntropyLoss
                outputs = self.model(inputs)

                # Điều chỉnh targets nếu dùng CrossEntropyLoss
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    if targets.dim() == 2 and targets.size(1) > 1:  # Nếu targets đang ở dạng one-hot
                        targets = torch.argmax(targets, dim=1)  # Chuyển về dạng số nguyên cho CrossEntropyLoss
                targets = targets.long()
                
                _, predicted = torch.max(outputs, 1)  # Lấy nhãn dự đoán có xác suất cao nhất
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Tính accuracy bằng cách so sánh predicted với targets
                correct = (predicted == targets).sum().item()
                total_correct += correct
                total_samples += targets.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def early_stopping(self, val_loss, save_path):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model(save_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Model loaded from {load_path}')

    def plot_training_history(self):
        """Hiển thị biểu đồ quá trình huấn luyện."""
        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def predict(self, input_data):
        """
        Hàm dự đoán cho ModelManagerClassification.
        :param input_data: Dữ liệu đầu vào (có thể là DataLoader hoặc tensor).
        :return: predicted_class: Nhãn dự đoán, probabilities: Xác suất của các nhãn.
        """
        self.model.eval()  # Đặt model ở chế độ đánh giá
        with torch.no_grad():
            if isinstance(input_data, torch.utils.data.DataLoader):
                predictions = []
                probabilities = []
                for inputs, _ in input_data:
                    outputs = self.model(inputs)
                    prob = torch.softmax(outputs, dim=1)  # Chuyển đổi logits thành xác suất
                    _, predicted_class = torch.max(prob, 1)  # Lấy nhãn dự đoán có xác suất cao nhất
                    predictions.append(predicted_class)
                    probabilities.append(prob)
                predictions = torch.cat(predictions)
                probabilities = torch.cat(probabilities)
            else:
                outputs = self.model(input_data)
                prob = torch.softmax(outputs, dim=1)
                _, predicted_class = torch.max(prob, 1)
                return predicted_class, prob
        return predictions, probabilities

    def plot_confusion_matrix(self, loader, class_names):
            """
            Vẽ ma trận nhầm lẫn (confusion matrix) sau quá trình huấn luyện.
            :param loader: DataLoader chứa dữ liệu kiểm thử hoặc validation.
            :param class_names: Danh sách tên các lớp.
            """
            self.model.eval()  # Chuyển mô hình sang chế độ đánh giá
            all_preds = []
            all_targets = []

            with torch.no_grad():  # Tắt chế độ tính toán gradient
                for inputs, targets in loader:
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            # Tính ma trận nhầm lẫn
            cm = confusion_matrix(all_targets, all_preds)
            
            # Vẽ ma trận nhầm lẫn
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

#endergion MM_Classification