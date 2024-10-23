import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime

class MachineLearningManager:
    def __init__(self, model_type="random_forest", **kwargs):
        """
        Khởi tạo class quản lý mô hình.
        :param model_type: Loại mô hình ("random_forest" hoặc "xgboost")
        :param kwargs: Các tham số cho mô hình
        """
        if model_type == "random_forest":
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == "xgboost":
            self.model = XGBRegressor(use_label_encoder=False, eval_metric='rmse', **kwargs)
        else:
            raise ValueError("Model type not supported. Use 'random_forest' or 'xgboost'.")
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Huấn luyện mô hình với dữ liệu huấn luyện và validation.
        """
        if isinstance(self.model, XGBRegressor):
            # Nếu là XGBoost, sử dụng eval_set để theo dõi quá trình huấn luyện
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))  # Thêm tập validation nếu có
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,  # Đặt tập đánh giá để theo dõi
                verbose=True  # Hiển thị thông tin trong quá trình huấn luyện
            )
        elif isinstance(self.model, RandomForestRegressor):
            # Nếu là RandomForest, chỉ huấn luyện với dữ liệu huấn luyện (không cần eval_set)
            self.model.fit(X_train, y_train)
        else:
            raise ValueError("Unsupported model type for training.")
        
        # Lưu lại mô hình sau khi train xong
        self.save_model()
    
    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình với dữ liệu kiểm thử.
        :param X_test: Dữ liệu kiểm thử (features)
        :param y_test: Nhãn của dữ liệu kiểm thử (labels)
        :return: Dictionary với các chỉ số đánh giá
        """
        # metrics = {
        #     "accuracy": accuracy_score(y_test, y_pred),
        #     "precision": precision_score(y_test, y_pred, average='weighted'),
        #     "recall": recall_score(y_test, y_pred, average='weighted'),
        #     "f1_score": f1_score(y_test, y_pred, average='weighted')
        # }
        # return metrics
        if isinstance(self.model, RandomForestRegressor):
            y_pred = self.model.predict(X_test)        
            
            # Kiểm tra hiệu suất của mô hình
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            return mse, rmse, mae
        elif isinstance(self.model, RandomForestRegressor):
            y_pred = self.model.predict(X_test)        
            
            # Kiểm tra hiệu suất của mô hình
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            return mse, rmse, mae
    
    def save_model(self):
        """
        Lưu mô hình với tên chứa tên lớp của mô hình.
        """
        os.makedirs('modelsML', exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f'best-{self.model.__class__.__name__}-{current_time}.pkl'  # Tạo tên file với tên lớp
        save_path = os.path.join('modelsML', model_name)

        # Lưu mô hình sử dụng joblib
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filepath):
        """
        Tải mô hình từ file.
        :param filepath: Đường dẫn file chứa mô hình
        """
        self.model = joblib.load(filepath)
        print(f'Model loaded from {filepath}')
    
    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới.
        :param X: Dữ liệu mới (features)
        :return: Nhãn dự đoán
        """
        return self.model.predict(X)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_test, y_pred):
    """
    Hàm đánh giá mô hình dựa trên các chỉ số: accuracy, precision, recall, f1_score.
    :param y_test: Nhãn thực tế
    :param y_pred: Nhãn dự đoán từ mô hình
    :return: Dictionary chứa các chỉ số đánh giá
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }
    
    # In ra các chỉ số
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
    
    return metrics