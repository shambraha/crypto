import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

class MachineLearningManager:
    def __init__(self, model_type="random_forest", task_type="classification", **kwargs):
        """
        Khởi tạo class quản lý mô hình.
        :param model_type: Loại mô hình ("random_forest" hoặc "xgboost")
        :param task_type: Loại bài toán ("regression" hoặc "classification")
        :param kwargs: Các tham số cho mô hình
        """
        self.task_type = task_type.lower()
        self.model = self._initialize_model(model_type, **kwargs)
        self.best_score = float("inf")  # Đặt giá trị ban đầu cho best_score
        # self.best_model_path = "modelsML/best_model.pkl"  # Tên tệp cố định cho mô hình tốt nhất
        self.best_model_path = f"modelsML/best_{model_type}_{task_type}.pkl"  # Đặt tên duy nhất cho mỗi loại

    def _initialize_model(self, model_type, **kwargs):
        if self.task_type == "regression":
            if model_type == "random_forest":
                return RandomForestRegressor(**kwargs)
            elif model_type == "xgboost":
                return XGBRegressor(use_label_encoder=False, eval_metric='rmse', **kwargs)
        elif self.task_type == "classification":
            if model_type == "random_forest":
                return RandomForestClassifier(**kwargs)
            elif model_type == "xgboost":
                return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **kwargs)
        raise ValueError("Unsupported model type or task type")
    
    def get_model_summary(self):
        """
        Trả về một chuỗi mô tả rút gọn của mô hình với các tham số chính.
        """
        if isinstance(self.model, (RandomForestRegressor, RandomForestClassifier)):
            return f"{self.model.__class__.__name__}(max_depth={self.model.max_depth}, n_estimators={self.model.n_estimators})"
        
        elif isinstance(self.model, (XGBRegressor, XGBClassifier)):
            params = self.model.get_params()
            return f"{self.model.__class__.__name__}(max_depth={params['max_depth']}, learning_rate={params['learning_rate']}, n_estimators={params['n_estimators']})"
        
        else:
            return f"{self.model.__class__.__name__}"

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Huấn luyện mô hình với dữ liệu huấn luyện và validation.
        """
        if isinstance(self.model, (XGBRegressor, XGBClassifier)):
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        else:
            self.model.fit(X_train, y_train)
        self.save_model()  # Tự động lưu mô hình sau khi train
    
    def train_with_best_model(self, X_train, y_train, X_val, y_val):
        """
        Huấn luyện mô hình với tập train và tập validation, lưu mô hình tốt nhất.
        """
        # Huấn luyện mô hình, lưu lại mô hình nếu đạt điểm tốt nhất trên tập validation
        self.model.fit(X_train, y_train)
        score = mean_squared_error(y_val, self.model.predict(X_val))

        if score < self.best_score:
            self.best_score = score
            self.save_model()  # Lưu mô hình tốt nhất mới với điểm số thấp nhất
            print(f"New best model saved with score: {self.best_score:.4f}")
                
    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình với dữ liệu kiểm thử.
        :return: Dictionary với các chỉ số đánh giá
        """
        y_pred = self.model.predict(X_test)
        if self.task_type == "regression":
            print("Evaluating regression model...")
            return self._evaluate_regression(y_test, y_pred)
        elif self.task_type == "classification":
            print("Evaluating classification model...")
            return self._evaluate_classification(y_test, y_pred)
        else:
            raise ValueError("Invalid task type. Supported task types are 'regression' and 'classification'.")
    
    def _evaluate_regression(self, y_test, y_pred):
        # Đảm bảo y_test và y_pred là một chiều
        y_test = np.ravel(y_test)
        y_pred = np.ravel(y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        results = {
            "MSE": f"{mse:.2f} (Trung bình bình phương sai số)",
            "RMSE": f"{rmse:.2f} (Căn bậc hai trung bình bình phương sai số)",
            "MAE": f"{mae:.2f} (Trung bình sai số tuyệt đối)"
        }

        print("Regression Metrics:", results)
        return results
    
    def _evaluate_classification(self, y_test, y_pred):
        # Đảm bảo y_test và y_pred là một chiều
        y_test = np.ravel(y_test)
        y_pred = np.ravel(y_pred)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        #results = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
        results = {
            "Accuracy": f"{accuracy:.2f} (Phần trăm dự đoán đúng trên tổng số mẫu)",
            "Precision": f"{precision:.2f} (Trong những dự đoán là 'đúng', có bao nhiêu phần trăm là đúng thật sự)",
            "Recall": f"{recall:.2f} (Trong tất cả các trường hợp 'đúng' thật sự, mô hình nhận diện được bao nhiêu phần trăm)",
            "F1 Score": f"{f1:.2f} (Sự cân bằng giữa độ chính xác của dự đoán và khả năng tìm được hết các trường hợp 'đúng')"
        }

        print("Classification Metrics:", results)
        return results

    def save_model(self):
        # """
        # Lưu mô hình với tên chứa tên lớp của mô hình.
        # """
        os.makedirs('modelsML', exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        save_path = self.best_model_path  # Đường dẫn cố định cho tệp mô hình tốt nhất
        
        try:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

        # os.makedirs('modelsML', exist_ok=True)
        # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # model_name = f'best-{self.model.__class__.__name__}-{current_time}.pkl'
        # save_path = os.path.join('modelsML', model_name)
        # try:
        #     joblib.dump(self.model, save_path)
        #     print(f"Model saved to {save_path}")
        # except Exception as e:
        #     print(f"Failed to save model: {e}")

    def load_model(self, filepath):
        """
        Tải mô hình từ file.
        """
        try:
            self.model = joblib.load(filepath)
            print(f'Model loaded from {filepath}')
        except Exception as e:
            print(f"Failed to load model: {e}")

    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới.
        """
        return self.model.predict(X)
