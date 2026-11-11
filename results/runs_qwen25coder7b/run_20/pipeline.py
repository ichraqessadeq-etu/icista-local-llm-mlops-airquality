import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.exceptions import NotFittedError

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess data and split into features and target."""
    try:
        X = data.drop(columns=['Anomaly'])
        y = data['Anomaly']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    try:
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def predict(model, X):
    """Predict anomalies using a trained model."""
    try:
        if not hasattr(model, 'predict'):
            raise NotFittedError("Model is not fitted")
        return model.predict(X)
    except Exception as e:
        raise Exception(f"Failed to make predictions: {e}")
