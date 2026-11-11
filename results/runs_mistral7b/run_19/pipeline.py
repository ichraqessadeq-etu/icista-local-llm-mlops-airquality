import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform any necessary data preprocessing such as normalization, encoding, etc.
    # For simplicity, I'll assume no preprocessing is required.
    return data[Features], data['Anomaly']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_artifact(f"models/random_forest_{mlflow.active_run().info.run_id}.pkl", artifact_path="models")
        return model
