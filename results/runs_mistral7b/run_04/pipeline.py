import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like normalization, encoding, etc.
    # For simplicity, let's assume no preprocessing is needed
    return data[Features], data['Anomaly']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_artifact(artifact=f"model_{mlflow.active_run().info.run_id}.pkl", local_path=f"model.pkl")
    return model
